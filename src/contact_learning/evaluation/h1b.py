"""H1-B: does forecast staleness after an action correction change control outcomes?

See docs/experimental_protocol.md (H1-B) for the registered design. Conditions:
    cached      forecasts only at chunk boundaries, made before the correction arrives
    refreshed   cached + one extra forecast at t_c for the corrected chunk
    live_only   no forecast term
    fixed_rate  forecasts every R steps on a correction-independent schedule
    oracle      as refreshed, but with the true simulated future (NOT deployable)
All conditions share the same live-feedback term and gains.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from contact_learning.controllers.scripted import nominal_plan, sample_plan_params
from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW, GripperLiftEnv
from contact_learning.evaluation.h1a import grasp_contact
from contact_learning.evaluation.rollout import Recorder, continue_rollout, rollout
from contact_learning.evaluation.stats import bootstrap_ci, holm, mcnemar_exact_p, wilson_ci
from contact_learning.predictors.features import flatten_inputs, history_at
from contact_learning.utils import seeding
from contact_learning.utils.provenance import provenance, sha256_file

CONDITIONS = ("cached", "refreshed", "live_only", "fixed_rate", "oracle")
PREDICTOR_FREE = ("live_only", "oracle")


def trial_setup(env: GripperLiftEnv, seed: int, mag: float, cfg: dict) -> dict | None:
    T = cfg["T"]
    rng = np.random.default_rng([seed, 3])
    p = sample_plan_params(rng)
    plan = nominal_plan(p, T, env.cfg.max_lift_speed, env.cfg.control_dt)
    nom = rollout(env, seed, plan)
    c = grasp_contact(nom.tactile_clean, env.cfg.contact_threshold)
    valid = [t for t in cfg["h1b"]["correction_times"] if c[t]]
    if not valid:
        return None
    t_c = int(rng.choice(valid))
    plan_c = plan.copy()
    plan_c[t_c:, 0] -= 2.0 * mag
    plan_c = np.clip(plan_c, ACTION_LOW, ACTION_HIGH)
    lift_start = p.close_start + p.close_steps + p.hold_steps
    return {"seed": seed, "plan": plan, "plan_c": plan_c, "t_c": t_c, "magnitude": mag,
            "nominal_success": env.held(_last_obs(nom), cfg["h1b"]["min_lift"]),
            "lift_in_progress_at_tc": bool(lift_start <= t_c and nom.actions[t_c, 1] > 0),
            "g_hold": p.g_hold}


def _last_obs(traj) -> dict:
    return {"box_z": np.array([traj.box_z[-1]]), "tactile_clean": traj.tactile_clean[-1]}


def run_condition(env: GripperLiftEnv, setup: dict, cond: str, predictor, cfg: dict) -> dict:
    T, K, H = cfg["T"], cfg["K"], cfg["H"]
    hb = cfg["h1b"]
    N_min, kappa, R = hb["N_min"], hb["kappa"], hb["fixed_rate_period"]
    t_c, plan, plan_c = setup["t_c"], setup["plan"], setup["plan_c"]
    obs = env.reset(setup["seed"])
    rec = Recorder(obs)
    u_f = u_l = 0.0
    n_fc, lat, g_exec, n_meas_after = 0, [], [], []

    def chunk(known_plan, t):
        a = known_plan[t:t + H].copy()
        if len(a) < H:  # pad beyond episode end with the last action
            a = np.vstack([a, np.repeat(a[-1:], H - len(a), 0)])
        g = (a[:, 0] + 1) / 2 + u_f + u_l
        a[:, 0] = 2 * np.clip(g, 0, 1) - 1
        return np.clip(a, ACTION_LOW, ACTION_HIGH)

    def forecast_deficit(known_plan, t):
        nonlocal n_fc
        A = chunk(known_plan, t)
        t0 = time.perf_counter()
        if cond == "oracle":
            s = env.get_state()
            fut = continue_rollout(env, A)["tactile_clean"]
            env.set_state(s)
        else:
            tr = rec.trajectory()
            h = history_at(tr.tactile, tr.proprio, tr.actions, t, K)
            fut = predictor.predict(flatten_inputs(h, None)[None], A.reshape(1, -1))[0]
        lat.append(time.perf_counter() - t0)
        n_fc += 1
        n_hat = fut[:, :2].mean(1)
        return max(0.0, N_min - float(n_hat.min()))

    for t in range(T):
        active = t >= t_c
        if active and t > t_c:  # live feedback on the latest measurement (all conditions)
            n_meas = float(obs["tactile"][:2].mean())
            u_l += max(0.0, N_min - n_meas) / kappa
        if active and cond != "live_only":
            # Scheduled forecasts happen before a same-step correction is known.
            pre_plan = plan if t == t_c else plan_c
            if cond == "fixed_rate":
                if t % R == 0:
                    u_f += forecast_deficit(pre_plan, t) / kappa
            elif t % H == 0:
                d_sched = forecast_deficit(pre_plan, t)
                if cond in ("refreshed", "oracle") and t == t_c:
                    # Event-triggered refresh: the corrected-chunk forecast REPLACES the
                    # just-made (now stale) forecast; the stale one still counts as compute.
                    u_f += forecast_deficit(plan_c, t) / kappa
                else:
                    u_f += d_sched / kappa
        known = plan_c if active else plan
        a = chunk(known, t)[0] if active else known[t]
        g_exec.append((a[0] + 1) / 2)
        obs = env.step(a)
        rec.add(a, obs)
        if active:
            n_meas_after.append(float(obs["tactile_clean"][:2].mean()))

    tr = rec.trajectory()
    slip = float((tr.hand_z[-1] - tr.box_z[-1]) - (tr.hand_z[0] - tr.box_z[0]))
    return {"success": env.held(obs, hb["min_lift"]), "min_normal_after_tc": float(min(n_meas_after)),
            "max_normal": float(tr.tactile_clean[:, :2].max()), "slip_m": slip, "n_forecasts": n_fc,
            "forecast_latency_ms_median": float(np.median(lat) * 1e3) if lat else 0.0,
            "u_f_final": u_f, "u_l_final": u_l, "_g": np.array(g_exec)}


def run_h1b(cfg, env, overwrite, load_models, paths, guard, write_json):
    eid, art, summ = paths(cfg)
    out_csv = art / "h1b_episodes.csv"  # per-episode raw table: artifacts, checksummed in summary
    guard(out_csv, overwrite)
    hb = cfg["h1b"]
    models = {s: load_models(cfg, s)["mlp"] for s in cfg["predictor_seeds"]}
    rows, gtraj = [], {}
    base = seeding.H1B_BASE
    t0 = time.time()
    n_skipped = 0
    for mi, mag in enumerate(hb["magnitudes"]):
        made, i = 0, 0
        while made < hb["n_trials_per_magnitude"]:
            seed = base + mi * 100_000 + i
            i += 1
            st = trial_setup(env, seed, mag, cfg)
            if st is None:
                n_skipped += 1  # no nominal grasp contact at any candidate t_c
                continue
            made += 1
            meta = {k: st[k] for k in ("seed", "t_c", "magnitude", "nominal_success", "lift_in_progress_at_tc", "g_hold")}
            for cond in CONDITIONS:
                seeds = [None] if cond in PREDICTOR_FREE else cfg["predictor_seeds"]
                for ps in seeds:
                    r = run_condition(env, st, cond, models.get(ps), cfg)
                    gtraj[(seed, cond, ps)] = r.pop("_g")
                    rows.append({**meta, "condition": cond, "pred_seed": -1 if ps is None else ps, **r})
        print(f"[h1b] magnitude {mag}: {made} trials ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    # Action difference vs the cached condition (same trial, same predictor seed).
    diffs = []
    for _, r in df.iterrows():
        ps = None if r.pred_seed < 0 else int(r.pred_seed)
        ref = gtraj[(r.seed, "cached", ps if ps is not None else cfg["predictor_seeds"][0])]
        diffs.append(float(np.abs(gtraj[(r.seed, r.condition, ps)] - ref).sum()))
    df["action_diff_vs_cached"] = diffs
    df.to_csv(out_csv, index=False)
    summary = summarize_h1b(df, cfg)
    summary["n_trials_skipped_no_contact"] = n_skipped
    summary["episodes_csv"] = str(out_csv.relative_to(out_csv.parents[3]))
    summary["episodes_csv_sha256"] = sha256_file(out_csv)
    summary["provenance"] = provenance()
    summary["config"] = cfg
    write_json(summ / f"{eid}_h1b_summary.json", summary)
    print(f"[h1b] wrote {out_csv.name} ({len(df)} rows)")


def summarize_h1b(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"by_magnitude": []}
    seeds = cfg["predictor_seeds"]
    pvals = []
    for mag, d in df.groupby("magnitude"):
        rec = {"magnitude": float(mag), "n_trials": int(d.seed.nunique())}
        # Per-trial success, averaged over predictor seeds where applicable.
        per_trial = d.groupby(["seed", "condition"]).success.mean().unstack()
        for cond in CONDITIONS:
            k = int(d[d.condition == cond].success.sum())
            n = int((d.condition == cond).sum())
            p, lo, hi = wilson_ci(k, n)
            rec[f"success_{cond}"] = [p, lo, hi]
            rec[f"mean_forecasts_{cond}"] = float(d[d.condition == cond].n_forecasts.mean())
            rec[f"mean_action_diff_vs_cached_{cond}"] = float(d[d.condition == cond].action_diff_vs_cached.mean())
        for a, b in (("refreshed", "cached"), ("refreshed", "live_only"), ("fixed_rate", "cached"), ("oracle", "refreshed")):
            m, lo, hi = bootstrap_ci((per_trial[a] - per_trial[b]).values, seed=1)
            rec[f"diff_{a}_minus_{b}"] = [m, lo, hi]
        per_seed = []
        for s in seeds:
            A = d[(d.condition == "cached") & (d.pred_seed == s)].set_index("seed").success
            B = d[(d.condition == "refreshed") & (d.pred_seed == s)].set_index("seed").success.loc[A.index]
            b_ = int((B & ~A).sum()); c_ = int((A & ~B).sum())
            per_seed.append({"seed": s, "B_only": b_, "A_only": c_, "p_mcnemar": mcnemar_exact_p(b_, c_),
                             "success_A": float(A.mean()), "success_B": float(B.mean())})
        rec["per_seed_B_vs_A"] = per_seed
        rec["identical_A_B_actions"] = bool((d[d.condition == "refreshed"].action_diff_vs_cached == 0).all())
        rec["mean_min_normal_after_tc"] = {c: float(d[d.condition == c].min_normal_after_tc.mean()) for c in CONDITIONS}
        out["by_magnitude"].append(rec)
        pvals.append([ps["p_mcnemar"] for ps in per_seed])
    # Holm across non-zero magnitudes, separately for each predictor seed.
    P = np.array(pvals, dtype=float)
    nz = [i for i, r in enumerate(out["by_magnitude"]) if r["magnitude"] > 0]
    for j in range(P.shape[1]):
        adj = holm([P[i, j] for i in nz])
        for i, a in zip(nz, adj):
            out["by_magnitude"][i]["per_seed_B_vs_A"][j]["p_holm"] = a
    return out
