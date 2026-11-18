"""H1-002 closed-loop harness: learned vs analytic anticipation, under gripper-speed
and sensing-latency variation. See docs/experiments/H1-002_PROTOCOL.md.

At latency L = 0, with the H1-001 predictor, conditions A/B/C/O reproduce the H1-001
H1-B harness (``evaluation/h1b.py``) exactly; this is tested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from contact_learning.controllers.scripted import nominal_plan, sample_plan_params
from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW, GripperLiftEnv
from contact_learning.evaluation.h1a import grasp_contact
from contact_learning.evaluation.rollout import Recorder, continue_rollout, rollout
from contact_learning.predictors.features import flatten_inputs, history_at

# condition -> (forecaster key, event-triggered refresh at t_c)
CONDITIONS = {
    "A_cached_learned": ("learned_corr", False),
    "B_refreshed_learned": ("learned_corr", True),
    "C_live_only": (None, False),
    "D_refreshed_analytic": ("analytic", True),
    "E_refreshed_nominal_learned": ("learned_nom", True),
    "O_oracle_sim": ("oracle", True),
}
LEARNED = ("A_cached_learned", "B_refreshed_learned", "E_refreshed_nominal_learned")
TRIAL_SEED_BASE = 80_000_000
PILOT_SEED_BASE = 85_000_000


@dataclass
class Cell:
    slew: float
    latency: int          # control steps of tactile sensing delay
    direction: str        # "grip_decrease" | "lift_increase"
    magnitude: float


def trial_setup(env: GripperLiftEnv, seed: int, cell: Cell, cfg: dict, speed_aware: bool = True) -> dict | None:
    """Same random stream as h1b.trial_setup (rng [seed, 3]); plan is speed-aware."""
    T = cfg["T"]
    rng = np.random.default_rng([seed, 3])
    p = sample_plan_params(rng)
    plan = nominal_plan(p, T, env.cfg.max_lift_speed, env.cfg.control_dt,
                        slew=env.cfg.grip_slew_per_step if speed_aware else None)
    nom = rollout(env, seed, plan)
    c = grasp_contact(nom.tactile_clean, env.cfg.contact_threshold)
    valid = [t for t in cfg["correction_times"] if c[t]]
    if not valid:
        return None
    t_c = int(rng.choice(valid))
    plan_c = plan.copy()
    if cell.direction == "grip_decrease":
        plan_c[t_c:, 0] -= 2.0 * cell.magnitude
    elif cell.direction == "lift_increase":
        plan_c[t_c:, 1] += 1.0 * cell.magnitude
    else:
        raise ValueError(cell.direction)
    plan_c = np.clip(plan_c, ACTION_LOW, ACTION_HIGH)
    return {"seed": seed, "plan": plan, "plan_c": plan_c, "t_c": t_c, "g_hold": p.g_hold,
            "nominal_success": env.held({"box_z": np.array([nom.box_z[-1]]), "tactile_clean": nom.tactile_clean[-1]},
                                        cfg["min_lift"])}


def run_condition(env: GripperLiftEnv, setup: dict, cond: str, forecaster, cfg: dict, latency: int = 0) -> dict:
    T, K, H = cfg["T"], cfg["K"], cfg["H"]
    N_min, kappa = cfg["N_min"], cfg["kappa"]
    L = int(latency)
    key, refresh = CONDITIONS[cond]
    t_c, plan, plan_c = setup["t_c"], setup["plan"], setup["plan_c"]
    obs = env.reset(setup["seed"])
    rec = Recorder(obs)
    tac, pro, act = [obs["tactile"]], [obs["proprio"]], []
    u_f = u_l = 0.0
    n_fc, lat, g_exec, n_after = 0, [], [], []

    def chunk(known_plan, t, n):
        a = known_plan[t:t + n].copy()
        if len(a) < n:
            a = np.vstack([a, np.repeat(a[-1:], n - len(a), 0)])
        g = (a[:, 0] + 1) / 2 + u_f + u_l
        a[:, 0] = 2 * np.clip(g, 0, 1) - 1
        return np.clip(a, ACTION_LOW, ACTION_HIGH)

    def deficit(known_plan, t):
        nonlocal n_fc
        t0 = time.perf_counter()
        if key == "oracle":  # NOT deployable: true future from the current state
            s = env.get_state()
            fut = continue_rollout(env, chunk(known_plan, t, H))["tactile_clean"]
            env.set_state(s)
            n_hat = fut[:, :2].mean(1)
        else:
            tau = max(0, t - L)  # latest tactile index available at time t
            h = history_at(np.asarray(tac), np.asarray(pro), np.asarray(act) if act else np.zeros((0, 2)), tau, K)
            executed = np.asarray(act[tau:t]).reshape(-1, 2)  # actions already sent but not yet observed
            A = np.vstack([executed, chunk(known_plan, t, H - len(executed))])
            fut = forecaster.predict(flatten_inputs(h, None)[None], A.reshape(1, -1))[0]
            n_hat = fut[len(executed):, :2].mean(1)  # only steps after t
        lat.append(time.perf_counter() - t0)
        n_fc += 1
        return max(0.0, N_min - float(n_hat.min()))

    for t in range(T):
        active = t >= t_c
        if active and t > t_c:  # live feedback on the latest AVAILABLE measurement
            n_meas = float(np.asarray(tac[max(0, t - L)])[:2].mean())
            u_l += max(0.0, N_min - n_meas) / kappa
        if active and key is not None:
            pre_plan = plan if t == t_c else plan_c
            if t % H == 0:
                d_sched = deficit(pre_plan, t)
                if refresh and t == t_c:
                    u_f += deficit(plan_c, t) / kappa  # replaces the stale forecast
                else:
                    u_f += d_sched / kappa
        known = plan_c if active else plan
        a = chunk(known, t, 1)[0] if active else known[t]
        g_exec.append((a[0] + 1) / 2)
        obs = env.step(a)
        rec.add(a, obs)
        tac.append(obs["tactile"]); pro.append(obs["proprio"]); act.append(a)
        if active:
            n_after.append(float(obs["tactile_clean"][:2].mean()))
    tr = rec.trajectory()
    success = env.held(obs, cfg["min_lift"])
    min_n = float(min(n_after))
    return {"success": bool(success), "min_normal_after_tc": min_n, "dipped": bool(min_n < N_min),
            "slip_m": float((tr.hand_z[-1] - tr.box_z[-1]) - (tr.hand_z[0] - tr.box_z[0])),
            "n_forecasts": n_fc, "inference_ms_median": float(np.median(lat) * 1e3) if lat else 0.0,
            "inference_ms_total": float(np.sum(lat) * 1e3), "u_f": u_f, "u_l": u_l, "_g": np.array(g_exec)}


def run_cell(args) -> list[dict]:
    """Worker: one (cell, trial-seed list) unit. Loads its own models (spawn-safe)."""
    cell, seeds_needed, seed_base, cfg, model_paths, analytic_cal = args
    import pickle

    import torch

    from contact_learning.environments.gripper_lift import EnvConfig
    from contact_learning.predictors.analytic import AnalyticGripForecaster
    from contact_learning.predictors.models import MLPPredictor

    torch.set_num_threads(1)
    env = GripperLiftEnv(EnvConfig.from_dict({**cfg["env"], "grip_slew_per_step": cell.slew}))
    K, H = cfg["K"], cfg["H"]

    def load(p):
        with open(p, "rb") as f:
            d = pickle.load(f)
        return MLPPredictor.from_state(d["mlp"] if "mlp" in d else d)

    fc = {"learned_corr": {s: load(p) for s, p in model_paths["corr"].items()},
          "learned_nom": {s: load(p) for s, p in model_paths["nom"].items()},
          "analytic": AnalyticGripForecaster(K, H, cell.slew, analytic_cal["kappa_hat"], analytic_cal["g0_prior"],
                                             env.cfg.contact_threshold)}
    rows, j, made, skipped = [], 0, 0, 0
    while made < seeds_needed:
        seed = seed_base + j
        j += 1
        st = trial_setup(env, seed, cell, cfg)
        if st is None:
            skipped += 1
            continue
        made += 1
        g_ref = {}
        res = []
        for cond, (key, _) in CONDITIONS.items():
            pseeds = sorted(fc["learned_corr"]) if cond in LEARNED else [-1]
            for ps in pseeds:
                f = fc[key][ps] if key in ("learned_corr", "learned_nom") else fc.get(key)
                r = run_condition(env, st, cond, f, cfg, cell.latency)
                res.append((cond, ps, r))
                if cond == "A_cached_learned":
                    g_ref[ps] = r["_g"]
        gA_mean = np.mean([g_ref[s] for s in g_ref], axis=0)
        for cond, ps, r in res:
            g = r.pop("_g")
            ref = g_ref[ps] if ps in g_ref else gA_mean
            rows.append({"slew": cell.slew, "latency": cell.latency, "direction": cell.direction,
                         "magnitude": cell.magnitude, "seed": seed, "t_c": st["t_c"], "g_hold": st["g_hold"],
                         "nominal_success": st["nominal_success"], "condition": cond, "pred_seed": ps,
                         "action_diff_vs_A": float(np.abs(g - ref).sum()), **r})
    for r in rows:
        r["cell_skipped_seeds"] = skipped
    return rows


# ----------------------------------------------------------------------------- statistics
def per_trial_success(df, cond: str, pred_seeds=None):
    """Matrix trials x seeds for learned conditions; vector for others (index = seed)."""
    d = df[df.condition == cond]
    if cond in LEARNED:
        m = d.pivot_table(index="seed", columns="pred_seed", values="success", aggfunc="mean")
        return m if pred_seeds is None else m[pred_seeds]
    return d.set_index("seed").success.astype(float)


def hier_bootstrap_diff(df, cond_x: str, cond_y: str, n_boot: int = 2000, seed: int = 0):
    """Mean success difference X - Y over trials, with a hierarchical bootstrap that
    resamples trial seeds (clusters) and, independently, predictor seeds for learned
    conditions. ``df`` may span several cells; a trial cluster = one trial seed (all its
    cells move together)."""
    rng = np.random.default_rng(seed)

    def mat(cond):
        d = df[df.condition == cond].copy()
        d["unit"] = list(zip(d.slew, d.latency, d.direction, d.magnitude, d.seed))
        if cond in LEARNED:
            return d.pivot_table(index="unit", columns="pred_seed", values="success", aggfunc="mean")
        return d.set_index("unit").success.astype(float).to_frame(-1)

    X, Y = mat(cond_x), mat(cond_y)
    units = X.index.intersection(Y.index)
    X, Y = X.loc[units].values, Y.loc[units].values
    cluster = np.array([u[-1] for u in units])
    uniq, inv = np.unique(cluster, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    point = float(X.mean() - Y.mean())
    bs = np.empty(n_boot)
    for b in range(n_boot):
        cl = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in cl])
        sx = rng.integers(0, X.shape[1], X.shape[1])
        # Both learned: resample predictor seeds jointly (paired by seed index).
        sy = sx if X.shape[1] == Y.shape[1] and X.shape[1] > 1 else rng.integers(0, Y.shape[1], Y.shape[1])
        bs[b] = X[idx][:, sx].mean() - Y[idx][:, sy].mean()
    per_seed = {}
    if X.shape[1] > 1:
        for j in range(X.shape[1]):
            per_seed[j] = float(X[:, j].mean() - (Y[:, j].mean() if Y.shape[1] == X.shape[1] else Y.mean()))
    return {"diff": point, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n_trial_units": int(len(units)), "n_trial_clusters": int(len(uniq)), "per_pred_seed": per_seed,
            "between_seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else 0.0}


def hier_bootstrap_rate(df, cond: str, n_boot: int = 2000, seed: int = 0):
    d = df[df.condition == cond]
    rng = np.random.default_rng(seed)
    if cond in LEARNED:
        M = d.pivot_table(index="seed", columns="pred_seed", values="success", aggfunc="mean").values
    else:
        M = d.groupby("seed").success.mean().values[:, None]
    bs = [M[rng.integers(0, len(M), len(M))][:, rng.integers(0, M.shape[1], M.shape[1])].mean() for _ in range(n_boot)]
    return [float(M.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
