"""Analyse saved H1 outputs and produce figures + compact summaries.

    python experiments/h1/analyze_results.py --experiment-id H1-001

Reads only files written by run_experiment.py (results/h1/summaries/*). Every
number in the figures is computed here from those files; nothing is hand-entered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contact_learning.evaluation.stats import bootstrap_ci, holm, wilcoxon_p, wilson_ci
from contact_learning.utils.config import REPO_ROOT

SUMM = REPO_ROOT / "results" / "h1" / "summaries"
FIG = REPO_ROOT / "results" / "h1" / "figures"
EDGES = [0.0, 1e-9, 0.035, 0.075, 0.15, 0.25, 0.4, 1.0]
BIN_LABELS = ["0", "(0,.035]", "(.035,.075]", "(.075,.15]", "(.15,.25]", "(.25,.4]", ">.4"]
# Reference palette slots (validated categorical order) + neutral ink.
C = {"refreshed": "#2a78d6", "cached": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "grey": "#8a8985", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
})


def mag_bin(x):
    return pd.cut(x, EDGES, labels=BIN_LABELS, include_lowest=True, right=True)


def trial_level(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """One row per trial: metrics averaged over predictor seeds."""
    d = df[df.predictor == predictor]
    meta = ["seed", "t_c", "stratum", "kind", "magnitude_requested", "magnitude_effective", "phase",
            "near_transition", "contact_changing", "outcome_change_rmse"]
    num = ["E_cached", "E_refreshed", "E_sham", "D", "D_sham", "En_cached", "En_refreshed", "CSE_cached",
           "CSE_refreshed", "E_cached_on_orig", "forecast_shift", "E_cached_clean", "E_refreshed_clean"]
    t = d.groupby(meta, as_index=False)[num].mean()
    t["bin"] = mag_bin(t.magnitude_effective)
    return t


def bin_table(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in t.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n"] = len(g)
        for col in ("E_cached", "E_refreshed", "D", "D_sham", "outcome_change_rmse", "forecast_shift"):
            m, lo, hi = bootstrap_ci(g[col].values, seed=0)
            r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
        r["p_wilcoxon_D"] = wilcoxon_p(g.D.values)
        r["frac_D_pos"] = float((g.D > 0).mean())
        r["rel_D"] = float(g.D.mean() / g.E_cached.mean()) if g.E_cached.mean() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def errbar(ax, x, tab, col, color, label, marker="o", offset=0.0):
    y = tab[col].values
    yerr = np.vstack([y - tab[col + "_lo"].values, tab[col + "_hi"].values - y])
    ax.errorbar(np.asarray(x) + offset, y, yerr=yerr, color=color, marker=marker, ms=5, capsize=2.5,
                lw=2, label=label)


def analyze_h1a(eid: str) -> dict:
    df = pd.read_csv(REPO_ROOT / "artifacts" / "h1" / eid / "h1a_trials.csv")
    preds = json.load(open(SUMM / f"{eid}_predictors.json"))
    out: dict = {"experiment_id": eid, "n_trials": int(df.seed.nunique()),
                 "predictor_seeds": sorted(df.pred_seed.unique().tolist())}

    # ---------- predictor quality vs baselines (held-out predictor-test episodes)
    q = {}
    for rep in preds["reports"]:
        for name, r in rep["models"].items():
            q.setdefault(name, []).append(r["test_rmse"])
    out["predictor_test_rmse"] = {k: {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                  for k, v in q.items()}
    out["mlp_beats_persistence_all_seeds"] = bool(all(m < p for m, p in zip(q["mlp"], q["persistence"])))
    out["mlp_beats_history_only_all_seeds"] = bool(all(m < p for m, p in zip(q["mlp"], q["mlp_history_only"])))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    names = ["persistence", "mlp_history_only", "ridge", "mlp"]
    labels = ["Persistence", "MLP (history only)", "Ridge (action-cond.)", "MLP (action-cond.)"]
    cols = [C["grey"], C["grey"], C["aqua"], C["refreshed"]]
    for i, n in enumerate(names):
        v = np.array(q[n])
        ax.barh(i, v.mean(), color=cols[i], height=0.6)
        ax.scatter(v, np.full(len(v), i), color=C["ink"], s=8, zorder=3)
        ax.text(v.mean() + 0.01, i, f"{v.mean():.3f}", va="center", color=C["ink"], fontsize=8)
    ax.set_yticks(range(len(names)), labels)
    ax.set_xlabel("Held-out forecast RMSE (N), 10-step horizon, 4 channels")
    ax.set_title(f"Predictor quality on held-out episodes (bars: mean; dots: {len(q['mlp'])} seeds)", loc="left")
    ax.grid(axis="y", visible=False)
    savefig(fig, f"{eid}_predictor_quality.png")

    # ---------- implementation checks on the trial data
    z = df[df.magnitude_effective == 0]
    out["check_zero_correction_D_exactly_zero"] = bool((z.D == 0).all())
    blind = df[df.predictor.isin(["persistence", "mlp_history_only"])]
    out["check_action_blind_D_exactly_zero"] = bool((blind.D == 0).all())
    out["check_prefix_reproducible"] = bool(df.prefix_reproducible.all())
    out["n_trials_zero_effective_magnitude"] = int(df[df.predictor == "mlp"].groupby("seed").magnitude_effective.first().eq(0).sum())

    results = {}
    for pname in ("mlp", "ridge"):
        t = trial_level(df, pname)
        tab = bin_table(t, ["bin"])
        nz = tab["bin"] != "0"
        tab.loc[nz, "p_holm_D"] = holm(tab.loc[nz, "p_wilcoxon_D"].tolist())
        tab.to_csv(SUMM / f"{eid}_h1a_{pname}_by_magnitude.csv", index=False)
        large = t[t.magnitude_effective >= 0.15]
        m, lo, hi = bootstrap_ci(large.D.values, seed=0)
        ms, los, his = bootstrap_ci(large.D_sham.values, seed=0)
        per_seed = []
        dd = df[(df.predictor == pname) & (df.magnitude_effective >= 0.15)]
        for s, g in dd.groupby("pred_seed"):
            a, b, c = bootstrap_ci(g.D.values, seed=int(s))
            per_seed.append({"seed": int(s), "mean_D": a, "ci95": [b, c], "n": int(len(g))})
        results[pname] = {
            "large_corrections_ge_0.15": {"n_trials": int(len(large)), "mean_D": m, "ci95": [lo, hi],
                                          "p_wilcoxon": wilcoxon_p(large.D.values), "mean_E_cached": float(large.E_cached.mean()),
                                          "mean_E_refreshed": float(large.E_refreshed.mean()),
                                          "frac_D_pos": float((large.D > 0).mean()),
                                          "mean_D_sham": ms, "D_sham_ci95": [los, his]},
            "per_seed_large": per_seed,
            "seeds_with_positive_mean_D_large": int(sum(p["mean_D"] > 0 for p in per_seed)),
            "by_magnitude": tab.to_dict(orient="records"),
        }
        for grp in (["kind", "bin"], ["phase", "bin"], ["contact_changing", "bin"], ["kind"], ["phase"], ["contact_changing"]):
            g = bin_table(t[t.magnitude_effective > 0], grp)
            g.to_csv(SUMM / f"{eid}_h1a_{pname}_by_{'_'.join(grp)}.csv", index=False)
            results[pname]["by_" + "_".join(grp)] = g.to_dict(orient="records")
        # Mismatch relative to realized contact change (is D just tracking the outcome change?)
        r = np.corrcoef(t.outcome_change_rmse, t.D)[0, 1]
        results[pname]["corr_D_vs_outcome_change"] = float(r)
        small = t[(t.magnitude_effective > 0) & (t.outcome_change_rmse < 0.05)]
        results[pname]["negligible_outcome_change"] = {"n": int(len(small)), "mean_D": bootstrap_ci(small.D.values)}
        results[pname]["refresh_hurts_frac_nonzero"] = float((t[t.magnitude_effective > 0].D < 0).mean())
    out["results"] = results

    # ---------- latency
    lat = df.groupby("predictor").latency_ms_median.median()
    out["median_single_forecast_latency_ms"] = lat.to_dict()

    # ---------- figures (primary predictor: mlp)
    t = trial_level(df, "mlp")
    tab = bin_table(t, ["bin"])
    x = np.arange(len(tab))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, col, color, name in ((axes[0], "E_cached", C["cached"], "Cached (original action)"),
                                 (axes[1], "E_refreshed", C["refreshed"], "Refreshed (corrected action)")):
        ax.scatter(np.searchsorted(EDGES[1:], t.magnitude_effective.values, side="left") + np.random.default_rng(0).uniform(-0.25, 0.25, len(t)),
                   t[col], s=3, color=color, alpha=0.18, lw=0)
        errbar(ax, x, tab, col, color, "mean ± 95% CI")
        ax.set_xticks(x, tab["bin"], rotation=30)
        ax.set_title(f"{name} forecast", loc="left")
    for ax in axes:
        ax.set_xlabel("Effective correction magnitude\n(fraction of action range)")
    axes[1].text(1.0, -0.42, "y-axis clipped at the 99th percentile of trial errors", transform=axes[1].transAxes, ha="right", fontsize=7, color=C["ink2"])
    axes[0].set_ylabel("Forecast RMSE vs. executed trajectory (N)")
    axes[0].legend(loc="upper left")
    ax2 = axes[0]
    ax2.set_ylim(0, np.quantile(np.r_[t.E_cached, t.E_refreshed], 0.99) * 1.05)
    savefig(fig, f"{eid}_fig1_error_vs_magnitude.png")

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    errbar(ax, x, tab, "D", C["refreshed"], "Refreshed with the true correction (D)", offset=-0.08)
    errbar(ax, x, tab, "D_sham", C["grey"], "Negative control: refreshed with a wrong correction", marker="s", offset=0.08)
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xticks(x, [f"{b}\nn={n}" for b, n in zip(tab["bin"], tab["n"])], fontsize=7)
    ax.set_xlabel("Effective correction magnitude (fraction of action range)")
    ax.set_ylabel("E_cached − E_refreshed (N)\n> 0: refreshing helps")
    ax.set_title("Paired forecast-error difference (MLP; trial-bootstrap 95% CI)", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    savefig(fig, f"{eid}_fig2_paired_difference.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), sharey=True)
    tnz = t[t.magnitude_effective > 0]
    kinds = [("grip_decrease", C["refreshed"]), ("grip_increase", C["cached"]), ("lift_decrease", C["aqua"]), ("lift_increase", C["yellow"])]
    bl = BIN_LABELS[1:]
    for k, (kind, col) in enumerate(kinds):
        g = bin_table(tnz[tnz.kind == kind], ["bin"])
        xi = [bl.index(b) for b in g["bin"]]
        errbar(axes[0], xi, g, "D", col, kind.replace("_", " "), offset=(k - 1.5) * 0.08)
    axes[0].set_title("By correction type", loc="left")
    for k, (ph, col) in enumerate([("near_transition", C["cached"]), ("stable_contact", C["refreshed"]), ("no_contact", C["grey"])]):
        g = bin_table(tnz[tnz.phase == ph], ["bin"])
        if len(g):
            xi = [bl.index(b) for b in g["bin"]]
            errbar(axes[1], xi, g, "D", col, f"{ph.replace('_', ' ')} (n={int(g.n.sum())})", offset=(k - 1) * 0.1)
    axes[1].set_title("By task phase at correction", loc="left")
    for k, (cc, col, lab) in enumerate([(True, C["cached"], "contact-changing"), (False, C["refreshed"], "contact-preserving")]):
        g = bin_table(tnz[tnz.contact_changing == cc], ["bin"])
        xi = [bl.index(b) for b in g["bin"]]
        errbar(axes[2], xi, g, "D", col, f"{lab} (n={int(g.n.sum())})", offset=(k - 0.5) * 0.1)
    axes[2].set_title("By realized contact outcome", loc="left")
    for ax in axes:
        ax.axhline(0, color=C["ink2"], lw=0.8)
        ax.set_xticks(range(len(bl)), bl, rotation=30, fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.set_xlabel("Effective correction magnitude\n(mean ± trial-bootstrap 95% CI)")
    axes[0].set_ylabel("E_cached − E_refreshed (N)")
    savefig(fig, f"{eid}_fig3_by_type_phase_outcome.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(tnz.outcome_change_rmse, tnz.D, s=4, color=C["refreshed"], alpha=0.35, lw=0)
    lim = max(tnz.outcome_change_rmse.max(), 0.1)
    ax.plot([0, lim], [0, lim], color=C["ink2"], lw=0.8, ls="--", label="D = realized change (perfect predictor)")
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("Realized contact change from the correction\nRMSE(Y_corr, Y_orig) (N)")
    ax.set_ylabel("E_cached − E_refreshed (N)")
    ax.set_title("Mismatch tracks how much the correction changed contact", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    savefig(fig, f"{eid}_fig4_D_vs_realized_change.png")

    # Per-horizon error for large corrections (raw npz has per-horizon arrays).
    npz_path = REPO_ROOT / "artifacts" / "h1" / eid / "h1a_trials.npz"
    if npz_path.exists():
        raw = np.load(npz_path)
        seeds_arr = raw["seeds"]
        m_eff = df[(df.predictor == "mlp") & (df.pred_seed == out["predictor_seeds"][0])].set_index("seed").loc[seeds_arr].magnitude_effective.values
        sel = m_eff >= 0.15
        pc = np.mean([raw[f"seed{s}_mlp_PH_cached"][sel] for s in out["predictor_seeds"]], axis=0)
        pr = np.mean([raw[f"seed{s}_mlp_PH_refreshed"][sel] for s in out["predictor_seeds"]], axis=0)
        H = pc.shape[1]
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for arr, col, lab in ((pc, C["cached"], "Cached"), (pr, C["refreshed"], "Refreshed")):
            mean = arr.mean(0)
            ci = np.array([bootstrap_ci(arr[:, k], n_boot=1000)[1:] for k in range(H)])
            kk = np.arange(1, H + 1)
            ax.plot(kk, mean, color=col, marker="o", ms=4, label=lab)
            ax.fill_between(kk, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
        ax.set_xlabel("Forecast step k (× 50 ms)")
        ax.set_ylabel("RMSE at step k (N)")
        ax.set_title(f"Error by horizon, corrections ≥ 0.15 (n={int(sel.sum())})", loc="left")
        ax.legend()
        savefig(fig, f"{eid}_fig5_error_by_horizon.png")
        out["per_horizon_large"] = {"cached": pc.mean(0).tolist(), "refreshed": pr.mean(0).tolist()}
    return out


def analyze_env(eid: str):
    p = SUMM / f"{eid}_env_validation.json"
    if not p.exists():
        return None
    v = json.load(open(p))
    gs = v["grip_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    axes[0].plot(gs["g"], gs["expected_normal"], color=C["grey"], ls="--", label="Rigid-contact model kp·(target − contact)")
    axes[0].plot(gs["g"], gs["normal_l"], color=C["refreshed"], marker="o", ms=3, lw=1.5, label="Simulated left-pad normal force")
    axes[0].set_xlabel("Grip command g")
    axes[0].set_ylabel("Steady normal force (N)")
    axes[0].set_title("Contact signal responds to grip command", loc="left")
    axes[0].legend(fontsize=7)
    rows = v["lift"]["rows"]
    w = [r["weight_N"] for r in rows if r["held"]]
    ts = [r["tangential_sum_N"] for r in rows if r["held"]]
    axes[1].scatter(w, ts, color=C["refreshed"], s=14)
    lim = [0, max(w + ts) * 1.1]
    axes[1].plot(lim, lim, color=C["grey"], ls="--", lw=1, label="tangential = weight")
    axes[1].set_xlabel("Object weight m·g (N)")
    axes[1].set_ylabel("Σ pad tangential force (N)")
    axes[1].set_title("Held object: friction carries the weight", loc="left")
    axes[1].legend(fontsize=7)
    savefig(fig, f"{eid}_env_validation.png")
    return {"grip_sweep_max_dev_N": gs["max_abs_dev_from_expected_N"], "coulomb_agreement": v["capacity"]["agreement"],
            "lift_max_rel_err": v["lift"]["max_rel_err_tangential_vs_weight"], "determinism": v["determinism"]}


def analyze_h1b(eid: str):
    p = SUMM / f"{eid}_h1b_summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    bm = s["by_magnitude"]
    mags = [r["magnitude"] for r in bm]
    conds = [("cached", C["cached"], "A cached"), ("refreshed", C["refreshed"], "B refreshed"),
             ("live_only", C["grey"], "C live feedback only"), ("fixed_rate", C["aqua"], "D fixed-rate (every 5 steps)"),
             ("oracle", C["magenta"], "E oracle (non-deployable)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for k, (c, col, lab) in enumerate(conds):
        y = np.array([r[f"success_{c}"][0] for r in bm])
        lo = np.array([r[f"success_{c}"][1] for r in bm])
        hi = np.array([r[f"success_{c}"][2] for r in bm])
        axes[0].errorbar(np.array(mags) + (k - 2) * 0.004, y, yerr=[y - lo, hi - y], color=col, marker="o", ms=4, capsize=2, lw=1.8, label=lab)
    unc = uncontrolled_baseline(s["config"])  # the exact config saved with this run
    s["posthoc_uncontrolled"] = unc
    axes[0].plot(mags, [unc[str(m)][0] for m in mags], color=C["ink2"], ls=":", lw=1.5, label="No safety controller (post-hoc reference)")
    axes[0].set_xlabel("Grip-decrease correction magnitude (fraction of grip range)")
    axes[0].set_ylabel("Task success rate (Wilson 95% CI)")
    axes[0].set_title("Closed-loop success by forecast condition", loc="left")
    axes[0].legend(fontsize=7, loc="lower left")
    for k, (a, b, col, lab) in enumerate([("refreshed", "cached", C["refreshed"], "B − A (refresh vs. cached)"),
                                          ("refreshed", "live_only", C["grey"], "B − C (refresh vs. live only)"),
                                          ("fixed_rate", "cached", C["aqua"], "D − A (fixed-rate vs. cached)")]):
        d = np.array([r[f"diff_{a}_minus_{b}"] for r in bm])
        axes[1].errorbar(np.array(mags) + (k - 1) * 0.004, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], color=col, marker="o", ms=4, capsize=2, lw=1.8, label=lab)
    axes[1].axhline(0, color=C["ink2"], lw=0.8)
    axes[1].set_xlabel("Grip-decrease correction magnitude")
    axes[1].set_ylabel("Paired success difference\n(trial-bootstrap 95% CI)")
    axes[1].set_title("Paired differences", loc="left")
    axes[1].legend(fontsize=7)
    fig.subplots_adjust(wspace=0.38)
    savefig(fig, f"{eid}_fig6_closed_loop_success.png")
    return s


def uncontrolled_baseline(cfg: dict):
    """POST-HOC diagnostic (not registered): success of the corrected plan with no
    safety controller at all, on exactly the H1-B trial seeds. Shows how much the
    corrections threaten the grasp. Deterministic; re-simulated here."""
    from contact_learning.environments.gripper_lift import EnvConfig, GripperLiftEnv
    from contact_learning.evaluation.h1b import trial_setup
    from contact_learning.evaluation.rollout import rollout
    from contact_learning.utils import seeding
    env = GripperLiftEnv(EnvConfig.from_dict(cfg.get("env")))
    res = {}
    for mi, mag in enumerate(cfg["h1b"]["magnitudes"]):
        ok, i = [], 0
        while len(ok) < cfg["h1b"]["n_trials_per_magnitude"]:
            st = trial_setup(env, seeding.H1B_BASE + mi * 100_000 + i, mag, cfg)
            i += 1
            if st is None:
                continue
            tr = rollout(env, st["seed"], st["plan_c"])
            ok.append(env.held({"box_z": np.array([tr.box_z[-1]]), "tactile_clean": tr.tactile_clean[-1]}, cfg["h1b"]["min_lift"]))
        res[str(mag)] = list(wilson_ci(int(np.sum(ok)), len(ok)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default="H1-001")
    a = ap.parse_args()
    eid = a.experiment_id
    out = {"env_validation": analyze_env(eid), "h1a": analyze_h1a(eid)}
    h1b = analyze_h1b(eid)
    if h1b is not None:
        out["h1b"] = {k: v for k, v in h1b.items() if k not in ("config", "provenance")}
        out["h1b_posthoc_uncontrolled_success"] = h1b["posthoc_uncontrolled"]
    with open(SUMM / f"{eid}_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out["h1a"][k] for k in out["h1a"] if k.startswith("check") or k.startswith("mlp_beats")}, indent=1))
    for p in ("mlp", "ridge"):
        print(p, json.dumps(out["h1a"]["results"][p]["large_corrections_ge_0.15"], indent=1))


if __name__ == "__main__":
    main()
