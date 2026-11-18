"""Analyse H1-002 saved outputs (protocol §8-10). Reads only files written by
experiments/h1/h1_002.py; writes figures to results/h1/figures/h1_002/ and summaries
to results/h1/summaries/h1_002/.

    python experiments/h1/analyze_h1_002.py [--which grid|pilot]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contact_learning.evaluation.h1_002 import LEARNED, hier_bootstrap_diff, hier_bootstrap_rate
from contact_learning.evaluation.stats import bootstrap_ci
from contact_learning.utils.config import REPO_ROOT, load_config

ART = REPO_ROOT / "artifacts" / "h1" / "H1-002"
SUMM = REPO_ROOT / "results" / "h1" / "summaries" / "h1_002"
FIG = REPO_ROOT / "results" / "h1" / "figures" / "h1_002"
DELTA = 0.03
CHALLENGING = (0.2, 0.3, 0.4)
C = {"A_cached_learned": "#eb6834", "B_refreshed_learned": "#2a78d6", "C_live_only": "#8a8985",
     "D_refreshed_analytic": "#1baf7a", "E_refreshed_nominal_learned": "#eda100", "O_oracle_sim": "#e87ba4"}
LABEL = {"A_cached_learned": "A cached learned", "B_refreshed_learned": "B refreshed learned",
         "C_live_only": "C live feedback only", "D_refreshed_analytic": "D refreshed analytic",
         "E_refreshed_nominal_learned": "E refreshed nominal-trained", "O_oracle_sim": "O oracle (non-deployable)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def sel(df, slew=None, lat=None, mags=None, direction="grip_decrease"):
    d = df[df.direction == direction]
    if slew is not None:
        d = d[np.isclose(d.slew, slew)]
    if lat is not None:
        d = d[d.latency == lat]
    if mags is not None:
        d = d[d.magnitude.round(3).isin([round(m, 3) for m in mags])]
    return d


def contrast(d, x, y, n_boot=2000):
    r = hier_bootstrap_diff(d, x, y, n_boot=n_boot)
    lo, hi = r["ci95"]
    r["verdict_vs_delta"] = ("x>y practically" if (r["diff"] >= DELTA and lo > 0) else
                             "equivalent within delta" if (lo > -DELTA and hi < DELTA) else
                             "y>x practically" if (r["diff"] <= -DELTA and hi < 0) else "inconclusive")
    return r


# ----------------------------------------------------------------------------- Stage 1
def stage1(cfg):
    out = {}
    tr = json.load(open(SUMM / "predictor_training.json"))
    P = pd.DataFrame(tr["predictors"])
    P["family"] = np.where(P.corrections_in_training, "corrected-trained", "nominal-trained")
    tab = P.groupby(["slew", "family"]).agg(
        test_corrected_rmse=("test_corrected_rmse", "mean"), test_corrected_rmse_sd=("test_corrected_rmse", "std"),
        test_nominal_rmse=("test_nominal_rmse", "mean"), test_corrected_normal_rmse=("test_corrected_normal_rmse", "mean"),
        test_nominal_normal_rmse=("test_nominal_normal_rmse", "mean"), n_seeds=("pred_seed", "count")).reset_index()
    B = pd.DataFrame(tr["baselines"])
    tab.to_csv(SUMM / "stage1_predictor_test_rmse.csv", index=False)
    B.to_csv(SUMM / "stage1_baselines_test_rmse.csv", index=False)
    out["predictor_test_rmse"] = tab.to_dict(orient="records")
    out["baseline_test_rmse"] = B.to_dict(orient="records")
    nom_fail = P[(~P.corrections_in_training) & (P.val_rmse > P.val_rmse_persistence)] if "val_rmse" in P else P.iloc[:0]
    out["nominal_training_failures"] = nom_fail[["slew", "pred_seed"]].to_dict(orient="records")

    F = pd.read_csv(SUMM / "stage1_forecast_h1a_trials.csv")
    t = F.groupby(["seed", "family"]).agg(E=("E_refreshed", "mean"), En=("En_refreshed", "mean"), Enc=("En_cached", "mean"),
                                          kind=("kind", "first"), me=("magnitude_effective", "first")).reset_index()
    W = t.pivot(index="seed", columns="family", values=["E", "En", "Enc"])
    meta = t.groupby("seed")[["kind", "me"]].first()
    gap = (W["E"]["nom"] - W["E"]["corr"]).values
    m, lo, hi = bootstrap_ci(gap, seed=0)
    out["h1a_trials_gap_nom_minus_corr_allchannels"] = {"mean": m, "ci95": [lo, hi], "n_trials": int(len(gap)),
                                                         "meaningful_gap_gt_0.05": bool(m > 0.05 and lo > 0)}
    rows = []
    edges = [0, 1e-9, 0.075, 0.15, 0.25, 1.0]
    labels = ["0", "(0,.075]", "(.075,.15]", "(.15,.25]", ">.25"]
    b = pd.cut(meta.me, edges, labels=labels, include_lowest=True)
    for (kind, bb), idx in meta.groupby([meta.kind, b], observed=True).groups.items():
        r = {"kind": kind, "bin": bb, "n": len(idx)}
        for fam in ("corr", "nom", "analytic"):
            r[f"En_refreshed_{fam}"] = float(W["En"][fam].loc[idx].mean())
        r["En_cached_corr"] = float(W["Enc"]["corr"].loc[idx].mean())
        r["gap_nom_minus_corr_En"] = bootstrap_ci((W["En"]["nom"] - W["En"]["corr"]).loc[idx].values)
        r["gap_analytic_minus_corr_En"] = bootstrap_ci((W["En"]["analytic"] - W["En"]["corr"]).loc[idx].values)
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(SUMM / "stage1_h1a_trials_by_kind_bin.csv", index=False)
    out["h1a_trials_by_kind_bin"] = S.to_dict(orient="records")
    for fam in ("corr", "nom", "analytic"):
        out[f"h1a_trials_mean_normal_rmse_{fam}"] = float(W["En"][fam].mean())

    # Figure 3: forecast performance, corrected vs nominal training (+ analytic)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    slews = sorted(P.slew.unique())
    x = np.arange(len(slews))
    for k, (fam, col, ls) in enumerate([("corrected-trained", C["B_refreshed_learned"], "-"),
                                         ("nominal-trained", C["E_refreshed_nominal_learned"], "-")]):
        for split, mk in (("corrected", "o"), ("nominal", "s")):
            g = P[P.family == fam].groupby("slew")[f"test_{split}_normal_rmse"]
            ax.errorbar(x + (k - 0.5) * 0.12, g.mean().values, yerr=g.std().values, color=col, marker=mk, ms=5, capsize=2,
                        ls=ls if split == "corrected" else ":", lw=1.6, label=f"{fam} MLP, {split} test episodes")
    for split, mk in (("corrected", "o"), ("nominal", "s")):
        g = B[B.model == "analytic"].set_index("slew").loc[slews][f"test_{split}_normal_rmse"]
        ax.plot(x, g.values, color=C["D_refreshed_analytic"], marker=mk, ms=5, lw=1.6, ls="-" if split == "corrected" else ":",
                label=f"analytic, {split} test episodes")
    ax.set_xticks(x, [f"{s:g}" for s in slews])
    ax.set_xlabel("Gripper slew per step (0.06 = H1-001)")
    ax.set_ylabel("Held-out normal-force RMSE (N)\n(mean ± SD over 5 seeds)")
    ax.set_title("Forecast error by training distribution and speed", loc="left")
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    kinds = ["grip_decrease", "grip_increase", "lift_increase", "lift_decrease"]
    xi = np.arange(len(labels) - 1)
    for fam, col, mk in (("corr", C["B_refreshed_learned"], "o"), ("nom", C["E_refreshed_nominal_learned"], "s"),
                         ("analytic", C["D_refreshed_analytic"], "^")):
        g = S[S.kind == "grip_decrease"].set_index("bin").reindex(labels[1:])
        ax.plot(xi, g[f"En_refreshed_{fam}"].values, color=col, marker=mk, label={"corr": "corrected-trained MLP (refreshed)",
                "nom": "nominal-trained MLP (refreshed)", "analytic": "analytic (refreshed)"}[fam])
    g = S[S.kind == "grip_decrease"].set_index("bin").reindex(labels[1:])
    ax.plot(xi, g["En_cached_corr"].values, color=C["A_cached_learned"], marker="o", ls="--", label="corrected-trained MLP (cached)")
    ax.set_xticks(xi, labels[1:])
    ax.set_xlabel("Effective correction magnitude (H1-001 H1-A trials, grip decrease)")
    ax.set_ylabel("Normal-force RMSE vs. executed (N)")
    ax.set_title("Corrected-action forecasts on H1-001 trials", loc="left")
    ax.legend(fontsize=7)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig3_forecast_training_distribution.png")
    return out


# ----------------------------------------------------------------------------- grid
def grid(cfg, which):
    df = pd.read_csv(ART / f"{which}_episodes.csv")
    out = {"which": which, "n_rows": int(len(df))}
    conds = list(LABEL)
    # Per-cell success with hierarchical CIs + secondary metrics
    rows = []
    for (slew, lat, dirn, mag), d in df.groupby(["slew", "latency", "direction", "magnitude"]):
        for cond in conds:
            dc = d[d.condition == cond]
            if dc.empty:
                continue
            p, lo, hi = hier_bootstrap_rate(d, cond, n_boot=1000)
            dip = dc[dc.dipped]
            rows.append({"slew": slew, "latency": lat, "direction": dirn, "magnitude": mag, "condition": cond,
                         "n_trials": int(dc.seed.nunique()), "success": p, "success_lo": lo, "success_hi": hi,
                         "failure": 1 - p, "recovery_rate": float(dip.success.mean()) if len(dip) else np.nan,
                         "n_dipped_runs": int(len(dip)), "min_normal_after_tc": float(dc.min_normal_after_tc.mean()),
                         "action_diff_vs_A": float(dc.action_diff_vs_A.mean()), "forecasts_per_episode": float(dc.n_forecasts.mean()),
                         "inference_ms_median": float(dc.inference_ms_median[dc.n_forecasts > 0].median()) if (dc.n_forecasts > 0).any() else 0.0,
                         "skipped_seeds": int(dc.cell_skipped_seeds.iloc[0])})
    T = pd.DataFrame(rows)
    T["end_to_end_ms"] = T.latency * 50 + T.inference_ms_median + 50
    T.to_csv(SUMM / f"{which}_cell_success.csv", index=False)
    if which == "pilot":  # exploratory implementation check only (protocol §12): no contrasts, no decisions
        out["note"] = "PILOT: exploratory, excluded from confirmatory analysis; not interpreted"
        return out
    # Cell flags (protocol §11)
    flags = T[(T.condition == "C_live_only") & (T.magnitude == 0) & (T.success < 0.95)][["slew", "latency", "direction"]]
    out["flagged_cells"] = flags.to_dict(orient="records")
    flagged = {(r.slew, r.latency) for r in flags.itertuples() if r.direction == "grip_decrease"}

    def unflagged(d):
        keep = ~d[["slew", "latency"]].apply(tuple, axis=1).isin(flagged)
        return d[keep]

    ref = cfg["reference_slew"]
    K = {}
    K["K1_B_minus_D_ref_L0"] = contrast(sel(df, ref, 0, CHALLENGING), "B_refreshed_learned", "D_refreshed_analytic")
    K["K2_B_minus_D_all_cells"] = contrast(unflagged(sel(df, mags=CHALLENGING)), "B_refreshed_learned", "D_refreshed_analytic")
    K["K3_B_minus_E_ref_L0"] = contrast(sel(df, ref, 0, CHALLENGING), "B_refreshed_learned", "E_refreshed_nominal_learned")
    K["K3_B_minus_E_all_cells"] = contrast(unflagged(sel(df, mags=CHALLENGING)), "B_refreshed_learned", "E_refreshed_nominal_learned")
    fast, slow = max(cfg["slews"]), min(cfg["slews"])
    K["K4_B_minus_C_fast_L0"] = contrast(sel(df, fast, 0, CHALLENGING), "B_refreshed_learned", "C_live_only")
    K["K4_D_minus_C_fast_L0"] = contrast(sel(df, fast, 0, CHALLENGING), "D_refreshed_analytic", "C_live_only")
    K["K5_B_minus_C_slow_L0"] = contrast(sel(df, slow, 0, CHALLENGING), "B_refreshed_learned", "C_live_only")
    K["K5_D_minus_C_slow_L0"] = contrast(sel(df, slow, 0, CHALLENGING), "D_refreshed_analytic", "C_live_only")
    K["K5_B_minus_C_ref_L4"] = contrast(sel(df, ref, 4, CHALLENGING), "B_refreshed_learned", "C_live_only")
    K["K5_D_minus_C_ref_L4"] = contrast(sel(df, ref, 4, CHALLENGING), "D_refreshed_analytic", "C_live_only")
    K["K6_A_minus_C_all_cells"] = contrast(unflagged(sel(df, mags=CHALLENGING)), "A_cached_learned", "C_live_only")
    out["contrasts"] = K

    # Decisions (protocol §10), applied mechanically
    k1, k2 = K["K1_B_minus_D_ref_L0"], K["K2_B_minus_D_all_cells"]
    inside = lambda r: r["ci95"][0] > -DELTA and r["ci95"][1] < DELTA
    if all(r["diff"] >= DELTA and r["ci95"][0] > 0 for r in (k1, k2)):
        rq_a = "Explanation A supported (learned forecasting beats analytic anticipation)"
    elif all(inside(r) or (r["diff"] <= -DELTA and r["ci95"][1] < 0) for r in (k1, k2)):
        rq_a = "Explanation B supported (simple analytic anticipation suffices)"
    else:
        rq_a = "Mixed / inconclusive"
    k3a, k3b = K["K3_B_minus_E_ref_L0"], K["K3_B_minus_E_all_cells"]
    rq_b = ("corrected training needed" if all(r["diff"] >= DELTA and r["ci95"][0] > 0 for r in (k3a, k3b)) else
            "corrected training not needed" if all(inside(r) for r in (k3a, k3b)) else "inconclusive")
    k4 = [K["K4_B_minus_C_fast_L0"], K["K4_D_minus_C_fast_L0"]]
    k5 = [K["K5_B_minus_C_slow_L0"], K["K5_B_minus_C_ref_L4"]]
    if all(inside(r) for r in k4) and any(r["diff"] >= DELTA and r["ci95"][0] > 0 for r in k5):
        rq_c = "Explanation C supported (anticipation benefit is regime-limited: absent with fast actuation and immediate feedback)"
    elif K["K4_B_minus_C_fast_L0"]["diff"] >= DELTA and K["K4_B_minus_C_fast_L0"]["ci95"][0] > 0:
        rq_c = "anticipation useful even with fast actuation and immediate feedback"
    else:
        rq_c = "inconclusive"
    out["decisions"] = {"RQ-a": rq_a, "RQ-b": rq_b, "RQ-c": rq_c}

    # Benefit maps per (slew, latency), challenging pool
    maps = []
    for (slew, lat), d in sel(df, mags=CHALLENGING).groupby(["slew", "latency"]):
        r = {"slew": slew, "latency": lat}
        for nm, (x, y) in {"B_minus_A": ("B_refreshed_learned", "A_cached_learned"), "B_minus_C": ("B_refreshed_learned", "C_live_only"),
                           "D_minus_C": ("D_refreshed_analytic", "C_live_only"), "B_minus_D": ("B_refreshed_learned", "D_refreshed_analytic"),
                           "E_minus_C": ("E_refreshed_nominal_learned", "C_live_only"), "B_minus_E": ("B_refreshed_learned", "E_refreshed_nominal_learned"),
                           "O_minus_B": ("O_oracle_sim", "B_refreshed_learned")}.items():
            c = hier_bootstrap_diff(d, x, y, n_boot=1000)
            r[nm], r[nm + "_lo"], r[nm + "_hi"] = c["diff"], *c["ci95"]
        maps.append(r)
    Mp = pd.DataFrame(maps)
    Mp.to_csv(SUMM / f"{which}_benefit_maps.csv", index=False)
    out["benefit_maps"] = Mp.to_dict(orient="records")

    # Nominal safety (m = 0)
    nom = T[(T.magnitude == 0) & (T.direction == "grip_decrease")]
    out["nominal_m0"] = {"min_success_by_condition": nom.groupby("condition").success.min().to_dict(),
                         "max_mean_action_diff_vs_A": nom.groupby("condition").action_diff_vs_A.max().to_dict()}
    # Direction block
    dirb = T[T.direction != "grip_decrease"]
    out["direction_block"] = dirb.to_dict(orient="records")
    # Cost
    out["cost"] = T.groupby("condition")[["forecasts_per_episode", "inference_ms_median"]].median().to_dict()

    figures(df, T, Mp, cfg, which)
    return out


def figures(df, T, Mp, cfg, which):
    pre = "" if which == "grid" else "pilot_"
    ref = cfg["reference_slew"]
    order = ["A_cached_learned", "B_refreshed_learned", "C_live_only", "D_refreshed_analytic", "E_refreshed_nominal_learned", "O_oracle_sim"]

    # Fig 1: success by controller vs magnitude at reference (L0) and one harder setting
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    for ax, (slew, lat) in zip(axes, [(ref, 0), (ref, 4), (min(cfg["slews"]), 0)]):
        d = T[np.isclose(T.slew, slew) & (T.latency == lat) & (T.direction == "grip_decrease")]
        for k, cond in enumerate(order):
            g = d[d.condition == cond].sort_values("magnitude")
            if g.empty:
                continue
            ax.errorbar(g.magnitude + (k - 2.5) * 0.004, g.success, yerr=[g.success - g.success_lo, g.success_hi - g.success],
                        color=C[cond], marker="o", ms=4, capsize=2, lw=1.6, label=LABEL[cond],
                        ls="--" if cond == "O_oracle_sim" else "-")
        ax.set_title(f"slew {slew:g}/step, sensing latency {lat * 50} ms", loc="left")
        ax.set_xlabel("Grip-decrease magnitude")
    axes[0].set_ylabel("Task success (hierarchical 95% CI)")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, f"{pre}fig1_success_by_controller.png")

    slews, lats = sorted(Mp.slew.unique()), sorted(Mp.latency.unique())

    def heat(ax, col, title, vmax=0.3):
        Z = np.full((len(slews), len(lats)), np.nan)
        for r in Mp.itertuples():
            Z[slews.index(r.slew), lats.index(r.latency)] = getattr(r, col)
        im = ax.imshow(Z, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        for i, s in enumerate(slews):
            for j, l in enumerate(lats):
                r = Mp[np.isclose(Mp.slew, s) & (Mp.latency == l)].iloc[0]
                ax.text(j, i, f"{getattr(r, col):+.3f}\n[{getattr(r, col + '_lo'):+.2f},{getattr(r, col + '_hi'):+.2f}]",
                        ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(lats)), [f"{l * 50}" for l in lats])
        ax.set_yticks(range(len(slews)), [f"{s:g}" for s in slews])
        ax.set_xlabel("Sensing latency (ms)")
        ax.set_ylabel("Gripper slew per step")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False)
        return im

    # Fig 2: learned vs analytic
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    im = heat(axes[0], "B_minus_D", "B − D: refreshed learned minus refreshed analytic")
    heat(axes[1], "B_minus_E", "B − E: corrected-trained minus nominal-trained")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Success difference (m ∈ {0.2,0.3,0.4})")
    save(fig, f"{pre}fig2_learned_vs_analytic.png")

    # Fig 4: benefit vs speed (L0)
    def line(ax, d, xcol, xs, xlabels, xlabel):
        for k, (col, cl, lab) in enumerate([("B_minus_A", C["B_refreshed_learned"], "B − A (refresh vs. cached)"),
                                            ("B_minus_C", C["A_cached_learned"], "B − C (learned anticipation vs. live)"),
                                            ("D_minus_C", C["D_refreshed_analytic"], "D − C (analytic anticipation vs. live)")]):
            g = d.set_index(xcol).loc[xs]
            ax.errorbar(np.arange(len(xs)) + (k - 1) * 0.06, g[col], yerr=[g[col] - g[col + "_lo"], g[col + "_hi"] - g[col]],
                        color=cl, marker="o", ms=4, capsize=2, lw=1.8, label=lab)
        ax.axhspan(-DELTA, DELTA, color=GRID, alpha=0.6, lw=0, label=f"±δ = {DELTA}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(xs)), xlabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Success difference (m ∈ {0.2,0.3,0.4})")
        ax.legend(fontsize=7)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    line(ax, Mp[Mp.latency == 0], "slew", slews, [f"{s:g}\n({1 / s * 0.05:.2f} s stroke)" for s in slews], "Gripper slew per step (sensing latency 0)")
    ax.set_title("Anticipation benefit vs. gripper speed", loc="left")
    save(fig, f"{pre}fig4_benefit_vs_speed.png")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    line(ax, Mp[np.isclose(Mp.slew, ref)], "latency", lats, [f"{l * 50} ms" for l in lats], f"Sensing latency (slew {ref:g}, H1-001 speed)")
    ax.set_title("Anticipation benefit vs. sensing latency", loc="left")
    save(fig, f"{pre}fig5_benefit_vs_latency.png")

    # Fig 6: interactions — success of C, B, D across magnitude for every speed x latency
    fig, axes = plt.subplots(len(slews), len(lats), figsize=(11, 2.1 * len(slews)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, s in enumerate(slews[::-1]):
        for j, l in enumerate(lats):
            ax = axes[i, j]
            d = T[np.isclose(T.slew, s) & (T.latency == l) & (T.direction == "grip_decrease")]
            for cond in ("C_live_only", "B_refreshed_learned", "D_refreshed_analytic", "A_cached_learned"):
                g = d[d.condition == cond].sort_values("magnitude")
                ax.plot(g.magnitude, g.success, color=C[cond], marker="o", ms=2.5, lw=1.3, label=LABEL[cond])
            ax.set_title(f"slew {s:g}, {l * 50} ms", fontsize=7.5, loc="left")
            ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Grip decrease m", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="lower left")
    save(fig, f"{pre}fig6_interactions_magnitude_speed_latency.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="grid", choices=["grid", "pilot"])
    a = ap.parse_args()
    cfg = load_config(REPO_ROOT / "configs" / "experiments" / "h1_002.yaml")
    out = {}
    if a.which == "grid":
        out["stage1"] = stage1(cfg)
    out[a.which] = grid(cfg, a.which)
    with open(SUMM / f"{a.which}_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out[a.which].get("decisions", {}), indent=1))
    for k, v in out[a.which].get("contrasts", {}).items():
        print(f"{k:28s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['verdict_vs_delta']}")


if __name__ == "__main__":
    main()
