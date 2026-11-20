"""Registered analysis for H2-001A (docs/experiments/H2-001A_PROTOCOL.md §8-9).

Usage: python experiments/h2/analyze_h2_001a.py --art artifacts/h2_001a [--tag h2_001a]
Writes results/h2/summaries/<tag>/*.{csv,json,md} and results/h2/figures/<tag>/*.png.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from contact_learning.evaluation.stats import holm  # noqa: E402
from contact_learning.h2a.stats import crossed_bootstrap, mean_ci, paired_matrix, verdict  # noqa: E402

DELTA = 0.03
NB = 4000
P_ORDER = ["A_cached", "B_refresh", "C_null", "E_fixed3", "F_gate", "Fn_gate_null", "G_random", "O_oracle"]
ALL = P_ORDER + ["D_live_only", "R_reactive", "X_teacher"]
LABEL = {"A_cached": "A cached", "B_refresh": "B refresh", "C_null": "C suppressed", "E_fixed3": "E every 3",
         "F_gate": "F gate", "Fn_gate_null": "Fn gate→null", "G_random": "G random", "O_oracle": "O oracle*",
         "D_live_only": "D live-only", "R_reactive": "R reactive", "X_teacher": "X teacher*"}
COLOR = {"A_cached": "#4C6FB5", "B_refresh": "#D9822B", "C_null": "#7A7A7A", "E_fixed3": "#C2A33A",
         "F_gate": "#3A9E6F", "Fn_gate_null": "#8BC4A3", "G_random": "#B05FA0", "O_oracle": "#BBBBBB",
         "D_live_only": "#2E2E2E", "R_reactive": "#A8432E", "X_teacher": "#DDDDDD"}


def load(art: Path):
    recs, probes, calib = [], [], {}
    for d in sorted(art.glob("L*/seed*")):
        L, s = int(d.parent.name[1:]), int(d.name[4:])
        if not (d / "records.pkl").exists():
            continue
        c = pickle.load(open(d / "calib.pkl", "rb"))
        calib[(L, s)] = {k: c[k] for k in ("theta", "ood_q99_nominal", "ood_median_nominal", "n_disc")}
        for r in pickle.load(open(d / "records.pkl", "rb")):
            for p in r.pop("probe", []) or []:
                probes.append(dict(p, latency=L, student_seed=s, seed=r["seed"], kind=r["kind"],
                                   recoverable=r["recoverable"], severity=r["severity"], ref=r["ref"]))
            r.pop("hidden", None); r.pop("disc_trace", None)
            recs.append(r)
    df = pd.DataFrame(recs)
    df["cls"] = np.where(df.kind == "none", "nominal", np.where(df.recoverable == True, "recoverable", "unrecoverable"))  # noqa: E712
    pr = pd.DataFrame(probes)
    pr["ood_thr"] = [calib[(l, s)]["ood_q99_nominal"] for l, s in zip(pr.latency, pr.student_seed)]
    return df, pr, calib


def sub(df, L, cls, kinds=None):
    x = df[(df.latency == L) & (df.cls == cls)]
    return x if kinds is None else x[x.kind.isin(kinds)]


def contrast(d, a, b, metric="success"):
    D = paired_matrix(d, a, b, metric)
    return crossed_bootstrap(D, NB) if D.size else None


def fmt(r):
    return f"{r['est']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts/h2_001a")
    ap.add_argument("--tag", default="h2_001a")
    a = ap.parse_args()
    art = ROOT / a.art
    sdir, fdir = ROOT / "results/h2/summaries" / a.tag, ROOT / "results/h2/figures" / a.tag
    sdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    df, pr, calib = load(art)
    lats = sorted(df.latency.unique())
    L1 = 1 if 1 in lats else lats[0]
    seen = ["load", "push"]
    out = {"n_records": len(df), "latencies": lats, "seeds": sorted(df.student_seed.unique().tolist()),
           "calibration": {f"L{k[0]}_s{k[1]}": v for k, v in calib.items()}}

    # ---------------- success table
    rows = []
    for L in lats:
        for cls in ("nominal", "recoverable", "unrecoverable"):
            for kind in (["none"] if cls == "nominal" else ["load", "push", "friction", "seen"]):
                d = sub(df, L, cls, seen if kind == "seen" else [kind])
                for c in ALL:
                    if (d.condition == c).any():
                        m = mean_ci(d, c, "success", NB)
                        rows.append({"latency": L, "class": cls, "kind": kind, "condition": c, "success": m["est"],
                                     "ci_lo": m["ci"][0], "ci_hi": m["ci"][1], "n_episodes": m["n_episodes"],
                                     "n_seeds": m["n_seeds"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(sdir / "success_by_condition.csv", index=False)

    # ---------------- gates
    gates = {}
    for L in lats:
        gates[f"L{L}_nominal_success_A"] = float(sub(df, L, "nominal").query("condition=='A_cached'").success.mean())
        gates[f"L{L}_teacher_recovery"] = float(sub(df, L, "recoverable").query("condition=='X_teacher'").success.mean())
        gates[f"L{L}_pass"] = bool(gates[f"L{L}_nominal_success_A"] >= 0.90 and gates[f"L{L}_teacher_recovery"] >= 0.95)
    out["gates"] = gates

    # ---------------- primary family (L=1, recoverable, seen kinds)
    prim = sub(df, L1, "recoverable", seen)
    nom1 = sub(df, L1, "nominal")
    fam = [("P1", "B_refresh", "A_cached"), ("P2", "C_null", "A_cached"), ("P3", "F_gate", "A_cached"),
           ("P4", "F_gate", "G_random")]
    res = {k: contrast(prim, x, y) for k, x, y in fam}
    padj = holm([res[k]["p_boot"] for k, _, _ in fam])
    primary = {}
    for (k, x, y), pa in zip(fam, padj):
        r = res[k]
        primary[k] = {"contrast": f"{x} - {y}", **{kk: r[kk] for kk in ("est", "ci", "ci_episodes_only", "ci_seeds_only", "p_boot", "n_seeds", "n_episodes", "per_seed")},
                      "p_holm": pa, "verdict": verdict(r, DELTA, pa)}
    p5 = contrast(nom1, "F_gate", "A_cached")
    primary["P5"] = {"contrast": "F_gate - A_cached (nominal)", **{kk: p5[kk] for kk in ("est", "ci", "p_boot")},
                     "non_inferior": bool(p5["ci"][0] > -DELTA)}
    out["primary"] = primary

    # ---------------- secondary contrasts
    sec = []
    for L in lats:
        for pop, d in (("seen_recoverable", sub(df, L, "recoverable", seen)),
                       ("friction_recoverable", sub(df, L, "recoverable", ["friction"])),
                       ("nominal", sub(df, L, "nominal")),
                       ("all_unrecoverable", sub(df, L, "unrecoverable"))):
            for x in ["B_refresh", "C_null", "E_fixed3", "F_gate", "Fn_gate_null", "G_random", "O_oracle", "D_live_only", "R_reactive", "X_teacher"]:
                r = contrast(d, x, "A_cached")
                if r:
                    sec.append({"latency": L, "population": pop, "contrast": f"{x} - A_cached", "est": r["est"],
                                "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "ci_ep_lo": r["ci_episodes_only"][0],
                                "ci_ep_hi": r["ci_episodes_only"][1], "ci_seed_lo": r["ci_seeds_only"][0],
                                "ci_seed_hi": r["ci_seeds_only"][1], "p_boot": r["p_boot"]})
            r = contrast(d, "F_gate", "G_random")
            if r:
                sec.append({"latency": L, "population": pop, "contrast": "F_gate - G_random", "est": r["est"],
                            "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "p_boot": r["p_boot"]})
    secdf = pd.DataFrame(sec)
    secdf.to_csv(sdir / "secondary_contrasts.csv", index=False)

    # severity terciles (seen, recoverable), success by condition
    sev = []
    for L in lats:
        d = sub(df, L, "recoverable", seen).copy()
        d["sev_bin"] = pd.cut(d.severity, [0.55, 0.65, 0.75, 0.85], include_lowest=True).astype(str)
        sev.append(d.groupby(["sev_bin", "condition"]).success.mean().unstack().assign(latency=L))
    pd.concat(sev).to_csv(sdir / "success_by_severity.csv")

    # ---------------- reactive / live-only question
    reac = {}
    for L in lats:
        d, n = sub(df, L, "recoverable", seen), sub(df, L, "nominal")
        best = max(["A_cached", "B_refresh", "C_null", "F_gate"], key=lambda c: d[d.condition == c].success.mean())
        for x in ("R_reactive", "D_live_only"):
            rr, rn = contrast(d, x, best), contrast(n, x, best)
            reac[f"L{L}_{x}_vs_{best}"] = {"recovery": fmt(rr), "nominal": fmt(rn),
                                           "matches": bool(rr["ci"][0] > -DELTA and rn["ci"][0] > -DELTA)}
    out["reactive"] = reac

    # ---------------- mechanisms
    mech = {}
    for L in lats:
        d = sub(df, L, "recoverable", seen)
        m1 = contrast(d, "A_cached", "B_refresh", "fc_err_post")
        a_dist = mean_ci(d, "A_cached", "fc_err_post", NB)
        a_nom = mean_ci(sub(df, L, "nominal"), "A_cached", "fc_err_pseudo", NB)
        M1 = bool(m1["ci"][0] > 0 and a_dist["ci"][0] > a_nom["ci"][1])
        # M2: nominal lift-off probes
        pn = pr[(pr.latency == L) & (pr.kind == "none")].copy()
        pn["d_null"] = (pn.g_cached - pn.g_null).abs()
        pn["d_refresh"] = (pn.g_cached - pn.g_refresh).abs()
        ep = pn.groupby(["student_seed", "seed"])[["d_null", "d_refresh"]].mean()
        sens = crossed_bootstrap(ep.d_null.unstack().to_numpy(), NB)
        c_nom = contrast(sub(df, L, "nominal"), "C_null", "A_cached")
        c_rec = contrast(d, "C_null", "A_cached")
        beh = any(abs(r["est"]) >= DELTA and (r["ci"][0] > 0 or r["ci"][1] < 0) for r in (c_nom, c_rec))
        M2 = "meaningful" if (sens["est"] >= 0.01 and beh) else ("weak" if sens["est"] >= 0.01 or beh else "absent")
        # M3: disturbed recoverable probes (seen)
        pd_ = pr[(pr.latency == L) & (pr.recoverable == True) & (pr.kind.isin(seen))].copy()  # noqa: E712
        for k in ("refresh", "null", "shuffle", "oracle"):
            if "g_" + k in pd_:
                pd_["chg_" + k] = (pd_.g_cached - pd_["g_" + k]).abs()
                pd_["int_" + k] = (pd_.g_cached - pd_.g_star).abs() - (pd_["g_" + k] - pd_.g_star).abs()
                pd_["under_" + k] = np.maximum(0, pd_.g_star - pd_.g_cached) - np.maximum(0, pd_.g_star - pd_["g_" + k])
        epm = pd_.groupby(["student_seed", "seed"]).mean(numeric_only=True)
        m3 = {k: {"change": crossed_bootstrap(epm["chg_" + k].unstack().to_numpy(), NB),
                  "interference": crossed_bootstrap(epm["int_" + k].unstack().to_numpy(), NB),
                  "under_grip": crossed_bootstrap(epm["under_" + k].unstack().to_numpy(), NB)}
              for k in ("refresh", "null", "shuffle", "oracle") if "chg_" + k in epm}
        M3 = bool(m3["refresh"]["interference"]["ci"][0] > 0 and m3["refresh"]["change"]["est"] >= 0.01)
        p1 = contrast(d, "B_refresh", "A_cached")
        M4 = bool(p1["est"] >= DELTA and p1["ci"][0] > 0 and (L != L1 or primary["P1"]["p_holm"] < 0.05))
        # OOD
        ood = {}
        allp = pr[pr.latency == L]
        for k in ("cached", "refresh", "null", "shuffle", "oracle"):
            col = "ood_" + k
            if col in allp:
                for scope, q in (("nominal_probes", allp.kind == "none"), ("disturbed_probes", allp.kind != "none")):
                    x = allp[q]
                    ood[f"{k}_{scope}_frac_above_nominal_q99"] = float((x[col] > x.ood_thr).mean())
        mech[f"L{L}"] = {
            "M1": {"established": M1, "fc_err_A_minus_B": fmt(m1), "A_disturbed": fmt(a_dist), "A_nominal_window": fmt(a_nom)},
            "M2": {"level": M2, "nominal_action_sensitivity_null": fmt(sens),
                   "nominal_action_sensitivity_refresh_mean": float(ep.d_refresh.mean()),
                   "C_minus_A_nominal_success": fmt(c_nom), "C_minus_A_recovery": fmt(c_rec)},
            "M3": {"established": M3, **{f"{k}_{m}": fmt(v[m]) for k, v in m3.items() for m in ("change", "interference", "under_grip")}},
            "M4": {"established": M4, "B_minus_A_recovery": fmt(p1)},
            "OOD": ood,
        }
    out["mechanisms"] = mech

    # ---------------- compute, false interventions, recovery time, squeeze
    comp = df.groupby(["latency", "cls", "condition"])[["refreshes_post", "refreshes_total", "forecaster_calls",
                                                         "recovery_time", "squeeze_ratio", "grip_err_post", "slip"]].mean()
    comp.to_csv(sdir / "compute_and_secondary_metrics.csv")
    fi = {}
    for L in lats:
        n = sub(df, L, "nominal").query("condition=='F_gate'")
        fi[f"L{L}"] = {"episodes_with_trigger": float((n.gate_triggers > 0).mean()),
                       "triggers_per_step": float(n.gate_triggers.sum() / (len(n) * 80)),
                       "disturbed_first_trigger_delay_steps_median": float(sub(df, L, "recoverable", seen).query("condition=='F_gate'").first_trigger_delay.median()),
                       "disturbed_episodes_with_trigger": float((sub(df, L, "recoverable", seen).query("condition=='F_gate'").gate_triggers > 0).mean())}
    out["false_intervention"] = fi
    bench = art / "bench.json"
    out["bench"] = json.loads(bench.read_text()) if bench.exists() else {}

    # ---------------- overall verdict (L1, per protocol §9)
    mk = mech[f"L{L1}"]
    allm = mk["M1"]["established"] and mk["M2"]["level"] == "meaningful" and mk["M3"]["established"] and mk["M4"]["established"]
    gating = primary["P3"]["verdict"].startswith("positive (>=") and primary["P5"]["non_inferior"] and primary["P4"]["verdict"].startswith("positive")
    p1ci = primary["P1"]["ci"]
    if allm and gating:
        v = "supported"
    elif allm:
        v = "mechanism only"
    elif (mk["M2"]["level"] in ("weak", "absent") and p1ci[0] > -DELTA and p1ci[1] < DELTA) or primary["P1"]["verdict"].startswith("negative (<="):
        v = "contradicted"
    else:
        v = "inconclusive"
    out["H2_verdict_L1"] = v
    out["gates_pass_L1"] = gates.get(f"L{L1}_pass")
    (sdir / "summary.json").write_text(json.dumps(out, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))

    # ---------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # fig1: recovery by condition, per population
    pops = [("seen", "recoverable", "Seen types (load, push)"), ("friction", "recoverable", "Unseen type (friction)"),
            ("none", "nominal", "Nominal")]
    fig, axes = plt.subplots(len(lats), 3, figsize=(11, 3.2 * len(lats)), squeeze=False)
    for i, L in enumerate(lats):
        for j, (kind, cls, title) in enumerate(pops):
            ax = axes[i, j]
            t = tab[(tab.latency == L) & (tab["class"] == cls) & (tab.kind == kind)].set_index("condition").reindex(ALL).dropna(subset=["success"])
            y = np.arange(len(t))
            ax.barh(y, t.success, color=[COLOR[c] for c in t.index], height=0.65)
            ax.errorbar(t.success, y, xerr=[t.success - t.ci_lo, t.ci_hi - t.success], fmt="none", ecolor="#333", lw=0.8, capsize=2)
            ax.set_yticks(y, [LABEL[c] for c in t.index])
            ax.invert_yaxis()
            ax.set_xlim(0, 1.02)
            ax.set_title(f"{title} — latency {L * 50} ms", fontsize=9)
            ax.set_xlabel("success rate" if cls == "nominal" else "recovery success")
    fig.text(0.01, 0.005, "* non-deployable diagnostic. Bars: mean over student seeds x episodes; whiskers: crossed-bootstrap 95% CI.", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(fdir / "fig1_success_by_condition.png", dpi=160)
    plt.close(fig)

    # fig2: mechanism chain at L1 (forecast error, action change, interference)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    d = sub(df, L1, "recoverable", seen)
    e = d[d.condition.isin(["A_cached", "B_refresh", "E_fixed3", "F_gate"])].groupby("condition").fc_err_post.mean()
    en = sub(df, L1, "nominal").query("condition=='A_cached'").fc_err_pseudo.mean()
    axes[0].bar(range(len(e) + 1), list(e.values) + [en], color=[COLOR[c] for c in e.index] + ["#9DB4DE"])
    axes[0].set_xticks(range(len(e) + 1), [LABEL[c] for c in e.index] + ["A, nominal\nwindow"], fontsize=7)
    axes[0].set_ylabel("forecast error, N (10 steps after onset)")
    axes[0].set_title("M1 mismatch", fontsize=9)
    pd1 = pr[(pr.latency == L1) & (pr.recoverable == True) & (pr.kind.isin(seen))]  # noqa: E712
    ks = [k for k in ("refresh", "null", "shuffle", "oracle") if "g_" + k in pd1]
    chg = [np.abs(pd1.g_cached - pd1["g_" + k]).mean() for k in ks]
    nom_sens = pr[(pr.latency == L1) & (pr.kind == "none")]
    axes[1].bar(range(len(ks)), chg, color="#4C6FB5")
    axes[1].axhline(0.01, color="#A8432E", lw=0.8, ls="--")
    axes[1].text(-0.45, 0.0102, "registered 0.01 threshold", ha="left", va="bottom", fontsize=7, color="#A8432E")
    axes[1].set_xticks(range(len(ks)), [f"cached vs\n{k}" for k in ks], fontsize=7)
    axes[1].set_ylabel("mean |Δ grip command| (post-onset probes)")
    axes[1].set_title(f"M2/M3 action change (nominal |A-null| = {np.abs(nom_sens.g_cached - nom_sens.g_null).mean():.4f})", fontsize=8)
    intf = (pd1.g_cached - pd1.g_star).abs() - (pd1.g_refresh - pd1.g_star).abs()
    axes[2].hist(intf, bins=60, color="#4C6FB5")
    axes[2].axvline(0, color="#333", lw=0.8)
    axes[2].set_xlabel("|g_cached - g*| - |g_refresh - g*|  (>0: stale worse)")
    axes[2].set_ylabel("probe steps")
    axes[2].set_title(f"M3 interference, mean {intf.mean():+.4f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig2_mechanism_chain.png", dpi=160)
    plt.close(fig)

    # fig3: recovery vs severity
    fig, axes = plt.subplots(1, len(lats), figsize=(5 * len(lats), 3.3), squeeze=False)
    for i, L in enumerate(lats):
        ax = axes[0, i]
        d = df[(df.latency == L) & (df.kind.isin(seen)) & (df.cls != "nominal")].copy()
        d["bin"] = pd.cut(d.severity, [0.55, 0.65, 0.75, 0.85, 1.05, 1.15, 1.25], include_lowest=True)
        for c in ["A_cached", "B_refresh", "C_null", "F_gate", "D_live_only", "R_reactive", "X_teacher"]:
            g = d[d.condition == c].groupby("bin", observed=True).agg(sev=("severity", "mean"), s=("success", "mean"))
            sty = dict(marker="o", ms=3, lw=1.2, color="#777" if c == "X_teacher" else COLOR[c],
                       ls="--" if c == "X_teacher" else "-")
            for k, part in enumerate((g[g.sev < 1.0], g[g.sev > 1.0])):  # no line across the unsampled gap
                ax.plot(part.sev, part.s, label=LABEL[c] if k == 0 else None, **sty)
        ax.axvline(1.0, color="#999", lw=0.8, ls=":")
        ax.set_xlabel("severity (required normal force / crush limit)")
        ax.set_ylabel("recovery success")
        ax.set_title(f"Seen disturbance types — latency {L * 50} ms", fontsize=9)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fdir / "fig3_recovery_vs_severity.png", dpi=160)
    plt.close(fig)

    # fig4: compute vs recovery (L1 primary population)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = sub(df, L1, "recoverable", seen)
    for c in ["A_cached", "B_refresh", "E_fixed3", "F_gate", "Fn_gate_null", "G_random", "C_null"]:
        x = d[d.condition == c]
        ax.scatter(x.forecaster_calls.mean(), x.success.mean(), color=COLOR[c], s=30, zorder=3)
        ax.annotate(LABEL[c], (x.forecaster_calls.mean(), x.success.mean()), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("forecaster calls per episode")
    ax.set_ylabel("recovery success")
    ax.set_title(f"Compute vs recovery (seen types, {L1 * 50} ms)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig4_compute_vs_recovery.png", dpi=160)
    plt.close(fig)

    # ---------------- markdown summary
    lines = [f"# {a.tag} summary (generated)", "", f"Gates: {json.dumps(gates)}", "",
             "## Primary family (L=1, recoverable, load+push; Holm over P1-P4)", "",
             "| | contrast | estimate [95% CI] | episodes-only CI | seeds-only CI | p (Holm) | verdict |", "|---|---|---|---|---|---|---|"]
    for k in ("P1", "P2", "P3", "P4"):
        r = primary[k]
        lines.append(f"| {k} | {r['contrast']} | {r['est']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
                     f"[{r['ci_episodes_only'][0]:+.3f}, {r['ci_episodes_only'][1]:+.3f}] | [{r['ci_seeds_only'][0]:+.3f}, {r['ci_seeds_only'][1]:+.3f}] | {r['p_holm']:.3g} | {r['verdict']} |")
    r = primary["P5"]
    lines += [f"| P5 | {r['contrast']} | {r['est']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | | | | non-inferior: {r['non_inferior']} |", "",
              "## Mechanisms", "", "```", json.dumps(mech, indent=1), "```", "", f"**H2 verdict (L=1, protocol §9): {v}**", "",
              "## Reactive / live-only", "", "```", json.dumps(reac, indent=1), "```", "",
              "## False interventions", "", "```", json.dumps(fi, indent=1), "```"]
    (sdir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
