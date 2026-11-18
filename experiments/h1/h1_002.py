"""H1-002 runner. Stages follow docs/experiments/H1-002_PROTOCOL.md §12.

    python experiments/h1/h1_002.py --stage repro       # Stage 0: H1-001 reproduction through the new harness
    python experiments/h1/h1_002.py --stage calibrate   # analytic baseline calibration (disjoint seeds)
    python experiments/h1/h1_002.py --stage train       # Stage 1: corrected (new speeds) + nominal predictors
    python experiments/h1/h1_002.py --stage pilot       # Stage 0 pilot (exploratory, n=5/cell, separate seeds)
    python experiments/h1/h1_002.py --stage forecast    # Stage 1: forecast accuracy comparison
    python experiments/h1/h1_002.py --stage grid        # Stages 2-3: confirmatory closed-loop grid
    python experiments/h1/h1_002.py --stage all

Raw outputs -> artifacts/h1/H1-002/ ; summaries -> results/h1/summaries/h1_002/.
H1-001 artifacts are only read. Nothing is overwritten without --overwrite.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from contact_learning.environments.gripper_lift import EnvConfig, GripperLiftEnv
from contact_learning.evaluation import h1b
from contact_learning.evaluation import metrics as M
from contact_learning.evaluation.h1_002 import (CONDITIONS, PILOT_SEED_BASE, TRIAL_SEED_BASE, Cell, run_cell,
                                                 run_condition, trial_setup)
from contact_learning.predictors.analytic import AnalyticGripForecaster, calibrate
from contact_learning.predictors.models import MLPPredictor
from contact_learning.predictors.train import train_mlp_variant
from contact_learning.utils.config import REPO_ROOT, load_config
from contact_learning.utils.provenance import provenance, sha256_file

CFG_PATH = REPO_ROOT / "configs" / "experiments" / "h1_002.yaml"
ART = REPO_ROOT / "artifacts" / "h1" / "H1-002"
SUMM = REPO_ROOT / "results" / "h1" / "summaries" / "h1_002"


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def guard(p: Path, overwrite: bool):
    if p.exists() and not overwrite:
        sys.exit(f"refusing to overwrite {p} (pass --overwrite)")


def env_for(cfg, slew):
    return GripperLiftEnv(EnvConfig.from_dict({**cfg["env"], "grip_slew_per_step": slew}))


def model_paths(cfg) -> dict:
    """slew -> {"corr": {seed: path}, "nom": {seed: path}}. Slew 0.06 corrected = H1-001 checkpoints."""
    out = {}
    for slew in cfg["slews"]:
        corr = {}
        for s in cfg["predictor_seeds"]:
            if np.isclose(slew, cfg["reference_slew"]):
                corr[s] = str(REPO_ROOT / cfg["h1_001_models"] / f"predictors_seed{s}.pkl")
            else:
                corr[s] = str(ART / "models" / f"corr_slew{slew}_seed{s}.pkl")
        nom = {s: str(ART / "models" / f"nom_slew{slew}_seed{s}.pkl") for s in cfg["predictor_seeds"]}
        out[slew] = {"corr": corr, "nom": nom}
    return out


# ----------------------------------------------------------------------------- Stage 0
def stage_repro(cfg, overwrite):
    """Re-run a subset of H1-001 H1-B trials through the NEW harness with the H1-001
    checkpoints and compare to the saved H1-001 episode table."""
    out = SUMM / "stage0_h1_001_reproduction.json"
    guard(out, overwrite)
    saved = pd.read_csv(REPO_ROOT / "artifacts" / "h1" / "H1-001" / "h1b_episodes.csv")
    h1cfg = {**cfg, "h1b": {"correction_times": cfg["correction_times"], "min_lift": cfg["min_lift"]}}
    env = env_for(cfg, cfg["reference_slew"])
    with open(REPO_ROOT / cfg["h1_001_models"] / "predictors_seed0.pkl", "rb") as f:
        mlp = MLPPredictor.from_state(pickle.load(f)["mlp"])
    name = {"A_cached_learned": "cached", "B_refreshed_learned": "refreshed", "C_live_only": "live_only", "O_oracle_sim": "oracle"}
    checked, mismatches = 0, []
    for mag in (0.2, 0.3):
        mi = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].index(mag)
        seeds = sorted(saved[saved.magnitude == mag].seed.unique())[:40]
        for seed in seeds:
            st = trial_setup(env, int(seed), Cell(cfg["reference_slew"], 0, "grip_decrease", mag), cfg)
            for new, old in name.items():
                r = run_condition(env, st, new, mlp, cfg, latency=0)
                ref = saved[(saved.seed == seed) & (saved.condition == old) & (saved.pred_seed.isin([0, -1]))].iloc[0]
                ok = (r["success"] == bool(ref.success) and np.isclose(r["u_f"], ref.u_f_final)
                      and np.isclose(r["u_l"], ref.u_l_final) and np.isclose(r["min_normal_after_tc"], ref.min_normal_after_tc))
                checked += 1
                if not ok:
                    mismatches.append({"seed": int(seed), "condition": new, "magnitude": mag})
        assert mi >= 0
    res = {"episodes_checked": checked, "mismatches": mismatches, "reproduced": not mismatches,
           "h1_001_table_sha256": sha256_file(REPO_ROOT / "artifacts" / "h1" / "H1-001" / "h1b_episodes.csv"),
           "provenance": provenance()}
    jdump(out, res)
    print(f"[repro] {checked} episodes, {len(mismatches)} mismatches")
    if mismatches:
        sys.exit("H1-001 reproduction FAILED: protocol §11 stop condition")


def stage_calibrate(cfg, overwrite):
    out = SUMM / "analytic_calibration.json"
    guard(out, overwrite)
    cal = calibrate(env_for(cfg, cfg["reference_slew"]))
    cal["note"] = "fitted once at slew 0.06; kappa and g0 are properties of the finger servo/object geometry, not of slew"
    jdump(out, cal)
    print(f"[calibrate] {cal}")


# ----------------------------------------------------------------------------- Stage 1 training
def _train_job(args):
    slew, seed, corrections, cfg, path = args
    import torch
    torch.set_num_threads(1)
    env = env_for(cfg, slew)
    m, rep, _ = train_mlp_variant(env, seed, cfg, corrections, speed_aware=True)
    with open(path, "wb") as f:
        pickle.dump({"mlp": m.state()}, f)
    rep["path"] = str(Path(path).relative_to(REPO_ROOT))
    rep["sha256"] = sha256_file(path)
    return rep


def stage_train(cfg, overwrite, workers):
    out = SUMM / "predictor_training.json"
    guard(out, overwrite)
    (ART / "models").mkdir(parents=True, exist_ok=True)
    jobs = []
    for slew, mp_ in model_paths(cfg).items():
        for s in cfg["predictor_seeds"]:
            if not np.isclose(slew, cfg["reference_slew"]):
                jobs.append((slew, s, True, cfg, mp_["corr"][s]))
            jobs.append((slew, s, False, cfg, mp_["nom"][s]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        reports = pool.map(_train_job, jobs)
    print(f"[train] {len(jobs)} predictors in {time.time() - t0:.0f}s")
    # Evaluate the reused H1-001 checkpoints on the same held-out windows (for a matched comparison).
    from contact_learning.predictors.data import generate_windows
    from contact_learning.utils import seeding
    env = env_for(cfg, cfg["reference_slew"])
    ts = seeding.test_seeds(cfg["n_pred_test_episodes"], seeding.TEST_BASE + 5_000_000)
    te = {"corrected": generate_windows(env, ts, cfg["T"], cfg["K"], cfg["H"], True, True),
          "nominal": generate_windows(env, ts, cfg["T"], cfg["K"], cfg["H"], False, True)}
    for s, p in model_paths(cfg)[cfg["reference_slew"]]["corr"].items():
        with open(p, "rb") as f:
            m = MLPPredictor.from_state(pickle.load(f)["mlp"])
        rep = {"pred_seed": s, "corrections_in_training": True, "slew": cfg["reference_slew"], "reused_from": "H1-001",
               "path": str(Path(p).relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
        for k, d in te.items():
            pr = m.predict(d["X_hist"], d["A_fut"])
            rep[f"test_{k}_rmse"] = float(np.sqrt(((pr - d["Y"]) ** 2).mean()))
            rep[f"test_{k}_normal_rmse"] = float(np.sqrt(((pr[..., :2] - d["Y"][..., :2]) ** 2).mean()))
        reports.append(rep)
    # Analytic + persistence on the same windows, per slew (normal channels).
    cal = json.load(open(SUMM / "analytic_calibration.json"))
    from contact_learning.predictors.models import PersistencePredictor
    base = []
    for slew in cfg["slews"]:
        env = env_for(cfg, slew)
        te = {"corrected": generate_windows(env, ts, cfg["T"], cfg["K"], cfg["H"], True, True),
              "nominal": generate_windows(env, ts, cfg["T"], cfg["K"], cfg["H"], False, True)}
        for name, f in (("analytic", AnalyticGripForecaster(cfg["K"], cfg["H"], slew, cal["kappa_hat"], cal["g0_prior"])),
                        ("persistence", PersistencePredictor(cfg["K"], cfg["H"]))):
            r = {"model": name, "slew": slew}
            for k, d in te.items():
                pr = f.predict(d["X_hist"], d["A_fut"])
                r[f"test_{k}_normal_rmse"] = float(np.sqrt(((pr[..., :2] - d["Y"][..., :2]) ** 2).mean()))
                if name == "persistence":
                    r[f"test_{k}_rmse"] = float(np.sqrt(((pr - d["Y"]) ** 2).mean()))
            base.append(r)
    jdump(out, {"predictors": reports, "baselines": base, "config": cfg, "provenance": provenance()})


# ----------------------------------------------------------------------------- Stage 1 forecast on H1-001 H1-A trials
def stage_forecast(cfg, overwrite):
    out = SUMM / "stage1_forecast_h1a_trials.csv"
    guard(out, overwrite)
    raw = np.load(REPO_ROOT / "artifacts" / "h1" / "H1-001" / "h1a_trials.npz")
    meta = pd.read_csv(REPO_ROOT / "artifacts" / "h1" / "H1-001" / "h1a_trials.csv")
    meta = meta[(meta.predictor == "mlp") & (meta.pred_seed == 0)].set_index("seed").loc[raw["seeds"]]
    X, Ac, Y = raw["X"], raw["A_corr"].reshape(len(raw["X"]), -1), raw["Y_corr"]
    Ao = raw["A_orig"].reshape(len(X), -1)
    cal = json.load(open(SUMM / "analytic_calibration.json"))
    paths = model_paths(cfg)[cfg["reference_slew"]]
    rows = []
    fc = {("analytic", -1): AnalyticGripForecaster(cfg["K"], cfg["H"], cfg["reference_slew"], cal["kappa_hat"], cal["g0_prior"])}
    for fam in ("corr", "nom"):
        for s, p in paths[fam].items():
            with open(p, "rb") as f:
                fc[(fam, s)] = MLPPredictor.from_state(pickle.load(f)["mlp"])
    for (fam, s), f in fc.items():
        fr, fcache = f.predict(X, Ac), f.predict(X, Ao)
        d = pd.DataFrame({"seed": raw["seeds"], "family": fam, "pred_seed": s,
                          "E_refreshed": M.forecast_rmse(fr, Y), "En_refreshed": M.normal_rmse(fr, Y),
                          "En_cached": M.normal_rmse(fcache, Y),
                          "kind": meta.kind.values, "magnitude_effective": meta.magnitude_effective.values,
                          "contact_changing": meta.contact_changing.values})
        rows.append(d)
    pd.concat(rows).to_csv(out, index=False)
    print(f"[forecast] wrote {out.relative_to(REPO_ROOT)}")


# ----------------------------------------------------------------------------- Stages 2-3 grid
def cells_for(cfg, which):
    if which == "pilot":
        pc = cfg["pilot"]
        return [Cell(s, l, "grip_decrease", m) for s in pc["slews"] for l in pc["latencies"] for m in pc["grip_decrease_magnitudes"]]
    cells = [Cell(s, l, "grip_decrease", m) for s in cfg["slews"] for l in cfg["latencies"] for m in cfg["grip_decrease_magnitudes"]]
    db = cfg["direction_block"]
    cells += [Cell(s, l, db["direction"], m) for s in db["slews"] for l in db["latencies"] for m in db["magnitudes"]]
    return cells


def stage_grid(cfg, overwrite, workers, which="grid"):
    out = ART / f"{which}_episodes.csv"
    guard(out, overwrite)
    ART.mkdir(parents=True, exist_ok=True)
    cal = json.load(open(SUMM / "analytic_calibration.json"))
    mpaths = model_paths(cfg)
    for slew, d in mpaths.items():
        for fam in d.values():
            for p in fam.values():
                assert Path(p).exists(), f"missing model {p} (run --stage train first)"
    n = cfg["pilot"]["n_trials_per_cell"] if which == "pilot" else cfg["n_trials_per_cell"]
    base = PILOT_SEED_BASE if which == "pilot" else TRIAL_SEED_BASE
    jobs = [(c, n, base, cfg, mpaths[c.slew], cal) for c in cells_for(cfg, which)]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_cell, jobs)):
            rows.extend(r)
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"[{which}] {k + 1}/{len(jobs)} cells ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    jdump(SUMM / f"{which}_manifest.json", {
        "episodes_csv": str(out.relative_to(REPO_ROOT)), "sha256": sha256_file(out), "bytes": out.stat().st_size,
        "n_rows": len(df), "n_cells": len(jobs), "n_trials_per_cell": n, "trial_seed_base": base,
        "seconds": time.time() - t0, "workers": workers, "config": cfg, "provenance": provenance()})
    print(f"[{which}] wrote {len(df)} episode rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["repro", "calibrate", "train", "pilot", "forecast", "grid", "all"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(CFG_PATH)
    SUMM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    stages = ["repro", "calibrate", "train", "pilot", "forecast", "grid"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "repro":
            stage_repro(cfg, a.overwrite)
        elif st == "calibrate":
            stage_calibrate(cfg, a.overwrite)
        elif st == "train":
            stage_train(cfg, a.overwrite, a.workers)
        elif st == "forecast":
            stage_forecast(cfg, a.overwrite)
        elif st in ("pilot", "grid"):
            stage_grid(cfg, a.overwrite, a.workers, st)


if __name__ == "__main__":
    main()
