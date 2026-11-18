"""Run H1 experiment stages.

    python experiments/h1/run_experiment.py --config configs/experiments/h1_001.yaml --stage all

Stages
    validate-env  contact-signal validation checks           -> results/h1/summaries/<id>_env_validation.json
    train         train predictor family for every seed        -> artifacts/h1/<id>/models/, summaries/<id>_predictors.json
    h1a           paired cached-vs-refreshed forecast trials   -> artifacts/h1/<id>/h1a_trials.{npz,csv}, summaries/<id>_h1a_manifest.json
    h1b           closed-loop control relevance (see h1b.py)   -> artifacts/h1/<id>/h1b_episodes.csv, summaries/<id>_h1b_summary.json
    all           validate-env, train, h1a, h1b

Raw outputs live under artifacts/ (git-ignored); compact summaries under results/.
Nothing is overwritten unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from contact_learning.environments.gripper_lift import EnvConfig, GripperLiftEnv
from contact_learning.evaluation import env_checks
from contact_learning.evaluation.h1a import score_trials, simulate_trials
from contact_learning.predictors.models import MLPPredictor, PersistencePredictor, RidgePredictor
from contact_learning.predictors.train import train_predictors
from contact_learning.utils import seeding
from contact_learning.utils.config import REPO_ROOT, load_config
from contact_learning.utils.provenance import provenance, sha256_file


def paths(cfg):
    eid = cfg["experiment_id"]
    art = REPO_ROOT / "artifacts" / "h1" / eid
    summ = REPO_ROOT / "results" / "h1" / "summaries"
    art.mkdir(parents=True, exist_ok=True)
    (art / "models").mkdir(exist_ok=True)
    summ.mkdir(parents=True, exist_ok=True)
    return eid, art, summ


def guard(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        sys.exit(f"refusing to overwrite existing output {path} (pass --overwrite to replace)")


def write_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def stage_validate(cfg, env, overwrite):
    eid, art, summ = paths(cfg)
    out = summ / f"{eid}_env_validation.json"
    guard(out, overwrite)
    res = env_checks.run_all(env)
    res["env_config"] = env.cfg.to_dict()
    res["provenance"] = provenance()
    write_json(out, res)
    print(f"[validate-env] grip-sweep max dev {res['grip_sweep']['max_abs_dev_from_expected_N']:.4f} N; "
          f"Coulomb agreement {res['capacity']['agreement']:.3f}; determinism {res['determinism']}")


def stage_train(cfg, env, overwrite):
    eid, art, summ = paths(cfg)
    out = summ / f"{eid}_predictors.json"
    guard(out, overwrite)
    reports, ckpts = [], {}
    for s in cfg["predictor_seeds"]:
        t0 = time.time()
        models, rep = train_predictors(env, s, cfg)
        p = art / "models" / f"predictors_seed{s}.pkl"
        with open(p, "wb") as f:
            pickle.dump({"ridge": models["ridge"],
                         "mlp": models["mlp"].state(), "mlp_history_only": models["mlp_history_only"].state()}, f)
        ckpts[str(s)] = {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for r in rep["models"].values():
            r.pop("test_episode_mse", None)
        reports.append(rep)
        m = rep["models"]
        print(f"[train] seed {s}: test RMSE (N) persistence {m['persistence']['test_rmse']:.4f} "
              f"ridge {m['ridge']['test_rmse']:.4f} mlp {m['mlp']['test_rmse']:.4f} "
              f"mlp_history_only {m['mlp_history_only']['test_rmse']:.4f} ({time.time() - t0:.0f}s)")
    write_json(out, {"experiment_id": eid, "config": cfg, "reports": reports, "checkpoints": ckpts, "provenance": provenance()})


def load_models(cfg, seed):
    _, art, _ = paths(cfg)
    with open(art / "models" / f"predictors_seed{seed}.pkl", "rb") as f:
        d = pickle.load(f)
    K, H = cfg["K"], cfg["H"]
    return {"persistence": PersistencePredictor(K, H), "ridge": d["ridge"],
            "mlp": MLPPredictor.from_state(d["mlp"]), "mlp_history_only": MLPPredictor.from_state(d["mlp_history_only"])}


def stage_h1a(cfg, env, overwrite):
    eid, art, summ = paths(cfg)
    out_csv = art / "h1a_trials.csv"  # per-trial raw table (large): artifacts, checksummed in manifest
    guard(out_csv, overwrite)
    hc = {**cfg["h1a"], "T": cfg["T"], "K": cfg["K"], "H": cfg["H"]}
    seeds = seeding.test_seeds(hc["n_trials"])
    t0 = time.time()
    trials = simulate_trials(env, seeds, hc)
    print(f"[h1a] simulated {len(trials)} trials in {time.time() - t0:.0f}s; "
          f"prefix reproducible: {all(t['prefix_reproducible'] for t in trials)}")
    meta_cols = [k for k in trials[0] if not k.startswith("_")]
    base = pd.DataFrame([{k: t[k] for k in meta_cols} for t in trials])
    rows, raw = [], {}
    for s in cfg["predictor_seeds"]:
        models = load_models(cfg, s)
        for name, m in models.items():
            sc = score_trials(trials, m, env.cfg.contact_threshold)
            df = base.copy()
            df["pred_seed"], df["predictor"] = s, name
            for k, v in sc.items():
                if k.startswith("PH_") or k == "latency_s":
                    continue
                df[k] = v
            df["latency_ms_median"] = float(np.median(sc["latency_s"]) * 1e3)
            rows.append(df)
            raw[f"seed{s}_{name}_PH_cached"] = sc["PH_cached"]
            raw[f"seed{s}_{name}_PH_refreshed"] = sc["PH_refreshed"]
            raw[f"seed{s}_{name}_latency_s"] = sc["latency_s"]
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out_csv, index=False)
    npz = art / "h1a_trials.npz"
    np.savez_compressed(npz, X=np.stack([t["_X"] for t in trials]), A_orig=np.stack([t["_A_orig"] for t in trials]),
                        A_corr=np.stack([t["_A_corr"] for t in trials]), A_sham=np.stack([t["_A_sham"] for t in trials]),
                        Y_corr=np.stack([t["_Y_corr"] for t in trials]), Y_orig=np.stack([t["_Y_orig"] for t in trials]),
                        seeds=np.array(seeds), **raw)
    write_json(summ / f"{eid}_h1a_manifest.json", {
        "experiment_id": eid, "n_trials": len(trials), "trial_seeds": [seeds[0], seeds[-1]],
        "predictor_seeds": cfg["predictor_seeds"], "raw_artifact": str(npz.relative_to(REPO_ROOT)),
        "raw_sha256": sha256_file(npz), "raw_bytes": npz.stat().st_size,
        "trials_csv": str(out_csv.relative_to(REPO_ROOT)), "trials_csv_sha256": sha256_file(out_csv),
        "trials_csv_bytes": out_csv.stat().st_size, "config": cfg, "provenance": provenance()})
    print(f"[h1a] wrote {out_csv.relative_to(REPO_ROOT)} ({len(res)} rows)")


def stage_h1b(cfg, env, overwrite):
    from contact_learning.evaluation.h1b import run_h1b  # imported lazily: optional stage
    run_h1b(cfg, env, overwrite, load_models, paths, guard, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["validate-env", "train", "h1a", "h1b", "all"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeding.set_global_seed(0)
    env = GripperLiftEnv(EnvConfig.from_dict(cfg.get("env")))
    stages = ["validate-env", "train", "h1a", "h1b"] if args.stage == "all" else [args.stage]
    fn = {"validate-env": stage_validate, "train": stage_train, "h1a": stage_h1a, "h1b": stage_h1b}
    for st in stages:
        if st == "h1b" and "h1b" not in cfg:
            print("[h1b] no h1b section in config; skipping")
            continue
        fn[st](cfg, env, args.overwrite)


if __name__ == "__main__":
    main()
