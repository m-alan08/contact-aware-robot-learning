"""H2-001A runner: train -> calibrate -> eval -> bench (CPU only).

Usage:
  python experiments/h2/h2_001a.py --config configs/experiments/h2_001a.yaml --stage all
  python experiments/h2/h2_001a.py --config configs/experiments/h2_001a_smoke.yaml --stage all --out artifacts/h2_001a_smoke

Outputs (git-ignored, archived as a release asset): <out>/L{L}/seed{s}/{models,calib,records}.pkl,
<out>/bench.json, <out>/provenance.json, <out>/log.txt.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from contact_learning.h2a import evaluate as E  # noqa: E402
from contact_learning.h2a import train as T  # noqa: E402
from contact_learning.h2a.env import H2AConfig  # noqa: E402
from contact_learning.h2a.nets import NumpyMLP  # noqa: E402
from contact_learning.h2a.runner import ReactiveParams  # noqa: E402
from contact_learning.utils.config import load_yaml  # noqa: E402
from contact_learning.utils.provenance import provenance  # noqa: E402

ALL_CONDITIONS = list(E.P_CONDITIONS) + ["D_live_only", "R_reactive", "X_teacher"]


def h2cfg(c, L):
    e = dict(c["env"])
    e["crush_ratio_range"] = tuple(e["crush_ratio_range"])
    return H2AConfig(latency=L, **e)


def states_of(m):
    return {k: m[k] for k in ("forecaster", "policy_p", "policy_d")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["train", "calibrate", "eval", "bench", "all"])
    ap.add_argument("--out", default="artifacts/h2_001a")
    ap.add_argument("--latencies", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    c = load_yaml(a.config)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    (out / "provenance.json").write_text(json.dumps({**provenance(), "config": c, "stage": a.stage}, indent=2, default=str))
    lats = a.latencies or c["latencies"]
    seeds = a.seeds or c["student_seeds"]
    rp = ReactiveParams(**c["reactive"])
    stages = ["train", "calibrate", "eval", "bench"] if a.stage == "all" else [a.stage]

    for L in lats:
        cfg = h2cfg(c, L)
        with T.pool(cfg, c["workers"]) as p:
            for s in seeds:
                d = out / f"L{L}" / f"seed{s}"
                d.mkdir(parents=True, exist_ok=True)
                if "train" in stages:
                    t0 = time.time()
                    m = T.train_seed(p, s, c["train"], log=lambda x: log(f"L{L} {x}"))
                    pickle.dump(m, open(d / "models.pkl", "wb"))
                    log(f"L{L} seed {s} trained in {time.time() - t0:.0f}s")
                if "calibrate" in stages:
                    t0 = time.time()
                    m = pickle.load(open(d / "models.pkl", "rb"))
                    st = states_of(m)
                    dev = T.SEED_BASE["dev"]
                    nom = E.eval_specs(dev, c["calibration"]["dev_nominal_eps"], {"recoverable": 0, "unrecoverable": 0})
                    r = E.run_set(p, nom, st, s, np.inf, rp, ["A_cached"], record_hidden=True)
                    disc = np.concatenate([x["disc_trace"][x["disc_trace"] > 0] for x in r])
                    H = np.concatenate([x["hidden"] for x in r])
                    theta = float(np.quantile(disc, 0.99))
                    maha = E.Mahalanobis(H)
                    ood_ref = maha(H)
                    pool_specs = E.eval_specs(dev + 100_000, c["calibration"]["dev_pool_eps"], {"recoverable": 0, "unrecoverable": 0})
                    pool = E.shuffle_pool(p, pool_specs, st)
                    calib = {"theta": theta, "maha": maha, "ood_q99_nominal": float(np.quantile(ood_ref, 0.99)),
                             "ood_median_nominal": float(np.median(ood_ref)), "pool": pool,
                             "n_disc": int(len(disc)), "disc_q50": float(np.median(disc))}
                    pickle.dump(calib, open(d / "calib.pkl", "wb"))
                    log(f"L{L} seed {s} calibrated: theta={theta:.4f} (n={len(disc)}), ood q99={calib['ood_q99_nominal']:.2f}, pool={len(pool)} ({time.time() - t0:.0f}s)")
                if "eval" in stages:
                    t0 = time.time()
                    m = pickle.load(open(d / "models.pkl", "rb"))
                    calib = pickle.load(open(d / "calib.pkl", "rb"))
                    specs = E.eval_specs(T.SEED_BASE["eval"], c["eval"]["n_nominal"], c["eval"]["n_per_cell"],
                                         tuple(c["eval"]["kinds"]))
                    recs = E.run_set(p, specs, states_of(m), s, calib["theta"], rp, ALL_CONDITIONS, probe=True,
                                     shuffle_pool=calib["pool"], maha=calib["maha"])
                    for r in recs:
                        r["latency"] = L
                    pickle.dump(recs, open(d / "records.pkl", "wb"))
                    log(f"L{L} seed {s} evaluated {len(specs)} episodes x {len(ALL_CONDITIONS)} conditions in {time.time() - t0:.0f}s")

    if "bench" in stages:
        bench = {}
        for L in lats:
            m = pickle.load(open(out / f"L{L}" / f"seed{seeds[0]}" / "models.pkl", "rb"))
            for k in ("forecaster", "policy_p", "policy_d"):
                net = NumpyMLP(m[k])
                x = np.random.default_rng(0).normal(size=(1, len(m[k]["mu"])))
                for _ in range(200):
                    net(x)
                t0 = time.perf_counter()
                n = 5000
                for _ in range(n):
                    net(x)
                bench[f"L{L}_{k}_us_per_call"] = (time.perf_counter() - t0) / n * 1e6
                bench[f"L{L}_{k}_params"] = int(sum(W.size + b.size for W, b in m[k]["W"]))
        (out / "bench.json").write_text(json.dumps(bench, indent=2))
        log(f"bench: {bench}")


if __name__ == "__main__":
    main()
