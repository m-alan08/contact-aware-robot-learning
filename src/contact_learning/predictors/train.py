"""Train the predictor family for one predictor seed and evaluate it on held-out
episodes (split by episode seed; windows of one episode never cross splits)."""
from __future__ import annotations

import time

import numpy as np

from contact_learning.environments.gripper_lift import GripperLiftEnv
from contact_learning.evaluation.stats import bootstrap_ci
from contact_learning.predictors.data import generate_windows
from contact_learning.predictors.models import MLPPredictor, PersistencePredictor, RidgePredictor
from contact_learning.utils import seeding


def check_disjoint(*splits: dict) -> None:
    sets = [set(np.unique(s["episode"]).tolist()) for s in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "episode seeds leak across splits"


def train_predictors(env: GripperLiftEnv, pred_seed: int, cfg: dict) -> tuple[dict, dict]:
    T, K, H = cfg["T"], cfg["K"], cfg["H"]
    pc = cfg["predictor"]
    seeding.set_global_seed(pred_seed)
    t0 = time.time()
    cit = cfg.get("corrections_in_training", True)
    tr = generate_windows(env, seeding.train_seeds(pred_seed, cfg["n_train_episodes"]), T, K, H, cit)
    va = generate_windows(env, seeding.val_seeds(pred_seed, cfg["n_val_episodes"]), T, K, H, cit)
    # Predictor test episodes are shared across predictor seeds, disjoint from all training/val.
    te = generate_windows(env, seeding.test_seeds(cfg["n_pred_test_episodes"], seeding.TEST_BASE + 5_000_000), T, K, H, cit)
    check_disjoint(tr, va, te)
    gen_s = time.time() - t0

    models = {
        "persistence": PersistencePredictor(K, H),
        "ridge": RidgePredictor(K, H, alpha=pc.get("ridge_alpha", 1.0)),
        "mlp": MLPPredictor(K, H, True, seed=pred_seed, **pc["mlp"]),
        "mlp_history_only": MLPPredictor(K, H, False, seed=pred_seed, **pc["mlp"]),
    }
    report = {"pred_seed": pred_seed, "n_train_windows": int(len(tr["Y"])), "n_val_windows": int(len(va["Y"])),
              "n_test_windows": int(len(te["Y"])), "n_train_episodes": cfg["n_train_episodes"],
              "data_gen_seconds": gen_s, "models": {}}
    for name, m in models.items():
        t1 = time.time()
        if isinstance(m, MLPPredictor):
            m.fit(tr["X_hist"], tr["A_fut"], tr["Y"], val=va)
        else:
            m.fit(tr["X_hist"], tr["A_fut"], tr["Y"])
        fit_s = time.time() - t1
        r = {"fit_seconds": fit_s}
        for split, d in (("train", tr), ("val", va), ("test", te)):
            p = m.predict(d["X_hist"], d["A_fut"])
            se = ((p - d["Y"]) ** 2).mean(axis=(1, 2))
            # Episode-level aggregation for CI: mean squared error per episode.
            eps = np.unique(d["episode"])
            ep_mse = np.array([se[d["episode"] == e].mean() for e in eps])
            mean, lo, hi = bootstrap_ci(ep_mse, seed=pred_seed)
            r[f"{split}_rmse"] = float(np.sqrt(se.mean()))
            r[f"{split}_rmse_ci95"] = [float(np.sqrt(lo)), float(np.sqrt(hi))]
            if split == "test":
                r["test_episode_mse"] = ep_mse.tolist()
        if isinstance(m, MLPPredictor):
            r["epochs_run"] = len(m.history)
            r["history"] = m.history
        report["models"][name] = r
    # Paired episode-level comparison on the shared test episodes.
    base = np.array(report["models"]["persistence"]["test_episode_mse"])
    for name in models:
        e = np.array(report["models"][name]["test_episode_mse"])
        report["models"][name]["test_frac_episodes_better_than_persistence"] = float((e < base).mean())
    return models, report


def train_mlp_variant(env: GripperLiftEnv, pred_seed: int, cfg: dict, corrections: bool, speed_aware: bool = True):
    """H1-002: train only the action-conditioned MLP on corrected (``corrections=True``)
    or nominal-only (``False``) episodes. Episode seeds, architecture, and budget are
    identical between the two variants; evaluation uses held-out corrected AND nominal
    test episodes (seeds disjoint from training/validation)."""
    T, K, H = cfg["T"], cfg["K"], cfg["H"]
    seeding.set_global_seed(pred_seed)
    t0 = time.time()
    tr = generate_windows(env, seeding.train_seeds(pred_seed, cfg["n_train_episodes"]), T, K, H, corrections, speed_aware)
    va = generate_windows(env, seeding.val_seeds(pred_seed, cfg["n_val_episodes"]), T, K, H, corrections, speed_aware)
    test_seeds = seeding.test_seeds(cfg["n_pred_test_episodes"], seeding.TEST_BASE + 5_000_000)
    te = {"corrected": generate_windows(env, test_seeds, T, K, H, True, speed_aware),
          "nominal": generate_windows(env, test_seeds, T, K, H, False, speed_aware)}
    check_disjoint(tr, va, te["corrected"])
    m = MLPPredictor(K, H, True, seed=pred_seed, **cfg["predictor"]["mlp"]).fit(tr["X_hist"], tr["A_fut"], tr["Y"], val=va)
    pers = PersistencePredictor(K, H)
    rep = {"pred_seed": pred_seed, "corrections_in_training": corrections, "slew": env.cfg.grip_slew_per_step,
           "n_train_windows": int(len(tr["Y"])), "epochs_run": len(m.history), "seconds": time.time() - t0,
           "train_rmse": float(np.sqrt(((m.predict(tr["X_hist"][:20000], tr["A_fut"][:20000]) - tr["Y"][:20000]) ** 2).mean()))}
    pv = m.predict(va["X_hist"], va["A_fut"])
    rep["val_rmse"] = float(np.sqrt(((pv - va["Y"]) ** 2).mean()))
    rep["val_rmse_persistence"] = float(np.sqrt(((pers.predict(va["X_hist"], va["A_fut"]) - va["Y"]) ** 2).mean()))
    for k, d in te.items():
        p = m.predict(d["X_hist"], d["A_fut"])
        rep[f"test_{k}_rmse"] = float(np.sqrt(((p - d["Y"]) ** 2).mean()))
        rep[f"test_{k}_normal_rmse"] = float(np.sqrt(((p[..., :2] - d["Y"][..., :2]) ** 2).mean()))
    return m, rep, te
