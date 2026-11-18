"""Non-learned, action-aware contact anticipation (H1-002 condition D).

Model (pad normal force only):
    applied grip   g_{k+1} = g_k + clip(cmd_k - g_k, -slew, +slew)   (known actuator limit)
    normal force   N_k     = kappa_hat * max(0, g_k - g0_hat)

* ``kappa_hat`` and ``g0_prior`` come from ``calibrate`` (steady-state grip sweeps on
  calibration objects whose seeds are disjoint from all training/evaluation seeds).
* ``g0_hat`` is re-estimated online from the latest *available* measurement:
  ``g0_hat = g_applied - N_meas / kappa_hat`` when contact is measured, else ``g0_prior``.
* The model never reads object mass, friction, or simulator contact parameters.

Interface matches the learned predictors: ``predict(X_hist, A_fut) -> (N, H, 4)``,
where X_hist is the flattened history (features.flatten_inputs layout). Tangential
channels are returned as persistence (they are not used by the controller).
"""
from __future__ import annotations

import numpy as np

from contact_learning.environments.gripper_lift import GripperLiftEnv

TAC = 4
PRO = 4
CALIBRATION_SEED_BASE = 90_000_000


def calibrate(env: GripperLiftEnv, n_objects: int = 20, grips=np.linspace(0.62, 1.0, 12), steps: int = 30,
              seed_base: int = CALIBRATION_SEED_BASE) -> dict:
    """Fit N_mean = kappa * (g - g0) by least squares on steady-state grip sweeps."""
    G, N = [], []
    for i in range(n_objects):
        for g in grips:
            env.reset(seed_base + i)
            for _ in range(steps):
                o = env.step(np.array([2 * g - 1, 0.0]))
            n = float(o["tactile"][:2].mean())  # measured (noisy) signal, as on a real robot
            if n > 0.3:
                G.append(g); N.append(n)
    A = np.vstack([np.array(G), np.ones(len(G))]).T
    (k, b), *_ = np.linalg.lstsq(A, np.array(N), rcond=None)
    resid = np.array(N) - A @ np.array([k, b])
    return {"kappa_hat": float(k), "g0_prior": float(-b / k), "n_points": len(G),
            "resid_rmse_N": float(np.sqrt((resid ** 2).mean())), "seeds": [seed_base, seed_base + n_objects - 1]}


class AnalyticGripForecaster:
    name = "analytic"
    action_conditioned = True

    def __init__(self, K: int, H: int, slew: float, kappa_hat: float, g0_prior: float, contact_threshold: float = 0.1):
        self.K, self.H, self.slew = K, H, slew
        self.kappa_hat, self.g0_prior, self.thr = kappa_hat, g0_prior, contact_threshold

    def predict(self, X_hist: np.ndarray, A_fut: np.ndarray) -> np.ndarray:
        K, H = self.K, self.H
        X_hist = np.atleast_2d(X_hist)
        A = np.asarray(A_fut).reshape(len(X_hist), H, 2)
        last_tac = X_hist[:, (K - 1) * TAC:K * TAC]
        last_pro = X_hist[:, K * TAC + (K - 1) * PRO:K * TAC + K * PRO]
        g = last_pro[:, 3].copy()                      # applied (slewed) grip at the snapshot time
        n_meas = last_tac[:, :2].mean(1)
        g0 = np.where(n_meas > self.thr, g - n_meas / self.kappa_hat, self.g0_prior)
        out = np.repeat(last_tac[:, None, :], H, axis=1).copy()
        for k in range(H):
            cmd = (A[:, k, 0] + 1.0) / 2.0
            g = np.clip(g + np.clip(cmd - g, -self.slew, self.slew), 0.0, 1.0)
            n_hat = self.kappa_hat * np.maximum(0.0, g - g0)
            out[:, k, 0] = n_hat
            out[:, k, 1] = n_hat
        return out
