"""Explicit predictor input/output definitions.

Inputs at decision time t (K = history length, H = horizon):
    tactile_hist  obs.tactile[t-K+1 : t+1]   (K, 4)  measured pad forces, N
    proprio_hist  obs.proprio[t-K+1 : t+1]   (K, 4)  finger q (m) x2, hand z (m), applied grip
    past_actions  actions[t-K : t]           (K, 2)
    future_actions candidate actions[t : t+H] (H, 2)   <- the conditioning variable
Output:
    tactile at obs[t+1 : t+H+1]              (H, 4)

Indices before the start of the episode are padded with obs[0] and with the
"rest" action [-1, 0] (gripper open, no lift), which is what the env is doing then.
Nothing at index > t of obs is ever read: see ``history_at`` and
``tests/test_leakage.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REST_ACTION = np.array([-1.0, 0.0])


@dataclass
class History:
    tactile: np.ndarray       # (K, 4)
    proprio: np.ndarray       # (K, 4)
    past_actions: np.ndarray  # (K, 2)
    t: int

    def key(self) -> bytes:
        """Byte fingerprint used to verify that paired forecasts share a history."""
        return np.concatenate([self.tactile.ravel(), self.proprio.ravel(), self.past_actions.ravel(), [self.t]]).tobytes()


def history_at(tactile: np.ndarray, proprio: np.ndarray, actions: np.ndarray, t: int, K: int) -> History:
    """Build the observation history available at time t. ``tactile``/``proprio`` may
    be longer than t+1; only rows <= t are read."""
    idx = np.clip(np.arange(t - K + 1, t + 1), 0, None)
    tac = tactile[idx].copy()
    pro = proprio[idx].copy()
    pa = np.tile(REST_ACTION, (K, 1))
    for j, s in enumerate(range(t - K, t)):
        if s >= 0:
            pa[j] = actions[s]
    return History(tac, pro, pa, t)


def flatten_inputs(h: History, future_actions: np.ndarray | None) -> np.ndarray:
    parts = [h.tactile.ravel(), h.proprio.ravel(), h.past_actions.ravel()]
    if future_actions is not None:
        parts.append(np.asarray(future_actions).ravel())
    return np.concatenate(parts)


def windows_from_trajectory(traj, K: int, H: int):
    """All (history, future_actions, target) windows of one episode. Returns arrays
    plus the decision times. Windows from one episode are correlated and must stay in
    the same split."""
    X_hist, A_fut, Y, ts = [], [], [], []
    for t in range(0, traj.T - H + 1):
        h = history_at(traj.tactile, traj.proprio, traj.actions, t, K)
        X_hist.append(flatten_inputs(h, None))
        A_fut.append(traj.actions[t:t + H].ravel())
        Y.append(traj.tactile[t + 1:t + H + 1])
        ts.append(t)
    return np.array(X_hist), np.array(A_fut), np.array(Y), np.array(ts)
