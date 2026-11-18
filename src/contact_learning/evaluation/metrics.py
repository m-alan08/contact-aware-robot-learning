"""Forecast-error metrics. All operate on arrays shaped (..., H, 4) in newtons.

forecast_rmse        sqrt(mean over horizon steps and the 4 channels of (y_hat - y)^2)  [N]
normal_rmse          same, restricted to the two pad normal-force channels             [N]
per_horizon_rmse     RMSE at each horizon step k = 1..H, over channels                [N]
contact_state_error  fraction of (step, pad) entries where the predicted binary
                     contact state (normal > threshold) differs from the observed one [0..1]
paired_difference    E_cached - E_refreshed; positive => the refreshed forecast is
                     closer to the executed trajectory.
"""
from __future__ import annotations

import numpy as np


def forecast_rmse(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(((y_hat - y) ** 2).mean(axis=(-1, -2)))


def normal_rmse(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(((y_hat[..., :2] - y[..., :2]) ** 2).mean(axis=(-1, -2)))


def per_horizon_rmse(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(((y_hat - y) ** 2).mean(axis=-1))


def contact_state(y: np.ndarray, threshold: float) -> np.ndarray:
    return y[..., :2] > threshold


def contact_state_error(y_hat: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
    return (contact_state(y_hat, threshold) != contact_state(y, threshold)).mean(axis=(-1, -2))


def paired_difference(e_cached: np.ndarray, e_refreshed: np.ndarray) -> np.ndarray:
    return np.asarray(e_cached) - np.asarray(e_refreshed)
