"""H1-B harness checks: zero correction => all predictive conditions act identically;
oracle forecasting does not perturb the executed episode; determinism."""
import numpy as np

from contact_learning.evaluation.h1b import run_condition, trial_setup
from contact_learning.predictors.models import PersistencePredictor

CFG = {"T": 60, "K": 4, "H": 10, "h1b": {"correction_times": [10, 20, 30], "N_min": 1.7, "kappa": 7.5,
                                         "fixed_rate_period": 5, "min_lift": 0.05}}


def _setup(env, seed, mag):
    s, i = None, 0
    while s is None:
        s = trial_setup(env, seed + i, mag, CFG); i += 1
    return s


def test_zero_magnitude_conditions_identical(env):
    st = _setup(env, 70_000_000, 0.0)
    pred = PersistencePredictor(4, 10)
    g = {c: run_condition(env, st, c, pred, CFG)["_g"] for c in ("cached", "refreshed", "live_only", "fixed_rate", "oracle")}
    # With no correction and nominal force above N_min, no compensation is triggered.
    if st["g_hold"] >= 0.6 + 1.7 / 7.5 + 0.02:
        for c in g:
            assert np.array_equal(g[c], g["cached"]), c


def test_condition_runs_are_deterministic(env):
    st = _setup(env, 70_000_100, 0.2)
    pred = PersistencePredictor(4, 10)
    a = run_condition(env, st, "oracle", pred, CFG)
    b = run_condition(env, st, "oracle", pred, CFG)
    assert np.array_equal(a["_g"], b["_g"]) and a["success"] == b["success"]


def test_refreshed_equals_cached_for_action_blind_predictor(env):
    """An action-blind predictor gives the same forecast for A_orig and A_corr, so the
    refreshed condition must act exactly like the cached one (only compute differs)."""
    for seed in (70_000_200, 70_000_300, 70_000_400):
        st = _setup(env, seed, 0.3)
        pred = PersistencePredictor(4, 10)
        A = run_condition(env, st, "cached", pred, CFG)
        B = run_condition(env, st, "refreshed", pred, CFG)
        assert np.array_equal(A["_g"], B["_g"])
        assert B["n_forecasts"] == A["n_forecasts"] + 1
