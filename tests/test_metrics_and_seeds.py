"""Checks 6 (seed ranges), 10 (metric computation), and correction bounds."""
import numpy as np

from contact_learning.controllers.scripted import Correction, apply_correction, effective_magnitude
from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW
from contact_learning.evaluation import metrics as M
from contact_learning.evaluation.stats import bootstrap_ci, holm, mcnemar_exact_p, wilson_ci
from contact_learning.utils import seeding


def test_rmse_known_value():
    y = np.zeros((2, 10, 4)); yh = np.ones((2, 10, 4)) * 2.0
    assert np.allclose(M.forecast_rmse(yh, y), 2.0)
    yh[1, :, :2] = 0.0
    assert np.allclose(M.normal_rmse(yh, y), [2.0, 0.0])


def test_contact_state_error():
    y = np.zeros((1, 4, 4)); y[0, :2, :2] = 1.0
    yh = np.zeros((1, 4, 4)); yh[0, :, :2] = 1.0
    assert np.isclose(M.contact_state_error(yh, y, 0.1)[0], 0.5)


def test_paired_difference_sign():
    assert M.paired_difference(np.array([3.0]), np.array([1.0]))[0] == 2.0


def test_seed_ranges_disjoint():
    s = set()
    for p in range(5):
        tr, va = set(seeding.train_seeds(p, 5000)), set(seeding.val_seeds(p, 1000))
        assert not (tr & va) and not (s & tr) and not (s & va)
        s |= tr | va
    assert not (s & set(seeding.test_seeds(10000)))
    assert not (s & set(seeding.test_seeds(1000, seeding.TEST_BASE + 5_000_000)))


def test_correction_respects_bounds_and_magnitude():
    chunk = np.tile([0.5, 0.4], (10, 1))
    c = apply_correction(chunk, Correction("grip_decrease", 0.1))
    assert np.allclose(c[:, 0], 0.3) and np.allclose(c[:, 1], 0.4)
    assert np.isclose(effective_magnitude(chunk, c), 0.1)
    big = apply_correction(chunk, Correction("lift_increase", 0.9))
    assert np.all(big <= ACTION_HIGH) and np.all(big >= ACTION_LOW)
    assert effective_magnitude(chunk, big) < 0.9  # clipping reduces the effective magnitude


def test_stats_helpers():
    m, lo, hi = bootstrap_ci(np.ones(50))
    assert m == lo == hi == 1.0
    p, lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi
    assert mcnemar_exact_p(0, 10) < 0.01
    assert holm([0.01, 0.04, np.nan])[:2] == [0.02, 0.04]
