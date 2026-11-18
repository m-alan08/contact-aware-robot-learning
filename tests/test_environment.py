"""Checks 5 (reset), 6 (seeds), 7 (contact extraction), 9 (action bounds)."""
import numpy as np

from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW
from contact_learning.evaluation import env_checks


def test_reset_is_deterministic(env):
    r = env_checks.determinism_check(env)
    assert r["identical_repeat"] and r["identical_after_other_episode"]


def test_different_seeds_differ(env):
    env.reset(1); d1 = dict(env.domain)
    env.reset(2); d2 = dict(env.domain)
    assert d1["mass"] != d2["mass"]


def test_no_contact_when_open(env):
    f = env_checks.steady_grip_force(env, 3, 0.0)
    assert np.all(f == 0.0)


def test_normal_force_monotone_in_grip_and_matches_model(env):
    r = env_checks.grip_sweep(env, n=11)
    n = np.array(r["normal_l"])
    assert np.all(np.diff(n) >= -1e-3)
    assert r["max_abs_dev_from_expected_N"] < 0.05
    assert np.allclose(r["normal_l"], r["normal_r"], atol=1e-3)


def test_tangential_balances_weight_when_held(env):
    r = env_checks.lift_check(env, list(range(900, 905)))
    assert r["n_held"] == 5
    assert r["max_rel_err_tangential_vs_weight"] < 0.15


def test_action_clipping(env):
    a = env.clip_action(np.array([5.0, -3.0]))
    assert np.all(a >= ACTION_LOW) and np.all(a <= ACTION_HIGH)
    env.reset(0)
    for _ in range(5):
        env.step(np.array([10.0, 10.0]))
    assert 0.0 <= env.grip_applied <= 1.0
    assert env.z_target <= env.cfg.z_max


def test_state_restore_reproduces_future(env):
    env.reset(11)
    for _ in range(20):
        env.step(np.array([0.8, 0.3]))
    s = env.get_state()
    a = [env.step(np.array([0.9, 0.5]))["tactile"] for _ in range(10)]
    env.set_state(s)
    b = [env.step(np.array([0.9, 0.5]))["tactile"] for _ in range(10)]
    assert np.array_equal(np.array(a), np.array(b))
