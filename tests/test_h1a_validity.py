"""Checks 1 (zero-correction equivalence), 2 (pairing), 3 (alignment), 4 (leakage)."""
import numpy as np

from contact_learning.controllers.scripted import nominal_plan, sample_plan_params
from contact_learning.evaluation.h1a import score_trials, simulate_trial
from contact_learning.evaluation.rollout import rollout
from contact_learning.predictors.data import generate_windows
from contact_learning.predictors.features import history_at, windows_from_trajectory
from contact_learning.predictors.models import MLPPredictor, PersistencePredictor, RidgePredictor


def _plan(env, seed):
    rng = np.random.default_rng([seed, 2])
    return nominal_plan(sample_plan_params(rng), 60, env.cfg.max_lift_speed, env.cfg.control_dt)


def test_history_never_reads_future_rows():
    tac = np.random.rand(30, 4); pro = np.random.rand(30, 4); act = np.random.rand(29, 2)
    t = 12
    tac2, pro2, act2 = tac.copy(), pro.copy(), act.copy()
    tac2[t + 1:] = np.nan; pro2[t + 1:] = np.nan; act2[t:] = np.nan
    h1, h2 = history_at(tac, pro, act, t, 4), history_at(tac2, pro2, act2, t, 4)
    assert np.array_equal(h1.key(), h2.key())
    assert np.isfinite(h2.tactile).all() and np.isfinite(h2.past_actions).all()


def test_window_alignment(env):
    a = _plan(env, 5)
    tr = rollout(env, 5, a)
    Xh, Af, Y, ts = windows_from_trajectory(tr, 4, 10)
    for i in (0, 7, 30):
        t = ts[i]
        assert np.array_equal(Y[i], tr.tactile[t + 1:t + 11])
        assert np.array_equal(Af[i].reshape(10, 2), tr.actions[t:t + 10])
        assert np.array_equal(Xh[i][12:16], tr.tactile[t])  # last history frame is obs[t]


def test_executed_corrected_trajectory_matches_full_rollout(env, h1cfg):
    """Y_corr from snapshot/restore equals a from-scratch rollout of the spliced
    action sequence: confirms state restore, noise-stream restore, and time alignment."""
    for seed in (101, 102, 103):
        tr = simulate_trial(env, seed, h1cfg)
        t_c = tr["t_c"]
        full = np.concatenate([_plan(env, seed)[:t_c], tr["_A_corr"]])
        ref = rollout(env, seed, full)
        assert np.array_equal(ref.tactile[t_c + 1:t_c + 11], tr["_Y_corr"])
        assert tr["prefix_reproducible"]


def test_zero_correction_gives_identical_forecasts(env):
    cfg = {"T": 60, "K": 4, "H": 10, "magnitudes": [0.0], "strata": ["uniform"]}
    trials = [simulate_trial(env, s, cfg) for s in range(200, 206)]
    for t in trials:
        assert np.array_equal(t["_A_orig"], t["_A_corr"])
    d = generate_windows(env, list(range(20)), 60, 4, 10)
    for m in (RidgePredictor(4, 10).fit(d["X_hist"], d["A_fut"], d["Y"]),
              MLPPredictor(4, 10, epochs=1).fit(d["X_hist"], d["A_fut"], d["Y"])):
        sc = score_trials(trials, m, 0.1, latency_samples=1)
        assert np.all(sc["D"] == 0.0)


def test_action_blind_negative_control_has_zero_difference(env, h1cfg):
    trials = [simulate_trial(env, s, h1cfg) for s in range(300, 310)]
    d = generate_windows(env, list(range(20)), 60, 4, 10)
    for m in (PersistencePredictor(4, 10), MLPPredictor(4, 10, action_conditioned=False, epochs=1).fit(d["X_hist"], d["A_fut"], d["Y"])):
        sc = score_trials(trials, m, 0.1, latency_samples=1)
        assert np.all(sc["D"] == 0.0)


def test_pairing_uses_same_history(env, h1cfg):
    """Both forecasts are computed from the single stored history vector _X; check it
    equals a history rebuilt from an independent rollout of the nominal prefix."""
    tr = simulate_trial(env, 400, h1cfg)
    ref = rollout(env, 400, _plan(env, 400)[:tr["t_c"]])
    h = history_at(ref.tactile, ref.proprio, ref.actions, tr["t_c"], 4)
    assert h.key() == tr["_hist_key"]
