"""H1-002 sanity checks (protocol §11 and the H1-002 request §9)."""
import numpy as np
import pytest

from contact_learning.controllers.scripted import nominal_plan, sample_plan_params, slow_gripper_lift_delay
from contact_learning.environments.gripper_lift import EnvConfig, GripperLiftEnv
from contact_learning.evaluation import h1b
from contact_learning.evaluation.h1_002 import Cell, hier_bootstrap_diff, run_condition, trial_setup
from contact_learning.evaluation.rollout import rollout
from contact_learning.predictors.analytic import CALIBRATION_SEED_BASE, AnalyticGripForecaster
from contact_learning.predictors.data import generate_windows, training_actions
from contact_learning.predictors.features import flatten_inputs, history_at
from contact_learning.predictors.models import MLPPredictor, PersistencePredictor
from contact_learning.utils import seeding

CFG = {"T": 60, "K": 4, "H": 10, "N_min": 1.7, "kappa": 7.5, "min_lift": 0.05, "correction_times": [10, 20, 30],
       "h1b": {"correction_times": [10, 20, 30], "N_min": 1.7, "kappa": 7.5, "fixed_rate_period": 5, "min_lift": 0.05}}


@pytest.fixture(scope="module")
def tiny_mlp():
    env = GripperLiftEnv()
    d = generate_windows(env, list(range(30)), 60, 4, 10)
    return MLPPredictor(4, 10, epochs=2).fit(d["X_hist"], d["A_fut"], d["Y"])


def _setup(env, seed, cell):
    s, i = None, 0
    while s is None:
        s = trial_setup(env, seed + i, cell, CFG); i += 1
    return s


# 1. H1-001 behaviour reproducible through the new harness (L = 0)
@pytest.mark.parametrize("mag", [0.0, 0.3])
def test_new_harness_reproduces_h1b_at_zero_latency(env, tiny_mlp, mag):
    cell = Cell(0.06, 0, "grip_decrease", mag)
    st = _setup(env, 60_000_000, cell)
    st_old = h1b.trial_setup(env, st["seed"], mag, CFG)
    assert np.array_equal(st["plan_c"], st_old["plan_c"]) and st["t_c"] == st_old["t_c"]
    for new, old in (("A_cached_learned", "cached"), ("B_refreshed_learned", "refreshed"),
                     ("C_live_only", "live_only"), ("O_oracle_sim", "oracle")):
        a = run_condition(env, st, new, tiny_mlp, CFG, latency=0)
        b = h1b.run_condition(env, st_old, old, tiny_mlp, CFG)
        assert np.array_equal(a["_g"], b["_g"]), new
        assert a["success"] == b["success"] and np.isclose(a["u_f"], b["u_f_final"]) and np.isclose(a["u_l"], b["u_l_final"])


# 4. analytic baseline has no privileged access
def test_analytic_ignores_true_object_parameters(env):
    f = AnalyticGripForecaster(4, 10, 0.06, 7.5, 0.6)
    env.reset(5)
    for _ in range(25):
        o = env.step(np.array([0.8, 0.0]))
    tr_tac = np.array([o["tactile"]] * 5); tr_pro = np.array([o["proprio"]] * 5)
    h = history_at(tr_tac, tr_pro, np.zeros((4, 2)), 4, 4)
    A = np.tile([0.4, 0.0], (10, 1))
    y1 = f.predict(flatten_inputs(h, None)[None], A.reshape(1, -1))
    env.model.body_mass[:] *= 3.0  # changing the simulator cannot change a forecast from identical inputs
    y2 = f.predict(flatten_inputs(h, None)[None], A.reshape(1, -1))
    assert np.array_equal(y1, y2)
    import inspect
    from contact_learning.predictors import analytic
    src = inspect.getsource(analytic.AnalyticGripForecaster)
    assert "domain" not in src and "body_mass" not in src and "friction" not in src


def test_analytic_forecast_tracks_slew_and_contact(env):
    f = AnalyticGripForecaster(4, 10, 0.06, 7.5, 0.6)
    X = np.zeros((1, 40)); X[0, 12:14] = 7.5 * 0.3  # measured 2.25 N per pad
    X[0, 16 + 12 + 3] = 0.9                         # applied grip 0.9
    A = np.tile([2 * 0.5 - 1, 0.0], (10, 1))        # command drops to 0.5
    y = f.predict(X, A.reshape(1, -1))[0, :, 0]
    g = 0.9 - 0.06 * np.arange(1, 11)
    assert np.allclose(y, 7.5 * np.maximum(0, g - 0.6))


def test_calibration_seeds_disjoint():
    used = set(seeding.test_seeds(10_000)) | set(seeding.test_seeds(10_000, 80_000_000)) | set(seeding.test_seeds(10_000, 85_000_000))
    for p in range(5):
        used |= set(seeding.train_seeds(p, 3000)) | set(seeding.val_seeds(p, 1000))
    assert not (used & set(range(CALIBRATION_SEED_BASE, CALIBRATION_SEED_BASE + 100)))


# 5./9. latency is causal
def test_latency_controller_never_sees_newer_measurements(env, monkeypatch):
    """The forecaster's input at t_c must be exactly the history at t_c - L (built
    from an independent rollout of the nominal prefix), i.e. nothing newer."""
    cell = Cell(0.06, 2, "grip_decrease", 0.3)
    st = _setup(env, 60_000_100, cell)
    seen = {}

    class Spy(PersistencePredictor):
        def predict(self, X, A):
            seen.setdefault("calls", []).append(X.copy())
            return super().predict(X, A)

    r = run_condition(env, st, "B_refreshed_learned", Spy(4, 10), CFG, latency=2)
    assert np.isfinite(r["_g"]).all()
    # The snapshot used at t_c must equal history built at t_c - 2 from a plain rollout.
    ref = rollout(env, st["seed"], st["plan"][:st["t_c"]])
    h = history_at(ref.tactile, ref.proprio, ref.actions, st["t_c"] - 2, 4)
    assert any(np.array_equal(flatten_inputs(h, None), x[0]) for x in seen["calls"])


def test_latency_changes_live_feedback_timing(env):
    cell = Cell(0.06, 0, "grip_decrease", 0.3)
    st = _setup(env, 60_000_200, cell)
    g0 = run_condition(env, st, "C_live_only", None, CFG, latency=0)["_g"]
    g4 = run_condition(env, st, "C_live_only", None, CFG, latency=4)["_g"]
    assert not np.array_equal(g0, g4)


# 6. speed changes actuator dynamics; planner unchanged at >= reference speed
def test_slew_changes_actuator_dynamics():
    out = {}
    for slew in (0.03, 0.24):
        env = GripperLiftEnv(EnvConfig(grip_slew_per_step=slew))
        env.reset(1)
        gs = [env.step(np.array([1.0, 0.0]))["proprio"][3] for _ in range(5)]
        out[slew] = gs
    assert np.isclose(out[0.03][0], 0.03) and np.isclose(out[0.24][0], 0.24)


def test_speed_aware_plan_identical_at_reference_and_faster(env):
    rng = np.random.default_rng([3, 3]); p = sample_plan_params(rng)
    base = nominal_plan(p, 60, 0.25, 0.05)
    for slew in (0.06, 0.12, 0.24):
        assert np.array_equal(base, nominal_plan(p, 60, 0.25, 0.05, slew=slew))
    assert slow_gripper_lift_delay(p.g_hold, 0.03) > 0


# 7./8. nominal predictor trained on intended distribution; splits separate
def test_nominal_training_episodes_have_no_corrections(env):
    for s in range(20):
        a1, m1 = training_actions(env, s, 60, True)
        a0, m0 = training_actions(env, s, 60, False)
        assert m0["corrections"] == [] and m0["plan"] == m1["plan"]
        if not m1["corrections"]:
            assert np.array_equal(a0, a1)  # identical when no correction was drawn


# 10. paired bootstrap arithmetic
def test_hier_bootstrap_diff_known_values():
    import pandas as pd
    rows = []
    for seed in range(40):
        for ps in range(5):
            rows.append({"slew": .06, "latency": 0, "direction": "g", "magnitude": .3, "seed": seed,
                         "condition": "B_refreshed_learned", "pred_seed": ps, "success": True})
        rows.append({"slew": .06, "latency": 0, "direction": "g", "magnitude": .3, "seed": seed,
                     "condition": "C_live_only", "pred_seed": -1, "success": seed % 4 != 0})
    r = hier_bootstrap_diff(pd.DataFrame(rows), "B_refreshed_learned", "C_live_only", n_boot=300)
    assert np.isclose(r["diff"], 0.25) and r["ci95"][0] < 0.25 < r["ci95"][1] and r["n_trial_clusters"] == 40
