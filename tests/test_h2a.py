"""H2-001A validity tests: disturbance physics, pairing, leakage, seeds, statistics."""
import numpy as np
import pytest
import torch

from contact_learning.h2a.env import DisturbedLiftEnv, H2AConfig
from contact_learning.h2a.evaluate import _gate_random_steps
from contact_learning.h2a.expert import disturbance_for_severity
from contact_learning.h2a.nets import MLP, NumpyMLP, train_mlp
from contact_learning.h2a.runner import (HP, TAC, Episode, EpisodeSpec, Models, RunConfig, policy_input,
                                         start_episode)
from contact_learning.h2a.stats import crossed_bootstrap
from contact_learning.h2a.train import SEED_BASE


@pytest.fixture(scope="module")
def env():
    return DisturbedLiftEnv(H2AConfig())


def _rollout(env, spec, agent="X", n=None):
    e = Episode(env, Models(), start_episode(env, spec), env.h2.latency, spec.seed)
    e.run_until(RunConfig(agent=agent), n or env.h2.T)
    return e


def test_disturbance_is_dynamic_not_sensor_edit(env):
    """Identical trajectories up to onset; afterwards the change comes from forces on the object."""
    nom = _rollout(env, EpisodeSpec(240_900_001, "none", float("nan"), 0.5))
    d = _rollout(env, EpisodeSpec(240_900_001, "load", 0.8, 0.5))
    onset = env.dist.onset
    assert np.array_equal(np.asarray(nom.tac[: onset + 1]), np.asarray(d.tac[: onset + 1]))
    assert not np.array_equal(np.asarray(nom.tac[onset + 3:]), np.asarray(d.tac[onset + 3:]))
    assert env.data.xfrc_applied[env._box_body][2] < 0  # external load acts on the body


@pytest.mark.parametrize("kind", ["load", "push", "friction"])
def test_severity_mapping(env, kind):
    spec = EpisodeSpec(240_900_002, kind, 0.8, 0.5)
    start_episode(env, spec)
    t_full = env.dist.onset + env.dist.ramp_steps + 1
    ratio = env.required_normal(t_full) / env.crush_limit
    assert abs(ratio - 0.8) < 1e-6


def test_teacher_recovers_recoverable_and_frozen_grip_fails(env):
    ok_teacher, ok_unrec = [], []
    for i in range(6):
        for kind in ("load", "push", "friction"):
            e = _rollout(env, EpisodeSpec(240_910_000 + i, kind, 0.8, 0.4))
            ok_teacher.append(e.outcome(env.dist.onset)["success"])
            e = _rollout(env, EpisodeSpec(240_910_000 + i, kind, 1.15, 0.4))
            ok_unrec.append(e.outcome(env.dist.onset)["success"])
    assert np.mean(ok_teacher) == 1.0
    assert np.mean(ok_unrec) == 0.0


def test_snapshot_branch_is_bit_exact(env):
    spec = EpisodeSpec(240_900_003, "push", 0.7, 0.3)
    e = Episode(env, Models(), start_episode(env, spec), 1, spec.seed)
    onset = env.dist.onset
    e.run_until(RunConfig(agent="X"), onset)
    snap = e.snapshot()
    e.run_until(RunConfig(agent="X"), env.h2.T)
    a = (np.asarray(e.tac), np.asarray(e.rel))
    e.restore(snap)
    e.run_until(RunConfig(agent="X"), env.h2.T)
    assert np.array_equal(a[0], np.asarray(e.tac)) and np.array_equal(a[1], np.asarray(e.rel))


def test_policy_input_respects_latency_and_isolates_forecast(env):
    spec = EpisodeSpec(240_900_004, "none")
    e = _rollout(env, spec, n=30)
    t, tau = 30, 29
    f1, f2 = np.ones((HP, TAC)), np.zeros((HP, TAC))
    x1 = policy_input(env, e.tac, e.prop, t, tau, f1, 3, True)
    x2 = policy_input(env, e.tac, e.prop, t, tau, f2, 3, True)
    xn = policy_input(env, e.tac, e.prop, t, tau, None, 0, True)
    n_base = len(policy_input(env, e.tac, e.prop, t, tau, None, 0, False))
    assert np.array_equal(x1[:n_base], x2[:n_base]) and np.array_equal(x1[:n_base], xn[:n_base])
    assert not np.array_equal(x1[n_base:], x2[n_base:]) and xn[-1] == 1.0
    # the newest tactile frame (index t) is not visible with latency 1
    tac2 = list(e.tac)
    tac2[t] = tac2[t] + 100.0
    assert np.array_equal(policy_input(env, tac2, e.prop, t, tau, f1, 3, True), x1)


def test_seed_ranges_disjoint_and_above_h1():
    bases = sorted(SEED_BASE.values())
    assert bases[0] >= 200_000_000  # all H1 ranges are below 1e8
    assert all(b - a >= 10_000_000 for a, b in zip(bases, bases[1:]))


def test_numpy_mlp_matches_torch():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(256, 7)), rng.uniform(size=(256, 1))
    st = train_mlp(X, Y, X[:32], Y[:32], seed=0, epochs=2, hidden=32, layers=2, out_act="sigmoid")
    net = MLP(7, 1, 32, 2, "sigmoid")
    lin = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    for m, (W, b) in zip(lin, st["W"]):
        m.weight.data = torch.tensor(W, dtype=torch.float32)
        m.bias.data = torch.tensor(b, dtype=torch.float32)
    xt = torch.tensor((X - st["mu"]) / st["sd"], dtype=torch.float32)
    yt = net(xt).detach().numpy()
    assert np.allclose(NumpyMLP(st)(X), yt, atol=1e-5)


def test_random_gating_matches_budget_and_avoids_chunk_steps():
    rng = np.random.default_rng(1)
    steps = _gate_random_steps(None, 30, 80, 4, rng)
    assert len(steps) == 4 and all(s >= 30 and s % 10 != 0 for s in steps)
    assert _gate_random_steps(None, 30, 80, 0, rng) == frozenset()


def test_crossed_bootstrap_basics():
    z = crossed_bootstrap(np.zeros((5, 40)), n_boot=200)
    assert z["est"] == 0 and z["p_boot"] == 1.0
    pos = crossed_bootstrap(np.full((5, 40), 0.1) + np.random.default_rng(0).normal(0, 0.01, (5, 40)), n_boot=200)
    assert pos["ci"][0] > 0 and pos["p_boot"] < 0.05
    assert pos["ci_episodes_only"][0] > 0 and pos["ci_seeds_only"][0] > 0
