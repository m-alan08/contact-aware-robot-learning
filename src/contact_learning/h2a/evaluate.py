"""Evaluation of H2-001A conditions on matched episodes.

Disturbed episodes: a common prefix is run with the deployed forecast-conditioned
policy (condition A, chunk-cached forecast) up to disturbance onset; every
forecast condition then branches from the identical simulator + controller
state (same sensor-noise sequence). The live-only student (D), the reactive
controller (R) and the teacher (X) run their own full episodes with the same
seed and disturbance.

Nominal episodes: every condition runs the full episode from t = 0, because
the forecast's nominal role (anticipating lift loads) happens before any
branch point.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from contact_learning.h2a.runner import CHUNK, Episode, EpisodeSpec, Models, ReactiveParams, RunConfig, start_episode
from contact_learning.h2a.train import _models
from contact_learning.h2a import train as _train

P_CONDITIONS = {
    "A_cached": "chunk", "B_refresh": "every", "C_null": "null", "E_fixed3": "fixed3",
    "F_gate": "gate", "Fn_gate_null": "gate_null", "G_random": "random", "O_oracle": "oracle",
}


class Mahalanobis:
    def __init__(self, H: np.ndarray, eps: float = 1e-3):
        self.mu = H.mean(0)
        C = np.cov(H.T) + eps * np.eye(H.shape[1])
        self.P = np.linalg.inv(C)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(h) - self.mu
        return np.sqrt(np.maximum(np.einsum("nd,de,ne->n", d, self.P, d), 0))


def _gate_random_steps(f_log_refresh: List[bool], ref: int, T: int, n: int, rng) -> frozenset:
    cand = [u for u in range(ref, T) if u % CHUNK != 0]
    if n <= 0 or not cand:
        return frozenset()
    return frozenset(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())


def eval_episode(args) -> List[Dict]:
    (spec, states, seed_s, theta, rparams, conditions, probe, shuffle_pool, record_hidden, maha) = args
    env, cfg = _train._ENV, _train._CFG
    models = _models(states)
    obs0 = start_episode(env, spec)
    disturbed = spec.kind != "none"
    ref = env.dist.onset if disturbed else env.plan.lift_start
    T = cfg.T
    common = {"seed": spec.seed, "student_seed": seed_s, "kind": spec.kind, "severity": spec.severity,
              "recoverable": (None if not disturbed else bool(spec.severity < 1.0)), "ref": ref,
              "mass": env.domain["mass"], "mu": env.domain["friction"], "crush_ratio": env.domain["crush_ratio"]}
    recs = []

    def new_ep():
        e = Episode(env, models, start_episode(env, spec), cfg.latency, spec.seed)
        e.reactive_params = rparams
        e.theta_log = theta
        e.shuffle_pool = shuffle_pool
        return e

    def finish(e, name, extra=None):
        r = dict(common, condition=name, **e.outcome(ref))
        if extra:
            r.update(extra)
        recs.append(r)
        return r

    # ---- forecast-pathway conditions (same trained policy P)
    pconds = [c for c in conditions if c in P_CONDITIONS]
    if pconds:
        if disturbed:
            e = new_ep()
            e.run_until(RunConfig(agent="P", refresh="chunk"), ref)
            snap = e.snapshot()
        f_gate_triggers = None
        order = sorted(pconds, key=lambda c: c == "G_random")  # G needs F's trigger count
        for c in order:
            if disturbed:
                e.restore(snap)
            else:
                e = new_ep()
            # probes: disturbed -> 10 steps from onset; nominal -> around lift-off, where the
            # forecast's nominal role (anticipating the lift load) is exercised
            window = (ref, ref + 10) if disturbed else (max(ref - 5, 0), ref + 5)
            rc = RunConfig(agent="P", refresh=P_CONDITIONS[c], theta=theta,
                           action_probe=(probe and c == "A_cached"),
                           probe_window=window, record_hidden=record_hidden and c == "A_cached")
            if c == "G_random":
                n = 0 if f_gate_triggers is None else f_gate_triggers
                rng = np.random.default_rng([spec.seed, seed_s, 77])
                rc.random_steps = _gate_random_steps(None, ref if disturbed else 0, T, n, rng)
            e.run_until(rc, T)
            extra = {}
            if c == "F_gate":
                refr = np.asarray(e.log["refresh"])
                start = ref if disturbed else 0
                gate_steps = [u for u in range(start, T) if refr[u] and u % CHUNK != 0]
                f_gate_triggers = len(gate_steps)
                extra["gate_triggers"] = f_gate_triggers
                extra["first_trigger_delay"] = (gate_steps[0] - ref) if (disturbed and gate_steps) else float("nan")
            if c == "Fn_gate_null":
                extra["gate_triggers"] = int(np.sum(np.asarray(e.log["disc"][(ref if disturbed else 0):]) > theta))
            if rc.action_probe:
                probes = []
                for pr in e.log["probe"]:
                    q = {k: v for k, v in pr.items() if k != "hidden"}
                    if maha is not None:
                        q.update({"ood_" + k: float(maha(h)[0]) for k, h in pr["hidden"].items()})
                    probes.append(q)
                extra["probe"] = probes
            if rc.record_hidden:
                extra["hidden"] = np.asarray(e.log["hidden"])
                extra["disc_trace"] = np.asarray(e.log["disc"])
            if not disturbed:
                # forecast error over the same window a disturbance would have started in
                extra["fc_err_pseudo"] = e.fc_err_window(spec.nominal_branch_step(env))
            finish(e, c, extra)

    # ---- separately trained / non-learned controllers (own full episodes)
    for c, agent in (("D_live_only", "D"), ("R_reactive", "R"), ("X_teacher", "X")):
        if c in conditions:
            e = new_ep()
            e.run_until(RunConfig(agent=agent), T)
            finish(e, c)
    return recs


def _slice_episode(args):
    spec, states, n_slices = args
    env, cfg = _train._ENV, _train._CFG
    e = Episode(env, _models(states), start_episode(env, spec), cfg.latency, spec.seed)
    rng = np.random.default_rng([spec.seed, 13])
    steps = set(rng.choice(np.arange(env.plan.close_start + 2, cfg.T), size=n_slices, replace=False).tolist())
    out = []
    while e.t < cfg.T:
        if e.t in steps:
            e.update_forecast(RunConfig(agent="P", refresh="chunk"))
            out.append(e.slice()[0].copy())
            e.step(RunConfig(agent="P", refresh="chunk"))
        else:
            e.step(RunConfig(agent="P", refresh="chunk"))
    return out


def shuffle_pool(p, specs, states, n_slices=4):
    """Cached forecast slices from OTHER (development) episodes: in-distribution
    forecasts with the wrong content, for the shuffle control."""
    out = p.map(_slice_episode, [(sp, states, n_slices) for sp in specs], chunksize=4)
    return np.asarray([s for o in out for s in o])


def run_set(p, specs: List[EpisodeSpec], states, seed_s, theta, rparams, conditions, probe=False,
            shuffle_pool=None, record_hidden=False, maha=None, chunksize=2):
    args = [(sp, states, seed_s, theta, rparams, tuple(conditions), probe, shuffle_pool, record_hidden, maha)
            for sp in specs]
    out = p.map(eval_episode, args, chunksize=chunksize)
    return [r for rs in out for r in rs]


def eval_specs(base: int, n_nominal: int, n_per_cell: Dict, kinds=("load", "push", "friction")) -> List[EpisodeSpec]:
    """Deterministic evaluation set: nominal episodes plus, per disturbance kind,
    recoverable (severity U[0.55, 0.85]) and unrecoverable (U[1.05, 1.25]) episodes."""
    specs, i = [], 0
    for _ in range(n_nominal):
        rng = np.random.default_rng([base + i, 4242])
        specs.append(EpisodeSpec(base + i, "none", float("nan"), float(rng.uniform())))
        i += 1
    for kind in kinds:
        for cls, (lo, hi) in (("recoverable", (0.55, 0.85)), ("unrecoverable", (1.05, 1.25))):
            for _ in range(n_per_cell[cls]):
                rng = np.random.default_rng([base + i, 4242])
                specs.append(EpisodeSpec(base + i, kind, float(rng.uniform(lo, hi)), float(rng.uniform())))
                i += 1
    return specs
