"""Training for H2-001A: forecaster (supervised) and students (DAgger).

Seed ranges (episode seeds; disjoint from every H1 range, which all lie below 1e8):
  forecaster data   200_000_000 + 1_000_000*s + i
  student P DAgger  210_000_000 + 1_000_000*s + 10_000*it + i
  student D DAgger  220_000_000 + 1_000_000*s + 10_000*it + i
  validation        230_000_000 + 1_000_000*s + i
  development set   240_000_000 + i     (calibration, thresholds, pilot)
  held-out eval     250_000_000 + i
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List, Optional, Sequence

import numpy as np

from contact_learning.h2a.env import DisturbedLiftEnv, H2AConfig
from contact_learning.h2a.nets import NumpyMLP, train_mlp
from contact_learning.h2a.runner import (HF, Episode, EpisodeSpec, Models, RunConfig, forecaster_input,
                                         sample_spec, start_episode)

SEED_BASE = {"forecaster": 200_000_000, "P": 210_000_000, "D": 220_000_000, "val": 230_000_000,
             "dev": 240_000_000, "eval": 250_000_000}

_ENV: Optional[DisturbedLiftEnv] = None
_CFG: Optional[H2AConfig] = None


def _init(cfg: H2AConfig):
    global _ENV, _CFG
    _CFG = cfg
    _ENV = DisturbedLiftEnv(cfg)


def pool(cfg: H2AConfig, workers: int):
    return mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))


def _models(states: Dict) -> Models:
    return Models(**{k: (NumpyMLP(v) if v is not None else None) for k, v in states.items()})


# ---------------------------------------------------------------- forecaster data
def _forecaster_episode(args):
    seed, = args
    env, cfg = _ENV, _CFG
    spec = EpisodeSpec(seed, "none")
    obs0 = start_episode(env, spec)
    ep = Episode(env, Models(), obs0, cfg.latency, seed)
    rc = RunConfig(agent="X", exec_noise=0.04)
    rng = np.random.default_rng([seed, 5])
    X, Y = [], []
    taus = set(range(env.plan.close_start, cfg.T - 1, 3))
    while ep.t < cfg.T:
        tau = ep.t
        if tau in taus:
            st = env.get_state()
            j = int(rng.integers(0, 3))
            cmds = list(ep.grip_cmds)
            tac = list(ep.tac)
            for _ in range(j):
                g = float(np.clip(env.grip_applied + rng.normal(0, 0.08), 0, 1))
                cmds.append(g)
                tac.append(env.step_grip(g)["tactile"].copy())
            g_hold = env.grip_applied
            while len(tac) < tau + 1 + HF:
                tac.append(env.step_grip(g_hold)["tactile"].copy())
            env.set_state(st)
            x = forecaster_input(ep.tac, ep.prop, np.asarray(cmds), env.lift_cmd, tau, tau + j, g_hold)
            y = (np.asarray(tac[tau + 1:tau + 1 + HF]) - ep.tac[tau]).ravel()
            X.append(x)
            Y.append(y)
        ep.step(rc)
    return np.asarray(X), np.asarray(Y)


def forecaster_data(p, seeds: Sequence[int]):
    out = p.map(_forecaster_episode, [(s,) for s in seeds], chunksize=8)
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


# ---------------------------------------------------------------- student data (DAgger)
def _student_episode(args):
    seed, states, agent, beta, noise, p_null, kinds, sev_range, p_nominal = args
    env, cfg = _ENV, _CFG
    spec = sample_spec(seed, kinds, sev_range, p_nominal)
    obs0 = start_episode(env, spec)
    ep = Episode(env, _models(states), obs0, cfg.latency, seed)
    rc = RunConfig(agent=agent, refresh="chunk", beta=beta, exec_noise=noise, record_features=True,
                   p_null_record=p_null)
    ep.run_until(rc, cfg.T)
    return np.asarray(ep.log["x"]), np.asarray(ep.log["y"])


def student_data(p, seeds, states, agent, beta, noise, p_null, kinds, sev_range, p_nominal):
    args = [(s, states, agent, beta, noise, p_null, tuple(kinds), tuple(sev_range), p_nominal) for s in seeds]
    out = p.map(_student_episode, args, chunksize=4)
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])[:, None]


# ---------------------------------------------------------------- full per-seed training
TRAIN_KINDS = ("load", "push")      # 'friction' is held out of training (unseen disturbance type)
TRAIN_SEVERITY = (0.5, 1.2)
TRAIN_P_NOMINAL = 0.35


def train_seed(p, s: int, size: Dict, log=print) -> Dict:
    """Train the forecaster, then students P (forecast-conditioned) and D (live-only)
    with identical DAgger schedules, episode budgets and optimiser settings."""
    fb = SEED_BASE["forecaster"] + 1_000_000 * s
    Xf, Yf = forecaster_data(p, [fb + i for i in range(size["forecaster_eps"])])
    Xfv, Yfv = forecaster_data(p, [SEED_BASE["val"] + 1_000_000 * s + 500_000 + i for i in range(size["val_eps"])])
    log(f"[seed {s}] forecaster data {Xf.shape} val {Xfv.shape}")
    fc = train_mlp(Xf, Yf, Xfv, Yfv, seed=1000 + s, epochs=size["forecaster_epochs"])
    log(f"[seed {s}] forecaster final train/val loss {fc['history'][-1][1]:.4f}/{fc['history'][-1][2]:.4f}")
    out = {"forecaster": fc, "history": {"forecaster": fc["history"]}}
    for agent in ("P", "D"):
        key = "policy_p" if agent == "P" else "policy_d"
        states = {"forecaster": fc, "policy_p": None, "policy_d": None}
        vseeds = [SEED_BASE["val"] + 1_000_000 * s + i for i in range(size["val_eps"])]
        Xv, Yv = student_data(p, vseeds, states, agent, 1.0, 0.0, 0.0, TRAIN_KINDS, TRAIN_SEVERITY, TRAIN_P_NOMINAL)
        Xs, Ys, hist = [], [], []
        for it, beta in enumerate(size["betas"]):
            seeds = [SEED_BASE[agent] + 1_000_000 * s + 10_000 * it + i for i in range(size["eps_per_iter"])]
            X, Y = student_data(p, seeds, states, agent, beta, size["exec_noise"], size["p_null"] if agent == "P" else 0.0,
                                TRAIN_KINDS, TRAIN_SEVERITY, TRAIN_P_NOMINAL)
            Xs.append(X); Ys.append(Y)
            last = it == len(size["betas"]) - 1
            st = train_mlp(np.concatenate(Xs), np.concatenate(Ys), Xv, Yv, seed=2000 + 10 * s + it + (0 if agent == "P" else 500),
                           out_act="sigmoid", epochs=size["final_epochs"] if last else size["iter_epochs"])
            states[key] = st
            hist.append({"iter": it, "beta": beta, "n": int(sum(len(x) for x in Xs)),
                         "train": st["history"][-1][1], "val": st["history"][-1][2]})
            log(f"[seed {s}] {agent} iter {it} beta {beta} n={hist[-1]['n']} train {hist[-1]['train']:.5f} val {hist[-1]['val']:.5f}")
        out[key] = states[key]
        out["history"][agent] = hist
    return out
