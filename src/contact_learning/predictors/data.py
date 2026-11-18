"""Training-data generation for the action-conditioned contact predictor.

Each training episode is a nominal grasp-and-lift plan, optionally perturbed by
random action corrections at random times, plus small action noise. Including
corrected actions in training is a deliberate design choice: it means the refreshed
forecast in H1 is not purely an out-of-distribution extrapolation. (Training only
on nominal plans is a separate condition, see ``corrections_in_training``.)
"""
from __future__ import annotations

import numpy as np

from contact_learning.controllers.scripted import CORRECTION_TYPES, Correction, nominal_plan, sample_plan_params
from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW, GripperLiftEnv
from contact_learning.evaluation.rollout import rollout
from contact_learning.predictors.features import windows_from_trajectory


def training_actions(env: GripperLiftEnv, seed: int, T: int, corrections_in_training: bool,
                     corr_rate: float = 1.5, max_mag: float = 0.5, action_noise: float = 0.02) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng([seed, 1])
    p = sample_plan_params(rng)
    a = nominal_plan(p, T, env.cfg.max_lift_speed, env.cfg.control_dt)
    corrs = []
    if corrections_in_training:
        for _ in range(rng.poisson(corr_rate)):
            start = int(rng.integers(0, T))
            length = int(rng.integers(3, 16))
            c = Correction(str(rng.choice(CORRECTION_TYPES)), float(rng.uniform(0, max_mag)))
            seg = slice(start, min(T, start + length))
            a[seg] = a[seg] + c.delta(seg.stop - seg.start)
            corrs.append({"start": start, "length": length, "kind": c.kind, "magnitude": c.magnitude})
    a = a + rng.normal(0, action_noise, size=a.shape)
    a = np.clip(a, ACTION_LOW, ACTION_HIGH)
    return a, {"plan": p.to_dict(), "corrections": corrs}


def generate_windows(env: GripperLiftEnv, seeds: list[int], T: int, K: int, H: int, corrections_in_training: bool = True):
    Xh, Af, Y, ep = [], [], [], []
    for s in seeds:
        a, meta = training_actions(env, s, T, corrections_in_training)
        traj = rollout(env, s, a, meta)
        xh, af, y, _ = windows_from_trajectory(traj, K, H)
        Xh.append(xh); Af.append(af); Y.append(y); ep.append(np.full(len(y), s))
    return {"X_hist": np.concatenate(Xh), "A_fut": np.concatenate(Af), "Y": np.concatenate(Y), "episode": np.concatenate(ep)}
