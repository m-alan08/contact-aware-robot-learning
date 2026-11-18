"""Episode rollout and logging of actions, observations, and contact signals.

Index convention (used everywhere):
    obs[0]      observation after reset
    actions[t]  action applied at control step t
    obs[t + 1]  observation after applying actions[t]
A forecast made "at time t" may use obs[:t+1] and actions[:t] only, and predicts
tactile obs[t+1 : t+H+1] given candidate actions[t : t+H].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contact_learning.environments.gripper_lift import GripperLiftEnv


@dataclass
class Trajectory:
    tactile: np.ndarray        # (T+1, 4) noisy measured
    tactile_clean: np.ndarray  # (T+1, 4)
    proprio: np.ndarray        # (T+1, 4)
    box_z: np.ndarray          # (T+1,)
    hand_z: np.ndarray         # (T+1,)
    actions: np.ndarray        # (T, 2)
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.actions)


class Recorder:
    def __init__(self, obs0: dict):
        self.buf = {k: [obs0[k]] for k in ("tactile", "tactile_clean", "proprio", "box_z", "hand_z")}
        self.actions: list[np.ndarray] = []

    def add(self, action: np.ndarray, obs: dict) -> None:
        self.actions.append(np.asarray(action, dtype=np.float64))
        for k in self.buf:
            self.buf[k].append(obs[k])

    def trajectory(self, meta: dict | None = None) -> Trajectory:
        return Trajectory(
            tactile=np.array(self.buf["tactile"]), tactile_clean=np.array(self.buf["tactile_clean"]),
            proprio=np.array(self.buf["proprio"]), box_z=np.array(self.buf["box_z"])[:, 0],
            hand_z=np.array(self.buf["hand_z"])[:, 0],
            actions=np.array(self.actions).reshape(-1, 2), meta=meta or {},
        )


def rollout(env: GripperLiftEnv, seed: int, actions: np.ndarray, meta: dict | None = None) -> Trajectory:
    obs = env.reset(seed)
    rec = Recorder(obs)
    for a in actions:
        a = env.clip_action(a)
        rec.add(a, env.step(a))
    m = {"seed": seed, **env.domain, **(meta or {})}
    return rec.trajectory(m)


def continue_rollout(env: GripperLiftEnv, actions: np.ndarray) -> dict[str, np.ndarray]:
    """Execute ``actions`` from the env's current state; return stacked obs arrays
    of the H resulting observations (obs[t+1..t+H])."""
    outs = {k: [] for k in ("tactile", "tactile_clean", "proprio", "box_z", "hand_z")}
    for a in actions:
        o = env.step(env.clip_action(a))
        for k in outs:
            outs[k].append(o[k])
    return {k: np.array(v) for k, v in outs.items()}
