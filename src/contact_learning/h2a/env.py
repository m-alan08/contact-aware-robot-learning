"""H2-001A disturbed grasp-lift-hold task (MuJoCo), built on the H1 gripper scene.

The H1 environment (``environments/gripper_lift.py``) is reused unchanged; this
module subclasses it to add, all through simulator dynamics:

* a fragile object: a per-episode crush limit on the pad normal force,
* an episode plan: when to close and a two-segment lift-speed profile,
* recoverable / unrecoverable contact disturbances:
    - ``load``     persistent downward force on the object (e.g. contents added),
    - ``push``     transient lateral force along the pad surface (y axis),
    - ``friction`` the object's friction coefficient drops (e.g. a lubricated surface).
  Forces are applied with ``xfrc_applied`` on the object body every substep and
  friction is changed in the model; tactile readings are never edited.

The "tactile" signal is still the idealised MuJoCo pad-force proxy with additive
noise described in ``sensors/contact.py``: a simulated proxy, not a sensor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import mujoco
import numpy as np

from contact_learning.environments.gripper_lift import EnvConfig, EnvState, GripperLiftEnv

G = 9.81
KAPPA = 7.5   # N per unit grip command beyond contact (finger kp 150 N/m x 0.05 m travel)
G0 = 0.6      # grip command at which the pads touch a 4 cm box (0.03 m / 0.05 m)
DISTURBANCE_TYPES = ("load", "push", "friction")


def grip_for_normal(n: float) -> float:
    """Grip command producing pad normal force ``n`` at steady state (quasi-static model)."""
    return float(np.clip(G0 + n / KAPPA, 0.0, 1.0))


@dataclass
class Plan:
    close_start: int
    lift_start: int
    v1: float            # lift command (fraction of max speed), first segment
    n1: int              # steps in the first segment
    v2: float            # second segment, until the lift height is reached
    lift_height: float

    def lift_commands(self, T: int, max_lift_speed: float, control_dt: float) -> np.ndarray:
        cmd = np.zeros(T)
        z = 0.0
        for t in range(self.lift_start, T):
            v = self.v1 if t < self.lift_start + self.n1 else self.v2
            dz = v * max_lift_speed * control_dt
            if z + dz >= self.lift_height:
                cmd[t] = (self.lift_height - z) / (max_lift_speed * control_dt)
                break
            cmd[t] = v
            z += dz
        return cmd

    def to_dict(self):
        return asdict(self)


@dataclass
class Disturbance:
    kind: str = "none"        # none | load | push | friction
    onset: int = -1           # control step at which it starts
    magnitude: float = 0.0    # N for load/push; friction multiplier (0..1) for friction
    ramp_steps: int = 8       # ramp-in duration, control steps (0.4 s; calibrated on dev seeds)
    duration: int = 10        # push only: active steps before ramping out
    severity: float = float("nan")  # required normal force after onset / crush limit
    recoverable: Optional[bool] = None

    def force(self, step: int, frac: float) -> np.ndarray:
        """World-frame force on the object at control step ``step`` + substep fraction."""
        f = np.zeros(6)
        if self.kind not in ("load", "push") or self.onset < 0:
            return f
        x = step - self.onset + frac
        if x <= 0:
            return f
        a = min(1.0, x / self.ramp_steps)
        if self.kind == "push":
            off = x - self.duration
            if off > 0:
                a *= max(0.0, 1.0 - off / self.ramp_steps)
            f[1] = a * self.magnitude
        else:
            f[2] = -a * self.magnitude
        return f

    def friction_factor(self, step: int) -> float:
        """Friction multiplier during control step ``step`` (gradual change over ramp_steps)."""
        if self.kind == "friction" and 0 <= self.onset <= step:
            a = min(1.0, (step - self.onset + 1) / self.ramp_steps)
            return 1.0 - a * (1.0 - self.magnitude)
        return 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class H2AConfig:
    T: int = 80
    min_lift_frac: float = 0.5          # success: object raised >= this fraction of the planned height
    slip_tol: float = 0.012             # m: recovery fails if the object slides this far in the grasp
    crush_ratio_range: tuple = (1.8, 3.0)  # crush limit / static required normal force
    latency: int = 1                    # control steps of tactile sensing delay
    env: dict = field(default_factory=dict)


class DisturbedLiftEnv(GripperLiftEnv):
    def __init__(self, cfg: H2AConfig):
        env_cfg = EnvConfig.from_dict({**{"mass_range": (0.04, 0.12)}, **cfg.env})
        super().__init__(env_cfg)
        self.h2 = cfg
        self.dist = Disturbance()
        self.plan: Optional[Plan] = None
        self.lift_cmd = np.zeros(cfg.T)
        self.crush_limit = np.inf
        self.mu0 = 1.0
        self._hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # -------------------------------------------------------------- episode setup
    def reset_episode(self, seed: int, dist: Optional[Disturbance] = None) -> dict:
        obs = self.reset(seed)
        rng = np.random.default_rng([seed, 31337])
        m, mu = self.domain["mass"], self.domain["friction"]
        self.mu0 = mu
        n_static = m * G / (2 * mu)
        r = float(rng.uniform(*self.h2.crush_ratio_range))
        self.crush_limit = r * n_static
        # Deployable prior on the object's weight (e.g. from its known class): the static
        # per-pad normal-force requirement with ~8% multiplicative error.
        self.n_static_est = float(n_static * np.exp(rng.normal(0.0, 0.08)))
        # Lift accelerations are limited so that the anticipated peak load, with the
        # teacher's margin, stays below the crush limit (the planner knows the object is fragile).
        a_max = G * (r / 1.54 - 1.0)
        dv_max = a_max * self.cfg.control_dt / self.cfg.max_lift_speed  # in command units
        close_start = int(rng.integers(2, 5))
        lift_start = close_start + int(rng.integers(12, 16))
        v1 = float(min(rng.uniform(0.3, 0.9), dv_max))
        v2 = float(np.clip(v1 + rng.uniform(-0.5, 0.5), 0.15, min(1.0, v1 + dv_max)))
        self.plan = Plan(close_start, lift_start, v1, int(rng.integers(3, 8)), v2,
                         float(rng.uniform(0.07, 0.11)))
        self.lift_cmd = self.plan.lift_commands(self.h2.T, self.cfg.max_lift_speed, self.cfg.control_dt)
        self.domain.update({"crush_limit": self.crush_limit, "crush_ratio": r, "n_static": n_static,
                            "n_static_est": self.n_static_est})
        self.dist = dist or Disturbance()
        self._apply_disturbance_params()
        self.box_rel0 = None
        return obs

    def _apply_disturbance_params(self) -> None:
        self.model.geom_friction[self._gid["box"], 0] = self.mu0 * self.dist.friction_factor(self.step_count)

    def set_state(self, s: EnvState) -> None:
        super().set_state(s)
        self._apply_disturbance_params()
        # restore the force acting at the end of the previous control step
        self.data.xfrc_applied[self._box_body] = self.dist.force(self.step_count - 1, 1.0)

    # -------------------------------------------------------------- dynamics
    def step(self, action: np.ndarray) -> dict:
        self._apply_disturbance_params()
        a = self.clip_action(action)
        g_cmd = 0.5 * (a[0] + 1.0)
        d = np.clip(g_cmd - self.grip_applied, -self.cfg.grip_slew_per_step, self.cfg.grip_slew_per_step)
        self.grip_applied = float(np.clip(self.grip_applied + d, 0.0, 1.0))
        z_prev = self.z_target
        self.z_target = float(np.clip(self.z_target + a[1] * self.cfg.max_lift_speed * self.cfg.control_dt,
                                      self.cfg.z_min, self.cfg.z_max))
        self.data.ctrl[1] = self.grip_applied * 0.05
        self.data.ctrl[2] = self.grip_applied * 0.05
        for i in range(self.n_substeps):
            self.data.ctrl[0] = z_prev + (self.z_target - z_prev) * (i + 1) / self.n_substeps
            self.data.xfrc_applied[self._box_body] = self.dist.force(self.step_count, (i + 1) / self.n_substeps)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return self._observe()

    def step_grip(self, g: float) -> dict:
        """Execute grip command g in [0, 1] with the plan's lift command for this step."""
        lift = self.lift_cmd[self.step_count] if self.step_count < self.h2.T else 0.0
        return self.step(np.array([2.0 * g - 1.0, lift]))

    # -------------------------------------------------------------- privileged quantities
    def box_rel_z(self) -> float:
        """Object height relative to the hand (privileged; slip measurement only)."""
        return float(self.data.qpos[self._box_qadr + 2] - self.data.qpos[self._jadr["hand_z"]])

    def required_normal(self, t: int, hand_acc: float = 0.0, include_disturbance: bool = True) -> float:
        """Per-pad normal force needed to hold the object at step t (Coulomb, two pads)."""
        m = self.domain["mass"]
        fz, fy, k = 0.0, 0.0, 1.0
        if include_disturbance:
            f = self.dist.force(t, 1.0)
            fz, fy = -f[2], f[1]
            k = self.dist.friction_factor(t)
        tang = np.hypot(m * (G + hand_acc) + fz, fy)
        return float(tang / (2 * self.mu0 * k))

    def planned_hand_acc(self) -> np.ndarray:
        v = self.lift_cmd * self.cfg.max_lift_speed
        acc = np.diff(np.concatenate([[0.0], v, [0.0]])) / self.cfg.control_dt
        return acc[: self.h2.T + 1]

    def crushed(self, n_pair: np.ndarray) -> bool:
        return bool(min(n_pair) > self.crush_limit)
