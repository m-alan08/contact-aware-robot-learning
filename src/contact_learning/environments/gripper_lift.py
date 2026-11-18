"""Parallel-jaw grasp-and-lift task in MuJoCo.

A minimal contact-rich task for testing action-conditioned contact prediction.

Scene
-----
* A floating "hand" with one vertical slide joint (``hand_z``), gravity-compensated
  and driven by a stiff position servo.
* Two fingers on prismatic joints that move symmetrically toward the hand axis.
  Each finger carries a box-shaped pad geom; contact between pad and object is the
  simulated "tactile" signal (see ``contact_learning.sensors.contact``).
* A box object on a floor with randomised mass and friction.

Action (per control step) -- valid bounds ACTION_LOW = [-1, 0], ACTION_HIGH = [1, 1]
------------------------------------------------------------------------------------
``a[0]`` grip command  g in [0, 1] mapped from [-1, 1]  (0 = fully open, 1 = fully closed).
          The finger position target is ``g * FINGER_TRAVEL``. The commanded grip is
          passed through a slew-rate limiter (``grip_slew_per_step``) that models a
          slow gripper servo.
``a[1]`` lift velocity command in [0, 1], scaled by ``max_lift_speed`` (m/s). The
          hand height target integrates this velocity and is clipped to the joint range.
          Lift is one-directional by design: allowing descent lets the stiff hand servo
          press the pads onto a dropped box, producing non-physical ~70 N "contacts"
          that dominated forecast error in an early version (see docs/experiments/H1-001.md).

Everything that determines the future (MuJoCo state + limiter state + height target)
is captured by :meth:`GripperLiftEnv.get_state` / :meth:`set_state`, which is what the
paired H1 trials use to execute counterfactual action sequences from the same state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import mujoco
import numpy as np

from contact_learning.sensors.contact import ContactReading, read_pad_contacts

ACTION_LOW = np.array([-1.0, 0.0])
ACTION_HIGH = np.array([1.0, 1.0])
FINGER_TRAVEL = 0.05  # m, from open (0) to closed (pad faces meet at the axis)
PAD_HALF_THICK = 0.004
FINGER_OPEN_X = 0.05 + PAD_HALF_THICK  # pad centre x when open (inner face at 0.05)


@dataclass
class EnvConfig:
    timestep: float = 0.002
    control_dt: float = 0.05  # 20 Hz control
    finger_kp: float = 150.0  # N/m position servo per finger
    finger_damping: float = 4.0
    hand_kp: float = 4000.0
    hand_damping: float = 40.0
    hand_force_limit: float = 30.0  # N, servo force saturation
    max_lift_speed: float = 0.25  # m/s
    grip_slew_per_step: float = 0.06  # max change in g per control step
    z_min: float = 0.0
    z_max: float = 0.25
    box_half_size: float = 0.02
    pad_friction: float = 0.3
    noslip_iterations: int = 20  # suppresses MuJoCo soft-contact friction creep
    # Domain randomisation ranges (sampled per episode from the episode seed)
    mass_range: tuple[float, float] = (0.05, 0.20)
    friction_range: tuple[float, float] = (0.6, 1.0)
    # Sensor model
    force_noise_std: float = 0.02  # N, additive Gaussian on each channel
    contact_threshold: float = 0.1  # N, binary contact if pad normal force exceeds this

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EnvConfig":
        d = dict(d or {})
        for k in ("mass_range", "friction_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_xml(cfg: EnvConfig) -> str:
    bh = cfg.box_half_size
    # Hand origin height so that finger pads are centred on the box at hand_z = 0.
    pad_z = 0.03  # pads span z in [0.005, 0.055]; box spans [0, 2*bh]
    return f"""
<mujoco model="gripper_lift">
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="{cfg.noslip_iterations}"/>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="0.8 0.005 0.0001"/>
    <body name="hand" pos="0 0 {pad_z}" gravcomp="1">
      <joint name="hand_z" type="slide" axis="0 0 1" range="{cfg.z_min} {cfg.z_max}"
             damping="{cfg.hand_damping}"/>
      <geom name="palm" type="box" size="0.07 0.02 0.005" pos="0 0 0.05" mass="0.2"
            contype="0" conaffinity="0"/>
      <body name="finger_l" pos="{-FINGER_OPEN_X} 0 0" gravcomp="1">
        <joint name="finger_l" type="slide" axis="1 0 0" range="0 {FINGER_TRAVEL}"
               damping="{cfg.finger_damping}"/>
        <geom name="pad_l" type="box" size="{PAD_HALF_THICK} 0.015 0.025" mass="0.02"
              friction="{cfg.pad_friction} 0.005 0.0001"/>
      </body>
      <body name="finger_r" pos="{FINGER_OPEN_X} 0 0" gravcomp="1">
        <joint name="finger_r" type="slide" axis="-1 0 0" range="0 {FINGER_TRAVEL}"
               damping="{cfg.finger_damping}"/>
        <geom name="pad_r" type="box" size="{PAD_HALF_THICK} 0.015 0.025" mass="0.02"
              friction="{cfg.pad_friction} 0.005 0.0001"/>
      </body>
    </body>
    <body name="box" pos="0 0 {bh}">
      <freejoint name="box_free"/>
      <geom name="box" type="box" size="{bh} {bh} {bh}" mass="0.1" priority="1"
            friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="hand" body2="finger_l"/>
    <exclude body1="hand" body2="finger_r"/>
    <exclude body1="finger_l" body2="finger_r"/>
  </contact>
  <actuator>
    <position name="act_hand" joint="hand_z" kp="{cfg.hand_kp}" ctrlrange="{cfg.z_min} {cfg.z_max}"
              forcelimited="true" forcerange="-{cfg.hand_force_limit} {cfg.hand_force_limit}"/>
    <position name="act_l" joint="finger_l" kp="{cfg.finger_kp}" ctrlrange="0 {FINGER_TRAVEL}"/>
    <position name="act_r" joint="finger_r" kp="{cfg.finger_kp}" ctrlrange="0 {FINGER_TRAVEL}"/>
  </actuator>
</mujoco>
"""


@dataclass
class EnvState:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    grip_applied: float
    z_target: float
    step_count: int
    noise_rng_state: dict = field(default_factory=dict)
    qacc_warmstart: np.ndarray | None = None  # solver warm start; needed for bit-exact replay


class GripperLiftEnv:
    """Grasp-and-lift environment. Observations are returned as dicts of numpy arrays."""

    ACTION_DIM = 2
    TACTILE_DIM = 4  # [N_left, N_right, T_left, T_right] in newtons
    PROPRIO_DIM = 4  # [finger_l q, finger_r q, hand z, applied grip]

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.cfg.control_dt / self.cfg.timestep))
        m = self.model
        self._gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("pad_l", "pad_r", "box", "floor")}
        self._box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._jadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("hand_z", "finger_l", "finger_r")}
        self._box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_free")]
        self.domain: dict[str, float] = {}
        self.grip_applied = 0.0
        self.z_target = 0.0
        self.step_count = 0
        self.noise_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int) -> dict[str, np.ndarray]:
        """Deterministic reset: the same seed gives the same domain parameters,
        initial state and sensor-noise sequence."""
        rng = np.random.default_rng(seed)
        mass = float(rng.uniform(*self.cfg.mass_range))
        mu = float(rng.uniform(*self.cfg.friction_range))
        self.model.body_mass[self._box_body] = mass
        # Box geom has priority=1, so its friction is used for pad-box contacts.
        self.model.geom_friction[self._gid["box"], 0] = mu
        # Inertia consistent with a uniform cube.
        s = 2 * self.cfg.box_half_size
        self.model.body_inertia[self._box_body] = mass * s * s / 6.0
        self.domain = {"mass": mass, "friction": mu, "seed": int(seed)}
        mujoco.mj_resetData(self.model, self.data)
        self.grip_applied = 0.0
        self.z_target = 0.0
        self.step_count = 0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Settle the box on the floor (deterministic).
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self.noise_rng = np.random.default_rng([seed, 7919])
        return self._observe()

    # ------------------------------------------------------------------ state
    def get_state(self) -> EnvState:
        return EnvState(
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), act=self.data.act.copy(),
            ctrl=self.data.ctrl.copy(), time=float(self.data.time), grip_applied=self.grip_applied,
            z_target=self.z_target, step_count=self.step_count,
            noise_rng_state=self.noise_rng.bit_generator.state,
            qacc_warmstart=self.data.qacc_warmstart.copy(),
        )

    def set_state(self, s: EnvState) -> None:
        self.data.qpos[:] = s.qpos
        self.data.qvel[:] = s.qvel
        if s.act.size:
            self.data.act[:] = s.act
        self.data.ctrl[:] = s.ctrl
        self.data.time = s.time
        self.grip_applied = s.grip_applied
        self.z_target = s.z_target
        self.step_count = s.step_count
        self.noise_rng = np.random.default_rng()
        self.noise_rng.bit_generator.state = s.noise_rng_state
        mujoco.mj_forward(self.model, self.data)
        if s.qacc_warmstart is not None:
            self.data.qacc_warmstart[:] = s.qacc_warmstart

    # ------------------------------------------------------------------ step
    @staticmethod
    def clip_action(a: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(a, dtype=np.float64), ACTION_LOW, ACTION_HIGH)

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        a = self.clip_action(action)
        g_cmd = 0.5 * (a[0] + 1.0)
        d = np.clip(g_cmd - self.grip_applied, -self.cfg.grip_slew_per_step, self.cfg.grip_slew_per_step)
        self.grip_applied = float(np.clip(self.grip_applied + d, 0.0, 1.0))
        z_prev = self.z_target
        self.z_target = float(np.clip(self.z_target + a[1] * self.cfg.max_lift_speed * self.cfg.control_dt,
                                      self.cfg.z_min, self.cfg.z_max))
        self.data.ctrl[1] = self.grip_applied * FINGER_TRAVEL
        self.data.ctrl[2] = self.grip_applied * FINGER_TRAVEL
        # The height target is ramped linearly over the substeps (velocity-style
        # command); a step change every control period produces impulsive hand
        # accelerations that shake the object loose.
        for i in range(self.n_substeps):
            self.data.ctrl[0] = z_prev + (self.z_target - z_prev) * (i + 1) / self.n_substeps
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return self._observe()

    # ------------------------------------------------------------------ observe
    def contact_reading(self) -> ContactReading:
        return read_pad_contacts(self.model, self.data, self._gid["pad_l"], self._gid["pad_r"], self._gid["box"])

    def _observe(self) -> dict[str, np.ndarray]:
        c = self.contact_reading()
        clean = np.array([c.normal_l, c.normal_r, c.tangential_l, c.tangential_r])
        noisy = clean + self.noise_rng.normal(0.0, self.cfg.force_noise_std, size=4) if self.cfg.force_noise_std > 0 else clean.copy()
        q = self.data.qpos
        proprio = np.array([q[self._jadr["finger_l"]], q[self._jadr["finger_r"]], q[self._jadr["hand_z"]], self.grip_applied])
        box_z = float(q[self._box_qadr + 2])
        return {
            "tactile": noisy,            # what a deployable model may see
            "tactile_clean": clean,      # noise-free, for diagnostics only
            "proprio": proprio,
            "box_z": np.array([box_z]),  # privileged: task-outcome evaluation only
            "hand_z": np.array([q[self._jadr["hand_z"]]]),
        }

    def held(self, obs: dict[str, np.ndarray], min_lift: float) -> bool:
        """Task-success predicate: box raised by at least ``min_lift`` and still
        between the pads (pad normal contact on both sides)."""
        box_lift = obs["box_z"][0] - self.cfg.box_half_size
        thr = self.cfg.contact_threshold
        return bool(box_lift >= min_lift and obs["tactile_clean"][0] > thr and obs["tactile_clean"][1] > thr)
