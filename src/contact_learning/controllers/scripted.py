"""Nominal open-loop grasp-and-lift planner and controlled action corrections.

Actions are (T, 2) arrays: column 0 = grip command in [-1, 1] (g = (a+1)/2),
column 1 = lift velocity command in [0, 1] (fraction of max lift speed).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from contact_learning.environments.gripper_lift import ACTION_HIGH, ACTION_LOW

ACTION_RANGE = ACTION_HIGH - ACTION_LOW  # per-dimension span: [2, 1]
CORRECTION_TYPES = ("grip_decrease", "grip_increase", "lift_decrease", "lift_increase")


@dataclass
class PlanParams:
    g_hold: float
    close_start: int
    close_steps: int
    hold_steps: int
    lift_speed: float  # fraction of max lift speed
    lift_height: float  # m

    def to_dict(self):
        return asdict(self)


def sample_plan_params(rng: np.random.Generator) -> PlanParams:
    return PlanParams(
        g_hold=float(rng.uniform(0.85, 0.98)),
        close_start=int(rng.integers(2, 7)),
        close_steps=int(rng.integers(6, 13)),
        hold_steps=int(rng.integers(3, 7)),
        lift_speed=float(rng.uniform(0.4, 1.0)),
        lift_height=float(rng.uniform(0.08, 0.14)),
    )


REFERENCE_SLEW = 0.06  # gripper slew (per step) the nominal planner was designed for (H1-001)


def slow_gripper_lift_delay(g_hold: float, slew: float | None, ref_slew: float = REFERENCE_SLEW) -> int:
    """Extra steps a speed-aware planner waits before lifting with a gripper slower than
    the reference, so that the grip has made the same progress at lift start as with
    the reference gripper. Zero for the reference speed and faster grippers, so plans
    there are identical to H1-001."""
    if slew is None or slew >= ref_slew:
        return 0
    return int(max(0, np.ceil(g_hold / slew) - np.ceil(g_hold / ref_slew)))


def nominal_plan(p: PlanParams, T: int, max_lift_speed: float, control_dt: float,
                 slew: float | None = None) -> np.ndarray:
    """Open-loop plan: wait, ramp grip to g_hold, hold, lift at constant speed to
    lift_height, then hold still. ``slew`` (optional) makes the plan speed-aware for
    grippers slower than the reference (see ``slow_gripper_lift_delay``); the default
    reproduces H1-001 exactly."""
    a = np.zeros((T, 2))
    g = np.zeros(T)
    for t in range(T):
        if t < p.close_start:
            g[t] = 0.0
        elif t < p.close_start + p.close_steps:
            g[t] = p.g_hold * (t - p.close_start + 1) / p.close_steps
        else:
            g[t] = p.g_hold
    a[:, 0] = 2 * g - 1
    lift_start = p.close_start + p.close_steps + p.hold_steps + slow_gripper_lift_delay(p.g_hold, slew)
    n_lift = int(np.ceil(p.lift_height / (p.lift_speed * max_lift_speed * control_dt)))
    a[lift_start:lift_start + n_lift, 1] = p.lift_speed
    return a


@dataclass
class Correction:
    kind: str  # one of CORRECTION_TYPES
    magnitude: float  # requested, as a fraction of the action range (0..1)

    def delta(self, n: int) -> np.ndarray:
        d = np.zeros((n, 2))
        dim = 0 if self.kind.startswith("grip") else 1
        sign = -1.0 if self.kind.endswith("decrease") else 1.0
        d[:, dim] = sign * self.magnitude * ACTION_RANGE[dim]
        return d


def apply_correction(chunk: np.ndarray, corr: Correction) -> np.ndarray:
    """Add a constant offset to one action dimension over the whole chunk, then clip
    to valid action bounds."""
    return np.clip(chunk + corr.delta(len(chunk)), ACTION_LOW, ACTION_HIGH)


def effective_magnitude(orig: np.ndarray, corr: np.ndarray) -> float:
    """Normalised correction magnitude actually applied (after clipping):
    RMS over time of the largest range-normalised per-dimension change.
    Equals the requested magnitude when no clipping occurs."""
    d = (np.abs(corr - orig) / ACTION_RANGE).max(axis=1)
    return float(np.sqrt(np.mean(d ** 2)))
