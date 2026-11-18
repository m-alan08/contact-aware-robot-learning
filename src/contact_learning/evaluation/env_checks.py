"""Checks that the simulated contact signal responds meaningfully to actions.
Produces numbers (and optionally a figure) recorded in the experiment report."""
from __future__ import annotations

import numpy as np

from contact_learning.environments.gripper_lift import FINGER_TRAVEL, GripperLiftEnv


def steady_grip_force(env: GripperLiftEnv, seed: int, g: float, steps: int = 30) -> np.ndarray:
    env.reset(seed)
    o = None
    for _ in range(steps):
        o = env.step(np.array([2 * g - 1, 0.0]))
    return o["tactile_clean"]


def grip_sweep(env: GripperLiftEnv, seed: int = 123, n: int = 26) -> dict:
    gs = np.linspace(0, 1, n)
    F = np.array([steady_grip_force(env, seed, g) for g in gs])
    # Expected (rigid-limit) model: no contact until the pads reach the box, then
    # normal force = kp * (target - contact position).
    g_contact = (FINGER_TRAVEL - env.cfg.box_half_size) / FINGER_TRAVEL
    expected = np.maximum(0.0, env.cfg.finger_kp * FINGER_TRAVEL * (gs - g_contact))
    return {"g": gs.tolist(), "normal_l": F[:, 0].tolist(), "normal_r": F[:, 1].tolist(),
            "expected_normal": expected.tolist(), "g_contact_expected": float(g_contact),
            "max_abs_dev_from_expected_N": float(np.abs(F[:, 0] - expected).max())}


def lift_check(env: GripperLiftEnv, seeds: list[int], g: float = 0.95, lift: float = 0.6, steps: int = 50) -> dict:
    rows = []
    for s in seeds:
        env.reset(s)
        for t in range(steps):
            o = env.step(np.array([2 * min(g, g * (t + 1) / 12) - 1, lift if t >= 15 else 0.0]))
        weight = env.domain["mass"] * 9.81
        tsum = o["tactile_clean"][2] + o["tactile_clean"][3]
        rows.append({"seed": s, "mass": env.domain["mass"], "friction": env.domain["friction"],
                     "held": env.held(o, 0.05), "weight_N": weight, "tangential_sum_N": float(tsum),
                     "normal_l_N": float(o["tactile_clean"][0])})
    held = [r for r in rows if r["held"]]
    rel = [abs(r["tangential_sum_N"] - r["weight_N"]) / r["weight_N"] for r in held]
    return {"rows": rows, "n_held": len(held), "max_rel_err_tangential_vs_weight": float(max(rel)) if rel else None}


def capacity_check(env: GripperLiftEnv, seeds: list[int], gs=(0.7, 0.75, 0.8, 0.85)) -> dict:
    """Coulomb prediction: box held iff 2 * mu * N >= m * g_grav (static, small accel)."""
    agree, rows = 0, []
    for s in seeds:
        for g in gs:
            env.reset(s)
            for t in range(50):
                o = env.step(np.array([2 * min(g, g * (t + 1) / 12) - 1, 0.4 if t >= 15 else 0.0]))
            N = env.cfg.finger_kp * FINGER_TRAVEL * max(0.0, g - 0.6)
            pred = 2 * env.domain["friction"] * N >= env.domain["mass"] * 9.81
            h = env.held(o, 0.05)
            margin = (2 * env.domain["friction"] * N) / (env.domain["mass"] * 9.81)
            rows.append({"seed": s, "g": g, "coulomb_pred_held": bool(pred), "held": h, "capacity_ratio": margin})
            agree += int(pred == h)
    return {"rows": rows, "agreement": agree / len(rows)}


def determinism_check(env: GripperLiftEnv, seed: int = 7) -> dict:
    acts = np.stack([np.linspace(-1, 0.9, 40), np.r_[np.zeros(20), 0.6 * np.ones(20)]], 1)
    outs = []
    for _ in range(2):
        env.reset(seed)
        outs.append(np.array([env.step(a)["tactile"] for a in acts]))
    env.reset(seed + 1)
    [env.step(a) for a in acts]
    env.reset(seed)
    outs.append(np.array([env.step(a)["tactile"] for a in acts]))
    return {"identical_repeat": bool(np.array_equal(outs[0], outs[1])),
            "identical_after_other_episode": bool(np.array_equal(outs[0], outs[2]))}


def run_all(env: GripperLiftEnv) -> dict:
    return {"grip_sweep": grip_sweep(env), "lift": lift_check(env, list(range(900, 920))),
            "capacity": capacity_check(env, list(range(950, 970))), "determinism": determinism_check(env)}
