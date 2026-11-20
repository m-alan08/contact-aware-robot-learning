"""Privileged teacher and disturbance sampling for H2-001A.

The teacher is NOT deployable: it reads the object's mass and friction, the
current external disturbance, and the lift plan. It labels the grip a careful
operator would choose: just enough normal force (with a margin) for the load
expected over the next few steps, never above a safety fraction of the crush
limit. Students are trained to imitate it from deployable inputs only.
"""
from __future__ import annotations

import numpy as np

from contact_learning.h2a.env import G, Disturbance, DisturbedLiftEnv, grip_for_normal

MARGIN = 1.4
LOOKAHEAD = 5          # steps; covers the grip slew time (~0.45 N per step)
CRUSH_SAFETY = 0.88


def expert_grip(env: DisturbedLiftEnv, t: int) -> float:
    if t < env.plan.close_start:
        return 0.0
    acc = env.planned_hand_acc()
    hi = min(len(acc), t + LOOKAHEAD + 1)
    # Current disturbance (privileged) is assumed to persist; future disturbances are unknown.
    n_req = max(env.required_normal(t, hand_acc=float(a), include_disturbance=True)
                for a in acc[t:hi]) if hi > t else env.required_normal(t)
    n_target = min(MARGIN * n_req, CRUSH_SAFETY * env.crush_limit)
    return grip_for_normal(n_target)


def disturbance_for_severity(env: DisturbedLiftEnv, kind: str, severity: float, onset: int,
                             duration: int = 10) -> Disturbance:
    """Disturbance whose static per-pad normal-force requirement after onset equals
    ``severity`` x crush limit. Severity < 1 is physically recoverable (holding is
    possible without crushing); > 1 is not."""
    m, mu, nc = env.domain["mass"], env.mu0, env.crush_limit
    w = m * G
    need = 2 * mu * severity * nc  # total tangential capacity required
    if kind == "load":
        mag = need - w
    elif kind == "push":
        mag = float(np.sqrt(max(need ** 2 - w ** 2, 0.0)))
    elif kind == "friction":
        mag = w / need  # friction multiplier
    else:
        raise ValueError(kind)
    return Disturbance(kind=kind, onset=onset, magnitude=float(mag), duration=duration,
                       severity=float(severity), recoverable=bool(severity < 1.0))
