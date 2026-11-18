"""Simulated contact ("tactile") signal extraction.

What this signal is
-------------------
For each finger pad we sum, over all MuJoCo contacts between that pad geom and the
object geom, the contact-frame force returned by ``mj_contactForce``:

* ``normal``     -- sum of normal force components (N)
* ``tangential`` -- magnitude of the summed tangential (friction) force vector (N)

This is an idealised, noise-free, infinitely fast *net contact wrench* readout. It is
NOT equivalent to a physical tactile sensor: real tactile arrays (e.g. those that
might be mounted on an SO-101 gripper) report spatially distributed, filtered,
hysteretic, temperature- and calibration-dependent signals, with their own bandwidth
and latency. MuJoCo forces also depend on the soft-contact parameters (solref/solimp)
which were not calibrated to any real material. Additive Gaussian noise is applied by
the environment as a crude stand-in for measurement noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ContactReading:
    normal_l: float
    normal_r: float
    tangential_l: float
    tangential_r: float
    n_contacts: int


def read_pad_contacts(model: mujoco.MjModel, data: mujoco.MjData, pad_l: int, pad_r: int, obj: int) -> ContactReading:
    f6 = np.zeros(6)
    normal = {pad_l: 0.0, pad_r: 0.0}
    tang = {pad_l: np.zeros(3), pad_r: np.zeros(3)}
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 == obj and g2 in normal:
            pad = g2
        elif g2 == obj and g1 in normal:
            pad = g1
        else:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        # Contact frame rows: [normal, tangent1, tangent2]. Tangential forces are
        # rotated to world frame before summing so multiple points add as vectors.
        frame = c.frame.reshape(3, 3)
        normal[pad] += abs(f6[0])
        tang[pad] += f6[1] * frame[1] + f6[2] * frame[2]
        n += 1
    return ContactReading(
        normal_l=float(normal[pad_l]), normal_r=float(normal[pad_r]),
        tangential_l=float(np.linalg.norm(tang[pad_l])), tangential_r=float(np.linalg.norm(tang[pad_r])),
        n_contacts=n,
    )
