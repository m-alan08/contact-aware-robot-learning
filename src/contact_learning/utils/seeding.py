"""Deterministic seed handling."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


# Disjoint seed ranges for episode generation. A predictor trained with
# predictor seed s uses data seeds in [TRAIN_BASE + s*STRIDE, ...), validation in
# [VAL_BASE + s*STRIDE, ...). Evaluation trials use TEST_BASE + i. Ranges never overlap
# for s < 100 and fewer than STRIDE episodes per split.
STRIDE = 100_000
TRAIN_BASE = 0
VAL_BASE = 10_000_000
TEST_BASE = 50_000_000
H1B_BASE = 60_000_000


def train_seeds(pred_seed: int, n: int) -> list[int]:
    assert n < STRIDE
    return [TRAIN_BASE + pred_seed * STRIDE + i for i in range(n)]


def val_seeds(pred_seed: int, n: int) -> list[int]:
    assert n < STRIDE
    return [VAL_BASE + pred_seed * STRIDE + i for i in range(n)]


def test_seeds(n: int, base: int = TEST_BASE) -> list[int]:
    return [base + i for i in range(n)]
