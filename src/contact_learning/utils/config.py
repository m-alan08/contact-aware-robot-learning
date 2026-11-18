"""Minimal YAML config system: an experiment config may name an environment and a
predictor config file; values in the experiment file override those defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    resolved: dict[str, Any] = {}
    for key, sub in (("environment_config", "env"), ("predictor_config", "predictor")):
        if key in cfg:
            p = Path(cfg[key])
            p = p if p.is_absolute() else REPO_ROOT / p
            resolved[sub] = load_yaml(p)
    return _deep_update(resolved, cfg)
