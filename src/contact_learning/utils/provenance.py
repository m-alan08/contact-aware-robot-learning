"""Record code version, software versions, and hardware for experiment logs."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from contact_learning.utils.config import REPO_ROOT


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def software_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("mujoco", "numpy", "torch", "scipy", "matplotlib", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def hardware_info() -> dict:
    info = {"machine": platform.machine(), "processor": platform.processor()}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return info


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance() -> dict:
    return {"git": git_info(), "software": software_versions(), "hardware": hardware_info()}
