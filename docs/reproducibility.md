# Reproducibility

## Environment

* Python 3.12 (MuJoCo/PyTorch wheels; `pyproject.toml` pins `>=3.10,<3.14`).
* Exact package versions used for H1-001: [`requirements.txt`](../requirements.txt).
* H1-001 was run on an Apple M5 (10 cores, 16 GB), macOS (Darwin 25.5), CPU only.
  Every run writes its git commit, package versions, and hardware into its
  summary JSON (`provenance` field).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev]"
```

**macOS gotcha.** On some macOS setups, `uv` marks the `.pth` file for the
editable install as hidden, and Python 3.12 then silently ignores it
(`ModuleNotFoundError: contact_learning`). Fix it with:

```bash
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
```

Tests set `pythonpath = ["src"]`, so `pytest` works either way.

## Determinism

* `GripperLiftEnv.reset(seed)` derives the domain parameters (mass, friction)
  and the sensor-noise stream from the seed alone. Repeated resets give
  bit-identical trajectories, including after another episode has run
  (`tests/test_environment.py`).
* `get_state`/`set_state` snapshot the MuJoCo state, the solver warm start, the
  gripper slew-limiter state, the height target, and the noise RNG. Paired
  trials replay bit-exactly (`tests/test_h1a_validity.py`). Without the
  warm-start field, replays differed by about 4e-14 N; this was found by the
  test and fixed.
* Torch: `torch.manual_seed`, `use_deterministic_algorithms(True, warn_only=True)`,
  and a single thread. CPU results should reproduce exactly on the same platform
  and library versions. Across platforms, expect small floating-point differences.
* Seed ranges for train, validation, predictor-test, H1-A trials, and H1-B trials
  are disjoint by construction (`src/contact_learning/utils/seeding.py`).

## Reproducing H1-001

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/h1/run_experiment.py --config configs/experiments/h1_001.yaml --stage all
.venv/bin/python experiments/h1/analyze_results.py --experiment-id H1-001
```

The run refuses to overwrite existing outputs unless `--overwrite` is passed.
Raw arrays and checkpoints go to `artifacts/h1/H1-001/` (git-ignored).
Checksums are recorded in `results/h1/summaries/H1-001_h1a_manifest.json` and
`H1-001_predictors.json`.

A smoke test of the full pipeline, with tiny sizes and **not** for interpretation:

```bash
.venv/bin/python experiments/h1/run_experiment.py --config configs/experiments/h1_001_smoke.yaml --stage all --overwrite
```
