# contact-aware-robot-learning

Research code for **Contact-Aware Robot Learning** (Praneeth Samineni, Aarav Kothari, Alan).

> **Overarching question:** when should a robot retain, revise, or disregard its
> predicted contact during execution?

The research plan and literature review are in
[`docs/literature/literature_review.pdf`](docs/literature/literature_review.pdf).
The questions and hypotheses (original wording preserved) are in
[`docs/research_questions.md`](docs/research_questions.md) and
[`docs/hypotheses.md`](docs/hypotheses.md).

## Current status (2026-11-18)

| Hypothesis | Status | Where |
|---|---|---|
| **H1-A**: correcting an action makes the original-action forecast less accurate than a refreshed one | **Preliminary support** in simulation, limited to corrections the predictor is sensitive to | [H1 preliminary results](docs/H1_PRELIMINARY_RESULTS.md) |
| **H1-B**: this mismatch can reduce task success | **Preliminary support, narrow**: one slow-gripper scenario; refreshing at the correction prevents drops that live feedback and fixed-rate refresh do not; a non-learned feed-forward alternative is untested | [H1-001 report](docs/experiments/H1-001.md) |
| **H1-002**: is *learned* forecasting necessary? | **No, not in this task.** A two-parameter analytic grip→force model matched the learned predictor at every gripper speed and sensing latency. Anticipating corrections helps most with sensing delay and is not a slow-gripper artifact. | [H1-002 results](docs/H1_002_RESULTS.md), [report](docs/experiments/H1-002.md) |
| H2, H3 | Not started (deliberately) | `experiments/h2`, `experiments/h3` |

**Start here:** [`docs/H1_002_RESULTS.md`](docs/H1_002_RESULTS.md) (latest), then
[`docs/H1_PRELIMINARY_RESULTS.md`](docs/H1_PRELIMINARY_RESULTS.md) (H1-001, with its 2026-11-18 erratum).
[`docs/experiment_registry.md`](docs/experiment_registry.md) lists every experiment.

All results so far come from a **simplified MuJoCo simulation**, not a robot.
The "tactile" signal is the net pad-object contact force reported by MuJoCo; it
is not a model of a real tactile sensor. Nothing here establishes
physical-robot performance.

## The setup in one paragraph

A two-finger parallel gripper on a vertical slide grasps and lifts a box with
randomized mass and friction (`src/contact_learning/environments/gripper_lift.py`).
The action is a grip command (slew-rate-limited, like a slow servo) plus a lift
velocity, at 20 Hz. The contact observation has four channels: pad normal and
tangential force, left and right, in newtons. A small residual MLP predicts the
next 10 steps (0.5 s) of contact from a 4-step history and a proposed 10-step
action chunk. It is compared against persistence, ridge-regression, and
history-only baselines. The H1 experiments compare a forecast made for the
*original* action chunk (cached) with one made for the *corrected* chunk
(refreshed), scored against what actually happens when the corrected chunk is
executed. The H1 experiments also check whether that difference changes the
outcome of a closed-loop grasp-safety controller.

## Repository layout

```
docs/                      research documentation (start with H1_PRELIMINARY_RESULTS.md)
  literature/              the research plan / literature review (PDF)
  experimental_protocol.md registered procedures and decision criteria (+ amendments)
  experiment_registry.md   every experiment, status, outcome
  experiments/H1-001.md    full report for H1-001
src/contact_learning/
  environments/            MuJoCo task (GripperLiftEnv)
  sensors/                 contact-signal extraction (what the "tactile" signal is)
  predictors/              features (explicit I/O), data generation, models, training
  controllers/             nominal planner and action corrections
  evaluation/              rollouts, H1-A paired trials, H1-B closed loop, metrics, stats
  utils/                   config, seeding, provenance
configs/                   YAML: environments/, predictors/, experiments/
experiments/h1/            run_experiment.py, analyze_results.py
experiments/h2, h3/        placeholders (not started)
results/h1/                version-controlled figures/ and summaries/ (small)
artifacts/                 raw outputs and checkpoints (git-ignored; checksums in summaries)
tests/                     sanity and validity tests (pytest)
```

## Installation

Requires Python 3.10–3.13 (developed with 3.12). MuJoCo runs on CPU and no GPU
is needed.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

(`python -m venv` and `pip` work too.) If `import contact_learning` fails on
macOS, see [`docs/reproducibility.md`](docs/reproducibility.md).

## Reproducing H1-001

```bash
# full run: environment validation, predictor training (5 seeds), H1-A, H1-B  (~15 min on an M-series laptop)
.venv/bin/python experiments/h1/run_experiment.py --config configs/experiments/h1_001.yaml --stage all
# figures + summary statistics from saved outputs
.venv/bin/python experiments/h1/analyze_results.py --experiment-id H1-001
# quick smoke test of the whole pipeline (results not meaningful)
.venv/bin/python experiments/h1/run_experiment.py --config configs/experiments/h1_001_smoke.yaml --stage all --overwrite
```

Runs refuse to overwrite existing outputs unless `--overwrite` is passed.

## Reproducing H1-002

```bash
gh release download h1-002-artifacts -R m-alan08/contact-aware-robot-learning   # H1-001 + H1-002 raw artifacts
tar -xzf h1-001-artifacts.tar.gz && tar -xzf h1-002-artifacts.tar.gz
.venv/bin/python experiments/h1/analyze_h1_002.py --which grid                   # figures + statistics from saved outputs
# or rerun everything: see docs/experiments/H1-002.md §10 (~20 min on 8-10 cores)
```

## Conventions for collaborators

* Register the analysis plan and decision criteria in `docs/experimental_protocol.md`
  **before** running a new experiment. Record any later change as a dated amendment.
* Give every experiment an ID (`H1-002`, …), a row in `docs/experiment_registry.md`,
  and a report in `docs/experiments/`. Never overwrite a previous experiment's
  outputs. If a bug invalidates results, mark them invalid and keep the record.
* Commit configs, small summaries, and figures. Keep raw data and checkpoints
  in `artifacts/`, with checksums recorded.
* Passing tests is not evidence for a hypothesis.

No license has been chosen for this repository yet.
