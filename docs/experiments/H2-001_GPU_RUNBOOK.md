# H2-001 — GPU runbook (ManiFeel route)

*Written 2026-11-18. The team chose route 1 from
[`H2-001_BASELINE_SELECTION.md`](H2-001_BASELINE_SELECTION.md) §6: ManiFeel on a
rented GPU. **Nothing here has been run.** The GPU box does not exist yet.
Every step below is a plan, and the report will record what actually
happened.*

## 1. What gets built, and what it is not

| Piece | Source | Status |
|---|---|---|
| Simulator and tasks | ManiFeel (Isaac Gym / TacSL), pinned `ebfa9e1` | external, unmodified |
| **Base policy** | ManiFeel's own Diffusion Policy, *vision + tactile force field* (`visff_wrist`), **Peg Insertion**, 50 demos | to be **reproduced** (§3) |
| Forecaster and corrector | `src/contact_learning/h2/forecast_corrector.py` (ours) | written; CPU unit tests pass (11 tests, synthetic data only) |
| Disturbance injection | external force or torque on the held plug, mid-episode, via Isaac Gym's rigid-body force tensors | not written. Needs the box to develop against. |

The combined system is **a TacPAC-structured policy of our own, not TacPAC**.
Every report must say so.

### Why Peg Insertion with force fields

ManiFeel's paper (arXiv 2505.18472, Table II) gives the largest tactile benefit
on this task. With Diffusion Policy:

| Inputs | Success |
|---|---|
| Vision only | 0.14 ± 0.01 |
| Vision + tactile RGB | 0.21 ± 0.02 |
| **Vision + tactile force field** | **0.40 ± 0.02** |

These are averages over the last 10 training epochs, 50 initial conditions,
and 3 seeds.

H2 needs live touch to be *informative*. This is the task where ManiFeel
shows it is. The force field is a 3 × 10 × 14 tensor (normal plus two shear
channels per taxel), which is small enough to forecast explicitly.

## 2. The box

**Requirements**
* Linux.
* One NVIDIA GPU with ≥ 24 GB. L40S and RTX PRO 6000 class are good choices.
  Avoid H100, H200, and B200: they are datacenter parts without full graphics
  support, and Isaac Gym's camera sensors need Vulkan rendering.
* Python 3.8 via Miniforge, installed by ManiFeel's own installer.
* About 30 GB of disk for each task's data and environment.

**Main risk:** container platforms that expose only CUDA compute. The
`check` step of `scripts/h2/setup_manifeel_box.sh` tests for this first. If it
fails, use a full VM instead.

**The rental service already connected to this workspace** can run GPU jobs,
but it has these limits:

| Item | Detail |
|---|---|
| Balance | 10 credits (free plan), checked 2026-11-18 |
| L40S price | 71 credits per GPU-hour |
| Command time | each command runs for at most 90 s, so long jobs must use `nohup` and polled logs |
| Graphics support | unknown |

Topping up credits, or renting anywhere, is the team's decision. Nothing has
been rented.

**Alternative:** a cloud VM with SSH access, added to `~/.ssh/config` on this
Mac. This avoids the 90-second limit.

## 3. Phase 2 — reproduce the base policy (before any H2 work)

1. `setup_manifeel_box.sh check`. Stop if there is no graphics-capable driver.
2. `clone`, then `isaacgym`.
   * The TacSL Isaac Gym binary comes from the authors' Google Drive.
   * The README and `install.sh` link *different* files.
   * Confirm the choice with the team, and record the file ID and its SHA-256.
3. `install`, then `data` (`pih_quan_June06.zip`, 4.7 GB), then `smoke`.
4. Train ManiFeel `visff_wrist` on Peg Insertion with 50 demos and ManiFeel's
   default config, for the paper's 3 seeds.
5. Evaluate with ManiFeel's runner: 50 initial conditions, 500 steps.
6. **Report the measured success next to the paper's 0.40 ± 0.02.** Keep two
   things separate:
   * *reproduced published performance*: same task, data, and config;
   * anything measured in a modified environment.
7. Record:
   * policy inference latency: 100 DDPM steps per chunk, re-planning every
     8 steps at 10 Hz;
   * the exact observation key order and timing (`n_obs_steps=2`);
   * whether the force field is stored raw or as a rendered image.

**Gate:** continue only if the reproduction lands within about ±0.08 of 0.40,
or the gap is explained.

### Third-party checkpoints

There are third-party ManiFeel checkpoints on Hugging Face
(`guitarorangedog/manifeel-*`). They are LeRobot-format, and their training
recipe (`kwk2696/robo-synapse`) is not public, so they are not the
reproduction target.

## 4. Phase 2b — the forecast-using policy, and the nominal gate

1. Freeze the reproduced base policy.
2. Train `TactileForecaster` on the demos' measured future force fields.
3. Train `TactileCorrector` in two variants:
   * with the forecast, using forecast dropout `p_null = 0.2`;
   * without the forecast (condition D), at a matched parameter count.
   Both use TacPAC-style targets: demonstrated actions minus the frozen base
   plan, on the unexecuted suffix, at sampled execution offsets.
4. **Nominal gate**, with no disturbances. Before any H2 claim, show both:
   * the corrected policy is not worse than the base policy;
   * nulling the forecast changes its actions measurably.
   If the corrector ignores its forecast, H2 cannot be tested on this policy.
   Report that as a negative result.

## 5. Phase 3 and later

* `H2-001_PROTOCOL.md` will be committed **before** any confirmatory
  disturbance run. It will define:
  * disturbances: a lateral push or torque on the held plug, with
    severity × timing set by a pilot so that some trials are recoverable;
  * conditions A–E;
  * the matched controls already implemented in `apply_intervention`:
    a shuffled forecast from another trial, a norm-matched perturbation, and
    a context-only refresh;
  * the OOD check (`ActivationStats`);
  * matched inference budgets;
  * paired seeds and the statistics.
* **The protocol's decision rules are not set here.**

## 6. Rough compute budget (estimate, not measured)

| Stage | GPU-hours |
|---|---|
| Reproduction: 3 seeds × ≤ 8 h (ManiFeel's own job limit on an A30; an L40S should be faster) | ≤ 24 |
| Forecaster and two corrector variants × 5 seeds | 20–40 |
| Pilot, confirmatory disturbance grid, and evaluation | 30–60 |
| **Total** | **about 75–125**, roughly 5,000–9,000 credits at the L40S price above |

Stopping after Phase 2 and the nominal gate (about 30–50 GPU-hours) already
answers whether the route is viable.
