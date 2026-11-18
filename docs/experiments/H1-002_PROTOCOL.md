# H1-002 protocol — Is learned tactile forecasting necessary?

**Status: registered 2026-11-18, before any confirmatory H1-002 trial was run.**
Changes made after registration are listed under *Amendments*, with dates and
reasons. The original text is never edited.

Builds on H1-001 ([report](H1-001.md), including its 2026-11-18 erratum). H1-001
artifacts, configurations, checkpoints, and conclusions are reused read-only.

## 1. Research questions

* **RQ-a.** Does a refreshed *learned* tactile forecast improve closed-loop
  success beyond a simple *non-learned, action-aware* anticipation rule that has
  the same information?
* **RQ-b.** Does a predictor need corrected actions in its training data in
  order to give useful refreshed forecasts?
* **RQ-c.** Does the benefit of anticipation, learned or analytic, depend on
  gripper speed and sensing latency? Does it persist when the gripper is fast
  and feedback is immediate?
* **RQ-d.** Does the H1-001 result generalize beyond its original speed, its
  latency, and the grip-decrease direction?

## 2. Competing explanations (none assumed)

* **Expl. A — learned forecasting matters.** The learned predictor captures
  action-conditioned contact dynamics that a simple rule cannot.
* **Expl. B — simple anticipation suffices.** A non-learned rule using the
  grip command, a calibrated grip→force gain, and known actuator limits gets
  the same benefit.
* **Expl. C — latency/actuation artifact.** The benefit of anticipating exists
  mainly because the gripper is slow or the feedback arrives late. It vanishes
  with fast actuation and immediate feedback.

Explanation C is not exclusive with A or B. It concerns *when* anticipation of
any kind matters.

## 3. Task and fixed elements (unchanged from H1-001 unless stated)

* `GripperLiftEnv` with `configs/environments/gripper_lift.yaml`; T = 60
  steps at 20 Hz; K = 4; H = 10.
* The safety controller is identical across conditions:
  * minimum force `N_min = 1.7 N`, controller gain `κ = 7.5 N` per unit grip;
  * a one-sided live term every step after `t_c`;
  * a forecast term at forecast times.
* Correction times `t_c ∈ {10, 20, 30}` with nominal grasp contact; the
  correction persists to the end of the episode.
* Success: box raised ≥ 5 cm and both pads in contact at T.
* **Speed-aware planner (new; changes nothing at or above the H1-001 speed).**
  For grippers slower than the reference slew 0.06, the nominal plan delays the
  lift by `ceil(g_hold/slew) − ceil(g_hold/0.06)` steps. Plans at slew ≥ 0.06
  are identical to H1-001. This was checked before registration: nominal success
  (no correction, no controller) is 200/200 at every tested speed.

## 4. Conditions (closed loop)

Lettering follows the H1-002 request. It differs from H1-001's lettering.

| ID | Condition | Forecast source | Refresh at `t_c` | Deployable |
|---|---|---|---|---|
| **A** | cached learned | H1-001 corrected-trained MLP (at slew 0.06 the H1-001 checkpoints; at other speeds retrained identically) | no (boundary forecasts only) | yes |
| **B** | refreshed learned | same predictor as A | yes, event-triggered | yes |
| **C** | live feedback only | none | — | yes |
| **D** | refreshed analytic | non-learned grip→force model (§5) | yes, event-triggered | yes |
| **E** | refreshed nominal-trained learned | MLP trained only on uncorrected actions (§6) | yes, event-triggered | yes |
| **O** | oracle | true simulated future from the current state | yes | **no** (diagnostic only; excluded from all deployable comparisons) |

* All conditions receive the same delayed tactile stream, the same
  proprioception, the same executed actions, and the same plan and correction
  information. They differ **only** in the forecast source and timing in the
  table.
* Forecast schedule: as in H1-001, a forecast at every chunk boundary
  (`t ≡ 0 mod 10`, made before a same-step correction is known), plus for
  B/D/E/O one event-triggered forecast at `t_c` that *replaces* the stale one.

## 5. Analytic anticipation baseline (D) — information and calibration

* **Model.**
  1. The future applied grip is computed exactly from the commanded chunk and
     the **known actuator slew limit** (from the actuator specification, the
     same value the simulator uses).
  2. Predicted mean pad normal force is `N̂_k = κ̂ · max(0, ĝ_k − ĝ0)`.
* **`κ̂` and the prior `ĝ0_prior`** are fitted once by a calibration
  procedure: steady-state grip sweeps on **20 calibration objects**, seeds
  `90,000,000 + i`, disjoint from all training and evaluation seeds. This
  stands in for calibrating a gripper on known objects.
* **Online `ĝ0`.** From the latest *available* (delayed) measurement:
  `ĝ0 = g_applied(τ) − N_meas(τ)/κ̂` if contact is measured; otherwise `ĝ0_prior`.
* **No privileged parameters.** The model never reads object mass, friction,
  or the simulator's contact parameters. Mass and friction enter only through
  the shared `N_min` (the worst case over the object distribution), exactly
  as for every other condition.
* The static condition `2μN ≥ mg` is **not** used per object, because μ and m
  are unobservable at deployment. A privileged per-object analytic oracle is
  *not* part of this experiment.

## 6. Nominal-trained predictor (E)

Same architecture, optimizer, epochs, early stopping, and episode seeds as the
corrected-trained predictor for the same speed and seed. The only difference is
that the random corrections are removed from the training and validation
episodes; the small N(0, 0.02) action noise is kept. Five predictor seeds
(0–4) for every speed. No corrected trajectory enters its training or
validation data. This is checked by the generator flag and a test.

## 7. Independent variables (confirmatory grid)

| Variable | Levels |
|---|---|
| Gripper slew per step | 0.03 (slower; full stroke 1.67 s; force rise 10–90% ≈ 350 ms), **0.06 (H1-001; 0.83 s; ≈ 150 ms)**, 0.12 (faster; 0.42 s; ≈ 50 ms), 0.24 (fastest; 0.21 s; within one control step) |
| Sensing latency L (tactile stream only) | 0, 1, 2, 4 control steps = 0, 50, 100, 200 ms |
| Correction (direction, magnitude) | grip decrease m ∈ {0, 0.1, 0.2, 0.3, 0.4} (full grid) |
| Direction block | lift increase m ∈ {0.3, 0.6} at slew 0.06 × L ∈ {0, 2} |
| Predictor training distribution | corrected (A, B) vs. nominal-only (E) |

That gives 4 × 4 × 5 = 80 grid cells plus 4 direction cells, with **n = 150
trials per cell**.

* **Trial seeds** `80,000,000 + j` are **shared across all cells**, so the
  same object and plan appear in every cell. A seed with no valid `t_c` in a
  cell is skipped, and the next seed is used; skips are counted.
* Learned conditions (A, B, E) run with each of the 5 predictor seeds. C, D,
  and O run once per trial.

**Latency handling (causal).**
* At step t the controller has tactile observations up to index `τ = t − L`.
  Proprioception used by forecasters is taken at τ too, so forecasters get a
  self-consistent snapshot. All forecasters receive the identical snapshot.
* Forecasters predict from τ using the actions already executed in
  `[τ, t)` plus the planned chunk. Only the predicted steps after t are used,
  so the horizon shrinks by L.
* The live term uses `N_meas(τ)`.
* Sensor latency (L), inference latency (measured wall time), and actuator
  delay (one control step zero-order hold, plus slew) are recorded separately.

## 8. Metrics

* **Primary:** task success, paired by trial.
* **Secondary:**
  * failure rate;
  * recovery rate: success among episodes whose measured force dipped below
    `N_min` after `t_c`;
  * minimum measured force after `t_c`;
  * action difference `Σ_t |g_X − g_A|`;
  * forecasts per episode;
  * per-forecast inference wall time;
  * end-to-end decision latency = L·50 ms + inference + 50 ms actuation hold;
  * MLP parameter count.
* **Forecast accuracy (Stage 1):**
  * on the H1-001 H1-A trial set (2100 trials, slew 0.06, read-only): refreshed
    error RMSE (all channels for MLPs; normal channels for all forecasters,
    including analytic);
  * per speed: held-out test windows from corrected and from nominal episodes
    (300 episodes each, seeds `55,000,000 + i`).

## 9. Statistics

* **Unit of analysis:** the trial. Learned conditions are averaged over
  predictor seeds within a trial.
* **Uncertainty:** a **hierarchical bootstrap** (2000 resamples) that
  resamples trial seeds (clusters, shared across cells) and, independently,
  the 5 predictor seeds. Per-seed estimates and between-seed SDs are reported
  separately.
* **Practical effect threshold:** δ = 0.03 absolute success (3 percentage points).
* **"Challenging" pool:** grip-decrease m ∈ {0.2, 0.3, 0.4}.

**Confirmatory contrasts** (challenging pool unless stated):
* **K1:** B − D at (slew 0.06, L 0).
* **K2:** B − D pooled over all 16 speed × latency cells (excluding flagged cells, §11).
* **K3:** B − E at (0.06, L 0) and pooled over all cells.
* **K4:** (B − C) and (D − C) at (slew 0.24, L 0).
* **K5:** (B − C) and (D − C) at (slew 0.03, L 0) and at (slew 0.06, L 4).
* **K6:** A − C pooled over all cells (does a stale forecast ever *hurt* relative to no forecast?).
* **Nominal safety:** at m = 0, every condition's success and its action
  difference from C (false-alarm compensation).

## 10. Decision criteria

| Question | Supports | Criterion |
|---|---|---|
| RQ-a | **Expl. A** (learned needed) | K1 **and** K2: B − D ≥ δ and 95% CI lower bound > 0 |
| RQ-a | **Expl. B** (simple suffices) | K1 **and** K2: 95% CI of B − D inside (−δ, +δ), **or** D − B ≥ δ with CI > 0 |
| RQ-a | Mixed / inconclusive | anything else; report the cell map |
| RQ-b | corrected training needed | K3 at both scopes: B − E ≥ δ, CI > 0 |
| RQ-b | not needed | K3 at both scopes: CI inside (−δ, +δ) |
| RQ-c | **Expl. C** (regime-limited) | K4: CIs of both B − C and D − C inside (−δ, +δ) **and** K5: B − C ≥ δ (CI > 0) in at least one of its two cells |
| RQ-c | anticipation useful even with fast actuation and immediate feedback | K4: B − C ≥ δ with CI > 0 |
| Stage 1 | meaningful generalization gap | mean paired RMSE(E) − RMSE(B) on corrected H1-A trials > 0.05 N with CI > 0 |

If simple anticipation performs as well as learned forecasting, we will report
that **the learned-forecasting contribution is not established**. We will not
conclude that H1's mechanism is false. The four questions are kept separate:
(1) the mismatch exists; (2) a refresh improves control; (3) learning is
necessary; (4) the result generalizes.

## 11. Failure, flagging, and invalidation conditions

* **Stop / invalid:**
  * Stage-0 reproduction of H1-001 fails. The new harness at (0.06, L 0) with
    the H1-001 checkpoints must reproduce the saved H1-001 episodes exactly
    (success and compensation terms) on a re-run subset of the H1-001 H1-B
    trials.
  * Any automated test fails.
  * Analytic calibration seeds overlap training or evaluation seeds.
  * A delayed-observation test shows access to measurements newer than `t − L`.
* **Cell flag (pre-registered exclusion from pooled contrasts, still reported):**
  live-only (C) success at m = 0 below 0.95 in that cell.
* **Training failure:** if a nominal-trained predictor's validation RMSE
  exceeds persistence, it is reported as a failed training. It is not retuned.
* **No retuning:** controller gains, `N_min`, thresholds, predictor
  hyperparameters, and grid levels are fixed by this document.

## 12. Stages

* **Stage 0:**
  * unit tests;
  * physics checks (actuator response per speed; already measured, §7);
  * H1-001 reproduction;
  * an exploratory **pilot**: n = 5 trials per cell on a subset of cells,
    seeds `85,000,000 + j`. The pilot is only for validating the code; it is
    excluded from all confirmatory analysis and its outputs are not interpreted.
* **Stage 1:** predictor training (corrected at the new speeds, nominal at all
  speeds) and forecast-accuracy comparison.
* **Stage 2:** controller comparison at (0.06, L 0) (a subset of the grid).
* **Stage 3:** the full speed × latency grid and the direction block.
* **Stage 4:** analysis per §9–10.

## Amendments

(none)
