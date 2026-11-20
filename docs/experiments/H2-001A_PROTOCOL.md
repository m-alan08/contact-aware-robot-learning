# H2-001A protocol — Local causal forecast-interference experiment

*Registered 2026-11-20, before any main training run or held-out evaluation.
Simulation only (MuJoCo, CPU). Branch `claude/h2-001a-local-interference`.
Config: [`configs/experiments/h2_001a.yaml`](../../configs/experiments/h2_001a.yaml).
Changes after this commit go in §12 (amendments), never by editing the text above it.*

## 1. Question and scope

**Can a stale predicted-contact representation actively interfere with recovery
from unexpected, recoverable contact disturbances in a *learned* controller, even
when informative live tactile measurements are available?**

H2 (plan §3, wording unchanged):

> Under recoverable contact disturbances, selective intervention on an outdated
> forecast improves recovery when live tactile measurements are informative.

H2-001A separates three mechanisms. A forecast-error difference alone is not
evidence for H2.

| # | Mechanism | Question |
|---|---|---|
| M1 | Forecast mismatch | Does the cached forecast disagree with measured contact after a disturbance? |
| M2 | Forecast dependence | Does the learned policy actually use its forecast input? |
| M3 | Action interference | With every other input held identical, does the stale forecast produce a worse action? |
| M4 | Recovery degradation | Does that lower the probability of recovering? |

**Context.** This is a local stand-in while ManiFeel GPU access is blocked. The
H2-001 audit, runbook, and code are preserved unchanged.

**Limits on interpretation.** The policy here is small and trained by
imitation. Its tactile input is an idealised net-force proxy. Results speak to
this controller class in this simulated task. They are not evidence about
pretrained world-action models such as TacPAC.

## 2. Task and simulator

The task is the H1 parallel-gripper scene, reused unmodified, via a subclass in
`src/contact_learning/h2a/env.py`.

**Task: grasp, lift, and hold a fragile box, for 80 steps (4 s at 20 Hz).**

* **Object.** Mass ~ U(0.04, 0.12) kg. Friction μ ~ U(0.6, 1.0).
* **Crush limit.** N_crush = r · N_static, with r ~ U(1.8, 3.0).
  N_static = m·g / (2μ) is the per-pad normal force needed to hold the box
  statically.
* **Plan.** The plan (open loop) fixes when the gripper may close and a
  two-segment lift-speed profile. Lift accelerations are limited so the
  teacher's anticipated peak stays below the crush limit.
* **Controller output.** Every controller outputs the grip command only. Lift
  follows the plan.

**Success.** All three must hold at the end of the episode:

* the object is lifted at least 50% of the planned height, with both pads in
  contact;
* it was not crushed: both pads above N_crush for ≥ 2 consecutive steps counts
  as crushed;
* it slid less than 12 mm in the grasp, measured from the reference step.

For disturbed episodes, this is **recovery success**.

**Tactile proxy.** [N_left, N_right, T_left, T_right] are MuJoCo net pad
forces with 0.02 N Gaussian noise. This is not a tactile sensor; see
`sensors/contact.py`.

* **Sensing latency L** control steps:
  * primary L = 1 (50 ms);
  * secondary L = 3 (150 ms).
* Proprioception is not delayed.

**Inputs available to deployable controllers:**

* tactile history (4 steps, delayed by L);
* current proprioception;
* the crush limit;
* a noisy weight prior (N_static × exp(N(0, 0.08)));
* the plan's current and previous lift command;
* a close-allowed flag;
* episode time.

## 3. Disturbances (simulator dynamics only)

Every disturbance is applied through the simulator; tactile values are never
edited:

| Kind | Mechanism | Seen in training? |
|---|---|---|
| `load` | Persistent downward force on the object (`xfrc_applied`), ramped in over 0.4 s | yes |
| `push` | Lateral force along the pad surface: ramp 0.4 s, hold 0.1 s, ramp out 0.4 s | yes |
| `friction` | Object friction coefficient falls gradually over 0.4 s | **no** (held-out type) |

**Severity.** s = (per-pad normal force required after onset) / N_crush.

* s < 1 is physically recoverable: holding without crushing is possible.
* s > 1 is not.
* Severity is converted to a force or friction multiplier per episode
  (`expert.disturbance_for_severity`).

**Onset** is uniform in [end of lift + 3, T − 25] steps.

**Evaluation classes:**

* recoverable: s ~ U(0.55, 0.85);
* unrecoverable: s ~ U(1.05, 1.25);
* nominal: no disturbance.

Calibration on development seeds (§10), with ramp 0.4 s and the teacher
reacting instantly:

| Severity | Teacher | Frozen grip |
|---|---|---|
| s ≤ 0.85 | 100% | 0% at s ≥ 0.75 |
| s = 1.1 | 0% | — |

## 4. Learned components

All learned components are trained and evaluated per student seed
s ∈ {0, …, 4}, and per latency.

* **Teacher (non-deployable).** Privileged: knows the mass, friction, current
  disturbance, and lift plan. It targets min(1.4 × peak required normal force
  over the next 5 steps, 0.88 N_crush). It never sees future disturbances.
* **Forecaster.**
  * MLP, 3 × 256.
  * Predicts the 4 tactile channels for the next 20 steps, from:
    * tactile and proprioceptive history at τ = t − L;
    * the grip commands already sent, followed by a held grip;
    * the plan's lift commands.
  * Trained on **nominal** episodes only, with targets from counterfactual
    held-grip rollouts: 2,000 episodes, 40 epochs.
* **Student P (forecast-conditioned).**
  * MLP, 3 × 256, sigmoid output.
  * Inputs: the deployable inputs in §2, plus a 9-step forecast slice
    (times τ+1 … τ+9), the forecast's age, and a null flag.
  * Trained by DAgger:
    * 5 iterations × 600 episodes; β = 1, 0.5, 0.25, 0.1, 0;
    * aggregated data;
    * 10 epochs per iteration, 30 for the final fit;
    * exploration noise 0.03.
  * During data collection the forecast is **chunk-cached**: refreshed every
    10 steps, the deployed default (condition A).
  * **Forecast dropout p = 0.2:** the forecast is replaced by the null input,
    so suppression (condition C) is in-distribution.
  * Training episodes: 35% nominal; otherwise `load` or `push` with
    s ~ U(0.5, 1.2).
* **Student D (live-only).** Identical to P except it has no forecast input.
  Its own DAgger run uses the same schedule, episode counts, and optimiser.
* **Checkpoint rule.** The final epoch of the final DAgger iteration is used.
  **No checkpoint is selected on any outcome**, including forecast dependence
  or harm.

## 5. Conditions

Conditions A–G use the **same trained student P**. Only the forecast pathway
differs.

| | Condition | Forecast pathway |
|---|---|---|
| A | Cached (deployed) | Refreshed at chunk boundaries (t mod 10 = 0) |
| B | Refreshed | Refreshed every step |
| C | Suppressed | Null input at every step |
| E | Fixed-rate refresh | Every 3 steps |
| F | Selective discrepancy gating (primary gate) | Chunk refresh, plus an extra refresh whenever mean abs(measured tactile at τ − cached prediction for τ) > θ |
| Fn | Selective gating, suppress variant (secondary) | Same trigger; forecast suppressed until the next chunk boundary |
| G | Matched-budget random gating | Chunk refresh, plus exactly as many extra refreshes as F made in the paired episode, at uniformly random non-boundary steps |
| O | Oracle forecast (**non-deployable diagnostic**) | True held-grip future from the simulator state, every step |
| D | Live-only student | n/a |
| R | Reactive controller (no learning) | n/a: grip normal force max(1.4 × weight prior, measured tangential / 0.4), capped at 0.88 N_crush. Parameters tuned on development seeds. |
| X | Teacher (non-deployable upper bound) | n/a |

**θ is set without looking at outcomes.** It is the 99th percentile of the
per-step discrepancy in condition A on 150 nominal **development** episodes,
per seed and latency.

**Pairing.**

* **Disturbed episodes.** The prefix up to onset runs under A. All of A–G and O
  branch from the bit-identical simulator, controller, and sensor-noise state.
* **Nominal episodes.** Every condition runs from t = 0, because the forecast's
  nominal role (anticipating lift load) occurs before any branch point.
* **D, R, and X** run their own full episodes, with the same seed and the same
  disturbance.

**Paired action-level probes.**

* *When:*
  * on the A branch, for the 10 steps after onset;
  * in nominal episodes, for steps −5 … +4 around lift-off.
* *Method:* the student's action is recomputed with the identical state, live
  touch, and context. Only the forecast slice is replaced, by each of:
  * **refresh:** the forecaster run now;
  * **null:** the trained null input;
  * **shuffle:** a cached slice from a *different* development episode;
    in-distribution but with the wrong content; the specificity control;
  * **oracle.**
* *OOD diagnostic:* the Mahalanobis distance of the student's last hidden layer
  to nominal development states. Report the fraction of probe states beyond the
  nominal 99th percentile.

## 6. Episodes and seeds

Seed ranges are disjoint from each other and from every H1 range (all H1 seeds
are below 1e8):

| Use | Seed range |
|---|---|
| Forecaster data | 200,000,000 + 1e6·s + i |
| P DAgger | 210,000,000 + 1e6·s + 1e4·iteration + i |
| D DAgger | 220,000,000 + 1e6·s + 1e4·iteration + i |
| Validation (logged only) | 230,000,000 + 1e6·s + i |
| **Development** (all tuning, θ, OOD reference, shuffle pool) | 240,000,000 + i |
| **Held-out evaluation** (never used for any choice) | 250,000,000 + i |

**Held-out set, per latency, shared by every student seed:**

* 200 nominal episodes;
* for each of `load`, `push`, and `friction`: 150 recoverable and 40
  unrecoverable episodes.

That is 770 episodes. With 5 seeds and 2 latencies, this gives 7,700 paired
episode instances.

## 7. Metrics

**Per episode and condition:**

* recovery success (disturbed) or task success (nominal);
* lifted, crushed, and slip distance;
* recovery time: from onset until the slip speed stays below 5 mm/s for 5 steps;
* forecast error of the forecast in use over the 10 steps after onset (nominal:
  the same window at the pseudo-onset);
* number of forecast refreshes and forecaster calls (the compute budget);
* gate triggers, and the delay to the first trigger;
* mean grip error to the teacher;
* squeeze ratio: mean normal force after onset / N_static, a grip-economy
  measure;
* disturbance severity.

**From the probes:**

* action change between variants;
* **interference** = |g_A − g*| − |g_B − g*|, where g* is the teacher's grip at
  that state. Positive means the stale forecast's action is worse;
* the under-grip component of interference.

**Other:**

* false-intervention rate: the fraction of nominal episodes in which F
  triggers, and the triggers per nominal step;
* nominal performance change;
* inference latency (µs per forecaster and policy call), with refreshes per
  episode as the compute measure.

## 8. Statistics

**Unit.** The episode, paired across conditions within a student seed.

**Uncertainty.** Episodes are shared across seeds, so episodes and seeds are
crossed. The bootstrap resamples episodes and seeds independently (4,000
replicates). Episode-only and seed-only intervals are also reported.

**p-values.** Two-sided bootstrap p.

**Practical threshold.** δ = 0.03 (3 points of success).

**Primary family** (Holm-adjusted). Population: L = 1, recoverable, `load` and
`push` (seen types), on recovery success. The family is P1–P4; P5 is decided by
its confidence interval only:

| | Comparison | Test |
|---|---|---|
| P1 | B − A | refreshing vs. the stale cache |
| P2 | C − A | suppressing vs. the stale cache |
| P3 | F − A | selective gating vs. the stale cache |
| P4 | F − G | selective vs. random gating at a matched budget |
| P5 | F − A on **nominal** success | non-inferiority: 95% CI lower bound > −δ |

**Secondary, reported in full but not confirmatory:**

* the same contrasts for `friction` (unseen type), for L = 3, and by severity
  tercile;
* unrecoverable episodes, as a sanity check: no condition should "rescue"
  physically impossible episodes;
* E, Fn, and O;
* D − A, R − best of {A, B, C, F}, and X;
* recovery time, squeeze ratio, and compute.

## 9. Decision criteria (registered before results)

**M1 (mismatch) is established if both hold:**

* disturbed-episode forecast error A − B > 0 with the CI excluding 0;
* the disturbed A error exceeds the nominal A error over the matched window
  (non-overlapping 95% CIs).

**M2 (dependence) is "meaningful" if both hold:**

* nominal action sensitivity, mean |g_A − g_null| at the lift-off probes,
  ≥ 0.01 grip units (≈ 0.075 N);
* **and** a behavioural effect of suppression: |C − A| ≥ δ with the CI
  excluding 0, on nominal success or on primary recovery.

Otherwise dependence is reported as "weak" (sensitivity only) or "absent".

**M3 (action interference) is established if both hold:**

* mean interference > 0 with the CI excluding 0;
* the mean post-onset action change |g_A − g_B| ≥ 0.01.

Specificity: report whether shuffle and null produce comparable changes.

**M4 (recovery degradation) is established if** P1 B − A ≥ δ, the CI lower
bound is > 0, and the Holm-adjusted p < 0.05.

**Overall H2 verdict for this setting:**

| Verdict | Condition |
|---|---|
| **Supported** | M1–M4 established, P3 F − A ≥ δ (significant), P5 holds, and P4 F > G |
| **Mechanism only** | M1–M4 established, but selective gating is not better than A or than G |
| **Contradicted** | M2 absent or weak **and** the P1 CI lies within ±δ; **or** P1 significantly ≤ −δ (refreshing hurts) |
| **Inconclusive** | anything else |

**Reactive question.** "A simple reactive controller matches" if the R − best(A,
B, C, F) CI lower bound is > −δ on primary recovery **and** on nominal success.
The same rule applies to D versus best P.

**Gates before interpreting H2:**

* *nominal adequacy:* student P under condition A has nominal success ≥ 0.90,
  per latency;
* *training sanity:* the teacher's recovery on held-out recoverable episodes is
  ≥ 0.95.

If a gate fails, the H2 contrasts are still reported, but flagged as not
interpretable.

## 10. Development history (disclosed)

All of this used development, training, or ad-hoc seeds below 250,000,000.
None used held-out episodes.

**1. Physics smoke test.** Teacher at 100% nominal; disturbances calibrated.

**2. A 0.1 s ramp made recovery impossible for any sensing controller.** The
object slid out before a 50 ms-delayed controller could respond; only the
instant teacher recovered. The ramp was changed to 0.4 s and the friction drop
made gradual.

**3. The first student fit failed nominally (63–82%).** Two causes:

* imitation error (~0.15 N) exceeded the teacher's 1.25× margin;
* the object's mass cannot be sensed before lift-off.

Changes:

* a weight prior was added to the deployable context;
* the crush range was widened from [1.6, 2.4] to [1.8, 3.0];
* the teacher margin was raised from 1.25 to 1.4, and its crush safety factor
  lowered from 0.92 to 0.88;
* the planner's acceleration limit was tied to the margin;
* severity bins were reset: recoverable [0.55, 0.85], unrecoverable
  [1.05, 1.25].

**4. Reactive-controller grid** (ratio × prior gain) on development seeds,
before the main runs.

* An earlier version without a weight prior reached 100% only by squeezing at
  0.92 N_crush.
* The registered version uses a floor of 1.4 × the prior (equal to the
  teacher's margin) and ratio 0.4. It reached 100% on development nominal and
  s = 0.7 episodes.

**5. Medium-scale pilot.** Seed 0, L = 1, 4 DAgger iterations × 500 episodes,
on 138 development episodes, with 20 per recoverable cell, so these are noisy:

| Measure | Result |
|---|---|
| Student P nominal success | 1.00, under every forecast condition except E (0.98) |
| Recoverable, seen types | A 0.63 overall (load 0.80, push 0.85); B, C, D, E between 0.52 and 0.62 |
| Friction (unseen type) | about 0.25 |
| R / X recovery | 0.95 / 1.00 |
| Probe action sensitivity | |g_A − g_B| ≈ 0.0015 and |g_A − g_null| ≈ 0.0020 grip units: **weak forecast dependence** |

**What the pilot implies.**

* The registered size is larger than the pilot, which may change the
  dependence result.
* **Neither the model nor the training was changed to increase forecast
  dependence.**
* The L = 3 secondary condition was added *a priori*, because delayed touch is
  where anticipation could plausibly matter more (H1-002). It was not chosen on
  observed outcomes: no L = 3 run existed when it was added.

## 11. Compute budget

Estimated from the smoke and pilot runs. Hardware: Apple M5, 10 cores, CPU
only.

| Stage | Time per (latency × seed) | Total for 10 combinations |
|---|---|---|
| Training | ~4–5 min | ~45 min |
| Calibration | < 1 min | |
| Evaluation (770 episodes × 11 conditions, plus probes) | ~2 min | ~20 min |
| **Total** | | **≈ 1–1.5 h** |

Every network has about 0.14–0.17 M parameters, and a call takes about
40 µs (NumPy, single sample).

No GPU, cluster, or paid infrastructure is used.

## 12. Amendments

*(None yet.)*
