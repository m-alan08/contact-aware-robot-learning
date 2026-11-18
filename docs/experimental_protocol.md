# Experimental protocol

This file fixes the procedures and decision criteria **before** results are seen.
Any later change is listed under "Amendments" with the date and reason, and the
original text is kept.

## Common conventions

* **Time indexing.** `obs[0]` is the observation after reset; `actions[t]` is
  applied at control step `t`; `obs[t+1]` is observed after it. A forecast made
  at time `t` may read `obs[:t+1]` and `actions[:t]`, and predicts tactile
  `obs[t+1 : t+H+1]` for candidate actions `actions[t : t+H]`.
* **Contact signal.** 4 channels per step: pad normal force left/right and pad
  tangential-force magnitude left/right, in newtons (net MuJoCo contact force
  between pad and object, plus N(0, 0.02 N) noise). This is a net contact
  wrench readout, **not** a model of a physical tactile array.
* **Binary contact.** A pad is "in contact" if its noise-free normal force is
  above 0.1 N. "Grasp contact" means both pads are in contact.
* **Unit of analysis.** One trial = one held-out episode seed = one correction.
  Timesteps within a trial are never treated as independent samples.
* **Splits.** Train, validation, predictor-test, and H1 trial episodes come from
  disjoint seed ranges (`src/contact_learning/utils/seeding.py`; checked by
  `tests/test_metrics_and_seeds.py` and at runtime in `predictors/train.py`).

## H1-A — forecast mismatch after action corrections (registered 2026-11-17, before the full run)

**Procedure** (`src/contact_learning/evaluation/h1a.py`): for each trial, execute
the nominal plan up to the correction time `t_c`, take the original chunk
`A_orig = plan[t_c : t_c+H]`, apply a constant offset correction of one of four
types (grip −/+, lift −/+) at a requested magnitude in
{0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} (a fraction of that action dimension's range),
clip it to the action bounds, and execute the corrected chunk from the saved state.
The cached forecast is `f(h, A_orig)` and the refreshed forecast is `f(h, A_corr)`,
computed from the **same stored history vector** `h`. Both are scored against
the same executed trajectory `Y_corr`. `t_c` is stratified: half the trials are
chosen so the nominal contact onset falls inside the forecast window, and half
are chosen uniformly.

**Primary predictor:** `mlp` (action-conditioned residual MLP).
**Primary metric:** `E` = RMSE over the H×4 forecast entries against measured
`Y_corr` (N). **Paired difference** `D = E_cached − E_refreshed`; a positive
value means refreshing helped.

**Primary analysis.**
1. For each trial, average `D` over the 5 predictor seeds, giving one value per
   trial. Bin the trials by *effective* normalized magnitude (after clipping),
   with edges `[0 (exact), 0.035, 0.075, 0.15, 0.25, 0.4, 1.0]`.
2. In each bin, report the mean `D` with a trial-bootstrap 95% CI (2000
   resamples), and a Wilcoxon signed-rank p-value with Holm correction across
   the non-zero bins.
3. Repeat per predictor seed.

**Decision criteria for H1-A "preliminary support":**
* (a) the MLP beats persistence on the held-out predictor-test episodes for every predictor seed;
* (b) for corrections with effective magnitude ≥ 0.15, mean `D` > 0, its 95%
  CI excludes 0, the Holm-adjusted p is below 0.05, and at least 4 of the 5
  seeds individually have mean `D` > 0;
* (c) zero-magnitude trials give `D` = 0 exactly (implementation check);
* (d) negative control: refreshing with a *wrong* correction of the same
  magnitude (`D_sham`) does not improve on the cached forecast on average.
  If it does, an implementation artifact is suspected.

**Not supported** if, under (b), the CI includes 0 or the mean `D` ≤ 0. The
"sufficiently large" threshold is estimated as the smallest bin whose
Holm-adjusted p is below 0.05 with mean `D` > 0.

**Secondary analyses (descriptive):** results broken down by correction type,
by phase (near the nominal contact transition, during stable contact, or no
contact), and by contact-changing vs. contact-preserving outcome. A correction is
contact-changing if the binary grasp-contact sequence under `A_corr` differs from
the one under `A_orig`. Also reported: per-horizon error, noise-free targets,
contact-state error, the ridge model as a second action-conditioned predictor,
the history-only MLP and persistence as action-blind controls (their `D` ≡ 0),
and inference latency.

**Known limitation of the design, stated in advance.** If a predictor is accurate
and responds to its action input, the cached forecast *must* become worse
whenever the correction changes the realized contact by more than the predictor's
error. H1-A is therefore partly expected by construction. The informative
quantities are: the magnitude at which the effect appears relative to predictor
error, whether a learned predictor actually achieves it on held-out data, which
correction types and phases produce it, and whether refreshing ever *hurts*
(for example, from out-of-distribution action inputs).

## H1-B — does the mismatch affect control? (registered 2026-11-17, before any H1-B trial was run)

**Scenario.** This is the same grasp-and-lift task. At a chunk boundary
`t_c ∈ {10, 20, 30}`, chosen from those at which nominal grasp contact already
exists, an upstream module issues a **persistent grip-decrease correction** of
magnitude `m ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3}`. The fraction `m` of the
grip range is subtracted from the grip command for the rest of the episode. A
contact-safety controller sits downstream of the correction. Its job is to keep
the predicted and measured pad normal force at or above `N_min = 1.7 N`. This is
the force needed to hold the heaviest, most slippery object in the training
distribution: 1.96 N weight at μ = 0.6.

**Controller (identical in every condition).** The grip command is
`g(t) = g_corrected(t) + u_f + u_l`, clipped to [0, 1].
* Live term (every step, **all** conditions): `u_l += max(0, N_min − N_meas(t)) / κ`,
  where `N_meas` is the mean measured pad normal force and `κ = 7.5 N` per unit `g`
  (the grip-to-force gain measured in environment validation).
* Forecast term (only at forecast times): `u_f += max(0, N_min − min_k N̂(t+k)) / κ`,
  where `N̂` is the mean predicted pad normal force over the H = 10 forecast steps
  for the chunk about to be executed, including the current compensation.
  Compensation is one-sided (it only increases grip).

**Conditions (only the forecast source and timing differ):**
* **A cached:** forecasts are made at chunk boundaries (`t ≡ 0 mod 10`)
  *before* the correction arrives. At `t_c` the controller keeps the forecast
  made for `A_orig`.
* **B refreshed:** as A, plus one extra forecast at `t_c` for the corrected chunk.
* **C live-only:** no forecast term (`u_f ≡ 0`).
* **D fixed-rate:** forecasts every 5 steps (`t ≡ 0 mod 5`) for the committed
  chunk, on a schedule that does not depend on when the correction occurs.
* **E oracle (non-deployable):** as B, but `N̂` is the true future normal force,
  obtained by simulating the chunk from a state snapshot.

**Held constant:** episode seeds and initial states, plans, correction time and
magnitude, the controller and its gains, the trained predictor (all 5 seeds, each
trial run with every seed), the observation history, H, and the control rate.
**Measured:** success, defined as the box raised ≥ 5 cm and both pads in contact
at T = 60; minimum measured normal force after `t_c`; slip, the box displacement
relative to the hand; number of forecasts; per-forecast latency; and the action
difference between conditions, `Σ_t |g_B(t) − g_A(t)|`.

**Primary comparison:** success of B vs. A, paired by trial. Within each
magnitude and for each predictor seed separately, this is tested with an exact
McNemar test. The pooled effect is the success-rate difference averaged over
seeds, with a trial-cluster bootstrap 95% CI (trials are resampled; seeds stay
together). The live-feedback (C) and oracle (E) conditions do not use the
learned predictor, so they run once per trial. The live term and forecast term
are active from `t_c` onward (the safety layer monitors the corrected stream);
before `t_c` all conditions execute the same nominal plan. Secondary
comparisons: B vs. C and D vs. A, with E as the upper bound.
**"H1-B preliminary support"** requires: success(B) > success(A) with McNemar
p < 0.05 (Holm across the non-zero magnitudes) in at least one magnitude for
at least 4 of the 5 predictor seeds, with a pooled bootstrap CI that excludes 0;
B ≥ C within CI (refreshing is not merely redundant with live feedback, or if it
is, that is reported as the result); and A = B exactly at `m = 0`. If A and B
do not differ, H1-B is "not supported under tested conditions". If B > A but
C ≥ B, the conclusion is that the mismatch is harmful only relative to a stale
plan, and live feedback alone suffices.

**Known caveat, stated in advance.** The scenario was chosen because it is one
where forecast staleness *could* matter: a slow gripper (slew limit) and a lift
in progress. It is not a random sample of tasks. Live-feedback gains were set from
the measured plant gain κ, not tuned on H1-B outcomes.

## Amendments

* **2026-11-17, H1-B, before the full H1-B run (smoke test only).** A smoke run
  exposed a bug. At `t_c`, the refreshed and oracle conditions *added* the
  deficit from the superseded pre-correction forecast on top of the refreshed
  forecast's deficit. They now *replace* it, as the design text above specifies.
  The superseded forecast still counts toward the forecast budget. No full
  H1-B results existed when this was fixed. Smoke outputs were discarded.
  A regression test was added (`tests/test_h1b_validity.py`).
