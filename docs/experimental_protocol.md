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

## H1-B — does the mismatch affect control? (registered before the H1-B run)

See the "H1-B" section below. It is added before any H1-B trial is run.

## Amendments

(none yet)
