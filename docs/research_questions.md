# Research questions

Source: *Contact-Aware Robot Learning: Literature and Research Plan*
(P. Samineni, A. Kothari, Alan), [`docs/literature/literature_review.pdf`](literature/literature_review.pdf),
SHA-256 `2b786fbf46f51b859d7b12fee0cc28c997a9aa200746efaea7e7d0cea9bfd9ec`.

## Overarching question

> When should a robot retain, revise, or disregard its predicted contact during execution?

## Prioritised questions (from §2 of the plan)

**RQ1 — Updating predictions after action corrections (§2.1).**
When an action chunk is modified during execution, does refreshing its
predicted tactile outcome improve control compared with retaining the original
prediction? This direction is worth pursuing only if the inconsistency causes a
control failure *and* refreshing is feasible within the control budget.
→ tested by **H1** (this repository's current scope).

**RQ2 — Resolving disagreement between prediction and measurement (§2.2).**
Under unexpected contact changes, when do predicted tactile states override
informative live feedback, and can selectively reducing their influence improve
recovery? → **H2** (not started).

## Alternatives and supporting questions (§5, not in current scope)

* §5.1 Do tactile predictors generalise from demonstration actions to policy-generated actions?
  (The plan's strongest alternative if the execution-time mismatch is not supported.)
* §5.2 Preserving pretrained capabilities during tactile adaptation.
* §5.3 Predicting contact events and their timing.

## Comparisons the plan requires (§1.3, §4.3)

* Retaining the original forecast vs. live feedback without predictions vs.
  fixed-rate refreshing vs. error-triggered refreshing.
* Account for extra inference cost; separate gains from forecasting from gains
  caused by faster access to measured touch (Reactive Diffusion Policy, TouchWorld,
  T-Rex, TacPAC are the relevant prior work cited in the plan).
