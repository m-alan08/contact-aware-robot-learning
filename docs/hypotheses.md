# Hypotheses

Original wording from the research plan (§3) is preserved verbatim. Status is
updated only from completed, registered experiments; see
[`experiment_registry.md`](experiment_registry.md).

| ID | Original wording | Status |
|----|------------------|--------|
| H1 | After sufficiently large action corrections, forecasts conditioned on the original action become less accurate than refreshed forecasts, and this mismatch can reduce task success. | See split below |
| H1-A | *(prediction component)* Correcting an action makes the original action-conditioned tactile forecast less accurate than a refreshed forecast. | **Preliminary support** (simulation, H1-001); limited to correction directions the predictor is sensitive to |
| H1-B | *(control component)* This forecast mismatch can reduce task success. | **Preliminary support, narrow** (simulation, H1-001); one scenario with a slow gripper; the stale forecast was inert rather than misleading |
| H2 | Under recoverable contact disturbances, selective intervention on an outdated forecast improves recovery when live tactile measurements are informative. | Not tested (H2-001 blocked at baseline gate, 2026-11-18) |
| H3 | A selective update rule can retain the benefits of prediction in nominal conditions while improving disturbed-condition performance at a comparable computation and latency budget. | Not tested |

The plan states: *"These hypotheses are conditional. We should drop or revise
them if the corresponding mismatch is absent, does not affect control, or can be
explained by sensor noise or execution delay."*

Proposed revisions, if any, are recorded in `H1_PRELIMINARY_RESULTS.md` §7 and
are clearly labelled as proposals; they do not replace the text above.
