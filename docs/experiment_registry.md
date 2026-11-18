# Experiment registry

Every experiment that produced results is listed here, including failed or
invalidated ones. IDs are never reused. Status values: *planned*, *running*,
*completed*, *invalid* (kept for the record).

| ID | Title | Hypothesis | Status | Primary outcome | Report |
|---|---|---|---|---|---|
| H1-001 | Action corrections, forecast mismatch, and closed-loop relevance | H1-A, H1-B | Completed 2026-11-17 | H1-A **preliminary support**: refreshed forecast error 0.18 N vs. cached 0.49 N for corrections ≥ 0.15 (D = +0.30 N [0.27, 0.34], 5/5 seeds), limited to grip corrections the predictor is sensitive to. H1-B **preliminary support, narrow**: event-triggered refresh raised success from 83.5% to 99.8% at a 0.3 grip decrease; live-only and fixed-rate refresh did no better than cached; analytic feed-forward untested. | [H1-001.md](experiments/H1-001.md) |
| H1-002 | Is learned tactile forecasting necessary? (learned vs analytic anticipation, speed × latency) | H1-B mechanism (H1-A secondary) | Completed 2026-11-18 | **Simple analytic anticipation suffices** (registered RQ-a): refreshed learned − refreshed analytic = +0.003 [−0.002, +0.010] at the H1-001 setting and +0.003 [−0.000, +0.006] over all cells; learned-forecasting contribution not established. Anticipation benefit is not a slow-gripper artifact (+0.09 at the fastest gripper, 0 ms) and grows strongly with sensing latency (+0.54 at 200 ms). Corrected-action training: inconclusive. | [H1-002.md](experiments/H1-002.md), [protocol](experiments/H1-002_PROTOCOL.md) |

## Errata

* **H1-001 erratum (2026-11-18).** The H1-B Holm family dropped zero-discordance comparisons, and the
  pooled success CIs over-counted seed-replicated runs. Corrected: B > A is significant at m ≥ 0.25
  (not ≥ 0.2), with wider success CIs. The H1-B verdict is unchanged. Original outputs are kept;
  the corrections are in `*_erratum.*`. See the box in `experiments/H1-001.md` §7.

## Not experiments (recorded for transparency)

* **H1-002 pilot** (2026-11-18; seeds `85,000,000+`, 8 cells × 5 trials): exploratory implementation check,
  excluded from the confirmatory analysis (`results/h1/summaries/h1_002/pilot_*`).
* **Smoke runs** (`configs/experiments/h1_001_smoke.yaml`, ID `H1-001-smoke`):
  tiny-size pipeline checks. Their outputs were discarded and are not results.
  One smoke run of H1-B exposed a refresh double-counting bug that was fixed
  before the full H1-B run (protocol amendment, 2026-11-17).
* **Pre-registration development runs** (2026-11-17, not saved): exploratory
  predictor fits used while building the environment. They exposed three
  environment issues (friction combination, impulsive lift commands, a
  non-physical ~70 N press-down contact mode), which were fixed before H1-001.
  They tested no hypothesis.

## Suggested future IDs (not run)

* ~~H1-002 — feed-forward ablation × gripper slew rate × sensing latency~~ (completed; see above)
* H1-005 — learned vs. analytic anticipation under plant conditions the analytic model cannot capture
  (compliance, varied geometry, mid-episode friction change, realistic taxel signal) — recommended next
* H1-003 — improve predictor sensitivity to lift corrections, then re-test H1-A by correction direction
* H1-004 — error-triggered refreshing baseline (named in plan §4.3)
