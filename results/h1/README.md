# results/h1/

## figures/ (H1-001)

| File | Shows |
|---|---|
| `H1-001_env_validation.png` | Contact signal vs. grip command (vs. the rigid model); tangential force vs. weight for held objects |
| `H1-001_predictor_quality.png` | Held-out forecast RMSE of the four predictors, 5 seeds |
| `H1-001_fig1_error_vs_magnitude.png` | Cached and refreshed forecast error vs. effective correction magnitude (trial points plus mean ± 95% CI) |
| `H1-001_fig2_paired_difference.png` | Paired difference D = E_cached − E_refreshed by magnitude, with the wrong-correction negative control |
| `H1-001_fig3_by_type_phase_outcome.png` | D by correction type, by task phase, and by contact-changing vs. contact-preserving outcome |
| `H1-001_fig4_D_vs_realized_change.png` | D vs. how much the correction actually changed contact |
| `H1-001_fig5_error_by_horizon.png` | Error at each forecast step for corrections ≥ 0.15 |
| `H1-001_fig6_closed_loop_success.png` | H1-B: task success by forecast condition and correction magnitude, with paired differences |

## summaries/

* `H1-001_env_validation.json`: environment checks
* `H1-001_predictors.json`: per-seed predictor training and held-out metrics,
  checkpoint paths and SHA-256 checksums
* `H1-001_h1a_manifest.json`: H1-A trial seeds, raw artifact paths and checksums, provenance
* `H1-001_h1a_{mlp,ridge}_by_*.csv`: binned H1-A statistics
* `H1-001_h1b_summary.json`: H1-B per-magnitude success rates, CIs, and McNemar tests
* `H1-001_analysis.json`: every number quoted in the reports

The raw per-trial tables (`h1a_trials.csv`, `h1a_trials.npz`, `h1b_episodes.csv`)
are in `artifacts/h1/H1-001/` (git-ignored). Their checksums are in the manifests.

## H1-002 (`figures/h1_002/`, `summaries/h1_002/`)

| Figure | Shows |
|---|---|
| `fig1_success_by_controller.png` | Success of A–E and O vs. grip-decrease size at three settings |
| `fig2_learned_vs_analytic.png` | B − D (learned vs. analytic) and B − E (corrected vs. nominal training) per speed × latency |
| `fig3_forecast_training_distribution.png` | Forecast error by training distribution and speed; corrected-action forecasts on the H1-001 trials |
| `fig4_benefit_vs_speed.png` | Anticipation benefit vs. gripper speed (latency 0) |
| `fig5_benefit_vs_latency.png` | Anticipation benefit vs. sensing latency (slew 0.06) |
| `fig6_interactions_magnitude_speed_latency.png` | Success vs. magnitude for every speed × latency cell |

Summaries:
* `grid_analysis.json`: every registered contrast and decision
* `grid_cell_success.csv` and `grid_benefit_maps.csv`
* `stage0_h1_001_reproduction.json`
* `analytic_calibration.json`
* `predictor_training.json`: per-checkpoint SHA-256
* `stage1_*.csv`
* `pilot_*`: exploratory only

The raw tables are in `artifacts/h1/H1-002/` and are archived as the GitHub Release asset `h1-002-artifacts`.
