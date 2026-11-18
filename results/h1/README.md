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
