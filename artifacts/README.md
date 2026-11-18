# artifacts/

Large or raw outputs (predictor checkpoints, raw trial arrays) are written here
by `experiments/*/run_experiment.py` and are **not version-controlled**.
Their paths, sizes, and SHA-256 checksums are recorded in
`results/h1/summaries/*_manifest.json`, `*_predictors.json`, and the experiment
reports in `docs/experiments/`. To regenerate them, rerun the reproduction
commands listed in the corresponding report.
