# H1 experiments

H1: *After sufficiently large action corrections, forecasts conditioned on the
original action become less accurate than refreshed forecasts, and this mismatch
can reduce task success.* It is tested as two parts: H1-A (prediction) and
H1-B (control).

| Script | Purpose |
|---|---|
| `run_experiment.py --config <yaml> --stage {validate-env,train,h1a,h1b,all}` | Runs the experiment. Raw outputs go to `artifacts/h1/<ID>/`; summaries go to `results/h1/summaries/`. |
| `analyze_results.py --experiment-id <ID>` | Reads only saved outputs and writes figures to `results/h1/figures/` and statistics to `results/h1/summaries/<ID>_analysis.json` and `*_by_*.csv`. |

Configs: `configs/experiments/h1_001.yaml` (full) and `h1_001_smoke.yaml`
(pipeline check only; its results are not meaningful).

The analysis plans and decision criteria are registered in
`docs/experimental_protocol.md`. The reports are in `docs/experiments/`.

Where the logic lives:
* `src/contact_learning/evaluation/h1a.py`: paired cached-vs-refreshed trials
  (simulation is kept separate from scoring, so every predictor sees identical trials)
* `src/contact_learning/evaluation/h1b.py`: closed-loop conditions A–E
* `src/contact_learning/predictors/`: feature definitions, data generation, models, training
