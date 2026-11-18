# H2 — selective intervention on outdated forecasts under disturbances

**Status: not started.** H2 is out of scope until H1 is resolved
(see `docs/H1_PRELIMINARY_RESULTS.md` §9).

Reusable pieces already in place: `GripperLiftEnv` (it can be extended with
external disturbances such as an applied object force or a friction change
mid-episode), the predictor family and training pipeline, the paired
snapshot/restore trial machinery (`evaluation/h1a.py`), the closed-loop
condition harness (`evaluation/h1b.py`), and the statistics helpers.
