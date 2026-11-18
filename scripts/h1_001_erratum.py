"""H1-001 erratum (2026-11-18): recompute H1-B statistics with two corrections,
WITHOUT overwriting the original H1-001 outputs.

1. Exact McNemar with zero discordant pairs now returns p = 1.0 instead of NaN,
   so all six registered non-zero magnitudes stay in each Holm family.
2. Pooled success CIs use the trial as the unit (seed-averaged, trial bootstrap)
   instead of a Wilson interval over seed-replicated runs.

Inputs : artifacts/h1/H1-001/h1b_episodes.csv (SHA-256 recorded in the original summary)
Outputs: results/h1/summaries/H1-001_h1b_summary_erratum.json
         results/h1/figures/H1-001_fig6_closed_loop_success_erratum.png
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "h1"))
from analyze_results import analyze_h1b  # noqa: E402

from contact_learning.evaluation.h1b import summarize_h1b  # noqa: E402
from contact_learning.utils.config import REPO_ROOT  # noqa: E402
from contact_learning.utils.provenance import provenance, sha256_file  # noqa: E402

SUMM = REPO_ROOT / "results" / "h1" / "summaries"
orig = json.load(open(SUMM / "H1-001_h1b_summary.json"))
csv = REPO_ROOT / orig["episodes_csv"]
assert sha256_file(csv) == orig["episodes_csv_sha256"], "raw H1-B table does not match the recorded checksum"
df = pd.read_csv(csv)
out = summarize_h1b(df, orig["config"])
out.update({"erratum_of": "H1-001_h1b_summary.json", "erratum_date": "2026-11-18",
            "corrections": ["McNemar p=1.0 (not NaN) for zero discordance; Holm family = all 6 non-zero magnitudes",
                            "pooled success CIs: trial-bootstrap of seed-averaged success (n = trials), replacing Wilson over all runs"],
            "source": "PR #1 automated review findings, verified against saved data",
            "episodes_csv": orig["episodes_csv"], "episodes_csv_sha256": orig["episodes_csv_sha256"],
            "config": orig["config"], "provenance": provenance()})
with open(SUMM / "H1-001_h1b_summary_erratum.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
analyze_h1b("H1-001", summary_suffix="_erratum", fig_suffix="_erratum")
for r in out["by_magnitude"]:
    print(r["magnitude"], [round(q["p_holm"], 5) for q in r["per_seed_B_vs_A"]] if r["magnitude"] > 0 else "-",
          {c: [round(x, 3) for x in r[f"success_{c}"]] for c in ("cached", "refreshed", "live_only")})
