#!/usr/bin/env python3
"""Verify the separate decoder-capacity comparison summary."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def check():
    with (ROOT / "related_evidence/capacity_primary_contrasts.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert len({r["contrast_id"] for r in rows}) == 12
    critical = float(t.ppf(1 - .05 / (2 * 12), 23))
    counts = Counter()
    maximum = 0.0
    for r in rows:
        assert int(r["degrees_of_freedom"]) == 23 and int(r["family_size"]) == 12
        assert int(r["n_replicates"]) == 24
        assert math.isclose(float(r["control_mean_test_mse"]) - float(r["candidate_mean_test_mse"]), float(r["mean_improvement"]), rel_tol=2e-10, abs_tol=2e-12)
        assert float(r["family_alpha"]) == .05
        mean = float(r["mean_improvement"])
        sd = float(r["sd_improvement"])
        se = sd / math.sqrt(24)
        lo, hi = mean - critical * se, mean + critical * se
        comparisons = ((critical, float(r["critical_t"])),
                       (se, float(r["se_improvement"])),
                       (lo, float(r["simultaneous_ci_lower"])),
                       (hi, float(r["simultaneous_ci_upper"])))
        for a, b in comparisons:
            maximum = max(maximum, abs(a - b))
            assert math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-12), (a, b)
        margin = float(r["practical_margin"])
        cls = "practical_benefit" if lo > margin else "harm" if hi < 0 else "direction_unresolved" if lo <= 0 <= hi else "positive_below_margin"
        assert cls == r["classification"], (cls, r["classification"])
        counts[cls] += 1
    assert counts == {"practical_benefit": 4, "harm": 6, "direction_unresolved": 2}
    return {"status": "PASS", "scope": "Summary arithmetic only; per-repetition inputs unavailable in this collection", "comparisons": len(rows), "classifications": dict(counts), "maximum_absolute_discrepancy": maximum}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
