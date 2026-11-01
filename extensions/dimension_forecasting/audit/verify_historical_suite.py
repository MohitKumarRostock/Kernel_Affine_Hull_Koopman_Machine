#!/usr/bin/env python3
"""Separate-process comparison to unchanged historical ten-method implementation.

Only the prespecified manufactured schema fixture is used. No production draws
or outcomes are loaded and no current corrected fitting module is imported.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from verify_task_artifacts import Audit, arrays, read, sha, FULL, GEOMETRIC


def main(root):
    audit = Audit()
    fixture = root / "verification/schema_fixture_final"
    f = read(fixture / "results/fit_summary.json")
    audit.require(f["task"].get("manufactured_only") is True, "historical gate only uses manufactured inputs")
    a, x = arrays(fixture / "abc.npz"), arrays(fixture / "queries.npz")["D_X"]
    current_c = arrays(fixture / "results/calibration_predictions.npz")
    current_d = arrays(fixture / "predictions/predictions.npz")
    reference = root / "runtime/step1/forecasting_corrected/reference_current"
    sys.path.insert(0, str(reference / "code"))
    from geometry_adapter import _implementation
    source, provenance = _implementation()
    suite = source.fit_suite(*(a[k] for k in ("A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y")),
                             seed=f["task"]["fit_seed"])
    for split, query, current in (("C", a["C_X"], current_c), ("D", x, current_d)):
        result = suite.predict(query)
        for method in FULL:
            audit.equal(result["predictions"][method], current[f"context_historical2__calibrated__{method}"],
                        "historical ten-method forecast parity", [split, method])
        for method in GEOMETRIC:
            audit.equal(result["uncalibrated_predictions"][method], current[f"historical2__gamma1__{method}"],
                        "historical uncalibrated geometry parity", [split, method])
    audit.require(suite.selected_policy == f["contexts"]["historical2"]["selected_policy"],
                  "historical C-selected policy identity")
    for method in FULL[:-1]:
        expected = f["shared" if method in ("ridge", "raw_rbf", "raw_hgb", "persistence")
                     else "candidates"]
        if method not in ("ridge", "raw_rbf", "raw_hgb", "persistence"):
            expected = expected["historical2"]
        audit.require(suite.diagnostics["B"][method] == expected["fit_diagnostics"]["B"][method],
                      "historical B grid and selected hyperparameter identity", method)
        audit.require(suite.diagnostics["C"][method] == expected["calibration"]["C"][method],
                      "historical C calibration identity", method)
    result = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "manufactured_only": True, "counts": dict(audit.counts),
              "source_provenance": provenance, "audit_source_sha256": sha(__file__),
              "fixture_ABC_sha256": sha(fixture / "abc.npz"),
              "fixture_query_sha256": sha(fixture / "queries.npz"),
              "scope": "Bitwise parity with unchanged historical source in a clean process: all ten calibrated procedures on C and D, five gamma1 geometry procedures, B grids and C calibration."}
    (root / "audit/historical_suite_checks.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root.resolve())
