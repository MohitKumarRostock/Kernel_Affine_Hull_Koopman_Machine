"""Focused adapter fixture: ordinary-path bytes and captured-design recovery."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import crossfit_model as model
import numerical_solver as adapter


def main():
    output = ROOT / "readiness/numerical_solver_fixture.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite fixture receipt")
    source = ROOT.parent / "crossfit_residual/readiness/svd_failure_duffing_rep01/failure_00"
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source.iterdir() if path.is_file()}
    runtime = model.fd.load_runtime()
    original = runtime._ridge_path
    numpy_svd = np.linalg.svd
    alphas = [.0001, .001, .01, .1, 1.]
    rng = np.random.default_rng(3195807)
    fixtures = [rng.normal(size=(75, 15)), rng.normal(size=(80, 8))]
    fixtures[1] = np.column_stack((fixtures[1], fixtures[1][:, :3], np.ones(80)))
    targets = [rng.normal(size=(len(x), 3)) for x in fixtures]
    with threadpool_limits(limits=1):
        expected = [pickle.dumps(original(x, y, alphas), protocol=5)
                    for x, y in zip(fixtures, targets)]
        provenance = adapter.install(runtime)
        assert adapter.install(runtime) == provenance
        adapter.clear_events()
        observed = [pickle.dumps(runtime._ridge_path(x, y, alphas), protocol=5)
                    for x, y in zip(fixtures, targets)]
        assert expected == observed
        assert adapter.events() == []
        assert np.linalg.svd is numpy_svd
        x, y = np.load(source / "x.npy"), np.load(source / "y.npy")
        centered, response = x - x.mean(0), y - y.mean(0)
        assert np.array_equal(centered, np.load(source / "centered.npy"))
        assert np.array_equal(response, np.load(source / "response.npy"))
        try:
            original(x, y, alphas)
        except np.linalg.LinAlgError:
            original_failed = True
        else:
            raise AssertionError("Captured failing design no longer fails on original solver")

        def _select_ridge(prefix):
            return runtime._ridge_path(x, y, alphas)

        heads = _select_ridge("fixture/fold0/standard__cap12")
        records = adapter.events()
        assert len(records) == 1 and records[0]["status"] == "recovered" and records[0]["recovered"]
        assert records[0]["context"]["ridge_prefix"] == "fixture/fold0/standard__cap12"
        assert records[0]["centered_array_sha256"] == model.fd.array_hash(centered)
        assert records[0]["response_array_sha256"] == model.fd.array_hash(response)
        assert records[0]["alphas"] == alphas
        assert len(heads) == len(alphas) and all(head["coef"].shape == (182, 3) for head in heads)
        assert list(heads[0]) == ["mean", "ymean", "coef", "alpha_by_horizon", "response_rms",
                                  "feature_rank", "feature_width", "training_rows"]
        assert np.linalg.svd is numpy_svd
        records[0]["status"] = "tampered detached copy"
        assert adapter.events()[0]["status"] == "recovered"
        after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected}
        assert protected == after
        receipt = {
            "status": "passed", "schema": "ridge-numerical-adapter-fixture-v1",
            "successful_path_fixture_count": len(fixtures), "successful_path_pickle_bytes_identical": True,
            "successful_path_head_sha256": [hashlib.sha256(value).hexdigest() for value in observed],
            "known_original_failure_reproduced": original_failed,
            "saved_design_and_response_exactly_recentered": True,
            "recovered_head_count": len(heads), "numpy_svd_identity_unchanged": True,
            "events_return_detached_copies": True, "install_idempotent": True,
            "input_evidence_unchanged": protected == after,
            "no_B_C_D_forecast_scores_computed": True, "provenance": provenance,
            "fallback_events": adapter.events(), "input_evidence_sha256": protected,
        }
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": receipt["status"], "receipt": str(output),
                          "successful_path_bytes_identical": True,
                          "fallback_status": adapter.events()[0]["status"],
                          "normal_equation_residual_max": max(row["relative_normal_equation_residual"]
                                                               for row in adapter.events()[0]["ridge_checks"])}))


if __name__ == "__main__":
    main()
