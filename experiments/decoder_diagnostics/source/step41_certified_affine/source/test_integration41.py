"""Execute all Step41 producers and independent gates on isolated artificial data.

This harness is run before the scientific protocol is frozen. It uses both task
identities, all 18 affine configurations and 44 baselines, all development/final
stages, chronological test access, and the actual deployment timing procedure.
Only the evidence root and synthetic calendar/values change. Independent audit
functions issue every gate; the harness does not manufacture PASS gates.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[_name] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True

import check_data41 as fixtures
import common41 as c
import data41 as d
import prepare_data41 as prep
import protocol41 as plan
import representations35 as rep
import run41 as run
import audit41 as audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_API = ("preflight", "fit_gate", "validation_gate", "selection_gate",
             "pretest", "postrun", "benchmark_gate", "aggregate")


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_source_snapshot():
    paths = list((PROJECT_ROOT / "source").glob("*.py"))
    paths += [p for p in (PROJECT_ROOT / "inputs/native").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += [PROJECT_ROOT / "inputs/inherited41.json"]
    return {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in sorted(paths)}


def public_input_snapshot():
    """Hashes opaque bytes; no measurement field is decoded."""
    return {str(p.relative_to(PROJECT_ROOT)): sha(p)
            for p in sorted((PROJECT_ROOT / "inputs/data41").rglob("*")) if p.is_file()}


def freeze_fixture(root, protocol):
    c.save(root / "protocol41.json", protocol)
    paths = [root / "protocol41.json"]
    for folder in ("source", "inputs", "analysis"):
        paths += [p for p in (root / folder).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    bound = {str(p.relative_to(root)): sha(p) for p in sorted(set(paths))}
    require(d.required_data_files() <= set(bound), "Artificial seal omits a data dependency")
    seal = {"schema": "step41-protocol-source-seal-v1", "sealed_utc": c.utc(),
            "protocol_sha256": sha(root / "protocol41.json"), "bound_files": bound,
            "new_measurement_tokens_decoded": 0, "single_primary_pass": True,
            "artificial_only": True, "public_measurement_tokens_decoded": 0}
    ref = c.save(root / "protocol_seal41.json", seal)
    d.event("protocol_frozen", protocol_seal_sha256=ref["sha256"],
            bound_file_count=len(bound), artificial_only=True)
    return seal


def assert_artificial_roots(root):
    require(root.is_relative_to((PROJECT_ROOT / "tmp").resolve()),
            "Artificial root must be inside the project temporary directory")
    require(all(Path(p).resolve() == root for p in (c.ROOT, d.ROOT, prep.ROOT, run.ROOT, audit.ROOT, rep.ROOT)),
            "A producer or auditor escaped the artificial evidence root")
    require(run.OUT == root / "output", "Runner output escaped the artificial evidence root")
    require(audit.ARTIFICIAL_FIXTURE is True, "Independent audit artificial mode not explicit")


def main():
    require(not (PROJECT_ROOT / "protocol_seal41.json").exists(),
            "Artificial integration must finish before the scientific protocol freeze")
    require(all(callable(getattr(audit, name, None)) for name in AUDIT_API),
            "Independent audit interface is not complete")
    require(hasattr(audit, "ARTIFICIAL_FIXTURE") and audit.ARTIFICIAL_FIXTURE is False,
            "Independent auditor lacks its safe default artificial-mode guard")
    require(rep._GEOMETRY is None and rep._GEOMETRY_MODULE not in sys.modules,
            "Run integration in a fresh process before native runtime initialization")

    started = time.perf_counter()
    source_hashes = scientific_source_snapshot()
    public_hashes = public_input_snapshot()
    old = {"c_root": c.ROOT, "run_root": run.ROOT, "run_out": run.OUT,
           "audit_root": audit.ROOT, "audit_artificial": audit.ARTIFICIAL_FIXTURE,
           "audit_stages": deepcopy(audit.STAGE_DATES), "audit_counts": deepcopy(audit.EXPECTED_N),
           "audit_test_dates": audit.test_dates, "rep_root": rep.ROOT,
           "geometry_path": rep.GEOMETRY_PATH, "geometry": rep._GEOMETRY,
           "sys_path": list(sys.path)}
    fixture = None
    root = None
    phases = []
    gate_receipts = []
    current_phase = "setup"
    native_module_names = []
    fixture_decode_count = 0

    def execute(label, function, *args):
        nonlocal current_phase
        current_phase = label
        assert_artificial_roots(root)
        t0 = time.perf_counter()
        result = function(*args)
        phases.append({"phase": label, "status": "PASS",
                       "wall_seconds": time.perf_counter() - t0})
        print(json.dumps({"artificial_phase": label, "status": "PASS"}), flush=True)
        return result

    def gate(label, function, *args):
        result = execute(label, function, *args)
        require(isinstance(result, dict) and result.get("status") == "PASS",
                "Independent gate did not return PASS: " + label)
        gate_receipts.append({"phase": label, "receipt": result})
        return result

    try:
        fixture = fixtures.IntegrationFixture()
        root = fixture.root.resolve()
        print(json.dumps({"artificial_root": str(root), "public_measurement_accesses": 0}), flush=True)

        # Copy source/native dependencies only. Public data directories are never
        # copied into this root; its raw inputs are deterministic synthetic bytes.
        for relative, wanted in source_hashes.items():
            source = PROJECT_ROOT / relative
            dest = root / relative
            require(sha(source) == wanted, "Source changed while preparing fixture")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                require(sha(dest) == wanted, "Fixture source snapshot differs")
            else:
                shutil.copyfile(source, dest)
        for filename in ("data_artificial_checks41.json", "certificate_artificial_checks41.json",
                         "certificate_design41.json", "baseline_grid41.json", "hardware41.json",
                         "interface41.txt"):
            source = PROJECT_ROOT / "analysis" / filename
            if source.is_file():
                dest = root / "analysis" / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)

        c.ROOT = root
        run.ROOT = root
        run.OUT = root / "output"
        audit.ROOT = root
        audit.ARTIFICIAL_FIXTURE = True
        audit.STAGE_DATES = {task: fixtures.integration_stages(task) for task in d.task_ids()}
        audit.EXPECTED_N = deepcopy(fixtures.INTEGRATION_N)
        audit.test_dates = fixtures.integration_test_dates
        audit.cached_archive_member.cache_clear()
        rep.ROOT = root
        rep.GEOMETRY_PATH = root / "inputs/native/step6_geometry_regimes/src/geometry.py"
        rep._GEOMETRY = None
        assert_artificial_roots(root)

        tracked_parse = d._parse_numeric_token
        def synthetic_parse_only(task, token, stamp):
            nonlocal fixture_decode_count
            assert_artificial_roots(root)
            source = d.read_json(root / f"inputs/data41/{task}/public_source41.json")
            require(source.get("artificial_only") is True and
                    all(a["source_url"].startswith("artificial://fixture/") for a in source["archives"]),
                    "Numeric parse attempted outside the artificial source contract")
            fixture_decode_count += 1
            return tracked_parse(task, token, stamp)
        d._parse_numeric_token = synthetic_parse_only

        marker = {"schema": "step41-isolated-artificial-integration-v1", "artificial_only": True,
                  "production_project_root": str(PROJECT_ROOT), "fixture_root": str(root),
                  "raw_source_formula": "(2000+(hour*17)%271+(hour*hour)%19+hour//24)/10; hour from each task year January1",
                  "training_sizes": [64, 72, 80, 88], "validation_sizes": [8, 8, 8], "test_size_per_task": 2,
                  "source_snapshot_sha256": source_hashes, "public_measurement_values_decoded": 0,
                  "independent_gates_required": True, "harness_manufactured_pass_gates": False}
        c.save(root / "analysis/artificial_fixture41.json", marker)
        protocol = plan.protocol()
        protocol["artificial_only"] = True
        protocol["artificial_fixture"] = {"marker_path": "analysis/artificial_fixture41.json",
                                           "marker_sha256": sha(root / "analysis/artificial_fixture41.json")}
        require(len(protocol["heads"]["configurations"]) == 18 and len(protocol["baselines"]["configurations"]) == 44 and
                len(protocol["evaluation"]["prediction_ids"]) == 62 and len(protocol["tasks"]) == 2,
                "Artificial integration changed the production candidate panel")
        freeze_fixture(root, protocol)
        gate("audit_preflight", audit.preflight)

        selected_before_test = {}
        for task in protocol["tasks"]:
            for stage in ("f1", "f2", "f3"):
                execute(task + ":fit:" + stage, run.fit, protocol, task, stage)
                gate(task + ":audit_fit:" + stage, audit.fit_gate, task, stage)
                execute(task + ":validate:" + stage, run.validate, protocol, task, stage)
                gate(task + ":audit_validation:" + stage, audit.validation_gate, task, stage)
            execute(task + ":select", run.select, protocol, task)
            gate(task + ":audit_selection", audit.selection_gate, task)
            selection = c.read(root / f"output/{task}/selection41.json")
            selected_before_test[task] = {label: r["candidate_id"] for label, r in selection["selections"].items()}
            execute(task + ":fit:final", run.fit, protocol, task, "final")
            gate(task + ":audit_fit:final", audit.fit_gate, task, "final")
            gate(task + ":audit_pretest", audit.pretest, task)
            execute(task + ":test", run.test, protocol, task)
            gate(task + ":audit_postrun", audit.postrun, task)
            execute(task + ":benchmark", run.benchmark, protocol, task)
            gate(task + ":audit_benchmark", audit.benchmark_gate, task)
        execute("aggregate", run.aggregate, protocol)
        gate("audit_aggregate", audit.aggregate)

        for task in protocol["tasks"]:
            result = c.read(root / f"output/{task}/primary_result41.json")
            require(result["n"] == 2 and len(result["candidate_rows"]) == 62,
                    "Artificial final result omits a query or configured predictor")
            require({label: r["candidate_id"] for label, r in result["selections"].items()} == selected_before_test[task],
                    "Artificial final assessment changed a development-selected identity")
            require(d._load_test_state(task)["status"] == "COMPLETE",
                    "Artificial chronological data state did not complete")
            for row in result["queries"]:
                bundle = c.read(root / row["prediction_path"])
                require(set(bundle["predictions"]) == set(protocol["evaluation"]["prediction_ids"]),
                        "Artificial test prediction panel is incomplete")
            benchmark = c.read(root / f"output/{task}/benchmark41.json")
            require(benchmark["n"] == 2 and benchmark["candidate_id"] == selected_before_test[task]["affine_rolling"],
                    "Benchmark did not exercise the actual selected final affine predictor")

        # A real start-marker collision proves the producer cannot replay a fit
        # into the same evidence tree. It occurs before source access or fitting.
        try:
            run.fit(protocol, protocol["tasks"][0], "f1")
        except FileExistsError:
            replay_rejected = True
        else:
            replay_rejected = False
        require(replay_rejected, "Producer accepted a duplicate artificial fitting stage")
        require(scientific_source_snapshot() == source_hashes, "Scientific source changed during artificial integration")
        require(public_input_snapshot() == public_hashes, "Public source/input bytes changed during artificial integration")
        require(not (PROJECT_ROOT / "protocol_seal41.json").exists(),
                "Scientific protocol was frozen before artificial integration finished")

        files = [{"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
        manifest = {"schema": "step41-artificial-fixture-manifest-v1", "artificial_only": True,
                    "public_measurements_decoded": 0, "files": files}
        c.save(root / "fixture_manifest41.json", manifest)
        result = {"schema": "step41-full-artificial-integration-check-v1", "status": "PASS",
                  "artificial_only": True, "public_measurement_values_decoded": 0,
                  "harness_manufactured_pass_gates": False,
                  "fixture_root": str(root.relative_to(PROJECT_ROOT)),
                  "fixture_manifest_sha256": sha(root / "fixture_manifest41.json"),
                  "fixture_files": len(files), "source_sha256": source_hashes,
                  "public_input_hashes_unchanged": public_hashes,
                  "independent_gate_count": len(gate_receipts),
                  "independent_gates": gate_receipts, "phases": phases,
                  "tasks": list(protocol["tasks"]), "affine_fits": 18 * 4 * 2,
                  "baseline_fits": 44 * 4 * 2,
                  "validation_predictions": 62 * 8 * 3 * 2,
                  "test_predictions": 62 * 2 * 2,
                  "deployment_benchmarks": 2,
                  "synthetic_measurement_token_decodes": fixture_decode_count,
                  "producer_fit_replay_rejected": replay_rejected,
                  "all_sources_and_public_inputs_unchanged": True,
                  "wall_seconds": time.perf_counter() - started}
        receipt = PROJECT_ROOT / "analysis/integration_artificial_checks41.json"
        receipt.write_bytes(c.canonical_bytes(result))
        print(json.dumps({"status": "PASS", "fixture_root": result["fixture_root"],
                          "independent_gates": len(gate_receipts), "fixture_files": len(files),
                          "seconds": result["wall_seconds"], "public_measurement_values_decoded": 0}), flush=True)
        return result
    except Exception as exc:
        if root is not None:
            failure = {"schema": "step41-artificial-integration-failure-v1", "status": "FAIL",
                       "artificial_only": True, "phase": current_phase,
                       "exception_type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc(), "public_measurement_values_decoded": 0}
            (root / "integration_failure41.json").write_bytes(c.canonical_bytes(failure))
            print(json.dumps({"status": "FAIL", "artificial_root": str(root),
                              "phase": current_phase, "message": str(exc)}), flush=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if rep._GEOMETRY is not None:
            native_module_names = list(getattr(rep._GEOMETRY, "_NATIVE_FILES", {}))
        if fixture is not None:
            fixture.close()
        c.ROOT = old["c_root"]
        run.ROOT, run.OUT = old["run_root"], old["run_out"]
        audit.ROOT, audit.ARTIFICIAL_FIXTURE = old["audit_root"], old["audit_artificial"]
        audit.STAGE_DATES, audit.EXPECTED_N = old["audit_stages"], old["audit_counts"]
        audit.test_dates = old["audit_test_dates"]
        audit.cached_archive_member.cache_clear()
        rep.ROOT, rep.GEOMETRY_PATH, rep._GEOMETRY = old["rep_root"], old["geometry_path"], old["geometry"]
        for name in native_module_names + [rep._GEOMETRY_MODULE]:
            module = sys.modules.get(name)
            filename = getattr(module, "__file__", None) if module is not None else None
            if root is not None and filename and Path(filename).resolve().is_relative_to(root):
                sys.modules.pop(name, None)
        sys.path[:] = old["sys_path"]


if __name__ == "__main__":
    main()
