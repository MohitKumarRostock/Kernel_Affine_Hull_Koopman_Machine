"""Manufactured-only tests of Step24 data boundaries and feature/target order."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
import data24 as D


def rejects(fn):
    try:
        fn()
    except (ValueError, PermissionError, RuntimeError, FileNotFoundError):
        return True
    raise AssertionError("A deliberately invalid request was accepted")


def manufactured_spec():
    spec = json.loads((D.STEP23 / "data/appliances_protocol23.json").read_text())
    spec["source"]["advertised_rows"] = 576
    spec["partitions"]["raw_segments_zero_based_half_open"] = {
        "outer_train": [0, 192], "validation": [192, 384], "test": [384, 576]}
    return spec


def manufactured_lines(spec, *, poison_roles=(), poison_value="SEALED_NUMERIC_VALUE"):
    """The entire CSV is manufactured; even excluded columns are nonnumeric."""
    header = spec["source"]["expected_csv_columns"]
    numeric_names = [spec["forecast"]["target_column"]] + spec["features"]["current_local_channels"]
    numeric_set = set(numeric_names)
    start = datetime(2000, 1, 3)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    for row in range(spec["source"]["advertised_rows"]):
        role = "outer_train" if row < 192 else "validation" if row < 384 else "test"
        values = [(start + timedelta(minutes=10 * row)).strftime("%Y-%m-%d %H:%M:%S")]
        for index, name in enumerate(header[1:], 1):
            if name not in numeric_set:
                value = "UNUSED_COLUMN_NOT_A_NUMBER"
            elif role in poison_roles:
                value = poison_value
            elif name == "Appliances":
                value = str(10 * row)
            else:
                value = str(1000 * row + index)
            values.append(value)
        writer.writerow(values)
    return output.getvalue().splitlines(keepends=True)


class ScriptedIntegers:
    def __init__(self, draws):
        self.draws = list(draws)
        self.position = 0
    def integers(self, low, high):
        if self.position >= len(self.draws):
            raise AssertionError("A manufactured trajectory requested an extra draw")
        expected_high, value = self.draws[self.position]
        self.position += 1
        assert low == 0 and high == expected_high and 0 <= value < high
        return value


def scripted_hmm_draws(symbols):
    # All hidden states stay in state0; every emission symbol remains possible.
    starts = (0, 24, 30, 31)
    draws = [(4, 0)]
    for i, symbol in enumerate(symbols):
        if i:
            draws.append((512, 0))
        draws.append((32, starts[symbol]))
    return draws


def main():
    out = PROJECT / "tests/data_preflight24.json"
    if out.exists():
        raise FileExistsError("The manufactured data receipt is write-once")
    checks, rejected = [], []
    with tempfile.TemporaryDirectory(prefix="step24_data_manufactured_") as tmp:
        tmp = Path(tmp)
        binding = D.source_bindings()
        implementation = {"schema": "step24-implementation-seal-v1", "passed": True,
                          "protocol_sha256": D.PROTOCOL_SHA256, "archive_sha256": D.ARCHIVE_SHA256,
                          "source_sha256": binding}
        imp = tmp / "implementation_seal.json"
        fit = tmp / "fit_seal.json"
        imp.write_text(json.dumps(implementation))
        assert D._check_gate("development", "hmm_0", imp)["protocol_sha256"] == D.PROTOCOL_SHA256
        for task in ("hmm_0", "appliances"):
            assert rejects(lambda task=task: D.load_development(task, gate_path=tmp / "absent.json"))
            assert rejects(lambda task=task: D.load_test(task, gate_path=tmp / "absent.json"))
        rejected.extend(["missing development seal for both domains", "missing test seal for both domains"])
        bad = deepcopy(implementation); bad["passed"] = False
        bad_path = tmp / "bad_implementation.json"; bad_path.write_text(json.dumps(bad))
        assert rejects(lambda: D.load_development("appliances", gate_path=bad_path))
        rejected.append("unpassed implementation seal")
        for key, value in [("protocol_sha256", "0" * 64), ("archive_sha256", "0" * 64)]:
            bad = deepcopy(implementation); bad[key] = value; bad_path.write_text(json.dumps(bad))
            assert rejects(lambda: D.load_development("hmm_0", gate_path=bad_path))
            rejected.append("wrong " + key)
        bad = deepcopy(implementation)
        bad["source_sha256"]["step24_cross_task/src/data24.py"] = "0" * 64
        bad_path.write_text(json.dumps(bad))
        assert rejects(lambda: D.load_development("appliances", gate_path=bad_path))
        rejected.append("changed loader source binding")
        complete_fit = {"schema": "step24-fit-seal-v1", "passed": True,
                        "protocol_sha256": D.PROTOCOL_SHA256, "archive_sha256": D.ARCHIVE_SHA256,
                        "source_sha256": binding, "all_task_ids": list(D.TASK_IDS),
                        "implementation_seal_sha256": D.file_sha256(imp)}
        fit.write_text(json.dumps(complete_fit))
        assert D._check_gate("test", "appliances", fit)["stage"] == "test"
        for change, value in [("all_task_ids", list(D.TASK_IDS)[:-1]),
                              ("implementation_seal_sha256", "0" * 64),
                              ("source_sha256", {})]:
            bad = deepcopy(complete_fit); bad[change] = value; fit.write_text(json.dumps(bad))
            assert rejects(lambda: D.load_test("hmm_0", gate_path=fit))
            rejected.append("invalid fit seal " + change)
        assert rejects(lambda: D.load_development("hmm_5", gate_path=imp))
        rejected.append("task outside the frozen panel")
        checks.append("Both public domain loaders refuse unsealed, altered, or incomplete stage access before generating/parsing data")
        checks.append("Test gate binds all six tasks, the implementation seal, unchanged source hashes, protocol, and archive identity")

    spec = manufactured_spec()
    lines = manufactured_lines(spec, poison_roles=("test",))
    parsed = D._stream_appliances_records(lines, spec, ("outer_train", "validation"))
    assert set(parsed["segments"]) == {"outer_train", "validation"}
    assert parsed["metadata"]["numeric_rows_by_role"] == {"outer_train": 192, "validation": 192, "test": 0}
    assert parsed["metadata"]["numeric_values_by_role"] == {"outer_train": 3840, "validation": 3840, "test": 0}
    assert parsed["metadata"]["source_rows_metadata_checked"] == 576
    assert rejects(lambda: D._stream_appliances_records(lines, spec, ("test",)))
    rejected.append("poisoned test values after test-role opening")
    checks.append("Manufactured unparseable test fields remain opaque during development, but fail a permitted test numeric read")

    test_lines = manufactured_lines(spec, poison_roles=("outer_train", "validation"))
    test_parsed = D._stream_appliances_records(test_lines, spec, ("test",))
    assert set(test_parsed["segments"]) == {"test"}
    assert test_parsed["metadata"]["numeric_rows_by_role"] == {"outer_train": 0, "validation": 0, "test": 192}
    checks.append("Test-role parsing likewise skips all development numeric fields and preserves excluded weather/random columns as unused")

    original_csv_reader = csv.reader
    reader_calls = []
    def traced_reader(rows, *args, **kwargs):
        materialized = list(rows)
        if any("SEALED_NUMERIC_VALUE" in row for row in materialized):
            raise AssertionError("Forbidden test records reached the numeric-field CSV parser")
        reader_calls.append(len(materialized))
        return original_csv_reader(materialized, *args, **kwargs)
    with patch.object(D.csv, "reader", traced_reader):
        D._stream_appliances_records(lines, spec, ("outer_train", "validation"))
    assert len(reader_calls) == 385
    checks.append("Instrumented parser proves forbidden-role records never reach CSV field splitting, while timestamp/schema integrity is checked globally")

    anchors = np.array([154, 143, 150], dtype=np.int64)
    x, y = D._appliances_examples(parsed["segments"]["outer_train"], anchors, spec, include_targets=True)
    assert x.shape == (3, 167) and y.shape == (3, 3)
    header = spec["source"]["expected_csv_columns"]
    channels = spec["features"]["current_local_channels"]
    for k, anchor in enumerate(anchors):
        assert np.array_equal(x[k, :144], 10 * (anchor - np.arange(144)))
        assert np.array_equal(x[k, 144:163], [1000 * anchor + header.index(name) for name in channels])
        assert np.array_equal(y[k], [10 * (anchor + h) for h in (1, 6, 36)])
        timestamp = datetime(2000, 1, 3) + timedelta(minutes=10 * int(anchor))
        f = (timestamp.hour * 3600 + timestamp.minute * 60) / 86400
        expected_calendar = [math.sin(2 * math.pi * f), math.cos(2 * math.pi * f),
                             math.sin(2 * math.pi * (timestamp.weekday() + f) / 7),
                             math.cos(2 * math.pi * (timestamp.weekday() + f) / 7)]
        assert np.allclose(x[k, 163:], expected_calendar, rtol=0, atol=1e-14)
        assert x[k, 0] == 10 * anchor
        for h in (1, 6, 36):
            assert x[k, 144 - h] == 10 * (anchor + h - 144)
    support, no_labels = D._appliances_examples(parsed["segments"]["outer_train"], anchors, spec, include_targets=False)
    assert no_labels is None and np.array_equal(support, x)
    checks.append("All167 feature coordinates, reverse-lag order, three direct target columns, reference priority, calendar values, persistence and seasonal indices match independent manufactured values")
    checks.append("Reference feature construction never supplies future reference labels")
    for bad_anchor in (142, 156):
        assert rejects(lambda bad_anchor=bad_anchor: D._appliances_examples(
            parsed["segments"]["outer_train"], np.array([bad_anchor], dtype=np.int64), spec, include_targets=True))
        rejected.append("outer history/target boundary " + str(bad_anchor))
    checks.append("Full common-horizon footprint checks reject both history and future-target boundary crossings")

    for name, bad_lines in [("missing source row", lines[:-1]),
                            ("extra source row", lines + [lines[-1]]),
                            ("blank record", lines[:10] + ["\n"] + lines[11:]),
                            ("duplicate timestamp", lines[:11] + [lines[10]] + lines[12:])]:
        assert rejects(lambda bad_lines=bad_lines: D._stream_appliances_records(bad_lines, spec, ("outer_train", "validation")))
        rejected.append(name)
    wrong_header = list(lines); wrong_header[0] = wrong_header[0].replace('"T1"', '"T_BAD"')
    assert rejects(lambda: D._stream_appliances_records(wrong_header, spec, ("outer_train", "validation")))
    rejected.append("header mutation")
    nonfinite = manufactured_lines(spec, poison_roles=("validation",), poison_value="NaN")
    assert rejects(lambda: D._stream_appliances_records(nonfinite, spec, ("outer_train", "validation")))
    rejected.append("nonfinite selected development field")
    checks.append("Malformed metadata and permitted numeric nonfiniteness fail closed without row dropping, sorting or replacement")

    h = D.hmm_module()
    observations = [i % 4 for i in range(32)] + [0] * 16
    observations[31] = 0
    observations[32], observations[35], observations[47] = 1, 2, 3
    for horizons in ((1, 4, 16), (16, 1, 4)):
        scripted = ScriptedIntegers(scripted_hmm_draws(observations))
        history, targets = h.simulate_one(scripted, history_length=32, horizons=horizons)
        assert history == tuple(observations[:32])
        assert targets == tuple(observations[31 + k] for k in horizons)
        assert scripted.position == 96
    scripted = ScriptedIntegers(scripted_hmm_draws(observations[:32]))
    history, targets = h.simulate_one(scripted, history_length=32, horizons=())
    assert history == tuple(observations[:32]) and targets == () and scripted.position == 64
    checks.append("Scripted HMM draws verify observation-first order, all three exact future indices, reordered horizons, and support termination without future draws")

    provider_calls = []
    def manufactured_provider(replication, role):
        provider_calls.append(role)
        n = h.ROLE_COUNTS[role]
        symbols = np.asarray(h.SYMBOL_VALUES, dtype=np.float64)
        X = np.tile(np.resize(symbols, 32), (n, 1))
        Y = None if role == "support" else np.tile(np.array([-1., .5, 1.]), (n, 1))
        return {"X": X, "Y": Y, "metadata": {
            "replication": replication, "role": role, "rows": n, "history_length": 32,
            "horizons": [] if role == "support" else [1, 4, 16],
            "seed_entropy": list(h.seed_entropy(replication, role)), "bit_generator": "PCG64DXSM",
            "one_example_per_independent_trajectory": True, "hidden_states_or_oracles_in_inputs": False,
            "support_future_targets_generated": False, "origin": "manufactured arrays only"}}
    packaged = D._hmm_roles(2, ("support", "train", "validation"), manufactured_provider)
    assert provider_calls == ["support", "train", "validation"]
    assert packaged["support"]["Y"] is None
    assert np.array_equal(packaged["train"]["Y"][0], [-1., .5, 1.])
    assert packaged["train"]["anchor_ids"][0] == 1_000_000
    assert packaged["validation"]["anchor_ids"][0] == 2_000_000
    def bad_provider(rep, role):
        record = manufactured_provider(rep, role)
        record["latent_state"] = 0
        return record
    assert rejects(lambda: D._hmm_roles(2, ("train",), bad_provider))
    rejected.append("latent/oracle field injection")
    checks.append("Manufactured HMM role packaging preserves fixed counts/horizon order, omits test generation during development, and rejects latent-field injection")

    receipt = {
        "schema": "step24-manufactured-data-preflight-v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "check_groups": len(checks), "deliberate_rejections": rejected,
        "actual_benchmark_trajectories_generated": 0, "actual_source_csv_rows_parsed": 0,
        "actual_test_numeric_values_converted": 0, "model_fits": 0, "ECG_or_reserve_access": False,
        "manufactured_csv_rows_per_fixture": 576,
        "instrumented_development_csv_reader_calls": len(reader_calls),
        "source_sha256": {**D.source_bindings(),
                          str(Path(__file__).resolve().relative_to(BASE)): D.file_sha256(Path(__file__).resolve())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    def forbidden_benchmark_rng(*args, **kwargs):
        raise AssertionError("Manufactured checks must not construct a benchmark PCG64DXSM stream")
    with patch.object(np.random, "PCG64DXSM", forbidden_benchmark_rng):
        main()
