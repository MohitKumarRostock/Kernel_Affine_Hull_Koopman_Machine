#!/usr/bin/env python3
"""Independent numerical verifier for the confirmatory dynamical campaign.

The verifier reads retained confirmatory evidence but does not modify it.

It independently verifies:
- the complete campaign SHA256SUMS inventory;
- exact configuration, protocol, execution-source, schedule and event records;
- all retained seed material and generated seed values;
- every retained time-index draw from the retained time seed;
- the committed independently verified omega=128 representation;
- every retained KAHM association matrix;
- nearest-center hard labels and selected-center distances;
- class counts, class means, SSE, f_hat and s_hat;
- frozen-predictor spectral norm, MSE and RMSE;
- every prespecified kappa certificate and feasibility decision;
- coordinate and physical-state range diagnostics.

No physical dynamical pair dataset is regenerated and no representation is
refit during verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kahkm_confirmatory_frozen_source import (
    EXPECTED_ARCHIVE_MANIFEST_SHA256,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_BUILD_EXECUTION_COMMIT,
    EXPECTED_PACKAGE_CHECKSUMS_SHA256,
    EXPECTED_VERIFIER_SOURCE_COMMIT,
    restore_repository_confirmatory_representation,
)
from kahkm_frozen_representation import load_frozen_representation
from kernel_affine_hull_koopman_machines import kahm_associations
from scripts.verify_dynamical_pilot_numerics import (
    ATOL,
    RTOL,
    close_array,
    close_scalar,
    exact_array,
    generated_seed_from_material,
    independent_certificate,
    independent_statistics,
    load_npz,
    nearest_center_labels,
    norm_budgets,
    require,
    safe_relative_path,
)


VERIFIER_ID = "dynamical_confirmatory_independent_numerics_v1"

EXPECTED_EXECUTION_COMMIT = (
    "18dc9f3b965e2cc634764e05785ff60d5fbd6422"
)
EXPECTED_RUNNER_ID = "dynamical_original_iid_confirmatory_runner_v1"
EXPECTED_CAMPAIGN_ID = "dynamical_original_iid_confirmatory_v1"
EXPECTED_CAMPAIGN_ROLE = "confirmatory"

CONFIG_PATH = (
    "reproduction/certificates/configs/"
    "dynamical_confirmatory_v1.json"
)
CONFIG_SHA256 = (
    "1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91"
)

PROTOCOL_PATH = (
    "reproduction/certificates/"
    "DYNAMICAL_CONFIRMATORY_PROTOCOL.md"
)
PROTOCOL_SHA256 = (
    "daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c"
)

EXPECTED_ATTEMPTS = 64
EXPECTED_RETAINED_PAIRS = 1_048_576
EXPECTED_SEED_STREAMS = 2_097_152
EXPECTED_EVIDENCE_INVENTORY_COUNT = 344

EXPECTED_EVIDENCE_SHA256SUMS_SHA256 = (
    "df9b1b7abd4a4c38b5bf8d0a5f43d7f6cb299ed23c070960b506f52641243221"
)

EXPECTED_SAMPLING_ID = "dynamical_certificate_sampling_v1"
EXPECTED_PAIR_LAW_ID = "independent_uniform_time_pair_v1"
EXPECTED_EVALUATION_ID = "frozen_kahkm_original_iid_evaluation_v1"
EXPECTED_REFERENCE_MAP_ID = "nearest_stored_kmeans_center_v1"
EXPECTED_SAMPLE_EVIDENCE_ID = "dynamical_sample_evidence_v1"
EXPECTED_EVALUATION_EVIDENCE_ID = "dynamical_evaluation_evidence_v1"

EXPECTED_REPRESENTATION_MANIFEST_SHA256 = (
    "c3f440a0d454bb3c5b9c96f455aa81079da0365a52554b723d013a429dc8035a"
)

SAMPLE_ARRAY_MEMBERS = {
    "current_states",
    "successor_states",
    "time_indices",
    "trajectory_seed_generated_seed",
    "trajectory_seed_uint32_words",
    "trajectory_seed_material",
    "time_seed_generated_seed",
    "time_seed_uint32_words",
    "time_seed_material",
}

EVALUATION_ARRAY_MEMBERS = {
    "phi_rows",
    "successor_rows",
    "labels",
    "selected_center_squared_distances",
    "class_counts",
    "class_successor_sse",
    "class_successor_means",
}

REQUIRED_VERIFIER_SOURCE_FILES = (
    "scripts/verify_dynamical_confirmatory_numerics.py",
    "test_verify_dynamical_confirmatory_numerics.py",
    "scripts/verify_dynamical_pilot_numerics.py",
    "experiment_23_certificate_dynamical_confirmatory.py",
    "kahkm_confirmatory_frozen_source.py",
    "kahkm_confirmatory_sampling.py",
    "kahkm_confirmatory_evaluation.py",
    "kahkm_dynamical_pairs.py",
    "kahkm_dynamical_evaluation.py",
    "kahkm_certificate_statistics.py",
    "kahkm_certificates.py",
    "kahkm_frozen_representation.py",
    "kahkm_reference_classes.py",
    "kernel_affine_hull_koopman_machines.py",
    CONFIG_PATH,
    PROTOCOL_PATH,
    (
        "reproduction/prospective_campaigns/"
        "dynamical_confirmatory_representation_v1_run001/"
        "ARCHIVE.json"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_confirmatory_representation_v1_run001/"
        "README.md"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_confirmatory_representation_v1_run001/"
        "SHA256SUMS"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_confirmatory_representation_v1_run001/"
        "evidence.tar.gz"
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_json_write(path: Path, payload: Any) -> None:
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe JSONL: {path.name}")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"JSONL is not newline terminated: {path.name}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        require(bool(line), f"Blank JSONL line in {path.name}:{line_number}")
        value = json.loads(line)
        require(isinstance(value, dict), f"Non-object JSONL record in {path.name}:{line_number}")
        records.append(value)
    return records


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
    )


def verifier_provenance() -> dict[str, Any]:
    actual_root = Path(
        git_bytes("rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    require(actual_root == ROOT, "Verifier repository root mismatch.")

    commit = git_bytes("rev-parse", "HEAD").decode().strip()
    status = git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).decode()
    require(status == "", "Verifier requires a clean committed source tree.")

    hashes: dict[str, str] = {}
    for relative in sorted(set(REQUIRED_VERIFIER_SOURCE_FILES)):
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"Missing verifier source: {relative}")

        actual = sha256_file(path)
        committed = hashlib.sha256(
            git_bytes("show", f"{commit}:{relative}")
        ).hexdigest()
        require(actual == committed, f"Verifier source differs from HEAD: {relative}")
        hashes[relative] = actual

    return {
        "verifier_source_commit": commit,
        "git_status_porcelain": status,
        "source_file_sha256": hashes,
    }


def check_inventory(
    folder: Path,
    *,
    expected_count: int | None = EXPECTED_EVIDENCE_INVENTORY_COUNT,
    expected_fingerprint: str | None = EXPECTED_EVIDENCE_SHA256SUMS_SHA256,
) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    manifest = folder / "SHA256SUMS"

    require(
        manifest.is_file() and not manifest.is_symlink(),
        "Missing or unsafe SHA256SUMS.",
    )

    fingerprint = sha256_file(manifest)
    if expected_fingerprint is not None:
        require(
            fingerprint == expected_fingerprint,
            "Evidence SHA256SUMS fingerprint mismatch.",
        )

    files: dict[str, str] = {}

    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        digest, separator, name = line.partition("  ")

        require(
            bool(separator)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"Malformed SHA256SUMS line {line_number}.",
        )

        relative = safe_relative_path(name)

        require(
            relative.name != "SHA256SUMS",
            "SHA256SUMS must not inventory a SHA256SUMS file.",
        )
        require(name not in files, f"Duplicate SHA256SUMS entry: {name}")

        path = folder / relative
        require(path.is_file() and not path.is_symlink(), f"Missing/unsafe inventoried file: {name}")

        actual = sha256_file(path)
        require(actual == digest, f"SHA256 mismatch: {name}")
        files[name] = digest

    if expected_count is not None:
        require(
            len(files) == expected_count,
            f"Evidence inventory count mismatch: {len(files)} != {expected_count}.",
        )

    physical = {
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
    }

    require(
        set(files) == physical,
        "Evidence file inventory differs from SHA256SUMS.",
    )

    return {
        "sha256sums_fingerprint": fingerprint,
        "verified_file_count": len(files),
        "files": files,
    }


def load_config_from_evidence(evidence: Path) -> dict[str, Any]:
    retained = evidence / "campaign_spec.json"
    repository = ROOT / CONFIG_PATH

    require(retained.is_file() and not retained.is_symlink(), "Missing retained campaign_spec.json.")
    require(sha256_file(retained) == CONFIG_SHA256, "Retained config fingerprint mismatch.")
    require(sha256_file(repository) == CONFIG_SHA256, "Repository config fingerprint mismatch.")
    require(retained.read_bytes() == repository.read_bytes(), "Retained config differs from repository bytes.")

    config = strict_json_load(retained)

    require(config["campaign_id"] == EXPECTED_CAMPAIGN_ID, "Campaign ID mismatch.")
    require(config["campaign_role"] == EXPECTED_CAMPAIGN_ROLE, "Campaign role mismatch.")

    design = config["confirmatory_design"]
    require(design["systems"] == ["vanderpol"], "Unexpected confirmatory systems.")
    require(design["sample_sizes"] == [16384], "Unexpected confirmatory sample size.")
    require(design["replicates_per_cell"] == 32, "Unexpected confirmatory replicate count.")
    require(design["planned_attempts"] == EXPECTED_ATTEMPTS, "Planned attempt count mismatch.")
    require(design["planned_retained_pairs"] == EXPECTED_RETAINED_PAIRS, "Planned pair count mismatch.")

    require(
        config["pair_sampling"]["evaluation_modes"]
        == ["matched_stochastic", "deterministic"],
        "Unexpected evaluation modes.",
    )
    require(config["pair_sampling"]["horizon_steps"] == 1200, "Unexpected horizon.")
    require(config["pair_sampling"]["implementation_id"] == EXPECTED_PAIR_LAW_ID, "Pair-law mismatch.")
    require(config["certificate"]["delta"] == 0.05, "Certificate delta mismatch.")

    seed = config["seed_scheme"]
    require(seed["campaign_key"] == 4, "Confirmatory campaign key mismatch.")
    require(seed["root_seed"] == 20261003, "Root seed mismatch.")
    require(seed["system_keys"]["vanderpol"] == 2, "Van der Pol key mismatch.")
    require(seed["mode_keys"] == {"deterministic": 2, "matched_stochastic": 1}, "Mode keys mismatch.")
    require(seed["stream_keys"] == {"time_seed": 2, "trajectory_seed": 1}, "Stream keys mismatch.")

    representation = config["confirmatory_representation"]
    require(representation["system"] == "vanderpol", "Representation system mismatch.")
    require(representation["omega"] == 128.0, "Representation omega mismatch.")
    require(representation["tau"] == 1e-6, "Representation tau mismatch.")
    require(representation["predictor_orientation"] == "B.T @ Phi", "Predictor orientation mismatch.")
    require(representation["project_stochastic"] is False, "Unexpected stochastic projection.")

    selection = config["selection_provenance"]
    require(selection["fresh_confirmatory_evaluation_pairs_required"] is True, "Fresh-pair firewall missing.")
    require(selection["pilot_evaluation_pairs_may_be_used_as_confirmatory_evidence"] is False, "Pilot firewall changed.")

    return config


def reconstruct_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = config["confirmatory_design"]
    pair = config["pair_sampling"]
    seed = config["seed_scheme"]

    jobs: list[dict[str, Any]] = []

    for system in design["systems"]:
        system_key = int(seed["system_keys"][system])

        for mode in pair["evaluation_modes"]:
            mode_key = int(seed["mode_keys"][mode])

            for sample_size_raw in design["sample_sizes"]:
                sample_size = int(sample_size_raw)

                for replicate_index in range(int(design["replicates_per_cell"])):
                    jobs.append(
                        {
                            "attempt_id": (
                                f"s{system_key}"
                                f"_q{mode_key}"
                                f"_m{sample_size}"
                                f"_r{replicate_index:03d}"
                            ),
                            "attempt_number": 1,
                            "system": system,
                            "system_key": system_key,
                            "evaluation_mode": mode,
                            "mode_key": mode_key,
                            "sample_size": sample_size,
                            "replicate_index": replicate_index,
                            "campaign_key": int(seed["campaign_key"]),
                            "root_seed": int(seed["root_seed"]),
                            "trajectory_stream_key": int(seed["stream_keys"]["trajectory_seed"]),
                            "time_stream_key": int(seed["stream_keys"]["time_seed"]),
                        }
                    )

    require(len(jobs) == EXPECTED_ATTEMPTS, "Reconstructed schedule attempt count mismatch.")
    require(
        sum(job["sample_size"] for job in jobs) == EXPECTED_RETAINED_PAIRS,
        "Reconstructed schedule pair count mismatch.",
    )
    require(
        len({job["attempt_id"] for job in jobs}) == EXPECTED_ATTEMPTS,
        "Duplicate reconstructed attempt identifiers.",
    )

    return jobs


def verify_execution_source_snapshot(source: Mapping[str, Any]) -> None:
    require(source.get("commit") == EXPECTED_EXECUTION_COMMIT, "Execution source commit mismatch.")
    require(source.get("git_status_porcelain") == "", "Execution source was dirty.")

    hashes = source.get("required_source_sha256")
    require(isinstance(hashes, dict), "Execution source hash inventory missing.")

    mandatory = {
        "experiment_23_certificate_dynamical_confirmatory.py",
        "kahkm_confirmatory_sampling.py",
        "kahkm_confirmatory_evaluation.py",
        "kahkm_confirmatory_frozen_source.py",
        CONFIG_PATH,
        PROTOCOL_PATH,
    }
    require(mandatory.issubset(hashes), "Execution source hash inventory lacks mandatory files.")

    for relative, retained_hash in hashes.items():
        committed = hashlib.sha256(
            git_bytes("show", f"{EXPECTED_EXECUTION_COMMIT}:{relative}")
        ).hexdigest()
        require(committed == retained_hash, f"Execution-source byte hash mismatch: {relative}")


def verify_campaign_documents(
    evidence: Path,
    config: Mapping[str, Any],
    expected_schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = evidence / "protocol.md"
    require(protocol.is_file() and not protocol.is_symlink(), "Missing retained protocol.")
    require(sha256_file(protocol) == PROTOCOL_SHA256, "Retained protocol fingerprint mismatch.")
    require(sha256_file(ROOT / PROTOCOL_PATH) == PROTOCOL_SHA256, "Repository protocol fingerprint mismatch.")
    require(protocol.read_bytes() == (ROOT / PROTOCOL_PATH).read_bytes(), "Retained protocol differs from repository bytes.")

    metadata = strict_json_load(evidence / "campaign_metadata.json")
    summary = strict_json_load(evidence / "campaign_summary.json")
    source_after = strict_json_load(evidence / "source_after.json")
    representation = strict_json_load(evidence / "representation_source.json")

    require(metadata["runner_id"] == EXPECTED_RUNNER_ID, "Runner ID mismatch.")
    require(metadata["campaign_id"] == EXPECTED_CAMPAIGN_ID, "Metadata campaign ID mismatch.")
    require(metadata["campaign_role"] == EXPECTED_CAMPAIGN_ROLE, "Metadata campaign role mismatch.")
    require(metadata["config_sha256"] == CONFIG_SHA256, "Metadata config SHA mismatch.")
    require(metadata["protocol_sha256"] == PROTOCOL_SHA256, "Metadata protocol SHA mismatch.")
    verify_execution_source_snapshot(metadata["source"])
    require(source_after == metadata["source"], "source_after differs from execution-source snapshot.")

    expected_summary = {
        "status": "completed",
        "exit_code": 0,
        "planned": EXPECTED_ATTEMPTS,
        "started": EXPECTED_ATTEMPTS,
        "sampled": EXPECTED_ATTEMPTS,
        "evaluated": EXPECTED_ATTEMPTS,
        "succeeded": EXPECTED_ATTEMPTS,
        "failed": 0,
        "not_started": 0,
        "started_without_terminal_event": 0,
        "planned_retained_pairs": EXPECTED_RETAINED_PAIRS,
        "sampled_pairs": EXPECTED_RETAINED_PAIRS,
        "evaluated_pairs": EXPECTED_RETAINED_PAIRS,
        "source_unchanged": True,
        "fresh_confirmatory_pairs_required": True,
        "pilot_evaluation_pairs_reused": False,
        "representation_refits_during_campaign": 0,
        "independent_evidence_verification": "not_yet_performed",
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_role": EXPECTED_CAMPAIGN_ROLE,
    }

    for key, expected in expected_summary.items():
        require(summary.get(key) == expected, f"Campaign summary mismatch: {key}")

    require(summary.get("active_attempt_id") is None, "Campaign ended with active attempt.")
    require(float(summary["elapsed_seconds"]) >= 0.0, "Campaign elapsed time is negative.")

    require(representation["frozen_source_id"] == "repository_verified_confirmatory_representation_v1", "Frozen source ID mismatch.")
    require(representation["system"] == "vanderpol", "Representation system mismatch.")
    require(representation["omega"] == 128.0, "Representation omega mismatch.")
    require(representation["tau"] == 1e-6, "Representation tau mismatch.")
    require(representation["archive_sha256"] == EXPECTED_ARCHIVE_SHA256, "Representation archive SHA mismatch.")
    require(representation["archive_manifest_sha256"] == EXPECTED_ARCHIVE_MANIFEST_SHA256, "Representation archive-manifest SHA mismatch.")
    require(representation["package_checksums_sha256"] == EXPECTED_PACKAGE_CHECKSUMS_SHA256, "Representation package-checksums SHA mismatch.")
    require(representation["build_execution_source_commit"] == EXPECTED_BUILD_EXECUTION_COMMIT, "Representation build commit mismatch.")
    require(representation["verifier_source_commit"] == EXPECTED_VERIFIER_SOURCE_COMMIT, "Representation verifier commit mismatch.")
    require(representation["representation_manifest_sha256"] == EXPECTED_REPRESENTATION_MANIFEST_SHA256, "Representation manifest SHA mismatch.")
    require(representation["state_space_kmeans_refits"] == 0, "Unexpected K-means refit.")
    require(representation["autoencoder_refits"] == 0, "Unexpected autoencoder refit.")
    require(representation["nlms_operator_refits"] == 0, "Unexpected NLMS refit during campaign.")
    require(representation["certificate_evaluation_pairs_generated_during_restore"] == 0, "Representation restore generated evaluation pairs.")

    schedule = jsonl_records(evidence / "schedule.jsonl")
    require(schedule == expected_schedule, "Retained schedule differs from reconstructed schedule.")

    return {
        "metadata": metadata,
        "summary": summary,
        "representation": representation,
    }


def verify_attempt_event_log(
    evidence: Path,
    expected_schedule: list[dict[str, Any]],
) -> None:
    records = jsonl_records(evidence / "attempts.jsonl")
    require(len(records) == 2 * EXPECTED_ATTEMPTS, "Attempt-event record count mismatch.")

    for index, job in enumerate(expected_schedule):
        started = records[2 * index]
        completed = records[2 * index + 1]

        require(started["attempt_id"] == job["attempt_id"], "Started-event attempt order mismatch.")
        require(completed["attempt_id"] == job["attempt_id"], "Completed-event attempt order mismatch.")
        require(started["attempt_number"] == 1 and completed["attempt_number"] == 1, "Attempt number mismatch.")
        require(started["event"] == "started", "Expected started event.")
        require(completed["event"] == "completed", "Expected completed event.")
        require("stage" not in started and "stage" not in completed, "Unexpected failure stage in successful event.")


def index_unique(
    records: list[dict[str, Any]],
    *,
    key: str,
    expected_count: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    require(len(records) == expected_count, f"{label} count mismatch.")
    result: dict[str, dict[str, Any]] = {}

    for record in records:
        value = record.get(key)
        require(isinstance(value, str), f"{label} key is not a string.")
        require(value not in result, f"Duplicate {label} key: {value}")
        result[value] = record

    return result


def expected_seed_material(
    *,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    pair_index: int,
    stream_name: str,
) -> tuple[int, ...]:
    seed = config["seed_scheme"]

    return (
        int(seed["campaign_key"]),
        int(job["system_key"]),
        int(job["mode_key"]),
        int(job["sample_size"]),
        int(job["replicate_index"]),
        int(pair_index),
        int(seed["stream_keys"][stream_name]),
        int(seed["root_seed"]),
    )


def _direct_seed_words(material: tuple[int, ...]) -> tuple[int, int]:
    words = np.random.SeedSequence(
        entropy=material
    ).generate_state(2, dtype=np.uint32)

    return (
        int(words[0]),
        int(words[1]),
    )


def verify_sample(
    *,
    attempt_dir: Path,
    job: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], int, int]:
    metadata_path = attempt_dir / "sample.json"
    arrays_path = attempt_dir / "sample_arrays.npz"

    metadata = strict_json_load(metadata_path)

    require(metadata["sample_evidence_id"] == EXPECTED_SAMPLE_EVIDENCE_ID, "Sample evidence ID mismatch.")
    require(metadata["sampling_id"] == EXPECTED_SAMPLING_ID, "Sampling ID mismatch.")
    require(metadata["pair_law_id"] == EXPECTED_PAIR_LAW_ID, "Pair-law ID mismatch.")
    require(metadata["system"] == job["system"], "Sample system mismatch.")
    require(metadata["evaluation_mode"] == job["evaluation_mode"], "Sample mode mismatch.")
    require(metadata["sample_size"] == job["sample_size"], "Sample size mismatch.")
    require(metadata["replicate_index"] == job["replicate_index"], "Sample replicate mismatch.")
    require(metadata["state_dimension"] == 2, "Sample state dimension mismatch.")
    require(metadata["dt"] == 0.02, "Sample dt mismatch.")
    require(metadata["horizon_steps"] == 1200, "Sample horizon mismatch.")
    require(metadata["vanderpol_mu"] == 1.0, "Sample Van der Pol mu mismatch.")
    require(metadata["evaluation_performed"] is False, "Sample metadata must predate evaluation.")
    require(set(metadata["array_members"]) == SAMPLE_ARRAY_MEMBERS, "Sample member inventory mismatch.")
    require(sha256_file(arrays_path) == metadata["array_sha256"], "Sample array SHA mismatch.")

    arrays = load_npz(arrays_path, SAMPLE_ARRAY_MEMBERS)
    M = int(job["sample_size"])

    current = arrays["current_states"]
    successor = arrays["successor_states"]
    time_indices = arrays["time_indices"]

    require(current.shape == (2, M) and current.dtype == np.float64, "Current-state array shape/dtype mismatch.")
    require(successor.shape == (2, M) and successor.dtype == np.float64, "Successor-state array shape/dtype mismatch.")
    require(time_indices.shape == (M,) and time_indices.dtype == np.int64, "Time-index array shape/dtype mismatch.")
    require(np.all(np.isfinite(current)) and np.all(np.isfinite(successor)), "Retained physical states contain nonfinite values.")
    require(np.all((0 <= time_indices) & (time_indices < 1200)), "Retained time index lies outside horizon.")

    for prefix in ("trajectory_seed", "time_seed"):
        require(
            arrays[f"{prefix}_material"].shape == (M, 8)
            and arrays[f"{prefix}_material"].dtype == np.uint64,
            f"{prefix} material shape/dtype mismatch.",
        )
        require(
            arrays[f"{prefix}_uint32_words"].shape == (M, 2)
            and arrays[f"{prefix}_uint32_words"].dtype == np.uint32,
            f"{prefix} words shape/dtype mismatch.",
        )
        require(
            arrays[f"{prefix}_generated_seed"].shape == (M,)
            and arrays[f"{prefix}_generated_seed"].dtype == np.uint64,
            f"{prefix} generated-seed shape/dtype mismatch.",
        )

    time_draws = 0
    seed_streams = 0

    for pair_index in range(M):
        generated_for_pair: dict[str, int] = {}

        for stream_name, prefix in (
            ("trajectory_seed", "trajectory_seed"),
            ("time_seed", "time_seed"),
        ):
            expected_material = expected_seed_material(
                config=config,
                job=job,
                pair_index=pair_index,
                stream_name=stream_name,
            )

            retained_material = tuple(
                int(value)
                for value in arrays[f"{prefix}_material"][pair_index]
            )

            require(
                retained_material == expected_material,
                f"{job['attempt_id']} seed material mismatch pair={pair_index} stream={stream_name}.",
            )

            helper_words, expected_seed = generated_seed_from_material(
                expected_material
            )

            expected_words = tuple(
                int(value)
                for value in helper_words
            )

            direct_words = _direct_seed_words(
                expected_material
            )

            require(
                expected_words == direct_words,
                f"{job['attempt_id']} helper/direct SeedSequence disagreement pair={pair_index} stream={stream_name}.",
            )

            retained_words = tuple(
                int(value)
                for value in arrays[f"{prefix}_uint32_words"][pair_index]
            )

            require(
                retained_words == expected_words,
                f"{job['attempt_id']} SeedSequence words mismatch pair={pair_index} stream={stream_name}.",
            )

            retained_seed = int(arrays[f"{prefix}_generated_seed"][pair_index])

            require(
                retained_seed == int(expected_seed),
                f"{job['attempt_id']} generated seed mismatch pair={pair_index} stream={stream_name}.",
            )

            generated_for_pair[stream_name] = retained_seed
            seed_streams += 1

        require(
            generated_for_pair["trajectory_seed"]
            != generated_for_pair["time_seed"],
            f"{job['attempt_id']} trajectory/time seed collision pair={pair_index}.",
        )

        recomputed_time = int(
            np.random.default_rng(
                generated_for_pair["time_seed"]
            ).integers(0, 1200)
        )

        require(
            int(time_indices[pair_index]) == recomputed_time,
            f"{job['attempt_id']} time-index draw mismatch pair={pair_index}.",
        )
        time_draws += 1

    return metadata, arrays, time_draws, seed_streams


def verify_evaluation(
    *,
    attempt_dir: Path,
    job: Mapping[str, Any],
    sample_arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    model: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = attempt_dir / "evaluation.json"
    arrays_path = attempt_dir / "evaluation_arrays.npz"

    metadata = strict_json_load(metadata_path)

    require(metadata["evaluation_evidence_id"] == EXPECTED_EVALUATION_EVIDENCE_ID, "Evaluation evidence ID mismatch.")
    require(metadata["evaluation_id"] == EXPECTED_EVALUATION_ID, "Evaluation ID mismatch.")
    require(metadata["reference_class_map_id"] == EXPECTED_REFERENCE_MAP_ID, "Reference map ID mismatch.")
    require(metadata["sample_evidence_present_before_evaluation_write"] is True, "Evidence-order flag mismatch.")
    require(metadata["system"] == job["system"], "Evaluation system mismatch.")
    require(metadata["evaluation_mode"] == job["evaluation_mode"], "Evaluation mode mismatch.")
    require(metadata["sample_size"] == job["sample_size"], "Evaluation sample-size mismatch.")
    require(metadata["replicate_index"] == job["replicate_index"], "Evaluation replicate mismatch.")
    require(metadata["n_pairs"] == job["sample_size"], "Evaluation n_pairs mismatch.")
    require(metadata["n_classes"] == 25, "Evaluation class-count dimension mismatch.")
    require(metadata["state_dimension"] == 2, "Evaluation state dimension mismatch.")
    require(metadata["delta"] == config["certificate"]["delta"], "Evaluation delta mismatch.")
    require(set(metadata["array_members"]) == EVALUATION_ARRAY_MEMBERS, "Evaluation member inventory mismatch.")
    require(sha256_file(arrays_path) == metadata["array_sha256"], "Evaluation array SHA mismatch.")

    arrays = load_npz(arrays_path, EVALUATION_ARRAY_MEMBERS)

    M = int(job["sample_size"])
    C = 25

    require(arrays["phi_rows"].shape == (M, C) and arrays["phi_rows"].dtype == np.float64, "Phi shape/dtype mismatch.")
    require(arrays["successor_rows"].shape == (M, C) and arrays["successor_rows"].dtype == np.float64, "Successor Phi shape/dtype mismatch.")
    require(arrays["labels"].shape == (M,) and arrays["labels"].dtype == np.int64, "Label shape/dtype mismatch.")
    require(arrays["selected_center_squared_distances"].shape == (M,), "Distance shape mismatch.")
    require(arrays["class_counts"].shape == (C,) and arrays["class_counts"].dtype == np.int64, "Class-count shape/dtype mismatch.")
    require(arrays["class_successor_sse"].shape == (C,), "Class-SSE shape mismatch.")
    require(arrays["class_successor_means"].shape == (C, C), "Class-mean shape mismatch.")

    current = np.asarray(sample_arrays["current_states"], dtype=np.float64)
    successor = np.asarray(sample_arrays["successor_states"], dtype=np.float64)

    phi_columns = np.asarray(
        kahm_associations(
            model.abstraction_model,
            current,
            omega=float(model.omega),
            tau=float(model.tau),
            n_jobs=-1,
            batch_size=256,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    successor_columns = np.asarray(
        kahm_associations(
            model.abstraction_model,
            successor,
            omega=float(model.omega),
            tau=float(model.tau),
            n_jobs=-1,
            batch_size=256,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    require(phi_columns.shape == (C, M), "Recomputed Phi shape mismatch.")
    require(successor_columns.shape == (C, M), "Recomputed successor-Phi shape mismatch.")

    phi_rows = np.asarray(phi_columns.T, dtype=np.float64)
    successor_rows = np.asarray(successor_columns.T, dtype=np.float64)

    close_array(arrays["phi_rows"], phi_rows, label=f"{job['attempt_id']} retained Phi(X)")
    close_array(arrays["successor_rows"], successor_rows, label=f"{job['attempt_id']} retained Phi(X+)")

    centers = np.asarray(
        model.abstraction_model["cluster_centers"],
        dtype=np.float64,
    )
    labels, distances = nearest_center_labels(centers, current)

    exact_array(arrays["labels"], labels, label=f"{job['attempt_id']} hard labels")
    close_array(
        arrays["selected_center_squared_distances"],
        distances,
        label=f"{job['attempt_id']} selected-center distances",
    )

    statistics = independent_statistics(
        phi_rows,
        successor_rows,
        labels,
    )

    exact_array(arrays["class_counts"], statistics["class_counts"], label=f"{job['attempt_id']} class counts")
    close_array(arrays["class_successor_sse"], statistics["class_successor_sse"], label=f"{job['attempt_id']} class successor SSE")
    close_array(arrays["class_successor_means"], statistics["class_successor_means"], label=f"{job['attempt_id']} class successor means")

    saved_statistics = metadata["statistics"]
    for key in (
        "f_hat",
        "s_hat",
        "current_max_row_sum_error",
        "successor_max_row_sum_error",
    ):
        close_scalar(
            saved_statistics[key],
            statistics[key],
            label=f"{job['attempt_id']} {key}",
        )

    require(saved_statistics["simplex_sum_atol"] == 1e-12, "Unexpected simplex_sum_atol.")

    B = np.asarray(model.B, dtype=np.float64)
    spectral_norm = float(np.linalg.norm(B, ord=2))
    prediction_columns = B.T @ phi_columns
    residual_columns = prediction_columns - successor_columns

    per_pair_squared_error = np.sum(
        residual_columns * residual_columns,
        axis=0,
        dtype=np.float64,
    )
    mse = float(np.mean(per_pair_squared_error, dtype=np.float64))
    rmse = float(np.sqrt(mse))

    close_scalar(metadata["frozen_predictor_spectral_norm"], spectral_norm, label=f"{job['attempt_id']} predictor spectral norm")
    close_scalar(metadata["frozen_predictor_mse"], mse, label=f"{job['attempt_id']} predictor MSE")
    close_scalar(metadata["frozen_predictor_rmse"], rmse, label=f"{job['attempt_id']} predictor RMSE")

    expected_kappas = norm_budgets(config, spectral_norm)
    saved_budgets = metadata["budget_certificates"]

    require(len(saved_budgets) == len(expected_kappas), "Budget-certificate count mismatch.")

    delta = float(config["certificate"]["delta"])

    for saved, kappa in zip(saved_budgets, expected_kappas, strict=True):
        close_scalar(saved["kappa"], kappa, label=f"{job['attempt_id']} kappa")

        certificate = independent_certificate(
            f_hat=statistics["f_hat"],
            s_hat=statistics["s_hat"],
            n_pairs=statistics["n_pairs"],
            kappa=kappa,
            delta=delta,
        )

        retained = saved["certificate"]
        require(int(retained["n_pairs"]) == certificate["n_pairs"], "Saved certificate pair count mismatch.")

        for key in (
            "kappa",
            "delta",
            "f_hat",
            "s_hat",
            "r_delta",
            "F_delta",
            "V_delta",
            "L_kappa_delta",
        ):
            close_scalar(
                retained[key],
                certificate[key],
                label=f"{job['attempt_id']} certificate {key}",
            )

        feasibility_atol = (
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, spectral_norm, kappa)
        )
        within_budget = bool(
            spectral_norm
            <= kappa + feasibility_atol
        )

        require(
            saved["frozen_predictor_within_budget"] is within_budget,
            f"{job['attempt_id']} predictor budget-membership mismatch.",
        )

        close_scalar(
            saved["frozen_predictor_rmse_minus_bound"],
            rmse - certificate["L_kappa_delta"],
            label=f"{job['attempt_id']} RMSE-minus-bound",
        )

    ranges = metadata["coordinate_ranges"]

    for key, value in (
        ("phi_min", np.min(phi_columns)),
        ("phi_max", np.max(phi_columns)),
        ("successor_min", np.min(successor_columns)),
        ("successor_max", np.max(successor_columns)),
    ):
        close_scalar(
            ranges[key],
            value,
            label=f"{job['attempt_id']} {key}",
        )

    state_ranges = metadata["state_ranges"]
    close_array(state_ranges["current_min"], np.min(current, axis=1), label=f"{job['attempt_id']} current-state min")
    close_array(state_ranges["current_max"], np.max(current, axis=1), label=f"{job['attempt_id']} current-state max")
    close_array(state_ranges["successor_min"], np.min(successor, axis=1), label=f"{job['attempt_id']} successor-state min")
    close_array(state_ranges["successor_max"], np.max(successor, axis=1), label=f"{job['attempt_id']} successor-state max")

    return metadata, arrays


def compare_restored_representation_trees(
    evidence_restored: Path,
    independently_restored: Path,
) -> int:
    left = {
        path.relative_to(evidence_restored).as_posix(): path
        for path in evidence_restored.rglob("*")
        if path.is_file()
    }
    right = {
        path.relative_to(independently_restored).as_posix(): path
        for path in independently_restored.rglob("*")
        if path.is_file()
    }

    require(set(left) == set(right), "Campaign-restored representation tree differs from independently restored tree.")

    for name in sorted(left):
        require(
            sha256_file(left[name]) == sha256_file(right[name]),
            f"Campaign-restored representation file mismatch: {name}",
        )

    return len(left)


def verify_evidence(
    evidence: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = evidence.expanduser().resolve()

    require(evidence.is_dir(), "Evidence directory does not exist.")
    require(not evidence.is_symlink(), "Evidence directory must not be a symlink.")

    inventory_before = check_inventory(evidence)
    config = load_config_from_evidence(evidence)
    expected_schedule = reconstruct_schedule(config)
    campaign = verify_campaign_documents(
        evidence,
        config,
        expected_schedule,
    )
    verify_attempt_event_log(
        evidence,
        expected_schedule,
    )

    sample_index = index_unique(
        jsonl_records(evidence / "samples.jsonl"),
        key="attempt_id",
        expected_count=EXPECTED_ATTEMPTS,
        label="sample index",
    )
    result_index = index_unique(
        jsonl_records(evidence / "results.jsonl"),
        key="attempt_id",
        expected_count=EXPECTED_ATTEMPTS,
        label="result index",
    )

    time_index_draws_recomputed = 0
    seed_streams_verified = 0
    evaluated_pairs = 0

    with tempfile.TemporaryDirectory(
        prefix="kahkm-confirmatory-verifier-"
    ) as temporary_parent:
        restoration_dir = (
            Path(temporary_parent)
            / "restored"
        )

        restored = restore_repository_confirmatory_representation(
            repository_root=ROOT,
            restoration_dir=restoration_dir,
        )

        require(restored.archive_sha256 == EXPECTED_ARCHIVE_SHA256, "Verifier-restored archive fingerprint mismatch.")
        require(restored.archive_manifest_sha256 == EXPECTED_ARCHIVE_MANIFEST_SHA256, "Verifier-restored manifest fingerprint mismatch.")
        require(restored.package_checksums_sha256 == EXPECTED_PACKAGE_CHECKSUMS_SHA256, "Verifier-restored package checksum mismatch.")

        restored_file_count = compare_restored_representation_trees(
            evidence / "representation_source",
            restored.restoration_dir,
        )

        model = load_frozen_representation(
            restored.vanderpol_artifact_dir,
            verify_hashes=True,
        )

        require(model.system == "vanderpol", "Verifier model system mismatch.")
        require(model.omega == 128.0, "Verifier model omega mismatch.")
        require(model.tau == 1e-6, "Verifier model tau mismatch.")
        require(
            sha256_file(restored.vanderpol_artifact_dir / "manifest.json")
            == EXPECTED_REPRESENTATION_MANIFEST_SHA256,
            "Verifier representation manifest mismatch.",
        )

        for job in expected_schedule:
            identifier = job["attempt_id"]
            attempt_relative = (
                Path("attempt_evidence")
                / identifier
            )
            attempt_dir = evidence / attempt_relative

            require(
                attempt_dir.is_dir()
                and not attempt_dir.is_symlink(),
                f"Missing/unsafe attempt directory: {identifier}",
            )

            sample_record = sample_index[identifier]
            result_record = result_index[identifier]

            require(
                sample_record["attempt_relative_dir"]
                == attempt_relative.as_posix(),
                f"Sample index path mismatch: {identifier}",
            )
            require(
                result_record["attempt_relative_dir"]
                == attempt_relative.as_posix(),
                f"Result index path mismatch: {identifier}",
            )

            (
                sample_metadata,
                sample_arrays,
                time_draw_count,
                seed_stream_count,
            ) = verify_sample(
                attempt_dir=attempt_dir,
                job=job,
                config=config,
            )

            time_index_draws_recomputed += time_draw_count
            seed_streams_verified += seed_stream_count

            require(
                sample_record["sample_metadata"]
                == sample_metadata,
                f"Compact sample metadata mismatch: {identifier}",
            )
            require(
                float(sample_record["sampling_seconds"]) >= 0.0,
                f"Negative sampling runtime: {identifier}",
            )

            evaluation_metadata, _evaluation_arrays = verify_evaluation(
                attempt_dir=attempt_dir,
                job=job,
                sample_arrays=sample_arrays,
                config=config,
                model=model,
            )

            for key in (
                "system",
                "evaluation_mode",
                "sample_size",
                "replicate_index",
            ):
                require(
                    result_record[key]
                    == job[key],
                    f"Compact result design mismatch: {identifier} {key}",
                )

            for key in (
                "frozen_predictor_spectral_norm",
                "frozen_predictor_mse",
                "frozen_predictor_rmse",
            ):
                close_scalar(
                    result_record[key],
                    evaluation_metadata[key],
                    label=f"{identifier} compact result {key}",
                )

            for key in (
                "f_hat",
                "s_hat",
            ):
                close_scalar(
                    result_record["statistics"][key],
                    evaluation_metadata["statistics"][key],
                    label=f"{identifier} compact result {key}",
                )

            require(
                result_record["statistics"]
                == evaluation_metadata["statistics"],
                f"Compact statistics record mismatch: {identifier}",
            )
            require(
                result_record["budget_certificates"]
                == evaluation_metadata["budget_certificates"],
                f"Compact budget-certificate record mismatch: {identifier}",
            )
            require(
                result_record["evaluation_array_sha256"]
                == evaluation_metadata["array_sha256"],
                f"Compact evaluation array SHA mismatch: {identifier}",
            )
            require(
                float(result_record["sampling_seconds"]) >= 0.0
                and float(result_record["evaluation_seconds"]) >= 0.0,
                f"Negative compact runtime: {identifier}",
            )

            evaluated_pairs += int(job["sample_size"])

    require(
        seed_streams_verified == EXPECTED_SEED_STREAMS,
        "Verified seed-stream count mismatch.",
    )
    require(
        time_index_draws_recomputed == EXPECTED_RETAINED_PAIRS,
        "Recomputed time-index draw count mismatch.",
    )
    require(
        evaluated_pairs == EXPECTED_RETAINED_PAIRS,
        "Verifier evaluated-pair count mismatch.",
    )

    inventory_after = check_inventory(evidence)
    require(
        inventory_after == inventory_before,
        "Evidence inventory changed during verification.",
    )

    report = {
        "verifier_id": VERIFIER_ID,
        "status": "passed",
        "verified_utc": utc_now(),
        "evidence_path": str(evidence),
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_role": EXPECTED_CAMPAIGN_ROLE,
        "config_sha256": CONFIG_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_source_commit": campaign["metadata"]["source"]["commit"],
        "expected_execution_source_commit": EXPECTED_EXECUTION_COMMIT,
        "verified_attempts": EXPECTED_ATTEMPTS,
        "verified_retained_pairs": EXPECTED_RETAINED_PAIRS,
        "verified_seed_streams": seed_streams_verified,
        "seed_material_injectivity_verified_from_frozen_coordinates": True,
        "time_index_draws_recomputed": time_index_draws_recomputed,
        "random_pair_datasets_regenerated": 0,
        "representations_refit": 0,
        "certificate_statistics_recomputed_independently": EXPECTED_ATTEMPTS,
        "certificate_bounds_recomputed_independently": EXPECTED_ATTEMPTS,
        "kahm_association_datasets_recomputed": EXPECTED_ATTEMPTS,
        "hard_label_datasets_recomputed_independently": EXPECTED_ATTEMPTS,
        "campaign_restored_representation_files_byte_compared": restored_file_count,
        "evidence_modified": False,
        "evidence_sha256sums_fingerprint": inventory_before["sha256sums_fingerprint"],
        "verified_file_count": inventory_before["verified_file_count"],
        "representation_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "representation_manifest_sha256": EXPECTED_REPRESENTATION_MANIFEST_SHA256,
        "rtol": RTOL,
        "atol": ATOL,
    }

    checksums = {
        "evidence_sha256sums_fingerprint": inventory_before["sha256sums_fingerprint"],
        "verified_file_count": inventory_before["verified_file_count"],
        "files": inventory_before["files"],
    }

    return report, checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output = args.output_dir.expanduser().resolve()

    if output.exists():
        print(
            f"Verification FAILED: output already exists: {output}",
            file=sys.stderr,
        )
        return 1

    try:
        source_before = verifier_provenance()

        report, checksums = verify_evidence(
            args.evidence
        )

        source_after = verifier_provenance()

        require(
            source_after == source_before,
            "Verifier source changed during verification.",
        )

        report["verifier_source"] = source_before

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.mkdir(
            exist_ok=False,
        )

        strict_json_write(
            output / "verification_report.json",
            report,
        )

        checksums["verification_report_sha256"] = sha256_file(
            output / "verification_report.json"
        )

        strict_json_write(
            output / "verification_checksums.json",
            checksums,
        )

    except BaseException as exc:
        print(
            "Verification FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
