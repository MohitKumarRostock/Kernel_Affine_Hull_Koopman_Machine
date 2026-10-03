#!/usr/bin/env python3
"""Independent numerical verification of retained dynamical pilot evidence.

This verifier does not regenerate dynamical trajectories and does not refit
KAHM representations.

It independently:

- validates the campaign SHA256 inventory and provenance;
- reconstructs the complete frozen 96-attempt schedule;
- reconstructs every eight-component SeedSequence namespace record;
- recomputes retained time-index draws from the recorded time-stream seeds;
- restores the frozen KAHM representations from the committed verified archive;
- recomputes KAHM associations on the retained raw current/successor states;
- independently recomputes nearest-center hard labels;
- independently recomputes f_hat and s_hat from their definitions;
- independently recomputes the frozen-predictor MSE/RMSE;
- independently evaluates every Eq. (8) certificate correction and lower bound;
- verifies all retained sufficient statistics and diagnostic ranges;
- verifies the evidence was not modified during verification.

The campaign evaluator, statistics helper, certificate helper, and dynamical pair
sampler are deliberately not called by this verifier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import log, sqrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

# Support direct execution as
#
#     python scripts/verify_dynamical_pilot_numerics.py
#
# Python otherwise places only the scripts/ directory at sys.path[0], which
# prevents imports of repository-root modules such as
# kahkm_dynamical_frozen_source.
_IMPORT_ROOT = Path(__file__).resolve().parents[1]

if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(_IMPORT_ROOT),
    )

from kahkm_dynamical_frozen_source import (
    EXPECTED_ARCHIVE_SHA256,
    restore_repository_frozen_representations,
)
from kahkm_frozen_representation import (
    load_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
)


ROOT = Path(__file__).resolve().parents[1]

VERIFIER_ID = "dynamical_pilot_independent_numerics_v1"

EXPECTED_EXECUTION_COMMIT = "4cda98e027cdc6b820e37f13869155e6f35956a8"

EXPECTED_RUNNER_ID = "dynamical_original_iid_pilot_runner_v1"

EXPECTED_CAMPAIGN_ID = "dynamical_original_iid_pilot_v1"
EXPECTED_CAMPAIGN_ROLE = "pilot"

CONFIG_PATH = (
    "reproduction/certificates/configs/"
    "dynamical_pilot_v1.json"
)

CONFIG_SHA256 = (
    "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1"
)

EXPECTED_ATTEMPTS = 96
EXPECTED_RETAINED_PAIRS = 86016
EXPECTED_EVIDENCE_INVENTORY_COUNT = 487

EXPECTED_SAMPLING_ID = "dynamical_certificate_sampling_v1"
EXPECTED_PAIR_LAW_ID = "independent_uniform_time_pair_v1"
EXPECTED_EVALUATION_ID = "frozen_kahkm_original_iid_evaluation_v1"
EXPECTED_REFERENCE_MAP_ID = "nearest_stored_kmeans_center_v1"

EXPECTED_SAMPLE_EVIDENCE_ID = "dynamical_sample_evidence_v1"
EXPECTED_EVALUATION_EVIDENCE_ID = "dynamical_evaluation_evidence_v1"

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

# Numerical comparisons are intentionally tight but not bitwise for KAHM
# association recomputation because threaded linear algebra may alter the final
# few binary64 bits without changing the mathematical result.
RTOL = 5e-12
ATOL = 5e-14

# These are the execution-time numerical source files used again by this
# verifier for frozen-model restoration/KAHM association evaluation. Their
# current bytes must still match the original execution commit.
EXECUTION_NUMERICAL_FILES = (
    "kahkm_dynamical_frozen_source.py",
    "kahkm_frozen_representation.py",
    "kernel_affine_hull_koopman_machines.py",
)


def utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ValueError(
            message
        )


def sha256_file(
    path: str | os.PathLike[str],
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def strict_json_load(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (
                (_ for _ in ()).throw(
                    ValueError(
                        f"Nonfinite JSON constant in {path}: {value}"
                    )
                )
            ),
        )


def strict_json_write(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            text
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )


def jsonl_records(
    path: Path,
) -> list[dict[str, Any]]:
    require(
        path.is_file()
        and not path.is_symlink(),
        f"Missing or unsafe JSONL file: {path.name}",
    )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            1,
        ):
            require(
                line.endswith(
                    "\n"
                ),
                f"{path.name}:{line_number} lacks newline termination.",
            )

            value = json.loads(
                line,
                parse_constant=lambda token: (
                    (_ for _ in ()).throw(
                        ValueError(
                            f"Nonfinite JSON constant in "
                            f"{path.name}:{line_number}: {token}"
                        )
                    )
                ),
            )

            require(
                isinstance(
                    value,
                    dict,
                ),
                f"{path.name}:{line_number} must contain an object.",
            )

            records.append(
                value
            )

    return records


def git_bytes(
    *args: str,
) -> bytes:
    return subprocess.check_output(
        [
            "git",
            *args,
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
    )


def verifier_provenance() -> dict[str, Any]:
    """Require clean committed verifier source and unchanged execution numerics."""
    repository = Path(
        git_bytes(
            "rev-parse",
            "--show-toplevel",
        )
        .decode()
        .strip()
    ).resolve()

    require(
        repository == ROOT,
        "Verifier is not running from the expected repository.",
    )

    status = (
        git_bytes(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        .decode()
    )

    require(
        status == "",
        "Verifier requires a clean worktree.",
    )

    commit = (
        git_bytes(
            "rev-parse",
            "HEAD",
        )
        .decode()
        .strip()
    )

    verifier_relative = (
        Path(__file__)
        .resolve()
        .relative_to(
            ROOT
        )
        .as_posix()
    )

    verifier_path = (
        ROOT
        / verifier_relative
    )

    actual_verifier_hash = (
        sha256_file(
            verifier_path
        )
    )

    try:
        committed_verifier = git_bytes(
            "show",
            f"{commit}:{verifier_relative}",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Verifier source is not committed at HEAD."
        ) from exc

    committed_verifier_hash = (
        hashlib.sha256(
            committed_verifier
        ).hexdigest()
    )

    require(
        actual_verifier_hash
        == committed_verifier_hash,
        "Verifier source bytes differ from HEAD.",
    )

    execution_hashes = {}

    for relative_name in (
        EXECUTION_NUMERICAL_FILES
    ):
        current_path = (
            ROOT
            / relative_name
        )

        require(
            current_path.is_file()
            and not current_path.is_symlink(),
            f"Missing/unsafe numerical source: {relative_name}",
        )

        current_hash = (
            sha256_file(
                current_path
            )
        )

        execution_bytes = git_bytes(
            "show",
            f"{EXPECTED_EXECUTION_COMMIT}:{relative_name}",
        )

        execution_hash = (
            hashlib.sha256(
                execution_bytes
            ).hexdigest()
        )

        require(
            current_hash
            == execution_hash,
            "Current numerical source differs from campaign execution "
            f"commit: {relative_name}",
        )

        execution_hashes[
            relative_name
        ] = current_hash

    return {
        "verifier_source_commit":
            commit,
        "verifier_source_sha256":
            actual_verifier_hash,
        "worktree_clean":
            True,
        "execution_numerical_source_sha256":
            execution_hashes,
    }


def safe_relative_path(
    name: str,
) -> Path:
    require(
        isinstance(
            name,
            str,
        )
        and bool(
            name
        ),
        "Checksum path must be a nonempty string.",
    )

    relative = Path(
        name
    )

    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative != Path(".")
        and "\\" not in name,
        f"Unsafe checksum path: {name!r}",
    )

    return relative


def check_inventory(
    folder: Path,
) -> dict[str, Any]:
    manifest = (
        folder
        / "SHA256SUMS"
    )

    require(
        manifest.is_file()
        and not manifest.is_symlink(),
        "Missing or unsafe SHA256SUMS.",
    )

    inventory: dict[
        str,
        str,
    ] = {}

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            1,
        ):
            require(
                line.endswith(
                    "\n"
                ),
                f"Malformed SHA256SUMS line {line_number}.",
            )

            stripped = (
                line[
                    :-1
                ]
            )

            parts = (
                stripped.split(
                    "  ",
                    1,
                )
            )

            require(
                len(
                    parts
                )
                == 2,
                f"Malformed SHA256SUMS line {line_number}.",
            )

            digest_text, name = (
                parts
            )

            require(
                len(
                    digest_text
                )
                == 64
                and all(
                    character
                    in "0123456789abcdef"
                    for character
                    in digest_text
                ),
                f"Malformed SHA256 digest on line {line_number}.",
            )

            relative = (
                safe_relative_path(
                    name
                )
            )

            canonical = (
                relative.as_posix()
            )

            require(
                canonical
                == name,
                f"Noncanonical SHA256SUMS path: {name!r}",
            )

            require(
                name
                != "SHA256SUMS",
                "SHA256SUMS must not inventory itself.",
            )

            require(
                name
                not in inventory,
                f"Duplicate SHA256SUMS entry: {name}",
            )

            path = (
                folder
                / relative
            )

            require(
                path.is_file()
                and not path.is_symlink(),
                f"Missing/unsafe inventoried file: {name}",
            )

            actual = (
                sha256_file(
                    path
                )
            )

            require(
                actual
                == digest_text,
                f"SHA256 mismatch for {name}",
            )

            inventory[
                name
            ] = actual

    require(
        len(
            inventory
        )
        == EXPECTED_EVIDENCE_INVENTORY_COUNT,
        "Unexpected evidence inventory count: "
        f"{len(inventory)} != {EXPECTED_EVIDENCE_INVENTORY_COUNT}",
    )

    actual_files = {
        path.relative_to(
            folder
        ).as_posix()
        for path
        in folder.rglob("*")
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS"
        )
    }

    require(
        actual_files
        == set(
            inventory
        ),
        "Evidence file inventory differs from SHA256SUMS.",
    )

    return {
        "sha256sums_fingerprint":
            sha256_file(
                manifest
            ),
        "verified_file_count":
            len(
                inventory
            ),
        "files":
            inventory,
    }


def load_config_from_evidence(
    evidence: Path,
) -> dict[str, Any]:
    campaign_spec = (
        evidence
        / "campaign_spec.json"
    )

    require(
        campaign_spec.is_file()
        and not campaign_spec.is_symlink(),
        "Missing/unsafe campaign_spec.json.",
    )

    raw = (
        campaign_spec.read_bytes()
    )

    require(
        hashlib.sha256(
            raw
        ).hexdigest()
        == CONFIG_SHA256,
        "Campaign specification SHA256 mismatch.",
    )

    repository_config = (
        ROOT
        / CONFIG_PATH
    ).read_bytes()

    require(
        raw
        == repository_config,
        "Campaign specification differs from committed frozen config.",
    )

    config = json.loads(
        raw.decode(
            "utf-8"
        ),
        parse_constant=lambda value: (
            (_ for _ in ()).throw(
                ValueError(
                    f"Nonfinite config constant: {value}"
                )
            )
        ),
    )

    require(
        isinstance(
            config,
            dict,
        ),
        "Campaign config must be an object.",
    )

    require(
        config.get(
            "campaign_id"
        )
        == EXPECTED_CAMPAIGN_ID,
        "Campaign ID mismatch.",
    )

    require(
        config.get(
            "campaign_role"
        )
        == EXPECTED_CAMPAIGN_ROLE,
        "Campaign role mismatch.",
    )

    require(
        config[
            "certificate"
        ][
            "implementation_id"
        ]
        == "original_iid_pair_certificate_v1",
        "Certificate configuration identifier mismatch.",
    )

    require(
        config[
            "pair_sampling"
        ][
            "implementation_id"
        ]
        == EXPECTED_PAIR_LAW_ID,
        "Pair-law identifier mismatch.",
    )

    require(
        config[
            "reference_class_map"
        ][
            "implementation_id"
        ]
        == EXPECTED_REFERENCE_MAP_ID,
        "Reference-map identifier mismatch.",
    )

    return config


def reconstruct_schedule(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pilot = config[
        "pilot_design"
    ]

    pair_sampling = config[
        "pair_sampling"
    ]

    seed_scheme = config[
        "seed_scheme"
    ]

    jobs = []

    for system in pilot[
        "systems"
    ]:
        system_key = int(
            seed_scheme[
                "system_keys"
            ][
                system
            ]
        )

        for mode in pair_sampling[
            "evaluation_modes"
        ]:
            mode_key = int(
                seed_scheme[
                    "mode_keys"
                ][
                    mode
                ]
            )

            for sample_size_raw in pilot[
                "sample_sizes"
            ]:
                sample_size = int(
                    sample_size_raw
                )

                for replicate_index in range(
                    int(
                        pilot[
                            "replicates_per_cell"
                        ]
                    )
                ):
                    jobs.append(
                        {
                            "attempt_id":
                                (
                                    f"s{system_key}"
                                    f"_q{mode_key}"
                                    f"_m{sample_size}"
                                    f"_r{replicate_index:03d}"
                                ),
                            "attempt_number":
                                1,
                            "system":
                                system,
                            "system_key":
                                system_key,
                            "evaluation_mode":
                                mode,
                            "mode_key":
                                mode_key,
                            "sample_size":
                                sample_size,
                            "replicate_index":
                                replicate_index,
                        }
                    )

    require(
        len(
            jobs
        )
        == EXPECTED_ATTEMPTS,
        "Reconstructed attempt count mismatch.",
    )

    require(
        len(
            {
                job[
                    "attempt_id"
                ]
                for job
                in jobs
            }
        )
        == EXPECTED_ATTEMPTS,
        "Reconstructed attempt IDs are not unique.",
    )

    require(
        sum(
            int(
                job[
                    "sample_size"
                ]
            )
            for job
            in jobs
        )
        == EXPECTED_RETAINED_PAIRS,
        "Reconstructed retained-pair total mismatch.",
    )

    return jobs


def verify_campaign_documents(
    evidence: Path,
    config: Mapping[str, Any],
    expected_schedule: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    metadata = strict_json_load(
        evidence
        / "campaign_metadata.json"
    )

    summary = strict_json_load(
        evidence
        / "campaign_summary.json"
    )

    source_after = strict_json_load(
        evidence
        / "source_after.json"
    )

    frozen_source = strict_json_load(
        evidence
        / "frozen_model_source.json"
    )

    require(
        isinstance(
            metadata,
            dict,
        ),
        "campaign_metadata.json must contain an object.",
    )

    require(
        isinstance(
            summary,
            dict,
        ),
        "campaign_summary.json must contain an object.",
    )

    require(
        metadata[
            "runner_id"
        ]
        == EXPECTED_RUNNER_ID,
        "Runner ID mismatch.",
    )

    require(
        metadata[
            "campaign_id"
        ]
        == EXPECTED_CAMPAIGN_ID,
        "Metadata campaign ID mismatch.",
    )

    require(
        metadata[
            "campaign_role"
        ]
        == EXPECTED_CAMPAIGN_ROLE,
        "Metadata campaign role mismatch.",
    )

    require(
        metadata[
            "config_sha256"
        ]
        == CONFIG_SHA256,
        "Metadata config SHA256 mismatch.",
    )

    require(
        metadata[
            "representation_archive_sha256"
        ]
        == EXPECTED_ARCHIVE_SHA256,
        "Metadata representation archive SHA mismatch.",
    )

    require(
        metadata[
            "expected_attempts"
        ]
        == EXPECTED_ATTEMPTS,
        "Metadata expected-attempt count mismatch.",
    )

    require(
        metadata[
            "expected_retained_pairs"
        ]
        == EXPECTED_RETAINED_PAIRS,
        "Metadata retained-pair count mismatch.",
    )

    execution_source = (
        metadata[
            "source"
        ]
    )

    require(
        execution_source[
            "execution_source_commit"
        ]
        == EXPECTED_EXECUTION_COMMIT,
        "Campaign execution commit mismatch.",
    )

    require(
        execution_source[
            "git_status_porcelain"
        ]
        == "",
        "Campaign execution source was not clean.",
    )

    require(
        source_after
        == execution_source,
        "Campaign source-after snapshot differs from source-before snapshot.",
    )

    expected_summary = {
        "status":
            "completed",
        "exit_code":
            0,
        "planned":
            EXPECTED_ATTEMPTS,
        "planned_retained_pairs":
            EXPECTED_RETAINED_PAIRS,
        "started":
            EXPECTED_ATTEMPTS,
        "sampled":
            EXPECTED_ATTEMPTS,
        "evaluated":
            EXPECTED_ATTEMPTS,
        "succeeded":
            EXPECTED_ATTEMPTS,
        "failed":
            0,
        "failed_sampling":
            0,
        "failed_evaluation":
            0,
        "not_started":
            0,
        "started_without_terminal_event":
            0,
        "sampled_without_success":
            0,
        "source_unchanged":
            True,
        "active_attempt_id":
            None,
        "independent_evidence_verification":
            "not_yet_performed",
    }

    for key, expected in (
        expected_summary.items()
    ):
        require(
            summary.get(
                key
            )
            == expected,
            f"Campaign summary mismatch for {key}.",
        )

    schedule = jsonl_records(
        evidence
        / "schedule.jsonl"
    )

    require(
        schedule
        == list(
            expected_schedule
        ),
        "Saved schedule differs from independently reconstructed schedule.",
    )

    restoration = (
        frozen_source[
            "restoration"
        ]
    )

    require(
        restoration[
            "archive_sha256"
        ]
        == EXPECTED_ARCHIVE_SHA256,
        "Frozen-model restoration archive SHA mismatch.",
    )

    require(
        restoration[
            "archived_verification_status"
        ]
        == "passed",
        "Archived frozen-model verification did not pass.",
    )

    return {
        "metadata":
            metadata,
        "summary":
            summary,
        "frozen_model_source":
            frozen_source,
    }


def index_unique(
    records: Sequence[
        Mapping[str, Any]
    ],
    *,
    key: str,
    expected_count: int,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed = {}

    for record in records:
        identifier = record.get(
            key
        )

        require(
            isinstance(
                identifier,
                str,
            )
            and identifier,
            f"{label} has invalid {key}.",
        )

        require(
            identifier
            not in indexed,
            f"Duplicate {label} {key}: {identifier}",
        )

        indexed[
            identifier
        ] = record

    require(
        len(
            indexed
        )
        == expected_count,
        f"{label} count mismatch.",
    )

    return indexed


def verify_attempt_event_log(
    evidence: Path,
    expected_schedule: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    events = jsonl_records(
        evidence
        / "attempts.jsonl"
    )

    require(
        len(
            events
        )
        == 2
        * EXPECTED_ATTEMPTS,
        "Attempt-event count mismatch.",
    )

    for index, job in enumerate(
        expected_schedule
    ):
        started = events[
            2
            * index
        ]

        completed = events[
            2
            * index
            + 1
        ]

        identifier = job[
            "attempt_id"
        ]

        require(
            started.get(
                "attempt_id"
            )
            == identifier
            and started.get(
                "event"
            )
            == "started",
            f"Missing/incorrect started event for {identifier}.",
        )

        require(
            completed.get(
                "attempt_id"
            )
            == identifier
            and completed.get(
                "event"
            )
            == "completed",
            f"Missing/incorrect completed event for {identifier}.",
        )

        require(
            float(
                completed[
                    "sampling_seconds"
                ]
            )
            >= 0.0,
            f"Negative sampling runtime for {identifier}.",
        )

        require(
            float(
                completed[
                    "evaluation_seconds"
                ]
            )
            >= 0.0,
            f"Negative evaluation runtime for {identifier}.",
        )


def load_npz(
    path: Path,
    expected_members: set[str],
) -> dict[str, np.ndarray]:
    require(
        path.is_file()
        and not path.is_symlink(),
        f"Missing/unsafe NPZ file: {path}",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        require(
            set(
                archive.files
            )
            == expected_members,
            f"Unexpected NPZ member inventory in {path.name}.",
        )

        return {
            name:
                np.array(
                    archive[
                        name
                    ],
                    copy=True,
                )
            for name
            in sorted(
                expected_members
            )
        }


def close_scalar(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    a = float(
        actual
    )

    e = float(
        expected
    )

    require(
        np.isfinite(
            a
        )
        and np.isfinite(
            e
        ),
        f"Nonfinite scalar comparison for {label}.",
    )

    require(
        bool(
            np.isclose(
                a,
                e,
                rtol=RTOL,
                atol=ATOL,
            )
        ),
        f"{label} mismatch: saved={a!r}, recomputed={e!r}",
    )


def close_array(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    a = np.asarray(
        actual
    )

    e = np.asarray(
        expected
    )

    require(
        a.shape
        == e.shape,
        f"{label} shape mismatch: {a.shape} != {e.shape}",
    )

    require(
        bool(
            np.allclose(
                a,
                e,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=False,
            )
        ),
        f"{label} numerical mismatch.",
    )


def exact_array(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    a = np.asarray(
        actual
    )

    e = np.asarray(
        expected
    )

    require(
        a.shape
        == e.shape
        and bool(
            np.array_equal(
                a,
                e,
            )
        ),
        f"{label} exact-array mismatch.",
    )


def generated_seed_from_material(
    material: tuple[int, ...],
) -> tuple[
    tuple[int, int],
    int,
]:
    sequence = np.random.SeedSequence(
        entropy=
            material
    )

    words = sequence.generate_state(
        2,
        dtype=np.uint32,
    )

    pair = (
        int(
            words[
                0
            ]
        ),
        int(
            words[
                1
            ]
        ),
    )

    generated = (
        pair[
            0
        ]
        | (
            pair[
                1
            ]
            << 32
        )
    )

    return (
        pair,
        generated,
    )


def expected_seed_material(
    *,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    pair_index: int,
    stream_name: str,
) -> tuple[int, ...]:
    scheme = (
        config[
            "seed_scheme"
        ]
    )

    return (
        int(
            scheme[
                "campaign_key"
            ]
        ),
        int(
            job[
                "system_key"
            ]
        ),
        int(
            job[
                "mode_key"
            ]
        ),
        int(
            job[
                "sample_size"
            ]
        ),
        int(
            job[
                "replicate_index"
            ]
        ),
        int(
            pair_index
        ),
        int(
            scheme[
                "stream_keys"
            ][
                stream_name
            ]
        ),
        int(
            scheme[
                "root_seed"
            ]
        ),
    )


def verify_sample(
    *,
    attempt_dir: Path,
    job: Mapping[str, Any],
    config: Mapping[str, Any],
    global_materials: set[tuple[int, ...]],
    global_generated_seeds: set[int],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    int,
]:
    metadata_path = (
        attempt_dir
        / "sample.json"
    )

    arrays_path = (
        attempt_dir
        / "sample_arrays.npz"
    )

    metadata = strict_json_load(
        metadata_path
    )

    require(
        isinstance(
            metadata,
            dict,
        ),
        "sample.json must contain an object.",
    )

    require(
        metadata[
            "sample_evidence_id"
        ]
        == EXPECTED_SAMPLE_EVIDENCE_ID,
        "Sample evidence ID mismatch.",
    )

    require(
        metadata[
            "sampling_id"
        ]
        == EXPECTED_SAMPLING_ID,
        "Sampling ID mismatch.",
    )

    require(
        metadata[
            "pair_law_id"
        ]
        == EXPECTED_PAIR_LAW_ID,
        "Pair law ID mismatch in sample evidence.",
    )

    for key in (
        "system",
        "evaluation_mode",
        "sample_size",
        "replicate_index",
    ):
        require(
            metadata[
                key
            ]
            == job[
                key
            ],
            f"Sample metadata mismatch for {key}.",
        )

    system_design = (
        config[
            "frozen_representation"
        ][
            "systems"
        ][
            job[
                "system"
            ]
        ]
    )

    expected_state_dimension = len(
        system_design[
            "initial_state_center"
        ]
    )

    require(
        int(
            metadata[
                "state_dimension"
            ]
        )
        == expected_state_dimension,
        "Sample state_dimension mismatch.",
    )

    require(
        float(
            metadata[
                "dt"
            ]
        )
        == float(
            system_design[
                "dt"
            ]
        ),
        "Sample dt mismatch.",
    )

    require(
        int(
            metadata[
                "horizon_steps"
            ]
        )
        == int(
            config[
                "pair_sampling"
            ][
                "horizon_steps"
            ]
        ),
        "Sample horizon_steps mismatch.",
    )

    require(
        float(
            metadata[
                "vanderpol_mu"
            ]
        )
        == float(
            system_design[
                "vanderpol_mu"
            ]
        ),
        "Sample vanderpol_mu mismatch.",
    )

    require(
        metadata[
            "evaluation_performed"
        ]
        is False,
        "Sample metadata must record evaluation_performed=false.",
    )

    require(
        metadata[
            "array_file"
        ]
        == "sample_arrays.npz",
        "Unexpected sample array filename.",
    )

    require(
        set(
            metadata[
                "array_members"
            ]
        )
        == SAMPLE_ARRAY_MEMBERS,
        "Sample array-member inventory mismatch.",
    )

    require(
        sha256_file(
            arrays_path
        )
        == metadata[
            "array_sha256"
        ],
        "Sample array SHA256 mismatch.",
    )

    arrays = load_npz(
        arrays_path,
        SAMPLE_ARRAY_MEMBERS,
    )

    M = int(
        job[
            "sample_size"
        ]
    )

    current = np.asarray(
        arrays[
            "current_states"
        ],
        dtype=np.float64,
    )

    successor = np.asarray(
        arrays[
            "successor_states"
        ],
        dtype=np.float64,
    )

    require(
        current.shape
        == (
            2,
            M,
        )
        and successor.shape
        == (
            2,
            M,
        ),
        "Physical state-array shape mismatch.",
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    current
                )
            )
        )
        and bool(
            np.all(
                np.isfinite(
                    successor
                )
            )
        ),
        "Physical state arrays contain nonfinite values.",
    )

    time_indices = np.asarray(
        arrays[
            "time_indices"
        ]
    )

    require(
        time_indices.shape
        == (
            M,
        )
        and time_indices.dtype.kind
        in "iu",
        "time_indices have invalid shape or dtype.",
    )

    horizon = int(
        config[
            "pair_sampling"
        ][
            "horizon_steps"
        ]
    )

    require(
        bool(
            np.all(
                time_indices
                >= 0
            )
        )
        and bool(
            np.all(
                time_indices
                < horizon
            )
        ),
        "time_indices lie outside frozen horizon.",
    )

    recomputed_time_draws = 0

    for stream_name in (
        "trajectory_seed",
        "time_seed",
    ):
        materials = np.asarray(
            arrays[
                f"{stream_name}_material"
            ]
        )

        words = np.asarray(
            arrays[
                f"{stream_name}_uint32_words"
            ]
        )

        seeds = np.asarray(
            arrays[
                f"{stream_name}_generated_seed"
            ]
        )

        require(
            materials.shape
            == (
                M,
                8,
            ),
            f"{stream_name} seed-material shape mismatch.",
        )

        require(
            words.shape
            == (
                M,
                2,
            ),
            f"{stream_name} uint32-word shape mismatch.",
        )

        require(
            seeds.shape
            == (
                M,
            ),
            f"{stream_name} generated-seed shape mismatch.",
        )

        for pair_index in range(
            M
        ):
            material = (
                expected_seed_material(
                    config=
                        config,
                    job=
                        job,
                    pair_index=
                        pair_index,
                    stream_name=
                        stream_name,
                )
            )

            exact_array(
                materials[
                    pair_index
                ],
                np.asarray(
                    material,
                    dtype=np.uint64,
                ),
                label=(
                    f"{job['attempt_id']} "
                    f"{stream_name} material[{pair_index}]"
                ),
            )

            expected_words, expected_seed = (
                generated_seed_from_material(
                    material
                )
            )

            exact_array(
                words[
                    pair_index
                ],
                np.asarray(
                    expected_words,
                    dtype=np.uint32,
                ),
                label=(
                    f"{job['attempt_id']} "
                    f"{stream_name} words[{pair_index}]"
                ),
            )

            require(
                int(
                    seeds[
                        pair_index
                    ]
                )
                == expected_seed,
                f"{job['attempt_id']} {stream_name} "
                f"generated seed mismatch at pair {pair_index}.",
            )

            require(
                material
                not in global_materials,
                "Duplicate seed material across frozen pilot schedule.",
            )

            require(
                expected_seed
                not in global_generated_seeds,
                "Generated 64-bit seed collision across frozen pilot schedule.",
            )

            require(
                expected_seed
                not in (
                    0,
                    1,
                    2,
                ),
                "Generated evaluation seed collides with literal training seed.",
            )

            global_materials.add(
                material
            )

            global_generated_seeds.add(
                expected_seed
            )

            if (
                stream_name
                == "time_seed"
            ):
                expected_time = int(
                    np.random.default_rng(
                        expected_seed
                    ).integers(
                        horizon
                    )
                )

                require(
                    int(
                        time_indices[
                            pair_index
                        ]
                    )
                    == expected_time,
                    f"{job['attempt_id']} time-index draw mismatch "
                    f"at pair {pair_index}.",
                )

                recomputed_time_draws += 1

    return (
        metadata,
        arrays,
        recomputed_time_draws,
    )


def norm_budgets(
    config: Mapping[str, Any],
    spectral_norm: float,
) -> tuple[float, ...]:
    design = (
        config[
            "norm_budgets"
        ]
    )

    values = [
        float(
            value
        )
        for value
        in design[
            "fixed_constants"
        ]
    ]

    for name in design[
        "training_derived"
    ]:
        if name == "nlms_spectral_norm":
            values.append(
                float(
                    spectral_norm
                )
            )

        elif name == "twice_nlms_spectral_norm":
            values.append(
                float(
                    2.0
                    * spectral_norm
                )
            )

        else:
            raise ValueError(
                f"Unknown norm-budget recipe: {name!r}"
            )

    require(
        design[
            "deduplicate_exact_binary64"
        ]
        is True
        and design[
            "sort_ascending"
        ]
        is True,
        "Unexpected frozen norm-budget policy.",
    )

    return tuple(
        sorted(
            set(
                values
            )
        )
    )


def independent_statistics(
    phi_rows: np.ndarray,
    successor_rows: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    current = np.asarray(
        phi_rows,
        dtype=np.float64,
    )

    successor = np.asarray(
        successor_rows,
        dtype=np.float64,
    )

    classes = np.asarray(
        labels,
        dtype=np.int64,
    )

    require(
        current.ndim
        == 2
        and successor.shape
        == current.shape,
        "Independent statistics require matching (M,C) arrays.",
    )

    M, C = (
        current.shape
    )

    require(
        classes.shape
        == (
            M,
        ),
        "Independent-statistics label shape mismatch.",
    )

    current_error = float(
        np.max(
            np.abs(
                np.sum(
                    current,
                    axis=1,
                    dtype=np.float64,
                )
                - 1.0
            )
        )
    )

    successor_error = float(
        np.max(
            np.abs(
                np.sum(
                    successor,
                    axis=1,
                    dtype=np.float64,
                )
                - 1.0
            )
        )
    )

    counts = np.bincount(
        classes,
        minlength=C,
    )

    sums = np.zeros(
        (
            C,
            C,
        ),
        dtype=np.float64,
    )

    np.add.at(
        sums,
        classes,
        successor,
    )

    means = np.full(
        (
            C,
            C,
        ),
        1.0
        / C,
        dtype=np.float64,
    )

    occupied = (
        counts
        > 0
    )

    means[
        occupied
    ] = (
        sums[
            occupied
        ]
        / counts[
            occupied,
            None,
        ]
    )

    residuals = (
        successor
        - means[
            classes
        ]
    )

    row_sse = np.sum(
        residuals
        * residuals,
        axis=1,
        dtype=np.float64,
    )

    class_sse = np.bincount(
        classes,
        weights=
            row_sse,
        minlength=C,
    )

    assignment_losses = (
        1.0
        - current[
            np.arange(
                M
            ),
            classes,
        ]
    ) ** 2

    f_hat = float(
        np.mean(
            assignment_losses,
            dtype=np.float64,
        )
    )

    s_hat = float(
        np.sum(
            class_sse,
            dtype=np.float64,
        )
        / M
    )

    return {
        "n_pairs":
            int(
                M
            ),
        "n_classes":
            int(
                C
            ),
        "f_hat":
            f_hat,
        "s_hat":
            s_hat,
        "class_counts":
            counts,
        "class_successor_means":
            means,
        "class_successor_sse":
            class_sse,
        "current_max_row_sum_error":
            current_error,
        "successor_max_row_sum_error":
            successor_error,
    }


def independent_certificate(
    *,
    f_hat: float,
    s_hat: float,
    n_pairs: int,
    kappa: float,
    delta: float,
) -> dict[str, float]:
    M = int(
        n_pairs
    )

    f = float(
        f_hat
    )

    s = float(
        s_hat
    )

    k = float(
        kappa
    )

    d = float(
        delta
    )

    require(
        M > 0,
        "Certificate pair count must be positive.",
    )

    require(
        0.0
        <= f
        <= 1.0
        and 0.0
        <= s
        <= 1.0,
        "Certificate statistics lie outside [0,1].",
    )

    require(
        k
        >= 0.0,
        "Certificate kappa is negative.",
    )

    require(
        0.0
        < d
        < 1.0,
        "Certificate delta lies outside (0,1).",
    )

    r = (
        log(
            2.0
        )
        - log(
            d
        )
    ) / M

    F = min(
        1.0,
        f
        + r
        + sqrt(
            r
            * r
            + 2.0
            * r
            * f
        ),
    )

    V = max(
        0.0,
        s
        - sqrt(
            2.0
            * r
        ),
    )

    L = max(
        0.0,
        sqrt(
            V
        )
        - k
        * sqrt(
            2.0
            * F
        ),
    )

    return {
        "n_pairs":
            M,
        "kappa":
            k,
        "delta":
            d,
        "f_hat":
            f,
        "s_hat":
            s,
        "r_delta":
            r,
        "F_delta":
            F,
        "V_delta":
            V,
        "L_kappa_delta":
            L,
    }


def nearest_center_labels(
    centers: np.ndarray,
    current_states: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    stored = np.asarray(
        centers,
        dtype=np.float64,
    )

    current = np.asarray(
        current_states,
        dtype=np.float64,
    )

    require(
        stored.ndim
        == 2
        and current.ndim
        == 2,
        "Nearest-center inputs must be matrices.",
    )

    require(
        stored.shape[
            0
        ]
        == current.shape[
            0
        ],
        "Nearest-center state dimension mismatch.",
    )

    difference = (
        current.T[
            :,
            None,
            :,
        ]
        - stored.T[
            None,
            :,
            :,
        ]
    )

    squared = np.sum(
        difference
        * difference,
        axis=2,
        dtype=np.float64,
    )

    labels = np.argmin(
        squared,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    distances = squared[
        np.arange(
            current.shape[
                1
            ]
        ),
        labels,
    ]

    return (
        labels,
        np.asarray(
            distances,
            dtype=np.float64,
        ),
    )


def verify_evaluation(
    *,
    attempt_dir: Path,
    job: Mapping[str, Any],
    sample_arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    model,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    metadata_path = (
        attempt_dir
        / "evaluation.json"
    )

    arrays_path = (
        attempt_dir
        / "evaluation_arrays.npz"
    )

    metadata = strict_json_load(
        metadata_path
    )

    require(
        isinstance(
            metadata,
            dict,
        ),
        "evaluation.json must contain an object.",
    )

    require(
        metadata[
            "evaluation_evidence_id"
        ]
        == EXPECTED_EVALUATION_EVIDENCE_ID,
        "Evaluation evidence ID mismatch.",
    )

    require(
        metadata[
            "evaluation_id"
        ]
        == EXPECTED_EVALUATION_ID,
        "Evaluation ID mismatch.",
    )

    require(
        metadata[
            "reference_class_map_id"
        ]
        == EXPECTED_REFERENCE_MAP_ID,
        "Reference-class map ID mismatch.",
    )

    for key in (
        "system",
        "evaluation_mode",
        "sample_size",
        "replicate_index",
    ):
        require(
            metadata[
                key
            ]
            == job[
                key
            ],
            f"Evaluation metadata mismatch for {key}.",
        )

    require(
        metadata[
            "sample_evidence_present_before_evaluation_write"
        ]
        is True,
        "Evaluation metadata must confirm sample evidence was present first.",
    )

    expected_pairs = int(
        job[
            "sample_size"
        ]
    )

    require(
        int(
            metadata[
                "n_pairs"
            ]
        )
        == expected_pairs,
        "Evaluation pair count mismatch.",
    )

    current_for_design = np.asarray(
        sample_arrays[
            "current_states"
        ]
    )

    require(
        current_for_design.ndim
        == 2,
        "Sample current-state array must be two-dimensional.",
    )

    expected_state_dimension = int(
        current_for_design.shape[
            0
        ]
    )

    require(
        int(
            metadata[
                "state_dimension"
            ]
        )
        == expected_state_dimension,
        "Evaluation state_dimension mismatch.",
    )

    centers_for_design = np.asarray(
        model.abstraction_model[
            "cluster_centers"
        ]
    )

    require(
        centers_for_design.ndim
        == 2,
        "Frozen cluster centers must be two-dimensional.",
    )

    expected_classes = int(
        centers_for_design.shape[
            1
        ]
    )

    require(
        int(
            metadata[
                "n_classes"
            ]
        )
        == expected_classes,
        "Evaluation class count mismatch.",
    )

    require(
        float(
            metadata[
                "delta"
            ]
        )
        == float(
            config[
                "certificate"
            ][
                "delta"
            ]
        ),
        "Evaluation delta mismatch.",
    )

    require(
        metadata[
            "array_file"
        ]
        == "evaluation_arrays.npz",
        "Unexpected evaluation array filename.",
    )

    require(
        set(
            metadata[
                "array_members"
            ]
        )
        == EVALUATION_ARRAY_MEMBERS,
        "Evaluation array-member inventory mismatch.",
    )

    require(
        sha256_file(
            arrays_path
        )
        == metadata[
            "array_sha256"
        ],
        "Evaluation-array SHA256 mismatch.",
    )

    arrays = load_npz(
        arrays_path,
        EVALUATION_ARRAY_MEMBERS,
    )

    current = np.asarray(
        sample_arrays[
            "current_states"
        ],
        dtype=np.float64,
    )

    successor = np.asarray(
        sample_arrays[
            "successor_states"
        ],
        dtype=np.float64,
    )

    phi_columns = np.asarray(
        kahm_associations(
            dict(
                model.abstraction_model
            ),
            current,
            omega=
                float(
                    model.omega
                ),
            tau=
                float(
                    model.tau
                ),
            n_jobs=
                int(
                    config[
                        "frozen_representation"
                    ][
                        "n_jobs"
                    ]
                ),
            batch_size=
                int(
                    config[
                        "frozen_representation"
                    ][
                        "batch_size"
                    ]
                ),
            show_progress=False,
        ),
        dtype=np.float64,
    )

    successor_columns = np.asarray(
        kahm_associations(
            dict(
                model.abstraction_model
            ),
            successor,
            omega=
                float(
                    model.omega
                ),
            tau=
                float(
                    model.tau
                ),
            n_jobs=
                int(
                    config[
                        "frozen_representation"
                    ][
                        "n_jobs"
                    ]
                ),
            batch_size=
                int(
                    config[
                        "frozen_representation"
                    ][
                        "batch_size"
                    ]
                ),
            show_progress=False,
        ),
        dtype=np.float64,
    )

    phi_rows = np.asarray(
        phi_columns.T,
        dtype=np.float64,
    )

    successor_rows = np.asarray(
        successor_columns.T,
        dtype=np.float64,
    )

    close_array(
        arrays[
            "phi_rows"
        ],
        phi_rows,
        label=(
            f"{job['attempt_id']} retained Phi(X)"
        ),
    )

    close_array(
        arrays[
            "successor_rows"
        ],
        successor_rows,
        label=(
            f"{job['attempt_id']} retained Phi(X+)"
        ),
    )

    centers = np.asarray(
        model.abstraction_model[
            "cluster_centers"
        ],
        dtype=np.float64,
    )

    labels, distances = (
        nearest_center_labels(
            centers,
            current,
        )
    )

    exact_array(
        arrays[
            "labels"
        ],
        labels,
        label=(
            f"{job['attempt_id']} hard labels"
        ),
    )

    close_array(
        arrays[
            "selected_center_squared_distances"
        ],
        distances,
        label=(
            f"{job['attempt_id']} selected-center distances"
        ),
    )

    statistics = independent_statistics(
        phi_rows,
        successor_rows,
        labels,
    )

    exact_array(
        arrays[
            "class_counts"
        ],
        statistics[
            "class_counts"
        ],
        label=(
            f"{job['attempt_id']} class counts"
        ),
    )

    close_array(
        arrays[
            "class_successor_sse"
        ],
        statistics[
            "class_successor_sse"
        ],
        label=(
            f"{job['attempt_id']} class successor SSE"
        ),
    )

    close_array(
        arrays[
            "class_successor_means"
        ],
        statistics[
            "class_successor_means"
        ],
        label=(
            f"{job['attempt_id']} class successor means"
        ),
    )

    saved_statistics = (
        metadata[
            "statistics"
        ]
    )

    for key in (
        "f_hat",
        "s_hat",
        "current_max_row_sum_error",
        "successor_max_row_sum_error",
    ):
        close_scalar(
            saved_statistics[
                key
            ],
            statistics[
                key
            ],
            label=(
                f"{job['attempt_id']} {key}"
            ),
        )

    require(
        saved_statistics[
            "simplex_sum_atol"
        ]
        == 1e-12,
        "Unexpected saved simplex_sum_atol.",
    )

    B = np.asarray(
        model.B,
        dtype=np.float64,
    )

    spectral_norm = float(
        np.linalg.norm(
            B,
            ord=2,
        )
    )

    prediction_columns = (
        B.T
        @ phi_columns
    )

    residual_columns = (
        prediction_columns
        - successor_columns
    )

    per_pair_squared_error = (
        np.sum(
            residual_columns
            * residual_columns,
            axis=0,
            dtype=np.float64,
        )
    )

    mse = float(
        np.mean(
            per_pair_squared_error,
            dtype=np.float64,
        )
    )

    rmse = sqrt(
        mse
    )

    close_scalar(
        metadata[
            "frozen_predictor_spectral_norm"
        ],
        spectral_norm,
        label=(
            f"{job['attempt_id']} predictor spectral norm"
        ),
    )

    close_scalar(
        metadata[
            "frozen_predictor_mse"
        ],
        mse,
        label=(
            f"{job['attempt_id']} predictor MSE"
        ),
    )

    close_scalar(
        metadata[
            "frozen_predictor_rmse"
        ],
        rmse,
        label=(
            f"{job['attempt_id']} predictor RMSE"
        ),
    )

    expected_kappas = (
        norm_budgets(
            config,
            spectral_norm,
        )
    )

    saved_budgets = (
        metadata[
            "budget_certificates"
        ]
    )

    require(
        len(
            saved_budgets
        )
        == len(
            expected_kappas
        ),
        "Budget-certificate count mismatch.",
    )

    delta = float(
        config[
            "certificate"
        ][
            "delta"
        ]
    )

    for saved, kappa in zip(
        saved_budgets,
        expected_kappas,
        strict=True,
    ):
        close_scalar(
            saved[
                "kappa"
            ],
            kappa,
            label=(
                f"{job['attempt_id']} kappa"
            ),
        )

        certificate = independent_certificate(
            f_hat=
                statistics[
                    "f_hat"
                ],
            s_hat=
                statistics[
                    "s_hat"
                ],
            n_pairs=
                statistics[
                    "n_pairs"
                ],
            kappa=
                kappa,
            delta=
                delta,
        )

        saved_certificate = (
            saved[
                "certificate"
            ]
        )

        require(
            int(
                saved_certificate[
                    "n_pairs"
                ]
            )
            == certificate[
                "n_pairs"
            ],
            "Saved certificate pair count mismatch.",
        )

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
                saved_certificate[
                    key
                ],
                certificate[
                    key
                ],
                label=(
                    f"{job['attempt_id']} certificate {key}"
                ),
            )

        feasibility_atol = (
            64.0
            * np.finfo(
                np.float64
            ).eps
            * max(
                1.0,
                spectral_norm,
                kappa,
            )
        )

        within_budget = bool(
            spectral_norm
            <= kappa
            + feasibility_atol
        )

        require(
            saved[
                "frozen_predictor_within_budget"
            ]
            is within_budget,
            f"{job['attempt_id']} predictor budget-membership mismatch.",
        )

        close_scalar(
            saved[
                "frozen_predictor_rmse_minus_bound"
            ],
            (
                rmse
                - certificate[
                    "L_kappa_delta"
                ]
            ),
            label=(
                f"{job['attempt_id']} RMSE-minus-bound"
            ),
        )

    ranges = (
        metadata[
            "coordinate_ranges"
        ]
    )

    close_scalar(
        ranges[
            "phi_min"
        ],
        np.min(
            phi_columns
        ),
        label=(
            f"{job['attempt_id']} phi_min"
        ),
    )

    close_scalar(
        ranges[
            "phi_max"
        ],
        np.max(
            phi_columns
        ),
        label=(
            f"{job['attempt_id']} phi_max"
        ),
    )

    close_scalar(
        ranges[
            "successor_min"
        ],
        np.min(
            successor_columns
        ),
        label=(
            f"{job['attempt_id']} successor_min"
        ),
    )

    close_scalar(
        ranges[
            "successor_max"
        ],
        np.max(
            successor_columns
        ),
        label=(
            f"{job['attempt_id']} successor_max"
        ),
    )

    state_ranges = (
        metadata[
            "state_ranges"
        ]
    )

    close_array(
        state_ranges[
            "current_min"
        ],
        np.min(
            current,
            axis=1,
        ),
        label=(
            f"{job['attempt_id']} current-state min"
        ),
    )

    close_array(
        state_ranges[
            "current_max"
        ],
        np.max(
            current,
            axis=1,
        ),
        label=(
            f"{job['attempt_id']} current-state max"
        ),
    )

    close_array(
        state_ranges[
            "successor_min"
        ],
        np.min(
            successor,
            axis=1,
        ),
        label=(
            f"{job['attempt_id']} successor-state min"
        ),
    )

    close_array(
        state_ranges[
            "successor_max"
        ],
        np.max(
            successor,
            axis=1,
        ),
        label=(
            f"{job['attempt_id']} successor-state max"
        ),
    )

    return (
        metadata,
        arrays,
    )


def verify_evidence(
    evidence: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    evidence = (
        evidence
        .expanduser()
        .resolve()
    )

    require(
        evidence.is_dir(),
        "Evidence directory does not exist.",
    )

    require(
        not evidence.is_symlink(),
        "Evidence directory must not be a symlink.",
    )

    inventory_before = (
        check_inventory(
            evidence
        )
    )

    config = (
        load_config_from_evidence(
            evidence
        )
    )

    expected_schedule = (
        reconstruct_schedule(
            config
        )
    )

    campaign = (
        verify_campaign_documents(
            evidence,
            config,
            expected_schedule,
        )
    )

    verify_attempt_event_log(
        evidence,
        expected_schedule,
    )

    sample_index = index_unique(
        jsonl_records(
            evidence
            / "samples.jsonl"
        ),
        key=
            "attempt_id",
        expected_count=
            EXPECTED_ATTEMPTS,
        label=
            "sample index",
    )

    result_index = index_unique(
        jsonl_records(
            evidence
            / "results.jsonl"
        ),
        key=
            "attempt_id",
        expected_count=
            EXPECTED_ATTEMPTS,
        label=
            "result index",
    )

    global_materials: set[
        tuple[int, ...]
    ] = set()

    global_generated_seeds: set[
        int
    ] = set()

    time_index_draws_recomputed = 0
    evaluated_pairs = 0

    with tempfile.TemporaryDirectory(
        prefix=
            "kahkm-dynamical-verifier-"
    ) as temporary_parent:
        restoration_dir = (
            Path(
                temporary_parent
            )
            / "restored"
        )

        restored = (
            restore_repository_frozen_representations(
                repository_root=
                    ROOT,
                restoration_dir=
                    restoration_dir,
            )
        )

        require(
            restored.archive_sha256
            == EXPECTED_ARCHIVE_SHA256,
            "Verifier-restored model archive fingerprint mismatch.",
        )

        model_paths = {
            "duffing":
                restored.duffing_artifact_dir,
            "vanderpol":
                restored.vanderpol_artifact_dir,
        }

        models = {
            system:
                load_frozen_representation(
                    path,
                    verify_hashes=True,
                )
            for system, path
            in model_paths.items()
        }

        for job in (
            expected_schedule
        ):
            identifier = (
                job[
                    "attempt_id"
                ]
            )

            attempt_relative = (
                Path(
                    "attempt_data"
                )
                / identifier
            )

            attempt_dir = (
                evidence
                / attempt_relative
            )

            require(
                attempt_dir.is_dir()
                and not attempt_dir.is_symlink(),
                f"Missing/unsafe attempt directory: {identifier}",
            )

            sample_record = (
                sample_index[
                    identifier
                ]
            )

            result_record = (
                result_index[
                    identifier
                ]
            )

            require(
                sample_record[
                    "attempt_relative_dir"
                ]
                == attempt_relative.as_posix(),
                f"Sample index attempt path mismatch: {identifier}",
            )

            require(
                result_record[
                    "attempt_relative_dir"
                ]
                == attempt_relative.as_posix(),
                f"Result index attempt path mismatch: {identifier}",
            )

            (
                sample_metadata,
                sample_arrays,
                time_draw_count,
            ) = verify_sample(
                attempt_dir=
                    attempt_dir,
                job=
                    job,
                config=
                    config,
                global_materials=
                    global_materials,
                global_generated_seeds=
                    global_generated_seeds,
            )

            time_index_draws_recomputed += (
                time_draw_count
            )

            require(
                sample_record[
                    "array_sha256"
                ]
                == sample_metadata[
                    "array_sha256"
                ],
                f"Sample index array SHA mismatch: {identifier}",
            )

            for key in (
                "system",
                "evaluation_mode",
                "sample_size",
                "replicate_index",
            ):
                require(
                    sample_record[
                        key
                    ]
                    == job[
                        key
                    ],
                    f"Sample index mismatch for {identifier} field {key}.",
                )

            (
                evaluation_metadata,
                _evaluation_arrays,
            ) = verify_evaluation(
                attempt_dir=
                    attempt_dir,
                job=
                    job,
                sample_arrays=
                    sample_arrays,
                config=
                    config,
                model=
                    models[
                        job[
                            "system"
                        ]
                    ],
            )

            require(
                result_record[
                    "system"
                ]
                == job[
                    "system"
                ]
                and result_record[
                    "evaluation_mode"
                ]
                == job[
                    "evaluation_mode"
                ]
                and result_record[
                    "sample_size"
                ]
                == job[
                    "sample_size"
                ]
                and result_record[
                    "replicate_index"
                ]
                == job[
                    "replicate_index"
                ],
                f"Result index design mismatch: {identifier}",
            )

            for key in (
                "f_hat",
                "s_hat",
                "frozen_predictor_spectral_norm",
                "frozen_predictor_mse",
                "frozen_predictor_rmse",
            ):
                expected = (
                    evaluation_metadata[
                        "statistics"
                    ][
                        key
                    ]
                    if key
                    in (
                        "f_hat",
                        "s_hat",
                    )
                    else evaluation_metadata[
                        key
                    ]
                )

                close_scalar(
                    result_record[
                        key
                    ],
                    expected,
                    label=(
                        f"{identifier} compact result {key}"
                    ),
                )

            require(
                result_record[
                    "budget_certificates"
                ]
                == evaluation_metadata[
                    "budget_certificates"
                ],
                f"Compact budget-certificate record mismatch: {identifier}",
            )

            require(
                float(
                    result_record[
                        "sampling_seconds"
                    ]
                )
                >= 0.0
                and float(
                    result_record[
                        "evaluation_seconds"
                    ]
                )
                >= 0.0,
                f"Negative compact runtime: {identifier}",
            )

            evaluated_pairs += int(
                job[
                    "sample_size"
                ]
            )

    require(
        len(
            global_materials
        )
        == 2
        * EXPECTED_RETAINED_PAIRS,
        "Global seed-material count mismatch.",
    )

    require(
        len(
            global_generated_seeds
        )
        == 2
        * EXPECTED_RETAINED_PAIRS,
        "Global generated-seed count mismatch.",
    )

    require(
        time_index_draws_recomputed
        == EXPECTED_RETAINED_PAIRS,
        "Recomputed time-index draw count mismatch.",
    )

    require(
        evaluated_pairs
        == EXPECTED_RETAINED_PAIRS,
        "Verifier evaluated-pair count mismatch.",
    )

    inventory_after = (
        check_inventory(
            evidence
        )
    )

    require(
        inventory_after
        == inventory_before,
        "Evidence inventory changed during verification.",
    )

    report = {
        "verifier_id":
            VERIFIER_ID,
        "status":
            "passed",
        "verified_utc":
            utc_now(),
        "evidence_path":
            str(
                evidence
            ),
        "campaign_id":
            EXPECTED_CAMPAIGN_ID,
        "campaign_role":
            EXPECTED_CAMPAIGN_ROLE,
        "config_sha256":
            CONFIG_SHA256,
        "execution_source_commit":
            campaign[
                "metadata"
            ][
                "source"
            ][
                "execution_source_commit"
            ],
        "expected_execution_source_commit":
            EXPECTED_EXECUTION_COMMIT,
        "verified_attempts":
            EXPECTED_ATTEMPTS,
        "verified_retained_pairs":
            EXPECTED_RETAINED_PAIRS,
        "verified_seed_streams":
            2
            * EXPECTED_RETAINED_PAIRS,
        "time_index_draws_recomputed":
            time_index_draws_recomputed,
        "random_pair_datasets_regenerated":
            0,
        "representations_refit":
            0,
        "certificate_statistics_recomputed_independently":
            EXPECTED_ATTEMPTS,
        "certificate_bounds_recomputed_independently":
            EXPECTED_ATTEMPTS,
        "kahm_association_datasets_recomputed":
            EXPECTED_ATTEMPTS,
        "evidence_modified":
            False,
        "evidence_sha256sums_fingerprint":
            inventory_before[
                "sha256sums_fingerprint"
            ],
        "verified_file_count":
            inventory_before[
                "verified_file_count"
            ],
        "representation_archive_sha256":
            EXPECTED_ARCHIVE_SHA256,
        "rtol":
            RTOL,
        "atol":
            ATOL,
    }

    checksums = {
        "evidence_sha256sums_fingerprint":
            inventory_before[
                "sha256sums_fingerprint"
            ],
        "verified_file_count":
            inventory_before[
                "verified_file_count"
            ],
        "files":
            inventory_before[
                "files"
            ],
    }

    return (
        report,
        checksums,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
        help=(
            "Completed dynamical pilot evidence directory."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "New verification-output directory outside the source repository "
            "and outside the evidence directory."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = (
        parse_args()
    )

    evidence = (
        args.evidence
        .expanduser()
        .resolve()
    )

    output = (
        args.output
        .expanduser()
        .resolve()
    )

    if output.exists():
        raise FileExistsError(
            f"Verification output already exists: {output}"
        )

    try:
        output.relative_to(
            ROOT
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "Verification output must be outside the source repository."
        )

    try:
        output.relative_to(
            evidence
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "Verification output must be outside the evidence directory."
        )

    provenance = (
        verifier_provenance()
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    report, checksums = (
        verify_evidence(
            evidence
        )
    )

    report[
        "verifier_provenance"
    ] = provenance

    strict_json_write(
        output
        / "verification_report.json",
        report,
    )

    strict_json_write(
        output
        / "verified_evidence_checksums.json",
        checksums,
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
