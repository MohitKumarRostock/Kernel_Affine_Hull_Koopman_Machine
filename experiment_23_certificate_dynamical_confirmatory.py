#!/usr/bin/env python3
"""Failure-preserving runner for the frozen dynamical confirmatory campaign.

This runner executes only the exact prospectively frozen configuration in

    reproduction/certificates/configs/dynamical_confirmatory_v1.json

and consumes only the repository-committed, independently verified
Van der Pol omega=128 representation.

The complete 64-attempt schedule is retained before preflight or sampling.
Every sampled dataset is persisted and fsynced before numerical evaluation.
There is no retry with replacement seed material.

Ordinary sampling/evaluation calculation failures are retained as terminal
attempt failures and execution proceeds to the next prespecified attempt.
Evidence-persistence failures abort the campaign.

No representation fitting occurs in this runner.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

from kahkm_certificates import (
    CERTIFICATE_ID,
)
from kahkm_confirmatory_evaluation import (
    CONFIRMATORY_EVALUATION_ADAPTER_ID,
    evaluate_confirmatory_dataset,
    write_confirmatory_evaluation_evidence,
    write_confirmatory_sample_evidence,
)
from kahkm_confirmatory_frozen_source import (
    CONFIRMATORY_FROZEN_SOURCE_ID,
    EXPECTED_ARCHIVE_MANIFEST_SHA256,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_BUILD_EXECUTION_COMMIT,
    EXPECTED_PACKAGE_CHECKSUMS_SHA256,
    EXPECTED_VERIFIER_SOURCE_COMMIT,
    restore_repository_confirmatory_representation,
)
from kahkm_confirmatory_sampling import (
    CONFIRMATORY_ATTEMPT_COUNT,
    CONFIRMATORY_CAMPAIGN_ID,
    CONFIRMATORY_CAMPAIGN_KEY,
    CONFIRMATORY_PAIR_COUNT,
    CONFIRMATORY_REPLICATES_PER_MODE,
    CONFIRMATORY_ROOT_SEED,
    CONFIRMATORY_SAMPLE_SIZE,
    CONFIRMATORY_SYSTEM,
    CONFIRMATORY_SYSTEM_KEY,
    DYNAMICAL_SAMPLING_ID,
    MODE_KEYS,
    STREAM_KEYS,
    confirmatory_schedule_audit,
    sample_confirmatory_dataset,
)
from kahkm_dynamical_evaluation import (
    DYNAMICAL_EVALUATION_ID,
)
from kahkm_dynamical_evidence import (
    EVALUATION_EVIDENCE_ID,
    SAMPLE_EVIDENCE_ID,
)
from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)
from kahkm_reference_classes import (
    REFERENCE_CLASS_MAP_ID,
)


ROOT = Path(
    __file__
).resolve().parent

RUNNER_ID = (
    "dynamical_original_iid_confirmatory_runner_v1"
)

CONFIG_PATH = (
    "reproduction/certificates/configs/"
    "dynamical_confirmatory_v1.json"
)

PROTOCOL_PATH = (
    "reproduction/certificates/"
    "DYNAMICAL_CONFIRMATORY_PROTOCOL.md"
)

SUPPORTED_CONFIG_SHA256 = (
    "1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91"
)

SUPPORTED_PROTOCOL_SHA256 = (
    "daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c"
)

REQUIRED_SOURCE_FILES = (
    "experiment_23_certificate_dynamical_confirmatory.py",
    "test_experiment_23_certificate_dynamical_confirmatory.py",
    "kahkm_confirmatory_sampling.py",
    "test_kahkm_confirmatory_sampling.py",
    "kahkm_confirmatory_evaluation.py",
    "test_kahkm_confirmatory_evaluation.py",
    "kahkm_confirmatory_frozen_source.py",
    "test_kahkm_confirmatory_frozen_source.py",
    "kahkm_dynamical_evaluation.py",
    "kahkm_dynamical_evidence.py",
    "kahkm_dynamical_pairs.py",
    "kahkm_dynamical_sampling.py",
    "kahkm_certificate_statistics.py",
    "kahkm_certificates.py",
    "kahkm_reference_classes.py",
    "kahkm_frozen_representation.py",
    "kernel_affine_hull_koopman_machines.py",
    "scripts/verify_environment.py",
    "requirements-lock-arm64.txt",
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


@dataclass(frozen=True)
class ModelBundle:
    system: str
    artifact_dir: Path
    abstraction_model: dict[str, Any]
    B: np.ndarray
    omega: float
    tau: float
    spectral_norm: float
    kappas: tuple[float, ...]
    representation_manifest_sha256: str


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def file_sha256(
    path: str | os.PathLike[str],
) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
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


def json_line(
    value: Any,
) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        + "\n"
    )


def append_record(
    handle,
    value: Any,
) -> None:
    handle.write(
        json_line(
            value
        )
    )

    handle.flush()

    os.fsync(
        handle.fileno()
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        append_record(
            handle,
            value,
        )


def source_snapshot() -> dict[str, Any]:
    repository = Path(
        git_bytes(
            "rev-parse",
            "--show-toplevel",
        ).decode().strip()
    ).resolve()

    if repository != ROOT:
        raise RuntimeError(
            "Runner repository root differs from expected ROOT."
        )

    commit = (
        git_bytes(
            "rev-parse",
            "HEAD",
        )
        .decode()
        .strip()
    )

    status = (
        git_bytes(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        .decode()
    )

    if status:
        raise RuntimeError(
            "A clean committed source tree is required:\n"
            + status
        )

    hashes: dict[str, str] = {}

    for relative_name in sorted(
        set(
            REQUIRED_SOURCE_FILES
        )
    ):
        path = (
            ROOT
            / relative_name
        )

        if (
            path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(
                "Missing or symlinked required source file: "
                f"{relative_name}"
            )

        actual = file_sha256(
            path
        )

        try:
            committed_bytes = git_bytes(
                "show",
                f"{commit}:{relative_name}",
            )

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Required source file is not committed at HEAD: "
                f"{relative_name}"
            ) from exc

        committed = hashlib.sha256(
            committed_bytes
        ).hexdigest()

        if actual != committed:
            raise RuntimeError(
                "Required source bytes differ from HEAD: "
                f"{relative_name}"
            )

        hashes[
            relative_name
        ] = actual

    return {
        "commit":
            commit,

        "git_status_porcelain":
            status,

        "required_source_sha256":
            hashes,
    }


def assert_source_unchanged(
    initial: Mapping[str, Any],
) -> None:
    current = source_snapshot()

    if current != initial:
        raise RuntimeError(
            "Committed source state changed during campaign execution."
        )


def load_confirmatory() -> tuple[
    bytes,
    dict[str, Any],
]:
    config_path = (
        ROOT
        / CONFIG_PATH
    )

    raw = config_path.read_bytes()

    if (
        hashlib.sha256(
            raw
        ).hexdigest()
        != SUPPORTED_CONFIG_SHA256
    ):
        raise ValueError(
            "Configuration differs from the frozen confirmatory bytes."
        )

    config = json.loads(
        raw
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Confirmatory configuration must be a JSON object."
        )

    exact_top = {
        "campaign_id":
            CONFIRMATORY_CAMPAIGN_ID,

        "campaign_role":
            "confirmatory",

        "config_schema_id":
            "dynamical_original_iid_confirmatory_config_v1",

        "protocol":
            PROTOCOL_PATH,

        "protocol_sha256":
            SUPPORTED_PROTOCOL_SHA256,
    }

    for key, expected in exact_top.items():
        if config.get(
            key
        ) != expected:
            raise ValueError(
                f"Unexpected confirmatory config field {key!r}."
            )

    protocol_path = (
        ROOT
        / PROTOCOL_PATH
    )

    if (
        file_sha256(
            protocol_path
        )
        != SUPPORTED_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Confirmatory protocol fingerprint changed."
        )

    design = config.get(
        "confirmatory_design"
    )

    if design != {
        "planned_attempts":
            64,

        "planned_retained_pairs":
            1_048_576,

        "replicates_per_cell":
            32,

        "sample_sizes": [
            16_384
        ],

        "systems": [
            "vanderpol"
        ],
    }:
        raise ValueError(
            "Confirmatory design differs from the frozen 64-attempt design."
        )

    certificate = config.get(
        "certificate"
    )

    if certificate != {
        "delta":
            0.05,

        "implementation_id":
            "original_iid_pair_certificate_v1",
    }:
        raise ValueError(
            "Unexpected confirmatory certificate configuration."
        )

    pair_sampling = config.get(
        "pair_sampling"
    )

    if pair_sampling != {
        "evaluation_modes": [
            "matched_stochastic",
            "deterministic",
        ],

        "horizon_steps":
            1200,

        "implementation_id":
            PAIR_LAW_ID,
    }:
        raise ValueError(
            "Unexpected confirmatory pair-sampling configuration."
        )

    seed_scheme = config.get(
        "seed_scheme"
    )

    if not isinstance(
        seed_scheme,
        dict,
    ):
        raise TypeError(
            "seed_scheme must be an object."
        )

    expected_seed_fields = {
        "bit_generator":
            "PCG64",

        "campaign_key":
            CONFIRMATORY_CAMPAIGN_KEY,

        "root_seed":
            CONFIRMATORY_ROOT_SEED,

        "seed_sequence":
            "numpy.random.SeedSequence",

        "mode_keys":
            {
                "deterministic":
                    2,

                "matched_stochastic":
                    1,
            },

        "stream_keys":
            {
                "time_seed":
                    2,

                "trajectory_seed":
                    1,
            },

        "system_keys":
            {
                "duffing":
                    1,

                "vanderpol":
                    2,
            },

        "evaluation_seed_components": [
            "campaign_key",
            "system_key",
            "mode_key",
            "sample_size",
            "replicate_index",
            "pair_index",
            "stream_key",
            "root_seed",
        ],
    }

    for key, expected in expected_seed_fields.items():
        if seed_scheme.get(
            key
        ) != expected:
            raise ValueError(
                "Unexpected seed-scheme field: "
                f"{key}"
            )

    representation = config.get(
        "confirmatory_representation"
    )

    if not isinstance(
        representation,
        dict,
    ):
        raise TypeError(
            "confirmatory_representation must be an object."
        )

    expected_representation_fields = {
        "implementation_id":
            "frozen_abstraction_omega_refit_nlms_v1",

        "system":
            "vanderpol",

        "omega":
            128.0,

        "tau":
            1e-6,

        "n_jobs":
            -1,

        "batch_size":
            256,

        "beta":
            0.1,

        "n_steps_train":
            1200,

        "nlms_epochs":
            20,

        "nlms_initial_matrix":
            "identity",

        "nlms_shuffle":
            False,

        "operator_refit_uses_training_data_only":
            True,

        "predictor_orientation":
            "B.T @ Phi",

        "project_stochastic":
            False,

        "train_seeds": [
            0,
            1,
            2,
        ],
    }

    for key, expected in expected_representation_fields.items():
        if representation.get(
            key
        ) != expected:
            raise ValueError(
                "Unexpected confirmatory representation field: "
                f"{key}"
            )

    reference = config.get(
        "reference_class_map"
    )

    if reference != {
        "implementation_id":
            REFERENCE_CLASS_MAP_ID
    }:
        raise ValueError(
            "Unexpected reference-class-map configuration."
        )

    norm_budgets = config.get(
        "norm_budgets"
    )

    if norm_budgets != {
        "deduplicate_exact_binary64":
            True,

        "fixed_constants": [
            1.0,
            1.4142135623730951,
        ],

        "sort_ascending":
            True,

        "training_derived": [
            "nlms_spectral_norm",
            "twice_nlms_spectral_norm",
        ],
    }:
        raise ValueError(
            "Unexpected norm-budget design."
        )

    execution = config.get(
        "execution"
    )

    if not isinstance(
        execution,
        dict,
    ):
        raise TypeError(
            "execution must be an object."
        )

    for key in (
        "archive_after_independent_verification",
        "no_retry_with_replacement_seed_material",
        "output_must_not_exist",
        "preflight_full_certificate_test_suite",
        "require_clean_source",
        "require_committed_configuration",
    ):
        if execution.get(
            key
        ) is not True:
            raise ValueError(
                "Required execution safeguard is not enabled: "
                f"{key}"
            )

    selection = config.get(
        "selection_provenance"
    )

    if not isinstance(
        selection,
        dict,
    ):
        raise TypeError(
            "selection_provenance must be an object."
        )

    if (
        selection.get(
            "fresh_confirmatory_evaluation_pairs_required"
        )
        is not True
        or selection.get(
            "pilot_evaluation_pairs_may_be_used_as_confirmatory_evidence"
        )
        is not False
    ):
        raise ValueError(
            "Pilot/confirmatory evidence firewall changed."
        )

    source_abstraction = config.get(
        "source_frozen_abstraction"
    )

    if not isinstance(
        source_abstraction,
        dict,
    ):
        raise TypeError(
            "source_frozen_abstraction must be an object."
        )

    if (
        source_abstraction.get(
            "refit_autoencoders"
        )
        is not False
        or source_abstraction.get(
            "refit_kmeans"
        )
        is not False
        or source_abstraction.get(
            "reuse_kahm_autoencoders"
        )
        is not True
        or source_abstraction.get(
            "reuse_state_space_kmeans_centers"
        )
        is not True
    ):
        raise ValueError(
            "Frozen-abstraction reuse contract changed."
        )

    audit = (
        confirmatory_schedule_audit()
    )

    if (
        audit[
            "attempt_count"
        ]
        != CONFIRMATORY_ATTEMPT_COUNT
        or audit[
            "pair_count"
        ]
        != CONFIRMATORY_PAIR_COUNT
        or audit[
            "campaign_key"
        ]
        != CONFIRMATORY_CAMPAIGN_KEY
        or audit[
            "pilot_seed_material_disjoint"
        ]
        is not True
    ):
        raise ValueError(
            "Confirmatory sampler audit disagrees with frozen design."
        )

    return (
        raw,
        config,
    )


def build_schedule(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    design = config[
        "confirmatory_design"
    ]

    pair_sampling = config[
        "pair_sampling"
    ]

    seed_scheme = config[
        "seed_scheme"
    ]

    jobs: list[
        dict[str, Any]
    ] = []

    for system in design[
        "systems"
    ]:
        if system != CONFIRMATORY_SYSTEM:
            raise ValueError(
                "Only Van der Pol is permitted in this confirmatory campaign."
            )

        system_key = int(
            seed_scheme[
                "system_keys"
            ][
                system
            ]
        )

        if system_key != CONFIRMATORY_SYSTEM_KEY:
            raise ValueError(
                "Van der Pol system key changed."
            )

        for mode in pair_sampling[
            "evaluation_modes"
        ]:
            if mode not in MODE_KEYS:
                raise ValueError(
                    "Unexpected evaluation mode."
                )

            mode_key = int(
                seed_scheme[
                    "mode_keys"
                ][
                    mode
                ]
            )

            if mode_key != MODE_KEYS[
                mode
            ]:
                raise ValueError(
                    "Evaluation-mode key mismatch."
                )

            for sample_size_raw in design[
                "sample_sizes"
            ]:
                sample_size = int(
                    sample_size_raw
                )

                if (
                    sample_size
                    != CONFIRMATORY_SAMPLE_SIZE
                ):
                    raise ValueError(
                        "Confirmatory sample size changed."
                    )

                for replicate_index in range(
                    int(
                        design[
                            "replicates_per_cell"
                        ]
                    )
                ):
                    attempt_id = (
                        f"s{system_key}"
                        f"_q{mode_key}"
                        f"_m{sample_size}"
                        f"_r{replicate_index:03d}"
                    )

                    jobs.append(
                        {
                            "attempt_id":
                                attempt_id,

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

                            "campaign_key":
                                int(
                                    seed_scheme[
                                        "campaign_key"
                                    ]
                                ),

                            "root_seed":
                                int(
                                    seed_scheme[
                                        "root_seed"
                                    ]
                                ),

                            "trajectory_stream_key":
                                int(
                                    STREAM_KEYS[
                                        "trajectory_seed"
                                    ]
                                ),

                            "time_stream_key":
                                int(
                                    STREAM_KEYS[
                                        "time_seed"
                                    ]
                                ),
                        }
                    )

    identifiers = [
        job[
            "attempt_id"
        ]
        for job in jobs
    ]

    if len(
        identifiers
    ) != len(
        set(
            identifiers
        )
    ):
        raise ValueError(
            "Duplicate confirmatory attempt identifiers."
        )

    if len(
        jobs
    ) != CONFIRMATORY_ATTEMPT_COUNT:
        raise ValueError(
            "Confirmatory schedule must contain exactly 64 attempts."
        )

    retained_pairs = sum(
        int(
            job[
                "sample_size"
            ]
        )
        for job in jobs
    )

    if (
        retained_pairs
        != CONFIRMATORY_PAIR_COUNT
    ):
        raise ValueError(
            "Confirmatory schedule pair count differs from 1,048,576."
        )

    if (
        int(
            design[
                "planned_attempts"
            ]
        )
        != len(
            jobs
        )
        or int(
            design[
                "planned_retained_pairs"
            ]
        )
        != retained_pairs
    ):
        raise ValueError(
            "Configuration planning totals disagree with constructed schedule."
        )

    return jobs


def run_command_check(
    *,
    logs: Path,
    name: str,
    command: list[str],
) -> None:
    write_json(
        logs
        / f"{name}.command.json",
        {
            "command":
                command,

            "cwd":
                str(
                    ROOT
                ),
        },
    )

    with (
        (
            logs
            / f"{name}.stdout.log"
        ).open(
            "xb"
        ) as stdout,
        (
            logs
            / f"{name}.stderr.log"
        ).open(
            "xb"
        ) as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )

    write_json(
        logs
        / f"{name}.result.json",
        {
            "returncode":
                completed.returncode
        },
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Preflight {name} failed; see retained check logs."
        )


def run_preflight(
    output: Path,
) -> None:
    logs = (
        output
        / "checks"
    )

    logs.mkdir()

    checks = [
        (
            "environment",
            [
                sys.executable,
                "scripts/verify_environment.py",
            ],
        ),
        (
            "pip_check",
            [
                sys.executable,
                "-m",
                "pip",
                "check",
            ],
        ),
        (
            "pip_freeze",
            [
                sys.executable,
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
        ),
        (
            "focused_confirmatory_tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "-v",
                "test_kahkm_confirmatory_frozen_source",
                "test_kahkm_confirmatory_sampling",
                "test_kahkm_confirmatory_evaluation",
                "test_experiment_23_certificate_dynamical_confirmatory",
            ],
        ),
        (
            "full_kahkm_test_suite",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_kahkm_*.py",
                "-v",
            ],
        ),
    ]

    for name, command in checks:
        run_command_check(
            logs=
                logs,

            name=
                name,

            command=
                command,
        )

    packages = []

    lock_path = (
        ROOT
        / "requirements-lock-arm64.txt"
    )

    for raw_line in lock_path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if (
            not line
            or line.startswith(
                "#"
            )
        ):
            continue

        name, separator, expected = (
            line.partition(
                "=="
            )
        )

        if (
            not separator
            or not name
            or not expected
        ):
            raise ValueError(
                "Unsupported reference lockfile entry: "
                + line
            )

        actual = (
            importlib.metadata.version(
                name
            )
        )

        packages.append(
            {
                "name":
                    name,

                "expected":
                    expected,

                "actual":
                    actual,

                "matches":
                    actual
                    == expected,
            }
        )

    write_json(
        logs
        / "locked_packages.json",
        packages,
    )

    if not all(
        item[
            "matches"
        ]
        for item in packages
    ):
        raise RuntimeError(
            "Installed packages differ from the reference lockfile."
        )


def _norm_budgets(
    *,
    config: Mapping[str, Any],
    spectral_norm: float,
) -> tuple[float, ...]:
    norm = float(
        spectral_norm
    )

    if (
        not np.isfinite(
            norm
        )
        or norm < 0.0
    ):
        raise ValueError(
            "Frozen spectral norm is invalid."
        )

    design = config[
        "norm_budgets"
    ]

    if design[
        "training_derived"
    ] != [
        "nlms_spectral_norm",
        "twice_nlms_spectral_norm",
    ]:
        raise ValueError(
            "Unexpected training-derived norm-budget specification."
        )

    values = [
        float(
            value
        )
        for value in design[
            "fixed_constants"
        ]
    ]

    values.extend(
        [
            norm,
            2.0
            * norm,
        ]
    )

    if design[
        "deduplicate_exact_binary64"
    ]:
        unique: dict[
            str,
            float,
        ] = {}

        for value in values:
            unique.setdefault(
                float(
                    value
                ).hex(),
                float(
                    value
                ),
            )

        values = list(
            unique.values()
        )

    if design[
        "sort_ascending"
    ]:
        values.sort()

    result = tuple(
        values
    )

    if (
        not result
        or any(
            not np.isfinite(
                value
            )
            or value < 0.0
            for value in result
        )
    ):
        raise ValueError(
            "Invalid norm-budget result."
        )

    return result


def restore_model_source(
    *,
    output: Path,
    config: Mapping[str, Any],
) -> tuple[
    ModelBundle,
    dict[str, Any],
]:
    # Canonicalize once so macOS /var -> /private/var resolution cannot
    # make the restored artifact appear to lie outside the campaign output.
    output = Path(
        output
    ).expanduser().resolve()

    restoration_dir = (
        output
        / "representation_source"
    )

    restored = (
        restore_repository_confirmatory_representation(
            repository_root=
                ROOT,

            restoration_dir=
                restoration_dir,
        )
    )

    if (
        restored.source_id
        != CONFIRMATORY_FROZEN_SOURCE_ID
        or restored.campaign_id
        != CONFIRMATORY_CAMPAIGN_ID
        or restored.system
        != CONFIRMATORY_SYSTEM
        or restored.omega
        != 128.0
        or restored.tau
        != 1e-6
        or restored.archive_sha256
        != EXPECTED_ARCHIVE_SHA256
        or restored.archive_manifest_sha256
        != EXPECTED_ARCHIVE_MANIFEST_SHA256
        or restored.package_checksums_sha256
        != EXPECTED_PACKAGE_CHECKSUMS_SHA256
        or restored.build_execution_source_commit
        != EXPECTED_BUILD_EXECUTION_COMMIT
        or restored.verifier_source_commit
        != EXPECTED_VERIFIER_SOURCE_COMMIT
        or restored.certificate_evaluation_pairs_generated
        != 0
    ):
        raise RuntimeError(
            "Restored confirmatory representation provenance mismatch."
        )

    loaded = (
        load_frozen_representation(
            restored.vanderpol_artifact_dir,
            verify_hashes=True,
        )
    )

    if (
        loaded.format_id
        != FORMAT_ID
        or loaded.system
        != CONFIRMATORY_SYSTEM
        or loaded.omega
        != 128.0
        or loaded.tau
        != 1e-6
    ):
        raise RuntimeError(
            "Loaded confirmatory representation identity mismatch."
        )

    representation_config = config[
        "confirmatory_representation"
    ]

    if (
        loaded.omega
        != float(
            representation_config[
                "omega"
            ]
        )
        or loaded.tau
        != float(
            representation_config[
                "tau"
            ]
        )
    ):
        raise RuntimeError(
            "Loaded representation differs from frozen config."
        )

    B = np.asarray(
        loaded.B,
        dtype=np.float64,
    )

    if (
        B.ndim
        != 2
        or B.shape[
            0
        ]
        != B.shape[
            1
        ]
        or not np.all(
            np.isfinite(
                B
            )
        )
    ):
        raise RuntimeError(
            "Loaded confirmatory NLMS operator is invalid."
        )

    spectral_norm = float(
        np.linalg.norm(
            B,
            ord=2,
        )
    )

    if (
        not np.isfinite(
            spectral_norm
        )
        or spectral_norm < 0.0
    ):
        raise RuntimeError(
            "Loaded confirmatory spectral norm is invalid."
        )

    training = loaded.manifest.get(
        "training_metadata"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise RuntimeError(
            "Confirmatory representation lacks training metadata."
        )

    retained_norm = float(
        training[
            "nlms_spectral_norm"
        ]
    )

    tolerance = (
        16.0
        * np.finfo(
            np.float64
        ).eps
        * max(
            1.0,
            abs(
                retained_norm
            ),
            abs(
                spectral_norm
            ),
        )
    )

    if (
        abs(
            retained_norm
            - spectral_norm
        )
        > tolerance
    ):
        raise RuntimeError(
            "Loaded spectral norm differs from retained training metadata."
        )

    kappas = (
        _norm_budgets(
            config=
                config,

            spectral_norm=
                spectral_norm,
        )
    )

    manifest_path = (
        restored.vanderpol_artifact_dir
        / "manifest.json"
    )

    manifest_sha = file_sha256(
        manifest_path
    )

    bundle = ModelBundle(
        system=
            CONFIRMATORY_SYSTEM,

        artifact_dir=
            restored.vanderpol_artifact_dir,

        abstraction_model=
            loaded.abstraction_model,

        B=
            B,

        omega=
            float(
                loaded.omega
            ),

        tau=
            float(
                loaded.tau
            ),

        spectral_norm=
            spectral_norm,

        kappas=
            kappas,

        representation_manifest_sha256=
            manifest_sha,
    )

    metadata = {
        "frozen_source_id":
            restored.source_id,

        "system":
            restored.system,

        "omega":
            restored.omega,

        "tau":
            restored.tau,

        "archive_sha256":
            restored.archive_sha256,

        "archive_manifest_sha256":
            restored.archive_manifest_sha256,

        "package_checksums_sha256":
            restored.package_checksums_sha256,

        "build_execution_source_commit":
            restored.build_execution_source_commit,

        "verifier_source_commit":
            restored.verifier_source_commit,

        "representation_manifest_sha256":
            manifest_sha,

        "representation_relative_dir":
            restored.vanderpol_artifact_dir.relative_to(
                output
            ).as_posix(),

        "spectral_norm":
            spectral_norm,

        "norm_budgets":
            list(
                kappas
            ),

        "state_space_kmeans_refits":
            0,

        "autoencoder_refits":
            0,

        "nlms_operator_refits":
            0,

        "certificate_evaluation_pairs_generated_during_restore":
            0,
    }

    return (
        bundle,
        metadata,
    )


def _compact_result_record(
    *,
    attempt_id: str,
    attempt_relative_dir: str,
    sampling_seconds: float,
    evaluation_seconds: float,
    evaluation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "attempt_id":
            attempt_id,

        "attempt_relative_dir":
            attempt_relative_dir,

        "sampling_seconds":
            float(
                sampling_seconds
            ),

        "evaluation_seconds":
            float(
                evaluation_seconds
            ),

        "system":
            evaluation_metadata[
                "system"
            ],

        "evaluation_mode":
            evaluation_metadata[
                "evaluation_mode"
            ],

        "sample_size":
            int(
                evaluation_metadata[
                    "sample_size"
                ]
            ),

        "replicate_index":
            int(
                evaluation_metadata[
                    "replicate_index"
                ]
            ),

        "frozen_predictor_spectral_norm":
            float(
                evaluation_metadata[
                    "frozen_predictor_spectral_norm"
                ]
            ),

        "frozen_predictor_mse":
            float(
                evaluation_metadata[
                    "frozen_predictor_mse"
                ]
            ),

        "frozen_predictor_rmse":
            float(
                evaluation_metadata[
                    "frozen_predictor_rmse"
                ]
            ),

        "statistics":
            evaluation_metadata[
                "statistics"
            ],

        "budget_certificates":
            evaluation_metadata[
                "budget_certificates"
            ],

        "evaluation_array_sha256":
            evaluation_metadata[
                "array_sha256"
            ],
    }


def execute_jobs(
    *,
    jobs: list[dict[str, Any]],
    model: ModelBundle,
    config: Mapping[str, Any],
    output: Path,
    state: dict[str, Any],
    source_before: Mapping[str, Any],
) -> None:
    certificate = config[
        "certificate"
    ]

    representation = config[
        "confirmatory_representation"
    ]

    attempts_root = (
        output
        / "attempt_evidence"
    )

    attempts_root.mkdir()

    with ExitStack() as stack:
        events = stack.enter_context(
            (
                output
                / "attempts.jsonl"
            ).open(
                "x",
                encoding="utf-8",
                newline="\n",
            )
        )

        samples = stack.enter_context(
            (
                output
                / "samples.jsonl"
            ).open(
                "x",
                encoding="utf-8",
                newline="\n",
            )
        )

        results = stack.enter_context(
            (
                output
                / "results.jsonl"
            ).open(
                "x",
                encoding="utf-8",
                newline="\n",
            )
        )

        for job in jobs:
            assert_source_unchanged(
                source_before
            )

            identifier = job[
                "attempt_id"
            ]

            state[
                "active_attempt_id"
            ] = identifier

            append_record(
                events,
                {
                    "attempt_id":
                        identifier,

                    "attempt_number":
                        1,

                    "event":
                        "started",

                    "utc":
                        utc_now(),
                },
            )

            state[
                "started"
            ] += 1

            stage = (
                "sampling"
            )

            sampling_start = (
                time.monotonic()
            )

            try:
                dataset = (
                    sample_confirmatory_dataset(
                        evaluation_mode=
                            job[
                                "evaluation_mode"
                            ],

                        replicate_index=
                            int(
                                job[
                                    "replicate_index"
                                ]
                            ),
                    )
                )

                sampling_seconds = (
                    time.monotonic()
                    - sampling_start
                )

            except Exception as exc:
                append_record(
                    events,
                    {
                        "attempt_id":
                            identifier,

                        "attempt_number":
                            1,

                        "event":
                            "failed",

                        "stage":
                            stage,

                        "utc":
                            utc_now(),

                        "error_type":
                            type(
                                exc
                            ).__name__,

                        "error":
                            str(
                                exc
                            ),

                        "traceback":
                            traceback.format_exc(),
                    },
                )

                state[
                    "failed"
                ] += 1

                state[
                    "active_attempt_id"
                ] = None

                continue

            attempt_dir = (
                attempts_root
                / identifier
            )

            # Deliberately outside the calculation exception handler:
            # persistence failure aborts the campaign.
            sample_metadata = (
                write_confirmatory_sample_evidence(
                    attempt_dir=
                        attempt_dir,

                    dataset=
                        dataset,
                )
            )

            append_record(
                samples,
                {
                    "attempt_id":
                        identifier,

                    "attempt_relative_dir":
                        attempt_dir.relative_to(
                            output
                        ).as_posix(),

                    "sampling_seconds":
                        float(
                            sampling_seconds
                        ),

                    "sample_metadata":
                        sample_metadata,
                },
            )

            state[
                "sampled"
            ] += 1

            state[
                "sampled_pairs"
            ] += int(
                job[
                    "sample_size"
                ]
            )

            stage = (
                "evaluation"
            )

            evaluation_start = (
                time.monotonic()
            )

            try:
                evaluation = (
                    evaluate_confirmatory_dataset(
                        dataset=
                            dataset,

                        abstraction_model=
                            model.abstraction_model,

                        B=
                            model.B,

                        omega=
                            model.omega,

                        tau=
                            model.tau,

                        kappas=
                            model.kappas,

                        delta=
                            float(
                                certificate[
                                    "delta"
                                ]
                            ),

                        n_jobs=
                            int(
                                representation[
                                    "n_jobs"
                                ]
                            ),

                        batch_size=
                            int(
                                representation[
                                    "batch_size"
                                ]
                            ),
                    )
                )

                evaluation_seconds = (
                    time.monotonic()
                    - evaluation_start
                )

            except Exception as exc:
                append_record(
                    events,
                    {
                        "attempt_id":
                            identifier,

                        "attempt_number":
                            1,

                        "event":
                            "failed",

                        "stage":
                            stage,

                        "utc":
                            utc_now(),

                        "error_type":
                            type(
                                exc
                            ).__name__,

                        "error":
                            str(
                                exc
                            ),

                        "traceback":
                            traceback.format_exc(),
                    },
                )

                state[
                    "failed"
                ] += 1

                state[
                    "active_attempt_id"
                ] = None

                continue

            # Evaluation persistence is also fatal on failure.
            evaluation_metadata = (
                write_confirmatory_evaluation_evidence(
                    attempt_dir=
                        attempt_dir,

                    dataset=
                        dataset,

                    evaluation=
                        evaluation,
                )
            )

            record = (
                _compact_result_record(
                    attempt_id=
                        identifier,

                    attempt_relative_dir=
                        attempt_dir.relative_to(
                            output
                        ).as_posix(),

                    sampling_seconds=
                        sampling_seconds,

                    evaluation_seconds=
                        evaluation_seconds,

                    evaluation_metadata=
                        evaluation_metadata,
                )
            )

            append_record(
                results,
                record,
            )

            append_record(
                events,
                {
                    "attempt_id":
                        identifier,

                    "attempt_number":
                        1,

                    "event":
                        "completed",

                    "utc":
                        utc_now(),
                },
            )

            state[
                "evaluated"
            ] += 1

            state[
                "evaluated_pairs"
            ] += int(
                job[
                    "sample_size"
                ]
            )

            state[
                "succeeded"
            ] += 1

            state[
                "active_attempt_id"
            ] = None

            assert_source_unchanged(
                source_before
            )


def write_checksums(
    output: Path,
) -> None:
    files = sorted(
        path
        for path in output.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS"
        )
    )

    checksum_path = (
        output
        / "SHA256SUMS"
    )

    with checksum_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for path in files:
            handle.write(
                f"{file_sha256(path)}  "
                f"{path.relative_to(output).as_posix()}\n"
            )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def run_campaign(
    output: Path,
) -> int:
    output = output.expanduser().resolve()

    if (
        output == ROOT
        or ROOT
        in output.parents
    ):
        raise ValueError(
            "Choose a new output directory outside this working tree."
        )

    if output.exists():
        raise FileExistsError(
            "Output already exists; it will not be reused or overwritten."
        )

    raw, config = (
        load_confirmatory()
    )

    source_before = (
        source_snapshot()
    )

    jobs = build_schedule(
        config
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    start = time.monotonic()

    state: dict[
        str,
        Any,
    ] = {
        "status":
            "initializing",

        "started":
            0,

        "sampled":
            0,

        "evaluated":
            0,

        "succeeded":
            0,

        "failed":
            0,

        "sampled_pairs":
            0,

        "evaluated_pairs":
            0,

        "planned":
            len(
                jobs
            ),

        "planned_retained_pairs":
            CONFIRMATORY_PAIR_COUNT,

        "active_attempt_id":
            None,

        "started_utc":
            utc_now(),

        "campaign_id":
            CONFIRMATORY_CAMPAIGN_ID,

        "campaign_role":
            "confirmatory",
    }

    exit_code = 2

    try:
        with (
            output
            / "campaign_spec.json"
        ).open(
            "xb"
        ) as handle:
            handle.write(
                raw
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        with (
            output
            / "protocol.md"
        ).open(
            "xb"
        ) as handle:
            handle.write(
                (
                    ROOT
                    / PROTOCOL_PATH
                ).read_bytes()
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        write_json(
            output
            / "campaign_metadata.json",
            {
                "runner_id":
                    RUNNER_ID,

                "campaign_id":
                    config[
                        "campaign_id"
                    ],

                "campaign_role":
                    config[
                        "campaign_role"
                    ],

                "run_id":
                    output.name,

                "output_path":
                    str(
                        output
                    ),

                "repository_path":
                    str(
                        ROOT
                    ),

                "config_path":
                    CONFIG_PATH,

                "config_sha256":
                    SUPPORTED_CONFIG_SHA256,

                "protocol_path":
                    PROTOCOL_PATH,

                "protocol_sha256":
                    SUPPORTED_PROTOCOL_SHA256,

                "source":
                    source_before,

                "python_executable":
                    sys.executable,

                "python_version":
                    sys.version,

                "platform":
                    platform.platform(),

                "architecture":
                    platform.machine(),

                "argv":
                    sys.argv,

                "process_id":
                    os.getpid(),

                "started_utc":
                    state[
                        "started_utc"
                    ],

                "numerical_environment":
                    {
                        name:
                            os.environ.get(
                                name
                            )
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                            "PYTHONHASHSEED",
                        )
                    },

                "implementation_ids":
                    {
                        "certificate_formula":
                            CERTIFICATE_ID,

                        "evaluation":
                            DYNAMICAL_EVALUATION_ID,

                        "confirmatory_evaluation_adapter":
                            CONFIRMATORY_EVALUATION_ADAPTER_ID,

                        "sample_evidence":
                            SAMPLE_EVIDENCE_ID,

                        "evaluation_evidence":
                            EVALUATION_EVIDENCE_ID,

                        "sampling":
                            DYNAMICAL_SAMPLING_ID,

                        "pair_law":
                            PAIR_LAW_ID,

                        "reference_class_map":
                            REFERENCE_CLASS_MAP_ID,

                        "frozen_representation_format":
                            FORMAT_ID,

                        "frozen_representation_source":
                            CONFIRMATORY_FROZEN_SOURCE_ID,
                    },
            },
        )

        with (
            output
            / "schedule.jsonl"
        ).open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for job in jobs:
                handle.write(
                    json_line(
                        job
                    )
                )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        state[
            "status"
        ] = "preflight"

        run_preflight(
            output
        )

        assert_source_unchanged(
            source_before
        )

        state[
            "status"
        ] = "restoring_representation"

        model, model_metadata = (
            restore_model_source(
                output=
                    output,

                config=
                    config,
            )
        )

        write_json(
            output
            / "representation_source.json",
            model_metadata,
        )

        assert_source_unchanged(
            source_before
        )

        state[
            "status"
        ] = "running"

        execute_jobs(
            jobs=
                jobs,

            model=
                model,

            config=
                config,

            output=
                output,

            state=
                state,

            source_before=
                source_before,
        )

        state[
            "status"
        ] = (
            "completed_with_failures"
            if state[
                "failed"
            ]
            else "completed"
        )

        exit_code = (
            1
            if state[
                "failed"
            ]
            else 0
        )

    except (
        Exception,
        KeyboardInterrupt,
    ) as exc:
        state[
            "aborted_stage"
        ] = state[
            "status"
        ]

        state[
            "status"
        ] = "aborted"

        state[
            "error_type"
        ] = type(
            exc
        ).__name__

        state[
            "error"
        ] = str(
            exc
        )

        state[
            "traceback"
        ] = traceback.format_exc()

        exit_code = (
            130
            if isinstance(
                exc,
                KeyboardInterrupt,
            )
            else 2
        )

    try:
        source_after = (
            source_snapshot()
        )

        source_unchanged = (
            source_after
            == source_before
        )

    except Exception as exc:
        source_after = {
            "inspection_error":
                str(
                    exc
                )
        }

        source_unchanged = False

    if not source_unchanged:
        if state[
            "status"
        ].startswith(
            "completed"
        ):
            state[
                "status"
            ] = "invalid_source"

        exit_code = 2

    if not (
        output
        / "source_after.json"
    ).exists():
        write_json(
            output
            / "source_after.json",
            source_after,
        )

    state.update(
        {
            "ended_utc":
                utc_now(),

            "elapsed_seconds":
                time.monotonic()
                - start,

            "exit_code":
                exit_code,

            "source_unchanged":
                source_unchanged,

            "not_started":
                state[
                    "planned"
                ]
                - state[
                    "started"
                ],

            "started_without_terminal_event":
                state[
                    "started"
                ]
                - state[
                    "succeeded"
                ]
                - state[
                    "failed"
                ],

            "fresh_confirmatory_pairs_required":
                True,

            "pilot_evaluation_pairs_reused":
                False,

            "representation_refits_during_campaign":
                0,

            "independent_evidence_verification":
                "not_yet_performed",
        }
    )

    if not (
        output
        / "campaign_summary.json"
    ).exists():
        write_json(
            output
            / "campaign_summary.json",
            state,
        )

    if not (
        output
        / "SHA256SUMS"
    ).exists():
        write_checksums(
            output
        )

    print(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "New campaign directory outside the source working tree."
        ),
    )

    args = parser.parse_args()

    try:
        return run_campaign(
            args.output
        )

    except Exception as exc:
        print(
            "STOP:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        print(
            "Retain any created output directory; do not overwrite it.",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
