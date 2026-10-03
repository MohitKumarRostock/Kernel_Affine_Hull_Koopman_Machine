#!/usr/bin/env python3
"""Failure-preserving runner for the frozen dynamical certificate pilot.

The campaign is the pilot frozen in

    reproduction/certificates/configs/dynamical_pilot_v1.json

and consumes KAHM representations only from the committed, independently
verified repository archive.

Core execution guarantees
-------------------------
- Require a clean committed source tree.
- Verify required source bytes against HEAD.
- Save the complete 96-attempt schedule before preflight or sampling.
- Restore the frozen representations only from the committed archive.
- Persist every successfully sampled physical-state dataset before evaluation.
- Do not retry failed sampling or evaluation attempts with replacement seeds.
- Abort on evidence-persistence failures.
- Recheck source bytes after sampling and after evaluation.
- Retain complete accounting and SHA256 inventory.
- A zero certificate is a valid scientific result, not an execution failure.

The pilot contains:
    2 systems
    x 2 evaluation modes
    x 3 sample sizes
    x 8 replicates
    = 96 replicate datasets
    = 86,016 independently generated retained pairs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from kahkm_certificates import (
    CERTIFICATE_ID,
)
from kahkm_dynamical_evaluation import (
    DYNAMICAL_EVALUATION_ID,
    evaluate_frozen_dynamical_pairs,
)
from kahkm_dynamical_evidence import (
    EVALUATION_EVIDENCE_ID,
    SAMPLE_EVIDENCE_ID,
    write_evaluation_evidence,
    write_sample_evidence,
)
from kahkm_dynamical_frozen_source import (
    EXPECTED_ARCHIVE_SHA256,
    RestoredFrozenRepresentations,
    restore_repository_frozen_representations,
)
from kahkm_dynamical_pairs import (
    PAIR_LAW_ID,
)
from kahkm_dynamical_sampling import (
    DYNAMICAL_SAMPLING_ID,
    sample_dynamical_dataset,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    LoadedFrozenRepresentation,
    load_frozen_representation,
)
from kahkm_reference_classes import (
    REFERENCE_CLASS_MAP_ID,
)


ROOT = Path(__file__).resolve().parent

RUNNER_ID = "dynamical_original_iid_pilot_runner_v1"

CONFIG_PATH = (
    "reproduction/certificates/configs/"
    "dynamical_pilot_v1.json"
)

SUPPORTED_CONFIG_SHA256 = (
    "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1"
)

# The frozen configuration uses a campaign-facing certificate implementation
# identifier. CERTIFICATE_ID is the lower-level equation/formula identifier
# exposed by kahkm_certificates.py; the two namespaces are intentionally
# distinct and must not be conflated.
CONFIG_CERTIFICATE_IMPLEMENTATION_ID = (
    "original_iid_pair_certificate_v1"
)

EXPECTED_ATTEMPTS = 96
EXPECTED_RETAINED_PAIRS = 86016

REPRESENTATION_ARCHIVE_DIR = (
    "reproduction/prospective_campaigns/"
    "dynamical_frozen_representations_pilot_v1_run001"
)

REQUIRED_SOURCE_FILES = (
    Path(__file__).name,
    CONFIG_PATH,
    "reproduction/certificates/DYNAMICAL_PROTOCOL.md",
    "requirements-lock-arm64.txt",
    "scripts/verify_environment.py",
    "kahkm_certificates.py",
    "kahkm_certificate_statistics.py",
    "kahkm_reference_classes.py",
    "kahkm_dynamical_pairs.py",
    "kahkm_dynamical_sampling.py",
    "kahkm_dynamical_evaluation.py",
    "kahkm_dynamical_evidence.py",
    "kahkm_dynamical_frozen_source.py",
    "kahkm_frozen_representation.py",
    "kernel_affine_hull_koopman_machines.py",
    "tests/test_kahkm_certificates.py",
    "tests/test_kahkm_certificate_statistics.py",
    "tests/test_kahkm_reference_classes.py",
    "tests/test_kahkm_dynamical_pairs.py",
    "tests/test_kahkm_dynamical_sampling.py",
    "tests/test_kahkm_dynamical_evaluation.py",
    "tests/test_kahkm_dynamical_evidence.py",
    "tests/test_kahkm_dynamical_frozen_source.py",
    "tests/test_kahkm_dynamical_campaign.py",
    f"{REPRESENTATION_ARCHIVE_DIR}/ARCHIVE.json",
    f"{REPRESENTATION_ARCHIVE_DIR}/README.md",
    f"{REPRESENTATION_ARCHIVE_DIR}/SHA256SUMS",
    f"{REPRESENTATION_ARCHIVE_DIR}/evidence.tar.gz",
)


@dataclass(frozen=True)
class ModelBundle:
    """One restored frozen model plus its prespecified norm budgets."""

    system: str
    loaded: LoadedFrozenRepresentation
    spectral_norm: float
    kappas: tuple[float, ...]


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


def file_sha256(
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
            digest.update(block)

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
    value: Mapping[str, Any],
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
    value: Mapping[str, Any],
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
    value: Mapping[str, Any],
) -> None:
    text = (
        json.dumps(
            value,
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


def source_snapshot() -> dict[str, Any]:
    """Require clean source and compare required-file bytes against HEAD."""
    repository = Path(
        git_bytes(
            "rev-parse",
            "--show-toplevel",
        )
        .decode()
        .strip()
    ).resolve()

    if repository != ROOT:
        raise RuntimeError(
            "Runner must execute from the repository root."
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
            "A clean source working tree is required:\n"
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
        "execution_source_commit":
            commit,
        "git_status_porcelain":
            status,
        "source_file_sha256":
            hashes,
    }


def assert_source_unchanged(
    initial: Mapping[str, Any],
) -> None:
    """Recheck HEAD/status and actual required-file bytes without git-show."""

    repository = Path(
        git_bytes(
            "rev-parse",
            "--show-toplevel",
        )
        .decode()
        .strip()
    ).resolve()

    if repository != ROOT:
        raise RuntimeError(
            "Repository root changed during campaign."
        )

    commit = (
        git_bytes(
            "rev-parse",
            "HEAD",
        )
        .decode()
        .strip()
    )

    if (
        commit
        != initial[
            "execution_source_commit"
        ]
    ):
        raise RuntimeError(
            "HEAD changed during campaign."
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
            "Source working tree changed during campaign:\n"
            + status
        )

    expected_hashes = initial[
        "source_file_sha256"
    ]

    if not isinstance(
        expected_hashes,
        Mapping,
    ):
        raise TypeError(
            "Initial source snapshot has invalid hash mapping."
        )

    for relative_name, expected_hash in expected_hashes.items():
        path = (
            ROOT
            / relative_name
        )

        if (
            path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(
                "Required source file disappeared or became symlinked: "
                f"{relative_name}"
            )

        actual = file_sha256(
            path
        )

        if actual != expected_hash:
            raise RuntimeError(
                "Required source file changed during campaign: "
                f"{relative_name}"
            )


def load_pilot() -> tuple[bytes, dict[str, Any]]:
    path = (
        ROOT
        / CONFIG_PATH
    )

    raw = path.read_bytes()

    actual_hash = hashlib.sha256(
        raw
    ).hexdigest()

    if actual_hash != SUPPORTED_CONFIG_SHA256:
        raise ValueError(
            "Unsupported dynamical pilot configuration SHA256: "
            f"{actual_hash}."
        )

    config = json.loads(
        raw.decode(
            "utf-8"
        ),
        parse_constant=lambda value: (
            (_ for _ in ()).throw(
                ValueError(
                    f"Nonfinite JSON constant: {value}"
                )
            )
        ),
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Pilot configuration must be a JSON object."
        )

    if config.get(
        "campaign_id"
    ) != "dynamical_original_iid_pilot_v1":
        raise ValueError(
            "Unexpected campaign_id."
        )

    if config.get(
        "campaign_role"
    ) != "pilot":
        raise ValueError(
            "Dynamical campaign must remain explicitly labelled pilot."
        )

    if (
        config.get(
            "certificate",
            {},
        ).get(
            "implementation_id"
        )
        != CONFIG_CERTIFICATE_IMPLEMENTATION_ID
    ):
        raise ValueError(
            "Certificate implementation identifier mismatch."
        )

    if (
        config.get(
            "pair_sampling",
            {},
        ).get(
            "implementation_id"
        )
        != PAIR_LAW_ID
    ):
        raise ValueError(
            "Independent-pair implementation identifier mismatch."
        )

    if (
        config.get(
            "reference_class_map",
            {},
        ).get(
            "implementation_id"
        )
        != REFERENCE_CLASS_MAP_ID
    ):
        raise ValueError(
            "Reference-class implementation identifier mismatch."
        )

    if (
        config.get(
            "frozen_representation",
            {},
        ).get(
            "format_id"
        )
        != FORMAT_ID
    ):
        raise ValueError(
            "Frozen-representation format identifier mismatch."
        )

    return raw, config


def build_schedule(
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

    systems = pilot[
        "systems"
    ]

    modes = pair_sampling[
        "evaluation_modes"
    ]

    sample_sizes = pilot[
        "sample_sizes"
    ]

    replicates = int(
        pilot[
            "replicates_per_cell"
        ]
    )

    jobs: list[
        dict[str, Any]
    ] = []

    for system in systems:
        system_key = int(
            seed_scheme[
                "system_keys"
            ][
                system
            ]
        )

        for mode in modes:
            mode_key = int(
                seed_scheme[
                    "mode_keys"
                ][
                    mode
                ]
            )

            for sample_size in sample_sizes:
                M = int(
                    sample_size
                )

                for replicate_index in range(
                    replicates
                ):
                    attempt_id = (
                        f"s{system_key}"
                        f"_q{mode_key}"
                        f"_m{M}"
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
                                M,
                            "replicate_index":
                                replicate_index,
                        }
                    )

    identifiers = [
        job[
            "attempt_id"
        ]
        for job
        in jobs
    ]

    if len(
        set(
            identifiers
        )
    ) != len(
        identifiers
    ):
        raise ValueError(
            "Duplicate scheduled attempt identifiers."
        )

    if len(
        jobs
    ) != EXPECTED_ATTEMPTS:
        raise ValueError(
            "Frozen pilot schedule has unexpected attempt count: "
            f"{len(jobs)} != {EXPECTED_ATTEMPTS}."
        )

    total_pairs = sum(
        int(
            job[
                "sample_size"
            ]
        )
        for job
        in jobs
    )

    if (
        total_pairs
        != EXPECTED_RETAINED_PAIRS
    ):
        raise ValueError(
            "Frozen pilot schedule has unexpected retained-pair count: "
            f"{total_pairs} != {EXPECTED_RETAINED_PAIRS}."
        )

    return jobs


def run_command_check(
    *,
    logs: Path,
    name: str,
    command: Sequence[str],
) -> None:
    write_json(
        logs
        / f"{name}.command.json",
        {
            "command":
                list(
                    command
                ),
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
        )
        as stdout,
        (
            logs
            / f"{name}.stderr.log"
        ).open(
            "xb"
        )
        as stderr,
    ):
        completed = subprocess.run(
            list(
                command
            ),
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
                int(
                    completed.returncode
                ),
        },
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Preflight {name} failed; "
            "see retained check logs."
        )


def run_preflight(
    output: Path,
) -> None:
    checks = (
        (
            "environment",
            (
                sys.executable,
                "scripts/verify_environment.py",
            ),
        ),
        (
            "pip_check",
            (
                sys.executable,
                "-m",
                "pip",
                "check",
            ),
        ),
        (
            "pip_freeze",
            (
                sys.executable,
                "-m",
                "pip",
                "freeze",
                "--all",
            ),
        ),
        (
            "kahkm_tests",
            (
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_kahkm_*.py",
                "-v",
            ),
        ),
        (
            "representation_runner_tests",
            (
                sys.executable,
                "-m",
                "unittest",
                "tests.test_build_dynamical_certificate_representations",
                "-v",
            ),
        ),
        (
            "representation_verifier_tests",
            (
                sys.executable,
                "-m",
                "unittest",
                "tests.test_verify_dynamical_frozen_representations",
                "-v",
            ),
        ),
    )

    logs = (
        output
        / "checks"
    )

    logs.mkdir()

    for name, command in checks:
        run_command_check(
            logs=logs,
            name=name,
            command=command,
        )

    packages: list[
        dict[str, Any]
    ] = []

    for line in (
        ROOT
        / "requirements-lock-arm64.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

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
            or not expected
        ):
            raise ValueError(
                "Unsupported lockfile entry: "
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
        {
            "packages":
                packages,
        },
    )

    if not all(
        item[
            "matches"
        ]
        for item
        in packages
    ):
        raise RuntimeError(
            "Installed packages differ from the reference lockfile."
        )


def _norm_budgets(
    *,
    config: Mapping[str, Any],
    spectral_norm: float,
) -> tuple[float, ...]:
    if (
        not np.isfinite(
            spectral_norm
        )
        or spectral_norm < 0.0
    ):
        raise ValueError(
            "Frozen spectral norm is invalid."
        )

    design = config[
        "norm_budgets"
    ]

    fixed = [
        float(
            value
        )
        for value
        in design[
            "fixed_constants"
        ]
    ]

    derived = []

    for name in design[
        "training_derived"
    ]:
        if (
            name
            == "nlms_spectral_norm"
        ):
            derived.append(
                float(
                    spectral_norm
                )
            )

        elif (
            name
            == "twice_nlms_spectral_norm"
        ):
            derived.append(
                float(
                    2.0
                    * spectral_norm
                )
            )

        else:
            raise ValueError(
                "Unsupported training-derived norm budget: "
                f"{name!r}."
            )

    values = (
        fixed
        + derived
    )

    if any(
        (
            not math.isfinite(
                value
            )
            or value < 0.0
        )
        for value
        in values
    ):
        raise ValueError(
            "Norm budgets must be finite and nonnegative."
        )

    if design.get(
        "deduplicate_exact_binary64"
    ) is not True:
        raise ValueError(
            "Frozen pilot requires exact-binary64 budget deduplication."
        )

    unique = set(
        values
    )

    if design.get(
        "sort_ascending"
    ) is not True:
        raise ValueError(
            "Frozen pilot requires ascending norm budgets."
        )

    return tuple(
        sorted(
            unique
        )
    )


def restore_models(
    *,
    output: Path,
    config: Mapping[str, Any],
) -> tuple[
    RestoredFrozenRepresentations,
    dict[str, ModelBundle],
]:
    restoration_dir = (
        output
        / "frozen_source"
    )

    restored = (
        restore_repository_frozen_representations(
            repository_root=
                ROOT,
            restoration_dir=
                restoration_dir,
        )
    )

    if (
        restored.archive_sha256
        != EXPECTED_ARCHIVE_SHA256
    ):
        raise RuntimeError(
            "Restored model source archive fingerprint mismatch."
        )

    artifact_paths = {
        "duffing":
            restored.duffing_artifact_dir,
        "vanderpol":
            restored.vanderpol_artifact_dir,
    }

    systems_config = (
        config[
            "frozen_representation"
        ][
            "systems"
        ]
    )

    bundles: dict[
        str,
        ModelBundle
    ] = {}

    model_records: dict[
        str,
        Any
    ] = {}

    for system in (
        "duffing",
        "vanderpol",
    ):
        loaded = (
            load_frozen_representation(
                artifact_paths[
                    system
                ],
                verify_hashes=True,
            )
        )

        frozen = systems_config[
            system
        ]

        if (
            loaded.abstraction_model[
                "n_clusters"
            ]
            != int(
                frozen[
                    "n_clusters"
                ]
            )
        ):
            raise RuntimeError(
                f"Restored n_clusters mismatch for {system}."
            )

        if (
            loaded.omega
            != float(
                frozen[
                    "omega"
                ]
            )
        ):
            raise RuntimeError(
                f"Restored omega mismatch for {system}."
            )

        if (
            loaded.tau
            != float(
                frozen[
                    "tau"
                ]
            )
        ):
            raise RuntimeError(
                f"Restored tau mismatch for {system}."
            )

        spectral_norm = float(
            np.linalg.norm(
                loaded.B,
                ord=2,
            )
        )

        training = (
            loaded.manifest[
                "training_metadata"
            ]
        )

        recorded_norm = float(
            training[
                "nlms_spectral_norm"
            ]
        )

        if not np.isclose(
            spectral_norm,
            recorded_norm,
            rtol=1e-13,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"Restored spectral norm mismatch for {system}."
            )

        kappas = _norm_budgets(
            config=config,
            spectral_norm=
                spectral_norm,
        )

        bundles[
            system
        ] = ModelBundle(
            system=
                system,
            loaded=
                loaded,
            spectral_norm=
                spectral_norm,
            kappas=
                kappas,
        )

        model_records[
            system
        ] = {
            "artifact_dir":
                artifact_paths[
                    system
                ],
            "n_clusters":
                int(
                    loaded.abstraction_model[
                        "n_clusters"
                    ]
                ),
            "omega":
                float(
                    loaded.omega
                ),
            "tau":
                float(
                    loaded.tau
                ),
            "nlms_spectral_norm":
                spectral_norm,
            "kappas":
                list(
                    kappas
                ),
            "train_closure_error":
                float(
                    loaded.train_closure_error
                ),
            "association_r2":
                float(
                    loaded.association_r2
                ),
        }

    write_json(
        output
        / "frozen_model_source.json",
        {
            "restoration":
                asdict(
                    restored
                ),
            "models":
                model_records,
            "evaluation_id":
                DYNAMICAL_EVALUATION_ID,
            "sampling_id":
                DYNAMICAL_SAMPLING_ID,
            "sample_evidence_id":
                SAMPLE_EVIDENCE_ID,
            "evaluation_evidence_id":
                EVALUATION_EVIDENCE_ID,
        },
    )

    return (
        restored,
        bundles,
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
        "system":
            evaluation_metadata[
                "system"
            ],
        "evaluation_mode":
            evaluation_metadata[
                "evaluation_mode"
            ],
        "sample_size":
            evaluation_metadata[
                "sample_size"
            ],
        "replicate_index":
            evaluation_metadata[
                "replicate_index"
            ],
        "sampling_seconds":
            float(
                sampling_seconds
            ),
        "evaluation_seconds":
            float(
                evaluation_seconds
            ),
        "f_hat":
            evaluation_metadata[
                "statistics"
            ][
                "f_hat"
            ],
        "s_hat":
            evaluation_metadata[
                "statistics"
            ][
                "s_hat"
            ],
        "frozen_predictor_spectral_norm":
            evaluation_metadata[
                "frozen_predictor_spectral_norm"
            ],
        "frozen_predictor_mse":
            evaluation_metadata[
                "frozen_predictor_mse"
            ],
        "frozen_predictor_rmse":
            evaluation_metadata[
                "frozen_predictor_rmse"
            ],
        "budget_certificates":
            evaluation_metadata[
                "budget_certificates"
            ],
    }


def execute_jobs(
    *,
    jobs: Sequence[
        Mapping[str, Any]
    ],
    models: Mapping[
        str,
        ModelBundle,
    ],
    config: Mapping[str, Any],
    output: Path,
    state: dict[str, Any],
    source_before: Mapping[str, Any],
) -> None:
    """Persist each sample before evaluation; never retry failed attempts."""

    attempts_dir = (
        output
        / "attempt_data"
    )

    attempts_dir.mkdir()

    pair_sampling = (
        config[
            "pair_sampling"
        ]
    )

    systems_config = (
        config[
            "frozen_representation"
        ][
            "systems"
        ]
    )

    delta = float(
        config[
            "certificate"
        ][
            "delta"
        ]
    )

    batch_size = int(
        config[
            "frozen_representation"
        ][
            "batch_size"
        ]
    )

    n_jobs = int(
        config[
            "frozen_representation"
        ][
            "n_jobs"
        ]
    )

    with ExitStack() as stack:
        files = {
            name:
                stack.enter_context(
                    (
                        output
                        / name
                    ).open(
                        "x",
                        encoding="utf-8",
                        newline="\n",
                    )
                )
            for name
            in (
                "attempts.jsonl",
                "samples.jsonl",
                "results.jsonl",
            )
        }

        events = files[
            "attempts.jsonl"
        ]

        samples = files[
            "samples.jsonl"
        ]

        results = files[
            "results.jsonl"
        ]

        for job in jobs:
            assert_source_unchanged(
                source_before
            )

            identifier = str(
                job[
                    "attempt_id"
                ]
            )

            state[
                "active_attempt_id"
            ] = identifier

            state[
                "started"
            ] += 1

            append_record(
                events,
                {
                    "attempt_id":
                        identifier,
                    "event":
                        "started",
                    "utc":
                        utc_now(),
                },
            )

            system = str(
                job[
                    "system"
                ]
            )

            mode = str(
                job[
                    "evaluation_mode"
                ]
            )

            M = int(
                job[
                    "sample_size"
                ]
            )

            replicate_index = int(
                job[
                    "replicate_index"
                ]
            )

            frozen = (
                systems_config[
                    system
                ]
            )

            sample_started = (
                time.monotonic()
            )

            stage = "sampling"

            try:
                dataset = (
                    sample_dynamical_dataset(
                        system=
                            system,
                        evaluation_mode=
                            mode,
                        sample_size=
                            M,
                        replicate_index=
                            replicate_index,
                        horizon_steps=
                            int(
                                pair_sampling[
                                    "horizon_steps"
                                ]
                            ),
                        dt=
                            float(
                                frozen[
                                    "dt"
                                ]
                            ),
                        vanderpol_mu=
                            float(
                                frozen[
                                    "vanderpol_mu"
                                ]
                            ),
                    )
                )

                sampling_seconds = float(
                    time.monotonic()
                    - sample_started
                )

            except Exception as exc:
                state[
                    "failed"
                ] += 1

                state[
                    "failed_sampling"
                ] += 1

                append_record(
                    events,
                    {
                        "attempt_id":
                            identifier,
                        "event":
                            "failed",
                        "stage":
                            stage,
                        "utc":
                            utc_now(),
                        "stage_seconds":
                            float(
                                time.monotonic()
                                - sample_started
                            ),
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
                    "active_attempt_id"
                ] = None

                continue

            attempt_relative = (
                Path(
                    "attempt_data"
                )
                / identifier
            )

            attempt_dir = (
                output
                / attempt_relative
            )

            # Deliberately outside the sampling exception handler:
            # sample-evidence persistence failure aborts the campaign.
            sample_metadata = (
                write_sample_evidence(
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
                        attempt_relative.as_posix(),
                    "sampling_seconds":
                        sampling_seconds,
                    "array_sha256":
                        sample_metadata[
                            "array_sha256"
                        ],
                    "system":
                        system,
                    "evaluation_mode":
                        mode,
                    "sample_size":
                        M,
                    "replicate_index":
                        replicate_index,
                },
            )

            state[
                "sampled"
            ] += 1

            assert_source_unchanged(
                source_before
            )

            evaluation_started = (
                time.monotonic()
            )

            stage = "evaluation"

            try:
                model = models[
                    system
                ]

                evaluation = (
                    evaluate_frozen_dynamical_pairs(
                        abstraction_model=
                            model.loaded.abstraction_model,
                        B=
                            model.loaded.B,
                        omega=
                            model.loaded.omega,
                        tau=
                            model.loaded.tau,
                        current_states=
                            dataset.current_states,
                        successor_states=
                            dataset.successor_states,
                        kappas=
                            model.kappas,
                        delta=
                            delta,
                        n_jobs=
                            n_jobs,
                        batch_size=
                            batch_size,
                    )
                )

                evaluation_seconds = float(
                    time.monotonic()
                    - evaluation_started
                )

            except Exception as exc:
                state[
                    "failed"
                ] += 1

                state[
                    "failed_evaluation"
                ] += 1

                append_record(
                    events,
                    {
                        "attempt_id":
                            identifier,
                        "event":
                            "failed",
                        "stage":
                            stage,
                        "utc":
                            utc_now(),
                        "stage_seconds":
                            float(
                                time.monotonic()
                                - evaluation_started
                            ),
                        "sample_persisted":
                            True,
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
                    "active_attempt_id"
                ] = None

                continue

            assert_source_unchanged(
                source_before
            )

            # Deliberately outside the evaluation exception handler:
            # evaluation-evidence persistence failure aborts the campaign.
            evaluation_metadata = (
                write_evaluation_evidence(
                    attempt_dir=
                        attempt_dir,
                    dataset=
                        dataset,
                    evaluation=
                        evaluation,
                )
            )

            compact = (
                _compact_result_record(
                    attempt_id=
                        identifier,
                    attempt_relative_dir=
                        attempt_relative.as_posix(),
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
                compact,
            )

            append_record(
                events,
                {
                    "attempt_id":
                        identifier,
                    "event":
                        "completed",
                    "utc":
                        utc_now(),
                    "sampling_seconds":
                        sampling_seconds,
                    "evaluation_seconds":
                        evaluation_seconds,
                },
            )

            state[
                "evaluated"
            ] += 1

            state[
                "succeeded"
            ] += 1

            state[
                "active_attempt_id"
            ] = None


def write_checksums(
    output: Path,
) -> None:
    files = sorted(
        path
        for path
        in output.rglob("*")
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS"
        )
    )

    with (
        output
        / "SHA256SUMS"
    ).open(
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
    output = (
        output
        .expanduser()
        .resolve()
    )

    if (
        output == ROOT
        or ROOT in output.parents
    ):
        raise ValueError(
            "Choose a new output directory outside this working tree."
        )

    if output.exists():
        raise FileExistsError(
            "Output already exists; it will not be reused or overwritten."
        )

    raw, config = load_pilot()

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

    started = (
        time.monotonic()
    )

    state: dict[str, Any] = {
        "status":
            "initializing",
        "planned":
            len(
                jobs
            ),
        "planned_retained_pairs":
            sum(
                int(
                    job[
                        "sample_size"
                    ]
                )
                for job
                in jobs
            ),
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
        "failed_sampling":
            0,
        "failed_evaluation":
            0,
        "active_attempt_id":
            None,
        "started_utc":
            utc_now(),
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
                    "pilot",
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
                "source":
                    source_before,
                "representation_archive_sha256":
                    EXPECTED_ARCHIVE_SHA256,
                "python_executable":
                    sys.executable,
                "python_version":
                    sys.version,
                "platform":
                    platform.platform(),
                "architecture":
                    platform.machine(),
                "argv":
                    list(
                        sys.argv
                    ),
                "process_id":
                    os.getpid(),
                "started_utc":
                    state[
                        "started_utc"
                    ],
                "expected_attempts":
                    EXPECTED_ATTEMPTS,
                "expected_retained_pairs":
                    EXPECTED_RETAINED_PAIRS,
                "numerical_environment": {
                    name:
                        os.environ.get(
                            name
                        )
                    for name
                    in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "PYTHONHASHSEED",
                    )
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
        ] = "model_restore"

        (
            _restoration,
            models,
        ) = restore_models(
            output=
                output,
            config=
                config,
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
            models=
                models,
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
        if str(
            state[
                "status"
            ]
        ).startswith(
            "completed"
        ):
            state[
                "status"
            ] = "invalid_source"

        exit_code = 2

    write_json(
        output
        / "source_after.json",
        source_after,
    )

    state.update(
        ended_utc=
            utc_now(),
        elapsed_seconds=
            float(
                time.monotonic()
                - started
            ),
        exit_code=
            int(
                exit_code
            ),
        source_unchanged=
            bool(
                source_unchanged
            ),
        not_started=
            state[
                "planned"
            ]
            - state[
                "started"
            ],
        started_without_terminal_event=
            state[
                "started"
            ]
            - state[
                "succeeded"
            ]
            - state[
                "failed"
            ],
        sampled_without_success=
            state[
                "sampled"
            ]
            - state[
                "succeeded"
            ],
        independent_evidence_verification=
            "not_yet_performed",
    )

    write_json(
        output
        / "campaign_summary.json",
        state,
    )

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

    return int(
        exit_code
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "New run directory outside the source working tree."
        ),
    )

    args = parser.parse_args()

    try:
        return run_campaign(
            args.output
        )

    except Exception as exc:
        print(
            f"STOP: {type(exc).__name__}: {exc}",
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
