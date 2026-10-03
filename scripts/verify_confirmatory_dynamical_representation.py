#!/usr/bin/env python3
"""Independent numerical verifier for confirmatory representation evidence.

The verifier does not modify retained build evidence and does not generate
certificate-evaluation pairs.

It independently:
- verifies the complete retained checksum inventory;
- verifies exact retained config/protocol bytes;
- verifies build-runner provenance and source commit;
- restores the earlier independently verified Van der Pol abstraction;
- checks that state-space centers and all autoencoder shards were reused
  byte-for-byte;
- regenerates the original training trajectories from seeds 0, 1, 2;
- recomputes Phi_128 and Chi_128;
- reruns the manuscript NLMS recursion from identity for 20 epochs;
- compares the recomputed operator, history, spectral norm, closure error,
  and association R2 against the retained artifact.
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
from typing import Any

import numpy as np

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from experiment_15_external_koopman_baselines import (
    make_concatenated_training_data,
)
from kahkm_confirmatory_representation import (
    CONFIRMATORY_CONFIG_RELATIVE,
    CONFIRMATORY_CONFIG_SHA256,
    CONFIRMATORY_PROTOCOL_RELATIVE,
    CONFIRMATORY_PROTOCOL_SHA256,
    DERIVATION_ID,
    EXPECTED_SOURCE_ARCHIVE_SHA256,
    EXPECTED_SOURCE_BUILD_COMMIT,
    EXPECTED_SOURCE_CONFIG_SHA256,
    EXPECTED_SOURCE_VERIFIER_COMMIT,
)
from kahkm_dynamical_frozen_source import (
    restore_repository_frozen_representations,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
    nlms_koopman_closure,
)


VERIFIER_ID = (
    "confirmatory_dynamical_representation_independent_numerics_v1"
)

EXPECTED_RUNNER_ID = (
    "confirmatory_dynamical_representation_build_v1"
)

EXPECTED_EXECUTION_COMMIT = (
    "4a0cedd557a6e06d331aa31b91223629c36d1997"
)

EXPECTED_CAMPAIGN_ID = (
    "dynamical_original_iid_confirmatory_v1"
)

EXPECTED_CAMPAIGN_ROLE = (
    "confirmatory"
)

EXPECTED_SYSTEM = (
    "vanderpol"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "c9711cac73f38c166c54025e9805c4b199e575131cb92d8e914af475e4cd2e2c"
)

EXPECTED_N_CLUSTERS = 25
EXPECTED_N_TRAIN_SNAPSHOTS = 3600
EXPECTED_OMEGA = 128.0
EXPECTED_TAU = 1e-6
EXPECTED_BETA = 0.1
EXPECTED_EPOCHS = 20

REQUIRED_VERIFIER_SOURCE_FILES = (
    "scripts/verify_confirmatory_dynamical_representation.py",
    "test_verify_confirmatory_dynamical_representation.py",
    "kahkm_confirmatory_representation.py",
    "kahkm_dynamical_frozen_source.py",
    "kahkm_frozen_representation.py",
    "kernel_affine_hull_koopman_machines.py",
    "experiment_15_external_koopman_baselines.py",
    CONFIRMATORY_CONFIG_RELATIVE,
    CONFIRMATORY_PROTOCOL_RELATIVE,
    (
        "reproduction/prospective_campaigns/"
        "dynamical_frozen_representations_pilot_v1_run001/"
        "ARCHIVE.json"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_frozen_representations_pilot_v1_run001/"
        "README.md"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_frozen_representations_pilot_v1_run001/"
        "SHA256SUMS"
    ),
    (
        "reproduction/prospective_campaigns/"
        "dynamical_frozen_representations_pilot_v1_run001/"
        "evidence.tar.gz"
    ),
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: str | os.PathLike[str],
) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024
                * 1024
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


def strict_json_load(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def strict_json_dump(
    path: Path,
    value: Any,
) -> None:
    serialized = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    with path.open(
        "xb"
    ) as handle:
        handle.write(
            serialized
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )


def verifier_source_snapshot() -> dict[str, Any]:
    actual_root = Path(
        git_bytes(
            "rev-parse",
            "--show-toplevel",
        )
        .decode()
        .strip()
    ).resolve()

    if actual_root != ROOT:
        raise RuntimeError(
            "Verifier must execute from repository root."
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

    hashes: dict[
        str,
        str,
    ] = {}

    for relative_name in sorted(
        set(
            REQUIRED_VERIFIER_SOURCE_FILES
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
                "Missing or symlinked verifier source file: "
                f"{relative_name}"
            )

        actual = sha256_file(
            path
        )

        try:
            committed_bytes = git_bytes(
                "show",
                f"{commit}:{relative_name}",
            )

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Verifier source file is not committed at HEAD: "
                f"{relative_name}"
            ) from exc

        committed = hashlib.sha256(
            committed_bytes
        ).hexdigest()

        if actual != committed:
            raise RuntimeError(
                "Verifier source bytes differ from HEAD: "
                f"{relative_name}"
            )

        hashes[
            relative_name
        ] = actual

    return {
        "verifier_source_commit":
            commit,

        "git_status_porcelain":
            status,

        "source_file_sha256":
            hashes,
    }


def verify_checksum_inventory(
    evidence: Path,
) -> dict[str, Any]:
    checksum_path = (
        evidence
        / "SHA256SUMS"
    )

    if (
        checksum_path.is_symlink()
        or not checksum_path.is_file()
    ):
        raise RuntimeError(
            "Evidence SHA256SUMS is missing or symlinked."
        )

    expected: dict[
        str,
        str,
    ] = {}

    for line_number, line in enumerate(
        checksum_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line:
            continue

        digest, separator, relative_name = (
            line.partition(
                "  "
            )
        )

        if (
            not separator
            or len(
                digest
            ) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                "Malformed SHA256SUMS line "
                f"{line_number}."
            )

        relative = Path(
            relative_name
        )

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative
            == Path(
                "."
            )
        ):
            raise ValueError(
                "Unsafe SHA256SUMS path: "
                f"{relative_name!r}."
            )

        normalized = (
            relative.as_posix()
        )

        if normalized in expected:
            raise ValueError(
                "Duplicate SHA256SUMS entry: "
                f"{normalized}"
            )

        expected[
            normalized
        ] = digest

    if not expected:
        raise ValueError(
            "SHA256SUMS is empty."
        )

    actual_paths = {
        path.relative_to(
            evidence
        ).as_posix()
        for path
        in evidence.rglob(
            "*"
        )
        if path.is_file()
        and path.name
        != "SHA256SUMS"
    }

    if set(
        expected
    ) != actual_paths:
        raise ValueError(
            "Top-level evidence inventory mismatch; "
            f"unlisted={sorted(actual_paths - set(expected))}, "
            f"missing={sorted(set(expected) - actual_paths)}."
        )

    for relative_name, expected_hash in expected.items():
        path = (
            evidence
            / relative_name
        )

        if path.is_symlink():
            raise ValueError(
                "Evidence file must not be a symlink: "
                f"{relative_name}"
            )

        actual_hash = (
            sha256_file(
                path
            )
        )

        if (
            actual_hash
            != expected_hash
        ):
            raise ValueError(
                "Top-level SHA256 mismatch for "
                f"{relative_name}: "
                f"{actual_hash} != {expected_hash}."
            )

    return {
        "verified_file_count":
            len(
                expected
            ),

        "sha256sums_fingerprint":
            sha256_file(
                checksum_path
            ),
    }


def verify_configuration_and_protocol(
    evidence: Path,
) -> dict[str, Any]:
    retained_config = (
        evidence
        / "configuration.json"
    )

    retained_protocol = (
        evidence
        / "protocol.md"
    )

    if not retained_config.is_file():
        raise FileNotFoundError(
            "Retained configuration.json is missing."
        )

    if not retained_protocol.is_file():
        raise FileNotFoundError(
            "Retained protocol.md is missing."
        )

    repository_config = (
        ROOT
        / CONFIRMATORY_CONFIG_RELATIVE
    )

    repository_protocol = (
        ROOT
        / CONFIRMATORY_PROTOCOL_RELATIVE
    )

    if (
        sha256_file(
            repository_config
        )
        != CONFIRMATORY_CONFIG_SHA256
    ):
        raise ValueError(
            "Repository confirmatory config fingerprint changed."
        )

    if (
        sha256_file(
            retained_config
        )
        != CONFIRMATORY_CONFIG_SHA256
    ):
        raise ValueError(
            "Retained confirmatory config fingerprint is wrong."
        )

    if (
        repository_config.read_bytes()
        != retained_config.read_bytes()
    ):
        raise ValueError(
            "Retained config is not byte-identical to repository config."
        )

    if (
        sha256_file(
            repository_protocol
        )
        != CONFIRMATORY_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Repository confirmatory protocol fingerprint changed."
        )

    if (
        sha256_file(
            retained_protocol
        )
        != CONFIRMATORY_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Retained confirmatory protocol fingerprint is wrong."
        )

    if (
        repository_protocol.read_bytes()
        != retained_protocol.read_bytes()
    ):
        raise ValueError(
            "Retained protocol is not byte-identical to repository protocol."
        )

    payload = strict_json_load(
        retained_config
    )

    if (
        payload.get(
            "campaign_id"
        )
        != EXPECTED_CAMPAIGN_ID
    ):
        raise ValueError(
            "Unexpected retained campaign ID."
        )

    if (
        payload.get(
            "campaign_role"
        )
        != EXPECTED_CAMPAIGN_ROLE
    ):
        raise ValueError(
            "Unexpected retained campaign role."
        )

    return {
        "configuration_sha256":
            CONFIRMATORY_CONFIG_SHA256,

        "protocol_sha256":
            CONFIRMATORY_PROTOCOL_SHA256,

        "repository_evidence_bytes_identical":
            True,
    }


def verify_build_summary(
    evidence: Path,
) -> dict[str, Any]:
    summary = strict_json_load(
        evidence
        / "build_summary.json"
    )

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(
            "build_summary.json must contain an object."
        )

    expected = {
        "runner_id":
            EXPECTED_RUNNER_ID,

        "campaign_id":
            EXPECTED_CAMPAIGN_ID,

        "campaign_role":
            EXPECTED_CAMPAIGN_ROLE,

        "status":
            "completed",

        "execution_source_commit":
            EXPECTED_EXECUTION_COMMIT,

        "configuration_sha256":
            CONFIRMATORY_CONFIG_SHA256,

        "protocol_sha256":
            CONFIRMATORY_PROTOCOL_SHA256,

        "representation_derivation_id":
            DERIVATION_ID,

        "num_systems_planned":
            1,

        "num_systems_built":
            1,

        "systems_built": [
            EXPECTED_SYSTEM
        ],

        "state_space_kmeans_refits":
            0,

        "autoencoder_refits":
            0,

        "nlms_operator_refits":
            1,

        "random_evaluation_pairs_generated":
            0,

        "certificate_evaluation_pairs_generated":
            0,

        "source_unchanged":
            True,

        "exit_code":
            0,
    }

    for key, expected_value in expected.items():
        actual = summary.get(
            key
        )

        if (
            actual
            != expected_value
        ):
            raise ValueError(
                "Build summary mismatch for "
                f"{key}: {actual!r} != {expected_value!r}."
            )

    records = summary.get(
        "build_records"
    )

    if (
        not isinstance(
            records,
            list,
        )
        or len(
            records
        ) != 1
    ):
        raise ValueError(
            "Build summary must contain exactly one build record."
        )

    record = records[
        0
    ]

    if (
        record.get(
            "system"
        )
        != EXPECTED_SYSTEM
    ):
        raise ValueError(
            "Build record system mismatch."
        )

    if (
        record.get(
            "source_archive_sha256"
        )
        != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError(
            "Build record source archive mismatch."
        )

    if (
        record.get(
            "source_manifest_sha256"
        )
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "Build record source manifest mismatch."
        )

    return {
        "execution_source_commit":
            summary[
                "execution_source_commit"
            ],

        "source_unchanged":
            True,

        "certificate_evaluation_pairs_generated":
            0,
    }


def _relative_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    numerator = float(
        np.sum(
            (
                target
                - prediction
            ) ** 2,
            dtype=np.float64,
        )
    )

    denominator = float(
        np.sum(
            target
            * target,
            dtype=np.float64,
        )
    )

    return (
        numerator
        / max(
            denominator,
            1e-12,
        )
    )


def _association_r2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    residual_ss = float(
        np.sum(
            (
                target
                - prediction
            ) ** 2,
            dtype=np.float64,
        )
    )

    centered = (
        target
        - np.mean(
            target,
            axis=1,
            keepdims=True,
        )
    )

    total_ss = float(
        np.sum(
            centered
            * centered,
            dtype=np.float64,
        )
    )

    if total_ss <= 0.0:
        raise ValueError(
            "Training target variance must be positive."
        )

    return (
        1.0
        - residual_ss
        / total_ss
    )


def verify_representation_numerics(
    evidence: Path,
    scratch: Path,
) -> dict[str, Any]:
    artifact = (
        evidence
        / "representations"
        / EXPECTED_SYSTEM
    )

    loaded = (
        load_frozen_representation(
            artifact,
            verify_hashes=True,
        )
    )

    if (
        loaded.format_id
        != FORMAT_ID
    ):
        raise ValueError(
            "Wrong frozen representation format."
        )

    if (
        loaded.system
        != EXPECTED_SYSTEM
    ):
        raise ValueError(
            "Wrong frozen representation system."
        )

    if (
        loaded.omega
        != EXPECTED_OMEGA
    ):
        raise ValueError(
            "Wrong confirmatory omega."
        )

    if (
        loaded.tau
        != EXPECTED_TAU
    ):
        raise ValueError(
            "Wrong confirmatory tau."
        )

    manifest = loaded.manifest

    training = manifest.get(
        "training_metadata"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise TypeError(
            "training_metadata is missing."
        )

    expected_training = {
        "derivation_id":
            DERIVATION_ID,

        "configuration_sha256":
            CONFIRMATORY_CONFIG_SHA256,

        "protocol_sha256":
            CONFIRMATORY_PROTOCOL_SHA256,

        "source_representation_archive_sha256":
            EXPECTED_SOURCE_ARCHIVE_SHA256,

        "source_representation_manifest_sha256":
            EXPECTED_SOURCE_MANIFEST_SHA256,

        "source_representation_build_execution_commit":
            EXPECTED_SOURCE_BUILD_COMMIT,

        "source_representation_verifier_source_commit":
            EXPECTED_SOURCE_VERIFIER_COMMIT,

        "source_representation_verification_status":
            "passed",

        "source_omega":
            4.0,

        "source_tau":
            1e-6,

        "omega":
            128.0,

        "tau":
            1e-6,

        "train_seeds": [
            0,
            1,
            2,
        ],

        "n_steps_train":
            1200,

        "n_train_snapshots":
            EXPECTED_N_TRAIN_SNAPSHOTS,

        "n_clusters":
            EXPECTED_N_CLUSTERS,

        "beta":
            EXPECTED_BETA,

        "nlms_epochs":
            EXPECTED_EPOCHS,

        "nlms_shuffle":
            False,

        "nlms_initial_matrix":
            "identity",

        "predictor_orientation":
            "B.T @ Phi",

        "project_stochastic":
            False,

        "reuse_state_space_kmeans_centers":
            True,

        "reuse_kahm_autoencoders":
            True,

        "refit_kmeans":
            False,

        "refit_autoencoders":
            False,
    }

    for key, expected_value in expected_training.items():
        actual = training.get(
            key
        )

        if (
            actual
            != expected_value
        ):
            raise ValueError(
                "Training metadata mismatch for "
                f"{key}: {actual!r} != {expected_value!r}."
            )

    restored = (
        restore_repository_frozen_representations(
            repository_root=
                ROOT,

            restoration_dir=
                scratch
                / "source_representation",
        )
    )

    if (
        restored.archive_sha256
        != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError(
            "Source archive SHA mismatch."
        )

    if (
        restored.configuration_sha256
        != EXPECTED_SOURCE_CONFIG_SHA256
    ):
        raise ValueError(
            "Source config SHA mismatch."
        )

    source = (
        load_frozen_representation(
            restored.vanderpol_artifact_dir,
            verify_hashes=True,
        )
    )

    source_manifest_sha = (
        sha256_file(
            Path(
                restored.vanderpol_artifact_dir
            )
            / "manifest.json"
        )
    )

    if (
        source_manifest_sha
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "Source manifest SHA mismatch."
        )

    for key in (
        "cluster_centers",
        "cluster_centers_init",
    ):
        if not np.array_equal(
            source.abstraction_model[
                key
            ],
            loaded.abstraction_model[
                key
            ],
        ):
            raise ValueError(
                f"Frozen abstraction array changed: {key}."
            )

    source_entries = source.manifest.get(
        "classifier_entries"
    )

    retained_entries = manifest.get(
        "classifier_entries"
    )

    if (
        source_entries
        != retained_entries
    ):
        raise ValueError(
            "Classifier entry inventory changed."
        )

    if (
        not isinstance(
            source_entries,
            list,
        )
        or len(
            source_entries
        ) != EXPECTED_N_CLUSTERS
    ):
        raise ValueError(
            "Unexpected classifier shard count."
        )

    source_hashes = source.manifest.get(
        "files_sha256"
    )

    retained_hashes = manifest.get(
        "files_sha256"
    )

    if (
        not isinstance(
            source_hashes,
            dict,
        )
        or not isinstance(
            retained_hashes,
            dict,
        )
    ):
        raise TypeError(
            "Internal representation hash inventory missing."
        )

    for entry in source_entries:
        relative = (
            "autoencoders/"
            + entry
        )

        if (
            source_hashes.get(
                relative
            )
            != retained_hashes.get(
                relative
            )
        ):
            raise ValueError(
                "Autoencoder shard differs from source: "
                f"{relative}"
            )

    source_model_meta = dict(
        source.manifest[
            "model_metadata"
        ]
    )

    retained_model_meta = dict(
        manifest[
            "model_metadata"
        ]
    )

    source_omega = source_model_meta.pop(
        "kahkm_omega"
    )

    retained_omega = retained_model_meta.pop(
        "kahkm_omega"
    )

    if source_omega != 4.0:
        raise ValueError(
            "Unexpected source model omega."
        )

    if retained_omega != 128.0:
        raise ValueError(
            "Unexpected retained model omega."
        )

    if (
        source_model_meta
        != retained_model_meta
    ):
        raise ValueError(
            "Model metadata changed beyond omega."
        )

    X0, X1 = (
        make_concatenated_training_data(
            EXPECTED_SYSTEM,
            train_seeds=(
                0,
                1,
                2,
            ),
            n_steps_train=
                1200,
            dt=
                0.02,
            vanderpol_mu=
                1.0,
        )
    )

    X0 = np.asarray(
        X0,
        dtype=np.float64,
    )

    X1 = np.asarray(
        X1,
        dtype=np.float64,
    )

    if (
        X0.shape
        != (
            2,
            EXPECTED_N_TRAIN_SNAPSHOTS,
        )
        or X1.shape
        != X0.shape
    ):
        raise ValueError(
            "Regenerated training snapshot shape mismatch."
        )

    recompute_model = dict(
        source.abstraction_model
    )

    recompute_model[
        "kahkm_omega"
    ] = EXPECTED_OMEGA

    recompute_model[
        "kahkm_tau"
    ] = EXPECTED_TAU

    Phi = np.asarray(
        kahm_associations(
            recompute_model,
            X0,
            omega=
                EXPECTED_OMEGA,
            tau=
                EXPECTED_TAU,
            n_jobs=
                -1,
            batch_size=
                256,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    Chi = np.asarray(
        kahm_associations(
            recompute_model,
            X1,
            omega=
                EXPECTED_OMEGA,
            tau=
                EXPECTED_TAU,
            n_jobs=
                -1,
            batch_size=
                256,
            show_progress=False,
        ),
        dtype=np.float64,
    )

    if (
        Phi.shape
        != (
            EXPECTED_N_CLUSTERS,
            EXPECTED_N_TRAIN_SNAPSHOTS,
        )
        or Chi.shape
        != Phi.shape
    ):
        raise ValueError(
            "Recomputed association shape mismatch."
        )

    B_recomputed, history_recomputed = (
        nlms_koopman_closure(
            Phi,
            Chi,
            beta=
                EXPECTED_BETA,
            epochs=
                EXPECTED_EPOCHS,
            shuffle=False,
            random_state=0,
            B0=None,
        )
    )

    B_recomputed = np.asarray(
        B_recomputed,
        dtype=np.float64,
    )

    B_retained = np.asarray(
        loaded.B,
        dtype=np.float64,
    )

    if not np.array_equal(
        B_recomputed,
        B_retained,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    B_recomputed
                    - B_retained
                )
            )
        )

        raise ValueError(
            "Independently recomputed B differs from retained B; "
            f"max_abs={max_abs:.17g}."
        )

    retained_history = tuple(
        float(
            value
        )
        for value in loaded.nlms_history
    )

    recomputed_history = tuple(
        float(
            value
        )
        for value in history_recomputed
    )

    if (
        recomputed_history
        != retained_history
    ):
        raise ValueError(
            "Independently recomputed NLMS history differs."
        )

    prediction = (
        B_recomputed.T
        @ Phi
    )

    error_recomputed = (
        _relative_error(
            prediction,
            Chi,
        )
    )

    r2_recomputed = (
        _association_r2(
            prediction,
            Chi,
        )
    )

    norm_recomputed = float(
        np.linalg.norm(
            B_recomputed,
            ord=2,
        )
    )

    comparisons = (
        (
            "spectral norm",
            norm_recomputed,
            float(
                training[
                    "nlms_spectral_norm"
                ]
            ),
        ),
        (
            "train closure error",
            error_recomputed,
            float(
                loaded.train_closure_error
            ),
        ),
        (
            "association R2",
            r2_recomputed,
            float(
                loaded.association_r2
            ),
        ),
    )

    for name, recomputed, retained in comparisons:
        if (
            recomputed
            != retained
        ):
            raise ValueError(
                f"Recomputed {name} differs: "
                f"{recomputed!r} != {retained!r}."
            )

    coordinate_min = float(
        min(
            np.min(
                Phi
            ),
            np.min(
                Chi
            ),
        )
    )

    coordinate_max = float(
        max(
            np.max(
                Phi
            ),
            np.max(
                Chi
            ),
        )
    )

    simplex_error = float(
        max(
            np.max(
                np.abs(
                    np.sum(
                        Phi,
                        axis=0,
                        dtype=np.float64,
                    )
                    - 1.0
                )
            ),
            np.max(
                np.abs(
                    np.sum(
                        Chi,
                        axis=0,
                        dtype=np.float64,
                    )
                    - 1.0
                )
            ),
        )
    )

    if coordinate_min < 0.0:
        raise ValueError(
            "Recomputed training associations have negative coordinates."
        )

    if coordinate_max > 1.0:
        raise ValueError(
            "Recomputed training associations exceed one."
        )

    if simplex_error > 1e-12:
        raise ValueError(
            "Recomputed training associations violate simplex normalization."
        )

    return {
        "system":
            EXPECTED_SYSTEM,

        "n_clusters":
            EXPECTED_N_CLUSTERS,

        "n_train_snapshots":
            EXPECTED_N_TRAIN_SNAPSHOTS,

        "omega":
            EXPECTED_OMEGA,

        "tau":
            EXPECTED_TAU,

        "source_manifest_sha256":
            source_manifest_sha,

        "autoencoder_shards_verified":
            len(
                source_entries
            ),

        "cluster_center_arrays_verified":
            2,

        "training_trajectories_regenerated":
            3,

        "training_snapshot_pairs_regenerated":
            EXPECTED_N_TRAIN_SNAPSHOTS,

        "association_matrices_recomputed":
            2,

        "nlms_operator_recomputed_independently":
            True,

        "operator_exact_binary64_match":
            True,

        "nlms_history_exact_binary64_match":
            True,

        "nlms_spectral_norm":
            norm_recomputed,

        "train_closure_error":
            error_recomputed,

        "association_r2":
            r2_recomputed,

        "training_coordinate_min":
            coordinate_min,

        "training_coordinate_max":
            coordinate_max,

        "training_max_simplex_error":
            simplex_error,

        "certificate_evaluation_pairs_generated":
            0,
    }


def verify_no_leftover_work(
    evidence: Path,
) -> None:
    pattern = (
        f".{evidence.name}."
        "*.representation-work"
    )

    leftovers = sorted(
        evidence.parent.glob(
            pattern
        )
    )

    if leftovers:
        raise ValueError(
            "Leftover representation work directories found: "
            + ", ".join(
                str(
                    path
                )
                for path
                in leftovers
            )
        )


def verify_evidence(
    *,
    evidence_dir: str | os.PathLike[str],
    scratch_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    evidence = Path(
        evidence_dir
    ).expanduser().resolve()

    scratch = Path(
        scratch_dir
    ).expanduser().resolve()

    if not evidence.is_dir():
        raise FileNotFoundError(
            f"Evidence directory not found: {evidence}"
        )

    if scratch.exists():
        raise FileExistsError(
            f"Scratch directory already exists: {scratch}"
        )

    scratch.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        inventory = (
            verify_checksum_inventory(
                evidence
            )
        )

        configuration = (
            verify_configuration_and_protocol(
                evidence
            )
        )

        build_summary = (
            verify_build_summary(
                evidence
            )
        )

        representation = (
            verify_representation_numerics(
                evidence,
                scratch,
            )
        )

        verify_no_leftover_work(
            evidence
        )

        return {
            "verifier_id":
                VERIFIER_ID,

            "status":
                "passed",

            "verified_utc":
                utc_now(),

            "evidence_dir":
                str(
                    evidence
                ),

            **inventory,
            **configuration,
            **build_summary,

            "representation":
                representation,

            "num_representations":
                1,

            "random_evaluation_pairs_generated":
                0,

            "certificate_evaluation_pairs_generated":
                0,

            "state_space_kmeans_refits":
                0,

            "autoencoder_refits":
                0,

            "retained_representation_modified":
                False,
        }

    finally:
        if scratch.exists():
            import shutil

            shutil.rmtree(
                scratch,
                ignore_errors=True,
            )


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

    parser.add_argument(
        "--scratch-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if output.exists():
        print(
            "Verification FAILED: output already exists: "
            f"{output}",
            file=sys.stderr,
        )

        return 1

    try:
        source = (
            verifier_source_snapshot()
        )

        report = (
            verify_evidence(
                evidence_dir=
                    args.evidence,

                scratch_dir=
                    args.scratch_dir,
            )
        )

        final_source = (
            verifier_source_snapshot()
        )

        if (
            final_source
            != source
        ):
            raise RuntimeError(
                "Verifier/source state changed during verification."
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.mkdir(
            exist_ok=False
        )

        report[
            "verifier_source"
        ] = source

        strict_json_dump(
            output
            / "verification_report.json",
            report,
        )

        strict_json_dump(
            output
            / "verification_checksums.json",
            {
                "verification_report.json":
                    sha256_file(
                        output
                        / "verification_report.json"
                    )
            },
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
    raise SystemExit(
        main()
    )
