#!/usr/bin/env python3
"""Verify retained frozen dynamical KAHKM representations without refitting.

The verifier checks:

- top-level retained SHA256SUMS;
- complete top-level file inventory;
- frozen configuration bytes and SHA256;
- build-summary provenance and execution commit;
- both Duffing and Van der Pol representation manifests;
- representation-internal file inventories and hashes;
- cluster-center and operator dimensions;
- recorded versus recomputed NLMS spectral norm;
- portable loading with hash verification;
- association evaluation at the stored cluster centers;
- simplex-valued association diagnostics;
- absence of leftover temporary representation-build directories.

No representation fitting and no certificate-pair generation occur.
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
from typing import Any, Mapping

import numpy as np

from kahkm_dynamical_representation import (
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
    DYNAMICAL_PILOT_CONFIG_SHA256,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
)


ROOT = Path(__file__).resolve().parents[1]

VERIFIER_ID = "dynamical_frozen_representation_verifier_v1"

EXPECTED_RUNNER_ID = (
    "dynamical_certificate_representation_build_v1"
)

EXPECTED_EXECUTION_COMMIT = (
    "1add19b94372257d16118f60c69e1e8afa668259"
)

EXPECTED_SYSTEMS: dict[str, dict[str, Any]] = {
    "duffing": {
        "n_clusters": 10,
        "omega": 2.0,
        "tau": 1e-6,
        "n_train_snapshots": 3600,
        "state_dimension": 2,
    },
    "vanderpol": {
        "n_clusters": 25,
        "omega": 4.0,
        "tau": 1e-6,
        "n_train_snapshots": 3600,
        "state_dimension": 2,
    },
}

REQUIRED_VERIFIER_SOURCE_FILES = (
    "scripts/verify_dynamical_frozen_representations.py",
    "tests/test_verify_dynamical_frozen_representations.py",
    "kahkm_dynamical_representation.py",
    "kahkm_frozen_representation.py",
    "kernel_affine_hull_koopman_machines.py",
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
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
                        "Nonfinite JSON constant "
                        f"in {path}: {value}"
                    )
                )
            ),
        )


def strict_json_dump(
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
        handle.write(text)
        handle.flush()
        os.fsync(
            handle.fileno()
        )


def verifier_source_snapshot() -> dict[str, Any]:
    """Require clean committed verifier/source bytes."""
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
            "Verifier must execute from its repository root."
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

    lines = checksum_path.read_text(
        encoding="utf-8"
    ).splitlines()

    expected: dict[str, str] = {}

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line:
            continue

        digest, separator, relative_name = (
            line.partition("  ")
        )

        if (
            not separator
            or len(digest) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in digest
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
            or relative == Path(".")
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
        in evidence.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
    }

    if set(expected) != actual_paths:
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

        actual_hash = sha256_file(
            path
        )

        if actual_hash != expected_hash:
            raise ValueError(
                "Top-level SHA256 mismatch for "
                f"{relative_name}: "
                f"{actual_hash} != {expected_hash}."
            )

    return {
        "verified_file_count":
            len(expected),
        "sha256sums_fingerprint":
            sha256_file(
                checksum_path
            ),
    }


def verify_configuration(
    evidence: Path,
) -> dict[str, Any]:
    repository_config = (
        ROOT
        / DYNAMICAL_PILOT_CONFIG_RELATIVE
    )

    retained_config = (
        evidence
        / "configuration.json"
    )

    if not retained_config.is_file():
        raise FileNotFoundError(
            "Retained configuration.json is missing."
        )

    repository_hash = sha256_file(
        repository_config
    )

    retained_hash = sha256_file(
        retained_config
    )

    if (
        repository_hash
        != DYNAMICAL_PILOT_CONFIG_SHA256
    ):
        raise ValueError(
            "Repository frozen configuration fingerprint changed."
        )

    if (
        retained_hash
        != DYNAMICAL_PILOT_CONFIG_SHA256
    ):
        raise ValueError(
            "Retained configuration fingerprint is incorrect."
        )

    if (
        repository_config.read_bytes()
        != retained_config.read_bytes()
    ):
        raise ValueError(
            "Retained configuration is not byte-identical "
            "to the repository configuration."
        )

    payload = strict_json_load(
        retained_config
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Retained configuration must be a JSON object."
        )

    systems = (
        payload.get(
            "pilot_design",
            {}
        ).get(
            "systems"
        )
    )

    if systems != [
        "duffing",
        "vanderpol",
    ]:
        raise ValueError(
            "Unexpected frozen pilot system design."
        )

    return {
        "configuration_sha256":
            retained_hash,
        "repository_evidence_bytes_identical":
            True,
    }


def verify_build_summary(
    evidence: Path,
) -> dict[str, Any]:
    summary_path = (
        evidence
        / "build_summary.json"
    )

    summary = strict_json_load(
        summary_path
    )

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(
            "build_summary.json must contain an object."
        )

    expected_scalars = {
        "runner_id":
            EXPECTED_RUNNER_ID,
        "status":
            "completed",
        "execution_source_commit":
            EXPECTED_EXECUTION_COMMIT,
        "configuration_sha256":
            DYNAMICAL_PILOT_CONFIG_SHA256,
        "num_systems_built":
            2,
        "source_unchanged":
            True,
        "certificate_evaluation_pairs_generated":
            0,
    }

    for key, expected in expected_scalars.items():
        actual = summary.get(
            key
        )

        if actual != expected:
            raise ValueError(
                "Build summary mismatch for "
                f"{key}: {actual!r} != {expected!r}."
            )

    if summary.get(
        "systems_built"
    ) != [
        "duffing",
        "vanderpol",
    ]:
        raise ValueError(
            "Build summary has unexpected systems_built."
        )

    records = summary.get(
        "build_records"
    )

    if (
        not isinstance(
            records,
            list,
        )
        or len(records) != 2
    ):
        raise ValueError(
            "Build summary must contain exactly two build records."
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


def verify_representation(
    evidence: Path,
    system: str,
) -> dict[str, Any]:
    design = EXPECTED_SYSTEMS[
        system
    ]

    artifact = (
        evidence
        / "representations"
        / system
    )

    if not artifact.is_dir():
        raise FileNotFoundError(
            f"Representation artifact missing: {system}."
        )

    loaded = load_frozen_representation(
        artifact,
        verify_hashes=True,
    )

    if loaded.format_id != FORMAT_ID:
        raise ValueError(
            f"Wrong frozen format for {system}."
        )

    if loaded.system != system:
        raise ValueError(
            f"Wrong system identity for {system}."
        )

    manifest = loaded.manifest

    if not isinstance(
        manifest,
        dict,
    ):
        raise TypeError(
            f"Manifest is not an object for {system}."
        )

    metadata = manifest.get(
        "model_metadata"
    )

    training = manifest.get(
        "training_metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            f"model_metadata missing for {system}."
        )

    if not isinstance(
        training,
        dict,
    ):
        raise TypeError(
            f"training_metadata missing for {system}."
        )

    checks = {
        "n_clusters":
            design[
                "n_clusters"
            ],
        "kahkm_omega":
            design[
                "omega"
            ],
        "kahkm_tau":
            design[
                "tau"
            ],
    }

    for key, expected in checks.items():
        actual = metadata.get(
            key
        )

        if actual != expected:
            raise ValueError(
                f"{system} model metadata mismatch for "
                f"{key}: {actual!r} != {expected!r}."
            )

    if (
        training.get(
            "configuration_sha256"
        )
        != DYNAMICAL_PILOT_CONFIG_SHA256
    ):
        raise ValueError(
            f"{system} configuration provenance mismatch."
        )

    if (
        training.get(
            "n_train_snapshots"
        )
        != design[
            "n_train_snapshots"
        ]
    ):
        raise ValueError(
            f"{system} training snapshot count mismatch."
        )

    centers = np.asarray(
        loaded.abstraction_model[
            "cluster_centers"
        ],
        dtype=np.float64,
    )

    centers_init = np.asarray(
        loaded.abstraction_model[
            "cluster_centers_init"
        ],
        dtype=np.float64,
    )

    expected_center_shape = (
        design[
            "state_dimension"
        ],
        design[
            "n_clusters"
        ],
    )

    if centers.shape != expected_center_shape:
        raise ValueError(
            f"{system} cluster-center shape mismatch: "
            f"{centers.shape} != {expected_center_shape}."
        )

    if centers_init.shape != centers.shape:
        raise ValueError(
            f"{system} initial/final center shapes differ."
        )

    operator = np.asarray(
        loaded.B,
        dtype=np.float64,
    )

    expected_operator_shape = (
        design[
            "n_clusters"
        ],
        design[
            "n_clusters"
        ],
    )

    if operator.shape != expected_operator_shape:
        raise ValueError(
            f"{system} operator shape mismatch: "
            f"{operator.shape} != {expected_operator_shape}."
        )

    if not np.all(
        np.isfinite(
            operator
        )
    ):
        raise ValueError(
            f"{system} operator contains nonfinite values."
        )

    spectral_norm = float(
        np.linalg.norm(
            operator,
            ord=2,
        )
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
        raise ValueError(
            f"{system} recorded/recomputed spectral norm mismatch."
        )

    phi = kahm_associations(
        loaded.abstraction_model,
        centers,
        omega=loaded.omega,
        tau=loaded.tau,
        n_jobs=1,
        batch_size=256,
        show_progress=False,
    )

    expected_phi_shape = (
        design[
            "n_clusters"
        ],
        design[
            "n_clusters"
        ],
    )

    if phi.shape != expected_phi_shape:
        raise ValueError(
            f"{system} association probe shape mismatch."
        )

    if not np.all(
        np.isfinite(phi)
    ):
        raise ValueError(
            f"{system} association probe contains nonfinite values."
        )

    coordinate_min = float(
        np.min(phi)
    )

    coordinate_max = float(
        np.max(phi)
    )

    if coordinate_min < -1e-12:
        raise ValueError(
            f"{system} association probe has negative coordinate."
        )

    simplex_error = float(
        np.max(
            np.abs(
                np.sum(
                    phi,
                    axis=0,
                    dtype=np.float64,
                )
                - 1.0
            )
        )
    )

    if simplex_error > 1e-10:
        raise ValueError(
            f"{system} association probe violates simplex normalization."
        )

    classifier_entries = manifest.get(
        "classifier_entries"
    )

    if (
        not isinstance(
            classifier_entries,
            list,
        )
        or len(
            classifier_entries
        )
        != design[
            "n_clusters"
        ]
    ):
        raise ValueError(
            f"{system} classifier shard count mismatch."
        )

    files_sha = manifest.get(
        "files_sha256"
    )

    if not isinstance(
        files_sha,
        dict,
    ):
        raise TypeError(
            f"{system} internal files_sha256 is invalid."
        )

    actual_internal = {
        path.relative_to(
            artifact
        ).as_posix()
        for path
        in artifact.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
    }

    if set(
        files_sha
    ) != actual_internal:
        raise ValueError(
            f"{system} internal artifact inventory mismatch."
        )

    return {
        "system":
            system,
        "n_clusters":
            design[
                "n_clusters"
            ],
        "shard_count":
            len(
                classifier_entries
            ),
        "state_dimension":
            design[
                "state_dimension"
            ],
        "nlms_spectral_norm":
            spectral_norm,
        "train_closure_error":
            float(
                loaded.train_closure_error
            ),
        "association_r2":
            float(
                loaded.association_r2
            ),
        "probe_coordinate_min":
            coordinate_min,
        "probe_coordinate_max":
            coordinate_max,
        "probe_max_simplex_error":
            simplex_error,
        "artifact_internal_file_count":
            len(
                actual_internal
            )
            + 1,
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
                str(path)
                for path
                in leftovers
            )
        )


def verify_evidence(
    *,
    evidence_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    evidence = Path(
        evidence_dir
    ).expanduser().resolve()

    if not evidence.is_dir():
        raise FileNotFoundError(
            f"Evidence directory not found: {evidence}"
        )

    evidence_inventory = (
        verify_checksum_inventory(
            evidence
        )
    )

    configuration = (
        verify_configuration(
            evidence
        )
    )

    build_summary = (
        verify_build_summary(
            evidence
        )
    )

    systems = [
        verify_representation(
            evidence,
            system,
        )
        for system
        in (
            "duffing",
            "vanderpol",
        )
    ]

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
        **evidence_inventory,
        **configuration,
        **build_summary,
        "representations":
            systems,
        "num_representations":
            len(
                systems
            ),
        "random_samples_regenerated":
            0,
        "representations_refit":
            0,
        "certificate_evaluation_pairs_generated":
            0,
        "evidence_modified":
            False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify frozen dynamical "
            "representation evidence without refitting."
        )
    )

    parser.add_argument(
        "--evidence-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "New external directory for verifier report. "
            "Must not already exist."
        ),
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
            f"Verification FAILED: output already exists: {output}",
            file=sys.stderr,
        )

        return 1

    try:
        source = (
            verifier_source_snapshot()
        )

        report = verify_evidence(
            evidence_dir=
                args.evidence_dir
        )

        # Recheck committed verifier/source state after all evidence reads.
        final_source = (
            verifier_source_snapshot()
        )

        if final_source != source:
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

        checksums = {
            "verification_report.json":
                sha256_file(
                    output
                    / "verification_report.json"
                )
        }

        strict_json_dump(
            output
            / "verification_checksums.json",
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
    raise SystemExit(
        main()
    )
