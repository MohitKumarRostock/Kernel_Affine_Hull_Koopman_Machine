"""Restore frozen dynamical KAHM models from repository-retained evidence.

The dynamical certificate pilot must consume the committed representation
archive rather than a mutable external build directory.

This module performs no representation fitting and no certificate sampling.
It verifies the exact archive bytes and provenance, safely restores the
archive into a new external directory, validates the archived independent
verification report, and checks that both portable representations load with
their internal hashes intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any

from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)


ARCHIVE_RELATIVE_DIR = (
    "reproduction/prospective_campaigns/"
    "dynamical_frozen_representations_pilot_v1_run001"
)

ARCHIVE_FILENAME = "evidence.tar.gz"

ARCHIVE_FORMAT_ID = (
    "dynamical_frozen_representations_archive_v1"
)

EXPECTED_ARCHIVE_SHA256 = (
    "04a5ad3d4dae3a8fafce2576512fdce06d6c738907ced7d3ad9fde600d8c0e2a"
)

EXPECTED_ARCHIVE_MEMBER_FILE_COUNT = 70

EXPECTED_CONFIGURATION_SHA256 = (
    "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1"
)

EXPECTED_BUILD_EXECUTION_COMMIT = (
    "1add19b94372257d16118f60c69e1e8afa668259"
)

EXPECTED_VERIFIER_SOURCE_COMMIT = (
    "65db148f865c94f988676381c6b5f12222356b31"
)

EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT = (
    "b999a59059c91fa8b50c532dc84199d58683d34dde8e8dedced1c891cb5f8bbd"
)

EXPECTED_VERIFICATION_REPORT_SHA256 = (
    "7e8808aae9109896a5fb02226a364fe67c7ea581f025ca0bd0dc65109177bb42"
)

EXPECTED_VERIFICATION_CHECKSUMS_SHA256 = (
    "5be2a8665dcfaf2d657235638618e34e9d8c2d75b33259c1a34837d08802053b"
)

EXPECTED_SYSTEMS = {
    "duffing": {
        "n_clusters": 10,
        "omega": 2.0,
        "tau": 1e-6,
    },
    "vanderpol": {
        "n_clusters": 25,
        "omega": 4.0,
        "tau": 1e-6,
    },
}


@dataclass(frozen=True)
class RestoredFrozenRepresentations:
    """Paths and provenance for one verified archive restoration."""

    restoration_dir: str
    archive_sha256: str
    configuration_sha256: str
    build_execution_commit: str
    verifier_source_commit: str
    duffing_artifact_dir: str
    vanderpol_artifact_dir: str
    archived_verification_status: str
    archive_member_file_count: int


def _sha256_file(
    path: str | os.PathLike[str],
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _strict_json_load(
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


def _require_safe_member(
    member: tarfile.TarInfo,
) -> None:
    path = Path(
        member.name
    )

    if (
        path.is_absolute()
        or ".." in path.parts
        or path == Path(".")
    ):
        raise ValueError(
            f"Unsafe archive member path: {member.name!r}."
        )

    if not member.isfile():
        raise ValueError(
            "Frozen representation archive may contain only regular files; "
            f"found {member.name!r} with tar type {member.type!r}."
        )


def _verify_archive_metadata(
    metadata: Any,
) -> dict[str, Any]:
    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "ARCHIVE.json must contain a JSON object."
        )

    expected = {
        "archive_format_id":
            ARCHIVE_FORMAT_ID,
        "archive_file":
            ARCHIVE_FILENAME,
        "archive_sha256":
            EXPECTED_ARCHIVE_SHA256,
        "archive_member_file_count":
            EXPECTED_ARCHIVE_MEMBER_FILE_COUNT,
        "configuration_sha256":
            EXPECTED_CONFIGURATION_SHA256,
        "build_execution_commit":
            EXPECTED_BUILD_EXECUTION_COMMIT,
        "verifier_source_commit":
            EXPECTED_VERIFIER_SOURCE_COMMIT,
        "evidence_sha256sums_fingerprint":
            EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT,
        "verification_report_sha256":
            EXPECTED_VERIFICATION_REPORT_SHA256,
        "verification_checksums_sha256":
            EXPECTED_VERIFICATION_CHECKSUMS_SHA256,
        "verification_status":
            "passed",
        "representations_verified":
            2,
        "random_samples_regenerated":
            0,
        "representations_refit":
            0,
        "certificate_evaluation_pairs_generated":
            0,
        "evidence_modified_by_verifier":
            False,
    }

    for key, expected_value in expected.items():
        actual = metadata.get(
            key
        )

        if actual != expected_value:
            raise ValueError(
                "Frozen representation archive metadata mismatch for "
                f"{key}: {actual!r} != {expected_value!r}."
            )

    return metadata


def _verify_archived_verification_report(
    restored_root: Path,
) -> dict[str, Any]:
    report_path = (
        restored_root
        / "verification"
        / "verification_report.json"
    )

    checksums_path = (
        restored_root
        / "verification"
        / "verification_checksums.json"
    )

    if _sha256_file(
        report_path
    ) != EXPECTED_VERIFICATION_REPORT_SHA256:
        raise ValueError(
            "Restored archived verification report SHA256 mismatch."
        )

    if _sha256_file(
        checksums_path
    ) != EXPECTED_VERIFICATION_CHECKSUMS_SHA256:
        raise ValueError(
            "Restored archived verification checksum record SHA256 mismatch."
        )

    checksums = _strict_json_load(
        checksums_path
    )

    if not isinstance(
        checksums,
        dict,
    ):
        raise TypeError(
            "Archived verification_checksums.json must be an object."
        )

    if checksums != {
        "verification_report.json":
            EXPECTED_VERIFICATION_REPORT_SHA256
    }:
        raise ValueError(
            "Archived verification checksum record has unexpected content."
        )

    report = _strict_json_load(
        report_path
    )

    if not isinstance(
        report,
        dict,
    ):
        raise TypeError(
            "Archived verification report must be an object."
        )

    expected = {
        "status":
            "passed",
        "num_representations":
            2,
        "execution_source_commit":
            EXPECTED_BUILD_EXECUTION_COMMIT,
        "configuration_sha256":
            EXPECTED_CONFIGURATION_SHA256,
        "sha256sums_fingerprint":
            EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT,
        "random_samples_regenerated":
            0,
        "representations_refit":
            0,
        "certificate_evaluation_pairs_generated":
            0,
        "evidence_modified":
            False,
        "source_unchanged":
            True,
        "repository_evidence_bytes_identical":
            True,
    }

    for key, expected_value in expected.items():
        actual = report.get(
            key
        )

        if actual != expected_value:
            raise ValueError(
                "Archived verification report mismatch for "
                f"{key}: {actual!r} != {expected_value!r}."
            )

    verifier_source = report.get(
        "verifier_source"
    )

    if not isinstance(
        verifier_source,
        dict,
    ):
        raise TypeError(
            "Archived verification report lacks verifier_source."
        )

    if (
        verifier_source.get(
            "verifier_source_commit"
        )
        != EXPECTED_VERIFIER_SOURCE_COMMIT
    ):
        raise ValueError(
            "Archived verification report verifier commit mismatch."
        )

    return report


def _verify_restored_representation(
    restored_root: Path,
    system: str,
) -> Path:
    design = EXPECTED_SYSTEMS[
        system
    ]

    artifact = (
        restored_root
        / "evidence"
        / "representations"
        / system
    )

    loaded = load_frozen_representation(
        artifact,
        verify_hashes=True,
    )

    if loaded.format_id != FORMAT_ID:
        raise ValueError(
            f"Unexpected frozen representation format for {system}."
        )

    if loaded.system != system:
        raise ValueError(
            f"Restored representation system mismatch for {system}."
        )

    if (
        loaded.abstraction_model[
            "n_clusters"
        ]
        != design[
            "n_clusters"
        ]
    ):
        raise ValueError(
            f"Restored n_clusters mismatch for {system}."
        )

    if loaded.omega != design[
        "omega"
    ]:
        raise ValueError(
            f"Restored omega mismatch for {system}."
        )

    if loaded.tau != design[
        "tau"
    ]:
        raise ValueError(
            f"Restored tau mismatch for {system}."
        )

    training = loaded.manifest.get(
        "training_metadata"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise TypeError(
            f"Restored training metadata missing for {system}."
        )

    if (
        training.get(
            "configuration_sha256"
        )
        != EXPECTED_CONFIGURATION_SHA256
    ):
        raise ValueError(
            f"Restored configuration provenance mismatch for {system}."
        )

    return artifact.resolve()


def restore_repository_frozen_representations(
    *,
    repository_root: str | os.PathLike[str],
    restoration_dir: str | os.PathLike[str],
) -> RestoredFrozenRepresentations:
    """Restore and verify the committed frozen representation archive.

    ``restoration_dir`` must not already exist and must be outside the source
    repository. No source archive bytes are modified.
    """
    repository = Path(
        repository_root
    ).expanduser().resolve()

    destination = Path(
        restoration_dir
    ).expanduser().resolve()

    try:
        destination.relative_to(
            repository
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "restoration_dir must be outside the source repository."
        )

    if destination.exists():
        raise FileExistsError(
            f"Restoration directory already exists: {destination}"
        )

    archive_dir = (
        repository
        / ARCHIVE_RELATIVE_DIR
    )

    archive_path = (
        archive_dir
        / ARCHIVE_FILENAME
    )

    metadata_path = (
        archive_dir
        / "ARCHIVE.json"
    )

    if (
        archive_path.is_symlink()
        or not archive_path.is_file()
    ):
        raise FileNotFoundError(
            "Committed frozen representation archive is missing "
            "or symlinked."
        )

    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
    ):
        raise FileNotFoundError(
            "Committed frozen representation ARCHIVE.json is missing "
            "or symlinked."
        )

    metadata = _verify_archive_metadata(
        _strict_json_load(
            metadata_path
        )
    )

    actual_archive_hash = _sha256_file(
        archive_path
    )

    if (
        actual_archive_hash
        != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError(
            "Committed frozen representation archive SHA256 mismatch: "
            f"{actual_archive_hash} != {EXPECTED_ARCHIVE_SHA256}."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.mkdir(
        exist_ok=False
    )

    try:
        with tarfile.open(
            archive_path,
            mode="r:gz",
        ) as tar:
            members = tar.getmembers()

            if len(
                members
            ) != EXPECTED_ARCHIVE_MEMBER_FILE_COUNT:
                raise ValueError(
                    "Frozen representation archive member count mismatch: "
                    f"{len(members)} != "
                    f"{EXPECTED_ARCHIVE_MEMBER_FILE_COUNT}."
                )

            for member in members:
                _require_safe_member(
                    member
                )

            names = [
                member.name
                for member in members
            ]

            if len(
                set(
                    names
                )
            ) != len(names):
                raise ValueError(
                    "Frozen representation archive contains duplicate paths."
                )

            tar.extractall(
                destination,
                members=members,
                filter="data",
            )

        restored_files = [
            path
            for path
            in destination.rglob("*")
            if path.is_file()
        ]

        if len(
            restored_files
        ) != EXPECTED_ARCHIVE_MEMBER_FILE_COUNT:
            raise ValueError(
                "Restored frozen representation file count mismatch."
            )

        report = (
            _verify_archived_verification_report(
                destination
            )
        )

        duffing = (
            _verify_restored_representation(
                destination,
                "duffing",
            )
        )

        vanderpol = (
            _verify_restored_representation(
                destination,
                "vanderpol",
            )
        )

        # The restored build evidence carries its original SHA256SUMS file.
        restored_evidence_checksums = (
            destination
            / "evidence"
            / "SHA256SUMS"
        )

        if (
            _sha256_file(
                restored_evidence_checksums
            )
            != EXPECTED_EVIDENCE_SHA256SUMS_FINGERPRINT
        ):
            raise ValueError(
                "Restored evidence SHA256SUMS fingerprint mismatch."
            )

        # Recheck the immutable repository archive bytes after restoration.
        if (
            _sha256_file(
                archive_path
            )
            != EXPECTED_ARCHIVE_SHA256
        ):
            raise RuntimeError(
                "Committed representation archive changed during restoration."
            )

    except BaseException:
        shutil.rmtree(
            destination,
            ignore_errors=True,
        )

        raise

    return RestoredFrozenRepresentations(
        restoration_dir=
            str(
                destination
            ),
        archive_sha256=
            actual_archive_hash,
        configuration_sha256=
            metadata[
                "configuration_sha256"
            ],
        build_execution_commit=
            metadata[
                "build_execution_commit"
            ],
        verifier_source_commit=
            metadata[
                "verifier_source_commit"
            ],
        duffing_artifact_dir=
            str(
                duffing
            ),
        vanderpol_artifact_dir=
            str(
                vanderpol
            ),
        archived_verification_status=
            str(
                report[
                    "status"
                ]
            ),
        archive_member_file_count=
            EXPECTED_ARCHIVE_MEMBER_FILE_COUNT,
    )
