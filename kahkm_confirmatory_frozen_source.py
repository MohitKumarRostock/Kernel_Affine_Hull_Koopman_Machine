"""Restore the verified confirmatory dynamical representation from Git.

This module is intentionally separate from the pilot frozen-source restore
path.  It restores only the repository-committed, independently verified
Van der Pol representation frozen for the confirmatory campaign.

No training trajectories are regenerated here and no certificate-evaluation
pairs are generated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any

from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)


CONFIRMATORY_FROZEN_SOURCE_ID = (
    "repository_verified_confirmatory_representation_v1"
)

CAMPAIGN_ID = (
    "dynamical_original_iid_confirmatory_v1"
)

SYSTEM = "vanderpol"
OMEGA = 128.0
TAU = 1e-6

PACKAGE_RELATIVE_DIR = Path(
    "reproduction"
) / "prospective_campaigns" / (
    "dynamical_confirmatory_representation_v1_run001"
)

ARCHIVE_RELATIVE_PATH = (
    PACKAGE_RELATIVE_DIR
    / "evidence.tar.gz"
)

ARCHIVE_MANIFEST_RELATIVE_PATH = (
    PACKAGE_RELATIVE_DIR
    / "ARCHIVE.json"
)

PACKAGE_CHECKSUMS_RELATIVE_PATH = (
    PACKAGE_RELATIVE_DIR
    / "SHA256SUMS"
)

EXPECTED_ARCHIVE_SHA256 = (
    "17fe746da6eddcc194da9fe19dedf7a80d3d887da397b103ae5e5f1f4718bcbc"
)

EXPECTED_ARCHIVE_MANIFEST_SHA256 = (
    "2f9cafc4c72ae1dca5dc9184301e7138a4cc6afc57a3a4144d6adfd750529404"
)

EXPECTED_PACKAGE_CHECKSUMS_SHA256 = (
    "b59e1c4b3279eeaf90b917569b62f5e05b166dbe8745c603579d0004d03ba452"
)

EXPECTED_BUILD_EVIDENCE_SHA256SUMS_SHA256 = (
    "76b6cd0a2b2369b6cd2e2468fe3d2e06a5ed33e4eb172c84372196c2e4530372"
)

EXPECTED_VERIFICATION_REPORT_SHA256 = (
    "25bbf7a326e6a9e6f10a459cca4a13aebe166d52335ce12efea17f037f60efbf"
)

EXPECTED_VERIFICATION_CHECKSUMS_SHA256 = (
    "10e6698b6c2d3d7995b72e1bc326cddb5f00f80b458947e778d89852b4af7447"
)

EXPECTED_CONFIGURATION_SHA256 = (
    "1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91"
)

EXPECTED_PROTOCOL_SHA256 = (
    "daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c"
)

EXPECTED_BUILD_EXECUTION_COMMIT = (
    "4a0cedd557a6e06d331aa31b91223629c36d1997"
)

EXPECTED_VERIFIER_SOURCE_COMMIT = (
    "cbfe43808ac54d76fb26a35e5361a634e526745c"
)

EXPECTED_ARCHIVE_MEMBER_COUNT = 58

EXPECTED_ARTIFACT_RELATIVE_PATH = (
    Path("evidence")
    / "representations"
    / SYSTEM
)


@dataclass(frozen=True)
class RestoredConfirmatoryRepresentation:
    source_id: str
    campaign_id: str
    system: str
    omega: float
    tau: float

    archive_sha256: str
    archive_manifest_sha256: str
    package_checksums_sha256: str

    configuration_sha256: str
    protocol_sha256: str

    build_execution_source_commit: str
    verifier_source_commit: str

    build_evidence_sha256sums_fingerprint: str
    verification_report_sha256: str
    verification_checksums_sha256: str

    restoration_dir: Path
    vanderpol_artifact_dir: Path

    certificate_evaluation_pairs_generated: int


def _sha256_file(
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


def _load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _safe_archive_name(
    name: str,
) -> PurePosixPath:
    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "Archive member name must be a string."
        )

    candidate = PurePosixPath(
        name
    )

    if (
        not name
        or name.startswith(
            "/"
        )
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
        or "\\" in name
    ):
        raise ValueError(
            f"Unsafe archive member path: {name!r}."
        )

    normalized = (
        candidate.as_posix()
    )

    if normalized != name:
        raise ValueError(
            f"Non-canonical archive member path: {name!r}."
        )

    return candidate


def _verify_package_checksums(
    package_dir: Path,
) -> None:
    checksum_path = (
        package_dir
        / "SHA256SUMS"
    )

    if (
        checksum_path.is_symlink()
        or not checksum_path.is_file()
    ):
        raise FileNotFoundError(
            "Confirmatory package SHA256SUMS is missing."
        )

    if (
        _sha256_file(
            checksum_path
        )
        != EXPECTED_PACKAGE_CHECKSUMS_SHA256
    ):
        raise ValueError(
            "Confirmatory package SHA256SUMS fingerprint changed."
        )

    expected = {
        "ARCHIVE.json":
            EXPECTED_ARCHIVE_MANIFEST_SHA256,

        "README.md":
            "5036536d4104cf7b96e3838e961696fd61ec0c979790de66c8149d8b6e890ff9",

        "evidence.tar.gz":
            EXPECTED_ARCHIVE_SHA256,
    }

    parsed: dict[
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

        digest, separator, name = (
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
                "Malformed confirmatory package SHA256SUMS "
                f"line {line_number}."
            )

        if name in parsed:
            raise ValueError(
                "Duplicate confirmatory package checksum entry: "
                f"{name}"
            )

        parsed[
            name
        ] = digest

    if parsed != expected:
        raise ValueError(
            "Confirmatory package checksum inventory changed."
        )

    for name, digest in expected.items():
        path = (
            package_dir
            / name
        )

        if (
            path.is_symlink()
            or not path.is_file()
        ):
            raise FileNotFoundError(
                "Missing or symlinked confirmatory package file: "
                f"{name}"
            )

        if (
            _sha256_file(
                path
            )
            != digest
        ):
            raise ValueError(
                "Confirmatory package payload hash mismatch: "
                f"{name}"
            )


def _verify_archive_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "Confirmatory ARCHIVE.json is missing."
        )

    if (
        _sha256_file(
            manifest_path
        )
        != EXPECTED_ARCHIVE_MANIFEST_SHA256
    ):
        raise ValueError(
            "Confirmatory ARCHIVE.json fingerprint changed."
        )

    manifest = _load_json(
        manifest_path
    )

    if not isinstance(
        manifest,
        dict,
    ):
        raise TypeError(
            "Confirmatory ARCHIVE.json must contain an object."
        )

    expected_scalars = {
        "schema_id":
            "verified_confirmatory_dynamical_representation_archive_v1",

        "run_id":
            "dynamical_confirmatory_representation_v1_run001",

        "campaign_id":
            CAMPAIGN_ID,

        "artifact_role":
            "verified_confirmatory_representation",

        "system":
            SYSTEM,

        "omega":
            OMEGA,

        "build_execution_source_commit":
            EXPECTED_BUILD_EXECUTION_COMMIT,

        "verifier_source_commit":
            EXPECTED_VERIFIER_SOURCE_COMMIT,

        "configuration_sha256":
            EXPECTED_CONFIGURATION_SHA256,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "original_evidence_sha256sums_fingerprint":
            EXPECTED_BUILD_EVIDENCE_SHA256SUMS_SHA256,

        "verification_report_sha256":
            EXPECTED_VERIFICATION_REPORT_SHA256,

        "verification_checksums_sha256":
            EXPECTED_VERIFICATION_CHECKSUMS_SHA256,

        "compressed_archive_sha256":
            EXPECTED_ARCHIVE_SHA256,

        "regular_file_member_count":
            EXPECTED_ARCHIVE_MEMBER_COUNT,

        "evidence_physical_file_count":
            56,

        "verification_file_count":
            2,

        "certificate_evaluation_pairs_generated":
            0,

        "operator_exact_binary64_verified":
            True,

        "nlms_history_exact_binary64_verified":
            True,

        "retained_representation_modified_by_verification":
            False,
    }

    for key, expected in expected_scalars.items():
        actual = manifest.get(
            key
        )

        if actual != expected:
            raise ValueError(
                "Confirmatory archive manifest mismatch for "
                f"{key}: {actual!r} != {expected!r}."
            )

    members = manifest.get(
        "members"
    )

    if (
        not isinstance(
            members,
            list,
        )
        or len(
            members
        )
        != EXPECTED_ARCHIVE_MEMBER_COUNT
    ):
        raise ValueError(
            "Confirmatory archive must declare exactly "
            f"{EXPECTED_ARCHIVE_MEMBER_COUNT} members."
        )

    seen: set[
        str
    ] = set()

    for record in members:
        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(
                "Invalid confirmatory archive member record."
            )

        name = record.get(
            "path"
        )

        _safe_archive_name(
            name
        )

        if name in seen:
            raise ValueError(
                "Duplicate confirmatory archive member record: "
                f"{name}"
            )

        seen.add(
            name
        )

        size = record.get(
            "bytes"
        )

        digest = record.get(
            "sha256"
        )

        if (
            not isinstance(
                size,
                int,
            )
            or size < 0
        ):
            raise ValueError(
                "Invalid confirmatory archive member size: "
                f"{name}"
            )

        if (
            not isinstance(
                digest,
                str,
            )
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
                "Invalid confirmatory archive member SHA-256: "
                f"{name}"
            )

    return manifest


def _verify_and_extract_archive(
    *,
    archive_path: Path,
    manifest: dict[str, Any],
    work_dir: Path,
) -> None:
    if (
        archive_path.is_symlink()
        or not archive_path.is_file()
    ):
        raise FileNotFoundError(
            "Confirmatory representation archive is missing."
        )

    if (
        _sha256_file(
            archive_path
        )
        != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError(
            "Confirmatory representation archive fingerprint changed."
        )

    records = {
        record[
            "path"
        ]:
            record
        for record in manifest[
            "members"
        ]
    }

    work_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as tar:
        members = (
            tar.getmembers()
        )

        if (
            len(
                members
            )
            != EXPECTED_ARCHIVE_MEMBER_COUNT
        ):
            raise ValueError(
                "Unexpected confirmatory tar member count."
            )

        names = [
            member.name
            for member in members
        ]

        if names != list(
            records
        ):
            raise ValueError(
                "Confirmatory tar member ordering differs from ARCHIVE.json."
            )

        for member in members:
            _safe_archive_name(
                member.name
            )

            if not member.isfile():
                raise ValueError(
                    "Confirmatory tar contains a non-regular member: "
                    f"{member.name}"
                )

            if (
                member.uid != 0
                or member.gid != 0
                or member.mtime != 0
                or member.mode != 0o644
            ):
                raise ValueError(
                    "Confirmatory tar deterministic metadata mismatch: "
                    f"{member.name}"
                )

            record = records[
                member.name
            ]

            if (
                member.size
                != record[
                    "bytes"
                ]
            ):
                raise ValueError(
                    "Confirmatory tar member-size mismatch: "
                    f"{member.name}"
                )

            extracted = tar.extractfile(
                member
            )

            if extracted is None:
                raise ValueError(
                    "Unable to read confirmatory tar member: "
                    f"{member.name}"
                )

            data = (
                extracted.read()
            )

            if len(
                data
            ) != member.size:
                raise ValueError(
                    "Short read from confirmatory tar member: "
                    f"{member.name}"
                )

            digest = hashlib.sha256(
                data
            ).hexdigest()

            if (
                digest
                != record[
                    "sha256"
                ]
            ):
                raise ValueError(
                    "Confirmatory tar member SHA-256 mismatch: "
                    f"{member.name}"
                )

            relative = PurePosixPath(
                member.name
            )

            destination = (
                work_dir.joinpath(
                    *relative.parts
                )
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with destination.open(
                "xb"
            ) as handle:
                handle.write(
                    data
                )
                handle.flush()
                os.fsync(
                    handle.fileno()
                )


def _verify_restored_evidence(
    work_dir: Path,
) -> Path:
    build_inventory = (
        work_dir
        / "evidence"
        / "SHA256SUMS"
    )

    verification_report = (
        work_dir
        / "verification"
        / "verification_report.json"
    )

    verification_checksums = (
        work_dir
        / "verification"
        / "verification_checksums.json"
    )

    if (
        _sha256_file(
            build_inventory
        )
        != EXPECTED_BUILD_EVIDENCE_SHA256SUMS_SHA256
    ):
        raise ValueError(
            "Restored build-evidence inventory fingerprint changed."
        )

    if (
        _sha256_file(
            verification_report
        )
        != EXPECTED_VERIFICATION_REPORT_SHA256
    ):
        raise ValueError(
            "Restored verification-report fingerprint changed."
        )

    if (
        _sha256_file(
            verification_checksums
        )
        != EXPECTED_VERIFICATION_CHECKSUMS_SHA256
    ):
        raise ValueError(
            "Restored verification-checksums fingerprint changed."
        )

    checksums_payload = _load_json(
        verification_checksums
    )

    if checksums_payload != {
        "verification_report.json":
            EXPECTED_VERIFICATION_REPORT_SHA256
    }:
        raise ValueError(
            "Restored verification-checksums payload changed."
        )

    report = _load_json(
        verification_report
    )

    if not isinstance(
        report,
        dict,
    ):
        raise TypeError(
            "Restored verification report must contain an object."
        )

    if report.get(
        "status"
    ) != "passed":
        raise ValueError(
            "Restored representation is not independently verified."
        )

    if report.get(
        "execution_source_commit"
    ) != EXPECTED_BUILD_EXECUTION_COMMIT:
        raise ValueError(
            "Restored verification report build-source mismatch."
        )

    if (
        report.get(
            "verifier_source",
            {}
        ).get(
            "verifier_source_commit"
        )
        != EXPECTED_VERIFIER_SOURCE_COMMIT
    ):
        raise ValueError(
            "Restored verification report verifier-source mismatch."
        )

    if report.get(
        "configuration_sha256"
    ) != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError(
            "Restored verification report config mismatch."
        )

    if report.get(
        "protocol_sha256"
    ) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "Restored verification report protocol mismatch."
        )

    if report.get(
        "retained_representation_modified"
    ) is not False:
        raise ValueError(
            "Restored verification report does not preserve immutability."
        )

    if report.get(
        "certificate_evaluation_pairs_generated"
    ) != 0:
        raise ValueError(
            "Representation verification generated evaluation pairs."
        )

    representation = report.get(
        "representation"
    )

    if not isinstance(
        representation,
        dict,
    ):
        raise TypeError(
            "Restored representation verification block is missing."
        )

    if representation.get(
        "system"
    ) != SYSTEM:
        raise ValueError(
            "Restored verified system mismatch."
        )

    if representation.get(
        "omega"
    ) != OMEGA:
        raise ValueError(
            "Restored verified omega mismatch."
        )

    if representation.get(
        "tau"
    ) != TAU:
        raise ValueError(
            "Restored verified tau mismatch."
        )

    if representation.get(
        "operator_exact_binary64_match"
    ) is not True:
        raise ValueError(
            "Restored operator lacks exact binary64 verification."
        )

    if representation.get(
        "nlms_history_exact_binary64_match"
    ) is not True:
        raise ValueError(
            "Restored NLMS history lacks exact binary64 verification."
        )

    artifact_dir = (
        work_dir
        / EXPECTED_ARTIFACT_RELATIVE_PATH
    )

    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            "Restored Van der Pol representation artifact is missing."
        )

    loaded = (
        load_frozen_representation(
            artifact_dir,
            verify_hashes=True,
        )
    )

    if loaded.format_id != FORMAT_ID:
        raise ValueError(
            "Restored representation format mismatch."
        )

    if loaded.system != SYSTEM:
        raise ValueError(
            "Restored representation system mismatch."
        )

    if loaded.omega != OMEGA:
        raise ValueError(
            "Restored representation omega mismatch."
        )

    if loaded.tau != TAU:
        raise ValueError(
            "Restored representation tau mismatch."
        )

    return artifact_dir


def restore_repository_confirmatory_representation(
    *,
    repository_root: str | os.PathLike[str],
    restoration_dir: str | os.PathLike[str],
) -> RestoredConfirmatoryRepresentation:
    """Restore and verify the repository-frozen confirmatory representation."""

    repository_root_path = Path(
        repository_root
    ).expanduser().resolve()

    restoration_path = Path(
        restoration_dir
    ).expanduser().resolve()

    package_dir = (
        repository_root_path
        / PACKAGE_RELATIVE_DIR
    )

    archive_path = (
        repository_root_path
        / ARCHIVE_RELATIVE_PATH
    )

    manifest_path = (
        repository_root_path
        / ARCHIVE_MANIFEST_RELATIVE_PATH
    )

    checksum_path = (
        repository_root_path
        / PACKAGE_CHECKSUMS_RELATIVE_PATH
    )

    if restoration_path.exists():
        raise FileExistsError(
            "Confirmatory restoration destination already exists: "
            f"{restoration_path}"
        )

    if (
        not repository_root_path.is_dir()
    ):
        raise FileNotFoundError(
            "Repository root does not exist: "
            f"{repository_root_path}"
        )

    _verify_package_checksums(
        package_dir
    )

    if (
        _sha256_file(
            checksum_path
        )
        != EXPECTED_PACKAGE_CHECKSUMS_SHA256
    ):
        raise ValueError(
            "Confirmatory package checksum file changed."
        )

    manifest = (
        _verify_archive_manifest(
            manifest_path
        )
    )

    restoration_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir = (
        restoration_path.parent
        / (
            "."
            + restoration_path.name
            + ".restore-work"
        )
    )

    if work_dir.exists():
        raise FileExistsError(
            "Confirmatory restoration work directory already exists: "
            f"{work_dir}"
        )

    try:
        _verify_and_extract_archive(
            archive_path=
                archive_path,

            manifest=
                manifest,

            work_dir=
                work_dir,
        )

        artifact_in_work = (
            _verify_restored_evidence(
                work_dir
            )
        )

        relative_artifact = (
            artifact_in_work.relative_to(
                work_dir
            )
        )

        os.replace(
            work_dir,
            restoration_path,
        )

    except BaseException:
        if work_dir.exists():
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

        raise

    artifact_dir = (
        restoration_path
        / relative_artifact
    )

    if not artifact_dir.is_dir():
        raise RuntimeError(
            "Confirmatory artifact missing after atomic restoration."
        )

    return RestoredConfirmatoryRepresentation(
        source_id=
            CONFIRMATORY_FROZEN_SOURCE_ID,

        campaign_id=
            CAMPAIGN_ID,

        system=
            SYSTEM,

        omega=
            OMEGA,

        tau=
            TAU,

        archive_sha256=
            EXPECTED_ARCHIVE_SHA256,

        archive_manifest_sha256=
            EXPECTED_ARCHIVE_MANIFEST_SHA256,

        package_checksums_sha256=
            EXPECTED_PACKAGE_CHECKSUMS_SHA256,

        configuration_sha256=
            EXPECTED_CONFIGURATION_SHA256,

        protocol_sha256=
            EXPECTED_PROTOCOL_SHA256,

        build_execution_source_commit=
            EXPECTED_BUILD_EXECUTION_COMMIT,

        verifier_source_commit=
            EXPECTED_VERIFIER_SOURCE_COMMIT,

        build_evidence_sha256sums_fingerprint=
            EXPECTED_BUILD_EVIDENCE_SHA256SUMS_SHA256,

        verification_report_sha256=
            EXPECTED_VERIFICATION_REPORT_SHA256,

        verification_checksums_sha256=
            EXPECTED_VERIFICATION_CHECKSUMS_SHA256,

        restoration_dir=
            restoration_path,

        vanderpol_artifact_dir=
            artifact_dir,

        certificate_evaluation_pairs_generated=
            0,
    )
