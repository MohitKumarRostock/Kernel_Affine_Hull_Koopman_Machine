#!/usr/bin/env python3
"""Build provenance-locked frozen representations for certificate evaluation.

This runner fits exactly the Duffing and Van der Pol representations frozen in

    reproduction/certificates/configs/dynamical_pilot_v1.json

and exports each one using the portable frozen-representation format.

The runner requires a clean committed source tree, retains full preflight
logs, verifies required source-file bytes against HEAD, checks source
immutability throughout execution, reloads each frozen representation, and
performs a small association-evaluation probe before declaring success.

No certificate-evaluation pairs are generated here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from kahkm_dynamical_representation import (
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
    DYNAMICAL_PILOT_CONFIG_SHA256,
    SUPPORTED_SYSTEMS,
    fit_and_freeze_dynamical_representation,
    load_frozen_dynamical_pilot_config,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
)


ROOT = Path(__file__).resolve().parent

RUNNER_ID = "dynamical_certificate_representation_build_v1"

REQUIRED_SOURCE_FILES = (
    Path(__file__).name,
    DYNAMICAL_PILOT_CONFIG_RELATIVE,
    "reproduction/certificates/DYNAMICAL_PROTOCOL.md",
    "requirements-lock-arm64.txt",
    "scripts/verify_environment.py",
    "kahkm_dynamical_representation.py",
    "kahkm_frozen_representation.py",
    "experiment_15_external_koopman_baselines.py",
    "kernel_affine_hull_koopman_machines.py",
    "tests/test_build_dynamical_certificate_representations.py",
    "tests/test_kahkm_dynamical_representation.py",
    "tests/test_kahkm_frozen_representation.py",
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


def strict_json_text(
    payload: Mapping[str, Any],
) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    text = strict_json_text(
        payload
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


def source_snapshot() -> dict[str, Any]:
    """Verify actual required-file bytes against the committed HEAD."""
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
            "Runner must execute from its repository root."
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

    for name in sorted(
        set(
            REQUIRED_SOURCE_FILES
        )
    ):
        path = ROOT / name

        if (
            path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(
                "Missing or symlinked required source file: "
                f"{name}"
            )

        actual = file_sha256(
            path
        )

        try:
            committed_bytes = git_bytes(
                "show",
                f"{commit}:{name}",
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Required source file is not present in HEAD: "
                f"{name}"
            ) from exc

        committed = hashlib.sha256(
            committed_bytes
        ).hexdigest()

        if actual != committed:
            raise RuntimeError(
                "Source bytes differ from HEAD: "
                f"{name}"
            )

        hashes[name] = actual

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
    current = source_snapshot()

    if current != dict(initial):
        raise RuntimeError(
            "Required source state changed during execution."
        )


def _is_within(
    path: Path,
    parent: Path,
) -> bool:
    try:
        path.relative_to(
            parent
        )
    except ValueError:
        return False

    return True


def validate_output_location(
    output: Path,
) -> None:
    resolved = output.expanduser().resolve()

    if _is_within(
        resolved,
        ROOT,
    ):
        raise ValueError(
            "Representation-build output must be outside the "
            "source repository so retained evidence cannot dirty "
            "the execution tree."
        )

    if resolved.exists():
        raise FileExistsError(
            f"Output already exists: {resolved}"
        )


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
                list(command),
            "cwd":
                str(ROOT),
        },
    )

    with (
        (
            logs
            / f"{name}.stdout.log"
        ).open("xb")
        as stdout,
        (
            logs
            / f"{name}.stderr.log"
        ).open("xb")
        as stderr,
    ):
        completed = subprocess.run(
            list(command),
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
    """Retain full environment/dependency/test logs."""
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
            "unit_tests",
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
    )

    logs = output / "checks"
    logs.mkdir()

    for name, command in checks:
        run_command_check(
            logs=logs,
            name=name,
            command=command,
        )

    packages: list[dict[str, Any]] = []

    lock_path = (
        ROOT
        / "requirements-lock-arm64.txt"
    )

    for line in lock_path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        name, separator, expected = (
            line.partition("==")
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
        item["matches"]
        for item in packages
    ):
        raise RuntimeError(
            "Installed packages differ from "
            "the reference lockfile."
        )


def artifact_probe(
    artifact_dir: Path,
) -> dict[str, Any]:
    """Reload and exercise a frozen representation from its portable files."""
    loaded = load_frozen_representation(
        artifact_dir,
        verify_hashes=True,
    )

    model = loaded.abstraction_model

    centers = np.asarray(
        model[
            "cluster_centers"
        ],
        dtype=np.float64,
    )

    phi = kahm_associations(
        model,
        centers,
        omega=loaded.omega,
        tau=loaded.tau,
        n_jobs=1,
        batch_size=256,
        show_progress=False,
    )

    expected_shape = (
        centers.shape[1],
        centers.shape[1],
    )

    if phi.shape != expected_shape:
        raise RuntimeError(
            "Frozen-artifact association probe has "
            f"unexpected shape: {phi.shape} != "
            f"{expected_shape}."
        )

    if not np.all(
        np.isfinite(phi)
    ):
        raise RuntimeError(
            "Frozen-artifact association probe "
            "contains nonfinite values."
        )

    if (
        float(
            np.min(phi)
        )
        < -1e-12
    ):
        raise RuntimeError(
            "Frozen-artifact association probe "
            "contains a materially negative coordinate."
        )

    column_sums = np.sum(
        phi,
        axis=0,
        dtype=np.float64,
    )

    max_simplex_error = float(
        np.max(
            np.abs(
                column_sums
                - 1.0
            )
        )
    )

    if max_simplex_error > 1e-10:
        raise RuntimeError(
            "Frozen-artifact association probe violates "
            "simplex normalization: "
            f"{max_simplex_error}."
        )

    operator = np.asarray(
        loaded.B,
        dtype=np.float64,
    )

    spectral_norm = float(
        np.linalg.norm(
            operator,
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
            "Reloaded frozen operator has invalid spectral norm."
        )

    training_metadata = (
        loaded.manifest.get(
            "training_metadata"
        )
    )

    if not isinstance(
        training_metadata,
        dict,
    ):
        raise RuntimeError(
            "Frozen artifact lacks training_metadata."
        )

    recorded_norm = float(
        training_metadata[
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
            "Reloaded operator spectral norm differs "
            "from the training-time record."
        )

    return {
        "format_id":
            loaded.format_id,
        "system":
            loaded.system,
        "n_clusters":
            int(
                centers.shape[1]
            ),
        "state_dimension":
            int(
                centers.shape[0]
            ),
        "probe_shape":
            [
                int(value)
                for value
                in phi.shape
            ],
        "probe_coordinate_min":
            float(
                np.min(phi)
            ),
        "probe_coordinate_max":
            float(
                np.max(phi)
            ),
        "probe_max_simplex_error":
            max_simplex_error,
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
    }


def write_checksums(
    output: Path,
) -> None:
    files = sorted(
        path
        for path
        in output.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen Duffing and Van der Pol "
            "representations for the dynamical certificate pilot."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "New external directory for retained representation "
            "artifacts and provenance. It must not already exist "
            "and must be outside the source repository."
        ),
    )

    return parser.parse_args()


def execute(
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the two-system frozen-representation build."""
    output = (
        output_dir
        .expanduser()
        .resolve()
    )

    validate_output_location(
        output
    )

    # Establish committed source state before creating any retained output.
    initial_source = source_snapshot()

    config = load_frozen_dynamical_pilot_config(
        repository_root=ROOT
    )

    if (
        file_sha256(
            ROOT
            / DYNAMICAL_PILOT_CONFIG_RELATIVE
        )
        != DYNAMICAL_PILOT_CONFIG_SHA256
    ):
        raise RuntimeError(
            "Configuration fingerprint changed "
            "between validation steps."
        )

    configured_systems = (
        config.get(
            "pilot_design",
            {}
        ).get(
            "systems"
        )
    )

    if configured_systems != [
        "duffing",
        "vanderpol",
    ]:
        raise RuntimeError(
            "Unexpected frozen pilot system order."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.mkdir(
        exist_ok=False
    )

    started_wall = time.time()
    started_utc = utc_now()

    write_json(
        output
        / "build_metadata.json",
        {
            "runner_id":
                RUNNER_ID,
            "status":
                "started",
            "started_utc":
                started_utc,
            "execution_source":
                dict(
                    initial_source
                ),
            "configuration_relative_path":
                DYNAMICAL_PILOT_CONFIG_RELATIVE,
            "configuration_sha256":
                DYNAMICAL_PILOT_CONFIG_SHA256,
            "frozen_representation_format":
                FORMAT_ID,
            "systems":
                list(
                    configured_systems
                ),
            "python_executable":
                sys.executable,
            "python_version":
                sys.version,
            "platform":
                platform.platform(),
            "argv":
                list(
                    sys.argv
                ),
        },
    )

    # Preserve the exact committed configuration bytes with the evidence.
    shutil.copyfile(
        ROOT
        / DYNAMICAL_PILOT_CONFIG_RELATIVE,
        output
        / "configuration.json",
    )

    run_preflight(
        output
    )

    assert_source_unchanged(
        initial_source
    )

    representations = (
        output
        / "representations"
    )

    representations.mkdir()

    build_records: list[
        dict[str, Any]
    ] = []

    for system in configured_systems:
        if system not in SUPPORTED_SYSTEMS:
            raise RuntimeError(
                f"Unsupported configured system: {system!r}."
            )

        system_started = (
            time.time()
        )

        artifact_dir = (
            representations
            / system
        )

        work_dir = (
            output.parent
            / (
                f".{output.name}."
                f"{system}.representation-work"
            )
        )

        if work_dir.exists():
            raise FileExistsError(
                "External representation work directory "
                f"already exists: {work_dir}"
            )

        result = (
            fit_and_freeze_dynamical_representation(
                repository_root=ROOT,
                system=system,
                output_dir=
                    artifact_dir,
                work_dir=
                    work_dir,
            )
        )

        if work_dir.exists():
            raise RuntimeError(
                "Representation builder left its temporary "
                f"work directory behind: {work_dir}"
            )

        assert_source_unchanged(
            initial_source
        )

        probe = artifact_probe(
            artifact_dir
        )

        if probe[
            "system"
        ] != system:
            raise RuntimeError(
                "Reloaded representation system mismatch."
            )

        if (
            probe[
                "n_clusters"
            ]
            != result.n_clusters
        ):
            raise RuntimeError(
                "Reloaded representation cluster count mismatch."
            )

        if not np.isclose(
            probe[
                "nlms_spectral_norm"
            ],
            result.nlms_spectral_norm,
            rtol=1e-13,
            atol=1e-15,
        ):
            raise RuntimeError(
                "Reloaded representation spectral norm "
                "differs from build result."
            )

        record = {
            **asdict(
                result
            ),
            "probe":
                probe,
            "build_seconds":
                float(
                    time.time()
                    - system_started
                ),
        }

        build_records.append(
            record
        )

        write_json(
            output
            / (
                f"{system}_build_result.json"
            ),
            record,
        )

        assert_source_unchanged(
            initial_source
        )

    final_source = source_snapshot()

    if final_source != initial_source:
        raise RuntimeError(
            "Source state changed before build finalization."
        )

    summary = {
        "runner_id":
            RUNNER_ID,
        "status":
            "completed",
        "started_utc":
            started_utc,
        "completed_utc":
            utc_now(),
        "elapsed_seconds":
            float(
                time.time()
                - started_wall
            ),
        "execution_source_commit":
            initial_source[
                "execution_source_commit"
            ],
        "configuration_sha256":
            DYNAMICAL_PILOT_CONFIG_SHA256,
        "systems_built":
            [
                record[
                    "system"
                ]
                for record
                in build_records
            ],
        "num_systems_built":
            len(
                build_records
            ),
        "build_records":
            build_records,
        "source_unchanged":
            True,
        "certificate_evaluation_pairs_generated":
            0,
    }

    write_json(
        output
        / "build_summary.json",
        summary,
    )

    write_checksums(
        output
    )

    return summary


def main() -> int:
    args = parse_args()

    output = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    try:
        summary = execute(
            output_dir=output
        )
    except BaseException as exc:
        if output.is_dir():
            failure = {
                "runner_id":
                    RUNNER_ID,
                "status":
                    "failed",
                "failed_utc":
                    utc_now(),
                "exception_type":
                    type(
                        exc
                    ).__name__,
                "exception_message":
                    str(
                        exc
                    ),
                "traceback":
                    traceback.format_exc(),
            }

            failure_path = (
                output
                / "failure.json"
            )

            if not failure_path.exists():
                try:
                    write_json(
                        failure_path,
                        failure,
                    )
                except BaseException:
                    pass

            checksums = (
                output
                / "SHA256SUMS"
            )

            if not checksums.exists():
                try:
                    write_checksums(
                        output
                    )
                except BaseException:
                    pass

        print(
            "Representation build FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return 1

    print(
        json.dumps(
            summary,
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
