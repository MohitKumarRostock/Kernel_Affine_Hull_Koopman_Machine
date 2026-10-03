"""Lossless per-attempt evidence for the dynamical certificate pilot.

Sampling evidence and evaluation evidence are deliberately separate.

The campaign writes:

    sample_arrays.npz
    sample.json

before calling the evaluator. Only afterwards may it write:

    evaluation_arrays.npz
    evaluation.json

This preserves the sampled independent pairs even if evaluation fails.

The NPZ writer is deterministic: array members are sorted, each member is an
ordinary NPY payload, and ZIP timestamps/permissions are fixed. No pickle
payloads are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from kahkm_dynamical_evaluation import (
    DynamicalEvaluationResult,
)
from kahkm_dynamical_sampling import (
    DYNAMICAL_SAMPLING_ID,
    SampledDynamicalDataset,
    SeedRecord,
)


SAMPLE_EVIDENCE_ID = "dynamical_sample_evidence_v1"
EVALUATION_EVIDENCE_ID = "dynamical_evaluation_evidence_v1"

SAMPLE_ARRAY_FILENAME = "sample_arrays.npz"
SAMPLE_METADATA_FILENAME = "sample.json"

EVALUATION_ARRAY_FILENAME = "evaluation_arrays.npz"
EVALUATION_METADATA_FILENAME = "evaluation.json"


@dataclass(frozen=True)
class LoadedEvidence:
    """Strictly loaded JSON metadata plus copied NumPy arrays."""

    metadata: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


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


def _strict_json_text(
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


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    text = _strict_json_text(
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


def _strict_json_load(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle,
            parse_constant=lambda value: (
                (_ for _ in ()).throw(
                    ValueError(
                        f"Nonfinite JSON constant in {path}: {value}"
                    )
                )
            ),
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path.name} must contain a JSON object."
        )

    return payload


def _npy_bytes(
    array: NDArray[Any],
) -> bytes:
    stream = io.BytesIO()

    np.lib.format.write_array(
        stream,
        np.asarray(
            array
        ),
        allow_pickle=False,
    )

    return stream.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    if not arrays:
        raise ValueError(
            "At least one array is required."
        )

    names = sorted(
        arrays
    )

    if len(
        set(names)
    ) != len(names):
        raise ValueError(
            "Duplicate array names are not allowed."
        )

    with path.open(
        "xb"
    ) as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name in names:
                if (
                    not isinstance(
                        name,
                        str,
                    )
                    or not name
                    or "/"
                    in name
                    or "\\"
                    in name
                ):
                    raise ValueError(
                        f"Unsafe array name: {name!r}."
                    )

                array = np.asarray(
                    arrays[
                        name
                    ]
                )

                if array.dtype.hasobject:
                    raise TypeError(
                        f"Object dtype is forbidden for {name}."
                    )

                info = zipfile.ZipInfo(
                    filename=
                        f"{name}.npy",
                    date_time=(
                        1980,
                        1,
                        1,
                        0,
                        0,
                        0,
                    ),
                )

                info.compress_type = (
                    zipfile.ZIP_STORED
                )

                info.create_system = 3
                info.external_attr = (
                    0o600
                    << 16
                )

                archive.writestr(
                    info,
                    _npy_bytes(
                        array
                    ),
                )

        raw.flush()
        os.fsync(
            raw.fileno()
        )


def _load_npz_exact(
    path: Path,
    expected_names: set[str],
) -> dict[str, NDArray[Any]]:
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        actual = set(
            archive.files
        )

        if actual != expected_names:
            raise ValueError(
                f"{path.name} members differ from expected set; "
                f"actual={sorted(actual)}, "
                f"expected={sorted(expected_names)}."
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
                expected_names
            )
        }


def _seed_record_arrays(
    *,
    records: tuple[SeedRecord, ...],
    expected_stream: str,
    dataset: SampledDynamicalDataset,
) -> dict[str, NDArray[Any]]:
    M = int(
        dataset.sample_size
    )

    if len(records) != M:
        raise ValueError(
            f"{expected_stream} record count does not match sample size."
        )

    seeds = np.empty(
        M,
        dtype=np.uint64,
    )

    words = np.empty(
        (
            M,
            2,
        ),
        dtype=np.uint32,
    )

    material = np.empty(
        (
            M,
            8,
        ),
        dtype=np.uint64,
    )

    for index, record in enumerate(
        records
    ):
        if record.stream_name != expected_stream:
            raise ValueError(
                f"Seed record {index} has wrong stream name."
            )

        expected_common = (
            record.sampling_id
            == DYNAMICAL_SAMPLING_ID
            and record.system
            == dataset.system
            and record.evaluation_mode
            == dataset.evaluation_mode
            and record.sample_size
            == dataset.sample_size
            and record.replicate_index
            == dataset.replicate_index
            and record.pair_index
            == index
        )

        if not expected_common:
            raise ValueError(
                f"Seed record {index} metadata does not match dataset."
            )

        if len(
            record.seed_material
        ) != 8:
            raise ValueError(
                f"Seed record {index} does not contain eight seed components."
            )

        seeds[
            index
        ] = np.uint64(
            record.generated_seed
        )

        words[
            index,
            :,
        ] = np.asarray(
            record.generated_uint32_words,
            dtype=np.uint32,
        )

        material[
            index,
            :,
        ] = np.asarray(
            record.seed_material,
            dtype=np.uint64,
        )

    return {
        f"{expected_stream}_generated_seed":
            seeds,
        f"{expected_stream}_uint32_words":
            words,
        f"{expected_stream}_material":
            material,
    }


def write_sample_evidence(
    *,
    attempt_dir: str | os.PathLike[str],
    dataset: SampledDynamicalDataset,
) -> dict[str, Any]:
    """Persist one sampled dataset before any evaluation is performed."""
    if not isinstance(
        dataset,
        SampledDynamicalDataset,
    ):
        raise TypeError(
            "dataset must be SampledDynamicalDataset."
        )

    destination = Path(
        attempt_dir
    ).expanduser().resolve()

    if destination.exists():
        raise FileExistsError(
            f"Attempt directory already exists: {destination}"
        )

    current = np.asarray(
        dataset.current_states,
        dtype=np.float64,
    )

    successor = np.asarray(
        dataset.successor_states,
        dtype=np.float64,
    )

    expected_shape = (
        2,
        int(
            dataset.sample_size
        ),
    )

    if (
        current.shape
        != expected_shape
        or successor.shape
        != expected_shape
    ):
        raise ValueError(
            "Sample physical-state arrays have unexpected shape."
        )

    if (
        not np.all(
            np.isfinite(
                current
            )
        )
        or not np.all(
            np.isfinite(
                successor
            )
        )
    ):
        raise ValueError(
            "Sample physical-state arrays contain nonfinite values."
        )

    time_indices = np.asarray(
        dataset.time_indices,
        dtype=np.int64,
    )

    if time_indices.shape != (
        dataset.sample_size,
    ):
        raise ValueError(
            "time_indices shape does not match sample size."
        )

    if (
        np.any(
            time_indices < 0
        )
        or np.any(
            time_indices
            >= dataset.horizon_steps
        )
    ):
        raise ValueError(
            "time_indices lie outside the frozen horizon."
        )

    arrays: dict[
        str,
        NDArray[Any],
    ] = {
        "current_states":
            np.array(
                current,
                copy=True,
                order="C",
            ),
        "successor_states":
            np.array(
                successor,
                copy=True,
                order="C",
            ),
        "time_indices":
            time_indices.copy(),
    }

    arrays.update(
        _seed_record_arrays(
            records=
                dataset.trajectory_seed_records,
            expected_stream=
                "trajectory_seed",
            dataset=
                dataset,
        )
    )

    arrays.update(
        _seed_record_arrays(
            records=
                dataset.time_seed_records,
            expected_stream=
                "time_seed",
            dataset=
                dataset,
        )
    )

    destination.mkdir(
        parents=True,
        exist_ok=False,
    )

    arrays_path = (
        destination
        / SAMPLE_ARRAY_FILENAME
    )

    metadata_path = (
        destination
        / SAMPLE_METADATA_FILENAME
    )

    try:
        _write_deterministic_npz(
            arrays_path,
            arrays,
        )

        arrays_sha = _sha256_file(
            arrays_path
        )

        metadata = {
            "sample_evidence_id":
                SAMPLE_EVIDENCE_ID,
            "sampling_id":
                dataset.sampling_id,
            "pair_law_id":
                dataset.pair_law_id,
            "system":
                dataset.system,
            "evaluation_mode":
                dataset.evaluation_mode,
            "sample_size":
                int(
                    dataset.sample_size
                ),
            "replicate_index":
                int(
                    dataset.replicate_index
                ),
            "state_dimension":
                int(
                    current.shape[0]
                ),
            "dt":
                float(
                    dataset.dt
                ),
            "horizon_steps":
                int(
                    dataset.horizon_steps
                ),
            "vanderpol_mu":
                float(
                    dataset.vanderpol_mu
                ),
            "array_file":
                SAMPLE_ARRAY_FILENAME,
            "array_sha256":
                arrays_sha,
            "array_members":
                sorted(
                    arrays
                ),
            "seed_record_fields_retained": [
                "generated_seed",
                "generated_uint32_words",
                "seed_material",
            ],
            "evaluation_performed":
                False,
        }

        _write_json_exclusive(
            metadata_path,
            metadata,
        )

    except BaseException:
        # Evidence-write failure is fatal at campaign level. Keep the attempt
        # directory and any successfully written bytes for forensic inspection;
        # do not make a failed persistence operation look as though it never
        # happened.
        raise

    return metadata


def write_evaluation_evidence(
    *,
    attempt_dir: str | os.PathLike[str],
    dataset: SampledDynamicalDataset,
    evaluation: DynamicalEvaluationResult,
) -> dict[str, Any]:
    """Persist evaluation outputs after the sample evidence already exists."""
    if not isinstance(
        dataset,
        SampledDynamicalDataset,
    ):
        raise TypeError(
            "dataset must be SampledDynamicalDataset."
        )

    if not isinstance(
        evaluation,
        DynamicalEvaluationResult,
    ):
        raise TypeError(
            "evaluation must be DynamicalEvaluationResult."
        )

    destination = Path(
        attempt_dir
    ).expanduser().resolve()

    if not destination.is_dir():
        raise FileNotFoundError(
            f"Attempt directory does not exist: {destination}"
        )

    sample_json = (
        destination
        / SAMPLE_METADATA_FILENAME
    )

    sample_arrays = (
        destination
        / SAMPLE_ARRAY_FILENAME
    )

    if (
        not sample_json.is_file()
        or not sample_arrays.is_file()
    ):
        raise RuntimeError(
            "Sample evidence must be persisted before evaluation evidence."
        )

    evaluation_json = (
        destination
        / EVALUATION_METADATA_FILENAME
    )

    evaluation_arrays = (
        destination
        / EVALUATION_ARRAY_FILENAME
    )

    if (
        evaluation_json.exists()
        or evaluation_arrays.exists()
    ):
        raise FileExistsError(
            "Evaluation evidence already exists for this attempt."
        )

    if (
        evaluation.n_pairs
        != dataset.sample_size
    ):
        raise ValueError(
            "Evaluation pair count does not match sampled dataset."
        )

    if (
        evaluation.state_dimension
        != dataset.current_states.shape[0]
    ):
        raise ValueError(
            "Evaluation state dimension does not match sampled dataset."
        )

    M = int(
        evaluation.n_pairs
    )

    C = int(
        evaluation.n_classes
    )

    phi = np.asarray(
        evaluation.phi_rows,
        dtype=np.float64,
    )

    successor_phi = np.asarray(
        evaluation.successor_rows,
        dtype=np.float64,
    )

    if (
        phi.shape
        != (
            M,
            C,
        )
        or successor_phi.shape
        != (
            M,
            C,
        )
    ):
        raise ValueError(
            "Retained association arrays do not have shape (M, C)."
        )

    labels = np.asarray(
        evaluation.labels,
        dtype=np.int64,
    )

    distances = np.asarray(
        evaluation.selected_center_squared_distances,
        dtype=np.float64,
    )

    if (
        labels.shape
        != (
            M,
        )
        or distances.shape
        != (
            M,
        )
    ):
        raise ValueError(
            "Evaluation labels/distances do not match pair count."
        )

    statistics = (
        evaluation.statistics
    )

    if (
        statistics.n_pairs
        != M
        or statistics.n_classes
        != C
    ):
        raise ValueError(
            "Evaluation statistics dimensions are inconsistent."
        )

    class_counts = np.asarray(
        statistics.class_counts,
        dtype=np.int64,
    )

    class_sse = np.asarray(
        statistics.class_successor_sse,
        dtype=np.float64,
    )

    class_means = np.asarray(
        statistics.class_successor_means,
        dtype=np.float64,
    )

    if (
        class_counts.shape
        != (
            C,
        )
        or class_sse.shape
        != (
            C,
        )
        or class_means.shape
        != (
            C,
            C,
        )
    ):
        raise ValueError(
            "Evaluation class sufficient statistics have invalid shape."
        )

    arrays: dict[
        str,
        NDArray[Any],
    ] = {
        "phi_rows":
            np.array(
                phi,
                copy=True,
                order="C",
            ),
        "successor_rows":
            np.array(
                successor_phi,
                copy=True,
                order="C",
            ),
        "labels":
            labels.copy(),
        "selected_center_squared_distances":
            distances.copy(),
        "class_counts":
            class_counts.copy(),
        "class_successor_sse":
            class_sse.copy(),
        "class_successor_means":
            class_means.copy(),
    }

    _write_deterministic_npz(
        evaluation_arrays,
        arrays,
    )

    arrays_sha = _sha256_file(
        evaluation_arrays
    )

    budget_records = []

    for item in (
        evaluation.budget_certificates
    ):
        certificate = (
            item.certificate
        )

        budget_records.append(
            {
                "kappa":
                    float(
                        item.kappa
                    ),
                "frozen_predictor_within_budget":
                    bool(
                        item.frozen_predictor_within_budget
                    ),
                "frozen_predictor_rmse_minus_bound":
                    float(
                        item.frozen_predictor_rmse_minus_bound
                    ),
                "certificate": {
                    "n_pairs":
                        int(
                            certificate.n_pairs
                        ),
                    "kappa":
                        float(
                            certificate.kappa
                        ),
                    "delta":
                        float(
                            certificate.delta
                        ),
                    "f_hat":
                        float(
                            certificate.f_hat
                        ),
                    "s_hat":
                        float(
                            certificate.s_hat
                        ),
                    "r_delta":
                        float(
                            certificate.r_delta
                        ),
                    "F_delta":
                        float(
                            certificate.F_delta
                        ),
                    "V_delta":
                        float(
                            certificate.V_delta
                        ),
                    "L_kappa_delta":
                        float(
                            certificate.L_kappa_delta
                        ),
                },
            }
        )

    metadata = {
        "evaluation_evidence_id":
            EVALUATION_EVIDENCE_ID,
        "evaluation_id":
            evaluation.evaluation_id,
        "reference_class_map_id":
            evaluation.reference_class_map_id,
        "system":
            dataset.system,
        "evaluation_mode":
            dataset.evaluation_mode,
        "sample_size":
            int(
                dataset.sample_size
            ),
        "replicate_index":
            int(
                dataset.replicate_index
            ),
        "n_pairs":
            M,
        "n_classes":
            C,
        "state_dimension":
            int(
                evaluation.state_dimension
            ),
        "delta":
            float(
                evaluation.delta
            ),
        "frozen_predictor_spectral_norm":
            float(
                evaluation.frozen_predictor_spectral_norm
            ),
        "frozen_predictor_mse":
            float(
                evaluation.frozen_predictor_mse
            ),
        "frozen_predictor_rmse":
            float(
                evaluation.frozen_predictor_rmse
            ),
        "statistics": {
            "f_hat":
                float(
                    statistics.f_hat
                ),
            "s_hat":
                float(
                    statistics.s_hat
                ),
            "current_max_row_sum_error":
                float(
                    statistics.current_max_row_sum_error
                ),
            "successor_max_row_sum_error":
                float(
                    statistics.successor_max_row_sum_error
                ),
            "simplex_sum_atol":
                float(
                    statistics.simplex_sum_atol
                ),
        },
        "budget_certificates":
            budget_records,
        "coordinate_ranges": {
            "phi_min":
                float(
                    evaluation.phi_coordinate_min
                ),
            "phi_max":
                float(
                    evaluation.phi_coordinate_max
                ),
            "successor_min":
                float(
                    evaluation.successor_coordinate_min
                ),
            "successor_max":
                float(
                    evaluation.successor_coordinate_max
                ),
        },
        "state_ranges": {
            "current_min":
                list(
                    evaluation.current_state_min
                ),
            "current_max":
                list(
                    evaluation.current_state_max
                ),
            "successor_min":
                list(
                    evaluation.successor_state_min
                ),
            "successor_max":
                list(
                    evaluation.successor_state_max
                ),
        },
        "array_file":
            EVALUATION_ARRAY_FILENAME,
        "array_sha256":
            arrays_sha,
        "array_members":
            sorted(
                arrays
            ),
        "sample_evidence_present_before_evaluation_write":
            True,
    }

    _write_json_exclusive(
        evaluation_json,
        metadata,
    )

    return metadata


def load_sample_evidence(
    attempt_dir: str | os.PathLike[str],
    *,
    verify_hash: bool = True,
) -> LoadedEvidence:
    destination = Path(
        attempt_dir
    ).expanduser().resolve()

    metadata_path = (
        destination
        / SAMPLE_METADATA_FILENAME
    )

    arrays_path = (
        destination
        / SAMPLE_ARRAY_FILENAME
    )

    metadata = _strict_json_load(
        metadata_path
    )

    if (
        metadata.get(
            "sample_evidence_id"
        )
        != SAMPLE_EVIDENCE_ID
    ):
        raise ValueError(
            "Unsupported sample evidence identifier."
        )

    if metadata.get(
        "array_file"
    ) != SAMPLE_ARRAY_FILENAME:
        raise ValueError(
            "Unexpected sample array filename."
        )

    if verify_hash:
        actual = _sha256_file(
            arrays_path
        )

        if (
            actual
            != metadata.get(
                "array_sha256"
            )
        ):
            raise ValueError(
                "Sample array SHA256 mismatch."
            )

    expected_names = {
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

    if set(
        metadata.get(
            "array_members",
            [],
        )
    ) != expected_names:
        raise ValueError(
            "Sample metadata array-member inventory mismatch."
        )

    arrays = _load_npz_exact(
        arrays_path,
        expected_names,
    )

    return LoadedEvidence(
        metadata=
            metadata,
        arrays=
            arrays,
    )


def load_evaluation_evidence(
    attempt_dir: str | os.PathLike[str],
    *,
    verify_hash: bool = True,
) -> LoadedEvidence:
    destination = Path(
        attempt_dir
    ).expanduser().resolve()

    metadata_path = (
        destination
        / EVALUATION_METADATA_FILENAME
    )

    arrays_path = (
        destination
        / EVALUATION_ARRAY_FILENAME
    )

    metadata = _strict_json_load(
        metadata_path
    )

    if (
        metadata.get(
            "evaluation_evidence_id"
        )
        != EVALUATION_EVIDENCE_ID
    ):
        raise ValueError(
            "Unsupported evaluation evidence identifier."
        )

    if metadata.get(
        "array_file"
    ) != EVALUATION_ARRAY_FILENAME:
        raise ValueError(
            "Unexpected evaluation array filename."
        )

    if verify_hash:
        actual = _sha256_file(
            arrays_path
        )

        if (
            actual
            != metadata.get(
                "array_sha256"
            )
        ):
            raise ValueError(
                "Evaluation array SHA256 mismatch."
            )

    expected_names = {
        "phi_rows",
        "successor_rows",
        "labels",
        "selected_center_squared_distances",
        "class_counts",
        "class_successor_sse",
        "class_successor_means",
    }

    if set(
        metadata.get(
            "array_members",
            [],
        )
    ) != expected_names:
        raise ValueError(
            "Evaluation metadata array-member inventory mismatch."
        )

    arrays = _load_npz_exact(
        arrays_path,
        expected_names,
    )

    return LoadedEvidence(
        metadata=
            metadata,
        arrays=
            arrays,
    )
