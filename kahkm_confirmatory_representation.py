"""Training-only derivation of the confirmatory Van der Pol representation.

This module intentionally does not modify or generalize the earlier pilot
representation builder.  It consumes the committed confirmatory design,
restores the already-verified Van der Pol KAHM abstraction, changes only the
prespecified association sharpness omega, recomputes training associations,
refits the manuscript NLMS operator from the original training trajectories,
and exports a new self-contained frozen representation.

No confirmatory evaluation pair is consumed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from experiment_15_external_koopman_baselines import (
    make_concatenated_training_data,
)
from kahkm_dynamical_frozen_source import (
    restore_repository_frozen_representations,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    load_frozen_representation,
    save_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    kahm_associations,
    nlms_koopman_closure,
)


DERIVATION_ID = "frozen_abstraction_omega_refit_nlms_v1"

CONFIRMATORY_CONFIG_RELATIVE = (
    "reproduction/certificates/configs/"
    "dynamical_confirmatory_v1.json"
)

CONFIRMATORY_CONFIG_SHA256 = (
    "1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91"
)

CONFIRMATORY_PROTOCOL_RELATIVE = (
    "reproduction/certificates/"
    "DYNAMICAL_CONFIRMATORY_PROTOCOL.md"
)

CONFIRMATORY_PROTOCOL_SHA256 = (
    "daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c"
)

EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "04a5ad3d4dae3a8fafce2576512fdce06d6c738907ced7d3ad9fde600d8c0e2a"
)

EXPECTED_SOURCE_CONFIG_SHA256 = (
    "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1"
)

EXPECTED_SOURCE_BUILD_COMMIT = (
    "1add19b94372257d16118f60c69e1e8afa668259"
)

EXPECTED_SOURCE_VERIFIER_COMMIT = (
    "65db148f865c94f988676381c6b5f12222356b31"
)

SYSTEM = "vanderpol"


@dataclass(frozen=True)
class ConfirmatoryRepresentationResult:
    """Metadata returned after one successful confirmatory derivation."""

    system: str
    artifact_dir: str
    derivation_id: str
    configuration_sha256: str
    protocol_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    n_train_snapshots: int
    n_clusters: int
    omega: float
    tau: float
    nlms_spectral_norm: float
    train_closure_error: float
    association_r2: float


def sha256_file(
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


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    return value


def _load_confirmatory_config(
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(
        repository_root
    ).expanduser().resolve()

    config_path = (
        root
        / CONFIRMATORY_CONFIG_RELATIVE
    )

    protocol_path = (
        root
        / CONFIRMATORY_PROTOCOL_RELATIVE
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Confirmatory configuration not found: {config_path}"
        )

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Confirmatory protocol not found: {protocol_path}"
        )

    actual_config_sha = sha256_file(
        config_path
    )

    if (
        actual_config_sha
        != CONFIRMATORY_CONFIG_SHA256
    ):
        raise ValueError(
            "Confirmatory configuration SHA256 mismatch: "
            f"{actual_config_sha} != "
            f"{CONFIRMATORY_CONFIG_SHA256}."
        )

    actual_protocol_sha = sha256_file(
        protocol_path
    )

    if (
        actual_protocol_sha
        != CONFIRMATORY_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Confirmatory protocol SHA256 mismatch: "
            f"{actual_protocol_sha} != "
            f"{CONFIRMATORY_PROTOCOL_SHA256}."
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Confirmatory configuration must be a JSON object."
        )

    if (
        payload.get(
            "campaign_id"
        )
        != "dynamical_original_iid_confirmatory_v1"
    ):
        raise ValueError(
            "Unexpected confirmatory campaign_id."
        )

    if (
        payload.get(
            "campaign_role"
        )
        != "confirmatory"
    ):
        raise ValueError(
            "Unexpected confirmatory campaign_role."
        )

    if (
        payload.get(
            "protocol"
        )
        != CONFIRMATORY_PROTOCOL_RELATIVE
    ):
        raise ValueError(
            "Unexpected confirmatory protocol path."
        )

    if (
        payload.get(
            "protocol_sha256"
        )
        != CONFIRMATORY_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Unexpected confirmatory protocol SHA256."
        )

    return payload


def _validate_design(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    design = _require_mapping(
        "confirmatory_design",
        config.get(
            "confirmatory_design"
        ),
    )

    if design.get(
        "systems"
    ) != [
        SYSTEM
    ]:
        raise ValueError(
            "Confirmatory design must contain only Van der Pol."
        )

    if design.get(
        "sample_sizes"
    ) != [
        16384
    ]:
        raise ValueError(
            "Unexpected confirmatory sample-size design."
        )

    if int(
        design.get(
            "replicates_per_cell",
            -1,
        )
    ) != 32:
        raise ValueError(
            "Unexpected confirmatory replicate count."
        )

    if int(
        design.get(
            "planned_attempts",
            -1,
        )
    ) != 64:
        raise ValueError(
            "Unexpected confirmatory attempt count."
        )

    if int(
        design.get(
            "planned_retained_pairs",
            -1,
        )
    ) != 1048576:
        raise ValueError(
            "Unexpected confirmatory retained-pair count."
        )

    return dict(design)


def _representation_settings(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source = _require_mapping(
        "source_frozen_abstraction",
        config.get(
            "source_frozen_abstraction"
        ),
    )

    confirm = _require_mapping(
        "confirmatory_representation",
        config.get(
            "confirmatory_representation"
        ),
    )

    if (
        confirm.get(
            "implementation_id"
        )
        != DERIVATION_ID
    ):
        raise ValueError(
            "Unexpected confirmatory representation implementation ID."
        )

    if (
        confirm.get(
            "system"
        )
        != SYSTEM
    ):
        raise ValueError(
            "Confirmatory representation system mismatch."
        )

    if (
        source.get(
            "representation_archive_sha256"
        )
        != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError(
            "Unexpected source representation archive SHA256."
        )

    if (
        source.get(
            "source_configuration_sha256"
        )
        != EXPECTED_SOURCE_CONFIG_SHA256
    ):
        raise ValueError(
            "Unexpected source configuration SHA256."
        )

    if (
        source.get(
            "representation_build_execution_commit"
        )
        != EXPECTED_SOURCE_BUILD_COMMIT
    ):
        raise ValueError(
            "Unexpected source build execution commit."
        )

    if (
        source.get(
            "representation_verifier_source_commit"
        )
        != EXPECTED_SOURCE_VERIFIER_COMMIT
    ):
        raise ValueError(
            "Unexpected source representation verifier commit."
        )

    for name in (
        "reuse_state_space_kmeans_centers",
        "reuse_kahm_autoencoders",
    ):
        if source.get(
            name
        ) is not True:
            raise ValueError(
                f"{name} must be true."
            )

    for name in (
        "refit_kmeans",
        "refit_autoencoders",
    ):
        if source.get(
            name
        ) is not False:
            raise ValueError(
                f"{name} must be false."
            )

    if (
        confirm.get(
            "operator_refit_uses_training_data_only"
        )
        is not True
    ):
        raise ValueError(
            "Operator refit must use training data only."
        )

    if float(
        confirm.get(
            "omega"
        )
    ) != 128.0:
        raise ValueError(
            "Confirmatory omega must equal 128."
        )

    if float(
        confirm.get(
            "tau"
        )
    ) != 1e-6:
        raise ValueError(
            "Confirmatory tau must equal 1e-6."
        )

    if confirm.get(
        "train_seeds"
    ) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            "Unexpected training seeds."
        )

    if int(
        confirm.get(
            "n_steps_train"
        )
    ) != 1200:
        raise ValueError(
            "Unexpected training trajectory length."
        )

    if float(
        confirm.get(
            "beta"
        )
    ) != 0.1:
        raise ValueError(
            "Unexpected NLMS beta."
        )

    if int(
        confirm.get(
            "nlms_epochs"
        )
    ) != 20:
        raise ValueError(
            "Unexpected NLMS epoch count."
        )

    if (
        confirm.get(
            "nlms_shuffle"
        )
        is not False
    ):
        raise ValueError(
            "Confirmatory NLMS must not shuffle."
        )

    if (
        confirm.get(
            "nlms_initial_matrix"
        )
        != "identity"
    ):
        raise ValueError(
            "Confirmatory NLMS must use identity initialization."
        )

    if (
        confirm.get(
            "predictor_orientation"
        )
        != "B.T @ Phi"
    ):
        raise ValueError(
            "Unexpected predictor orientation."
        )

    if (
        confirm.get(
            "project_stochastic"
        )
        is not False
    ):
        raise ValueError(
            "Confirmatory B must not be stochastically projected."
        )

    return dict(confirm)


def _relative_closure_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    pred = np.asarray(
        prediction,
        dtype=np.float64,
    )

    truth = np.asarray(
        target,
        dtype=np.float64,
    )

    numerator = float(
        np.sum(
            (
                truth
                - pred
            ) ** 2,
            dtype=np.float64,
        )
    )

    denominator = float(
        np.sum(
            truth
            * truth,
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
    pred = np.asarray(
        prediction,
        dtype=np.float64,
    )

    truth = np.asarray(
        target,
        dtype=np.float64,
    )

    residual_ss = float(
        np.sum(
            (
                truth
                - pred
            ) ** 2,
            dtype=np.float64,
        )
    )

    centered = (
        truth
        - np.mean(
            truth,
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

    if (
        not math.isfinite(
            total_ss
        )
        or total_ss <= 0.0
    ):
        raise RuntimeError(
            "Training association total variance must be positive."
        )

    value = (
        1.0
        - residual_ss
        / total_ss
    )

    if not math.isfinite(
        value
    ):
        raise RuntimeError(
            "Nonfinite training association R2."
        )

    return value


def derive_confirmatory_representation(
    *,
    repository_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
) -> ConfirmatoryRepresentationResult:
    """Derive and freeze the prespecified confirmatory Van der Pol model."""

    root = Path(
        repository_root
    ).expanduser().resolve()

    output = Path(
        output_dir
    ).expanduser().resolve()

    work = Path(
        work_dir
    ).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    if work.exists():
        raise FileExistsError(
            f"Work directory already exists: {work}"
        )

    config = _load_confirmatory_config(
        root
    )

    _validate_design(
        config
    )

    settings = _representation_settings(
        config
    )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        restored = (
            restore_repository_frozen_representations(
                repository_root=root,
                restoration_dir=(
                    work
                    / "restored_source"
                ),
            )
        )

        if (
            restored.archive_sha256
            != EXPECTED_SOURCE_ARCHIVE_SHA256
        ):
            raise RuntimeError(
                "Restored source archive SHA256 mismatch."
            )

        if (
            restored.configuration_sha256
            != EXPECTED_SOURCE_CONFIG_SHA256
        ):
            raise RuntimeError(
                "Restored source configuration SHA256 mismatch."
            )

        if (
            restored.build_execution_commit
            != EXPECTED_SOURCE_BUILD_COMMIT
        ):
            raise RuntimeError(
                "Restored source build commit mismatch."
            )

        if (
            restored.verifier_source_commit
            != EXPECTED_SOURCE_VERIFIER_COMMIT
        ):
            raise RuntimeError(
                "Restored source verifier commit mismatch."
            )

        if (
            restored.archived_verification_status
            != "passed"
        ):
            raise RuntimeError(
                "Source representation archive is not verified-passed."
            )

        source_artifact = Path(
            restored.vanderpol_artifact_dir
        )

        source_manifest_path = (
            source_artifact
            / "manifest.json"
        )

        source_manifest_sha = (
            sha256_file(
                source_manifest_path
            )
        )

        source_model = (
            load_frozen_representation(
                source_artifact,
                verify_hashes=True,
            )
        )

        if (
            source_model.system
            != SYSTEM
        ):
            raise RuntimeError(
                "Restored source system mismatch."
            )

        if (
            source_model.format_id
            != FORMAT_ID
        ):
            raise RuntimeError(
                "Restored source format mismatch."
            )

        if float(
            source_model.omega
        ) != 4.0:
            raise RuntimeError(
                "Restored source omega must equal 4."
            )

        if float(
            source_model.tau
        ) != 1e-6:
            raise RuntimeError(
                "Restored source tau must equal 1e-6."
            )

        source_training = _require_mapping(
            "source training_metadata",
            source_model.manifest.get(
                "training_metadata"
            ),
        )

        if (
            source_training.get(
                "train_seeds"
            )
            != settings[
                "train_seeds"
            ]
        ):
            raise RuntimeError(
                "Source training seeds differ from confirmatory design."
            )

        if int(
            source_training.get(
                "n_steps_train",
                -1,
            )
        ) != int(
            settings[
                "n_steps_train"
            ]
        ):
            raise RuntimeError(
                "Source training length differs from confirmatory design."
            )

        if float(
            source_training.get(
                "beta"
            )
        ) != float(
            settings[
                "beta"
            ]
        ):
            raise RuntimeError(
                "Source beta differs from confirmatory design."
            )

        if int(
            source_training.get(
                "nlms_epochs",
                -1,
            )
        ) != int(
            settings[
                "nlms_epochs"
            ]
        ):
            raise RuntimeError(
                "Source NLMS epochs differ from confirmatory design."
            )

        n_clusters = int(
            source_model.abstraction_model[
                "n_clusters"
            ]
        )

        if n_clusters != 25:
            raise RuntimeError(
                "Confirmatory source must contain 25 classes."
            )

        X0_train, X1_train = (
            make_concatenated_training_data(
                SYSTEM,
                train_seeds=tuple(
                    int(value)
                    for value in settings[
                        "train_seeds"
                    ]
                ),
                n_steps_train=int(
                    settings[
                        "n_steps_train"
                    ]
                ),
                dt=float(
                    source_training[
                        "dt"
                    ]
                ),
                vanderpol_mu=float(
                    source_training[
                        "vanderpol_mu"
                    ]
                ),
            )
        )

        X0_train = np.asarray(
            X0_train,
            dtype=np.float64,
        )

        X1_train = np.asarray(
            X1_train,
            dtype=np.float64,
        )

        expected_snapshots = (
            len(
                settings[
                    "train_seeds"
                ]
            )
            * int(
                settings[
                    "n_steps_train"
                ]
            )
        )

        if (
            X0_train.shape
            != (
                2,
                expected_snapshots,
            )
            or X1_train.shape
            != X0_train.shape
        ):
            raise RuntimeError(
                "Unexpected confirmatory training snapshot shape."
            )

        abstraction_model = dict(
            source_model.abstraction_model
        )

        abstraction_model[
            "kahkm_omega"
        ] = float(
            settings[
                "omega"
            ]
        )

        abstraction_model[
            "kahkm_tau"
        ] = float(
            settings[
                "tau"
            ]
        )

        Phi = np.asarray(
            kahm_associations(
                abstraction_model,
                X0_train,
                omega=float(
                    settings[
                        "omega"
                    ]
                ),
                tau=float(
                    settings[
                        "tau"
                    ]
                ),
                n_jobs=int(
                    settings[
                        "n_jobs"
                    ]
                ),
                batch_size=int(
                    settings[
                        "batch_size"
                    ]
                ),
                show_progress=False,
            ),
            dtype=np.float64,
        )

        Chi = np.asarray(
            kahm_associations(
                abstraction_model,
                X1_train,
                omega=float(
                    settings[
                        "omega"
                    ]
                ),
                tau=float(
                    settings[
                        "tau"
                    ]
                ),
                n_jobs=int(
                    settings[
                        "n_jobs"
                    ]
                ),
                batch_size=int(
                    settings[
                        "batch_size"
                    ]
                ),
                show_progress=False,
            ),
            dtype=np.float64,
        )

        expected_association_shape = (
            n_clusters,
            expected_snapshots,
        )

        if (
            Phi.shape
            != expected_association_shape
            or Chi.shape
            != expected_association_shape
        ):
            raise RuntimeError(
                "Unexpected training association shape."
            )

        if (
            not np.all(
                np.isfinite(
                    Phi
                )
            )
            or not np.all(
                np.isfinite(
                    Chi
                )
            )
        ):
            raise RuntimeError(
                "Training associations contain nonfinite values."
            )

        for name, array in (
            (
                "Phi",
                Phi,
            ),
            (
                "Chi",
                Chi,
            ),
        ):
            if float(
                np.max(
                    np.abs(
                        np.sum(
                            array,
                            axis=0,
                            dtype=np.float64,
                        )
                        - 1.0
                    )
                )
            ) > 1e-12:
                raise RuntimeError(
                    f"{name} violates simplex row-sum tolerance."
                )

            if (
                float(
                    np.min(
                        array
                    )
                ) < 0.0
                or float(
                    np.max(
                        array
                    )
                ) > 1.0
            ):
                raise RuntimeError(
                    f"{name} lies outside [0, 1]."
                )

        B, history = (
            nlms_koopman_closure(
                Phi,
                Chi,
                beta=float(
                    settings[
                        "beta"
                    ]
                ),
                epochs=int(
                    settings[
                        "nlms_epochs"
                    ]
                ),
                shuffle=False,
                random_state=0,
                B0=None,
            )
        )

        B = np.asarray(
            B,
            dtype=np.float64,
        )

        if B.shape != (
            n_clusters,
            n_clusters,
        ):
            raise RuntimeError(
                "Unexpected refitted operator shape."
            )

        if not np.all(
            np.isfinite(
                B
            )
        ):
            raise RuntimeError(
                "Refitted operator contains nonfinite values."
            )

        if len(
            history
        ) != int(
            settings[
                "nlms_epochs"
            ]
        ):
            raise RuntimeError(
                "Unexpected NLMS history length."
            )

        prediction = (
            B.T
            @ Phi
        )

        train_error = (
            _relative_closure_error(
                prediction,
                Chi,
            )
        )

        association_r2 = (
            _association_r2(
                prediction,
                Chi,
            )
        )

        spectral_norm = float(
            np.linalg.norm(
                B,
                ord=2,
            )
        )

        if (
            not math.isfinite(
                spectral_norm
            )
            or spectral_norm < 0.0
        ):
            raise RuntimeError(
                "Invalid refitted operator spectral norm."
            )

        training_metadata = {
            "derivation_id":
                DERIVATION_ID,

            "configuration_relative_path":
                CONFIRMATORY_CONFIG_RELATIVE,

            "configuration_sha256":
                CONFIRMATORY_CONFIG_SHA256,

            "protocol_relative_path":
                CONFIRMATORY_PROTOCOL_RELATIVE,

            "protocol_sha256":
                CONFIRMATORY_PROTOCOL_SHA256,

            "system":
                SYSTEM,

            "source_representation_archive_sha256":
                restored.archive_sha256,

            "source_representation_manifest_sha256":
                source_manifest_sha,

            "source_representation_build_execution_commit":
                restored.build_execution_commit,

            "source_representation_verifier_source_commit":
                restored.verifier_source_commit,

            "source_representation_verification_status":
                restored.archived_verification_status,

            "source_omega":
                float(
                    source_model.omega
                ),

            "source_tau":
                float(
                    source_model.tau
                ),

            "reuse_state_space_kmeans_centers":
                True,

            "reuse_kahm_autoencoders":
                True,

            "refit_kmeans":
                False,

            "refit_autoencoders":
                False,

            "train_seeds":
                list(
                    settings[
                        "train_seeds"
                    ]
                ),

            "n_steps_train":
                int(
                    settings[
                        "n_steps_train"
                    ]
                ),

            "n_train_snapshots":
                int(
                    X0_train.shape[1]
                ),

            "state_dimension":
                int(
                    X0_train.shape[0]
                ),

            "n_clusters":
                n_clusters,

            "dt":
                float(
                    source_training[
                        "dt"
                    ]
                ),

            "vanderpol_mu":
                float(
                    source_training[
                        "vanderpol_mu"
                    ]
                ),

            "omega":
                float(
                    settings[
                        "omega"
                    ]
                ),

            "tau":
                float(
                    settings[
                        "tau"
                    ]
                ),

            "beta":
                float(
                    settings[
                        "beta"
                    ]
                ),

            "nlms_epochs":
                int(
                    settings[
                        "nlms_epochs"
                    ]
                ),

            "nlms_shuffle":
                False,

            "nlms_initial_matrix":
                "identity",

            "predictor_orientation":
                "B.T @ Phi",

            "project_stochastic":
                False,

            "batch_size":
                int(
                    settings[
                        "batch_size"
                    ]
                ),

            "n_jobs":
                int(
                    settings[
                        "n_jobs"
                    ]
                ),

            "nlms_spectral_norm":
                spectral_norm,
        }

        save_frozen_representation(
            output_dir=output,
            system=SYSTEM,
            abstraction_model=abstraction_model,
            B=B,
            train_closure_error=train_error,
            association_r2=association_r2,
            nlms_history=history,
            training_metadata=training_metadata,
        )

        reloaded = (
            load_frozen_representation(
                output,
                verify_hashes=True,
            )
        )

        if reloaded.system != SYSTEM:
            raise RuntimeError(
                "Reloaded confirmatory system mismatch."
            )

        if float(
            reloaded.omega
        ) != float(
            settings[
                "omega"
            ]
        ):
            raise RuntimeError(
                "Reloaded confirmatory omega mismatch."
            )

        if float(
            reloaded.tau
        ) != float(
            settings[
                "tau"
            ]
        ):
            raise RuntimeError(
                "Reloaded confirmatory tau mismatch."
            )

        if not np.array_equal(
            reloaded.B,
            B,
        ):
            raise RuntimeError(
                "Reloaded operator differs from saved operator."
            )

        reloaded_training = _require_mapping(
            "reloaded training_metadata",
            reloaded.manifest.get(
                "training_metadata"
            ),
        )

        if (
            reloaded_training.get(
                "derivation_id"
            )
            != DERIVATION_ID
        ):
            raise RuntimeError(
                "Reloaded derivation ID mismatch."
            )

        if (
            reloaded_training.get(
                "configuration_sha256"
            )
            != CONFIRMATORY_CONFIG_SHA256
        ):
            raise RuntimeError(
                "Reloaded configuration SHA256 mismatch."
            )

        return ConfirmatoryRepresentationResult(
            system=SYSTEM,
            artifact_dir=str(
                output
            ),
            derivation_id=
                DERIVATION_ID,
            configuration_sha256=
                CONFIRMATORY_CONFIG_SHA256,
            protocol_sha256=
                CONFIRMATORY_PROTOCOL_SHA256,
            source_archive_sha256=
                restored.archive_sha256,
            source_manifest_sha256=
                source_manifest_sha,
            n_train_snapshots=int(
                X0_train.shape[1]
            ),
            n_clusters=
                n_clusters,
            omega=float(
                settings[
                    "omega"
                ]
            ),
            tau=float(
                settings[
                    "tau"
                ]
            ),
            nlms_spectral_norm=
                spectral_norm,
            train_closure_error=
                train_error,
            association_r2=
                association_r2,
        )

    except BaseException:
        if output.exists():
            shutil.rmtree(
                output,
                ignore_errors=True,
            )

        raise

    finally:
        if work.exists():
            shutil.rmtree(
                work,
                ignore_errors=True,
            )
