"""Fit and freeze KAHKM representations for dynamical certificate studies.

This module intentionally takes every representation hyperparameter from the
frozen certificate configuration rather than from Experiment 15's
``SYSTEM_CONFIGS`` defaults. The latter differ from the retained tuned
representations used by the certificate protocol.

No fitting occurs at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from experiment_15_external_koopman_baselines import (
    make_concatenated_training_data,
)
from kahkm_frozen_representation import (
    FORMAT_ID,
    save_frozen_representation,
)
from kernel_affine_hull_koopman_machines import (
    fit_kahkm,
)


DYNAMICAL_PILOT_CONFIG_RELATIVE = (
    "reproduction/certificates/configs/"
    "dynamical_pilot_v1.json"
)

DYNAMICAL_PILOT_CONFIG_SHA256 = (
    "3fdee83ec36f8e7a62032407875738e5687d2da06884ae0a124fb43498da50b1"
)

SUPPORTED_SYSTEMS = (
    "duffing",
    "vanderpol",
)


@dataclass(frozen=True)
class FrozenBuildResult:
    """Metadata returned after successfully fitting and freezing one system."""

    system: str
    artifact_dir: str
    training_snapshot_count: int
    n_clusters: int
    omega: float
    tau: float
    nlms_spectral_norm: float
    train_closure_error: float
    association_r2: float
    configuration_sha256: str


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


def load_frozen_dynamical_pilot_config(
    *,
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load the exact committed pilot configuration and verify its bytes."""
    root = Path(
        repository_root
    ).expanduser().resolve()

    path = (
        root
        / DYNAMICAL_PILOT_CONFIG_RELATIVE
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen dynamical pilot configuration not found: {path}"
        )

    actual_hash = sha256_file(
        path
    )

    if actual_hash != DYNAMICAL_PILOT_CONFIG_SHA256:
        raise ValueError(
            "Frozen dynamical pilot configuration SHA256 mismatch: "
            f"{actual_hash} != {DYNAMICAL_PILOT_CONFIG_SHA256}."
        )

    with path.open(
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
            "Frozen dynamical pilot configuration must be a JSON object."
        )

    return payload


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


def _require_positive_int(
    name: str,
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    result = int(value)

    if result <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _require_nonnegative_int(
    name: str,
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    result = int(value)

    if result < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )

    return result


def _require_positive_float(
    name: str,
    value: Any,
) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real scalar."
        )

    try:
        result = float(
            value
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a real scalar."
        ) from exc

    if (
        not np.isfinite(result)
        or result <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and positive."
        )

    return result


def _system_training_settings(
    config: Mapping[str, Any],
    system: str,
) -> dict[str, Any]:
    if system not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"Unsupported system: {system!r}."
        )

    frozen = _require_mapping(
        "frozen_representation",
        config.get(
            "frozen_representation"
        ),
    )

    systems = _require_mapping(
        "frozen_representation.systems",
        frozen.get(
            "systems"
        ),
    )

    system_cfg = _require_mapping(
        f"systems.{system}",
        systems.get(
            system
        ),
    )

    train_seeds_raw = frozen.get(
        "train_seeds"
    )

    if not isinstance(
        train_seeds_raw,
        list,
    ) or not train_seeds_raw:
        raise ValueError(
            "train_seeds must be a non-empty JSON list."
        )

    train_seeds = tuple(
        _require_nonnegative_int(
            f"train_seeds[{index}]",
            value,
        )
        for index, value
        in enumerate(
            train_seeds_raw
        )
    )

    if len(
        set(train_seeds)
    ) != len(train_seeds):
        raise ValueError(
            "train_seeds must be unique."
        )

    max_train = frozen.get(
        "max_train_per_cluster"
    )

    if max_train is not None:
        max_train = _require_positive_int(
            "max_train_per_cluster",
            max_train,
        )

    kmeans_kind = frozen.get(
        "kmeans_kind"
    )

    if kmeans_kind not in {
        "auto",
        "full",
        "minibatch",
    }:
        raise ValueError(
            "Unsupported kmeans_kind in frozen configuration."
        )

    if frozen.get(
        "save_ae_to_disk"
    ) is not True:
        raise ValueError(
            "Frozen dynamical representation requires save_ae_to_disk=true."
        )

    if frozen.get(
        "project_stochastic"
    ) is not False:
        raise ValueError(
            "Frozen dynamical representation requires project_stochastic=false."
        )

    format_id = frozen.get(
        "format_id"
    )

    if format_id != FORMAT_ID:
        raise ValueError(
            "Frozen-representation format ID mismatch."
        )

    return {
        "train_seeds":
            train_seeds,
        "n_steps_train":
            _require_positive_int(
                "n_steps_train",
                frozen.get(
                    "n_steps_train"
                ),
            ),
        "subspace_dim":
            _require_positive_int(
                "subspace_dim",
                frozen.get(
                    "subspace_dim"
                ),
            ),
        "Nb":
            _require_positive_int(
                "Nb",
                frozen.get(
                    "Nb"
                ),
            ),
        "beta":
            _require_positive_float(
                "beta",
                frozen.get(
                    "beta"
                ),
            ),
        "nlms_epochs":
            _require_positive_int(
                "nlms_epochs",
                frozen.get(
                    "nlms_epochs"
                ),
            ),
        "kmeans_kind":
            str(
                kmeans_kind
            ),
        "max_train_per_cluster":
            max_train,
        "random_state":
            _require_nonnegative_int(
                "representation_random_state",
                frozen.get(
                    "representation_random_state"
                ),
            ),
        "batch_size":
            _require_positive_int(
                "batch_size",
                frozen.get(
                    "batch_size"
                ),
            ),
        "n_jobs":
            int(
                frozen.get(
                    "n_jobs"
                )
            ),
        "dt":
            _require_positive_float(
                "dt",
                system_cfg.get(
                    "dt"
                ),
            ),
        "n_clusters":
            _require_positive_int(
                "n_clusters",
                system_cfg.get(
                    "n_clusters"
                ),
            ),
        "omega":
            _require_positive_float(
                "omega",
                system_cfg.get(
                    "omega"
                ),
            ),
        "tau":
            _require_positive_float(
                "tau",
                system_cfg.get(
                    "tau"
                ),
            ),
        "vanderpol_mu":
            _require_positive_float(
                "vanderpol_mu",
                system_cfg.get(
                    "vanderpol_mu"
                ),
            ),
    }


def fit_and_freeze_dynamical_representation(
    *,
    repository_root: str | os.PathLike[str],
    system: str,
    output_dir: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
) -> FrozenBuildResult:
    """Fit and freeze one representation using only the committed pilot config.

    The work directory contains temporary disk-backed autoencoder shards
    during fitting and is removed after a successful portable export.

    Both ``output_dir`` and ``work_dir`` must not exist on entry.
    """
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

    if output == work:
        raise ValueError(
            "output_dir and work_dir must differ."
        )

    config = load_frozen_dynamical_pilot_config(
        repository_root=root
    )

    settings = _system_training_settings(
        config,
        system,
    )

    # Construct training data exactly through Experiment 15's established
    # concatenated-trajectory helper. C and omega do NOT come from
    # Experiment 15 SYSTEM_CONFIGS; they come from the frozen certificate
    # configuration above.
    X0_train, X1_train = (
        make_concatenated_training_data(
            system,  # type: ignore[arg-type]
            train_seeds=
                settings[
                    "train_seeds"
                ],
            n_steps_train=
                settings[
                    "n_steps_train"
                ],
            dt=settings[
                "dt"
            ],
            vanderpol_mu=
                settings[
                    "vanderpol_mu"
                ],
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
        * settings[
            "n_steps_train"
        ]
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
            "Unexpected concatenated training-data shape: "
            f"X0={X0_train.shape}, X1={X1_train.shape}, "
            f"expected=(2, {expected_snapshots})."
        )

    if (
        not np.all(
            np.isfinite(
                X0_train
            )
        )
        or not np.all(
            np.isfinite(
                X1_train
            )
        )
    ):
        raise RuntimeError(
            "Training data contain nonfinite values."
        )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    ae_dir = (
        work
        / "autoencoders"
    )

    try:
        fit = fit_kahkm(
            X0_train,
            X1_train,
            n_clusters=
                settings[
                    "n_clusters"
                ],
            subspace_dim=
                settings[
                    "subspace_dim"
                ],
            Nb=settings[
                "Nb"
            ],
            omega=settings[
                "omega"
            ],
            tau=settings[
                "tau"
            ],
            beta=settings[
                "beta"
            ],
            nlms_epochs=
                settings[
                    "nlms_epochs"
                ],
            random_state=
                settings[
                    "random_state"
                ],
            kmeans_kind=
                settings[
                    "kmeans_kind"
                ],  # type: ignore[arg-type]
            max_train_per_cluster=
                settings[
                    "max_train_per_cluster"
                ],
            save_ae_to_disk=True,
            ae_dir=str(
                ae_dir
            ),
            overwrite_ae_dir=False,
            n_jobs=settings[
                "n_jobs"
            ],
            batch_size=
                settings[
                    "batch_size"
                ],
            project_stochastic=False,
            preload_classifier_after_fit=False,
            verbose=False,
        )

        abstraction_model = getattr(
            fit,
            "abstraction_model",
        )

        B = np.asarray(
            getattr(
                fit,
                "B",
            ),
            dtype=np.float64,
        )

        n_clusters = int(
            settings[
                "n_clusters"
            ]
        )

        if B.shape != (
            n_clusters,
            n_clusters,
        ):
            raise RuntimeError(
                "Unexpected fitted NLMS matrix shape: "
                f"{B.shape}."
            )

        if not np.all(
            np.isfinite(B)
        ):
            raise RuntimeError(
                "Fitted NLMS matrix contains nonfinite values."
            )

        actual_effective_clusters = int(
            abstraction_model.get(
                "n_clusters",
                -1,
            )
        )

        if (
            actual_effective_clusters
            != n_clusters
        ):
            raise RuntimeError(
                "Effective fitted cluster count differs from frozen "
                f"requested count: {actual_effective_clusters} != "
                f"{n_clusters}."
            )

        nlms_norm = float(
            np.linalg.norm(
                B,
                ord=2,
            )
        )

        if (
            not np.isfinite(
                nlms_norm
            )
            or nlms_norm < 0.0
        ):
            raise RuntimeError(
                "Invalid fitted NLMS spectral norm."
            )

        training_metadata = {
            "configuration_relative_path":
                DYNAMICAL_PILOT_CONFIG_RELATIVE,
            "configuration_sha256":
                DYNAMICAL_PILOT_CONFIG_SHA256,
            "system":
                system,
            "train_seeds":
                list(
                    settings[
                        "train_seeds"
                    ]
                ),
            "n_steps_train":
                settings[
                    "n_steps_train"
                ],
            "n_train_snapshots":
                int(
                    X0_train.shape[1]
                ),
            "state_dimension":
                int(
                    X0_train.shape[0]
                ),
            "dt":
                settings[
                    "dt"
                ],
            "vanderpol_mu":
                settings[
                    "vanderpol_mu"
                ],
            "n_clusters":
                n_clusters,
            "subspace_dim":
                settings[
                    "subspace_dim"
                ],
            "Nb":
                settings[
                    "Nb"
                ],
            "omega":
                settings[
                    "omega"
                ],
            "tau":
                settings[
                    "tau"
                ],
            "beta":
                settings[
                    "beta"
                ],
            "nlms_epochs":
                settings[
                    "nlms_epochs"
                ],
            "kmeans_kind":
                settings[
                    "kmeans_kind"
                ],
            "max_train_per_cluster":
                settings[
                    "max_train_per_cluster"
                ],
            "representation_random_state":
                settings[
                    "random_state"
                ],
            "batch_size":
                settings[
                    "batch_size"
                ],
            "n_jobs":
                settings[
                    "n_jobs"
                ],
            "project_stochastic":
                False,
            "nlms_spectral_norm":
                nlms_norm,
        }

        save_frozen_representation(
            output_dir=output,
            system=system,
            abstraction_model=
                abstraction_model,
            B=B,
            train_closure_error=
                getattr(
                    fit,
                    "train_closure_error",
                ),
            association_r2=
                getattr(
                    fit,
                    "association_r2",
                ),
            nlms_history=
                getattr(
                    fit,
                    "nlms_history",
                ),
            training_metadata=
                training_metadata,
        )

        result = FrozenBuildResult(
            system=system,
            artifact_dir=str(
                output
            ),
            training_snapshot_count=
                int(
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
                nlms_norm,
            train_closure_error=
                float(
                    getattr(
                        fit,
                        "train_closure_error",
                    )
                ),
            association_r2=
                float(
                    getattr(
                        fit,
                        "association_r2",
                    )
                ),
            configuration_sha256=
                DYNAMICAL_PILOT_CONFIG_SHA256,
        )

    except BaseException:
        # A failed build must not leave a portable artifact that could later be
        # mistaken for a completed representation.
        if output.exists():
            shutil.rmtree(
                output,
                ignore_errors=True,
            )

        raise

    finally:
        # The portable artifact contains its own copied shard tree, so the
        # temporary fit workspace is never part of retained evidence.
        if work.exists():
            shutil.rmtree(
                work,
                ignore_errors=True,
            )

    return result
