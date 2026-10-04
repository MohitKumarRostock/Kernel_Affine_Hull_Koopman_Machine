#!/usr/bin/env python3
"""Summarize the verified confirmatory dynamical certificate campaign.

This summarizer is intentionally read-only with respect to campaign evidence.

Before aggregation it invokes the independent confirmatory numerical verifier
again. No random dynamical pair dataset is generated and no representation is
refit.

Outputs:
- mode_summary.csv:
    primary fitted-operator-norm certificate summary by evaluation mode;
- budget_summary.csv:
    diagnostic summary for all four frozen norm budgets;
- primary_replicates.csv:
    one row per confirmatory replicate at the valid fitted-norm budget;
- numerical_audit.json:
    fresh independent verifier report/checksum record;
- summary_metadata.json:
    provenance and accounting;
- README.md and SHA256SUMS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import scipy
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


SUMMARY_ID = "dynamical_confirmatory_summary_v1"

EXPECTED_CAMPAIGN_ID = "dynamical_original_iid_confirmatory_v1"
EXPECTED_CAMPAIGN_ROLE = "confirmatory"

EXPECTED_EXECUTION_COMMIT = (
    "18dc9f3b965e2cc634764e05785ff60d5fbd6422"
)

EXPECTED_VERIFIER_COMMIT = (
    "090eac5fa2a28879378d165c898a875f11573aac"
)

EXPECTED_CONFIG_SHA256 = (
    "1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91"
)

EXPECTED_EVIDENCE_FINGERPRINT = (
    "df9b1b7abd4a4c38b5bf8d0a5f43d7f6cb299ed23c070960b506f52641243221"
)

EXPECTED_SYSTEM = "vanderpol"
EXPECTED_SAMPLE_SIZE = 16384
EXPECTED_REPLICATES_PER_MODE = 32
EXPECTED_REPLICATES = 64

EXPECTED_MODES = (
    "matched_stochastic",
    "deterministic",
)

CI_CONFIDENCE = 0.95
QUANTILE_METHOD = "linear"

SUMMARY_FILES = (
    "README.md",
    "mode_summary.csv",
    "budget_summary.csv",
    "primary_replicates.csv",
    "numerical_audit.json",
    "summary_metadata.json",
)

PRIMARY_BUDGET_ROLE = "fitted_operator_norm"

BUDGET_ROLES = (
    "fixed_1",
    "fitted_operator_norm",
    "fixed_sqrt2",
    "twice_fitted_operator_norm",
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ValueError(
            message
        )


def file_sha256(
    path: str | Path,
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


def write_json(
    path: Path,
    value: Any,
) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(
            rows
        ),
        "Cannot write an empty summary table",
    )

    columns = list(
        rows[
            0
        ]
    )

    require(
        all(
            set(row)
            == set(columns)
            for row in rows
        ),
        "Inconsistent summary-table columns",
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def distribution(
    values: Iterable[float],
) -> dict[str, float]:
    array = np.asarray(
        list(
            values
        ),
        dtype=np.float64,
    )

    require(
        array.ndim == 1
        and array.size > 0,
        "Distribution requires nonempty one-dimensional values",
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    array
                )
            )
        ),
        "Distribution contains nonfinite values",
    )

    return {
        "mean":
            float(
                np.mean(
                    array,
                    dtype=np.float64,
                )
            ),

        "sd":
            (
                float(
                    np.std(
                        array,
                        ddof=1,
                    )
                )
                if array.size > 1
                else 0.0
            ),

        "minimum":
            float(
                np.min(
                    array
                )
            ),

        "q05":
            float(
                np.quantile(
                    array,
                    0.05,
                    method=QUANTILE_METHOD,
                )
            ),

        "q25":
            float(
                np.quantile(
                    array,
                    0.25,
                    method=QUANTILE_METHOD,
                )
            ),

        "median":
            float(
                np.quantile(
                    array,
                    0.5,
                    method=QUANTILE_METHOD,
                )
            ),

        "q75":
            float(
                np.quantile(
                    array,
                    0.75,
                    method=QUANTILE_METHOD,
                )
            ),

        "q95":
            float(
                np.quantile(
                    array,
                    0.95,
                    method=QUANTILE_METHOD,
                )
            ),

        "maximum":
            float(
                np.max(
                    array
                )
            ),
    }


def optional_distribution(
    values: Iterable[float],
) -> dict[str, float | None]:
    values = list(
        values
    )

    keys = (
        "mean",
        "sd",
        "minimum",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "maximum",
    )

    if not values:
        return {
            key:
                None
            for key in keys
        }

    return distribution(
        values
    )


def binomial_rate(
    flags: Iterable[bool],
    confidence: float = CI_CONFIDENCE,
) -> dict[str, int | float | None]:
    values = list(
        flags
    )

    require(
        all(
            type(value) is bool
            for value in values
        ),
        "Binomial trials must be Boolean",
    )

    trials = len(
        values
    )

    count = sum(
        values
    )

    if trials == 0:
        return {
            "count":
                0,

            "trials":
                0,

            "rate":
                None,

            "ci_low":
                None,

            "ci_high":
                None,
        }

    result = binomtest(
        count,
        trials,
    )

    interval = result.proportion_ci(
        confidence_level=confidence,
        method="exact",
    )

    return {
        "count":
            count,

        "trials":
            trials,

        "rate":
            count / trials,

        "ci_low":
            float(
                interval.low
            ),

        "ci_high":
            float(
                interval.high
            ),
    }


def prefixed_distribution(
    prefix: str,
    values: Iterable[float],
) -> dict[str, float]:
    return {
        f"{prefix}_{key}":
            value
        for key, value
        in distribution(
            values
        ).items()
    }


def prefixed_optional_distribution(
    prefix: str,
    values: Iterable[float],
) -> dict[str, float | None]:
    return {
        f"{prefix}_{key}":
            value
        for key, value
        in optional_distribution(
            values
        ).items()
    }


def prefixed_rate(
    prefix: str,
    flags: Iterable[bool],
) -> dict[str, int | float | None]:
    return {
        f"{prefix}_{key}":
            value
        for key, value
        in binomial_rate(
            flags
        ).items()
    }


def source_provenance() -> dict[str, Any]:
    """Require clean committed summary/verifier source and locked packages."""

    def git(
        *args: str,
    ) -> bytes:
        return subprocess.check_output(
            [
                "git",
                *args,
            ],
            cwd=ROOT,
            stderr=subprocess.PIPE,
        )

    require(
        not git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).strip(),
        "Commit the dynamical summary script and tests; "
        "use a clean source tree before generating summaries",
    )

    commit = git(
        "rev-parse",
        "HEAD",
    ).decode().strip()

    names = (
        "scripts/summarize_dynamical_confirmatory.py",
        "scripts/verify_dynamical_confirmatory_numerics.py",
        "tests/test_kahkm_dynamical_confirmatory_summary.py",
        "requirements-lock-arm64.txt",
    )

    hashes: dict[str, str] = {}

    for name in names:
        path = ROOT / name

        require(
            path.is_file()
            and not path.is_symlink(),
            f"Missing or symlinked source: {name}",
        )

        hashes[
            name
        ] = file_sha256(
            path
        )

        committed = git(
            "show",
            f"{commit}:{name}",
        )

        require(
            hashes[
                name
            ]
            == hashlib.sha256(
                committed
            ).hexdigest(),
            f"Source bytes differ from HEAD: {name}",
        )

    locked: dict[str, str] = {}

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

        name, separator, version = line.partition(
            "=="
        )

        require(
            bool(
                separator
            )
            and bool(
                version
            ),
            f"Unsupported lockfile entry: {line}",
        )

        locked[
            name
        ] = version

    versions = {
        "numpy":
            np.__version__,

        "scipy":
            scipy.__version__,
    }

    require(
        all(
            locked.get(
                name
            )
            == version
            for name, version
            in versions.items()
        ),
        "NumPy/SciPy differ from reference lockfile; use .venv-arm64",
    )

    return {
        "summary_source_commit":
            commit,

        "source_sha256":
            hashes,

        "python_version":
            sys.version,

        "package_versions":
            versions,
    }


def load_verifier():
    from scripts import (
        verify_dynamical_confirmatory_numerics
        as verifier,
    )

    return verifier


def budget_roles(
    spectral_norm: float,
) -> list[
    tuple[
        str,
        float,
    ]
]:
    values = [
        (
            "fixed_1",
            1.0,
        ),
        (
            "fitted_operator_norm",
            float(
                spectral_norm
            ),
        ),
        (
            "fixed_sqrt2",
            math.sqrt(
                2.0
            ),
        ),
        (
            "twice_fitted_operator_norm",
            2.0
            * float(
                spectral_norm
            ),
        ),
    ]

    values.sort(
        key=lambda item:
            item[
                1
            ]
    )

    require(
        [role for role, _value in values]
        == list(
            BUDGET_ROLES
        ),
        "Unexpected norm-budget ordering",
    )

    require(
        len(
            {
                np.float64(
                    value
                ).tobytes()
                for _role, value
                in values
            }
        )
        == 4,
        "Norm budgets are not distinct in binary64",
    )

    return values


def primary_budget_index(
    spectral_norm: float,
) -> int:
    roles = budget_roles(
        spectral_norm
    )

    matches = [
        index
        for index, (
            role,
            _value,
        )
        in enumerate(
            roles
        )
        if role
        == PRIMARY_BUDGET_ROLE
    ]

    require(
        matches
        == [
            1
        ],
        "Unexpected primary fitted-norm budget index",
    )

    return matches[
        0
    ]


def validate_results(
    records: list[dict[str, Any]],
    *,
    spectral_norm: float,
) -> dict[str, list[dict[str, Any]]]:
    require(
        len(
            records
        )
        == EXPECTED_REPLICATES,
        "Unexpected confirmatory result count",
    )

    by_mode: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {
        mode:
            []
        for mode in EXPECTED_MODES
    }

    seen: set[str] = set()

    expected_budgets = budget_roles(
        spectral_norm
    )

    for record in records:
        identifier = record[
            "attempt_id"
        ]

        require(
            identifier not in seen,
            f"Duplicate attempt identifier: {identifier}",
        )

        seen.add(
            identifier
        )

        require(
            record[
                "system"
            ]
            == EXPECTED_SYSTEM,
            f"Unexpected system: {identifier}",
        )

        mode = record[
            "evaluation_mode"
        ]

        require(
            mode in by_mode,
            f"Unexpected evaluation mode: {mode}",
        )

        require(
            int(
                record[
                    "sample_size"
                ]
            )
            == EXPECTED_SAMPLE_SIZE,
            f"Unexpected sample size: {identifier}",
        )

        replicate = int(
            record[
                "replicate_index"
            ]
        )

        require(
            0
            <= replicate
            < EXPECTED_REPLICATES_PER_MODE,
            f"Unexpected replicate index: {identifier}",
        )

        expected_mode_key = (
            1
            if mode
            == "matched_stochastic"
            else 2
        )

        expected_id = (
            f"s2_q{expected_mode_key}"
            f"_m{EXPECTED_SAMPLE_SIZE}"
            f"_r{replicate:03d}"
        )

        require(
            identifier
            == expected_id,
            f"Attempt identifier/design mismatch: {identifier}",
        )

        budgets = record[
            "budget_certificates"
        ]

        require(
            len(
                budgets
            )
            == 4,
            f"Unexpected budget count: {identifier}",
        )

        for saved, (
            _role,
            expected_kappa,
        ) in zip(
            budgets,
            expected_budgets,
            strict=True,
        ):
            require(
                math.isclose(
                    float(
                        saved[
                            "kappa"
                        ]
                    ),
                    expected_kappa,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                f"Budget mismatch: {identifier}",
            )

        require(
            math.isclose(
                float(
                    record[
                        "frozen_predictor_spectral_norm"
                    ]
                ),
                spectral_norm,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"Predictor norm mismatch: {identifier}",
        )

        by_mode[
            mode
        ].append(
            record
        )

    require(
        len(
            seen
        )
        == EXPECTED_REPLICATES,
        "Confirmatory attempt accounting mismatch",
    )

    for mode in EXPECTED_MODES:
        rows = sorted(
            by_mode[
                mode
            ],
            key=lambda row:
                int(
                    row[
                        "replicate_index"
                    ]
                ),
        )

        require(
            len(
                rows
            )
            == EXPECTED_REPLICATES_PER_MODE,
            f"Wrong replicate count: {mode}",
        )

        require(
            [
                int(
                    row[
                        "replicate_index"
                    ]
                )
                for row in rows
            ]
            == list(
                range(
                    EXPECTED_REPLICATES_PER_MODE
                )
            ),
            f"Missing/duplicate replicates: {mode}",
        )

        by_mode[
            mode
        ] = rows

    return by_mode


def primary_replicate_rows(
    by_mode: Mapping[
        str,
        list[
            dict[str, Any]
        ],
    ],
    *,
    spectral_norm: float,
    delta: float,
) -> list[
    dict[str, Any]
]:
    index = primary_budget_index(
        spectral_norm
    )

    rows: list[
        dict[str, Any]
    ] = []

    for mode in EXPECTED_MODES:
        for record in by_mode[
            mode
        ]:
            budget = record[
                "budget_certificates"
            ][
                index
            ]

            certificate = budget[
                "certificate"
            ]

            within = bool(
                budget[
                    "frozen_predictor_within_budget"
                ]
            )

            require(
                within,
                "Primary fitted-norm budget must contain the frozen predictor",
            )

            require(
                math.isclose(
                    float(
                        certificate[
                            "delta"
                        ]
                    ),
                    delta,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "Primary certificate delta mismatch",
            )

            margin = float(
                budget[
                    "frozen_predictor_rmse_minus_bound"
                ]
            )

            rows.append(
                {
                    "attempt_id":
                        record[
                            "attempt_id"
                        ],

                    "system":
                        EXPECTED_SYSTEM,

                    "evaluation_mode":
                        mode,

                    "sample_size":
                        EXPECTED_SAMPLE_SIZE,

                    "replicate_index":
                        int(
                            record[
                                "replicate_index"
                            ]
                        ),

                    "delta":
                        delta,

                    "budget_role":
                        PRIMARY_BUDGET_ROLE,

                    "kappa":
                        float(
                            budget[
                                "kappa"
                            ]
                        ),

                    "frozen_predictor_spectral_norm":
                        float(
                            record[
                                "frozen_predictor_spectral_norm"
                            ]
                        ),

                    "f_hat":
                        float(
                            record[
                                "statistics"
                            ][
                                "f_hat"
                            ]
                        ),

                    "s_hat":
                        float(
                            record[
                                "statistics"
                            ][
                                "s_hat"
                            ]
                        ),

                    "r_delta":
                        float(
                            certificate[
                                "r_delta"
                            ]
                        ),

                    "F_delta":
                        float(
                            certificate[
                                "F_delta"
                            ]
                        ),

                    "V_delta":
                        float(
                            certificate[
                                "V_delta"
                            ]
                        ),

                    "L_kappa_delta":
                        float(
                            certificate[
                                "L_kappa_delta"
                            ]
                        ),

                    "frozen_predictor_mse":
                        float(
                            record[
                                "frozen_predictor_mse"
                            ]
                        ),

                    "frozen_predictor_rmse":
                        float(
                            record[
                                "frozen_predictor_rmse"
                            ]
                        ),

                    "rmse_minus_L":
                        margin,

                    "predictor_within_budget":
                        within,

                    "positive_certificate":
                        bool(
                            float(
                                certificate[
                                    "L_kappa_delta"
                                ]
                            )
                            > 0.0
                        ),

                    "empirical_lower_bound_covered":
                        bool(
                            margin
                            >= 0.0
                        ),
                }
            )

    require(
        len(
            rows
        )
        == EXPECTED_REPLICATES,
        "Primary replicate-table count mismatch",
    )

    return rows


def mode_summary_rows(
    primary_rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    output: list[
        dict[str, Any]
    ] = []

    for mode in EXPECTED_MODES:
        rows = [
            row
            for row in primary_rows
            if row[
                "evaluation_mode"
            ]
            == mode
        ]

        require(
            len(
                rows
            )
            == EXPECTED_REPLICATES_PER_MODE,
            f"Primary mode-summary count mismatch: {mode}",
        )

        base: dict[str, Any] = {
            "system":
                EXPECTED_SYSTEM,

            "evaluation_mode":
                mode,

            "sample_size":
                EXPECTED_SAMPLE_SIZE,

            "replicates":
                EXPECTED_REPLICATES_PER_MODE,

            "delta":
                rows[
                    0
                ][
                    "delta"
                ],

            "budget_role":
                PRIMARY_BUDGET_ROLE,

            "kappa":
                rows[
                    0
                ][
                    "kappa"
                ],

            "frozen_predictor_spectral_norm":
                rows[
                    0
                ][
                    "frozen_predictor_spectral_norm"
                ],

            "monte_carlo_ci_confidence":
                CI_CONFIDENCE,
        }

        metrics = {
            "f_hat":
                [
                    row[
                        "f_hat"
                    ]
                    for row in rows
                ],

            "s_hat":
                [
                    row[
                        "s_hat"
                    ]
                    for row in rows
                ],

            "certificate_L":
                [
                    row[
                        "L_kappa_delta"
                    ]
                    for row in rows
                ],

            "frozen_predictor_rmse":
                [
                    row[
                        "frozen_predictor_rmse"
                    ]
                    for row in rows
                ],

            "rmse_minus_L":
                [
                    row[
                        "rmse_minus_L"
                    ]
                    for row in rows
                ],
        }

        for name, values in metrics.items():
            base.update(
                prefixed_distribution(
                    name,
                    values,
                )
            )

        base.update(
            prefixed_rate(
                "positive_certificate",
                [
                    row[
                        "positive_certificate"
                    ]
                    for row in rows
                ],
            )
        )

        base.update(
            prefixed_rate(
                "empirical_lower_bound_coverage",
                [
                    row[
                        "empirical_lower_bound_covered"
                    ]
                    for row in rows
                ],
            )
        )

        base[
            "empirical_lower_bound_violation_count"
        ] = sum(
            not row[
                "empirical_lower_bound_covered"
            ]
            for row in rows
        )

        output.append(
            base
        )

    return output


def budget_summary_rows(
    by_mode: Mapping[
        str,
        list[
            dict[str, Any]
        ],
    ],
    *,
    spectral_norm: float,
    delta: float,
) -> list[
    dict[str, Any]
]:
    role_values = budget_roles(
        spectral_norm
    )

    output: list[
        dict[str, Any]
    ] = []

    for mode in EXPECTED_MODES:
        rows = by_mode[
            mode
        ]

        for budget_index, (
            role,
            kappa,
        ) in enumerate(
            role_values
        ):
            budgets = [
                row[
                    "budget_certificates"
                ][
                    budget_index
                ]
                for row in rows
            ]

            certificates = [
                budget[
                    "certificate"
                ]
                for budget in budgets
            ]

            feasible = [
                bool(
                    budget[
                        "frozen_predictor_within_budget"
                    ]
                )
                for budget in budgets
            ]

            positive = [
                float(
                    certificate[
                        "L_kappa_delta"
                    ]
                )
                > 0.0
                for certificate in certificates
            ]

            margins = [
                float(
                    budget[
                        "frozen_predictor_rmse_minus_bound"
                    ]
                )
                for budget in budgets
            ]

            feasible_margins = [
                margin
                for flag, margin
                in zip(
                    feasible,
                    margins,
                    strict=True,
                )
                if flag
            ]

            coverage = [
                margin
                >= 0.0
                for flag, margin
                in zip(
                    feasible,
                    margins,
                    strict=True,
                )
                if flag
            ]

            base: dict[str, Any] = {
                "system":
                    EXPECTED_SYSTEM,

                "evaluation_mode":
                    mode,

                "sample_size":
                    EXPECTED_SAMPLE_SIZE,

                "replicates":
                    EXPECTED_REPLICATES_PER_MODE,

                "delta":
                    delta,

                "budget_role":
                    role,

                "primary_budget":
                    role
                    == PRIMARY_BUDGET_ROLE,

                "kappa":
                    kappa,

                "frozen_predictor_spectral_norm":
                    spectral_norm,

                "monte_carlo_ci_confidence":
                    CI_CONFIDENCE,
            }

            base.update(
                prefixed_distribution(
                    "certificate_L",
                    [
                        float(
                            certificate[
                                "L_kappa_delta"
                            ]
                        )
                        for certificate
                        in certificates
                    ],
                )
            )

            base.update(
                prefixed_rate(
                    "predictor_feasible",
                    feasible,
                )
            )

            base.update(
                prefixed_rate(
                    "positive_certificate",
                    positive,
                )
            )

            base.update(
                prefixed_optional_distribution(
                    "feasible_rmse_minus_L",
                    feasible_margins,
                )
            )

            base.update(
                prefixed_rate(
                    "empirical_lower_bound_coverage_among_feasible",
                    coverage,
                )
            )

            base[
                "empirical_lower_bound_violation_count_among_feasible"
            ] = sum(
                not flag
                for flag in coverage
            )

            output.append(
                base
            )

    require(
        len(
            output
        )
        == 8,
        "Budget-summary row count mismatch",
    )

    return output


def summarize_records(
    config: Mapping[str, Any],
    representation: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    require(
        config[
            "campaign_id"
        ]
        == EXPECTED_CAMPAIGN_ID,
        "Unexpected campaign identifier",
    )

    require(
        config[
            "campaign_role"
        ]
        == EXPECTED_CAMPAIGN_ROLE,
        "Expected confirmatory campaign evidence",
    )

    design = config[
        "confirmatory_design"
    ]

    require(
        design[
            "systems"
        ]
        == [
            EXPECTED_SYSTEM
        ],
        "Unexpected confirmatory system design",
    )

    require(
        design[
            "sample_sizes"
        ]
        == [
            EXPECTED_SAMPLE_SIZE
        ],
        "Unexpected confirmatory sample size design",
    )

    require(
        design[
            "replicates_per_cell"
        ]
        == EXPECTED_REPLICATES_PER_MODE,
        "Unexpected replicates-per-mode design",
    )

    require(
        design[
            "planned_attempts"
        ]
        == EXPECTED_REPLICATES,
        "Unexpected confirmatory attempt design",
    )

    require(
        config[
            "pair_sampling"
        ][
            "evaluation_modes"
        ]
        == list(
            EXPECTED_MODES
        ),
        "Unexpected confirmatory evaluation modes",
    )

    delta = float(
        config[
            "certificate"
        ][
            "delta"
        ]
    )

    require(
        delta
        == 0.05,
        "Unexpected confirmatory certificate delta",
    )

    require(
        representation[
            "system"
        ]
        == EXPECTED_SYSTEM,
        "Unexpected representation system",
    )

    require(
        float(
            representation[
                "omega"
            ]
        )
        == 128.0,
        "Unexpected representation omega",
    )

    spectral_norm = float(
        representation[
            "spectral_norm"
        ]
    )

    require(
        math.isfinite(
            spectral_norm
        )
        and spectral_norm > 0.0,
        "Invalid frozen predictor spectral norm",
    )

    expected_kappas = [
        value
        for _role, value
        in budget_roles(
            spectral_norm
        )
    ]

    retained_kappas = [
        float(
            value
        )
        for value in representation[
            "norm_budgets"
        ]
    ]

    require(
        len(
            retained_kappas
        )
        == 4,
        "Unexpected retained norm-budget count",
    )

    for actual, expected in zip(
        retained_kappas,
        expected_kappas,
        strict=True,
    ):
        require(
            math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "Representation norm-budget mismatch",
        )

    by_mode = validate_results(
        records,
        spectral_norm=spectral_norm,
    )

    primary = primary_replicate_rows(
        by_mode,
        spectral_norm=spectral_norm,
        delta=delta,
    )

    modes = mode_summary_rows(
        primary
    )

    budgets = budget_summary_rows(
        by_mode,
        spectral_norm=spectral_norm,
        delta=delta,
    )

    return (
        modes,
        budgets,
        primary,
    )


def summarize_evidence(
    evidence: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    evidence = Path(
        evidence
    ).resolve()

    output = Path(
        output
    ).resolve()

    require(
        evidence.is_dir()
        and not evidence.is_symlink(),
        "Evidence directory not found or unsafe",
    )

    require(
        not output.exists()
        and not output.is_symlink(),
        "Output already exists; do not overwrite it",
    )

    require(
        output != ROOT
        and ROOT not in output.parents,
        "Output must be outside the source tree",
    )

    require(
        output != evidence
        and evidence not in output.parents
        and output not in evidence.parents,
        "Output must be separate from evidence tree",
    )

    source_before = source_provenance()

    verifier = load_verifier()

    report, checksums = verifier.verify_evidence(
        evidence
    )

    require(
        report[
            "status"
        ]
        == "passed",
        "Fresh numerical audit did not pass",
    )

    require(
        report[
            "verified_attempts"
        ]
        == EXPECTED_REPLICATES,
        "Numerical audit attempt count mismatch",
    )

    require(
        report[
            "verified_retained_pairs"
        ]
        == (
            EXPECTED_REPLICATES
            * EXPECTED_SAMPLE_SIZE
        ),
        "Numerical audit retained-pair count mismatch",
    )

    require(
        report[
            "execution_source_commit"
        ]
        == EXPECTED_EXECUTION_COMMIT,
        "Numerical audit execution commit mismatch",
    )

    require(
        report[
            "config_sha256"
        ]
        == EXPECTED_CONFIG_SHA256,
        "Numerical audit configuration mismatch",
    )

    require(
        report[
            "evidence_sha256sums_fingerprint"
        ]
        == EXPECTED_EVIDENCE_FINGERPRINT,
        "Numerical audit evidence fingerprint mismatch",
    )

    config = json.loads(
        (
            evidence
            / "campaign_spec.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    representation = json.loads(
        (
            evidence
            / "representation_source.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    records = verifier.jsonl_records(
        evidence
        / "results.jsonl"
    )

    (
        mode_rows,
        budget_rows,
        primary_rows,
    ) = summarize_records(
        config,
        representation,
        records,
    )

    current_inventory = verifier.check_inventory(
        evidence
    )

    require(
        current_inventory[
            "sha256sums_fingerprint"
        ]
        == EXPECTED_EVIDENCE_FINGERPRINT,
        "Evidence changed during aggregation",
    )

    require(
        source_provenance()
        == source_before,
        "Source changed during aggregation",
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output
        / "mode_summary.csv",
        mode_rows,
    )

    write_csv(
        output
        / "budget_summary.csv",
        budget_rows,
    )

    write_csv(
        output
        / "primary_replicates.csv",
        primary_rows,
    )

    numerical_audit = {
        "report":
            report,

        "evidence_checksums":
            checksums,
    }

    write_json(
        output
        / "numerical_audit.json",
        numerical_audit,
    )

    readme = """# Confirmatory dynamical certificate summaries

These tables summarize the independently verified confirmatory Van der Pol
certificate campaign.

`mode_summary.csv` is the primary scientific summary. It reports the
finite-sample certificate and frozen-predictor distributions separately for the
matched-stochastic and deterministic evaluation laws at the valid fitted
operator-norm budget `kappa = ||B||_2`.

`budget_summary.csv` reports all four prospectively frozen norm budgets:
`1`, `||B||_2`, `sqrt(2)`, and `2||B||_2`. Predictor-specific interpretation
requires the frozen predictor to lie inside the corresponding budget. In
particular, the `kappa = 1` rows are diagnostic only when `||B||_2 > 1`.

`primary_replicates.csv` retains one row for each of the 64 independent
confirmatory replicate datasets at the fitted-norm budget. No replicate is
dropped.

The `certificate_L_*`, predictor-RMSE, `f_hat`, `s_hat`, and margin quantiles
are descriptive replicate quantiles using NumPy's linear interpolation. They
are not confidence intervals.

Rates use independent replicate datasets as trials. Their intervals are
two-sided pointwise 95% exact Clopper-Pearson intervals. They are not
simultaneous across modes, budgets, or metrics. Zero observed violations do
not establish a zero violation probability.

`positive_certificate` means `L_kappa_delta > 0`.
`empirical_lower_bound_coverage` means the retained frozen predictor satisfies
`RMSE >= L_kappa_delta` on that replicate. This empirical check is descriptive;
the certificate itself is the finite-sample lower bound derived under the
frozen i.i.d.-pair design.

`numerical_audit.json` records a fresh independent recomputation of all 64
confirmatory association datasets, hard labels, sufficient statistics,
predictor errors, certificate bounds, seed streams, and time-index draws.
No random dynamical pair dataset is regenerated, no representation is refit,
and the original campaign evidence is not modified.

The campaign execution commit, independent verifier commit, and later summary
source commit have distinct provenance roles and must not be interchanged.
"""

    with (
        output
        / "README.md"
    ).open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            readme
        )

    require(
        source_provenance()
        == source_before,
        "Source changed while saving summaries",
    )

    final_inventory = verifier.check_inventory(
        evidence
    )

    require(
        final_inventory[
            "sha256sums_fingerprint"
        ]
        == EXPECTED_EVIDENCE_FINGERPRINT,
        "Evidence changed while saving summaries",
    )

    metadata = {
        "summary_id":
            SUMMARY_ID,

        "status":
            "completed",

        "campaign_id":
            EXPECTED_CAMPAIGN_ID,

        "campaign_role":
            EXPECTED_CAMPAIGN_ROLE,

        "system":
            EXPECTED_SYSTEM,

        "evaluation_modes":
            list(
                EXPECTED_MODES
            ),

        "sample_size":
            EXPECTED_SAMPLE_SIZE,

        "replicates_per_mode":
            EXPECTED_REPLICATES_PER_MODE,

        "summarized_replicates":
            EXPECTED_REPLICATES,

        "mode_summary_rows":
            len(
                mode_rows
            ),

        "budget_summary_rows":
            len(
                budget_rows
            ),

        "primary_replicate_rows":
            len(
                primary_rows
            ),

        "primary_budget_role":
            PRIMARY_BUDGET_ROLE,

        "primary_kappa":
            float(
                representation[
                    "spectral_norm"
                ]
            ),

        "execution_source_commit":
            report[
                "execution_source_commit"
            ],

        "verifier_source_commit":
            EXPECTED_VERIFIER_COMMIT,

        "source":
            source_before,

        "config_sha256":
            report[
                "config_sha256"
            ],

        "evidence_manifest_sha256":
            report[
                "evidence_sha256sums_fingerprint"
            ],

        "evidence_directory":
            str(
                evidence
            ),

        "output_directory":
            str(
                output
            ),

        "numerical_audit_status":
            report[
                "status"
            ],

        "random_samples_generated":
            0,

        "random_pair_datasets_regenerated":
            report[
                "random_pair_datasets_regenerated"
            ],

        "representations_refit":
            report[
                "representations_refit"
            ],

        "evidence_modified":
            False,

        "quantile_method":
            QUANTILE_METHOD,

        "monte_carlo_ci_method":
            "clopper_pearson",

        "monte_carlo_ci_confidence":
            CI_CONFIDENCE,

        "command":
            [
                sys.executable,
                *sys.argv,
            ],
    }

    write_json(
        output
        / "summary_metadata.json",
        metadata,
    )

    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
    )

    require(
        {
            path.name
            for path in files
        }
        == set(
            SUMMARY_FILES
        ),
        "Unexpected pre-checksum summary file set",
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
                f"{path.name}\n"
            )

    return metadata


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
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        metadata = summarize_evidence(
            args.evidence,
            args.output,
        )

    except BaseException as exc:
        print(
            "Summary FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return 1

    print(
        json.dumps(
            metadata,
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
