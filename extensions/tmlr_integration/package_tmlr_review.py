"""Build an anonymous saved-forecast supplement for the TMLR manuscript.

The package contains the original numeric forecast archives, compact copies of
training targets and variance scales, reported analysis tables, and a standalone
statistical verifier. Source experimental records are read without modification.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from verify_saved_forecasts import (
    COMPARATORS, HORIZONS, LAWS, METHODS, NONLINEAR_LAWS, NOISE_LEVELS,
    SEEDS, TABLE_KEYS, VARIANCE_FLOOR, VerificationError, array_sha256,
    read_json, require, sha256, verify,
)

ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_LIMIT_BYTES = 100_000_000
ARCHIVE_NAME = "KAHM_TMLR_Saved_Forecasts.zip"

README = r'''# Saved-forecast reproduction for KAHM reconstruction departures

This supplement reproduces the numerical forecasting comparison in the
manuscript's reconstruction-departure experiment. It contains the original
forecast arrays for every method and task, the training targets that determine
the loss scales, the reported numerical analysis, and an independently
implemented verification program.

## Scope

The experiment contains 192 task suites: four simulation laws, two added-noise
levels, and 24 independent repetitions. Every suite contains 1,664 evaluation
origins and the three forecast offsets 1, 4, and 12. Ten forecasting procedures
produce 1,920 arrays containing 9,584,640 scalar forecasts. All specified tasks
and methods are included, regardless of their forecasting accuracy.

The verifier recomputes normalized forecast errors and the six primary
statistical comparisons from saved arrays. The supplied material supports
verification of these numerical results. It does not generate trajectories,
fit forecasting models, repeat hyperparameter or correction selection,
reproduce feature-bound diagnostics, or measure computation time. The complete
fitting implementation and experiment records are distributed separately from
this anonymous review archive. The manuscript gives the mathematical methods,
selection rules, and scope of each experiment.

## Reproduction

Use Python 3.11 or later with the dependencies in `requirements.txt`. From this
directory, run:

```bash
python -m pip install -r requirements.txt
python verify_saved_forecasts.py
```

To write the recomputed CSV tables and a machine-readable verification report
to a new directory outside this archive, run:

```bash
python verify_saved_forecasts.py --write-results ../recomputed_forecasts
```

The program first checks the SHA256 digest and size of every packaged file.
It then checks the full task and method roster, array dimensions, finite values,
source-array digests, training variance scales, and reported analysis tables.
All NumPy loads use `allow_pickle=False`. No model object is deserialized and
no network access is required by the verifier.

Successful verification reports two established primary criteria: improvement
by at least 3% over the common geometric basis on the nonlinear aggregate, and
noninferiority within 5% of ridge on the linear control. The other four useful
improvement criteria are not established. Their point estimates and lower
bounds are reported in `reported/primary_contrasts.csv`. Failure to establish
a margin does not demonstrate equivalence. The statistical
unit is an independent repetition, rather than an individual forecast window.

## Files

- `forecasts/`: 192 original NPZ archives, copied without changing their bytes.
  Each contains `Y`, the evaluation targets, and one array per method. Every
  array has shape `(1664, 3)` with columns ordered by offsets `(1, 4, 12)`.
- `normalization/`: one compact NPZ archive per task, containing `A_Y` with
  shape `(512, 3)` and the three stored `loss_scale` values. These arrays are
  extracted without numerical changes from the prepared experiment inputs.
- `tasks.json`: the task roster, file mapping, selected non-KAHM method,
  original input-archive and prediction-archive hashes, and source-array hashes.
- `study.json`: the experimental roster and statistical constants.
- `reported/`: eight original CSV analysis tables and a concise statistical
  result. The verifier compares every column of each CSV with recomputation.
- `SOURCE_PROVENANCE.json`: hashes identifying the original prepared-input
  manifest, statistical decision record, and analysis tables.
- `MANIFEST.json`: SHA256 digests and sizes of all other packaged files.
- `verify_saved_forecasts.py`: the standalone verifier.

The normalization archives contain only two arrays from each larger prepared
input archive. Their file hashes therefore differ from those of the original
input archives. Their array hashes use the original convention: SHA256 of a
sorted JSON description of the NumPy dtype and shape, followed by a zero byte
and the contiguous array values in C order. The original input archive hash is
recorded separately. The target and forecast array hashes connect the compact
supplement to the original numerical records.

## Forecasting procedures

| Array identifier | Procedure |
| --- | --- |
| `ridge` | Raw-history ridge baseline |
| `raw_rbf` | Raw-history Nyström RBF forecast with calibrated ridge blending |
| `raw_hgb` | Histogram gradient boosting with calibrated ridge blending |
| `persistence` | Persistence forecast with calibrated ridge blending |
| `common` | Shared gates and gated principal coordinates |
| `kahm_departure` | Common basis and bounded KAHM reconstruction departures |
| `rbf_departure` | Common basis and bounded positive-RBF reconstruction departures |
| `kahm_unbounded` | Common basis and unbounded KAHM reconstruction departures |
| `rbf_unbounded` | Common basis and unbounded positive-RBF reconstruction departures |
| `nonkahm_policy` | Complete non-KAHM procedure selected on split C |

The stored policy forecast is checked against its declared selected method's
D forecast. This checks the prediction record; it does not repeat model
selection on split C.

## Losses and statistical comparisons

For task `t` and forecast offset `h`, the normalization is

```text
scale[t,h] = max(var(A_Y[:,h], ddof=0), 1e-12)
NMSE[t,m,h] = mean((prediction[m][:,h] - Y[:,h])**2) / scale[t,h]
```

Task loss averages the three offset-specific normalized errors. Within each
repetition, the nonlinear aggregate weights the three nonlinear laws and two
noise levels equally; the linear aggregate weights its two noise levels
equally. Thus each primary contrast has 24 paired repetition values.

For each nonlinear comparator `m`, the useful-improvement gap is
`d[r] = 0.97 * loss[r,m] - loss[r,KAHM]`. The comparators are the common
basis, bounded RBF departure, raw RBF, boosting, and the C-selected non-KAHM
policy. The linear noninferiority gap is
`d[r] = 1.05 * loss[r,ridge] - loss[r,KAHM]`. Each simultaneous one-sided
lower bound is

```text
mean(d) - t.ppf(1 - 0.05/6, df=23) * std(d, ddof=1) / sqrt(24)
```

A criterion is established when its lower bound is strictly positive. The
six bounds use a Bonferroni family at level 0.05 and the Student approximation
for independent repetition differences. The relative point gain is
`1 - mean(loss[KAHM]) / mean(loss[comparator])`; it is distinct from the
margin-adjusted gap and its lower bound. The 3% and 5% margins were fixed
research criteria, rather than application-derived utility requirements.

File and source-array hashes are checked exactly. Recomputed floating-point
quantities are compared with relative tolerance `1e-10` and absolute tolerance
`1e-12`, allowing minor differences in numerical-library arithmetic. These
checks establish consistency of the supplied forecasts and statistics; they
do not constitute independent experimental replication.
'''


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def zip_info(name: str) -> zipfile.ZipInfo:
    entry = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    return entry


def write_numeric_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Store numeric arrays with reproducible ZIP member metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in sorted(arrays.items()):
            require(array.dtype.kind in "fiu", f"Nonnumeric normalization array: {name}")
            content = io.BytesIO()
            np.save(content, array, allow_pickle=False)
            archive.writestr(zip_info(name + ".npy"), content.getvalue(), compresslevel=9)


def make_bundle(study: Path, output: Path) -> dict[str, Any]:
    study, output = study.resolve(), output.resolve()
    require(not output.exists(), "The output directory must be new")
    require((study / "prepared_inputs" / "manifest.json").is_file(), "Prepared input manifest is unavailable")
    require((study / "analysis" / "primary_contrasts.csv").is_file(), "Reported analysis is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        bundle = staging / "saved_forecast_supplement"
        for directory in (bundle, bundle / "forecasts", bundle / "normalization", bundle / "reported"):
            directory.mkdir(exist_ok=True)
        original_manifest = read_json(study / "prepared_inputs" / "manifest.json")
        original_decision = read_json(study / "analysis" / "decision.json")
        require(original_decision.get("technical_complete") is True, "The original numerical evaluation is incomplete")
        records = []
        for task in sorted(original_manifest["tasks"], key=lambda item: (item["seed"], LAWS.index(item["law"]), item["noise_level"])):
            identifier = task["task_id"]
            require(Path(identifier).name == identifier and identifier not in (".", ".."), "Invalid source task identifier")
            original_input = (study / task["input"]).resolve()
            require(original_input.is_relative_to(study), "Prepared source path escapes its study directory")
            original_predictions = study / "results" / identifier / "predictions.npz"
            original_summary = read_json(study / "results" / identifier / "summary.json")
            require(original_summary.get("status") == "completed", f"Incomplete source task: {identifier}")
            require(sha256(original_input) == task["sha256"] == original_summary["input_sha256"], f"Prepared input identity differs: {identifier}")
            prediction_digest = sha256(original_predictions)
            require(prediction_digest == original_summary["files"]["predictions.npz"], f"Prediction identity differs: {identifier}")
            with np.load(original_input, allow_pickle=False) as arrays:
                selected_arrays = {name: arrays[name].copy() for name in ("A_Y", "loss_scale")}
                for name, array in selected_arrays.items():
                    require(array_sha256(array) == task["array_sha256"][name], f"Prepared array identity differs: {identifier}/{name}")
                require(array_sha256(arrays["D_Y"]) == task["array_sha256"]["D_Y"], f"Prepared evaluation target identity differs: {identifier}")
            forecast_file = f"forecasts/{identifier}.npz"
            normalization_file = f"normalization/{identifier}.npz"
            shutil.copyfile(original_predictions, bundle / forecast_file)
            write_numeric_npz(bundle / normalization_file, selected_arrays)
            records.append({
                "task_id": identifier, "law": task["law"], "noise_level": task["noise_level"],
                "seed": task["seed"], "repetition": task["repetition"],
                "forecast_file": forecast_file, "normalization_file": normalization_file,
                "selected_policy_method": original_summary["fit_diagnostics"]["policy"]["selected_method"],
                "source_input_archive_sha256": task["sha256"],
                "source_prediction_file_sha256": prediction_digest,
                "source_arrays_sha256": {name: task["array_sha256"][name] for name in ("A_Y", "loss_scale", "D_Y")},
                "source_prediction_arrays_sha256": original_summary["prediction_array_sha256"],
                "normalization_extraction": {"source": "prepared task input archive", "members": ["A_Y", "loss_scale"]},
            })
        write_json(bundle / "tasks.json", records)
        write_json(bundle / "study.json", {
            "schema": "kahm-saved-forecast-study-v1", "laws": list(LAWS),
            "nonlinear_laws": list(NONLINEAR_LAWS), "seeds": list(SEEDS),
            "noise_levels": list(NOISE_LEVELS), "horizons": list(HORIZONS),
            "methods": list(METHODS), "training_target_rows": 512,
            "evaluation_origins": 1664, "variance_floor": VARIANCE_FLOOR,
            "variance_ddof": 0, "primary_candidate": "kahm_departure",
            "primary_comparators": list(COMPARATORS), "family_alpha": 0.05,
            "family_size": 6, "superiority_margin": 0.03,
            "linear_noninferiority_margin": 0.05,
        })
        source_tables = []
        for name in TABLE_KEYS:
            source = study / "analysis" / f"{name}.csv"
            relative = f"reported/{name}.csv"
            shutil.copyfile(source, bundle / relative)
            source_tables.append({"path": relative, "source_sha256": sha256(source)})
        established = [row["contrast_id"] for row in original_decision["primary_contrasts"] if row["criterion_established"]]
        write_json(bundle / "reported" / "statistical_result.json", {
            "task_suites": original_decision["completed_valid_tasks"],
            "independent_repetitions": original_decision["independent_repetitions"],
            "established_contrasts": established,
            "all_primary_criteria_established": original_decision["statistical_gate_pass"],
            "interpretation": "Two of the six prespecified accuracy criteria are established. The complete set of useful forecasting improvement criteria is not supported by this experiment.",
        })
        write_json(bundle / "SOURCE_PROVENANCE.json", {
            "schema": "kahm-saved-forecast-provenance-v1",
            "prepared_input_manifest_sha256": sha256(study / "prepared_inputs" / "manifest.json"),
            "statistical_decision_record_sha256": sha256(study / "analysis" / "decision.json"),
            "reported_tables": source_tables,
            "forecast_archives": "Copied byte for byte; original file hashes appear in tasks.json.",
            "normalization_archives": "Numeric A_Y and loss_scale members extracted without numerical changes; original archive and array hashes appear in tasks.json.",
        })
        (bundle / "README.md").write_text(README, encoding="utf-8")
        (bundle / "requirements.txt").write_text("numpy==2.3.5\nscipy==1.17.0\n", encoding="utf-8")
        verifier_source = Path(__file__).resolve().with_name("verify_saved_forecasts.py")
        shutil.copyfile(verifier_source, bundle / verifier_source.name)
        files = [{"path": path.relative_to(bundle).as_posix(), "bytes": path.stat().st_size,
                  "sha256": sha256(path)} for path in sorted(bundle.rglob("*")) if path.is_file()]
        write_json(bundle / "MANIFEST.json", {"schema": "kahm-saved-forecast-manifest-v1", "files": files})
        result = verify(bundle)
        archive_path = staging / ARCHIVE_NAME
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    relative = "saved_forecast_supplement/" + path.relative_to(bundle).as_posix()
                    archive.writestr(zip_info(relative), path.read_bytes(), compresslevel=9)
        archive_bytes = archive_path.stat().st_size
        require(archive_bytes < ZIP_LIMIT_BYTES, f"Review archive exceeds 100 MB: {archive_bytes} bytes")
        summary = {
            "archive": ARCHIVE_NAME, "archive_bytes": archive_bytes,
            "archive_sha256": sha256(archive_path), "maximum_archive_bytes": ZIP_LIMIT_BYTES,
            "verification": result,
            "scope": "Anonymous saved-forecast reproduction of the reconstruction-departure experiment.",
        }
        write_json(staging / "PACKAGE_SUMMARY.json", summary)
        staging.replace(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reconstruction_mechanism")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New destination containing the directory, deterministic ZIP, and package summary")
    arguments = parser.parse_args()
    try:
        summary = make_bundle(arguments.study_dir, arguments.output_dir)
    except (VerificationError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    print(json.dumps({key: summary[key] for key in ("archive", "archive_bytes", "archive_sha256", "scope")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
