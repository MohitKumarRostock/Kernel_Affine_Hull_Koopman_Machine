# KAHM input-scaling investigation

This study compares training-mean/standard-deviation preprocessing with training-fitted range scaling to `[-1, 1]` for KAHM features. Min–max scaling improves soft KAHM forecasts on several tasks, including an approximately 20% PenDigits MSE reduction across both fitting-seed sets, but worsens logistic and Van der Pol forecasts. Standardization remains the reference default, and scaling alternatives are explicit options. The retained [study summary](report_decision.json) records the complete outcomes.

## Intervention and primary outcome

| Arm | KAHM geometry input |
|---|---|
| `standard` | Original training mean/standard-deviation transformation |
| `minmax` | Primary treatment: per-coordinate training minima/maxima mapped to `[-1, 1]`; later values may extrapolate beyond that interval |
| `minmax_clip` | Same training-fitted transformation, followed by clipping to `[-1, 1]` |
| `raw` | Existing mathematically bounded histories, without standardization |

All raw observation processes already have known bounds inside `[-1, 1]`; the original history standardization changed the inputs subsequently seen by KAHM. Scaler statistics use **all training histories before geometry subsampling**, and are frozen for validation, calibration, test, and shifted Koopman histories. Training-constant coordinates map to zero in the two min–max arms. Targets, error units, output clipping, candidate families, seeds, origins, sample sizes, ridge, coefficient budgets, sharpness, and non-KAHM preprocessing are held fixed within each paired comparison. KAHM direct, hard, Frobenius, duplicate, and rollout variants share the corresponding changed KAHM geometry.

The primary family is `kahm_nested`. Each arm selects its representative using **mean validation MSE across horizons**. Original main-campaign stable tie behavior is preserved; other campaign ties use feature count and name. The primary outcome is arithmetic mean test MSE over horizons and fitting seeds, separately for each task. PenDigits averages coordinate MSE and weights eligible held-out writers equally. No test score chooses a preprocessing arm or candidate.

Relative MSE changes below are `(minmax / standard - 1) × 100`; negative values indicate improvement. Sources are [original task comparisons](analysis_original/taskwise_comparison.csv) and [confirmation task comparisons](analysis_confirmation/taskwise_comparison.csv).

| Task | Original schedule | Confirmation | Improving confirmation seeds |
|---|---:|---:|---:|
| Harmonic | −2.030% | −1.303% | 3/3 |
| Quasiperiodic | −2.316% | −6.394% | 3/3 |
| Nonlinear AR | +0.182% | −1.312% | 2/3 |
| Logistic | +2.645% | +6.968% | 0/3 |
| Van der Pol | +3.822% | +2.262% | 1/3 |
| Duffing | −7.072% | −7.294% | 3/3 |
| PenDigits | −19.787% | −19.650% | 3/3 |

Clipped min–max gives essentially the same main pattern; `raw` is a separately reported diagnostic. Improvements in the mean do not imply improvement at every horizon or for every KAHM variant. In particular, hard KAHM assignment worsens on PenDigits, and the soft model still trails stronger autoregressive/quadratic baselines there. The counts of repetitions whose upper endpoints meet the all-horizon tolerances are unchanged between `standard` and `minmax`; see [certification_comparison.csv](certification_comparison.csv). Their statistical interpretation depends on which evaluation streams are used, as described below.

## Calibration scope

The original-seed replay was designed after inspection of the earlier study
and reuses its calibration and test streams. It is a matched descriptive
comparison. Fixing a new arm before rerunning those streams does not establish
independence from previously observed outcomes, so the replayed new-arm
endpoints are nominal calculations without an additional adaptive-validity
argument.

The synthetic confirmation uses new training, validation, calibration, and
test episodes for seeds 4701--4703. Its existing confidence allocations apply
within each prespecified arm and repetition; they do not provide a joint event
for searching across four preprocessing arms. Cross-arm improvements remain
descriptive. Conditional test-episode standard errors summarize finite-episode
variation for fixed models and do not include variation across independent fits.

Both PenDigits phases reuse the same 13 eligible held-out writers. New fitting
subsamples provide no new test cohort. A population confidence interpretation
would require iid writers and a study design independent of the evaluation
writer outcomes; neither is established for this follow-up. The retained
writer-level upper endpoints are therefore nominal, and writer differences
are descriptive. These qualifications leave all numerical records unchanged.

## Exact provenance and counts

The manuscript's original source commit is:

```text
a8bcf6c8016499e1a4b64f7dd4196af43c60bd78
```

The frozen investigation adapter SHA-256 is:

```text
a4b3487c392b96ac528e09e7e3cdf7767f647909739d3d76886ee3198b4dde95
```

The paired scaling protocol was fixed before the new scaling comparisons, after the earlier manuscript and results had been inspected. It is a follow-up, not an externally preregistered original experiment. Its [protocol](protocols/protocol.json), [fresh synthetic protocol](protocols/confirmation_protocol.json), and [fresh-fitting PenDigits protocol](protocols/pendigits_confirmation_protocol.json) retain the precise designs. Per-run manifests record actual source, configuration, adapter, and protocol hashes and software versions. Exact original investigation scripts and configurations are copied unchanged into [input_scaling_replay](../../experiments/input_scaling_replay/); its [source snapshot manifest](../../experiments/input_scaling_replay/source_snapshot_manifest.json) verifies 25 files.

| Schedule component | Seeds / design | Recorded evaluations across four arms |
|---|---|---:|
| Original four main laws | 1701–1705 | 80 synthetic run folders |
| Original two oscillators | 3701–3703 | 24 synthetic run folders |
| Original budget sensitivity | 2701–2703; four laws; budgets 1, 4, 16 | 144 synthetic run folders |
| Fresh six synthetic laws | 4701–4703 | 72 synthetic run folders |
| Original PenDigits | 1701–1703; two coordinates | 24 coordinate/seed families, in four campaign folders |
| Fresh-fitting PenDigits | 4701–4703; two coordinates | 24 coordinate/seed families, in four campaign folders |

There are **320 complete synthetic score archives** and **48 PenDigits coordinate/seed families**, comprising 11,984 fitted candidates. These are paired evaluation records: arms share data streams, budget comparisons reuse streams, and PenDigits fits share a writer cohort. They are not independent datasets or independent population replications. Fresh synthetic seeds regenerate training, validation, calibration, and test episodes; fresh PenDigits seeds change fitting randomness on the **same 13 eligible held-out writers**. The analytical two-atom and duplicate-invariance illustrations do not fit KAHM input geometry and require no scaling intervention.

## Tables and machine-readable results

Both [analysis_original](analysis_original/) and [analysis_confirmation](analysis_confirmation/) contain the following files:

| File in each analysis directory | Contents |
|---|---|
| `all_candidates.csv` | Every candidate and horizon, including unsuccessful models |
| `selected_candidates.csv` | Validation-selected representatives; PenDigits coordinates remain separate |
| `selected_pooled_horizons.csv` | Selected results with PenDigits coordinate pooling |
| `horizon_summary.csv` | Per-horizon RMSE means and fitting-seed standard deviations |
| `comparison_by_horizon.csv` | Paired arm comparisons for each horizon |
| `taskwise_comparison.csv` | Primary all-horizon MSE effects, fitting-seed signs, and uncertainty summaries |
| `paired_effects.csv` | Per-fitting-seed paired loss differences |
| `pendigits_writer_differences.csv` | Descriptive effects for individual held-out writers |
| `summary.json` | Consolidated outcomes, scope notes, input hashes, and repair records |
| `audit.json` | Baseline replay and unchanged non-KAHM control checks |

Original analysis retains 24,432 candidate–horizon rows; confirmation retains 11,520. The manuscript's generated scaling tables are [primary](../../paper/tables/input_scaling_primary.tex), [PenDigits](../../paper/tables/input_scaling_pendigits.tex), [diagnostics](../../paper/tables/input_scaling_diagnostics.tex), [budget sensitivity](../../paper/tables/input_scaling_sensitivity.tex), and [uncertainty](../../paper/tables/input_scaling_uncertainty.tex). [Numeric macros](../../paper/tables/input_scaling_numbers.tex) and the [asset manifest](../../paper/tables/input_scaling_assets_manifest.json) connect prose/table values to their source analyses. Historical experiment tables are preserved.

The synthetic normal intervals in these CSVs are unadjusted Monte Carlo calculations for fixed fitted models. They omit training randomness and are separate from the finite-sample risk certificates. For the original-seed replay, reuse after earlier inspection also limits a repeated-sampling confidence interpretation; the recorded calculations are descriptive. Fitting-seed variation is reported separately. PenDigits writer effects are descriptive. Fresh-synthetic risk allocations apply within each prespecified arm and repetition, not simultaneously across four arms, all tasks, or all fitting seeds.

## Complete records and separately supplied primary scores

| Repository artifact | Contents / size |
|---|---|
| [records_original.tar.gz](records_original.tar.gz) | 1,302 text records; 7,814,592 bytes |
| [records_confirmation.tar.gz](records_confirmation.tar.gz) | 424 text records; 3,691,841 bytes |
| [episode_archive_index.json](episode_archive_index.json) | Paths, SHA-256 hashes, sizes, and repair records for all 320 valid full score NPZs |

The tar files preserve per-arm paths, every result/frozen-candidate/manifest/diagnostic file, full PenDigits writer CSVs, completion records, and textual repair evidence. Every member hash is checked. They exclude all NPZ binaries. The indexed all-candidate NPZs total **1,601,972,160 bytes** and are regenerable from the frozen designs.

A compact evidence file is supplied separately:

```text
KAHM_scaling_primary_evidence.zip
Size: 61146155 bytes
SHA-256: 84abfa225d2a0f73cfa5be1b03c32a8e4175d73fe0fe7edb1e14e55278f6bbb1
```

It contains 320 validation-selected `kahm_nested` NPZs with **960 full-precision float64 calibration-loss, pair-score, and test-loss arrays**, 1,872 selected PenDigits writer rows, and selection/provenance indices. All 1,104 selected horizon values and candidate identities were checked against the canonical CSVs. It is **primary-score evidence**, not the complete 1.6 GB all-candidate archive. Scientific arrays under `scores/` pass exact round-trip checks.

Four truncated full score archives were detected before analysis and regenerated with identical frozen source, configurations, and seeds. All original non-timing result and frozen-candidate values matched before replacement. The repository retains hashes and `artifact_repair.json` records in the indexed/tar evidence. The separately supplied ZIP additionally preserves all **13,399,125 original corrupt bytes** under `forensics/`, with exact repair records and explanatory README/index. Those four inner NPZ structures are deliberately incomplete and are not analysis inputs; the outer ZIP passes CRC validation.

## Verification and reproduction

[Verification records](verification/) cover seven isolated preprocessing tests, exact original-standard replay, unchanged non-KAHM outputs, source/configuration guards, scaler fitting before geometry subsampling, and complete archive integrity. The [production validation](verification/portable_runner_validation.json) records **33 passing tests** and 20 reduced route/storage checks; the reduced checks are not accuracy evidence. The [full production replay](verification/production_replay.json) repeats complete `harmonic_1701` standard and min–max fits: **240 arrays and 1,966,080 float64 values agree exactly** with the investigation. Only timing and explicitly added scaler metadata are excluded from the stated JSON comparison. [Packaging validation](verification/evidence_packaging_validation.json) and [manuscript QA](verification/manuscript_qa.json) document the delivered artifacts.

From `extensions/certified_forecasting`, using the pinned [reference environment](../../requirements-reference.txt):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/prepare_pendigits.py --output-dir data_cache/pendigits
python experiments/run_input_scaling.py \
  --output /tmp/kahm-scaling-replay --workers 4 \
  --prepared data_cache/pendigits/pendigits_original.npz
```

Use a new output directory. This current production runner executes both phases and all four arms, records its actual sources and preprocessing metadata, and writes complete score archives atomically. Standardization remains the default in the underlying API; explicit opt-in uses `input_scaling="minmax"`, configuration `"kahm_input_scaling": "minmax"`, or the example CLI's `--kahm-input-scaling minmax`.

To summarize **new production-run records**, the archived analyzer can skip historical schema equivalence while retaining selection and non-KAHM checks:

```bash
python experiments/input_scaling_replay/analyze_comparison.py \
  --results-root /tmp/kahm-scaling-replay/results --config-root configs \
  --skip-original-replay --output /tmp/kahm-scaling-replay/analysis
python experiments/input_scaling_replay/analyze_comparison.py \
  --results-root /tmp/kahm-scaling-replay/confirmation_results \
  --config-root results/input_scaling_study/protocols/confirmation_configs \
  --campaigns main_v1 oscillators_v1 pendigits_v1 --skip-original-replay \
  --output /tmp/kahm-scaling-replay/confirmation_analysis
```

Here `--skip-original-replay` explicitly makes **no original-baseline equivalence claim**. It does not make reused original seeds fresh. The frozen analyzer's legacy status label contains `fresh_confirmation_seeds` whenever this switch is used; for an original-seed production rerun it means only that historical comparison was skipped. Production records add scaler metadata that the historical auditor intentionally does not ignore. The checked full example instead uses [verify_scaling_production_replay.py](../../scripts/verify_scaling_production_replay.py), whose documented exclusions are narrow and whose numerical comparisons are exact.

For **exact historical source/schema reproduction**, start from the repository root and create the sibling directory names expected by the unchanged scripts:

```bash
scaling_source_repo="$PWD"
scaling_replay_parent="$(mktemp -d)"
git worktree add --detach "$scaling_replay_parent/kahm-study" \
  a8bcf6c8016499e1a4b64f7dd4196af43c60bd78
mkdir "$scaling_replay_parent/scaling-investigation"
cp -R "$scaling_source_repo/extensions/certified_forecasting/experiments/input_scaling_replay/." \
  "$scaling_replay_parent/scaling-investigation/"
cd "$scaling_replay_parent/scaling-investigation"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ../kahm-study/extensions/certified_forecasting/experiments/prepare_pendigits.py \
  --output-dir data_cache/pendigits
python run_scaling.py --workers 4
python run_confirmation.py
for mode in standard minmax minmax_clip raw; do
  python run_pendigits_scaling.py --mode "$mode"
  python run_pendigits_confirmation.py --mode "$mode"
done
python analyze_comparison.py --output analysis
python analyze_comparison.py --results-root confirmation_results \
  --config-root confirmation_configs --campaigns main_v1 oscillators_v1 pendigits_v1 \
  --skip-original-replay --output confirmation_analysis
```

Reference numerical-library versions are recorded; bitwise equality on arbitrary architectures is not promised. The archived scripts refuse overwrites and preserve failures; their unchanged bytes provide the source-hash reference for historical reproduction. Dataset preparation verifies the pinned UCI archive; downloaded data are not stored in this evidence directory.

To regenerate manuscript tables or the standalone report from the committed analyses, run from the extension directory:

```bash
python scripts/make_scaling_assets.py
python scripts/build_scaling_report.py --output /tmp/kahm-scaling-report
```

Report rendering additionally requires ReportLab and DejaVu fonts. [package_scaling_evidence.py](../../scripts/package_scaling_evidence.py) provides portable arguments for rebuilding the tar records, full-archive index, and compact primary-score ZIP from a completed investigation directory. Its scope and byte-level checks are recorded in the packaging validation file.
