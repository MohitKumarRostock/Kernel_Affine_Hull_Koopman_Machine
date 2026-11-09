# Numerical checks and scientific records

This directory contains numerical audits, statistical summaries, reporting programs, and source-identity records for the forecasting studies. Run commands from the repository root and choose explicit output paths. Each program checks a specified part of an experiment; its result applies to that scope.

## Primary comparisons from stored losses

With SciPy installed, recompute the four primary comparisons for the local-position forecasting study:

```bash
python3 reproduction/audits/local_position_forecasting/verify_primary.py \
  --evidence extensions/soft_state_ablation_20261102/analysis \
  --output /tmp/kahm_local_position_checks.json
```

This uses the retained repetition losses and reported comparison tables. It checks paired statistical summaries conditional on those losses. The source-data generation, model fitting, and forecast production have their own study-specific requirements.

## Reconstruction and forecast audits

The programs in `audits/reconstruction_mechanism/` have the following inputs and outputs:

| Program | Input and calculation | Output |
| --- | --- | --- |
| `audit_method_theory.py` | Small synthetic examples checking kernel identities, feature spans, radial transformations, and ridge calculations | Writes `method_theory_checks.json` beside the script |
| `independent_compact_forecast_checks.py` | A compact evidence directory containing its manifest, task records, targets, and stored forecast arrays; recomputes six primary comparisons | Path supplied with `--output` |
| `independent_saved_forecast_checks.py` | The pinned reconstruction study's numerical inputs, forecasts, selection records, and statistical tables; checks loss, aggregation, and recorded resource arithmetic | Path supplied with `--output` |
| `retrieve_pinned_empirical_evidence.py` | A pinned Git tree; downloads selected immutable source and evidence files, checking their identities | Directory supplied with `--destination` |
| `restore_numeric_evidence.py` | The complete pinned transport parts and `PACKED_EVIDENCE.json`; verifies checksums and extracts selected numerical inputs | Evidence directory and JSON path supplied explicitly |

The compact and expanded forecast audits require NumPy and SciPy. The geometry diagnostics use NumPy and provide floating-point checks, not formal proofs or verified interval arithmetic. To keep their generated result separate, copy that script to an output directory before running it:

```bash
mkdir -p /tmp/kahm_geometry_checks
cp reproduction/audits/reconstruction_mechanism/audit_method_theory.py /tmp/kahm_geometry_checks/
python3 /tmp/kahm_geometry_checks/audit_method_theory.py
```

For the forecast audits, replace the evidence paths with directories containing the required numerical records:

```bash
python3 reproduction/audits/reconstruction_mechanism/independent_compact_forecast_checks.py \
  --evidence-dir /path/to/compact_evidence \
  --output /tmp/kahm_compact_checks.json

python3 reproduction/audits/reconstruction_mechanism/independent_saved_forecast_checks.py \
  --study-dir /path/to/pinned_evidence/extensions/reconstruction_mechanism \
  --output /tmp/kahm_saved_forecast_checks.json
```

Pinned retrieval accepts `--tree extensions/dimension_forecasting/vendor_original/repository_tree.json` and an explicit `--destination`. Its `--parts` option also downloads the archived numerical transport. Extraction then uses `--evidence-root` pointing to that destination and `--output` for its JSON inventory. These commands require access to the pinned files; their source code alone is insufficient to run the forecast checks.

The `recorded/` subdirectory contains the supplied audit results. Their environments, checked quantities, numerical tolerances, and limitations are part of each record. Saved runtime measurements remain measurements of the recorded execution.

## Reporting and derived statistics

The `reporting/` programs construct derived statistics and table or figure files from saved study results. They perform no model fitting. Use explicit paths because their default locations depend on the script location.

For the completed ablation's reporting calculations:

```bash
python3 reproduction/reporting/soft_state_ablation/build_soft_state_assets.py \
  --study-root extensions/soft_state_ablation_20261102 \
  --output /tmp/kahm_soft_state_reporting
```

That output directory must not already exist or overlap the study inputs. The program requires NumPy, SciPy, and Matplotlib. It verifies the bound analysis and audit records and recomputes the four paired bounds from repetition summaries.

The dimension program accepts `--study-root` and `--output`. The follow-up program accepts `--preprocessing-root`, `--crossfit-root`, and `--output-root`. The policy program accepts `--evidence-root` and `--output-root`; its evidence root must contain `REMOTE_BLOBS.json` alongside the corresponding `extensions/...` files at the exact recorded Git identities. Use a pinned evidence bundle for this check. The `recorded/` directories retain derived numerical values and their input manifests.

## Source identities and interpretation

[The scientific input snapshot](source_provenance/integrated_studies/scientific_input_snapshot.json) records 144 scientific SHA-256 identities at its specified source commit. [The policy evidence scope](source_provenance/policy_augmentation/EVIDENCE_SCOPE.md) explains that study's comparisons, aggregation rules, results, and limitations. Other records under `source_provenance/` identify numerical archives, immutable source files, and their checksums.

A matching checksum establishes file identity. Statistical conclusions still depend on their declared sampling units and confidence procedures. Checks of stored forecast errors do not establish that every upstream data-acquisition, feature-generation, or training step has been independently reproduced.
