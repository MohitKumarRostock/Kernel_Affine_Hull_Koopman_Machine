# Numerical checks and reporting

This directory contains statistical checks, kernel-identity diagnostics, reporting programs, and source-identity records for the supported forecasting studies. Run commands from the repository root and use explicit output paths.

## Primary comparisons from saved losses

With SciPy installed, recompute the four primary comparisons for the local-position forecasting study:

```bash
python reproduction/audits/local_position_forecasting/verify_primary.py \
  --evidence extensions/soft_state_ablation_20261102/analysis \
  --output /tmp/kahm_local_position_checks.json
```

This checks paired statistical summaries conditional on the supplied repetition losses. Simulation, model fitting, and forecast production have separate requirements described in the study's [reproduction guide](../extensions/soft_state_ablation_20261102/reporting/README.md).

## Kernel and reconstruction identities

`audits/reconstruction_mechanism/audit_method_theory.py` checks kernel identities, feature spans, radial transformations, and ridge calculations on small numerical examples. To keep its output separate from the recorded evidence:

```bash
mkdir -p /tmp/kahm_geometry_checks
cp reproduction/audits/reconstruction_mechanism/audit_method_theory.py /tmp/kahm_geometry_checks/
python /tmp/kahm_geometry_checks/audit_method_theory.py
```

The program requires NumPy. Its checks are floating-point calculations, not formal proofs or verified interval arithmetic. The `recorded/` directory contains the supplied numerical audit result.

## Reporting from study results

The `reporting/` programs generate statistics, tables, and figures from saved results without fitting models. Their recorded outputs retain their source manifests. Use explicit source paths because default locations depend on the program location.

For local-position reporting:

```bash
python reproduction/reporting/soft_state_ablation/build_soft_state_assets.py \
  --study-root extensions/soft_state_ablation_20261102 \
  --output /tmp/kahm_local_position_reporting
```

The output directory must not already exist or overlap study inputs. This program requires NumPy, SciPy, and Matplotlib. It checks the supplied analysis and audit records and recomputes the four paired bounds.

The dimension program accepts `--study-root` and `--output`. The follow-up program accepts `--preprocessing-root`, `--crossfit-root`, and `--output-root`; use `extensions/crossfit_residual_reproduction_20261101` for the residual study. The policy program accepts `--evidence-root` and `--output-root`; its evidence root must contain `REMOTE_BLOBS.json` alongside the study files at the specified Git identities. The source records under `source_provenance/policy_augmentation/` identify these inputs and describe the comparisons and their limits.

A matching checksum establishes file identity. Statistical conclusions also depend on the stated sampling units and confidence procedures. Recomputing a summary from saved losses does not independently verify data acquisition, feature construction, or model training.
