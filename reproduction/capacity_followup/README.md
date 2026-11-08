# Decoder-capacity follow-up for geometric forecasting features

This experiment asks whether a forecasting advantage of Kernel Affine Hull
Machine (KAHM) local-coordinate features persists when the predictor has a larger Gaussian-kernel feature
budget. It compares those features with raw histories and matched affine
local-coordinate features, using **64 or 256 training landmarks** in a finite
Nyström approximation. The predictor also includes a linear component.

The capacity study uses **24 independent repetitions of three synthetic laws**:
nonlinear autoregression, Van der Pol, and Duffing. Each repetition generates
separate training, validation, and test episodes. Competing procedures within
a repetition share the observations and fitted geometric reference models;
each procedure selects its predictor settings by validation loss. The
**12 primary comparisons** are three laws × two landmark budgets × two
references, each compared with KAHM local-coordinate features.

This is the supplementary capacity study. It is separate from the enrichment
paper's main 64-repetition, four-law, 40-comparison experiment. Its loss is
**MSE**, averaged over origins and forecast horizons within an episode and then
over test episodes. It is not the variance-standardized SMSE of the main
experiment. These studies should not be pooled into one confidence family.

## Read and interpret the evidence

| Question | Local source |
| --- | --- |
| What was specified for the capacity comparison? | [`study/FOLLOWUP_PROTOCOL.txt`](study/FOLLOWUP_PROTOCOL.txt), [`study/CAPACITY_ANALYSIS_SPEC.txt`](study/CAPACITY_ANALYSIS_SPEC.txt), [`study/followup_config.json`](study/followup_config.json) |
| How are observations, geometric features, and predictors constructed? | [`study/datasets.py`](study/datasets.py), [`study/geometry.py`](study/geometry.py), [`study/heads.py`](study/heads.py), [`study/followup_capacity.py`](study/followup_capacity.py) |
| What are all 12 primary outcomes? | [`archive/capacity/capacity_primary_contrasts.csv`](archive/capacity/capacity_primary_contrasts.csv) |
| Which repetition records support those outcomes? | [`archive/capacity/capacity_records.json`](archive/capacity/capacity_records.json), [`archive/capacity/capacity_paired_differences.csv`](archive/capacity/capacity_paired_differences.csv) |
| How are intervals and decisions computed? | [`study/audit_capacity.py`](study/audit_capacity.py), [`archive/capacity/capacity_independent_analysis.json`](archive/capacity/capacity_independent_analysis.json) |
| Where do the practical margins come from? | [`study/capacity_margins.json`](study/capacity_margins.json) and the preserved first-stage records in [`archive/stage1/`](archive/stage1/) |
| What is checked for numerical sensitivity? | [`study/numerical_sensitivity.py`](study/numerical_sensitivity.py), [`archive/numerical_sensitivity/`](archive/numerical_sensitivity/) |

The primary CSV reports **reference MSE minus KAHM MSE**, so positive differences
favor KAHM. Its intervals use the 24 complete repetition differences, with a
paired Student approximation and Bonferroni adjustment across 12 comparisons.
Origins, horizons, and episodes do not increase the number of repetitions.
Each law's practical margin is 5% of its first-stage raw-input predictor's mean
MSE, fixed before the capacity data were generated.

The recorded outcomes include four practical benefits on Van der Pol, six
comparisons with higher KAHM loss, and two unresolved directions. The original
CSV calls these `practical_benefit`, `harm`, and `direction_unresolved`.
An unresolved direction does not establish equivalence. A benefit at 256
landmarks applies to that tested finite predictor; it does not establish an
advantage over every raw-input predictor. Changing the landmark budget can
also change the sampled landmarks and the training-derived bandwidth.

The 648 capacity records contain eight feature-by-budget procedures and one
validation-selected procedure for each of 72 law–repetition pairs. Full KAHM
score features and the selector are secondary results. Supporting first-stage
and sensitivity records remain because the original protocols, practical
margins, and numerical checks depend on them.

## Verify files and recompute saved-record statistics

Run commands from this directory:

```bash
cd reproduction/capacity_followup
python -m reproduce verify
```

`verify` needs only the Python standard library. It checks all **51 scientific
snapshot entries and 17 archive entries**, including decompressed hashes for
gzip files. A missing or changed file is an error. The two original manifests
are [`study/SNAPSHOT.json`](study/SNAPSHOT.json) and
[`archive/MANIFEST.json`](archive/MANIFEST.json).

For the statistical audit, use Python 3.12 and the pinned dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m reproduce audit --output runs/archive-audit
```

This command copies the unchanged scientific files to its output directory,
checks the protocol hashes and margin provenance, and recomputes the 29
first-stage and 12 capacity contrasts from saved repetition records. It does
not generate observations or fit predictors. Results are written under
`runs/archive-audit/stage1/` and `runs/archive-audit/capacity/`, with the audit
summary in `runs/archive-audit/archive_audit.json`.

The capacity audit also reports two historical differences between the
margin-setting metadata and the final capacity specification and analyzer.
The final versions are pinned by `study/followup_freeze.json`; both are
preserved unchanged here. These reported differences do not indicate changed
files in this bundle. The first-stage record bytes that determined the margins
are checked separately.

The compact archive supports repetition-level statistical recomputation.
It does not contain all fitted models or prediction arrays. Reconstructing
episode losses directly from predictions requires a full computational rerun.

## Regenerate the original experiments

The full command regenerates the first-stage experiments, capacity follow-up,
and numerical-sensitivity checks from their published random streams. It fits
models and can take substantially longer than the saved-record audit:

```bash
.venv/bin/python -m reproduce full --workers 4 --run-dir runs/full
```

Use the same command with `--resume` to continue a compatible incomplete run.
The command records source hashes and the execution environment, limits BLAS
to one thread per worker, checks saved predictions and losses, and compares
regenerated MSEs with the historical archive at `rtol=1e-6`, `atol=1e-10`.
Allow approximately 1 GB for generated files. Reusing the published random
streams checks computational reproduction; it supplies no new independent
repetitions. Runtime and memory measurements depend on the execution environment.

For a smaller fitting check, `python -m reproduce smoke --run-dir runs/smoke`
runs reduced examples and the archived arithmetic audit. Its reduced examples
do not reproduce the reported means. `python -m reproduce --help` lists the
available commands. No command in this bundle builds a manuscript.

## Provenance and licenses

The experiment files come from `extensions/predictive_geometry/` at commit
[`a67c40190458bab7b3e56b4fac76770798d074b3`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/a67c40190458bab7b3e56b4fac76770798d074b3/extensions/predictive_geometry).
All scientific snapshot files, archive entries, and their original manifests
are preserved unchanged. [`PROVENANCE.json`](PROVENANCE.json) identifies the
source commit and the experiment-only command adaptation. Historical protocol
files are evidence of the original design; executing them again does not
constitute a new preregistration.

Code licensing is retained in [`LICENSE`](LICENSE) and
[`study/vendor/upstream/LICENSE`](study/vendor/upstream/LICENSE).
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records upstream and Airfoil
data attribution. [`study/CITATIONS.bib`](study/CITATIONS.bib) is scientific
bibliography metadata.
