# Kernel Affine Hull Koopman Machine

## Reconstruction-based representations for forecasting

A reconstruction-based representation maps an observed time-series history to normalized similarity weights for a set of reference regions. Each weight measures how well a local model reconstructs the history. Normalization makes the weights easy to compare, but it can remove information needed to predict future observations. When a fitted forecaster performs poorly, its error alone does not identify whether the representation, the permitted predictor class, or the fitted model is responsible.

This research studies how to retain useful information in geometric representations and how to assess their forecasting limits. Kernel Affine Hull Machines (KAHMs) provide the main reconstruction method; affine and kernel-weighted averaging reconstructions provide comparison methods.

## Choose a research question

| Question | Contribution | Reading and code |
| --- | --- | --- |
| What can normalized reconstruction weights predict, and when do they support an autonomous transition model? | Prediction limits and conditions for a closed Koopman model of the representation. | [Foundational preprint](https://arxiv.org/abs/2609.32652v3) |
| Which additional features should accompany normalized reconstruction weights? | Explicit recovery relations for normalization factors and weighted local coordinates, followed by matched forecasting comparisons. | [Feature augmentation](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) and its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) |
| Could any predictor in a specified representation class meet an error requirement? | A hierarchy of lower risk bounds, calibrated decisions, and experiments distinguishing class limitations from unsuccessful fitted predictors. | [Prediction-risk bounds](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) and its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) |

### Feature augmentation

**Feature Augmentation for Forecasting with Reconstruction-Based Representations** asks what a forecasting model can gain when geometric memberships are augmented. Memberships with log normalizers recover the reconstruction scores. Membership-weighted local coordinates can preserve the input under explicit positivity and rank conditions. Whether a fitted predictor benefits also depends on its architecture, regularization, and training data.

The central experiment uses 64 independent repetitions under each of four data-generating distributions. Its 40 adjusted paired comparisons favor KAHM local-position features over KAHM memberships in six of eight distribution–predictor settings. Comparisons with affine features, kernel-weighted averaging features, and raw histories are mixed. These findings support task-specific evaluation of enrichment.

The enrichment branch supplies 17 experiment records, including the per-run loss table. Its reproduction guide describes the available data, analysis scripts and their limits. An independent checker rebuilds all 40 primary comparisons from the complete selected-loss table. A separate supplementary predictor-capacity bundle includes its original experiment source, archived records, and verification, audit and full-rerun commands. The central experiment's original generator, runner, exact task roster and numerical model/forecast archives remain unavailable; saved-loss analysis does not regenerate its predictions. See the linked reproduction guide for each command's scope.

### Lower bounds on prediction risk

**A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations** asks whether a feature map and a specified predictor class can meet a mean-squared-error requirement. The class includes a distance and a limit on how much predictions may change as features move apart. This sensitivity restriction is part of the claim.

The framework combines lower bounds on the best class risk with upper bounds for admissible predictors to support exclusion, acceptance, or an unresolved decision. Population conclusions require the stated sampling, calibration, and numerical assumptions. Other experiments assess capacity on fixed datasets or the numerical evaluation of stored forecasts.

The risk branch includes seven experiment groups and a driver that recomputes checks from fixed records. Replaying those records, refitting the original models, and collecting new observations are distinct tasks. Its guide explains what each command does and which inputs it uses.

## Research code and data

| Entry point | Purpose |
| --- | --- |
| [`main`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/main) | Repository navigation and the original Koopman-machine software and reproducibility records below |
| [`paper/geometry-induced-soft-state-clean-2026-11-02`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) | Feature augmentation: central saved-loss analysis and the supplementary capacity experiment |
| [`research/risk-certificates-consolidated-2026-11-08`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) | Prediction-risk bounds: seven experiment groups, fixed-record checks and a worked example of predictor-class capacity |

The [study archive](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a) provides code, data and protocols at commit `5123365917ae630b684f8a3bebb0294ee695ce7a`. The [source index](docs/BRANCH_ARCHIVE.json) identifies code snapshots for individual studies. Each study has its own observations and analysis; its results are not additional repetitions of the central feature-augmentation experiment.

## Related studies and original software

The [research index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/docs/RESEARCH_INDEX.md) links the controlled experiments, predictor-capacity comparisons, residual forecasting studies and model-selection results. Their different targets, populations, and comparison criteria should be interpreted separately.

The material below describes the Koopman-machine software. The two linked studies provide their own code, data and reproduction instructions.

## Original Koopman-machine software

Research software and reproducibility artifacts for Kernel Affine Hull Koopman Machine (KAHKM) experiments.

The repository contains the KAHKM implementation, experiment runners, hyperparameter-search scripts, archived numerical outputs, figure and table generators, and provenance records used to reproduce the reported results.

## Reproducibility

The authoritative mapping from each reported result to its generator, experiment runner, configuration, seeds, raw or per-run output, and aggregate artifact is:

`reproduction/reported_results.tsv`

Entries marked `PASS` identify results with documented inputs, code and outputs. Entries marked `REMOVE` are excluded from the reported result set.

### Reference environment

The reference environment is native Apple Silicon (`arm64`) macOS.

Key files:

- `requirements.txt` — direct Python dependencies.
- `requirements-lock-arm64.txt` — fixed reference dependency environment.
- `scripts/bootstrap_arm64.sh` — native Apple Silicon environment bootstrap.
- `scripts/verify_environment.py` — architecture, dependency, and import verification.
- `provenance/reference_arm64_environment/` — captured hardware, operating-system, Python, NumPy, and package metadata.

Create the reference environment with:

```bash
bash scripts/bootstrap_arm64.sh
```

Verify it with:

```bash
.venv-arm64/bin/python scripts/verify_environment.py
```

### Table and figure generators

The summary tables and figures can be generated with:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Other reported tables and figures use the corresponding `step_*` generators and experiment runners identified in `reproduction/reported_results.tsv`.

## Reproducibility evidence

The following files describe the data and checks supporting the results:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — evidence summary and limitations.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — raw scientific inputs.
- `reproduction/run_accounting/README.md` — coverage checks for archived experiment runs.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — separate 27-job Duffing reference experiment.

The run-accounting audit documents an incomplete Duffing reference-search archive. A separate 27-job experiment records every scheduled attempt, exit status and output checksum. The two sets of records describe different executions.

The available records do not establish that every failed execution in every
experiment was saved.

## Repository organization

- `experiment_*.py` — primary experiment implementations.
- `run_*.py` — experiment and hyperparameter-search runners.
- `step_*.py` — table and figure generation workflows.
- `scripts/` — environment and reproducibility utilities.
- `reproduction/` — result-to-source mapping and generated tables, figures and checks.
- `provenance/` — environment and dependency provenance.
- `figures/` — generated figures used by the associated research work.
- `kahkm_*` directories and archives — experiment outputs and derived artifacts.

## Licensing

Original source code in this repository is licensed under the MIT License. See `LICENSE`.

Unless otherwise indicated, original generated research data and non-code research outputs are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See `DATA_LICENSE`.

Third-party dependencies and any separately identified third-party material remain under their respective licenses.

## Citation

Citation metadata is provided in `CITATION.cff`.

Use the repository and article metadata in `CITATION.cff`, together with the commit identifier for the code used.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
