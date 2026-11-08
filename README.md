# Kernel Affine Hull Koopman Machine

## Geometry-induced representations for forecasting

A geometry-induced state abstraction summarizes an observed history through its affinities to reference regions. Such a summary can have a useful geometric interpretation while still losing distinctions needed for prediction. When a fitted forecaster performs poorly, its error alone does not identify whether the representation, the permitted predictor class, or the fitted model is responsible.

This research studies how to retain useful information in geometric representations and how to assess their forecasting limits. Kernel Affine Hull Machines (KAHMs) provide the main reconstruction method; affine and positive-kernel reconstructions provide comparison methods.

## Choose a research question

| Question | Contribution | Reading and code |
| --- | --- | --- |
| What can a geometry-induced abstraction predict, and when can its state evolve autonomously? | Analysis of prediction limits and Koopman closure for geometric soft states. | [Foundational preprint](https://arxiv.org/abs/2609.32652v3) |
| What information should accompany geometric memberships? | Explicit recovery relations for normalization factors and weighted local coordinates, followed by matched forecasting comparisons. | [Predictive enrichment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) and its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) |
| Could any predictor in a specified representation class meet an error requirement? | A hierarchy of lower risk bounds, calibrated decisions, and experiments distinguishing class limitations from unsuccessful fitted predictors. | [Risk certificates](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) and its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) |

### Predictive enrichment

**Predictive Enrichment of Geometry-Induced Soft-State Abstractions: Information Retention and Decoder Access** asks what a forecasting model can gain when geometric memberships are augmented. Memberships with log normalizers recover the reconstruction scores. Membership-weighted local coordinates can preserve the input under explicit positivity and rank conditions. Whether a fitted predictor benefits also depends on its architecture, regularization, and training data.

The central experiment uses 64 independent repetitions under each of four simulation laws. Its 40 adjusted paired comparisons favor KAHM local-position features over KAHM memberships in six of eight law–predictor settings. Comparisons with affine features, positive-kernel features, and raw histories are mixed. These findings support task-specific evaluation of enrichment.

The enrichment branch supplies the paper and a central saved-loss bundle with 17 original evidence records. An independent checker rebuilds all 40 primary comparisons from the complete selected-loss table. A separate supplementary decoder-capacity bundle includes its original experiment source, archived records, and verification, audit and full-rerun commands. The central experiment's original generator, runner, exact task roster and numerical model/forecast archives remain unavailable; saved-loss analysis does not regenerate its predictions. See the linked reproduction guide for each command's scope.

### Risk certificates

**A Hierarchy of Risk Certificates for Forecasting Representations** asks whether a feature map and a specified predictor class can meet a mean-squared-error requirement. The class includes a distance and a limit on how much predictions may change as features move apart. This sensitivity restriction is part of the claim.

The framework combines lower bounds on the best class risk with upper bounds for admissible predictors to support exclusion, acceptance, or an unresolved decision. Population conclusions require the stated sampling, calibration, and numerical assumptions. Other experiments assess capacity on fixed datasets or the numerical evaluation of stored forecasts.

The risk branch includes seven experiment groups and a driver that recomputes checks from fixed records. Replaying those records, refitting the original models, and collecting new observations are distinct tasks. Its guide explains what each command does and which inputs it uses.

## Active branches and historical archive

| Entry point | Purpose |
| --- | --- |
| [`main`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/main) | Repository navigation and the original Koopman-machine software and reproducibility records below |
| [`paper/geometry-induced-soft-state-clean-2026-11-02`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) | Predictive enrichment: central saved-loss analysis and the supplementary capacity experiment |
| [`research/risk-certificates-consolidated-2026-11-08`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) | Risk certificates: seven experiment groups, fixed-record checks and a worked class-adequacy example |

The retained historical archive is [`archive/pre-manuscript-cleanup-2026-11-02`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5123365917ae630b684f8a3bebb0294ee695ce7a). Commit `5123365917ae630b684f8a3bebb0294ee695ce7a` defines its immutable contents. The [branch archive catalog](docs/BRANCH_ARCHIVE.json) records the retained historical head identities and their preservation links. The research index identifies other historical studies by commit. These records preserve their original protocols and outcomes; they are not extra repetitions of the central enrichment experiment.

## Related studies and original software

The [research index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/docs/RESEARCH_INDEX.md) links the earlier controlled, capacity, correction, and model-selection studies with their original results. Their different targets, populations, and comparison criteria should be interpreted separately.

The material below concerns the original Koopman-machine software on this default branch. The two papers above use their own existing branches and reproduction instructions.

## Original Koopman-machine software

Research software and reproducibility artifacts for Kernel Affine Hull Koopman Machine (KAHKM) experiments.

The repository contains the KAHKM implementation, experiment runners, hyperparameter-search scripts, archived numerical outputs, figure and table generators, and provenance records used to reproduce the reported results.

## Reproducibility

The authoritative mapping from each reported result to its generator, experiment runner, configuration, seeds, raw or per-run output, and aggregate artifact is:

`reproduction/reported_results.tsv`

Entries marked `PASS` correspond to retained reported results with an identified reproducibility chain. Entries marked `REMOVE` identify legacy outputs that are not part of the canonical reported result set.

### Reference environment

The reference environment is native Apple Silicon (`arm64`) macOS.

Key files:

- `requirements.txt` — direct Python dependencies.
- `requirements-lock-arm64.txt` — frozen reference dependency environment.
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

### Canonical derived-result generators

Composite or newly canonicalized reported artifacts can be regenerated with:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Other reported tables and figures use the corresponding `step_*` generators and experiment runners identified in `reproduction/reported_results.tsv`.

## Reproducibility evidence

For a reviewer-facing summary of the current evidence state, see:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence summary and claim boundaries.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical audit and prospective gap-closing campaign are intentionally
kept separate. The historical run-accounting audit preserves the original
Duffing reference-search gap, while the prospective campaign records all 27
scheduled attempts and their exit statuses from a frozen clean commit.

The repository does not claim that every failed attempt from every historical
campaign was preserved when the available historical evidence does not support
that stronger statement.

## Repository organization

- `experiment_*.py` — primary experiment implementations.
- `run_*.py` — experiment and hyperparameter-search runners.
- `step_*.py` — table and figure generation workflows.
- `scripts/` — environment and canonical reproducibility utilities.
- `reproduction/` — result-provenance matrix and canonical generated artifacts.
- `provenance/` — environment and dependency provenance.
- `figures/` — generated figures used by the associated research work.
- `kahkm_*` directories and archives — experiment outputs and derived artifacts.

## Licensing

Original source code in this repository is licensed under the MIT License. See `LICENSE`.

Unless otherwise indicated, original generated research data and non-code research outputs are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See `DATA_LICENSE`.

Third-party dependencies and any separately identified third-party material remain under their respective licenses.

## Citation

Citation metadata is provided in `CITATION.cff`.

A persistent archival identifier such as a DOI should be cited once an immutable archival release is available. Until then, cite the repository and associated research work using the metadata in `CITATION.cff`.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
