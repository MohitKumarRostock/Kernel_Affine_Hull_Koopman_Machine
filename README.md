# Kernel Affine Hull Koopman Machine

Research software and reproducibility artifacts for Kernel Affine Hull Koopman Machine (KAHKM) experiments.

The repository contains the KAHKM implementation, experiment runners, hyperparameter-search scripts, archived numerical outputs, figure and table generators, and provenance records used to reproduce the reported results.

The software and reproducibility records are maintained independently of the
publication venue of any accompanying manuscript.

## KAHM forecasting and risk certification

The [certified forecasting extension](extensions/certified_forecasting/)
develops KAHM soft representations for direct prediction of future observed
outputs, retained representation refinement, and prediction-risk certification.
Its [editable manuscript](extensions/certified_forecasting/paper/) is titled
**A Hierarchy of Risk Certificates for Forecasting Representations**.
For bounded scalar targets, centered Lipschitz regression on independent
blocks connects the pair obstruction to a monotone population hierarchy that
converges to optimal class risk on compact metric representation spaces.
Larger blocks need not improve finite-sample endpoints; fixed block sizes
can remain arbitrarily loose. An independently audited eight-law benchmark
retains every block size and the often stronger full-sample comparator.
Its application studies evaluate synthetic dynamics, pen
trajectories, and speech energy; compare input scaling and geometric
representations under matched decoder choices; and assess lower risk bounds
and minimum feasible representation cost. The [scientific study guide](extensions/certified_forecasting/docs/STUDY_GUIDE.md)
explains the questions, independent sampling units, and interpretation of each
experiment. The comparisons report both improvements and deteriorations, along
with inconclusive certification outcomes.

The speech input-scaling comparison evaluates standardization, raw histories,
and training-only scaling to [-1,+1], with and without query clipping. It
reports all preprocessing arms and retains complete paired group scores;
preprocessing is treated as a model choice determined using development data.

A new [learned-KAHM decision study](extensions/certified_forecasting/experiments/learned_kahm_decision/)
fits the actual KAHM implementation on fresh continuous synthetic histories.
At 4,096 independent calibration episodes, stratified lower certificates
establish minimum feasible cost in all twenty repairable trials and exclude
both declared classes in all twenty irreducible trials. All trials at 256
episodes remain unresolved. A single 0.05 failure allocation covers both
regimes, every trial, all four sample sizes, and the entire certificate family.
These are controlled mechanism results; natural-data lower certificates retain
their reported limitations.

The [learned-geometry hierarchy study](extensions/certified_forecasting/experiments/learned_kahm_hierarchy/)
connects the new theory to all forty archived learned KAHM banks using fresh
independent calibration episodes. On the independent-future control, blocks
of eight or sixteen observations exclude both declared classes for all twenty
banks at 1,024 episodes, while unconditional pairs do not. Full-sample empirical
minimization is stronger at the matched 256-episode size. The study retains
all 1,760 endpoints, optimization gaps, and unresolved cases under one 0.05
failure allocation. Its conclusions are conditional on fixed fitted banks;
they do not establish that noise can be repaired by changing representations.

The main paper keeps all theorem proofs. A separately compiled
[experimental supplement](extensions/certified_forecasting/paper/supplement.pdf)
retains complete secondary tables and implementation details. The
[revision record](extensions/certified_forecasting/docs/JMLR_REPRESENTATION_REVISION.md)
explains the scientific changes, evidence, and verification scope.

Use the extension's own [reproduction guide](extensions/certified_forecasting/docs/REPRODUCIBILITY.md)
for its Linux/Python environment and experiment commands. The parent KAHKM
study and its reference environment below retain their original provenance.

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

The reported composite tables and figures can be regenerated with:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Other reported tables and figures use the corresponding `step_*` generators and experiment runners identified in `reproduction/reported_results.tsv`.

## Reproducibility evidence

The reproducibility evidence is documented in:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence and its scope.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical Duffing reference archive contains 18 of the 27 scheduled jobs.
A separate prospective campaign records all 27 attempts and their exit statuses
from a fixed source commit. These later results establish the reproducibility
of the complete grid; the historical archive remains incomplete. Execution
ledgers covering every failed attempt are available for the prospective campaign
but not for all historical campaigns.

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

`CITATION.cff` identifies the repository and associated research work. It does
not currently specify an archival DOI; the commit identifier distinguishes the
software revision used in a study.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
