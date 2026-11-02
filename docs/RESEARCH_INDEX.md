# Research and evidence index

The current forecasting revision is **Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines**. Its canonical home is [papers/soft_state_forecasting/](../papers/soft_state_forecasting/). This index maps the current manuscript and new representation study to the preserved evidence on which the research developed.

**Status: all 256 tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `2b7d19b5`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f). The [audit receipt](../extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) records verification coverage and limits. Three of four primary criteria are established; the KAHM-versus-RBF superiority margin is not.

The [research branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-soft-state-ablation-2026-11-02) is `research/kahm-soft-state-ablation-2026-11-02`. This revision maintains the manuscript and outcome documentation alongside its published frozen scientific source. Relative links below resolve within this revision.

## Current entry points

| Resource | Location and scope |
| --- | --- |
| Main paper | [main.pdf](../papers/soft_state_forecasting/main.pdf), [main.tex](../papers/soft_state_forecasting/main.tex) |
| Supplementary studies | [supplement.pdf](../papers/soft_state_forecasting/supplement.pdf), [supplement.tex](../papers/soft_state_forecasting/supplement.tex) |
| Editable files and build guide | [Source directory](../papers/soft_state_forecasting/), [README_BUILD.md](../papers/soft_state_forecasting/README_BUILD.md) |
| G/P/K/R representation ablation | [Study directory](../extensions/soft_state_ablation_20261102/), [primary contrasts](../extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv), [analysis summary](../extensions/soft_state_ablation_20261102/analysis/analysis_summary.json) |
| Interpretation of completed studies | [TIME_SERIES_RESEARCH.md](../TIME_SERIES_RESEARCH.md) |
| Scientific reproduction and audit scope | [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) |
| Paper scope and review materials | [TMLR.md](../TMLR.md) |

The new ablation has completed 32 fresh repetition units and 256 tasks, with frozen analysis complete and independent production audit passed. Its primary P-selected shared-bank family establishes nonlinear P/G and K/P superiority and linear K/P noninferiority; K/R superiority is unestablished, giving three of four criteria. The nonlinear gains are 69.201%, 10.459% and 0.0292%, respectively. Linear K loss is 1.111% higher than P while meeting the 5% tolerance. Independent-cap comparisons are secondary. These results are separate from the completed earlier policy and representation experiments.

## Forecasting evidence retained in the current research

These locations are stable study identifiers. Moving or renaming them would obscure source manifests, archive membership, or historical execution commands, so the navigation changes leave their contents and paths intact.

| Study directory | Evidence and interpretation entry point |
| --- | --- |
| `extensions/dimension_forecasting/` | [Findings](../extensions/dimension_forecasting/FINDINGS.md), [reproduction](../extensions/dimension_forecasting/README.md): earlier principal validation-selected dimension comparison, 24 repetition units and its own six-criterion family |
| `extensions/preprocessing_dimension/` | [Findings](../extensions/preprocessing_dimension/FINDINGS.md), [records](../extensions/preprocessing_dimension/RESULTS_PACKAGE.md): a separate four-criterion study and secondary fixed-common corrections |
| `extensions/crossfit_residual_reproduction_20261101/` | [Results](../extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md), [execution guide](../extensions/crossfit_residual_reproduction_20261101/README.md): same-cohort reproduction with prior outcome knowledge, not fresh independent repetitions |
| `extensions/policy_augmentation_20261101/` | [Results](../extensions/policy_augmentation_20261101/RESULTS.md), [reproduction](../extensions/policy_augmentation_20261101/README.md): fresh 24-repetition policy study; neither prescribed nonlinear superiority criterion established |
| `extensions/policy_augmentation_natural_20261101/` | [Findings](../extensions/policy_augmentation_natural_20261101/FINDINGS.md), [reproduction](../extensions/policy_augmentation_natural_20261101/REPRODUCTION.md): descriptive chronological evaluation of 48 M4 training series; no official Test data |

## Earlier forecasting and development investigations

| Directory | Scientific role |
| --- | --- |
| [reconstruction_mechanism](../extensions/reconstruction_mechanism/README.md) | Historical fixed-cap reconstruction-departure comparison, algebraic examples, and saved-forecast verification |
| [predictive_geometry](../extensions/predictive_geometry/README.md) | Controlled representation and finite-decoder experiments |
| [natural_forecasting](../extensions/natural_forecasting/README.md) | Natural-data forecasting and source provenance |
| [scaling_sensitivity](../extensions/scaling_sensitivity/README.md) | Scaling changes on previously evaluated data |
| [forecasting_utility](../extensions/forecasting_utility/README.md) | Additive corrections to a retained raw-input forecast |
| [target_geometry](../extensions/target_geometry/README.md) | Supervised metric learning and donor-residual comparisons |
| [direct_reconstruction](../extensions/direct_reconstruction/README.md) | Forecast training of local bandwidths and gates |
| [adaptive_development](../extensions/adaptive_development/REPRODUCING.md) | Online adaptation, numerical recovery, and cost/accuracy evaluation |
| [forecast_gate_pilot](../extensions/forecast_gate_pilot/README.md) | Exploratory continuation workbench and pilot evidence |
| [crossfit_residual](../extensions/crossfit_residual/README.md) | Original residual-study attempt, preserved with its own execution history |
| [crossfit_residual_v2](../extensions/crossfit_residual_v2/README.md) | Numerical amendment and scientific source inherited by the later reproduction |

A pilot, post hoc sensitivity analysis, same-seed reproduction, or numerical amendment does not acquire a new inferential status because it is linked from the current manuscript. Each study retains its original population, comparison family, failure accounting, and evidence limits.

## Earlier documents and Koopman architecture

The older coordinated forecasting manuscript remains under [extensions/tmlr_integration/](../extensions/tmlr_integration/), with its [document release record](../extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json). It is an earlier edition, not the current paper location.

The original KAHKM experiment chain uses the root `experiment_*.py`, `run_exp*.py`, and `step_*.py` files and the associated `kahkm_*` outputs. Their result mapping is [reproduction/reported_results.tsv](../reproduction/reported_results.tsv), and their evidence summary is [reproduction/REPRODUCIBILITY_EVIDENCE.md](../reproduction/REPRODUCIBILITY_EVIDENCE.md). The platform-specific environment remains under [provenance/](../provenance/). The root Python requirements belong to that architecture.

KAHM local reconstruction, the current gate-and-observable forecasting representation, the companion reconstruction-score-normalized soft abstraction, and the earlier KAHKM architecture are different objects. They need not have the same prediction target or decoder. A direct forecast of an observed output is not a test of autonomous coordinate propagation, and exact closure of one representation is not transferred to another by a shared name.

## Provenance and use

Use the current paper directory for manuscript builds, the particular study directory for scientific execution, and the study's archive manifest for numerical restoration. A hash check, analysis from saved forecasts, selected-model replay, refitting, and external replication have distinct scopes. Some historical audits include an entire old repository tree and require that historical checkout; they do not validate the present documentation edition.

The manuscript and documentation may be updated without changing earlier frozen experiments. Source freezes, target-release records, and numerical archives retain their separate identities. Cite the paper version, specific study, and immutable repository commit used rather than treating a moving branch name as a scientific checkpoint.

The completed four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from the frozen scientific source and manuscript edition.
