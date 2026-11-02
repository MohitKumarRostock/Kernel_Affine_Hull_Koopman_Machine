# Geometry-induced soft-state abstractions for forecasting

This repository contains the manuscripts, implementations, and retained evidence for research on Kernel Affine Hull Machines (KAHMs). The current forecasting paper is **Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines**. It studies whether geometric memberships, local position coordinates, and reconstruction-derived observables help a finite decoder predict the same future observations.

**Joint-selection status: all 256 production tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `8e025027`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/8e0250279074aefa39079fa33205d3c74f835848). The analysis and audit establish four of six primary criteria. It does not establish the KAHM-versus-RBF or joint-versus-standard 3% superiority margin. The [complete results](extensions/soft_state_joint_selection_20261102/RESULTS.md) retain favorable and unfavorable comparisons.

## Start here

| Purpose | Entry point |
| --- | --- |
| Current forecasting manuscript | [Main PDF](papers/soft_state_forecasting/main.pdf), [supplement](papers/soft_state_forecasting/supplement.pdf) |
| Editable manuscript and build instructions | [Source directory](papers/soft_state_forecasting/), [build guide](papers/soft_state_forecasting/README_BUILD.md) |
| Joint scaling and dimension selection | [Study and execution guide](extensions/soft_state_joint_selection_20261102/), [results](extensions/soft_state_joint_selection_20261102/RESULTS.md) |
| Preserved standard-only four-arm ablation | [extensions/soft_state_ablation_20261102/](extensions/soft_state_ablation_20261102/) |
| Map of current and historical research | [Research index](docs/RESEARCH_INDEX.md) |
| Findings and scientific limits across studies | [Research evidence guide](TIME_SERIES_RESEARCH.md) |
| Restore records, reproduce analyses, and distinguish audit scopes | [Reproducibility guide](REPRODUCIBILITY.md) |
| Manuscript scope and TMLR presentation | [Manuscript guide](TMLR.md) |

The revision uses the [research branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-soft-state-joint-selection-2026-11-02), `research/kahm-soft-state-joint-selection-2026-11-02`. This revision maintains the manuscript and outcome documentation alongside its published frozen scientific source. The canonical manuscript is maintained as PDFs and individual editable files in its source directory. Earlier manuscript editions remain under [extensions/tmlr_integration/](extensions/tmlr_integration/); they retain their original release records and are not the current paper entry point. Historical experiment directories and frozen records remain at their original paths.

## Current representation question

Within each reference partition, nonnegative locality gates sum to one. Their concatenation is a product of simplices. Gate-weighted principal coordinates add position within those reference regions, and KAHM reconstruction departures add signed observables. The complete enriched descriptor is not simplex-valued, and neither sufficiency nor a Markov property is assumed. Matched KAHM and positive-kernel RBF reconstructions share the gates and reference geometry.

The new joint-selection study has completed its fixed 32 fresh repetitions and 256 tasks under four known simulation laws, with horizons 1, 4, and 12. It selects among three A-fitted input scalers and five native dimension caps using B. Its independent production audit passed 7,422,883 checks. The four representations are:

| Arm | Representation |
| --- | --- |
| G | Locality memberships |
| P | Memberships and local principal coordinates |
| K | P with KAHM reconstruction departures |
| R | P with matched RBF reconstruction departures |

The comparison asks what local position and reconstruction enrichment add to prediction of a common external target. P selects the common scaler/cap pair for four structural primary contrasts. Two additional primary contrasts compare independently tuned joint K with standard-only K using the exact same cohort and cached standard candidate fits. All six receive one Bonferroni adjustment. Other independently selected comparisons, including K/R, remain descriptive. The study's [protocol](extensions/soft_state_joint_selection_20261102/PROTOCOL.md) defines the fixed design and inference; earlier observations are not pooled.

| Primary comparison | Panel | Relative gain | Prescribed criterion | Established |
| --- | --- | ---: | --- | --- |
| Shared P versus G | Nonlinear | 68.69% | 3% superiority | Yes |
| Shared K versus P | Nonlinear | 8.86% | 3% superiority | Yes |
| Shared K versus R | Nonlinear | -0.9186% | 3% superiority | No |
| Shared K versus P | Linear | -1.26% | 5% noninferiority | Yes |
| Independent joint K versus standard-only K | Nonlinear | 0.0727% | 3% superiority | No |
| Independent joint K versus standard-only K | Linear | -0.2491% | 5% noninferiority | Yes |

These audited outcomes establish four of six criteria. Local position improves the memberships-only readout, and KAHM departures improve the common-coordinate readout under the prescribed procedure. K has 0.9186% higher nonlinear loss than matched R. Joint tuning changes independent K's nonlinear mean by only 0.0727% in the favorable direction and does not establish the prescribed improvement. Both linear comparisons satisfy their 5% noninferiority criteria despite higher mean loss. The two unmet superiority margins are not equivalence findings. These are direct observed-output forecasts, not evidence of sufficient Markov states or autonomous soft-state dynamics.

See the [six primary contrasts](extensions/soft_state_joint_selection_20261102/analysis/primary_contrasts.csv), [panel losses](extensions/soft_state_joint_selection_20261102/analysis/panel_summary.csv), and [analysis summary](extensions/soft_state_joint_selection_20261102/analysis/analysis_summary.json). Per-law independent K tuning effects are mixed: mean gains of 0.2269% on Van der Pol and 0.5176% on Duffing, with increases in loss of 0.4884% on nonlinear AR and 0.2491% on linear AR. These per-law comparisons and scaler/cap frequencies are descriptive.

The [passed independent audit](extensions/soft_state_joint_selection_20261102/verification/independent_audit_summary.json) records 7,422,883 checks, independently refits and replays all 76,800 B penalty-path arrays, and replays all 4,096 C and 4,096 D forecast arrays. It confirms all six primary bounds and records zero numerical solver recoveries. Physical-law integration and partition k-means fitting are not independently reimplemented; this is internal computational verification, not external scientific replication.

The mathematical analysis distinguishes raw feature spans, regularized decoder access, and exact dynamical closure. Some matched KAHM and RBF augmented raw spans coincide; a specified irregular reference geometry separates them. Equal spans have the same exact Koopman-invariance status under a specified deterministic evolution. That conditional statement does not demonstrate closure for a fitted representation. Bounded-feature ridge algebra controls correction magnitude, without guaranteeing accuracy or stable coordinate propagation. The forecasting experiments use direct external-target readouts.

## Earlier forecasting evidence

The previous studies remain separate scientific experiments with their own procedures, samples, and comparison families.

| Study | Main finding and scope |
| --- | --- |
| [Standard-only G/P/K/R ablation](extensions/soft_state_ablation_20261102/RESULTS.md) | Separate 32-repetition, 256-task cohort and passed audit; three of four criteria established. Shared P/G and K/P gains are 69.201% and 10.459%; the 0.0292% K/R gain does not establish its 3% margin. |
| [Validation-selected local dimension](extensions/dimension_forecasting/FINDINGS.md) | 24 repetition units, 192 tasks. Three of six primary criteria established: KAHM gains over common features and raw-input RBF, plus linear noninferiority. The selected non-KAHM procedure has lower mean error. |
| [Preprocessing and dimension](extensions/preprocessing_dimension/FINDINGS.md) | One of four primary criteria established. Secondary fixed-common comparisons show useful residual corrections; they do not replace the primary complete-procedure result. |
| [Residual-target computational reproduction](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md) | Two of four original criteria established. The same 24 original simulation units were regenerated with prior outcome knowledge; this contributes no additional independent repetitions. |
| [Fresh policy augmentation](extensions/policy_augmentation_20261101/RESULTS.md) | 24 fresh repetitions, 192 tasks. Adding KAHM lowers nonlinear mean loss by 0.816% versus the base policy and 0.658% versus the added-RBF control. Neither prescribed 3% superiority criterion is established; both linear noninferiority criteria pass. |
| [Separate natural policy evaluation](extensions/policy_augmentation_natural_20261101/FINDINGS.md) | A descriptive 48-series M4 training-data panel. KAHM augmentation has 2.273% higher aggregate loss than the base policy and changes three decisions. No official M4 Test data or independent-source inference is used. |

These findings support conditional predictive value of particular corrections. They do not establish general KAHM superiority, nor are their observations pooled into one test. Equal added search counts in the policy study do not equate feature capacity or runtime. See the [evidence guide](TIME_SERIES_RESEARCH.md) for the retained numerical findings and limitations, including the older controlled, natural-data, training, and adaptation studies.

## Reproduction and manuscript builds

Use the [current paper directory](papers/soft_state_forecasting/) for document builds and the [joint-selection study directory](extensions/soft_state_joint_selection_20261102/) for its scientific execution instructions. A document build uses supplied figures and tables; it is distinct from regenerating forecasts or independently refitting a model.

The [reproducibility guide](REPRODUCIBILITY.md) links each historical study's environment, archive manifest, analysis commands, and audit receipt. For example, the dimension-selected experiment requires its own six numerical archives and pinned Python environment. The root `requirements.txt` belongs to the earlier KAHKM architecture and is not a universal forecasting-study environment.

Keep numerical reruns and regenerated analysis outputs separate from the retained evidence. A hash check, saved-forecast analysis, selected-model replay, refitting, and external scientific replication are different operations. Internal audits are not external replications, and source freezes are not automatically public preregistrations.

## Repository map

| Location | Role |
| --- | --- |
| `papers/soft_state_forecasting/` | Canonical current forecasting manuscript and document build entry point |
| `extensions/soft_state_joint_selection_20261102/` | Joint scaling/cap selection and same-cohort standard-only controls; independent audit passed |
| `extensions/soft_state_ablation_20261102/` | Preserved completed standard-only G/P/K/R representation study |
| `extensions/` | Study-specific protocols, code, reporting, audits, and retained evidence; see the [research index](docs/RESEARCH_INDEX.md) |
| `extensions/tmlr_integration/` | Preserved earlier coordinated manuscript editions |
| `reproduction/`, `provenance/` | Earlier KAHKM experiment mappings, environments, and provenance |
| Root `experiment_*.py`, `step_*.py`, and `kahkm_*` outputs | Earlier KAHKM runners and derived artifacts, retained at their historical paths |

KAHM denotes the local geometric reconstruction. KAHKM denotes the earlier Kernel Affine Hull Koopman Machine architecture. Their targets, representations, and readouts differ; evidence does not transfer automatically between them. The companion soft-abstraction analysis also uses reconstruction-score-normalized memberships, a different map from the locality gates evaluated in the current forecasting pipeline.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Unless otherwise indicated, original generated research data and non-code research outputs are distributed under [CC BY 4.0](DATA_LICENSE). Third-party data, dependencies, and template files retain their stated licenses and attribution.

Repository citation metadata appears in [CITATION.cff](CITATION.cff). Cite the particular manuscript version, study, and repository commit used. Anonymous TMLR review formatting does not assert submission, acceptance, or endorsement; editable source preserves required attribution.

The predecessor four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from that study's frozen scientific source and manuscript edition.
