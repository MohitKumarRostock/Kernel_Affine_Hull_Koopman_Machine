# Geometry-induced soft-state abstractions for forecasting

This repository contains the manuscripts, implementations, and retained evidence for research on Kernel Affine Hull Machines (KAHMs). The current forecasting paper is **Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines**. It studies whether geometric memberships, local position coordinates, and reconstruction-derived observables help a finite decoder predict the same future observations.

**Status: all 256 tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `2b7d19b5`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f). The [audit receipt](extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) records verification coverage and limits. Three of four primary criteria are established; the KAHM-versus-RBF superiority margin is not.

## Start here

| Purpose | Entry point |
| --- | --- |
| Current forecasting manuscript | [Main PDF](papers/soft_state_forecasting/main.pdf), [supplement](papers/soft_state_forecasting/supplement.pdf) |
| Editable manuscript and build instructions | [Source directory](papers/soft_state_forecasting/), [build guide](papers/soft_state_forecasting/README_BUILD.md) |
| New four-arm representation ablation | [extensions/soft_state_ablation_20261102/](extensions/soft_state_ablation_20261102/) |
| Map of current and historical research | [Research index](docs/RESEARCH_INDEX.md) |
| Findings and scientific limits across studies | [Research evidence guide](TIME_SERIES_RESEARCH.md) |
| Restore records, reproduce analyses, and distinguish audit scopes | [Reproducibility guide](REPRODUCIBILITY.md) |
| Manuscript scope and TMLR presentation | [Manuscript guide](TMLR.md) |

The revision uses the [research branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/kahm-soft-state-ablation-2026-11-02), `research/kahm-soft-state-ablation-2026-11-02`. This revision maintains the manuscript and outcome documentation alongside its published frozen scientific source. The canonical manuscript is maintained as PDFs and individual editable files in its source directory. Earlier manuscript editions remain under [extensions/tmlr_integration/](extensions/tmlr_integration/); they retain their original release records and are not the current paper entry point. Historical experiment directories and frozen records remain at their original paths.

## Current representation question

Within each reference partition, nonnegative locality gates sum to one. Their concatenation is a product of simplices. Gate-weighted principal coordinates add position within those reference regions, and KAHM reconstruction departures add signed observables. The complete enriched descriptor is not simplex-valued, and neither sufficiency nor a Markov property is assumed. Matched KAHM and positive-kernel RBF reconstructions share the gates and reference geometry.

The new study has completed its fixed 32 fresh repetitions and 256 tasks under four known simulation laws, with horizons 1, 4, and 12. The independent production audit has passed. The four representations are:

| Arm | Representation |
| --- | --- |
| G | Locality memberships |
| P | Memberships and local principal coordinates |
| K | P with KAHM reconstruction departures |
| R | P with matched RBF reconstruction departures |

The comparison asks what local position and reconstruction enrichment add to prediction of a common external target. In the primary context, P selects the common bank on B; the four claims are nonlinear P versus G, K versus P, and K versus R at a 3% superiority margin, plus linear K versus P at a 5% noninferiority margin. Independently selected banks form a secondary context. The study's own protocol and execution records define its design, inferential family, and status. No result from an earlier study is a result of this new ablation.

| Primary comparison | Panel | Relative gain | Prescribed criterion | Established |
| --- | --- | ---: | --- | --- |
| P versus G | Nonlinear | 69.201% | 3% superiority | Yes |
| K versus P | Nonlinear | 10.459% | 3% superiority | Yes |
| K versus R | Nonlinear | 0.0292% | 3% superiority | No |
| K versus P | Linear | -1.111% | 5% noninferiority | Yes |

Three of four primary criteria are established. Local position materially improves the memberships-only readout, and KAHM departures improve the common-coordinate readout under the prescribed procedure. The KAHM-specific superiority margin over matched RBF remains unestablished; the small mean difference is not an equivalence finding. Linear KAHM loss is 1.111% higher than P while satisfying the 5% noninferiority criterion. These are direct observed-output forecasts, not evidence of sufficient Markov states or autonomous soft-state dynamics.

See the [four primary contrasts](extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv), [panel losses](extensions/soft_state_ablation_20261102/analysis/panel_summary.csv), and [analysis summary](extensions/soft_state_ablation_20261102/analysis/analysis_summary.json).

The mathematical analysis distinguishes raw feature spans, regularized decoder access, and exact dynamical closure. Some matched KAHM and RBF augmented raw spans coincide; a specified irregular reference geometry separates them. Equal spans have the same exact Koopman-invariance status under a specified deterministic evolution. That conditional statement does not demonstrate closure for a fitted representation. Bounded-feature ridge algebra controls correction magnitude, without guaranteeing accuracy or stable coordinate propagation. The forecasting experiments use direct external-target readouts.

## Earlier forecasting evidence

The previous studies remain separate scientific experiments with their own procedures, samples, and comparison families.

| Study | Main finding and scope |
| --- | --- |
| [Validation-selected local dimension](extensions/dimension_forecasting/FINDINGS.md) | 24 repetition units, 192 tasks. Three of six primary criteria established: KAHM gains over common features and raw-input RBF, plus linear noninferiority. The selected non-KAHM procedure has lower mean error. |
| [Preprocessing and dimension](extensions/preprocessing_dimension/FINDINGS.md) | One of four primary criteria established. Secondary fixed-common comparisons show useful residual corrections; they do not replace the primary complete-procedure result. |
| [Residual-target computational reproduction](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md) | Two of four original criteria established. The same 24 original simulation units were regenerated with prior outcome knowledge; this contributes no additional independent repetitions. |
| [Fresh policy augmentation](extensions/policy_augmentation_20261101/RESULTS.md) | 24 fresh repetitions, 192 tasks. Adding KAHM lowers nonlinear mean loss by 0.816% versus the base policy and 0.658% versus the added-RBF control. Neither prescribed 3% superiority criterion is established; both linear noninferiority criteria pass. |
| [Separate natural policy evaluation](extensions/policy_augmentation_natural_20261101/FINDINGS.md) | A descriptive 48-series M4 training-data panel. KAHM augmentation has 2.273% higher aggregate loss than the base policy and changes three decisions. No official M4 Test data or independent-source inference is used. |

These findings support conditional predictive value of particular corrections. They do not establish general KAHM superiority, nor are their observations pooled into one test. Equal added search counts in the policy study do not equate feature capacity or runtime. See the [evidence guide](TIME_SERIES_RESEARCH.md) for the retained numerical findings and limitations, including the older controlled, natural-data, training, and adaptation studies.

## Reproduction and manuscript builds

Use the [current paper directory](papers/soft_state_forecasting/) for document builds and the [new study directory](extensions/soft_state_ablation_20261102/) for its scientific execution instructions. A document build uses supplied figures and tables; it is distinct from regenerating forecasts or independently refitting a model.

The [reproducibility guide](REPRODUCIBILITY.md) links each historical study's environment, archive manifest, analysis commands, and audit receipt. For example, the dimension-selected experiment requires its own six numerical archives and pinned Python environment. The root `requirements.txt` belongs to the earlier KAHKM architecture and is not a universal forecasting-study environment.

Keep numerical reruns and regenerated analysis outputs separate from the retained evidence. A hash check, saved-forecast analysis, selected-model replay, refitting, and external scientific replication are different operations. Internal audits are not external replications, and source freezes are not automatically public preregistrations.

## Repository map

| Location | Role |
| --- | --- |
| `papers/soft_state_forecasting/` | Canonical current forecasting manuscript and document build entry point |
| `extensions/soft_state_ablation_20261102/` | New G/P/K/R representation study; its status is separate from all earlier results |
| `extensions/` | Study-specific protocols, code, reporting, audits, and retained evidence; see the [research index](docs/RESEARCH_INDEX.md) |
| `extensions/tmlr_integration/` | Preserved earlier coordinated manuscript editions |
| `reproduction/`, `provenance/` | Earlier KAHKM experiment mappings, environments, and provenance |
| Root `experiment_*.py`, `step_*.py`, and `kahkm_*` outputs | Earlier KAHKM runners and derived artifacts, retained at their historical paths |

KAHM denotes the local geometric reconstruction. KAHKM denotes the earlier Kernel Affine Hull Koopman Machine architecture. Their targets, representations, and readouts differ; evidence does not transfer automatically between them. The companion soft-abstraction analysis also uses reconstruction-score-normalized memberships, a different map from the locality gates evaluated in the current forecasting pipeline.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Unless otherwise indicated, original generated research data and non-code research outputs are distributed under [CC BY 4.0](DATA_LICENSE). Third-party data, dependencies, and template files retain their stated licenses and attribution.

Repository citation metadata appears in [CITATION.cff](CITATION.cff). Cite the particular manuscript version, study, and repository commit used. Anonymous TMLR review formatting does not assert submission, acceptance, or endorsement; editable source preserves required attribution.

The completed four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from the frozen scientific source and manuscript edition.
