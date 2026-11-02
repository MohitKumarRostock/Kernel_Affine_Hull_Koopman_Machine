# Compact evidence for the completed joint-selection study

This directory supports the manuscript's joint input-scaling and native-dimension
follow-up. It contains exact copies of 30 small source, analysis, audit-summary,
and archive records from `extensions/soft_state_joint_selection_20261102/`, plus
this inventory and `EVIDENCE_MANIFEST.json`. It is a reporting subset of the
completed study, not a standalone scientific runtime or a replacement numerical
archive. No forecasts, losses, selections, or statistical results were recomputed
when making these copies.

## Completed study and provenance

The source was published before production as commit
`8e0250279074aefa39079fa33205d3c74f835848`, tree
`1946da072f230244ad92795f9517ef89d1f06487`, on branch
`research/kahm-soft-state-joint-selection-2026-11-02`.
The study completed 256 tasks and 32 independent repetitions, retaining 4,096
terminal forecast arrays. Four of six prespecified primary criteria were
established. The independent audit passed 7,422,883 checks, refitted/replayed
76,800 B penalty paths, and replayed all 4,096 C and 4,096 D arrays. No numerical
solver recoveries occurred.

| Evidence identity | SHA-256 |
| --- | --- |
| Source freeze | `0c3737ae8d4071983c03280a3b61b693d879da76d6f8dd5034ec6134743421a8` |
| Complete analysis summary | `1345429169d0bd263d17a54fb469718a768a0be218db18059b0084176d2421e9` |
| Original full audit, held in the complete study | `8ba3931c3e705838d9f3ec924f374b8cb9e492aa6f90803684412d8151a9c89c` |
| Included audit summary | `37ac5ff28909a01f7f86df6011f95964376989064eed36e7304f97e017a24b19` |
| Included compressed archive member index | `7a4a102a66accc703890b6da1f69191b5737bac86da0a55b1cb4903b946296e8` |

`EVIDENCE_MANIFEST.json` records every included file's original repository path,
byte size, and SHA-256. CSV row counts are checked against the completed analysis
summary. Source copies, including their CSV line endings and JSON formatting,
are byte-for-byte identical. The copied compact-release records are the final
version confirmed by the numerical packaging task, following independent
fresh-process verification of all five archive files.

## Inventory and uses

| Included records | Purpose |
| --- | --- |
| `PROTOCOL.md`, `config.json`, `freeze.json`, `PUBLICATION.json` | Prespecified procedure, configuration, frozen dependencies, and actual public source identity |
| `FIT_LOCK.json`, `PREDICTION_LOCK.json`, `TARGET_RELEASE.json` | Recorded fit, prediction, and terminal-target release links |
| `analysis/analysis_summary.json` | Complete analysis outcomes, phase hashes, and the original 23-table index |
| `analysis/primary_contrasts.csv`, `primary_paired_repetitions.csv`, `primary_difference_distribution.csv` | All six primary contrasts, the 192 paired repetition records, and descriptive difference summaries |
| `analysis/task_losses.csv`, `repetition_losses.csv`, `panel_summary.csv` | All task/view/arm losses and their repetition/panel aggregates |
| `analysis/condition_summary.csv`, `horizon_summary.csv` | Descriptive law/noise and horizon summaries |
| `analysis/optimization_summary.csv`, `optimization_repetition_pairs.csv` | Joint-versus-standard comparisons and paired repetition distributions |
| `analysis/secondary_descriptive.csv` | The retained descriptive contrast roster, including unfavorable outcomes |
| `analysis/selection_by_task.csv` | Exact task/view/arm validation choices, selected scaler/cap, feature dimensions, and selected penalties |
| `analysis/scaling_selection_frequencies.csv`, `cap_selection_frequencies.csv`, `joint_selection_frequencies.csv` | Scaler, cap, and joint selection frequencies |
| `analysis/resource_summary.csv` | Measured computation summaries, including shared work and rejected candidates |
| `verification/independent_audit_summary.json` | Passed production audit's top-level findings, numerical coverage, tolerances, and reference to the original full receipt |
| `reporting/numerical_archive_summary.json`, `numerical_archive_index.json.gz` | Final numerical ZIP identities and exact compressed full member index |
| `reporting/compact_release_manifest.json`, `compact_release_validation.json` | Lossless compact-release identities, checks, and restoration contract |
| `reporting/clean_clone_restoration_validation.json` | Subsequent isolated restoration of all three large compressed records, exact-byte checks, and idempotent second restoration |

In rows listing several names, the leading directory applies to all names in
that row. The full audit is deliberately omitted here: the included summary
removes only its task-detail payload and records the full original identity.
It is not a new audit execution. The member index describes the full numerical
archives; the archive payloads themselves are not present in this folder.

## Interpretation and scope

The primary family uses six one-sided Bonferroni-adjusted paired-t contrasts
with 32 independent repetition aggregates and 31 degrees of freedom. The first
four compare representations in a P-selected joint bank; the last two compare
independently selected joint K with standard-only K on the same cohort and
cached standard candidate fits. Relative gain is
`100 * (1 - mean(candidate) / mean(reference))`. A criterion is established
only when its margin-adjusted simultaneous upper bound is strictly negative.
The 3% superiority and 5% noninferiority margins remain the prespecified
research criteria. The two unmet superiority margins are retained.

Other independent-arm comparisons, per-law/horizon effects, selection
frequencies, and other paired intervals remain descriptive. Task cells,
windows, and horizons are not independent repetition units. A larger candidate
menu cannot worsen its selector's validation objective, but it can worsen
terminal loss. Raw observed-output forecasting does not establish sufficient
Markov states or autonomous soft-coordinate dynamics.

The audit independently checks numerical histories/targets, transforms,
reference geometry, readouts, forecast replay, losses, and bounds within its
recorded scope. It does not independently reimplement physical-law integration
or partition k-means fitting, and it is not external scientific replication.
Seals record procedural order; they do not supply cryptographic target isolation.

## Full records and restoration

Excluded payloads include raw B candidate scores, basis exports, C forecast
losses, feature-width exports, full audit/task details, model/forecast NPZ files,
and the five numerical ZIP archives. The retained source/analysis manifests
therefore refer to some files outside this reporting subset. The compact-release
manifest likewise describes large compressed records that remain in the full
study and are intentionally not copied here.

For numerical restoration and complete replay, use the study's
[reporting guide](../../../../extensions/soft_state_joint_selection_20261102/reporting/README.md)
in a complete repository checkout. In a standalone manuscript source archive,
that external study path is not included; use its immutable repository source
and the separately distributed numerical archives. The current archive summary
records availability without inventing download URLs.

Compiling the manuscript uses the supplied generated display files, not this
folder or the numerical archives. This evidence collection does not alter
compiler inputs. All 76 compact evidence files from the prior completed edition
`6c39a003e61ace058851ec6badca3e8dd13ef183` remain unchanged at their earlier paths.
Their identities were checked against the archived pre-joint-selection source
manifest before and after this folder was created.
