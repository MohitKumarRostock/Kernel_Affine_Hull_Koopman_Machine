# Reproducing the KAHM forecasting studies

This repository contains experiments with distinct source checkpoints, targets, comparison families, and computational environments. The current paper is under [papers/soft_state_forecasting/](papers/soft_state_forecasting/); the new joint-selection study is under [extensions/soft_state_joint_selection_20261102/](extensions/soft_state_joint_selection_20261102/). The [research index](docs/RESEARCH_INDEX.md) separates current entry points from preserved earlier studies. Use the environment and execution guide for the study being reproduced; the root Python requirements belong to the original KAHKM architecture.

**Joint-selection status: all 256 production tasks, the frozen analysis, and the independent production audit are complete.** The scientific source is [commit `8e025027`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/8e0250279074aefa39079fa33205d3c74f835848). The analysis and audit establish four of six criteria. The [complete results](extensions/soft_state_joint_selection_20261102/RESULTS.md) identify the completed evidence and the passed independent verification.

## Current joint scaling and native-dimension selection

The [publication receipt](extensions/soft_state_joint_selection_20261102/PUBLICATION.json) records actual source commit `8e0250279074aefa39079fa33205d3c74f835848`, tree `1946da072f230244ad92795f9517ef89d1f06487`, and branch `research/kahm-soft-state-joint-selection-2026-11-02`. All 19 extension file identities matched the approved source, and the public commit tree matched the reviewed tree. Publication was verified before production preparation. The separate local reviewed commit is retained as provenance and is not substituted for the actual publication.

The fixed experiment contains 32 new independent repetitions, 256 tasks, 15 scaler/cap candidates, and 16 forecast views/arms per task. Standard-only views reuse the same standard candidate fits on this cohort. All fitted states were sealed before D generation, and all predictions were sealed before target release and scoring. The [complete analysis](extensions/soft_state_joint_selection_20261102/analysis/analysis_summary.json) contains 4,096 terminal forecast arrays and 23 CSV tables, with source and phase digests. Its SHA-256 is `1345429169d0bd263d17a54fb469718a768a0be218db18059b0084176d2421e9`.

Four shared-bank structural comparisons and the two independent joint-versus-standard K comparisons comprise one six-criterion family. The frozen analysis establishes shared nonlinear P/G and K/P superiority and both linear noninferiority criteria. Neither shared K/R superiority nor independent K joint-tuning superiority is established. Independent K/R, per-law comparisons, and selection-frequency summaries remain descriptive. This family is separate from every predecessor study.

Use the [study README](extensions/soft_state_joint_selection_20261102/README.md), its pinned requirements, and the [reporting/restoration guide](extensions/soft_state_joint_selection_20261102/reporting/README.md). Preserve the original numerical records. After restoring complete records, reanalysis and independent replay use new output paths:

```bash
python extensions/soft_state_joint_selection_20261102/code/pipeline.py verify
python extensions/soft_state_joint_selection_20261102/analysis_code/analyze_soft_state.py \
  --output /tmp/KAHM_Joint_Selection_Reanalysis
python extensions/soft_state_joint_selection_20261102/audit/verify_soft_state.py \
  --analysis /tmp/KAHM_Joint_Selection_Reanalysis \
  --output /tmp/KAHM_Joint_Selection_Replayed_Audit.json
```

The [independent production audit summary](extensions/soft_state_joint_selection_20261102/verification/independent_audit_summary.json) records 7,422,883 passed checks across all 256 tasks. It independently refits and replays all 76,800 B penalty-path arrays and replays every selected C and D forecast, 4,096 arrays on each split. It verifies all six primary bounds, A-fitted scaler statistics, retained histories and targets, local references, native dimensions, and kernel identities. No numerical solver recoveries occurred. Maximum absolute forecast differences are 1.127675730572264e-10 on B paths, 2.4236168627567167e-12 on C, and 6.276451580689013e-12 on D, within frozen tolerances of relative 2e-7 and absolute 2e-9. The [full compressed receipt](extensions/soft_state_joint_selection_20261102/verification/independent_audit.json.gz) restores exact JSON with SHA-256 `8ba3931c3e705838d9f3ec924f374b8cb9e492aa6f90803684412d8151a9c89c` and binds the exact analysis digest above.

Physical-law integration and partition k-means fitting are not independently reimplemented. Manufactured-data checks are separate implementation evidence. This internal numerical audit is not external scientific replication. Stage seals and publication record procedural order; they are not public preregistration or cryptographic target isolation. Reusing the declared seeds reproduces this sample, without adding independent repetitions. Fresh realizations of the inherited laws do not create new application domains.

## Predecessor standard-only G/P/K/R representation ablation

The published source is identified by [PUBLICATION.json](extensions/soft_state_ablation_20261102/PUBLICATION.json): commit `2b7d19b5b47f46b5b21b28d24960688f7969d96f`, published on 2 November 2026 at 09:20:24 UTC. All 25 published study-file identities were checked against the exact local source. The receipt binds source publication to the local freeze before production preparation. All 256 A/B-fitted task states were sealed before D generation; terminal forecasts were sealed before scoring. The complete analysis contains 2,048 task/view/arm forecast arrays, 32 independent repetition aggregates and 17 CSV tables. The independent production audit has passed all 256 tasks. Primary comparisons share a P-selected bank; independently selected caps are secondary. The four-criterion family does not pool earlier repetitions or natural-series observations.

Use the [study directory](extensions/soft_state_ablation_20261102/) for the protocol, source freeze, publication receipt and execution commands. The [analysis summary](extensions/soft_state_ablation_20261102/analysis/analysis_summary.json) records all four primary contrasts, phase hashes and table identities. The analysis establishes P/G and K/P nonlinear superiority and K/P linear noninferiority, while leaving K/R nonlinear superiority unestablished: three of four criteria. The [independent production audit](extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json) records 2,563,814 checks, independently refits and replays 25,600 B penalty-path arrays, and replays all 2,048 C and 2,048 D forecast arrays. It recomputes the primary losses and bounds and checks local references, native dimensions and kernel identities. No numerical solver recoveries occurred. Maximum absolute forecast differences are 1.20209e-11 on B paths, 2.42889e-12 on C and 5.58575e-12 on D. Physical-law integration and partition k-means fitting are not independently reimplemented; this is an internal numerical audit, not external scientific replication. Earlier artificial receipts concern manufactured data and source review; they do not verify this production cohort. Publication and seals document procedural order, not independent preregistration or cryptographic target isolation. Fresh seeds under known laws extend sampling evidence, not application domains.

Preserve the original input, fit, prediction, target, and analysis records when replaying a completed study. Use separate output directories or a clean checkout for regenerated files. Earlier terminal results informed the new design; the new comparison therefore requires its own reserved evaluation.

## Earlier dimension-selected forecasting experiment

The [dimension-study README](extensions/dimension_forecasting/README.md) is the entry point for the earlier principal ten-procedure, 192-task-suite experiment. It preserves the six-comparison primary family, of which three criteria were established. Its evidence includes prepared inputs, fitted and candidate states, validation/calibration decisions, terminal forecasts, analysis tables and execution records.

Use the pinned Python 3.12 environment described in that guide. From the repository root:

```bash
python -m pip install -r extensions/dimension_forecasting/requirements.txt
```

Restore all six `KAHM_Forecasting_Dimension_Records_01.zip` through `_06.zip`
archives at the repository root. Their checkout-relative paths restore under
`extensions/dimension_forecasting/`; the [archive manifest](extensions/dimension_forecasting/documentation/KAHM_Forecasting_Dimension_Records_manifest.json)
records membership and checksums. Then verify the seals and reconstruct the
analysis using a new output directory:

```bash
cd extensions/dimension_forecasting
python code/study_seals.py verify --root .
python analysis_code/analyze_forecasting.py --root . --output verification/reanalysis
```

The [independent final audit](extensions/dimension_forecasting/audit/post_release/final_results_audit.json)
covers all 192 tasks, 31,104 retained forecast arrays, 18,816 local PCA bases
and 27 analysis tables. Full staged regeneration and fitting commands are in
the study README. Run a new numerical execution in a clean copy of the frozen
source, preserving the reported evidence. Reusing the declared seeds reproduces
that sample; it does not create new independent repetitions.

## Preprocessing and residual-target follow-ups

| Study | Reproduction entry point | Preserved inference and evidence |
| --- | --- | --- |
| Preprocessing and local dimension | [Completed package and restoration guide](extensions/preprocessing_dimension/RESULTS_PACKAGE.md), [frozen execution guide](extensions/preprocessing_dimension/README.md) | Four primary comparisons, one established; a separate secondary disjoint-episode fixed-common mechanism study |
| Cross-fitted and in-sample residual corrections | [Execution and provenance guide](extensions/crossfit_residual_reproduction_20261101/README.md), [completed results](extensions/crossfit_residual_reproduction_20261101/reporting/RESULTS.md) | Original four primary comparisons, two established; same 24 original cross-fit units and zero additional independent repetitions |

The preprocessing package contains one analysis ZIP and twelve disjoint records
ZIPs. Its [archive index](extensions/preprocessing_dimension/reporting/archive_index.json)
and restoration guide specify their layout. The extension depends on the
unchanged `dimension_forecasting` runtime. Its [audit](extensions/preprocessing_dimension/audit/final_results_audit.json)
checks all aggregate arithmetic and the primary comparisons; raw
forecast-to-loss reconstruction covers the prespecified eight repetition-zero
tasks, not every task. Retain that distinction when describing verification.

The cross-fit reproduction package contains one analysis ZIP and sixteen
standalone records ZIPs. Its [exact manifest](extensions/crossfit_residual_reproduction_20261101/reporting/ARCHIVE_MANIFEST.json)
and [delivery verification](extensions/crossfit_residual_reproduction_20261101/reporting/archive_delivery_verification.json)
identify the retained numerical bytes. The [independent reproduction audit](extensions/crossfit_residual_reproduction_20261101/verification/reproduction_audit.json)
covers all 192 tasks, 8,064 forecast arrays and 24,192 horizon losses. Restore
the checkout-relative members over the pinned source. Preserve the sibling
`dimension_forecasting`, `preprocessing_dimension` and `crossfit_residual_v2`
directories required by its runtime and provenance audit.

The cross-fit model, pipeline, solver and scientific analyzer were recovered
from published source and retained without scientific changes. The original
completed numerical records were unavailable, and prior reported outcomes
were known. Regenerating the same streams under newly recorded gates verifies
the new execution; it cannot authenticate the missing original run's bytes,
audit or chronology. The new freeze is not a new prospective preregistration.
In-sample residual comparisons remain secondary, and their post hoc mean
comparison with the expanded policy does not add a primary claim.

## Fresh policy augmentation and natural-series evaluation

The [policy-augmentation execution guide](extensions/policy_augmentation_20261101/README.md)
specifies the new 24-repetition, 192-task simulation experiment. Its
[completed results](extensions/policy_augmentation_20261101/RESULTS.md) retain
four prespecified criteria: neither nonlinear 3% superiority criterion is
established, and both linear 5% noninferiority criteria are established.
These new repetitions are not pooled with the previous studies.

The [natural reproduction guide](extensions/policy_augmentation_natural_20261101/REPRODUCTION.md)
specifies a separate 48-series chronological evaluation using pinned M4
training files, with no official Test-file evaluation. It preserves a
deterministic roster, local source freeze, stage seals and independent
numerical verification. Natural contrasts remain descriptive.

Restore the analysis ZIP and every records ZIP listed in the
[archive index](extensions/policy_augmentation_20261101/reporting/ARCHIVE_INDEX.json)
at the repository root. The full archive manifest records every member's
size and SHA-256, and the archive README specifies verification and replay.
The immutable combined scientific-source reference is commit
`b1915813e98a59572a4bbb800adc2bd1f3fccc1b`. The synthetic source was published
before D generation; natural source was locally frozen before parsing new
values and published after evaluation. Neither chronology is a public
preregistration.

The [synthetic audit](extensions/policy_augmentation_20261101/verification/policy_audit.json)
checks all 192 tasks, 9,600 saved forecast arrays and four primary bounds.
The [natural audit](extensions/policy_augmentation_natural_20261101/audit/verification.json)
checks 48 series, 2,016 terminal arrays, B/C choices, policy forwarding and
selected ridge normal equations. Each audit states which inherited feature
maps and numerical components it reuses. Preserve all sealed records,
including the explicitly sealed noncanonical filenames identified in the
archive manifest. Reporting scripts are subsequent to the scientific freeze.

## Historical fixed-cap reconstruction-departure experiment

The [reconstruction-mechanism guide](extensions/reconstruction_mechanism/README.md)
retains the earlier fixed-cap ten-procedure, 192-task experiment. Its original
numeric supplement, verification programs and eight-table analysis have that
historical scope. They are not substitutes for the dimension-selected
principal study or either follow-up.

Use Python 3.12 and install:

```bash
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
```

From the repository root, restore the lossless historical evidence and verify
its scientific checkpoint:

```bash
python -m extensions.reconstruction_mechanism.evidence unpack
python -m extensions.reconstruction_mechanism.run verify
```

Recompute its analysis in a separate output directory and check it with the
independent evaluation implementation:

```bash
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-analysis
python -m extensions.reconstruction_mechanism.audit \
  --analysis-json /tmp/kahm-analysis/decision.json \
  --output /tmp/kahm-audit.json
```

The historical study guide also describes selected-state replay and refitting.
A new execution must use a separate results directory or disposable checkout.
The original evidence is retained so that repeated computation can be
distinguished from the reported experiment.

### Historical saved-forecast analysis and numerical replay portability

The anonymous fixed-cap saved-forecast supplement reconstructs all 192 task suites and its complete primary statistical analysis directly from numeric forecast arrays and A-target variance scales. This operation is separate from evaluating serialized fitted states. Its successful verification does not establish that every processor reproduces each selected-state prediction within the declared pointwise tolerance.

Selected-state replay compares predictions using relative tolerance `1e-10` and absolute tolerance `1e-12`. The reference Linux environment uses Python 3.12.14 and glibc 2.39, with the pinned numerical packages. Processor-dependent vectorized kernels can still affect the comparison. In a targeted test, disabling NumPy AVX512 dispatch changes 49 of 4,992 raw-RBF predictions on `vanderpol__noise0__seed946100` beyond the original tolerance, with maximum absolute difference `1.0663936400590046e-10`; default dispatch reproduces those raw-RBF predictions exactly. The [diagnostic record](extensions/reconstruction_mechanism/publication_validation/NUMERICAL_REPLAY_PORTABILITY.json) specifies the reproducible invocation and identifies the original source arrays and fitted state. The strict replay criterion concerns numerical agreement between model evaluations and archived predictions. The reported statistical analysis uses the archived predictions directly.

## Evidence levels

| Operation | What it checks |
| --- | --- |
| Hash verification | The source, inputs, and stored outputs have the recorded bytes. |
| Analysis reconstruction | The reported losses, aggregates, and statistical contrasts follow from the saved forecasts. |
| Selected-model replay | The saved fitted state reproduces its recorded forecasts in a specified numerical environment. |
| Independent-code evaluation | A separate implementation evaluates saved states or statistics consistently with the original implementation, within its stated coverage. |
| Refitting | Re-execution of the specified fitting and selection procedure yields reproducible results, subject to numerical and timing variation. |
| Same-seed computational reproduction | A newly retained execution follows the same source, design and simulation streams; it adds no independent sampling evidence and does not authenticate unavailable earlier outputs. |
| External replication | A separate research team reproduces or independently tests the scientific findings. This is not claimed by the repository's internal checks. |

A successful checksum or replay does not, by itself, validate a scientific interpretation. The manuscript supplies the assumptions, comparison design and inferential scope needed to interpret the verified outputs. The earlier six-comparison dimension family, each follow-up family, the four-comparison standard-only ablation, and the new six-comparison joint-selection family remain separate. A favorable secondary result does not replace an unmet primary criterion.

## Other forecasting studies

| Study | Instructions |
| --- | --- |
| Controlled representations and decoder access | [Reproduction guide](extensions/predictive_geometry/REPRODUCING.md) |
| Natural-data evaluation | [Study README](extensions/natural_forecasting/README.md) |
| Input-scaling sensitivity | [Study README](extensions/scaling_sensitivity/README.md) |
| Additive geometric corrections | [Study README](extensions/forecasting_utility/README.md) |
| Supervised metric and donor residuals | [Study README](extensions/target_geometry/README.md) |
| Direct forecast training | [Study README](extensions/direct_reconstruction/README.md) |
| Online adaptation | [Study README](extensions/adaptive_development/REPRODUCING.md) |

The corresponding protocols specify random seeds, chronology, selection rules and failure handling. Raw forecasts and fitted states are retained where described. Some older audits verify a complete historical repository checkpoint, including its documentation. Those audits require that historical checkout or its corresponding archive; they should not be interpreted as checks of the present manuscript edition. Current scientific-source verification uses the study-specific frozen dependencies.

Numerical package versions alone need not guarantee bitwise agreement. The direct reconstruction study documents CPU-dispatch sensitivity in Duffing generation and one raw-RBF selected-state replay. Such differences are reported against the original tolerances rather than treated as successful exact reproduction. Timing measurements depend on hardware, numerical libraries and scheduling.

## Original KAHKM experiments

The original architecture's result-to-code mapping is [reproduction/reported_results.tsv](reproduction/reported_results.tsv). The map identifies generators, runners, settings, seeds, retained per-run outputs and derived artifacts. Entries marked `PASS` identify retained reported results with a reproducibility chain; entries marked `REMOVE` identify legacy outputs excluded from that reported set.

Its reference platform is native Apple Silicon macOS on an Apple M1, using Python 3.14. The direct dependencies are in `requirements.txt`; the resolved environment is in `requirements-lock-arm64.txt`. Platform metadata are recorded under `provenance/reference_arm64_environment/`.

On that platform:

```bash
bash scripts/bootstrap_arm64.sh
.venv-arm64/bin/python scripts/verify_environment.py
```

The full evidence summary is [reproduction/REPRODUCIBILITY_EVIDENCE.md](reproduction/REPRODUCIBILITY_EVIDENCE.md), with a machine-readable counterpart in `reproduction/reproducibility_evidence.json`. The historical Duffing reference search has a documented gap in run accounting. A separate prospective 27-job campaign records all scheduled attempts, exit statuses, logs and output checksums. These records support complete attempt accounting for that campaign, not for every historical execution in the repository.

## Manuscript reproduction and reporting provenance

The canonical current source is [papers/soft_state_forecasting/](papers/soft_state_forecasting/). Its `main.tex` and `supplement.tex` use supplied generated displays. With the documented TeX environment, compile from the repository root:

```bash
cd papers/soft_state_forecasting
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The outputs are `main.pdf` and `supplement.pdf` in that directory. GitHub retains those PDFs and the individual editable files in the [source directory](papers/soft_state_forecasting/), with the [build guide](papers/soft_state_forecasting/README_BUILD.md). A standalone source ZIP may be delivered separately; it is not an additional repository artifact. Compiling supplied tables and figures does not require scientific refitting or numerical-record archives; regenerating their values requires the corresponding study evidence. Follow the paper's `README_BUILD.md` for its actual dependencies and reporting commands.

The preserved [earlier manuscript guide](extensions/tmlr_integration/README.md), [historical build guide](extensions/tmlr_integration/manuscript/README_BUILD.md), and [coordinated release record](extensions/tmlr_integration/review/COORDINATED_MANUSCRIPT_RELEASE.json) describe the earlier edition. Its source remains at `extensions/tmlr_integration/manuscript/`; it is not the current build entry point.

Manuscript reporting changes do not alter frozen study code, configurations, protocols, analyzers, numerical records, or phase seals. The standard-only G/P/K/R ablation retains its completed execution and passed independent audit. The new joint-selection follow-up has completed its frozen analysis and passed its independent production audit. Historical packaging utilities and saved-forecast verifiers retain their original schemas. Document builds and numerical audits are distinct evidence, and TMLR review formatting makes no claim of submission or acceptance.

The predecessor four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from that study's frozen scientific source and manuscript edition.
