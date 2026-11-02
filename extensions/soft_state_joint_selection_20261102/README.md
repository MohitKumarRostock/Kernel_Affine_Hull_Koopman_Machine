# Joint input scaling and native dimension selection

This separate follow-up extends the current G/P/K/R representation-only study
with A-fitted input scaling selected jointly with native dimension using B.
It preserves the completed predecessor experiment unchanged.

**Current status: all 256 production tasks, the frozen analysis, and the
independent production audit are complete.** The audit passed 7,422,883 checks.
The [results report](RESULTS.md) records all
six primary comparisons, all 16 panel forecasts, descriptive per-law effects,
and scaler/cap selections. Four of six criteria are established by the frozen
analysis. The nonlinear KAHM-over-RBF and joint-tuning superiority margins are
not established.

The scientific source was published before production preparation as
[`8e0250279074aefa39079fa33205d3c74f835848`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/8e0250279074aefa39079fa33205d3c74f835848),
on branch `research/kahm-soft-state-joint-selection-2026-11-02`.
[PUBLICATION.json](PUBLICATION.json) binds that actual public commit to the
local [source freeze](freeze.json). The earlier reviewed local commit is not
substituted for the published reference.

The full design is in [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json).
The 15 geometry candidates combine standardization, min-max, and tanh with
native dimension caps 1, 2, 4, 8, 12. Every candidate fits all four arms and the
same five ridge penalties. Shared and independent joint selection are compared
with the corresponding standard-only selections on the same fresh cohort.

The cohort contains 32 fresh independent repetitions, four inherited laws, two
paired added-noise levels, history width 12, and horizons 1, 4, and 12. All
4,096 task/view/arm forecast arrays and 23 analysis tables are retained. The
six primary comparisons include the four shared-bank structural questions and
paired independent joint-versus-standard KAHM comparisons. No C or D outcome
participates in tuning. Earlier cohorts are not pooled with this study.

## Findings and records

Shared-bank P improves nonlinear mean loss over G by 68.69%, and K improves P
by 8.86%; both establish the prescribed 3% margin. K has 0.9186% higher loss
than matched R and does not establish that superiority margin. Independent
joint K improves over same-cohort standard-only K by 0.0727%, which does not
establish the prescribed 3% gain. Linear loss increases by 1.26% for shared
K versus P and by 0.2491% for joint versus standard-only K; both satisfy the
5% noninferiority criteria. The independent audit reproduces all six primary
comparisons and their outcomes.

- [Complete results and interpretation](RESULTS.md)
- [Primary contrasts](analysis/primary_contrasts.csv) and [paired repetitions](analysis/primary_paired_repetitions.csv)
- [Panel losses](analysis/panel_summary.csv) and [per-law/noise losses](analysis/condition_summary.csv)
- [Joint-versus-standard comparisons](analysis/optimization_summary.csv)
- [Selections](analysis/selection_by_task.csv) and [scaler frequencies](analysis/scaling_selection_frequencies.csv)
- [Analysis summary and table identities](analysis/analysis_summary.json)
- [Passed independent audit summary](verification/independent_audit_summary.json) and [full compressed receipt](verification/independent_audit.json.gz)
- [Independent audit implementation and scope](audit/README.md)
- [Numerical packaging, restoration, and replay](reporting/README.md)

## Reproduction

Use the pinned [study requirements](requirements.txt) in a dedicated environment.
Preserve the original input, model, prediction, target, and analysis records.
After restoring the full numerical records, verify the study and reconstruct
statistics into a new directory from the repository root:

```bash
python extensions/soft_state_joint_selection_20261102/code/pipeline.py verify
python extensions/soft_state_joint_selection_20261102/analysis_code/analyze_soft_state.py \
  --output /tmp/KAHM_Joint_Selection_Reanalysis
python extensions/soft_state_joint_selection_20261102/audit/verify_soft_state.py \
  --analysis /tmp/KAHM_Joint_Selection_Reanalysis \
  --output /tmp/KAHM_Joint_Selection_Replayed_Audit.json
```

Full regeneration uses `code/pipeline.py` in the protocol's staged order:
published freeze, A/B/C preparation, fitting, global fit seal, D generation,
prediction, global prediction seal, target release, analysis, and independent
audit. Use a clean copy for regeneration and retain the original execution.
Reusing its seeds reproduces this cohort; it does not add new independent
repetitions. Compiling the supplied manuscript is a separate operation and
does not require fitting models.

The production audit independently refits and replays all 76,800 B penalty-path
arrays and replays every selected C and D forecast: 4,096 arrays on each split.
It checks all task losses, aggregate comparisons, scaler statistics, local
references, native dimensions, and kernel identities. There were no numerical
solver recoveries. The [results report](RESULTS.md) records maximum numerical
differences and the exact analysis/audit hashes. Physical-law integration and
partition k-means fitting are not independently reimplemented.

The ten model checks, 42 statistical preflight checks, and 39,324 staged
artificial audit checks concern manufactured data and remain separate from
the completed production audit. Internal numerical verification is not
external scientific replication. The finite grid selects among declared
scalers and caps; it does not globally optimize all KAHM parameters.
