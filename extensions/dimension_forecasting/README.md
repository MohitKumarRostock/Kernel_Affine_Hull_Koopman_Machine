# KAHM forecasting with validation-selected local dimensions

This extension implements a staged ten-procedure forecasting comparison with
local KAHM dimensions selected on B, C-only blend calibration and policy
selection, and globally sealed models and held-out predictions. The study uses
the original four numerical observation laws and comparator grids, with 24
fresh repetitions, history length 12 and horizons 1, 4 and 12.

The completed study establishes three of the six declared primary comparisons.
The full joint performance criterion is not established. B-selected KAHM
improves over historical cap 2, while native cap 20 has higher mean nonlinear
error. The C-selected non-KAHM policy has lower aggregate error than KAHM.
See [FINDINGS.md](FINDINGS.md) for the complete ten-procedure table, simultaneous
margin tests, heterogeneity, numerical diagnostics and interpretation.
Numerical reporting and figure-generation entry points are described
in [reporting/README.md](reporting/README.md).

## Study definition and records

- [config.json](config.json) is the machine-readable design and full grid.
- [PROTOCOL.md](PROTOCOL.md) defines the data roles, selection, calibration,
  primary family and secondary comparisons.
- [analysis_code/ANALYSIS_PROTOCOL.md](analysis_code/ANALYSIS_PROTOCOL.md)
  defines the estimands, exports and inference.
- [freeze.json](freeze.json) records every frozen source/configuration hash.
- [analysis/](analysis/) contains all 27 exported tables and the analysis
  integrity receipt.
- [audit/post_release/final_results_audit.json](audit/post_release/final_results_audit.json)
  records the independent audit of all 192 tasks, 261,450 CSV rows and 18,816
  independently reconstructed local bases.

The seven independently fitted banks are historical2 and native1, native2,
native4, native8, native12 and native20. Each bank supplies common, bounded
KAHM, bounded RBF, unbounded KAHM and unbounded RBF heads. The four shared
procedures are ridge, raw Nyström RBF, HGB and persistence. A ten-procedure
context adds the C-selected non-KAHM policy. The contexts historical2 and
native20 are secondary; selected is the sole primary context. A separate
matched view uses bounded KAHM's selected bank for all five geometry heads.

The numerical runtime is preserved under [runtime/step1/](runtime/step1/) and
the unmodified observation generators under [vendor_original/](vendor_original/).
Their manifests and licenses accompany the source. The principal lineage is
commit `7dda356138675934aab5e327ec667bde0a5eda79` of
[Kernel_Affine_Hull_Koopman_Machine](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine).
The verified central dimension policy has SHA-256
`c03d9a1923b5d9c8ade8a4198fd06e2b24c6add9bf0e0992c469f90d7621efa2`.

## Environment

Run commands from `extensions/dimension_forecasting` in the repository source
tree. Install [requirements.txt](requirements.txt) into an appropriate Python
environment. The recorded execution used Python 3.12 with the pinned numerical
library versions. The worker runner uses Linux CPU-affinity APIs and one
numerical thread per fresh process. It accepts one to three workers and binds
each worker to a distinct CPU in the calling process's allowed CPU set.

All import and manifest paths are relative to the extension source tree;
the original working-directory path is not required. If CPU placement is
desired, invoke a stage with `taskset -c <allowed CPU IDs>`. Resource values
are descriptive shared-host measurements, not isolated speed benchmarks.

## Restore the complete numerical records

The numerical archives have this division:

| Archive | Repetition indices | Global records |
| --- | --- | --- |
| KAHM_Forecasting_Dimension_Records_01.zip | 0–3 | Included |
| KAHM_Forecasting_Dimension_Records_02.zip | 4–7 | None |
| KAHM_Forecasting_Dimension_Records_03.zip | 8–11 | None |
| KAHM_Forecasting_Dimension_Records_04.zip | 12–15 | None |
| KAHM_Forecasting_Dimension_Records_05.zip | 16–19 | None |
| KAHM_Forecasting_Dimension_Records_06.zip | 20–23 | None |

All members have the prefix `extensions/dimension_forecasting/`. Extract all
six archives at the repository root so that their contents restore under this
extension. The archives are disjoint. They contain every file in `prepared/`,
`results/`, `predictions/` and `execution/`, as well as the source freeze and
FIT/PREDICTION/TARGET seals. Source code and analysis tables are supplied by
the source-tree distribution rather than duplicated in the numerical archives.
The accompanying [archive manifest](documentation/KAHM_Forecasting_Dimension_Records_manifest.json) provides
archive SHA-256 values and every original member's size, SHA-256 and CRC32.
Each archive is below 500 MB.

The raw record layout is:

| Path | Content |
| --- | --- |
| prepared/base_abc/ | Original A/B/C base episodes and paired unit noise |
| prepared/abc/ | Retained A/B/C histories, targets and row identities |
| prepared/withheld_d_base/ | Reserved D base episodes and unit noise |
| prepared/queries/ | D histories and origin/group identities, without targets |
| prepared/targets/ | D targets and origin/group identities |
| results/TASK/ | Complete B/C evidence, actual bases, eight fitted states and fit summary |
| predictions/TASK/ | All 162 forecast views and fixed-state query diagnostics |
| execution/ | Per-task logs and fit/prediction stage receipts |

## Recompute the analysis from the sealed forecasts

After restoring all numerical records, run:

```bash
python code/study_seals.py verify --root .
python analysis_code/analyze_forecasting.py --root . --output verification/reanalysis
```

Use a new output directory for each replay. The analyzer refuses to overwrite
an existing result and validates the source, fitted-state, prediction and
target-release seals before reading terminal targets. It does not fit models
or choose any additional settings. The primary context is selected/calibrated;
historical2, native20, matched controls and gamma-one results are secondary.

## Reproduce the complete numerical execution

Use a clean copy of the frozen source tree without `prepared/`, `results/`,
`predictions/`, `execution/`, `FIT_LOCK.json`, `PREDICTION_LOCK.json`, or
`TARGET_RELEASE.json`. Preserve the existing source freeze and every file it
lists. The following commands reproduce the same declared seeds and design;
they do not create a new independent research sample.

```bash
python code/study_seals.py verify --root .
python code/prepare_dimension_data.py abc --root . --freeze-file freeze.json
python code/run_dimension_stage.py --phase fit --root . --freeze-file freeze.json --workers 3
python code/study_seals.py fits --root .
python code/prepare_dimension_data.py d --root . --freeze-file freeze.json --fit-lock FIT_LOCK.json
python code/run_dimension_stage.py --phase predict --root . --freeze-file freeze.json --fit-lock FIT_LOCK.json --workers 3
python code/study_seals.py predictions --root .
python code/study_seals.py release --root .
python analysis_code/analyze_forecasting.py --root . --output verification/reproduced_analysis
```

ABC generation requires the source freeze. B cap choices are durably saved
before C is loaded for calibration. All 192 fitted tasks must pass the global
fit seal before any D episode is generated. Prediction receives D histories
only, and every fitted-state hash must remain unchanged. Target release
requires the complete prediction seal. Unexpected failures stop the stage;
tasks or candidates are not omitted and existing attempts are not replaced.
Different machine timings and timestamps are expected in an execution replay.

## Construction and staging verification

The supplied receipts document the manufactured-input gates that preceded
production. To run them again, choose new fixture and receipt paths so that
the frozen evidence remains unchanged:

```bash
python code/verify_focused_adapter.py --fixture-dir verification/new_schema_fixture --receipt verification/new_adapter_check.json
python code/verify_staged_chain.py --fixture-dir verification/new_staged_fixture --receipt verification/new_staged_check.json
python code/verify_child_lock_optimization.py --fixture-root verification/new_staged_fixture --output verification/new_child_replay --receipt verification/new_child_check.json
```

These checks use manufactured inputs or separate small fixtures. They verify
the complete ten-procedure fixed-cap parity, B-only cap selection interface,
zero-multiplier identity, explicit failure handling, phase boundaries and
target-free prediction replay. Historical2/native2 and native12/native20
numerical identity and common reference sampling are additionally enforced
inside every production task. Independent audit programs are retained in
`audit/`; their original receipts remain distinct from any later replay.

## Prediction and tuning schema

Every production prediction archive contains 162 arrays of shape `(1664, 3)`.
The columns are horizons 1, 4 and 12. Array keys have the form
`VIEW__MODE__METHOD`, where modes are `calibrated` and `gamma1`:

- Each of the seven candidate views contains ridge and the five geometry
  methods.
- `shared` contains ridge, raw_rbf, raw_hgb and persistence.
- `context_historical2`, `context_native20` and `context_selected` contain
  all ten complete procedures.
- `matched` contains the five geometry heads at bounded KAHM's B-selected cap.

In the gamma-one policy view, the already C-selected method is evaluated with
multiplier one; the policy is not selected again. `pre_c_selection.json`
contains all five B cap choices and pre-C state hashes. The validation and
calibration NPZ files retain predictions used to verify the numerical scores;
`basis_diagnostics.npz` retains actual reference and basis arrays. The fit
summary records original raw tuning grids, requested/realized ranks, the
context mappings and C policy scores. Query summaries preserve feature guards,
readout bounds, state identities and per-context resource nodes.

## Scope

The conclusions concern these four numerical laws and this finite construction
and tuning protocol. They do not establish performance on natural-data cohorts,
general adoption suitability, or risk-certificate coverage. Local rank is not
the rank of the stacked feature representation. Independently selected RBF
comparisons are comparisons of complete tuning procedures; within-bank
mechanism statements use the separately declared matched view.
