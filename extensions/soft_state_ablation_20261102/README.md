# Predictive utility of geometry-induced soft-state abstractions

This study separates three contributions to a geometric forecasting
representation: soft memberships, position within their reference regions,
and nonlinear reconstruction departures. Every arm predicts the same future
observation with a direct affine ridge head. The scientific design is in
[PROTOCOL.md](PROTOCOL.md); the exact settings are in [config.json](config.json).

| Arm | Readout input |
| --- | --- |
| G | Separately transformed membership gates |
| P | G plus separately transformed gate-weighted local principal coordinates |
| K | P plus separately transformed KAHM departures |
| R | P plus separately transformed positive-kernel RBF departures |

The raw gates form a product of simplices. The transformed inputs and signed
enrichments need not be simplex-valued. KAHM and RBF share the memberships and
reference geometry within each matched bank. No arm retains a raw-history
forecast, residual target, or calibrated correction multiplier.

The fixed production cohort has 32 independent repetitions, four inherited
laws and two added-noise levels, yielding 256 tasks. History width is 12 and
horizons are 1, 4 and 12. A fits geometry and coefficients; B chooses penalties
and dimensions; C supplies diagnostics only; D supplies final evaluation.
The primary view uses P's B-selected cap for all arms. Independently selected
caps are secondary. There is one four-contrast family; neither diagnostic
subsets nor the secondary view can replace an unsuccessful primary result.

## Status and evidence

**Source-publication status:** the scientific source and fixed analysis are
frozen, and artificial checks are complete. This publication contains no
production data or forecasting outcomes. A genuine commit-bound
`PUBLICATION.json` is recorded after remote source verification and before
production preparation. An earlier rejected publication attempt is retained
as a dated historical record; the author subsequently authorized publication.
Completed results and verification will be linked after the fixed run.

A source freeze records the implemented procedure;
`PUBLICATION.json` binds that freeze to its verified GitHub commit before
production preparation. Stage seals document ordering, not access control or
external scientific replication.

The study was designed with knowledge of earlier results. Its fresh seeds add
independent repetitions under known laws, not new application domains. No
earlier observations are pooled. See
[analysis_code/HISTORICAL_PRECISION.md](analysis_code/HISTORICAL_PRECISION.md)
for the budget rationale and its limitations.

## Execution

Use a complete repository checkout containing the exact inherited sources
recorded in `freeze.json`. Install the versions in
[requirements.txt](requirements.txt) in a dedicated environment. Production
workers each use one numerical thread; the prescribed run uses three workers.
The checked inherited ridge solver records any NumPy-SVD to SciPy-gesvd
recovery, without changing the ridge objective or silently excluding tasks.

The following commands are run from the repository root. They use the
published source freeze and its genuine publication receipt. Do not create a
new freeze or invent a receipt to reproduce the original procedure.

```bash
python extensions/soft_state_ablation_20261102/code/pipeline.py abc
python extensions/soft_state_ablation_20261102/code/pipeline.py fit --workers 3
python extensions/soft_state_ablation_20261102/code/pipeline.py seal-fits
python extensions/soft_state_ablation_20261102/code/pipeline.py d
python extensions/soft_state_ablation_20261102/code/pipeline.py predict --workers 3
python extensions/soft_state_ablation_20261102/code/pipeline.py seal-predictions
python extensions/soft_state_ablation_20261102/code/pipeline.py release
python extensions/soft_state_ablation_20261102/analysis_code/analyze_soft_state.py
python extensions/soft_state_ablation_20261102/audit/verify_soft_state.py
```

The run is append-only: an occupied output is checked or rejected, rather
than overwritten. For a separate reproduction, use a clean checkout and move
only the released analysis and verification outputs to a clearly named
comparison directory before running their writers. Preserve the original
source, freeze, publication receipt, and released output hashes. Execution
timestamps and byte-level archive hashes can change across runs even where
numerical values agree; compare substantive arrays and reported tolerances.

The production order is enforced: D generation requires the complete fit
seal, and terminal analysis requires the full prediction seal and release.
Prediction workers receive query arrays only. D targets are physically present
after D preparation; these procedural checks are not cryptographic isolation.

## Retained records and audit scope

`prepared/` holds episode, noise, row and target records; `results/` contains
portable JSON/NPZ fitted states, all B candidate predictions, C diagnostics,
and numerical/resource logs. `predictions/` contains the eight logical D
forecast arrays per task and query diagnostics. Large numerical records are
kept in the separately indexed data archive rather than copied into Git.
Compact analysis tables, source identities and verification receipts remain
with this study. Public source is sufficient to regenerate the cohort.

The independent audit reconstructs histories and targets from saved base
episodes and noise, reconstructs the four feature maps from portable state,
checks native rank rules and exact shared-block identities, and refits every
B penalty path using a Cholesky calculation distinct from the production SVD.
It replays selected C and D forecasts and independently recomputes losses,
repetition aggregates and all four adjusted upper bounds. Its final receipt
states actual coverage and numerical tolerances. This computational audit is
not an independent replication by another research group.

## Connection to the manuscript

The current paper is in [../../papers/soft_state_forecasting/](../../papers/soft_state_forecasting/).
This experiment evaluates external-target predictive usefulness of the
representation hierarchy. It does not fit autonomous soft-coordinate
propagation or establish Markov sufficiency, Koopman invariance, or universal
forecasting superiority. Earlier dimension, residual and policy studies keep
their original procedures and comparison families; this study does not
relabel their results.
