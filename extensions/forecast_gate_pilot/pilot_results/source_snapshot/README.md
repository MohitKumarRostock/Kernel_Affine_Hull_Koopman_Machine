# KAHM research continuation workbench

Prepared 31 October 2026. This package turns the manuscript's proposed next
directions into executable development tools and a separate confirmatory
study design. It does not alter the supplied manuscript or reproduce the
original frozen experiments.

## What is included

- `geometry.py`: matched fixed and directly forecast-trained KAHM, affine,
  locality and RBF-local gates, plus raw ridge and raw-input-plus-Nystrom RBF.
- `run_pilot.py`: numerical-law generators, episode-separated A/B/C/D splits,
  an automatic pre-run source/configuration freeze, saved inputs/predictions,
  and complete nine-task exploratory evaluation.
- `audit_pilot.py`: loss, selection, hash, split, roster and cost-dependency
  checks on the supplied saved evidence.
- `online.py`: fixed-feature bounded sliding-window ridge and cached KAHM
  reconstruction, with explicit geometry-version guards.
- `test_geometry.py`, `test_online.py`, and their check records: numerical
  correctness checks, not an accuracy or production-latency benchmark.
- `CONFIRMATORY_PROTOCOL_DRAFT.md`: the next source-reserved comparison,
  streaming protocol, failure rules and decision criteria. Not registered
  or executed; application cost limits and reserved source roster remain open.
- `attachment_integrity.json`: independent checks of all 118 files listed in
  the supplied manuscript-source manifest. This is file integrity, not a
  reproduction of the original paper's 768 forecast outputs.
- `pilot_results/`: the frozen exploratory run, including source snapshots,
  configuration, raw episodes, every selected C/D forecast, candidate losses,
  optimizer evaluations, diagnostics, timing records and audit.

## Scope of the new experiment

The narrow hypothesis is whether training the final forecast through gate
temperatures helps KAHM beyond fixed KAHM and equally trained alternatives.
The reconstruction bank, input standardizer, references, PCA and kernel
regularizer remain fixed. This is direct forecast-trained gating, not
end-to-end learning of the KAHM reconstruction operator or a new input metric.

All four geometries receive three bounded log inverse-temperatures per
horizon, initialized to zero, and the same hard limit of 160 actual objective
evaluations per horizon/penalty. A ridge correction is refitted on A at each
evaluation; the objective is actual forecast MSE on B. C selects head penalty
and mixing separately by horizon, with zero first and strict improvement.
The ridge baseline is selected within A and then held fixed. The API does
not accept D targets. B is geometry-training data, not confirmation.

The study uses nonlinear AR, Van der Pol and Duffing forms documented in
the manuscript; three new seeds per law; 64/32/32/64 independent episodes
in A/B/C/D; 72 observations; history12; offsets1/4/12; 13 origins per episode;
row caps512/256/256/unlimited. D has832 rows per task. These are known laws
with new random streams and a new implementation, not new-source evidence.
The first ODE observation is explicitly t=10.0, followed by0.1-unit spacing;
the original execution code was unavailable for a bitwise replication check.

The raw RBF reference in this package concatenates raw standardized history
with Nystrom features. It is not an identical reimplementation of the
historical raw RBF procedure. Its landmarks are from A and its bandwidth/
penalty are selected on B. All reported comparisons concern this new suite.

## Run and audit

Use Python3.12 or a compatible version with NumPy, SciPy and scikit-learn.
Exact tested versions are recorded in `pilot_results/protocol.json`.
The runner sets common numerical-library thread variables to1 before imports.

```bash
python test_geometry.py
python test_online.py
python run_pilot.py --quick --output ../kahm_quick_new
python run_pilot.py --run --output ../kahm_pilot_new
python run_pilot.py --audit --output ../kahm_pilot_new
python run_pilot.py --audit --output pilot_results
```

Output directories for new runs must be absent or empty. The preserved
run should be audited rather than overwritten. Repeated fitting can show
small platform-dependent floating-point and timing differences.

The audit reconstructs every saved input window from raw observations,
recomputes C/D losses, checks full method and candidate rosters, replays
strict first-minimum decisions, verifies exact zero-correction outputs,
and sums recorded cost dependencies. It does not independently refit every
candidate or certify that the method generalizes. All exploratory methods,
seeds and failures remain visible; no confidence interval is attached to
the three-repetition means.

## Bounded-update module

`BoundedWindowRidge` in `online.py` is a generic numerical building block.
It minimizes sum squared errors plus a fixed positive ridge penalty on
every supplied feature, including an appended constant. That intercept
convention differs from `geometry.py`, which uses an unpenalized intercept.
It is not silently interchangeable with the static forecaster.

Updates maintain capped-window sufficient statistics via rank-one additions
and deletions, with logged bounded replay on numerical trouble. A geometry
version must accompany operations. Refreshes require reset or explicit
replay under the new map. The module does not update PCA, references or
metrics and does not implement a complete delayed-label forecasting stream.
Read the module docstring and confirmatory protocol before integration.

## Interpretation and provenance

The supplied manuscript established conditional gains against particular
baselines and documented stronger matched controls and costs. The workbench
does not assume those limitations have been resolved. Its exploratory output
must be followed by a separate frozen evaluation on genuinely reserved
sources or periods before a predictive-contribution claim.

The source ZIP contains the editable manuscript and generated displays,
not the referenced reproduction ZIP or experimental Python source. A search
did not locate the companion archive, and the linked repository was not
retrievable in this environment. The present code was consequently written
from the equations and protocol descriptions. The attachment's original
files and reported historical outcomes are preserved unchanged.

Supporting precedents: Bertinetto et al. (ICLR2019), differentiable closed-form
ridge solvers; Van Vaerenbergh et al. (ICASSP2010), fixed-budget kernel RLS;
Cawley and Talbot (JMLR2010), model-selection bias. Generic differentiation
through ridge and bounded kernel updates are not claimed as novel here.

The potential contribution is a demonstrated KAHM-specific improvement under
matched forecast supervision, reserved evidence, and complete cost accounting.
