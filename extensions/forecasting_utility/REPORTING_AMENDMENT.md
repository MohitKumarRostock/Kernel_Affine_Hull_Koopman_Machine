# Reporting amendment: retained geometric fitting failures

This note was written after the complete frozen execution. It changes the
downstream reporting and artifact-completion checks, not the original model,
dataset roster, protocol, execution freeze, or any recorded outcome.

## What happened

The public checkpoint `954fda6bc6f4a6a5cce971dff897049bd1f1e8e0` preceded the
first full run. All 168 scheduled tasks were attempted. The 144 synthetic tasks
completed. Of the 24 archive tasks, 22 completed all eight methods and both
policies.

Both requested-capacity settings for the selected Car Parts series `T1324`
(`car_parts_monthly__series0`) failed in the four geometric procedures. This
training split has 23 windows and only seven distinct input histories. The
fixed bank requests a block of eight nonempty clusters. The inherited
implementation rejects this condition with:

> Fewer nonempty clusters than requested; no states were dropped

The concatenated KAHM method and the KAHM, affine, and locality corrections each
failed on those two tasks. Both geometric-selection policies therefore also
failed on those tasks. Naive, raw ridge, raw RBF, and the fixed tree control
returned forecasts for every archive task.

There are 1,336 successful standalone method forecasts among 1,344 scheduled
method fits, eight method-fit failures, and four policy failures among 336
scheduled policy decisions. Failure tracebacks, warnings, unsuccessful timed
nodes, and successful baseline forecasts remain in the original task records.

## What remains unchanged

No series is replaced, no input is imputed, no clustering rule is relaxed, no
model is refitted after observing the failures, and no failed method receives
an invented forecast or an uncharged fallback. The original scientific files
retain the hashes in `freeze.json`. The original analyzer records all available
prediction audits and returns `complete: false` for the study; that result is
preserved in `results/utility_v1/analysis/summary.json`.

The zero-correction option preserves a fitted raw forecast when the additive
procedure can be fitted. It does not make the geometric search total on every
training sample. This distinction is part of the reported result.

## Explicit reporting scope

The prespecified full-roster archive performance aggregate is **undefined** for
the geometric methods and their policies. It must not be silently replaced by
an available-case mean.

The downstream `reporting.py` adds the following transparently separated views:

1. The complete 144-task synthetic study and its 24 prespecified paired
   contrasts, which are unaffected by the archive fitting failures.
2. Per-method and per-source success/failure counts across all 24 archive tasks.
3. Full-roster, source-balanced archive performance for the four baselines
   that completed every task.
4. A **post-hoc common-support diagnostic** comparing all methods on the same
   22 successful archive tasks. It includes 11 series: three each from Weather,
   Solar Weekly, and COVID Deaths, and two from Car Parts. Sources retain equal
   weight; Car Parts now averages two series instead of three. This is a
   different descriptive estimand, not completion of the full-roster study or
   a new confirmation sample.
5. The recorded validation choices, timing components, and residual-error
   decomposition on their explicitly supported tasks.

No tests on time windows or population-domain claims are added. A separate
report directory preserves the original analyzer's incomplete-study status.

## Artifact checks

An artifact can contain a completely accounted experiment with model failures.
The revised delivery checks require all 168 attempts, exact frozen-input and
saved-array verification, all 1,336 successful standalone forecasts, explicit
accounting for the eight failed fits and four failed policies, and the
unchanged original incomplete-study summary. They do not relabel every model
as successful. The frozen transport producer's original success-only build
precondition is not used for this failure-aware artifact; its checksum-checked
restore format remains valid.
