# A1: stable correction-head arithmetic

## Why a repeat was necessary

The original frozen study ran all 144 planned combinations and scored 387,900
forecasts. Its independent consistency audit passed 6,704,494 checks. The
prespecified scientific decision was nevertheless TECHNICAL_BLOCK because
four correction-head updates raised Cholesky errors. The harness retained
8,659 fallback forecasts and did not drop the failed runs. There were no
capacity violations.

The failures occurred in the small tier: fixed KAHM on recurring seeds
20261101, 20261102 and 20261103, and adaptive KAHM on recurring seed 20261103.
They occurred in the residual decoder's normal equations, not in the KAHM
kernel factorization. A diagnostic replay found severe subtractive
cancellation after large standardized feature rows left the bounded replay.
The intended ridge matrix remained positive definite when rebuilt from its
actual rows.

The full original attempt is preserved at `archives/initial_attempt.zip`:

* SHA-256: `6ec5bb7cae6336fb53d417b82d907c16b43fc25db2d99af32ce04647c20cdbe3`
* Original source commit: `3d374de47d694f6ae3267e4b6ac82787755eeca5`
* Original freeze SHA-256: `8b5b77100b285fff3a6cb81af3b1e3b13081771255fe898e2d6d39da6ab0473f`
* Contents: original source, protocol, configuration, prepared and regeneration
  inputs, forecasts, traces, summaries, analysis, decision, audit and licenses.

The ZIP was checked against every included file's SHA-256 before source edits.
The exact original code and records can be recovered without relying on the
amended implementation.

## Scope of repair

The correction head now solves its declared regularized least-squares
objective directly from its existing bounded replay. The objective remains

    min_(b,w) ||r - b*1 - F*w||_2^2 + alpha*||w||_2^2,

where r is y minus the CURRENT raw-ridge forecast on those replay rows and
alpha remains 1. The intercept is unpenalized. Stable linear algebra replaces
accumulated correction normal equations; this is not a new forecasting
objective or a tuned penalty. Every geometric arm receives the same repair.
All solve work remains inside the charged observation service interval.

No data, split, target chronology, scaler rule, geometry, reference update,
kernel bandwidth, penalty, feature clipping, mixing-weight grid, timing
allowance or advancement threshold is changed. The full 144-run roster is
repeated in fresh serial workers. Failed original runs are neither replaced
silently nor excluded from the retained first-attempt evidence.

## What this repair does not address

The large-tier recurring forecasts also contained extreme errors. Separate
fresh least-squares fits reproduced the largest KAHM and affine corrections.
Those bursts are genuine extrapolation under the existing feature-scaling
and adaptation policy, not merely the Cholesky defect. They are not grounds
for clipping predictions, altering normalization, or tuning against the
evaluation period during this amendment. They remain part of the development
decision.

## Status of the evidence

The first attempt's outcomes were known when this numerical repair was made.
The amended freeze documents code and rules before the repeat, but is not
independent confirmation or a claim that the second attempt was planned before
any result was seen. The same natural collections were already inspected in
earlier manuscript development. This repeat can support a transparent local
development decision; it cannot establish broad forecasting effectiveness.

Targeted regression tests, numerical diagnostics and the repeated independent
audit accompany the final report. The original and amended results must be
distinguished when reporting execution counts and conclusions.
