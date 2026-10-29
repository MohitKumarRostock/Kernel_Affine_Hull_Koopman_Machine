# Independent verification of the representation-focused revision

The numerical and provenance audits in this directory import no project or
experiment modules. They support the new learned-KAHM hierarchy study while
preserving its conditional inference scope and its disclosed execution event.

## Completed numerical reconstruction

`independent_hierarchy_audit.py` produced
`independent_hierarchy_audit.json` with a passing result. It independently:

- Replays 40 fresh calibration streams containing 163,840 observations and
  checks their separation from the earlier training, validation, calibration,
  and development benchmark seeds.
- Evaluates the frozen KAHM reconstruction formulas directly from portable
  numerical banks, then reproduces all 80 selected forecasts and partitions.
- Rebuilds feature distances and evaluates every one of 143,440 signed-edge
  feasible duals and primal predictions in extended precision, without running
  or importing the study's optimizer.
- Recomputes all 1,760 confidence endpoints, solves 480 stratum-mass linear
  programs independently, and reconstructs all 880 class decisions.
- Preserves 379 stopping-limit cases and their actual gaps. The maximum
  normalized gap is `6.409599374590513e-7`; the maximum primal feasibility
  residual is `2.1791940457056563e-16`.

There are 3,626,324 scalar comparisons. The independent feature calculation
differs by at most `3.34e-16`; signed dual SSE values differ by at most
`4.27e-14`. Lower confidence endpoints differ by approximately `1.01e-12`
because the study applies a downward numerical guard while the audit computes
the mathematical KL root directly. No decision changes.

For the independent-future regime only, the class optimum is known exactly:
the target is independent of every history representation and the constant
zero predictor is admissible. All 760 lower endpoints for that regime are
below its risk `0.4242004444444446`. This is a finite diagnostic on that
specific controlled regime, not a coverage proof or a claim about the
repairable classes' unknown optima.

An initial audit found that the default HiGHS dual tolerance did not resolve
the smallest conditional objective coefficients precisely. Positive scaling
of the objective and an explicit `1e-10` feasibility tolerance eliminated
the discrepancy; the optimization problem is unchanged. The completed audit
uses that independently conditioned LP calculation. No study endpoint or
data record was changed in response.

## Completed provenance reconstruction

`independent_provenance_audit.py` produced
`independent_provenance_audit.json` with a passing result. It verifies that the
only source-manifest mismatch is the auxiliary dense benchmark log, that all
executables/configurations/banks retain their frozen bytes, and that every
already-saved numerical output retains its pre-finalization hash. It checks
both preserved log versions and confirms that the frozen code references the
log only in its source inventory function. The original development-process
description was incorrect and remains explicitly corrected in the incident
record. Timings may include process contention and are not isolated benchmarks.

## Recheck

From `extensions/certified_forecasting`, first restore the two byte-identical
archived payloads from their checksummed parts, then use new audit output paths:

```bash
python scripts/restore_hierarchy_payloads.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python verification/jmlr_representation_revision/independent_hierarchy_audit.py \
  --results results/learned_kahm_hierarchy_v1 \
  --output verification/learned_hierarchy_numerical_recheck.json
python verification/jmlr_representation_revision/independent_provenance_audit.py \
  --results results/learned_kahm_hierarchy_v1 \
  --output verification/learned_hierarchy_provenance_recheck.json
```

The numerical check also applies to a fresh replay of this protocol. The
exception-specific provenance check applies to the retained v1 execution.
Numerical guards and extended precision do not constitute interval arithmetic.
Source hashes and recorded timestamps do not supply external attestation or
establish the independence assumptions for naturally acquired data.
