# Certificate-validation protocol

Status: DRAFT. No final experimental campaign is specified or claimed here.

## Baseline

- Baseline tag: `baseline-2026-10-02`
- Baseline commit: `9cf76095bf9b3f905b7966ac12865d63bc0be3aa`
- Development branch: `certificate-validation`
- Environment lockfile: `requirements-lock-arm64.txt`
- Historical experiments and evidence remain unchanged.

## Initial scope

Implement and validate the original finite-sample exclusion certificate
on the controlled four-state construction before expanding to dynamical
benchmarks. Alternative certificates require separate derivations, tests,
and clearly labeled results.

Environment verification is not validation of scientific results.
Earlier exploratory simulations inform planning but are not final evidence.

## Mathematical checks before simulation

- Transcribe the original certificate and document all assumptions.
- Use vector RMSE, without an additional per-coordinate normalization.
- Interpret the matrix-norm budget as the spectral/operator norm.
- Independently establish the optimal population prediction risk for each
  selected construction and norm budget.
- Include exact closure, detectable obstruction, and positive optimal
  error for which the original population certificate is inconclusive.
- Check deterministic examples and numerical edge cases before Monte Carlo runs.

## Sampling and configuration

The first study uses independent evaluation pairs with a fixed representation.
Correlated transitions must not be counted as independent observations.

Before the final campaign, commit a configuration specifying the construction,
sampling law, parameter grid, norm budgets, confidence levels, exclusion
tolerances, replicate counts, and random-number generator and seed policy.
Pilot and final campaigns must be separately identified.

Distinguish pointwise confidence claims from simultaneous claims.
Any certificate-based selection requires an appropriate correction or fresh
independent final evaluation data.

## Evidence retained

Retain per-replicate inputs or sufficient statistics that reproduce every
reported calculation, including state counts, class occupancies, empirical
certificate components, the final lower bound, and the analytical optimum.

Report empirical coverage, zero-certificate frequency, exclusion power,
and the gap to the known optimum. Include uncertainty intervals for estimated
probabilities. Document any numerical comparison tolerances.

A zero certificate is a valid inconclusive result, not an execution failure.

## Execution provenance

Commit executable code, tests, and configuration before the final campaign.
Run from a clean source commit and record that exact execution commit.

Record the environment, dependency-lockfile hash, commands, configuration
hashes, random seeds, and output checksums. Preserve every scheduled attempt,
including failures, logs, exit statuses, and separately identified retries.

A later commit that archives outputs must not be substituted for the commit
that generated them. Never overwrite an existing campaign's evidence.

## Reporting and reproduction

Generate tables and figures programmatically from retained replicate data.
Check that aggregation can be reproduced without rerunning the simulation.
Perform a small clean-checkout reproduction under the reference environment.

Add completed manuscript results to `reproduction/reported_results.tsv`.
Do not retroactively assign unverified execution provenance to historical results.
