# Confirmatory dynamical certificate summaries

These tables summarize the independently verified confirmatory Van der Pol
certificate campaign.

`mode_summary.csv` is the primary scientific summary. It reports the
finite-sample certificate and frozen-predictor distributions separately for the
matched-stochastic and deterministic evaluation laws at the valid fitted
operator-norm budget `kappa = ||B||_2`.

`budget_summary.csv` reports all four prospectively frozen norm budgets:
`1`, `||B||_2`, `sqrt(2)`, and `2||B||_2`. Predictor-specific interpretation
requires the frozen predictor to lie inside the corresponding budget. In
particular, the `kappa = 1` rows are diagnostic only when `||B||_2 > 1`.

`primary_replicates.csv` retains one row for each of the 64 independent
confirmatory replicate datasets at the fitted-norm budget. No replicate is
dropped.

The `certificate_L_*`, predictor-RMSE, `f_hat`, `s_hat`, and margin quantiles
are descriptive replicate quantiles using NumPy's linear interpolation. They
are not confidence intervals.

Rates use independent replicate datasets as trials. Their intervals are
two-sided pointwise 95% exact Clopper-Pearson intervals. They are not
simultaneous across modes, budgets, or metrics. Zero observed violations do
not establish a zero violation probability.

`positive_certificate` means `L_kappa_delta > 0`.
`empirical_lower_bound_coverage` means the retained frozen predictor satisfies
`RMSE >= L_kappa_delta` on that replicate. This empirical check is descriptive;
the certificate itself is the finite-sample lower bound derived under the
frozen i.i.d.-pair design.

`numerical_audit.json` records a fresh independent recomputation of all 64
confirmatory association datasets, hard labels, sufficient statistics,
predictor errors, certificate bounds, seed streams, and time-index draws.
No random dynamical pair dataset is regenerated, no representation is refit,
and the original campaign evidence is not modified.

The campaign execution commit, independent verifier commit, and later summary
source commit have distinct provenance roles and must not be interchanged.
