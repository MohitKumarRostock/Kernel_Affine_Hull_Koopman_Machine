# Final four-state certificate summaries

These tables summarize the final controlled certificate-validation campaign.

`cell_summary.csv` contains one row per final design case and evaluation sample
size. It reports the finite-sample certificate distribution, population-bound
slack, sampling gap, zero-certificate frequency, and empirical coverage.

`exclusion_summary.csv` contains one row per case, sample size, and tolerance.
Scientific threshold relations use the exact rational p metadata frozen in the
final configuration. They do not use binary64 near-equality heuristics.

`exact_optimum_relation` describes whether the true optimal RMSE is above,
below, or exactly on the tolerance.

`population_limit_relation` separately describes whether the limiting
population certificate is above, below, or exactly on the tolerance. This
distinguishes structural non-detectability from finite-sample underpower.

`interpretation` is based on the exact optimal error:
- exclusion_power
- false_exclusion_rate
- boundary_exclusion_rate

`population_detectability` is based on the exact population certificate limit:
- population_bound_can_exclude
- population_bound_cannot_exclude
- population_bound_boundary

Probability intervals are two-sided pointwise 95% Clopper-Pearson intervals
over independent replicate datasets. They are not simultaneous across cells
or tolerances. Certificate q05, median, and q95 are descriptive replicate
quantiles, not confidence intervals.

Zero observed failures do not imply a zero failure probability. All zero
certificates, signed gaps, coverage outcomes, exclusions, and exact threshold
boundaries are retained.

`numerical_audit.json` is a fresh independent recomputation of the complete
retained final evidence. No random samples are regenerated. Original evidence
is not modified.
