# Four-state pilot summaries

Pilot only, not final manuscript evidence. Every scheduled replicate is retained.
cell_summary.csv has one row per fixed case and sample size. Probability columns
use independent replicate datasets as trials, not individual evaluation pairs.
exclusion_summary.csv has one row per cell and tolerance. The interpretation
column distinguishes exclusion power, false-exclusion rate, and numerical
boundary cases. A threshold is labeled `boundary` when the analytical optimum
and tolerance differ by at most the configured comparison_atol. This prevents
binary floating-point roundoff from being promoted into scientific ground truth.
The original saved raw-float comparison is retained in a separate audit column.

Probability intervals are two-sided pointwise 95% Clopper-Pearson intervals.
They are not simultaneous across cells, metrics, or tolerances. Zero observed
failures do not establish a zero failure probability. Certificate q05, median,
and q95 are descriptive replicate quantiles using linear interpolation, not
confidence intervals. Rates are fractions in [0,1]. Signed gaps are not clipped.
A blank certificate/optimum ratio means undefined because the optimum is zero.

Source/configuration/evidence identities are in summary_metadata.json. A fresh
independent numerical audit is retained in numerical_audit.json. No samples were
drawn and no original files or campaign summaries were modified. The execution
commit and the later summary source commit describe different operations.
A completed package requires summary_metadata.json and a valid SHA256SUMS.
