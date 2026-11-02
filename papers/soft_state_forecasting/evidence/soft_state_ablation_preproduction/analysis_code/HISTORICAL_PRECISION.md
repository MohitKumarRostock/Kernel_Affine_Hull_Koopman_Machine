# Historical-only precision assessment

The production budget is fixed at **32 independent repetitions** before any
new production data are generated. It will not increase in response to B, C,
or D results. The assessment below describes scale; it is not a claim that the
new representation-only comparisons have known power.

For four one-sided Bonferroni Student bounds at family alpha 0.05, the n=32
critical value is **2.3555682821599135** (31 degrees of freedom). The allowance
above the observed mean margin difference is therefore **0.4164095764658054**
times its paired sample standard deviation. With n=24 it was
0.4894641979460952 times that standard deviation. Holding variance fixed, the
new allowance is 14.93% smaller.

The preceding preprocessing study's selected KAHM-versus-RBF nonlinear margin
difference had sample SD 0.0004358132702010258 and comparator mean SMSE
0.028292129119783228. Reusing only that historical SD for illustration gives a
32-repetition allowance of 0.00018147681926258678 SMSE, or 0.64144% of its
historical comparator mean. The preceding policy augmentation contrasts had
smaller historical allowances (approximately 0.36% and 0.50% of their
reference means), but their policies frequently forwarded the same forecast.
Those particularly small variances should not be assumed for a direct
representation ablation.

Under an independent Gaussian paired-difference planning model, a true
improvement beyond the specified margin of approximately 0.56828 paired SD
gives 80% power for an individual member of this family; 0.64951 SD gives 90%.
These values come from the noncentral t distribution with noncentrality
`sqrt(32) * standardized_margin_excess`. They concern one prespecified
contrast, not simultaneous success of all four. If the preprocessing
KAHM/RBF SD and reference mean applied, 80% power would require about 0.875
percentage points of improvement beyond the 3% margin. That is an illustration,
not a forecast for the new study.

The new experiment removes the raw-history baseline, changes block
transformations to preserve exact nesting, and selects the primary cap through
P. No historical gate-only arm supplies a variance estimate for P versus G.
Accordingly, the fixed budget is a feasible, moderately more precise follow-up,
not a validated power guarantee for all proposed mechanisms. Exact finite
sample t coverage requires paired normality; otherwise the intervals use the
stated approximation. No production observations enter this planning note.

## Reproducible historical sources

- `extensions/preprocessing_dimension/analysis/analysis_summary.json`,
  SHA-256 `b58944bd887fcd9ca96f56e8927378321ebfa0d1437c401ab5ee6c0b8678a735`,
  `primary_contrasts`, selected KAHM versus selected RBF.
- `extensions/policy_augmentation_20261101/analysis/analysis_summary.json`,
  SHA-256 `a0bf3226273454339247a2078166d8114be04efd7cc9e19d90e0dbed7cf7c473`,
  `primary_results`, nonlinear policy comparisons.

The companion script `historical_precision.py` recomputes this assessment from
those existing summaries and the fixed n=32/family-four allocation.
