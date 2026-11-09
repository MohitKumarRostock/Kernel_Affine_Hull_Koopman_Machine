# Feature augmentation with reconstruction-based representations

Protocol `KAHM-SCORE-20261102-v1` evaluates KAHM, affine, and kernel-weighted averaging reconstruction-score representations to forecast future observations directly. In the primary analysis, each reconstruction-based procedure selects its maximum local dimension separately on validation data. Each method produces its own scores and normalized affinity weights. The experiment separately adds log normalization constants or affinity-weighted local coordinates to those weights. The predictors are linear ridge regression on the representation and ridge regression on the representation concatenated with Gaussian-kernel Nyström features from 64 landmarks. Both use the same observed targets.

The study contains 64 repetitions for each of four data-generating distributions. Training split A is used to fit candidate models. Validation split B selects settings separately for each complete procedure. Split C supplies diagnostic results, and test split D evaluates the selected predictors. The complete 40-comparison Bonferroni family has 16 lower candidate mean losses, seven higher losses, and 17 unresolved comparisons. Appending local coordinates to KAHM memberships improves prediction in six of eight combinations of distribution and predictor. All four nonlinear-predictor KAHM/kernel-weighted averaging comparisons are unresolved. Unresolved intervals are not equivalence findings; the paired-Student approximation is not distribution-free.

## Published records

| Record | Scope |
| --- | --- |
| [Primary comparisons](primary_contrasts.csv) | Exact released 40-row summary, including unfavorable and unresolved outcomes |
| [Protocol](protocol.json) | Experimental settings and comparison definitions |
| [Protocol source identity](PROSPECTIVE_FREEZE.json) | Source identifier recorded before data generation; not evidence of public preregistration |
| [Results release](RESULTS_RELEASE.json) | Recorded execution summary and file identifiers; some referenced execution files are unavailable |
| [Availability](AVAILABILITY.md) | Supplied data, supported checks and missing execution materials |

Run `python verify_primary_summary.py` with SciPy installed to check summary arithmetic and original byte identity. This is not a repetition-level or model-level audit. The [article and supplement](../../papers/score_enrichment/) contain all definitions, the full primary decision matrix, and the complete static tables.

The independently selected procedures are primary. In the secondary analysis, a shared dimension limit is selected by the affine reconstruction control with a linear ridge predictor. Its favorable outcomes do not replace primary comparisons. The study does not fit an autonomous transition model for the representation, establish Koopman closure, or demonstrate universal KAHM superiority.
