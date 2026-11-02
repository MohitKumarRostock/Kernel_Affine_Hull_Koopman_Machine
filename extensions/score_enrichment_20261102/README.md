# Reconstruction-score enrichment study

Protocol `KAHM-SCORE-20261102-v1` evaluates KAHM, affine, and positive-kernel reconstruction-score representations for direct observed-response prediction. Geometry is matched at a fixed native cap, but each reconstruction produces its own scores and memberships. Normalizers and gated local coordinates are parallel enrichments. Linear ridge and a 64-landmark Gaussian Nyström decoder use the same observed targets.

The study contains 64 repetitions for each of four laws. A fits candidates, B selects independently within each complete procedure, C supplies diagnostics, and D evaluates fixed predictions. The complete 40-comparison Bonferroni family has 16 lower candidate mean losses, seven higher losses, and 17 unresolved comparisons. Local enrichment improves KAHM memberships in six of eight law–decoder cells. All four nonlinear-decoder KAHM/positive-kernel comparisons are unresolved. Unresolved intervals are not equivalence findings; the paired-Student approximation is not distribution-free.

## Published records

| Record | Scope |
| --- | --- |
| [Primary comparisons](primary_contrasts.csv) | Exact released 40-row summary, including unfavorable and unresolved outcomes |
| [Protocol](protocol.json) | Original fixed configuration; historical title and status preserved |
| [Prospective freeze](PROSPECTIVE_FREEZE.json) | Original local source identity, not public preregistration |
| [Results release](RESULTS_RELEASE.json) | Original completed-release receipt; referenced files belong to the full scientific package |
| [Availability](AVAILABILITY.md) | Precise distinction between this branch and separately delivered evidence |

Run `python verify_primary_summary.py` with SciPy installed to check summary arithmetic and original byte identity. This is not a repetition-level or model-level audit. The [article and supplement](../../papers/score_enrichment/) contain all definitions, the full primary decision matrix, and the complete static tables.

The independently selected procedures are primary. The secondary common cap is selected by affine-local linear ridge, an asymmetric reference procedure. Its favorable outcomes do not replace primary comparisons. The study does not fit autonomous coordinate dynamics, establish Koopman closure, or demonstrate universal KAHM superiority.
