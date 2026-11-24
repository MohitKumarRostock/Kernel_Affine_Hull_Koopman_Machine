# Results and interpretation

The following comparisons concern specified representations, data distributions and fitted decoders. The executable studies have distinct sampling and evaluation protocols; results are not pooled. Exact population approximation risks, fitted excess MSE, standardized forecasting MSE and numerical recovery diagnostics are different quantities.

## Population calculations

In the [eight-point construction](SCIENTIFIC_REFERENCE.md), membership-weighted and raw-plus-membership representations have the same exact information and affine dimension four. The weighted encoding has smaller optimal excess risk for a prediction law aligned with the local directions and larger risk for the reversed law. For an aligned law, optimal excess risk tends to zero as the inverse Lipschitz constant diverges. In that model their product, with the inverse constant squared, tends to $v^2$. This is a result about the specified support and decoder class, not a guarantee for arbitrary fitted forecasting procedures.

The executable finite-law checks evaluate 1,350 task and parameter combinations, 54 optimal-contrast cases, 300 weighted-recovery cases and 270 inverse-stability cases. All use binary64 arithmetic and do not constitute interval-arithmetic verification or independent mathematical proofs.

## Fixed-geometry forecasting

The 12 primary paired intervals compare validation-selected weighted and separated representations at cap four and contrast 16. The outcomes are classified by the sign of the complete interval.

| Law | KAHM | Positive kernel | Affine |
| --- | --- | --- | --- |
| Linear AR | Higher | Higher | Higher |
| Nonlinear AR | Higher | Higher | Higher |
| Van der Pol | Lower | Lower | Lower |
| Duffing | Unresolved | Unresolved | Lower |

There are four lower, six higher and two unresolved comparisons. Complete values and intervals are in [primary comparisons](../ablation/primary_comparisons.csv). In the fixed slope-df-eight sensitivity comparison, weighting has higher observed mean loss in all twelve settings. For KAHM on Van der Pol the validation-selected mean SMSE is 0.04933 versus 0.06147 for Sep; at df eight the means are 0.22004 versus 0.06660. A favorable validation-selected result therefore does not establish a favorable comparison at the matched complexity constraint. The matched comparison is descriptive, not part of the adjusted interval family.

## Controlled predictive alignment

At contrast 16, KAHM-weighted and separated local coordinates with matched slope df 2.85 have the following mean excess MSEs over 32 repetitions.

| Target | Weighted | Separated |
| --- | ---: | ---: |
| $V$ | 0.00856 | 0.19138 |
| $U$ | 0.20700 | 0.18610 |
| $V+0.3U$ | 0.02306 | 0.09682 |

Information equivalence holds in exact arithmetic and the two encodings have the same feature width and centered rank three. The aligned target favors the weighted encoding, whereas the minor-coordinate target does not. All 32 reference fits recover the two regimes on the reference sample after label permutation. Alignment is deliberately imposed by the target law; the result does not establish that alignment is common in natural forecasting.

With validation-selected ridge, the KAHM-weighted aligned-target mean loss decreases from 0.08063 at contrast one to 0.00164 at contrast 64, while the mean of the per-repetition assessment median weighted singular values decreases from 0.356 to about $1.05\times10^{-16}$. This is consistent with the population construction but is not a proof about the inverse of the complete fitted encoding.

## Chronological CO2 study

For the first reference fit and contrast 16, the assessment aggregate SMSEs are:

| Encoding or control | Validation-selected penalty | Matched slope df 8 |
| --- | ---: | ---: |
| KAHM memberships | 0.58787 | Not applicable |
| KAHM Local | 0.48605 | 0.66910 |
| KAHM Sep | 0.41039 | 0.54407 |
| KAHM Raw+Mem | 0.41002 | 0.51680 |
| KAHM Gated raw | 0.39470 | 0.68410 |
| Raw ridge | 0.42259 | 0.52123 |
| Raw boosted trees | 0.43016 | Not applicable |
| Seasonal drift | 0.86347 | Not applicable |
| Persistence | 1.28977 | Not applicable |

In all 20 reconstruction-method/reference-seed combinations at contrast 16, Local has higher observed aggregate assessment loss than Sep under both penalty rules. This is a sensitivity analysis on one time series, not a sample of 20 independent series. For the first KAHM fit, Local's validation-selected losses at contrasts 1, 4, 16 and 64 are 0.43533, 0.45665, 0.48605 and 0.51262; Sep's are 0.42216, 0.41280, 0.41039 and 0.41102. The corresponding assessment median weighted singular values decrease from 0.0544 to $6.33\times10^{-10}$. The changing predictive and recovery behavior contrasts with the aligned synthetic task. The real-data result does not demonstrate a weighting advantage or KAHM-specific superiority.

## Earlier forecasting experiment

The archived 40 paired comparisons concern independently selected procedures and not controlled weighting alone. KAHM Local has lower mean loss than KAHM Mem in six of eight settings; normalizer enrichment resolves three favorable comparisons. Across all 40 intervals, 16 are lower, seven higher and 17 unresolved. In particular, comparisons with affine-local, positive-kernel-local and raw-history controls are mixed. The [complete saved-loss table](../reproduction/score_enrichment/evidence/primary_contrasts.csv) and its [independent checker](../reproduction/score_enrichment/check_saved_losses.py) support statistical recomputation conditional on the recorded losses, not original forecast regeneration.

## Supporting evidence

The [supporting study descriptions](RELATED_EVIDENCE.md) identify the target-relevance, decoder-capacity, center-weighting and forecast-correction results. The separate 12-row capacity summary is available for arithmetic verification. Other related findings are reported summaries, not regenerated experiments.

## Numerical records

The [ablation bundle](../ablation/) contains 9,472 forecasting and 12,384 target-changing records, 256 forecasting and 512 target-changing diagnostic records, complete summaries and interval calculations. The [observed-series bundle](../real_data/) contains 212 model-evaluation records, 390,504 assessment forecast values, 2,544 annual loss records, 1,050 validation scores and 23 geometry diagnostics. Selection labels identify the procedure rather than independent replications. All evaluation records, including unfavorable and unresolved comparisons, are retained. Results are rounded for display; use the CSV and NPZ arrays for exact recorded values.
