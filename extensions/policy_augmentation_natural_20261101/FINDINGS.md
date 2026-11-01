# Policy augmentation on reserved natural-series periods

## Complete-policy result

On the declared 48-series panel, adding the centered in-sample KAHM option
increased the base selection policy's aggregate standardized mean squared
error (SMSE) by **2.2725%**. The KAHM-enlarged policy had **2.7184%** lower
SMSE than the alternative enlargement with an additional Nyström RBF residual
family. The base policy remained better than either enlargement. These are
descriptive results on newly selected exact M4 series identities, with
anonymous source dependence; no independent-source confidence interval or
general forecasting superiority claim is made.

| Panel | P0 | P1: P0 + KAHM | P0plus: P0 + extra RBF | P1 gain over P0 | P1 gain over P0plus |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weekly, 24 series | 0.0793919770 | 0.0839246350 | 0.0914622650 | -5.7092% | +8.2412% |
| Hourly, 24 series | 0.1758324893 | 0.1770998295 | 0.1768562025 | -0.7208% | -0.1378% |
| Equal-frequency aggregate | 0.1276122331 | 0.1305122322 | 0.1341592338 | -2.2725% | +2.7184% |

Positive gains favor P1. Loss averages three horizons within each series,
then series within frequency, then frequencies equally. Each horizon MSE is
divided by its A-only target population variance. All 48 series are retained.
The complete policy comparisons are in
[`analysis/policy_contrasts.csv`](analysis/policy_contrasts.csv), with exact
means in [`analysis/aggregates.csv`](analysis/aggregates.csv).

## Shared-baseline correction and policy selection

The standalone calibrated KAHM residual procedure has aggregate SMSE
**0.1519562535**, compared with **0.1571825808** for its identical
common-plus-shared-bias baseline and **0.1552962153** for independently
selected in-sample RBF. These represent descriptive gains of **3.3250%** and
**2.1507%**, respectively. These mechanism comparisons do not replace the
complete-policy comparison above: the baseline policy P0 has lower mean loss
than the standalone correction procedure.

C selected the KAHM option for 3 of 48 series. For the other 45 series, P1
forwards exactly the same forecast as P0. Among the three selected series,
KAHM improved one and worsened two, giving a 23.0889% increase in their
combined mean SMSE. This subset is defined by the frozen C selection rule
and is descriptive, not a new primary claim.

| Selected series | P0 choice | P0 SMSE | P1 SMSE | P1 gain | Selected cap | Selected scaler | C multiplier at h = 1, 4, 12 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| W160 | Drift | 0.0030622647 | 0.0026435211 | +13.6743% | 1 | tanh(z) | 0, 0, 1 |
| W196 | Drift | 0.5442943807 | 0.6534969164 | -20.0631% | 4 | Standard | 0.5, 1, 0.5 |
| H368 | Bounded RBF departure | 0.0555294216 | 0.0859455858 | -54.7749% | 2 | tanh(z) | 0.5, 0.25, 1 |

For these three series, the ratios of P1 to P0 maximum absolute error are
0.7800, 1.0809, and 0.9522, respectively. The two mean-loss increases are not
accompanied by orders-of-magnitude amplification of the maximum error in
this diagnostic. All forecasts are finite, and inherited centered-correction
magnitude checks pass. These checks do not imply forecast accuracy.

Full per-series results and the C-selected subset table appear in
[`analysis/task_methods.csv`](analysis/task_methods.csv) and
[`analysis/selected_subset.csv`](analysis/selected_subset.csv).

## Dimension and preprocessing choices

The established validation policy was retained. KAHM selected caps
1, 2, 4, 8, and 12 for 12, 10, 15, 5, and 6 series, respectively.
Its selected scalers were standardization for 12 series, min–max for 16, and
tanh(z) for 20. Independently selected RBF chose caps 1, 2, 4, 8, and 12 for
22, 7, 9, 3, and 7 series, and selected standardization, min–max, and tanh(z)
for 10, 14, and 24 series. These selection frequencies do not identify a
universally preferred scaler or rank.

Actual local ranks are retained separately from requested caps. The selected
KAHM banks contain 672 local basis occurrences with ranks 1, 2, 4, 8, and 12.
Selected RBF banks additionally include one realized rank7 occurrence.
Bank occurrences are not independent samples. Exact distributions, selected
penalties, multipliers, and source hashes are in
[`analysis/report_metrics.json`](analysis/report_metrics.json).

## Secondary absolute-error scores

Study-specific seasonal MASE is **0.6260662464** for P0,
**0.6309459390** for P1, and **0.6295275368** for P0plus. P1 therefore has
0.7794% higher seasonal MASE than P0. Weekly scaling uses an explicitly
declared lag52 annual difference, while Hourly scaling uses lag24.

The supplementary frequency-lag MASE uses lag1 Weekly and lag24 Hourly:
**1.3809301905** for P0, **1.3902146722** for P1, and **1.4049320903**
for P0plus. P1 is 0.6723% above P0 on this measure. No MASE denominator floor
was activated. None of these repeated-origin, selected-offset scores is an
official full-horizon M4 ranking.

## Computation and verification

Every declared fit and prediction completed. The fit records were sealed
before D query preparation; every forecast was sealed before terminal target
release and scoring. No official M4 Test file was downloaded. There were
zero solver fallback events and zero failed tasks.

The independent audit passed for all **48 series**, **2,016 terminal forecast
arrays**, **10,800 B alpha-path forecasts**, **576 C multiplier decisions**,
**144 complete-policy choices**, and **576 selected ridge normal equations**.
It reconstructed source indexing, chronological maturity, causal direct
forecasts, all losses, and equal-frequency aggregation. It reuses inherited
feature maps and does not independently refit every unselected candidate.
The full scope and audit status are retained in
[`audit/verification.json`](audit/verification.json).

Summed fitting-worker CPU time was **381.13 seconds**, summed worker elapsed
time **382.68 seconds**, and maximum worker resident memory **238.59 MiB**.
Compressed fitted states total **195,950,102 bytes**. These measurements cover
the complete finite research suite, including rejected candidates and
diagnostic controls. They are not isolated deployment latency or an assertion
of equal computational cost between KAHM and the extra RBF family.

## Scientific implication

The panel supplies limited additional evidence that a centered KAHM
correction can improve an identical retained predictor. It does not establish
net utility from enlarging the stronger forecasting policy with that option.
Validation selection, even with the corrected dimension and preprocessing
searches, did not transfer enough of the correction's conditional value to
improve the complete policy on these reserved periods. The broader forecasting
claim therefore remains conditional and method-specific. The alternative
enlargement's higher loss cannot by itself establish a benefit over the base
policy.
