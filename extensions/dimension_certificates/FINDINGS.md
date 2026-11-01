# Local subspace dimension in retained KAHM forecasting representations

Validation selection of local subspace dimension reduced mean test error relative to a fixed cap of two on five of the six synthetic laws and reproduced the cap-two result on the sixth. The native cap of twenty had mixed effects. These forecasting differences did not change the tolerance decisions for the validation-selected forecasts, and the simultaneous paired comparisons did not certify an improvement between dimension policies. The results support treating local dimension as a development parameter while retaining a separate assessment of certificate utility. [Numerical summary](runs/20261101/analysis/summary.json); [independent audit](audit/post_release/final_results_audit.json).

## Design

The study comprised 26 independent fitting repetitions: five each for harmonic, quasiperiodic, nonlinear autoregressive and logistic laws, and three each for Van der Pol and Duffing. Each repetition used 128 training, 512 validation, 4096 calibration and 2048 test episodes. Histories of lengths 4 and 12 shared thirteen forecast origins, with horizons 1, 4 and 12. Inputs were standardized using training statistics. The master seed was 2026110104. All development fits and selections were sealed before calibration generation; the complete calibration stage was sealed before test generation. [Protocol](PROTOCOL.md); [frozen schedule](runs/20261101/schedule.json).

Three KAHM policies used cap two, native cap twenty, or validation selection over caps 1, 2, 4, 8, 12 and 20. Each policy selected one complete history–cap bank by the best prefix’s mean validation MSE across horizons, then retained all four prefixes for certification. Their coordinate counts were 2, 6, 14 and 30. The decoder used sum-of-block Frobenius norm budget 4 and ridge penalty 0.0001. The local construction requested min(cap, ambient dimension, local reference count minus one), followed by the native numerical rank rules. [Configuration](runs/20261101/config.json).

## Forecasting error

The table reports test MSE averaged over the three horizons within each law, followed in parentheses by the sample standard deviation over independent fitting repetitions. Percentage changes compare the means for selected cap and cap two. Errors are not pooled across laws as a common physical risk. [Policy summaries](runs/20261101/analysis/selected_policy_summary.csv); [paired policy summaries](runs/20261101/analysis/policy_comparison_summary.csv).

| Law | Cap 2: MSE (SD) | Native 20: MSE (SD) | Selected cap: MSE (SD) | Selected versus cap 2 |
|---|---:|---:|---:|---:|
| Harmonic | 0.051868 (0.003817) | 0.051727 (0.003231) | 0.048401 (0.005200) | -6.68% |
| Quasiperiodic | 0.037484 (0.001373) | 0.038452 (0.001291) | 0.037484 (0.001373) | +0.00% |
| Nonlinear AR | 0.004591 (0.000087) | 0.004534 (0.000069) | 0.004534 (0.000069) | -1.24% |
| Logistic | 0.164773 (0.003695) | 0.167156 (0.003606) | 0.163252 (0.003213) | -0.92% |
| Van der Pol | 0.060314 (0.006781) | 0.058421 (0.004241) | 0.052366 (0.004645) | -13.18% |
| Duffing | 0.028750 (0.002039) | 0.034536 (0.002095) | 0.027932 (0.002038) | -2.85% |

The selected policy improved the mean across horizons in 18 of 26 individual repetitions and tied in eight; it worsened none in this sample. Selected cap and cap two chose the same history and prefix in every repetition. The selected caps were one in eight repetitions, two in eight, and four in ten. All 78 selected KAHM forecasts across the three policies used the 30-coordinate prefix. Harmonic, quasiperiodic, nonlinear AR and logistic selected history length four throughout; the selected policy used length twelve in two Van der Pol and all three Duffing repetitions. [Selected forecasts](runs/20261101/analysis/selected_forecasts.csv).

Native cap twenty decreased mean MSE on harmonic, nonlinear AR and Van der Pol, and increased it on quasiperiodic, logistic and Duffing. The Duffing increase was 20.12%, with higher error in all three repetitions. In the selected native banks, the actual local ranks were four for 674 encoders, twelve for 90, and three for 16. The rank-three cases arose from the native projected-range reduction. Thus “cap twenty” denotes the construction limit, not a realized rank of twenty. [Local-rank records](runs/20261101/analysis/selected_bank_local_ranks.csv).

### Comparators

The original RBF and affine-hull maps retained their distinct definitions. Ridge autoregression and quadratic regression selected history length on validation. The following means use the same held-out episodes; the cross-family contrasts are descriptive. [Complete family summaries](runs/20261101/analysis/selected_policy_summary.csv).

| Law | RBF | Affine hull | Ridge AR | Quadratic direct |
|---|---:|---:|---:|---:|
| Harmonic | 0.052144 | 0.054388 | 0.032028 | 0.022765 |
| Quasiperiodic | 0.035580 | 0.035714 | 0.025087 | 0.022347 |
| Nonlinear AR | 0.004646 | 0.004730 | 0.005027 | 0.005132 |
| Logistic | 0.165873 | 0.173346 | 0.225949 | 0.164866 |
| Van der Pol | 0.064008 | 0.050111 | 0.049516 | 0.045978 |
| Duffing | 0.023949 | 0.021399 | 0.008442 | 0.006895 |

Selected KAHM had a lower mean than these four comparators on nonlinear AR and logistic. Quadratic regression remained stronger on harmonic, quasiperiodic, Van der Pol and Duffing. The dimension policy therefore improves the KAHM comparison without establishing a general advantage over conventional forecasting models. Equal numerical norm budgets and feature counts also do not imply equal statistical capacity.

## Certificate utility

A campaign-wide failure probability of 0.05 covered 3744 fitted-loss upper and class-witness lower endpoints, together with 936 fitted-improvement lower endpoints. The probability allocated to each designated endpoint was 0.05/4680. The family contained 24 evaluated model aliases per repetition, with three horizons. [Endpoint records](runs/20261101/analysis/endpoints.csv); [comparison records](runs/20261101/analysis/paired_contrasts.csv).

For every repetition and each declared tolerance, the three policies gave the same decision for their validation-selected forecasts. Counts below apply separately to each policy.

| RMSE tolerance | Accepted / 26 | Excluded / 26 | Unresolved / 26 |
|---|---:|---:|---:|
| 0.20 | 5 | 0 | 21 |
| 0.35 | 13 | 0 | 13 |
| 0.50 | 21 | 0 | 5 |

All 234 horizon-specific lower endpoints for the selected KAHM forecasts were zero. Across all retained KAHM prefixes, cap two, native twenty and selected cap had 91, 81 and 88 positive lower endpoints, respectively. At tolerance 0.35, their 104 retained-prefix models yielded 30/3/71, 29/3/72 and 30/3/71 accepted/excluded/unresolved decisions. The selected policy did not increase resolution relative to cap two at this tolerance. [Decisions](runs/20261101/analysis/decisions.csv); [resolution by law](runs/20261101/analysis/certificate_resolution.csv).

Of 702 adjacent-prefix fitted-loss comparisons, 132 had positive simultaneous improvement lower endpoints: 47 for cap two, 40 for native twenty and 45 for selected cap. None of the 234 comparisons between validation-selected dimension policies had a positive improvement lower endpoint. These statements concern fitted forecasts; they do not compare unknown class-optimal risks. Across the full family, 1430 of 1872 lower endpoints were zero, no upper endpoint reached the loss bound of four, and no conflicting tolerance decision occurred.

## Interpretation and reproducibility

The forecasting evidence favors validation selection over a universal prescription to increase local rank. The largest improvements came from selecting cap one on Van der Pol, while cap two remained the selected choice on quasiperiodic. The declared certificate decisions remained unchanged despite these error reductions. The unknown class-optimal risks prevent interpreting test-error containment as an estimate of true coverage; independent test means are utility and consistency diagnostics. The campaign-wide confidence allocation also differs from earlier per-repetition studies, so cross-study interval widths cannot isolate the effect of rank.

All 1820 candidate fits completed within their prescribed optimization criterion, and every evaluated coefficient budget and prediction-range check passed. The independent audit checked all nine analysis tables and 9020 rows, all 4680 designated endpoints, 9360 development KAHM rank records and 288 full prediction-to-loss joins from the six prespecified repetition-zero jobs. The complete campaign received score-level reconstruction; full prediction arrays were retained for those six jobs. No production failure or scientific outcome was omitted. [Independent verification receipt](audit/post_release/final_results_audit.json).

The scope is six bounded synthetic laws with three or five fitting repetitions per law, standard preprocessing and the block-direct decoder. The controlled two-dimensional mechanism experiments, other decoder families and natural-data cohorts retain their separate scope. [Machine-readable report metrics](REPORT_METRICS.json) identify the exact source freeze, final test seal, audit receipt and numerical outcomes.
