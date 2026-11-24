# Supporting controlled and forecasting evidence

The investigations below concern separate populations, encodings, decoder configurations and selection rules. They are contextual evidence for the analysis of predictive accessibility, not additional repetitions of the principal fixed-geometry or chronological studies. The [capacity-comparison record](../related_evidence/capacity_primary_contrasts.csv) contains the complete reported comparison family. The other findings are reported summaries; the corresponding complete inputs, fitted predictions and experimental implementations are not distributed in this collection. No claim of end-to-end regeneration is made for these related evaluations.

## Target relevance with fixed inputs and geometry

Let $S$ be an equiprobable sign, $T\sim U[-1,1]$ and $\epsilon\sim U[-0.1,0.1]$, mutually independent. For curvature $c\in\{0,1\}$ and target parameter $\gamma\in\{0,1/2,1\}$, the static regression study uses

$$X=(T,c\cos(\pi T)+0.5S)^\top,\qquad Y_\gamma=\sqrt{1-\gamma}S+\sqrt{3\gamma}T+\epsilon.$$

Within a repetition and curvature, the inputs, noise values and fitted geometry are fixed across targets. The predictable signal has variance one and the noise variance is $1/300$. For straight branches, exact invariance of the encoding to reflection of $T$ would imply $\mathbb E[T\mid\Gamma(X)]=0$ and an irreducible signed-position contribution $\gamma$. The reported numerical near-invariance is not exact and does not establish this population conclusion.

| Curvature | Target parameter | Affine membership MSE | KAHM membership MSE |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.003334 | 0.003347 |
| 0 | 0.5 | 0.503448 | 0.013995 |
| 0 | 1 | 1.002694 | 0.024668 |
| 1 | 0 | 0.282887 | 0.281126 |
| 1 | 0.5 | 0.160813 | 0.165207 |
| 1 | 1 | 0.062366 | 0.051042 |

The reported affine-score control has MSE 0.004641 for the straight-branch position target. Local-coordinate controls also improve access to position; the benefit is not exclusive to KAHM. The six comparisons are part of a distinct 29-comparison family, not the archived 40-comparison or executable 12-comparison families.

## Decoder landmark counts with constrained local ranks

The separate capacity study uses 24 independent repetitions under each of nonlinear autoregression, Van der Pol and Duffing. It compares raw histories, affine-local and KAHM-local features using 64 and 256 Nyström landmarks, with local rank constrained by $\min(2,d-1,n_j-1)$ and partitions requesting 2, 4 and 8 groups. Its normalization, small-kernel handling and fitting conventions differ from the archived forecasting procedures.

The [complete comparison CSV](../related_evidence/capacity_primary_contrasts.csv) contains 12 rows. Here $D$ is control MSE minus KAHM MSE, so positive values favor KAHM. With sample mean $\bar D$ and sample standard deviation $s_D$, intervals are $\bar D\pm t_{23,1-0.05/(2\cdot12)}s_D/\sqrt{24}$. The critical value is 3.18066309590. Practical improvement requires the lower endpoint to exceed the fixed positive margin; a negative upper endpoint favors the control. Four comparisons meet their practical margins, six favor the controls, and two have unresolved direction. These MSEs, margins and signs differ from the archived 40-comparison SMSE family. At 256 landmarks on Van der Pol the mean MSEs for raw, affine-local and KAHM-local are 0.001654150, 0.001078003 and 0.000871969. Comparisons at two landmark counts do not constitute a test of the interaction with landmark count.

Run `python scripts/check_capacity_summary.py` from the root to verify the recorded summary arithmetic, Student critical value, interval endpoints and decision counts. It does not reconstruct the per-repetition differences or the fitted models.

## Common center-distance weights and reconstruction features

A four-construction study uses distance-to-center weights: G contains the weights, P adds weighted local coordinates, and K and R additionally append coordinates from KAHM and matched positive-kernel reconstruction. These weights differ from reconstruction-score memberships. The study uses 32 repetitions and 256 law/noise tasks. P selects the reference geometry, and all constructions use affine ridge. Linear and nonlinear refer to the observation-law groups, not decoder architectures.

The reported nonlinear-law gains are 69.201% for P over G, 10.459% for K over P and 0.029% for K over R. The first two meet the specified 3% improvement criterion, but the K/R comparison does not. K has 1.111% higher mean loss than P in the linear-law group and meets the specified 5% noninferiority criterion. Failure to meet an improvement margin does not establish equivalence.

## Forecast corrections and validation-based model selection

Residual prediction adds a learned correction to a baseline forecast; a policy selects among candidate forecasting families using validation data.

| Investigation | Reported outcome |
| --- | --- |
| Dimension selection (24 repetitions, 192 law/noise tasks) | KAHM meets the improvement criteria over common features and a raw-history Gaussian-kernel control and meets the noninferiority criterion; the other three improvement criteria are not established. C is used for calibration, unlike its diagnostic-only role in the archived forecasting study. |
| Preprocessing and dimension selection (24 repetitions) | One of four criteria is met (noninferiority). The KAHM procedure has higher mean loss in its specified improvements over selected positive-kernel and non-KAHM policies. |
| Residual-prediction repeated calculation | Two of four criteria are met on the same 24 prescribed units with prior outcome knowledge; reuse supplies no independent observations. The original execution records are unavailable. |
| Policy augmentation (24 repetitions, 192 tasks) | The base policy selects among ten non-KAHM families. KAHM correction gives point gains of 0.816% over the base and 0.658% over a raw-history Nyström correction; neither 3% improvement criterion is met, while both noninferiority criteria are met. |
| M4 training-series evaluation | 24 Weekly and 24 Hourly series, excluding pilot identifiers and official M4 Test data, use chronological A/B/C/D splits. Mean SMSE increases from 0.127612 to 0.130512 (2.273%) after KAHM augmentation. Of three changed model selections, one improves and two deteriorate. Dependence across series is not characterized; the comparison is descriptive. |

The M4 series and metadata are identified by the [M4-methods data snapshot](https://github.com/Mcompetitions/M4-methods/tree/6c1067e5a57161249b17289a565178dc7a3fb3ca); its observations and evaluation implementation are not distributed here.

These findings concern specified direct prediction procedures. They do not establish a universal advantage of reconstruction-based features, exact Markov sufficiency or autonomous linear evolution. Additional models with learned metrics, altered scaling or online references are outside the implemented study scope.
