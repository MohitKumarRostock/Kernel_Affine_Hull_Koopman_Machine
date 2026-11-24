# Experimental protocols

## Common representation and decoder specification

The mathematical definitions, reconstruction methods, and assumptions are given in [Scientific reference](SCIENTIFIC_REFERENCE.md). The executable studies use positive affinities with offset $\tau=10^{-6}$; the primary contrast is $\omega=16$. The same reference histories, partitions, retained directions, and reconstruction scores are used for the weighted and separated encodings within each comparison. All representations predict observed future quantities directly.

Reference partitions are fitted by k-means with five initializations and a maximum of 150 iterations. Up to 24 histories per group are sampled without replacement. Reference arrays are rounded to binary32 and converted back to binary64 for fitting. A PCA direction is retained only for eigenvalues exceeding machine epsilon; the projected-range safeguard reduces the dimension when necessary. See `ablation/run_ablation.py` for the precise implementation, including the reference-only regularizer iteration and denominator fallback. Reconstruction, gate calculation and fitting use binary64. The fixed-geometry ablation includes the KAHM (`K`), positive kernel (`R`), affine (`A`), and center-only (`center`) score constructions.

The ridge objective is mean squared training error plus $\lambda\|w\|_2^2$, with an unpenalized intercept. When input standardization is applied, it uses training statistics only; the target-changing study uses its specified analytical scale. Feature standardization uses training statistics only. Feature columns with standard deviation below $10^{-8}$ use scale one. The penalty grid is $10^{-6},10^{-5},\ldots,10^2$. The forecasting studies select one common penalty across horizons by mean validation standardized MSE; controlled targets are fitted separately. The first penalty in increasing order attaining the minimum validation loss is selected. No assessment outcomes enter selection. No refit on validation data is performed.

For centered standardized training features with singular values $s_i$ and $n$ rows, slope effective degrees of freedom are $\sum_i (s_i^2/n)/(s_i^2/n+\lambda)$. The numerical rank threshold is the larger of $10^{-12}$ times the leading squared singular value divided by $n$ and $10^{-15}$. The matched-complexity comparison is performed only when this numerical rank exceeds its specified target; all singular values are retained in the ridge calculation. This criterion matches a particular measure of fitted complexity, not all aspects of statistical capacity. Equal affine dimension, equal feature count and equal effective degrees of freedom are distinct conditions.

## 1. Fixed-geometry forecasting

### Observation laws

Four independently simulated episode laws generate 72 observed values per episode.

| Law | Specification |
| --- | --- |
| Linear AR | $u_t=0.65u_{t-1}-0.15u_{t-2}+\eta_t$, $\eta_t\sim N(0,0.1^2)$; zero initial state, 256 burn-in values |
| Nonlinear AR | $u_t=\tanh(1.25u_{t-1}-0.65u_{t-2}+0.25\sin(3u_{t-1})+\eta_t)$, $\eta_t\sim U[-0.1,0.1]$; independent initial coordinates uniform on $[-0.8,0.8]$, 80 burn-in values |
| Van der Pol | $\dot q=v,\ \dot v=(1-q^2)v-q$ |
| Duffing | $\dot q=v,\ \dot v=-0.2v+q-q^3+0.3\cos(1.2s+\theta)$, with $\theta\sim U[0,2\pi]$ |

For both oscillator laws the initial coordinates are independent uniform variables on $[-2,2]$. Classical RK4 uses step 0.025; samples are taken at time 10.1 and every 0.1 thereafter. Observations are $0.95\tanh(q)+\eta$, with independent uniform sensor noise on $[-0.02,0.02]$.

### Sampling and model selection

Each law has 32 independent repetitions with random streams derived from `SeedSequence([2026112404, law, repetition, stream])`. In each repetition, training, validation and assessment comprise 64, 32 and 128 disjoint independently generated episodes. Twelve-observation histories end at zero-based indices 11, 15, ..., 59; targets are at horizons 1, 4, and 12. Training and validation subsample 512 and 256 windows without replacement. All 1,664 assessment windows are used. Features are standardized using training inputs. Two partitions request two and four groups; local dimension caps four and twelve are compared using common reference sets. Each method evaluates memberships, weighted local positions, separated positions, raw plus memberships, gated raw coordinates, and the corresponding raw-history control. All procedures use common observations within a repetition.

The response loss is the average of the three horizon-specific MSEs divided by their corresponding training target variances (floor $10^{-12}$). The primary family consists of Local minus Sep differences under validation-selected ridge, cap four, contrast 16, and methods K, R and A across four laws. Each interval uses the mean paired difference plus or minus $t_{31,1-0.05/24}$ times its standard error. The repetition, not an individual forecast window, is the inferential unit. Nominal simultaneous coverage relies on the paired-Student approximation. No equivalence or practical-effect threshold is imposed. An interval containing zero is unresolved. Matched slope df eight, cap twelve, center scores and all other comparisons are descriptive.

## 2. Target-changing alignment

Let $S$ be an equiprobable binary regime, $R\sim U[1,2]$, $\Theta\sim U[0.12,0.5]$, and $\xi_1,\xi_2$ independent equiprobable signs, mutually independent. Set

$$V=\xi_1 R\cos\Theta/\sqrt{7/6},\qquad U=\xi_2 R\sin\Theta/\sqrt{7/6}.$$

The input is $(V,U)$ in the first regime and $(U,V)$ in the second. Across 32 repetitions, 512 independent reference observations are clustered without labels using projective direction features $(w_1^2-w_2^2,2w_1w_2)$ for unit direction $w$. The two groups use cap one. Independent training, validation and assessment sets have sizes 512, 256 and 8192. Target laws are $V$, $U$ and $V+0.3U$; Gaussian noise with standard deviation 0.05 is added to training and validation targets, using common noise across targets. Assessment uses noiseless conditional means, so losses are fitted excess MSE. The same input law, geometry, partitions and contrast apply to all targets. The reference-regime labels are used only for post-fit alignment diagnostics; they do not enter clustering or prediction.

The four reconstruction methods are evaluated at contrasts 1, 4, 16 and 64. Local, Sep and Raw+Mem are also compared at matched slope df 2.85 when their numerical rank permits. Reported loss means, positive reconstruction-margin fractions, and weighted recovery singular values are descriptive across repetitions. Conditioning statistics refer to the weighted linear recovery system, not a general certificate of inverse instability.

## 3. Chronological atmospheric CO2 forecasting

The supplied weekly [CO2 snapshot](../real_data/co2.csv) has 2,284 weekly slots from March 1958 to December 2001, including 59 missing values. For a weekly origin $t$ and horizons $h\in\{1,4,12\}$, the 15 input fields comprise the most recent 12 differences and the three seasonal changes $y_{t-52+h}-y_{t-52}$. The target is $y_{t+h}-y_t$; adding the observed origin level gives the level forecast and leaves forecast errors unchanged. Origins requiring any missing observation are excluded. No imputation is performed.

Training targets are strictly before 1980; validation origins begin in 1980 and their targets are before 1990; assessment origins begin in 1990. All three targets must remain in the corresponding period. The resulting counts are 789 training, 454 validation, and 614 assessment origins. Input scaling and target-variance normalization use training data alone. Predictions use observations available at the given origin, including seasonal inputs from the past. The reference fits use seeds 2026112406 through 2026112410, cap four and the two- and four-group partitions. The first seed supplies contrast sensitivity at 1, 4, 16 and 64 for KAHM; all five seeds use contrast 16 for K, R, A and center scores. Validation-selected ridge and matched slope df eight are compared.

Controls are raw ridge, persistence (zero forecast change), seasonal drift (the recorded seasonal change) and histogram gradient boosting on raw inputs. Boosting fits separate horizon models with maximum leaf counts 7, 15 or 31, 100 or 300 iterations, learning rate 0.05, minimum leaf size 20, and disabled early stopping. One common configuration minimizes mean validation SMSE. Assessment results include aggregate SMSE, horizon RMSE in ppm, annual summaries and forecasts. Forecast origins overlap and reference seeds reuse the same time series; none is treated as an independent replication of the data law. All real-series findings are descriptive.

## 4. Earlier forecasting experiment

The [archived forecasting specification](ARCHIVED_FORECASTING.md) defines the complete reported observation, fitting and comparison conventions. The [saved-loss archive](../reproduction/score_enrichment/) supplies the protocol, per-repetition losses, original selection and diagnostic records, and independent statistical analysis. It comprises 64 repetitions per law and 40 primary comparisons, with separately selected configurations and linear or Nyström-augmented ridge predictors. It is a distinct experiment from the fixed-geometry simulations. Its original simulation, complete task list, fitted model states and forecast arrays are unavailable. Statistical reconstruction of recorded losses is not regeneration of those forecasts. No observations or comparison families are pooled between these studies.
