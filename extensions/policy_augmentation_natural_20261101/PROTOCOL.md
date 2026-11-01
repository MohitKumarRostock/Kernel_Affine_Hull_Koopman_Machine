# Natural-series policy augmentation

## Question and evidence status

This study evaluates whether adding the existing centered in-sample bounded
KAHM residual procedure changes the forecasting accuracy of a declared
non-KAHM selection policy. A second enlargement adds a credible non-KAHM
residual family with the same number of feature configurations, ridge-penalty
scores, and calibration choices. The study follows the revised forecasting
manuscript at commit `2acd4f5f8b4970ef0e066f1b91a3636d36ad932f`.
Its hypothesis and method choices were informed by preceding KAHM studies.
It is an external benchmark-development experiment, separately reported from
the fresh simulation comparison; observations are never pooled across them.

The M4 source is pinned at
`6c1067e5a57161249b17289a565178dc7a3fb3ca` in
<https://github.com/Mcompetitions/M4-methods>. Only official Train files and
metadata are downloaded. Terminal portions of these Train series are reserved
within this procedure. Official M4 Test files are not used. Public benchmark
availability, earlier method development, and anonymous source dependence
preclude describing this as a blinded trial or an independently sampled set
of application domains.

## Roster fixed from identifiers and lengths

The complete previous M4 pilot roster of 72 Quarterly, Monthly, and Hourly
identities is recorded in `roster.json`, recovered from the prior
`KAHM_Step3_Evaluation_Report.pdf`, pages 10–12. Those identities are excluded.
Within Weekly and Hourly, require at least 475 observations. Order remaining
series by SHA256 of
`kahm-policy-augmentation-natural-2026-11-01-v1|Frequency|ID`, and retain the
first 24 of each frequency. No observation magnitude, validation score,
forecast error, or model success enters eligibility or replacement.

The Weekly file contains 359 series; 258 meet length eligibility. The Hourly
file contains 414 series; 24 previously studied identities are excluded and
the remaining 390 meet length eligibility. All 48 selected identities appear
in `roster.json`. New identities do not imply independent underlying sources;
the anonymous series may share a source process, shocks, or collection history.

## Chronology and retained data

History width is 12 and direct horizons are 1, 4, and 12. Forecast origin t
means that observations through t are available. A history is
`(y[t-11],...,y[t])`; its target vector is `y[t+[1,4,12]]`.

Choose D as the last 32 origins with all three targets available in the
official Train record. Moving backwards, retain C64 and B64 origins, with
12 intervening origins omitted at each boundary. Retain up to 512 preceding
A origins, again separated by 12 omitted origins; require at least 256 A
origins. Thus A has `min(512,N-219)` rows and N must be at least475.
The largest target index of each earlier block is strictly earlier than the
first origin of the next. No target observation is shared between blocks.
Histories and origins within blocks overlap; these are dependent observations.

For selected N700 Hourly series A has481 rows; N960 series have512. The
selected Weekly W38 has N510 and A291; the remaining selected Weekly series
have512 A rows. No A/B/C refit is performed after C selection. During D,
histories at later origins can include earlier D-period observations that have
become available. This is fixed-model rolling-origin prediction, with no
parameter updates or future observations used by a query.

The source/configuration/roster freeze precedes numerical parsing of study
values. ABC preparation converts only the source prefix through the last C
target. After every fit and C policy choice is sealed, query preparation
provides D histories and causal direct-control forecasts without D target
arrays. After every forecast is sealed, terminal targets are parsed and
released for scoring. Raw public Train bytes are present throughout; these
are procedural records, not cryptographic target isolation or independently
attested preregistration.

## Fixed candidate construction and selection

The inherited corrected `preprocessing_model.fit_primary` is unchanged. Its
seven non-KAHM procedures are raw ridge, raw Nyström RBF, boosted trees,
calibrated persistence, common geometry, bounded RBF departure, and unbounded
RBF departure. Their original finite grids and B/C roles are retained.

The added residual suite retains the full-A, B-selected common forecast g,
with common multiplier1. A shared bias is `mean_B(Y_B-g(X_B))`. Geometry and
all history/feature scalers are fitted on A. Slopes use centered A features
and centered in-sample residual responses `Y_A-g(X_A)`; no additional
response-mean intercept is restored. Both KAHM and positive-kernel RBF
departures independently search native caps {1,2,4,8,12}, history scalers
{standard,minmax,tanh(z)}, and ridge penalties
{0.0001,0.001,0.01,0.1,1}. For each feature configuration B selects each
horizon's penalty; its normalized aggregate B loss selects one cap/scaler.
Candidate ties use cap ascending, then standard/minmax/tanh order.
Native rank initialization is `min(cap,d,N_local-1)`, with the established
positive-variance and projected-range reductions. Cap12 includes the native
cap20 construction when ambient width is12; it does not imply realized rank12.
Realized local dimensions are retained in the fit records.

C selects horizon-specific correction multipliers from {0,0.25,0.5,1}; zero
exactly recovers the identical common-plus-bias forecast. KAHM-matched RBF,
with KAHM's selected scaler/cap and its own penalty/multiplier, is retained as
a diagnostic and excluded from the policy menus. Single natural series do
not supply independent episodes, so the simulation's out-of-fold RBF option
is explicitly omitted. No artificial independent-fold interpretation is used.

The extra non-KAHM residual family is the simulation extension's unchanged
bounded centered raw-history/Nyström RBF readout around the same common
forecast and bias. It searches the same three scalers, five bandwidth factors
{0.25,0.5,1,2,4}, five ridge penalties, and four C multipliers. It uses64
A-selected landmarks. Thus each added family has15 feature configurations,
75 B penalty scores per horizon, and4 C multiplier choices. Feature widths,
fitting costs, and functions differ; matching these counts does not establish
equal compute or capacity. The source hash of this extension is retained.

Two direct controls are added to all policies. Seasonal naive uses the latest
available value of the same seasonal position, with lag52 for Weekly and24
for Hourly. Lag52 is a deliberately declared annual-lag Weekly control,
not the official M4 Weekly frequency/Naive2 convention. Drift predicts
`y_t+h*(y_t-y_start)/(t-start)`, where start is the first raw observation used
by an A history. Both use only observations available at each queried origin.

P0 contains the seven inherited non-KAHM procedures, shared-bias common,
in-sample RBF, seasonal naive, and drift:11 options. P1 adds centered in-sample
KAHM:12 options. P0plus adds the extra Nyström RBF residual family to P0:
12 options. Each policy selects one family across all horizons by calibrated
C standardized MSE. Ties use the exact order in `config.json`, with the new
candidate last. B and C reuse is part of the evaluated finite procedure and
does not provide an unbiased development performance estimate.

## Analysis and interpretation

Primary descriptive loss is SMSE. For each horizon, divide D mean squared
error by its full-A target population variance, floored at1e-12. Average three
horizons within series, then24 series within each frequency, then the two
frequencies equally. The primary contrasts are complete P1 against P0 and
P0plus. Relative gain is `100*(1-L_P1/L_reference)` on aggregate losses.

Secondary study-specific seasonal MASE divides mean absolute error by the
A-only raw-prefix mean absolute seasonal difference at lag52 Weekly/24 Hourly.
Frequency MASE additionally uses lag1 Weekly/24 Hourly. The raw prefix runs
from the first observation used by an A history to the last A target.
Both scales are floored at1e-12 and floor activations are reported. These
selected-offset, repeated-origin scores are not the official full-horizon M4
leaderboard evaluation, even where a scale's lag matches the frequency.

Report every series, all methods, both frequencies, requested and realized
dimensions, selected scalers, policy choices, and total research-suite search
cost. Report KAHM-selected and nonselected subsets descriptively; they cannot
replace the full-roster result. No confidence interval, p-value, population
bootstrap, or independent-series sampling claim is used. The study cannot
establish general superiority from this selected source panel.

Unexpected numerical or data failures stop execution. Preserve failures and
do not substitute an easier series, remove failed outputs, change the grid,
or introduce a favorable post-outcome stopping rule. The existing checked
SVD recovery rule remains unchanged. The verifier independently reconstructs
raw-source indexing, chronology, direct-control forecasts, B alpha/family
choices, C multipliers/policies, selected ridge normal equations, all D
losses, and frequency aggregation. It reuses inherited feature maps and does
not independently refit every unselected candidate.

## Execution

Run3 fresh workers with one numerical thread per worker. Where available,
workers use allowed CPU indices3–5, leaving indices0–2 for the concurrent
simulation campaign. Record worker CPU/elapsed time and memory. Summed
worker elapsed time is not campaign wall time; whole-suite time includes
diagnostic and rejected models and is not isolated deployment latency.

No revised manuscript claim is inserted automatically. This extension first
produces a complete, auditable empirical result whose direction is unknown
when the above source/configuration is frozen.
