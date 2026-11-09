# Forecasting features, residual corrections, and matched evaluation

This guide connects the local-dimension, preprocessing, residual-target, and reconstruction experiments. The implementation and reported evidence are specific to each study.

- [Dimension forecasting](../dimension_forecasting/README.md) and [findings](../dimension_forecasting/FINDINGS.md)
- [Preprocessing](../preprocessing_dimension/README.md) and [findings](../preprocessing_dimension/FINDINGS.md)
- [Cross-fit computational reproduction](../crossfit_residual_reproduction_20261101/README.md) and [findings](../crossfit_residual_reproduction_20261101/reporting/RESULTS.md)
- [Fixed reconstruction departures](../reconstruction_mechanism/README.md)
- [Target geometry](../target_geometry/README.md)
- [Direct reconstruction training](../direct_reconstruction/README.md)
- [Online adaptation](../adaptive_development/REPRODUCING.md)

## Scientific scope

The analysis develops conditional feature-span equivalence and
distinctness results for KAHM and RBF reconstructions, together with a uniform
bound on the ridge residual correction including its free intercept. A
reconstruction can change the functions accessible to a finite predictor;
neither deterministic reconstruction nor a correction-amplitude bound
guarantees improved forecasting accuracy.

The principal experiment compares ten complete procedures on four numerical
laws, two added-noise levels and 24 independent repetitions. The local
dimension cap is selected on validation data separately for each geometric
procedure. Calibration subsequently chooses blending coefficients and one
complete non-KAHM procedure; terminal forecasts are fixed before target release.

Validation-selected KAHM reduces mean nonlinear standardized MSE by 3.26%
relative to the historical cap-two construction, while native cap twenty
increases it by 8.39%. Three of the six primary margin criteria are established:
the nonlinear comparisons with common features and raw-input RBF, and
noninferiority to ridge on the linear control. The full comparison criterion
is not established. The selected non-KAHM procedure has lower aggregate error,
and the KAHM advantage over an independently tuned RBF departure is small.
Separate matched-bank comparisons account for the dimension-selection gain
through improved common-feature predictor loss and a weaker additional KAHM
departure benefit. Local dimension, stacked-basis rank and predictive accuracy
are distinct quantities. A fixed cap is not an accuracy guarantee.

### Preprocessing and residual-target follow-ups

The follow-up studies use distinct evaluation procedures and comparison families:

| Study | Primary family | Established criteria | Main limitation |
| --- | --- | --- | --- |
| Local dimension selection | Six comparisons | Three | No declared-margin advantage over independently tuned RBF or the selected non-KAHM policy |
| Preprocessing and local dimension | Four comparisons | One | Tanh and jointly selected KAHM do not establish nonlinear superiority |
| Cross-fit computational reproduction | Original four comparisons | Two | Independent-RBF margin and policy superiority are not established; no additional independent repetitions |

In the preprocessing study, tanh/tanh has 6.486% higher nonlinear error than
standard/standard. Jointly selected KAHM has 4.022% higher error than the
non-KAHM policy and 0.281% higher error than independently selected bounded
RBF. A separate secondary architecture holds each common predictor and bias
fixed and fits corrections to residuals from disjoint episodes. Across its
five settings, KAHM improves over the shared baseline by 10.780–14.934% and
independently selected RBF by 2.372–4.140%. These results concern a different
training allocation from the primary complete-procedure comparison.

The cross-fit study strengthens the deployed common predictor to all 64
training episodes and forms OOF residuals with 48-episode fold predictors.
The computational reproduction gives a 9.1968% nonlinear KAHM gain over the
shared common-plus-bias baseline. The 2.8662% gain over independently selected
OOF RBF misses the prescribed 3% margin, and KAHM has 5.1246% higher error than
the expanded non-KAHM policy. Linear-control noninferiority is established.

In-sample KAHM is secondary: its gains are 12.4259% over the shared baseline
and 3.9643% over independently selected in-sample RBF. Its mean error remains
1.3862% above the expanded policy in a post hoc descriptive comparison.
Cross-fitted KAHM has 3.6873% higher error than the in-sample correction.
The different fold and deployment training sizes motivate a residual-alignment
hypothesis; the comparison does not establish its cause or a general ordering
between residual-fitting methods.

The strongest supported interpretation is conditional utility of bounded
KAHM features and residual corrections. A useful portfolio-specialist role
remains a future hypothesis: the completed policies exclude KAHM, and their
performance does not establish the marginal value of adding it. A prospective
comparison would hold the baseline policy fixed, add a prespecified KAHM
candidate, and control the extra search opportunity on fresh data.

The controlled, natural-data, scaling, additive-correction, target-geometry, direct-training, and adaptation experiments each have their own construction choices, sampling units, and uncertainty statements. Their outcomes are not pooled with the dimension-selected experiment.

## Reproduce the numerical evidence

The [dimension-study guide](../dimension_forecasting/README.md) supplies
restoration, fitting, prediction, analysis and independent-audit commands.
The complete numerical records are distributed as six disjoint archives,
`KAHM_Forecasting_Dimension_Records_01.zip` through `_06.zip`. They restore
under `extensions/dimension_forecasting/`. Their
[archive manifest](../dimension_forecasting/documentation/KAHM_Forecasting_Dimension_Records_manifest.json)
records every member and checksum. All 27 compact analysis tables and the
independent final audit accompany the source extension.

The principal inference uses 24 independent repetition aggregates and six
approximate simultaneous paired-t bounds. It does not treat overlapping
forecast origins as independent observations and does not supply the
bounded-loss certificates studied in the separate risk study. Fixed-cap
sensitivity results and matched-bank mechanism comparisons remain secondary.

The [preprocessing package guide](../preprocessing_dimension/RESULTS_PACKAGE.md)
and [archive index](../preprocessing_dimension/reporting/archive_index.json)
describe its analysis ZIP and twelve numerical-record ZIPs. Its independent
audit checks all aggregate arithmetic and the four primary contrasts; raw
forecast-to-loss replay covers the prespecified eight repetition-zero tasks.

The [cross-fit report](../crossfit_residual_reproduction_20261101/reporting/RESULTS.md),
[archive manifest](../crossfit_residual_reproduction_20261101/reporting/ARCHIVE_MANIFEST.json)
and [delivery verification](../crossfit_residual_reproduction_20261101/reporting/archive_delivery_verification.json)
identify its analysis ZIP, sixteen records ZIPs and 3,311 retained files.
The independent audit covers all 192 tasks and 8,064 forecast arrays. This
execution regenerates the same 24 original cross-fit units after earlier
reported outcomes were known. It adds zero independent repetitions and does
not authenticate the unavailable original completed-run bytes or seals.

## Scientific verification utilities

The package modules retain source-identity, numerical-audit, and archive-verification routines. Prepare the recorded experimental source context using the [target-geometry guide](../target_geometry/README.md), then run these study-specific audits inside it:

```bash
cd /tmp/kahm-scientific-context
python -m extensions.reconstruction_mechanism.package_results audit
python -m extensions.tmlr_integration.package_utility audit
python -m extensions.tmlr_integration.package_target audit
python -m extensions.tmlr_integration.package_direct audit
```

These commands require the numerical inputs named by each study's guide. The fixed-reconstruction saved-forecast checker, `verify_saved_forecasts.py`, expects its own 192-task evidence schema; those tasks and eight recomputed tables do not substitute for the dimension-selected or follow-up experiments.

Derived numerical tables are stored with their studies under `reporting/`. The reporting generators read completed results and write to separate output directories. They do not change model selection or supply new independent repetitions.
