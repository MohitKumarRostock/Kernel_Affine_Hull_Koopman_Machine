# Direct reconstruction training: scope and relation to subsequent studies

The direct-training experiment evaluates a specified intervention on KAHM bandwidths and gate temperatures. The representation keeps its reference observations, principal directions, covariance estimates, and original reference regularization fixed. Forecasting error on the geometry-training partition determines the bounded parameters, and a separate validation partition selects the residual-predictor penalty and correction strength.

## Findings and remaining questions

The complete adjusted synthetic comparison family has two favorable, six unfavorable, and ten unresolved intervals for KAHM. One favorable comparison identifies a substantial improvement over its own gate-only variant on Van der Pol. No favorable primary interval establishes superiority over an equally trained non-KAHM local control. The [findings](FINDINGS.md) report all methods, measured costs, and primary comparisons.

Three planned rideshare tasks contain no complete geometry-training windows. The full four-collection natural-data aggregate is therefore undefined. The seven available series provide descriptive evidence from three collections; they do not replace the complete planned aggregate. These restrictions remain part of the study's interpretation.

The central unresolved issue is the incremental forecasting value of the KAHM construction after controlling for predictive supervision, decoder capacity, and simpler geometric features. A lower training loss or a gain over a restricted baseline does not, by itself, answer that question.

## Distinct interventions in the research program

| Study | Intervention | Evidence and scope |
| --- | --- | --- |
| Direct reconstruction training | Horizon-specific bandwidths and gate temperatures, with matched trainable local controls | Static forecasting with a fixed optimization budget; no established KAHM advantage over equally trained local alternatives |
| [Online adaptation](../adaptive_development/REPRODUCING.md) | Reference refreshes and residual-predictor updates with finite storage | A separate chronological development study; both capacity tiers fail the stated accuracy criteria |
| [Reconstruction departures](../reconstruction_mechanism/README.md) | Full-coordinate departures from affine reconstruction, radial feature bounds, and a centered ridge predictor | A separate matched experiment; improvement over the common basis, without establishing the prescribed advantage over all forecasting controls |

These interventions have different data, selection rules, and statistical units. Their outcomes are reported separately. The online study does not continually optimize the direct-training study's bandwidth and temperature parameters. The reconstruction-departure experiment evaluates a fixed geometry with a bounded correction and does not establish online geometry stability.

## Interpretation

A useful geometric representation must make relevant functions accessible to the chosen decoder. Deterministic features do not add information to an observed history, but they can change how efficiently a restricted predictor expresses a relationship. Exact feature-span arguments, finite-sample error, numerical stability, and measured computational cost address different parts of that claim.

The direct-training results support a conditional effect of forecast supervision within KAHM. They do not prove that every KAHM forecaster fails, and they do not justify assuming that additional parameters, references, or adaptation will improve prediction. A broader claim requires a specified target population, comparison set, selection budget, and evaluation procedure.

The [integrated manuscript](../tmlr_integration/README.md) presents the resulting theoretical and empirical account. Each original protocol and scientific checkpoint continues to identify the exact procedure that produced its corresponding results.
