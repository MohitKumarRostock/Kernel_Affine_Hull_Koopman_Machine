# Archived forecasting experiment: statistical reconstruction

This bundle lets readers check the numerical evidence for adding normalization-scale and local-position information to Kernel Affine Hull Machine (KAHM) state features. It reconstructs the reported statistical comparisons from saved losses of the selected predictors. The study asks whether those additions improve forecasting with a specified predictor family, and how they compare with affine, positive-kernel and raw-history controls.

The study identifier is **KAHM-SCORE-20261102-v1**. Its 26,624 saved loss rows cover four generating laws, 64 repetitions per law, 26 procedures, two selection rules and two assessment splits. A procedure includes a feature construction, a predictor family and its validation-based setting selection. The comparisons assess these complete procedures; they do not isolate a change in one coordinate while holding every selected setting fixed.

## Run the check

From the repository root, with Python 3.12 (validated with Python 3.12.14):

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The command prints a JSON report and does not write files. To also save the rebuilt tables, choose a new directory outside this bundle:

```bash
python3 reproduction/score_enrichment/check_saved_losses.py --output /tmp/kahm-score-analysis
```

An existing output directory is rejected. A successful run reports `PASS`, 416 procedure summaries, 2,560 paired differences and 40 primary comparisons. It reconstructs **16 lower-loss, 7 higher-loss and 17 unresolved comparisons**, using the candidate-minus-reference sign convention. The output directory contains three `rebuilt_*.csv` files and `verification.json`; the supplied records are never overwritten.

The checker uses SciPy only for the Student critical value. Its pinned requirements specify the NumPy/SciPy versions used here. To run its small integrity and decision tests:

```bash
python3 -B -m unittest discover -s reproduction/score_enrichment/tests -v
```

## What is reconstructed

| Input or check | Meaning |
| --- | --- |
| `evidence/selected_losses.csv.gz` | Saved losses and selected-setting metadata for every law/repetition/procedure/selection-rule/split combination. The checker reads the compressed file directly in memory. |
| 416 procedure summaries | Mean standardized mean squared error (SMSE) and sample standard deviation across 64 repetitions for each law, selection rule, split and procedure. |
| 2,560 paired differences | The candidate loss minus its reference loss in the same repetition, for all 40 primary comparisons. |
| 40 primary comparisons | Paired mean differences, sample standard deviations, Student intervals, sign classifications, relative gains, repetition counts and the other numeric fields of `primary_contrasts.csv`. |
| Selection consistency | The complete unique record roster; the same selected predictor on C and D; allowed dimension caps, penalties and bandwidths; the shared-cap rule; and equality across selection rules when the selected cap is the same. |
| 17 evidence-file identities | Exact SHA256 and byte-size checks for the protocol, saved losses, summaries, audit, and execution records. The file-identity check does not execute the computations described by those records. |

**SMSE** is squared prediction error standardized separately for each forecast horizon by the training-target variance (with floor `1e-12`), then averaged over horizons 1, 4 and 12. The checker verifies each saved three-horizon average. The training targets and their variances are not included, so it cannot independently recompute the standardization from raw predictions.

The primary `independent` selection rule lets each procedure choose its dimension cap and predictor settings using split B. The secondary `shared` rule takes the cap selected by the affine-local linear procedure (`A__local__linear`), then selects each procedure's predictor settings at that cap. Raw histories do not have a dimension cap. All candidates were specified to be fitted on A, selected on B and evaluated without an A+B refit. C is the diagnostic split; D supplies the primary terminal comparisons. These roles come from the recorded protocol. The saved selected rows allow consistency checks, but do not expose all B candidate scores or independently prove that each recorded choice was the B optimum.

The 40 primary comparisons use **D and the independent selection rule only**. For each law and predictor family, they compare local-position enrichment with memberships, normalization-scale enrichment with memberships, and KAHM local-position features with the affine, positive-kernel and raw-history controls. The inference unit is a paired repetition difference: there are 64 per comparison. The code does not pool laws, windows, forecast horizons or selection rules as additional independent observations.

The interval is the paired mean plus or minus `t_(63, 1 - 0.05/(2*40))` times the paired sample standard deviation divided by `sqrt(64)`. The critical value is approximately `3.3796264638754705`. An interval wholly below zero gives `candidate_lower_mean_loss`; one wholly above zero gives `candidate_higher_mean_loss`; otherwise the outcome is `unresolved`. An unresolved comparison is not evidence of equivalence. These Bonferroni-adjusted Student intervals have approximate coverage, as specified in the protocol. No practical-margin or equivalence test is reconstructed. The pinned records have no zero-variance primary contrast; the checker rejects such a case rather than applying a Student interval to it.

Relative gain is `100 * (1 - mean_candidate_loss / mean_reference_loss)`, with positive values indicating lower candidate loss. This descriptive percentage is not itself a separate significance test. Numerical comparisons with the supplied tables use relative tolerance `2e-10` and absolute tolerance `2e-12` to allow summation-order differences; classifications and identities must match exactly. These tolerances concern summary arithmetic, not the original forecast-validation tolerances or prediction error.

## Evidence identity and limits

`MANIFEST.json` records every bundle file other than itself, and separately records the original logical identity of all 17 evidence files. The large selected-loss CSV is stored as deterministic gzip (`mtime=0`, no embedded filename). Its compressed hash identifies the distributed file; its decompressed hash identifies the unchanged original CSV:

```text
selected_losses.csv.gz  1,644,630 bytes
SHA256 9ac64cd11e165c7cffcabfec892b3f5f9ccb2756aeae9134a08d0c550687930a
selected_losses.csv     6,003,675 bytes after decompression
SHA256 f2cf659e19a58aadc9ff74aeefaff15568f19063367e1fb0f82263d90b3ea81b
```

The other 16 evidence files preserve their supplied bytes. The checker pins the logical identities of all 17 evidence files and checks the bundle manifest. Hashes establish content identity; they do not establish scientific correctness or authenticate execution order.

`check_saved_losses.py` independently implements the saved-loss statistical calculation. The experiment's `verify_saved_analysis.py`, its complete source snapshot, and its source manifest are unavailable. The execution record identifies that experimental source manifest by SHA256 `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. The bundle manifest identifies the files supplied here and does not replace that experimental source identity.

The complete generator, encoding and fitting implementation, seed/task roster, episode arrays, fitted states, and predictions are unavailable. The checker therefore cannot regenerate episodes, fit models, reproduce features or forecasts, or repeat the computational audit. Auxiliary horizon, selection, resource, and secondary-comparison tables are preserved and hash-checked, but are not reconstructed by this command. Audit and execution records document their respective calculations; the checker reads them for identity verification.

The separate 24-repetition capacity study has a different protocol. Its [comparison summary](../../related_evidence/capacity_primary_contrasts.csv) is not one of the 17 archived evidence records. The [archived forecasting specification](../../docs/ARCHIVED_FORECASTING.md) documents the observation laws, fit and selection conventions.
