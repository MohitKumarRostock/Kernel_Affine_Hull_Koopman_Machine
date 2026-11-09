# Statistical analysis of the forecasting study

This bundle lets readers check the numerical evidence for adding log normalization constants and local-coordinate features to Kernel Affine Hull Machine (KAHM) features. It reconstructs the reported statistical comparisons from saved losses of the selected predictors. The study asks whether those additions improve forecasting with a specified predictor family, and how they compare with affine projection, kernel-weighted averaging, and raw-history controls.

The study identifier is **KAHM-SCORE-20261102-v1**. Its 26,624 saved loss rows cover four data-generating processes, 64 repetitions per data-generating process, 26 procedures, two selection rules and two evaluation splits. A procedure includes a feature construction, a predictor family and its validation-based setting selection. The comparisons assess these complete procedures; they do not isolate a change in one coordinate while holding every selected setting fixed.

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
| `evidence/selected_losses.csv.gz` | Saved losses and selected-setting metadata for every process/repetition/procedure/selection-rule/split combination. The checker reads the compressed file directly in memory. |
| 416 procedure summaries | Mean standardized mean squared error (SMSE) and sample standard deviation across 64 repetitions for each data-generating process, selection rule, split and procedure. |
| 2,560 paired differences | The candidate loss minus its reference loss in the same repetition, for all 40 primary comparisons. |
| 40 primary comparisons | Paired mean differences, sample standard deviations, Student intervals, sign classifications, relative gains, repetition counts and the other numeric fields of `primary_contrasts.csv`. |
| Selection consistency | The complete set of uniquely identified records; the same selected predictor on C and D; allowed dimension caps, penalties and bandwidths; the shared-cap rule; and equality across selection rules when the selected cap is the same. |
| 17 record identities | Exact SHA256 and byte-size checks, including the supplied protocol, numerical audit and execution receipts. Their presence does not mean every computation described in them has been repeated. |

**SMSE** is squared prediction error standardized separately for each forecast horizon by the training-target variance (with floor `1e-12`), then averaged over horizons 1, 4 and 12. The checker verifies each saved three-horizon average. The training targets and their variances are not included, so it cannot independently recompute the standardization from raw predictions.

The primary `independent` selection rule lets each procedure choose its dimension cap and predictor settings using split B. The secondary `shared` rule takes the cap selected by the affine-local linear procedure (`A__local__linear`), then selects each procedure's predictor settings at that cap. Raw histories do not have a dimension cap. All candidates were specified to be fitted on A, selected on B and evaluated without an A+B refit. C is the diagnostic split; D supplies the primary test-set comparisons. These roles come from the recorded protocol. The saved selected rows allow consistency checks, but do not expose all B candidate scores or independently prove that each recorded choice was the B optimum.

The 40 primary comparisons use **D and the independent selection rule only**. For each data-generating process and predictor family, they compare KAHM memberships with and without each feature augmentation, and compare KAHM local-coordinate features with the affine projection, kernel-weighted averaging, and raw-history controls. The inference unit is a paired repetition difference: there are 64 per comparison. The code does not pool data-generating processes, windows, forecast horizons or selection rules as additional independent observations.

The interval is the paired mean plus or minus `t_(63, 1 - 0.05/(2*40))` times the paired sample standard deviation divided by `sqrt(64)`. The critical value is approximately `3.3796264638754705`. An interval wholly below zero gives `candidate_lower_mean_loss`; one wholly above zero gives `candidate_higher_mean_loss`; otherwise the outcome is `unresolved`. An unresolved comparison is not evidence of equivalence. These Bonferroni-adjusted Student intervals have approximate coverage, as specified in the protocol. No practical-margin or equivalence test is reconstructed. The pinned records have no zero-variance primary contrast; the checker rejects such a case rather than applying a Student interval to it.

Relative gain is `100 * (1 - mean_candidate_loss / mean_reference_loss)`, with positive values indicating lower candidate loss. This descriptive percentage is not itself a separate significance test. Numerical comparisons with the supplied tables use relative tolerance `2e-10` and absolute tolerance `2e-12` to allow summation-order differences; classifications and identities must match exactly. These tolerances concern summary arithmetic, not the forecast-validation tolerances or prediction error.

## Evidence identity and limits

`MANIFEST.json` records every bundle file other than itself, and separately records the logical identity of all 17 evidence files. The large selected-loss CSV is stored as deterministic gzip (`mtime=0`, no embedded filename). Its compressed hash identifies the distributed file; its decompressed hash identifies the decompressed CSV:

```text
selected_losses.csv.gz  1,644,630 bytes
SHA256 9ac64cd11e165c7cffcabfec892b3f5f9ccb2756aeae9134a08d0c550687930a
selected_losses.csv     6,003,675 bytes after decompression
SHA256 f2cf659e19a58aadc9ff74aeefaff15568f19063367e1fb0f82263d90b3ea81b
```

The other 16 evidence files have the byte-size and SHA256 identities listed in the manifest. The checker pins the identities of all evidence files and checks the bundle manifest. Hashes establish content identity; they do not establish scientific correctness or the order of execution.

`check_saved_losses.py` independently implements the statistical analysis of the saved per-run losses. The experiment's `verify_saved_analysis.py` is unavailable. The execution receipt identifies the experiment source manifest by SHA256 `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`; that manifest and the complete source files it identifies are unavailable. The bundle manifest identifies the supplied analysis files and is distinct from that experiment source manifest.

This check does not regenerate episodes, reconstruct the complete seed/task list, fit models, reproduce features or forecasts, repeat the computational audit, or supply the unavailable numerical archives. Auxiliary horizon, selection, resource and secondary-comparison tables are supplied and hash-checked, but are not recalculated by this command. Execution receipts document the computations they report; their successful status does not mean the supplied checker repeats every computation. 

The separate 24-repetition local-coordinate study is not part of this bundle. Its capacity comparison CSV has a different protocol and is deliberately excluded from the 17 central-study records.
