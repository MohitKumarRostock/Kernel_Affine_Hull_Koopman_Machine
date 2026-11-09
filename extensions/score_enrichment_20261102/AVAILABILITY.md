# Data, code and verification

The central experiment is identified by protocol `KAHM-SCORE-20261102-v1`. This directory supplies its protocol, complete 40-row primary-summary table and summary-arithmetic checker. The [per-run analysis package](../../reproduction/score_enrichment/) supplies 17 experiment records, including all 26,624 selected-loss rows and an independent checker that reconstructs 416 summaries, 2,560 paired differences and all 40 primary comparisons.

The statistical checker is conditional on the saved losses. It does not generate observations, construct features, fit predictors, reproduce forecast arrays or rerun the recorded computational audit. The experiment's complete implementation, exact task list, complete environment record, fitted models and forecast arrays are unavailable.

## Recorded execution-package identifiers

The experiment records identify the following files. These identifiers establish the expected file identities; the files are not supplied and no public download location is provided.

| Package | Bytes | SHA256 |
| --- | ---: | --- |
| `KAHM_Step8_Repository_Update.zip` | 6348062 | `8ef7bca05707c86f87922de1698719e971d0ce2b182c2ce264cd6a3c6c414637` |
| `KAHM_Step8_Complete_Author_Package.zip` | 10209590 | `b48ea9fbdcaa9c17d47e08a1a99b371b9ab47735b05e32b4112ddf8dd7de87fe` |

The recorded source-manifest SHA256 is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. The manifest for the per-run statistical-analysis package identifies its own files and does not supply this missing experimental source manifest.

## Numerical archives

The experiment records describe sixteen episode, model and prediction archives totaling 9.024 GiB. The complete set is unavailable for independent integrity checks. Archive volume is a distribution constraint, separate from the unavailable implementation, exact task list and complete environment record. Regeneration would require that execution source and its inputs; saved model outputs alone are not a replacement for them.

The recorded computational audit uses shared feature-encoding code and shared numerical routines. Its scope and limitations remain relevant when interpreting its results. The [reproduction guide](../../REPRODUCIBILITY.md) gives the supported commands and identifies the inputs required by each task.
