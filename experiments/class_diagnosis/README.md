# Minimum empirical risk and fitted-predictor error on Bike Sharing

## Design and results

The experiment evaluates 54 Lipschitz predictor classes for casual, registered, and total rentals. Each target has 172 fixed evaluation observations. A class specifies stored feature coordinates, output interval `[0, 1]`, and Lipschitz constant 2 under Euclidean distance. Stored coordinates include multiplication by the reciprocal median of positive pairwise Euclidean distances between the unscaled feature vectors of 32 training-history probes. The scale is one if no positive distance exists. Targets outside the output interval remain in the losses.

Verified feasible prediction vectors establish that all 54 classes contain decoders meeting their empirical mean squared error (MSE) thresholds. Both fitted predictors fail in 39 classes. All 108 fitted predictors have a strictly positive lower bound on empirical excess risk: their empirical MSE minus the minimum empirical MSE over their class. Initializing the feasible-prediction construction from assessment targets also meets all 54 thresholds without an optimizer proposal; constant predictions meet none. The maximum interval width for minimum empirical risk is approximately `2.181e-14` in normalized MSE.

## Reproduction

Use Python 3.10 or newer and NumPy. Run from the repository root:

```bash
python -m pip install -r experiments/class_diagnosis/requirements.txt
python experiments/class_diagnosis/replay.py --output /tmp/class-diagnosis-replay
```

The command checks package and input hashes, reconstructs 794,124 pairwise distance bounds, checks three feasible prediction candidates per class, evaluates dual lower bounds, and recomputes all 54 minimum empirical-risk intervals and 108 empirical excess-risk intervals. It uses an independent arithmetic checker and does not run an optimizer or fit models.

| Output | Contents |
| --- | --- |
| `summary.json` | Verification counts, per-target results, exact checks, and numerical assumptions |
| `class_risk.csv` | 54 minimum empirical-risk intervals and MSE thresholds |
| `head_excess.csv` | 108 predictor-risk intervals and recomputed empirical excess-risk intervals |

## Interpretation and numerical assumptions

Feasible predictions use evaluation targets. They establish what a class can achieve on those observations, not the accuracy of a predictor trained independently of those targets. Total rentals equal casual plus registered rentals, so the three targets are not independent application replications.

Stored binary64 feature coordinates are treated as exact inputs. The checker verifies distance bounds, feasibility, dual objectives, and empirical-risk arithmetic under this assumption. It does not bound feature-construction error or identify the cause of a fitted predictor's excess risk.

Predictor membership and achieved-risk checks come from the public-screening analysis. This package verifies their file hashes and recomputes excess risk; it does not regenerate those fitted predictions. Checksums establish file consistency, not independent authentication of execution history.

## Files and provenance

| Path | Contents |
| --- | --- |
| `inputs/decision/` | Three evaluation datasets, 54 feature arrays, representation identities, thresholds, and predictor metadata |
| `inputs/prior/` | Public-screening protocols, model hashes, numerical checks, and data-source metadata |
| `records/classes/` | 54 complete-graph primal and dual optimization records |
| `records/result.json` | Original aggregate result, including 108 predictor rows |
| `records/original_numerical_audit.json` | Original independent numerical checks |
| `protocol/` | Protocol, checksums, and software requirements |
| `source/` | Data-generation, input-preparation, and independent checking code |
| `MANIFEST.json` | Package file sizes and SHA-256 hashes |

The data are [Bike Sharing, UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing), by Hadi Fanaee-T (2013), DOI `10.24432/C5W894`. `replay.py` is the supported offline entry point. Regenerating features and fitted predictors from raw data requires the public-screening source and data.
