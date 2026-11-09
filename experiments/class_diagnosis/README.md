# Predictor-class capacity and fitted-model error on fixed datasets

This study asks whether unsuccessful fitted predictors indicate a predictor class that cannot meet the required error threshold. It checks 54 fixed classes on the three related Bike Sharing targets: casual, registered, and total rentals. Every class uses the same 172 decision observations for its target, a fixed stored feature map, output range `[0, 1]`, and Lipschitz constant 2 with respect to Euclidean distance. Original targets outside that range remain in the loss.

All 54 classes contain feasible prediction vectors whose mean squared error meets the specified threshold. In 39 classes, both fitted predictors exceed that threshold. All 108 predictors have a strictly positive lower bound on their empirical excess risk relative to their class optimum. All 54 classes also meet the threshold using feasible predictions constructed directly from the targets and Lipschitz constraints, without an optimizer proposal. Constant predictions meet the threshold in none.

## Reproduce the numerical evidence

Use Python 3.10 or newer and NumPy. From the repository root:

```bash
python -m pip install -r experiments/class_diagnosis/requirements.txt
python experiments/class_diagnosis/replay.py --output /tmp/class-diagnosis-replay
```

The command verifies package checksums and the correspondence between input files and their recorded hashes, then calls the independent arithmetic auditor for every class. It reconstructs all 794,124 pairwise distance bounds, checks the three feasible primal candidates per class, evaluates lower bounds from signed dual vectors, recomputes all class-risk brackets, and recomputes all 108 predictor-excess intervals. Exact arithmetic can take several minutes. No optimizer or model fit runs.

The output directory contains:

- `summary.json`: verification counts, per-target outcomes, exact class checks, and the numerical-assurance scope.
- `class_risk.csv`: all 54 exact class-risk brackets and tolerances.
- `head_excess.csv`: all 108 recorded predictor-risk intervals and recomputed excess intervals.

Expected totals are 54 adequate classes, 39 classes with both predictors failing, 108 positive excess intervals, 54 adequate target-based feasible vectors, and zero adequate constants. The maximum class-risk bracket width is approximately `2.181e-14` in normalized MSE.

## Interpretation and limits

This is a deterministic analysis of the specified observations. The feasible vectors are constructed using their target labels and do not measure the performance of predictors trained independently of those labels. The three targets are related: total rentals equal casual plus registered rentals. They do not provide three independent application replications.

Stored binary64 feature coordinates are treated as exact inputs. The replay verifies distance, feasibility, dual, risk-bracket, and excess arithmetic under that premise; it does not enclose upstream feature-construction error or establish future-risk coverage. It cannot identify the cause of a fitted predictor's excess.

The public-screening audit supplies the checks of predictor-class membership and achieved risk. This package verifies the corresponding file hashes and independently recomputes excess risk; it does not regenerate the fitted predictors' forecast arrays. The checksums and audit records establish consistency among the supplied files, without independently authenticating their execution history.

## Package contents and data provenance

| Path | Contents |
|---|---|
| `inputs/decision/` | The three evaluation datasets, 54 fixed feature arrays, map identities, specified tolerances and fitted-predictor metadata. |
| `inputs/prior/` | Public-screening protocols, model checksums, numerical audit and public-source metadata. |
| `records/classes/` | All 54 complete-graph primal and dual optimization records. |
| `records/result.json` | Original complete aggregate result with all 108 predictor rows. |
| `records/original_numerical_audit.json` | Retained independent audit of the original run. |
| `protocol/` | Scientific protocol, checksums and recorded software requirements. |
| `source/` | Scientific data-generation code, input preparation and independent auditors. |
| `MANIFEST.json` | File sizes and SHA-256 hashes for this package. |

The underlying data are [Bike Sharing, UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing), by Hadi Fanaee-T (2013), DOI `10.24432/C5W894`. This package contains the fixed inputs needed for the analysis of a fixed evaluation dataset. The original generation code is retained for scientific inspection and provenance; `replay.py` is the supported offline reproduction entry point. Full regeneration of representations and fitted predictors from raw data requires the public-screening source and data and is not performed by this replay.
