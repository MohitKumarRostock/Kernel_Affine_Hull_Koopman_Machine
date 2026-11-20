# Population-risk confidence bounds with independent episodes

## Design and results

The experiment compares two lower-confidence-bound procedures for minimum population mean squared error (MSE) over a specified Lipschitz predictor class. The first averages normalized minimum-loss bounds from disjoint observation blocks. The second uses feasible dual values from those blocks to bound minimum empirical loss over the full calibration dataset before statistical calibration.

Each of two bounded synthetic settings has 12 complete training repetitions and two nested feature sets per repetition. At 1,024 calibration observations, both procedures resolve all 12 repetitions in each setting. At 256 observations, reuse of the dual values excludes both classes in 11 additional repetitions when the target is independent of the history. Neither procedure resolves the history-dependent setting at that size. Outputs include all three calibration sizes and unresolved decisions.

## Reproduction

Run from the repository root:

```bash
python experiments/controlled_calibration/replay.py --output results/controlled_calibration.json
```

The command regenerates 4,096 calibration observations for each learning case from fixed random streams, recomputes selected fitted predictions, evaluates 190,464 stored block optimization records, and reconstructs 1,008 one-sided confidence endpoints and paired decision counts. It does not call an optimizer or refit models. Inputs use NumPy arrays and JSON, not pickles.

## Files and numerical assumptions

`records/` contains dual vectors, candidate prediction vectors, numerical diagnostics, features, observations, fitted coefficients, and expected endpoints. `protocol/` contains the fixed design. `source/` contains the data-generation code, independent checker, model code, and numerical routines. `provenance/` records source identities and checksums.

The calculations use floating-point safeguards and recorded feature checks. Reproduction does not enclose every numerical error in feature construction and subsequent arithmetic, and it does not repeat feature construction. The population interpretation requires valid feature values, conservative optimization and loss bounds, and fitted predictors belonging to their prescribed classes. The original archive, `geometry_step17_reproducibility.zip`, also contains all training candidates and coordinate-wise feature bounds.

Inputs are synthetic. The checker requires Python, NumPy, and SciPy and makes no network requests. Original training additionally uses scikit-learn and the recorded numerical software.
