# PenDigits run accounting

The successful run is `pendigits_v1b`, using the fixed
`configs/pendigits_v1.json` forecasting protocol. All 240 candidates were fitted
and frozen before the test cohort was scored. All six seed/coordinate families,
including their unselected models, are retained.

An earlier attempt in `results/pendigits_v1` stopped before test prediction
because the loader correctly found that the 32-point eligibility filter leaves
only 13 of the original 14 held-out writers. Writer 5 has no first pen-down
stroke with at least 32 points. The failed attempt's frozen candidates and
failure record are preserved. The correction changes the eligible-writer count
to its observed value and uses 13 in the confidence calculation. It does not
change the 32-point filter or tune anything using a test loss.

The dataset does not establish iid writer sampling. The raw cohort prediction
errors are descriptive. The additional population UCBs are explicitly
conditional on an iid-writer assumption and are practically uninformative at
the reported tolerances. All windows of one writer are aggregated into one
score before the confidence calculation.

Verification performed on the completed run:

- 240 candidate results and 9,360 candidate-writer-horizon loss rows.
- Every family has 40 candidates; six coordinate/seed families were frozen.
- All input source hashes still matched the run manifest at verification.
- Recomputing every candidate/horizon cohort MSE from `writer_losses.csv`
  differed by at most 1.3877787807814457e-17.
- No recorded block norm exceeds 4 (tolerance 1e-9).
- No model reports failure to converge.
- Largest recorded projected-optimization gap: 9.950945423081166e-08.

Runtime recorded by this run: 39.5506 seconds. Runtime is hardware dependent.
