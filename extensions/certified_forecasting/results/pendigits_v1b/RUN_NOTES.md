# PenDigits experiment: cohort definition and numerical verification

The completed `pendigits_v1b` experiment uses `configs/pendigits_v1.json`.
Three fitting seeds and two scalar coordinates produce six candidate families,
each containing 40 models. All 240 candidates and their validation selections
were fixed before test scoring; unselected candidates are included in the
released records.

## Eligibility and sampling units

The first execution, retained under `results/pendigits_v1`, stopped before test
prediction because a check expected all 14 original test-writer groups to
remain after the 32-point first-stroke filter. Test writer 5 has no qualifying
first stroke. The completed execution uses the 13 eligible writers in its
loss aggregation and confidence calculation. The correction changes neither
the eligibility rule nor a parameter chosen using test error. The earlier
candidate records and exception are retained as experimental provenance.

The test cohort contains 1,751 eligible digit trajectories. Loss is averaged
over origins and strokes within each writer, and then equally across the
13 eligible writers. The two coordinate MSEs are pooled before taking the
square root. The dataset does not establish independent, identically
distributed writer sampling. Cohort prediction errors are descriptive; the
additional population upper bounds require that sampling assumption and do
not certify the reported tolerances. Fitting-seed variability is measured on
this same held-out cohort.

## Archived numerical checks

The completed experiment records:

- 240 candidate results and 9,360 candidate–writer–horizon loss rows;
- 40 candidates in each of six coordinate–seed families;
- agreement between the source files used for the execution and the hashes
  recorded in its manifest at the time of the archived check;
- maximum discrepancy `1.3877787807814457e-17` when recomputing every
  candidate–horizon cohort MSE from `writer_losses.csv`;
- no recorded block norm above 4, using numerical tolerance `1e-9`;
- convergence for every reported constrained fit, with maximum first-order
  optimality gap `9.950945423081166e-08`.

An independent scalar-record check reproduced the writer-level aggregation,
coordinate pooling, table rounding, and Hoeffding arithmetic from the retained
scalar records. This check did not refit models or establish the iid-writer
assumption. Historical source hashes identify the executed implementation and
remain unchanged after subsequent editorial revisions.

The recorded complete-run time is 39.5506 seconds. It is specific to the
execution environment and does not support a general runtime comparison.
