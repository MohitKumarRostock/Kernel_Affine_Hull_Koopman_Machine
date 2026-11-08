# Finite-cohort exclusion on PTB-XL

This experiment evaluates class-risk lower witnesses on the fixed
8,192-patient cohort used by the manuscript. It compares three separately
selected geometries of 2, 6, and 14 coordinates, two partition rules, and
block sizes 32, 64, and 128. The geometries are not a nested feature family.

## Reproduce the manuscript results

From the repository root:

```bash
python experiments/ecg_cohort/replay.py --output results/ecg_cohort.json
```

The command reconstructs the frozen partitions, verifies outward integer
distance bounds from the retained feature boxes, and evaluates every one of
the 2,688 newly proposed signed duals in exact integer/rational arithmetic.
It reconstructs the nested choice among new, child, and zero witnesses,
checks all 18 configuration results, and recomputes the unchanged reference
threshold from the retained targets and reference predictions. No solver,
model fitting, waveform download, or reserved-patient access occurs.

The two-coordinate class is excluded at the unchanged MSE threshold. The
six- and fourteen-coordinate classes remain unresolved. The replay also
recomputes the necessary sensitivity budget from the same fixed witnesses;
this is a necessary condition and does not establish adequacy above it.

## Evidence and scope

The `records` directory contains cohort targets, stored predictions and
feature boxes, all newly proposed signed duals, row indices, and original
exact block values. Repeated stored distance matrices and embedded dual
copies are omitted because the replay reconstructs them. `protocol` contains
the original cohort and certificate designs. `source` retains the original
producer, exact evaluator, independent auditor, and consumed source modules.
`provenance` links the compact records to the original source files and seals.

The exact claim is conditional on the inherited canonical-feature
enclosures. It concerns these observations and does not establish a
future-patient population guarantee or an end-to-end feature-transform
proof. Full model fitting and upstream feature generation are not rerun by
this command. Their original records are in `step19_reproducibility.zip`.

Data originate from PTB-XL version 1.0.3, distributed by PhysioNet under
CC BY 4.0: https://physionet.org/content/ptb-xl/1.0.3/.
The manuscript contains the dataset citation. The replay needs Python and
NumPy and performs no network access.
