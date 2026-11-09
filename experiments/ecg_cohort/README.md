# Class-risk lower bounds on the PTB-XL evaluation dataset

This experiment evaluates lower bounds on the best mean squared error in a predictor class on the fixed
8,192-patient evaluation dataset. It compares three separately
selected feature maps of 2, 6, and 14 coordinates, two partition rules, and
block sizes 32, 64, and 128. These feature maps do not form a nested family.

## Reproduce the numerical results

From the repository root:

```bash
python experiments/ecg_cohort/replay.py --output results/ecg_cohort.json
```

The command reconstructs the fixed partitions, verifies outward integer
distance bounds from the recorded coordinate-wise feature bounds, and evaluates every one of
the 2,688 stored signed dual vectors in exact integer/rational arithmetic.
For each group it chooses among that group's dual bound, bounds assembled
from smaller groups and zero. It checks all 18 configuration results and
recomputes the reference threshold from the retained targets and reference predictions. No solver,
model fitting, waveform download, or reserved-patient access occurs.

The two-coordinate class is excluded at the specified MSE threshold. The
six- and fourteen-coordinate classes remain unresolved. The replay also
recomputes a necessary lower bound on the Lipschitz constant from the same stored
optimization records; values above that bound do not by themselves establish adequacy.

## Evidence and scope

The `records` directory contains evaluation dataset targets, stored predictions and
coordinate-wise feature bounds, all stored signed dual vectors, row indices, and original
exact block values. Repeated stored distance matrices and embedded dual
copies are omitted because the replay reconstructs them. `protocol` contains
the original evaluation dataset and certificate designs. `source` retains the data-generation code, exact evaluator, independent auditor and source modules
used by the experiment.
`provenance` links the compact records to the source files and checksums.

The exact result is conditional on the recorded bounds on feature values. It concerns these observations and does not establish a
future-patient population guarantee or verification of every step of feature generation. Full model fitting and upstream feature generation are not rerun by
this command. Their original records are in `step19_reproducibility.zip`.

Data originate from PTB-XL version 1.0.3, distributed by PhysioNet under
CC BY 4.0: https://physionet.org/content/ptb-xl/1.0.3/.
The dataset release has DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45);
Wagner et al. describe PTB-XL in *Scientific Data* 7, 154 (2020),
DOI [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
The replay needs Python and NumPy and performs no network access.
