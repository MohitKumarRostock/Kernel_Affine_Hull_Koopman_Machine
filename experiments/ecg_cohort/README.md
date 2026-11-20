# Minimum empirical-risk lower bounds on PTB-XL

## Design and results

The experiment bounds minimum mean squared error (MSE) over specified Lipschitz predictor classes on a fixed dataset of 8,192 patients. It evaluates separately selected 2-, 6-, and 14-coordinate feature maps, two partition rules, and block sizes 32, 64, and 128. The maps are selected separately, and their predictor classes are not assumed to be nested.

The two-coordinate class has a minimum empirical-risk lower bound above the specified MSE threshold. The six- and fourteen-coordinate classes remain unresolved. A separate calculation gives a necessary lower bound on the Lipschitz constant; exceeding that constant alone does not establish that the class meets the MSE threshold.

## Reproduction

Run from the repository root:

```bash
python experiments/ecg_cohort/replay.py --output results/ecg_cohort.json
```

The checker reconstructs partitions and outward integer distance bounds from recorded feature intervals. It evaluates all 2,688 stored dual vectors using exact integer/rational arithmetic. Each block uses the maximum of its dual lower bound, lower bounds combined from smaller blocks, and zero. The command checks all 18 configurations and recomputes the MSE threshold from targets and reference predictions. It requires Python and NumPy; no solver, model fitting, waveform download, or reserved-patient access occurs.

## Files and numerical assumptions

`records/` contains targets, predictions, coordinate-wise feature intervals, dual vectors, row indices, and exact block values. Redundant distance matrices and duplicate dual records are reconstructed. `protocol/` contains the dataset and optimization designs. `source/` contains data-generation code, exact evaluators, and independent checking code; `provenance/` records source identities and checksums.

The bounds are deterministic results for these observations, conditional on the recorded intervals enclosing the intended feature coordinates. The command does not regenerate features or establish future-patient risk guarantees. Original fitting and feature-generation records are in `step19_reproducibility.zip`.

## Data attribution

The data are [PTB-XL version 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), distributed by PhysioNet under CC BY 4.0, DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45). Wagner et al. describe the dataset in *Scientific Data* 7, 154 (2020), DOI [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
