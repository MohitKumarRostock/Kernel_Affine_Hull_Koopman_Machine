# Fixed-geometry and target-changing experiments

This directory implements two distinct synthetic investigations of information-equivalent reconstruction-score encodings. The [protocol](../docs/PROTOCOLS.md) specifies the sampling laws, reference fitting, feature construction, regularization and comparison families. The [scientific reference](../docs/SCIENTIFIC_REFERENCE.md) defines the representations and mathematical quantities.

`run_ablation.py` generates both experiments using fixed seeds and binary64 numerical fitting, with four spawned workers and single-threaded numerical libraries. It writes the four loss and diagnostic CSV files, the 12 primary comparisons and execution metadata into its own directory. `summarize.py` checks rosters and selected matched-df records, recomputes the paired intervals, and writes the descriptive summaries and verification JSON. Use the root reproduction driver to regenerate into a separate directory without modifying the distributed records.

```bash
python reproduce.py check
python reproduce.py replay --output runs/replay
```

Run from the root of the scientific distribution. Primary inference is limited to the declared 12-comparison family. Other contrasts, complexity comparisons and target-changing results are descriptive. See [Results](../docs/RESULTS.md) and [Reproduction](../REPRODUCIBILITY.md).
