# Reproducing the natural-series evaluation

The experimental source, configuration, roster, and numerical analysis are
fixed by `freeze.json`. The public M4 input files are distributed by the M4
organizers and can be recovered with the helper below. It checks each complete
file against the frozen SHA-256 identity and never requests official Test data.

```bash
python extensions/policy_augmentation_natural_20261101/reporting/fetch_inputs.py
```

The input files are `Hourly-train.csv`, `Weekly-train.csv`, and `M4-info.csv`.
Use the stage commands in `README.md` in a clean output checkout. Existing
execution outputs are append-only; do not replace a completed or failed task.
The archived numerical records permit independent scoring without refitting.

This experiment uses terminal segments of 48 previously unused series
identifiers from the public Train archive. Its selected horizons and rolling
origins differ from the official M4 competition evaluation. New identifiers
do not establish independence between the original anonymous data sources.
