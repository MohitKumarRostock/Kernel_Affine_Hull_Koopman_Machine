# Rebuild the report

This reporting script was written after the pilot code/configuration freeze.
It reads saved outcomes; it does not fit or select a forecasting model.
Its figures and prose are post-run reporting, not preregistration.

From the package root, with the scientific and report dependencies installed:

```bash
python report_support/build_report.py --project . --output reports/KAHM_Research_Continuation.pdf
```

The script also writes PILOT_FINDINGS.json and a comparison figure next to the PDF.
The original numerical source and checks remain under pilot_results/source_snapshot.
