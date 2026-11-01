# Scientific report

The report combines the completed forecasting comparison with the separately
audited certificate experiment. Its tables are read from the numerical exports;
the builder verifies their recorded SHA-256 digests before use. Scientific
inference remains defined by the frozen study protocols and analysis code.

The complete document specification and its figures are retained here. From
the forecasting repository root, the existing specification renders with:

```bash
python extensions/dimension_forecasting/reporting/render_report.py \
  extensions/dimension_forecasting/reporting/report_specification.json \
  KAHM_Dimension_Selection_Research_Report.pdf
```

The versions in the extension's `requirements.txt` include the reporting
dependencies. Rendering also requires the DejaVu Serif and DejaVu Sans fonts
at their standard Linux paths under `/usr/share/fonts/truetype/dejavu`.

To rebuild the figures and the complete specification from both analyses:

```bash
python extensions/dimension_forecasting/reporting/build_figures.py \
  extensions/dimension_forecasting \
  extensions/dimension_forecasting/reporting/figures
python extensions/dimension_forecasting/reporting/build_report_spec.py \
  extensions/dimension_forecasting \
  /path/to/risk_checkout/extensions/dimension_certificates \
  extensions/dimension_forecasting/reporting
```

The risk checkout must contain the completed `runs/20261101/analysis` exports
and the independent final audit. The figure paths in the specification are
relative to this directory. Rebuilding this presentation does not fit models,
change selection rules, or recompute the experimental endpoints.
