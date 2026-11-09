# Numerical reporting resources

This directory contains figure-generation code that reads the saved numerical records. The builders verify the recorded SHA-256 digests of their numerical inputs. Statistical inference remains defined by each study's protocol and analysis code.

From the repository root, regenerate the forecasting figures with:

```bash
python extensions/dimension_forecasting/reporting/build_figures.py \
  extensions/dimension_forecasting \
  extensions/dimension_forecasting/reporting/figures
```

The extension's `requirements.txt` specifies the reporting dependencies. Figure generation reads saved results and does not fit models or change selection rules. The [study guide](../README.md) describes restoration, fitting, statistical analysis, and independent checks.
