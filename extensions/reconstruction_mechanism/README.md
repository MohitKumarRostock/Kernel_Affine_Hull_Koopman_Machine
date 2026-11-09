# Paired statistical comparisons

`analyze.py` provides the `paired_margin_contrast` reference function used by the [dimension-study analysis checks](../dimension_forecasting/analysis_code/verify_analysis.py). The check compares paired differences, Student statistics, adjusted bounds, and decisions on simulated numerical fixtures.

The function requires NumPy and SciPy. The source verifier checks its recorded SHA-256 identity. The supported forecasting studies and their evidence are listed in the [research index](../../docs/RESEARCH_INDEX.md).
