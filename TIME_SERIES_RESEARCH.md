# KAHM for time-series research

Kernel Affine Hull Machines (KAHMs) provide geometric features for finite
predictors. This repository studies **when those features help forecasting and
whether their improvement justifies their additional cost**.

The current research extension preserves a raw-history forecast and tests an
optional KAHM residual correction against matched affine and locality controls.
It varies training data and raw-decoder capacity separately and reports both
favorable and unfavorable results. KAHM features are deterministic functions of
the observed history; their role here is to help a specified fitting procedure
use predictive structure.

## Start here

| Resource | Purpose |
|---|---|
| [Forecasting-utility study](extensions/forecasting_utility/README.md) | Protocol, code, saved predictions, independent audit, and reproduction commands |
| [Revised manuscript PDF](extensions/tmlr_integration/review/KAHM_Forecasting_Utility_Manuscript.pdf) · [editable source](extensions/tmlr_integration/manuscript/main.tex) | Unsubmitted research draft integrating the original studies and the new extension |
| [Quick forecasting example](examples/forecasting_utility.py) | Small chronological example of a preserved raw forecast plus an optional geometric correction |
| [KAHM classification entry repository](https://github.com/MohitKumarRostock/KAHM) | Installation and the separate classification example with fair baselines |

The forecasting extension is a bounded research study. It does not establish
universal superiority, Bayes-risk improvement from deterministic features, or
the necessity of a more complex representation class. The separate risk-
certificate manuscript addresses class-level questions under explicit
assumptions.

## Preserved Koopman research

The [original repository README](README.md) and citation metadata remain unchanged
because they are part of a verified historical checkpoint. Current forecasting
evidence and entry points are described above.
