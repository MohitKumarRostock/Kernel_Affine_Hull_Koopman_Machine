# Joint-selection manuscript displays

Generated only after complete analysis and a passed matching independent audit. Table fragments have no captions or labels. `numbers.tex` defines the JS macro namespace; `report_metrics.json` records every macro value and the figure data. For each primary suffix, `JSRelativeLossChange` is the negative of `JSGain`, so a positive value denotes a percentage loss increase. Percentage gains and loss changes use four decimals below 1% in magnitude and two decimals otherwise.

- `primary_table.tex`: all six simultaneous primary criteria. Gain is 100(1-candidate/reference); U is the one-sided margin-adjusted paired-t upper bound, with six-comparison Bonferroni adjustment.
- `aggregate_table.tex`: all 16 forecast means on both panels. P-shared geometry uses P's validation choice; Own uses each arm's validation choice.
- `selection_table.tex`: independent P/K/R scaler counts and percentages (192 nonlinear tasks or 64 linear tasks) and median selected cap, not median effective local rank.
- `descriptive_table.tex`: equal-noise per-law independent-selection comparisons; no per-law primary inference.
- `independent_K_joint_vs_standard.pdf` and `.png`: exact same-cohort means, with separate y scales.

The figure, per-law table and selection frequencies are descriptive. The two declared joint-versus-standard K primary contrasts remain in the six-comparison family. The asset builder verifies CSV/reporting arithmetic; raw-array replay belongs to the bound independent audit.
