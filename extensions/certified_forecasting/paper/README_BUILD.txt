JMLR-target research draft: Learning Geometric Soft State Forecasts with Two-Sided Prediction Risk Certificates

Author-review draft, 29 October 2026. Includes the original results and a separate input-scaling follow-up with fresh synthetic confirmation and new PenDigits training subsamples on the same held-out writers.

Build: cd paper; latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Requirements: a normal TeX Live distribution with pdflatex, BibTeX, amsmath, mathtools, enumitem, microtype, booktabs, and lastpage. The official JMLR style (preprint mode) is included unchanged from https://github.com/JmlrOrg/jmlr-style-file.

Code/results branch: https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/certified-soft-state-forecasting/extensions/certified_forecasting
Original experimental code baseline: a8bcf6c8016499e1a4b64f7dd4196af43c60bd78
Draft pull request: https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/pull/1

Generate the new scaling tables from the released audited CSVs, from the extension directory:
python scripts/make_scaling_assets.py
The generator accepts --analysis and --confirmation for alternative complete analysis directories and writes only the new input_scaling_* assets. Original numerical tables and figures are unchanged from the attached 28 October manuscript package.

Reproduce the scaling schedule, from the extension directory after preparing PenDigits:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src
python experiments/prepare_pendigits.py
python experiments/run_input_scaling.py --output results/reproduced_input_scaling --workers 4
See the experiment documentation for scoped --phases, --campaigns, and --modes runs, comparison analysis, and the distinction between original and fresh-seed evidence. The fresh PenDigits fits do not introduce new test writers.

This is a complete scientific draft, not an accepted publication. Author affiliations and the inherited author list must be checked by the authors, who must also supply their funding/conflict disclosures and independently verify the derivations and novelty before submission. AI assistance and the actual empirical limitations are disclosed in the manuscript.
