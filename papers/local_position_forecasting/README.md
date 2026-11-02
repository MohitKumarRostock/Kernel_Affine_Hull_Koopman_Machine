# Local Position and Reconstruction Features for Forecasting

## A matched study of Kernel Affine Hull Machines

The manuscript studies whether local position and reconstruction features improve a direct affine readout of the same future observations. It reports all four primary comparisons from the matched G/P/K/R experiment, including the unsuccessful KAHM-versus-RBF superiority comparison. The supplementary material gives the simulation laws, implementation details, complete comparison tables, computational audit scope, and related-study results.

## Documents

`main.tex` contains the main article and its raw-feature-span proofs. `supplement.tex` contains the supporting protocol, results, ridge bound, and reproducibility specification. The bibliography and unmodified TMLR template files are included. Third-party template licensing is recorded in `LICENSE_TMLR.txt` and `TEMPLATE_PROVENANCE.json`.

Build both documents from this directory:

```bash
python scripts/build.py
```

The build requires Python 3 and a TeX distribution providing pdfLaTeX and the packages imported in the source. BibTeX is used when available; the supplied `.bbl` files also permit compilation without it. Document compilation uses the supplied static tables and does not regenerate simulation data, refit models, or reproduce the computational audit.

## Check the primary statistics

The `evidence/` directory contains the repetition-level losses and compact analysis records. With NumPy and SciPy installed, run:

```bash
python scripts/verify_primary.py --output primary_check.json
```

This check reconstructs all 128 paired repetition differences, the four Bonferroni-adjusted upper bounds, and 24 panel summaries. It checks their agreement with the supplied analysis tables. It is a check of saved-loss arithmetic, not an independent reconstruction of forecasts. The original recorded numerical environment is identified in the supplementary material and study-specific requirements.

## Scientific source and full regeneration

The forecasting implementation is under `extensions/soft_state_ablation_20261102/` in the parent research repository. Its original scientific source is commit `2b7d19b5b47f46b5b21b28d24960688f7969d96f`. It depends on the dimension-study runtime and the policy-study numerical solver. The manuscript directory alone is not a full simulation environment. The repository-level `REPRODUCIBILITY.md` specifies the complete regeneration procedure and distinguishes compact-record checks from model replay and scientific replication.

## Review and attribution

The PDFs use anonymous TMLR review formatting. This author/source package preserves software licenses, template attribution, public scientific source identities, and original evidence metadata. It is not an anonymized code archive: author-identifying source and license information must not be uploaded as anonymous supplementary material without an appropriate separate review package. The default template header does not assert that the work has been submitted or accepted.
