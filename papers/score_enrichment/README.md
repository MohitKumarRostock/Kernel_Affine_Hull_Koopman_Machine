# Forecasting with reconstruction-based features

The study examines what information normalized reconstruction affinities preserve and whether adding normalization constants or local coordinates improves fitted forecasts. It compares Kernel Affine Hull Machines with affine reconstruction, kernel-weighted averaging and raw-history predictors. Input reconstruction, numerical conditioning and prediction accuracy are separate questions.

`main.tex` contains the article and mathematical proofs. `supplement.tex` contains the numerical specification, all 40 primary intervals, procedure means under both hyperparameter-selection rules, related studies and evidence-availability information. The directory includes a bibliography, 14 static tables and TMLR style files.

From this directory, run:

```bash
python scripts/build.py
```

The command produces `main.pdf` and `supplement.pdf` from the supplied document inputs. It does not generate observations or fit models. [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) identifies the build inputs, and [DOCUMENT_BUILD.json](DOCUMENT_BUILD.json) records the PDF checksums and build diagnostics. PDF bytes can differ between TeX environments.

The [experiment directory](../../extensions/score_enrichment_20261102/) supplies the protocol and primary-summary table. The [reproduction guide](../../REPRODUCIBILITY.md) identifies the saved per-run losses, statistical checker and missing materials required to regenerate the experiment.

The public source contains author attribution and third-party notices. Review-format headers do not indicate submission or acceptance.
