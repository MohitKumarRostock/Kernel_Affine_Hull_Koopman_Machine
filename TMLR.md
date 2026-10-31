# Manuscript for TMLR review

**Kernel Affine Hull Representations for Time-Series Forecasting: Distinctness, Stability, and Predictive Limits** studies the relationship between geometric reconstruction and forecasting with a finite decoder.

- [Manuscript PDF](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript.pdf)
- [Editable LaTeX source](extensions/tmlr_integration/review/KAHM_Reconstruction_Prediction_Manuscript_Source.zip)
- [Build instructions](extensions/tmlr_integration/README.md)
- [Principal experimental findings](extensions/reconstruction_mechanism/FINDINGS.md)
- [Reproduction instructions](extensions/reconstruction_mechanism/README.md)
- [Document and supplement verification](extensions/tmlr_integration/TMLR_SUBMISSION_VALIDATION.json)

## Contribution and scope

The paper gives exact identities for normalized KAHM reconstruction, sufficient conditions for redundancy with a matched RBF feature span, and an explicit construction in which those spans differ. It then derives a uniform correction bound for bounded features followed by centered ridge regression. The mathematical claims state their assumptions separately from the empirical forecasting claims.

The principal experiment evaluates ten procedures in 192 task suites. The KAHM procedure improves the shared local-coordinate basis by 12.58% in aggregate nonlinear normalized mean squared error. Its 1.17% gain over matched RBF departures and 3.92% gain over raw-history RBF features do not establish the prespecified 3% improvement under the simultaneous analysis. Boosted trees and the validation-selected non-KAHM procedure have lower aggregate errors. These findings establish useful behavior in a specified comparison while limiting claims of practical KAHM superiority.

The manuscript also reports controlled representation experiments, natural-data evaluations, direct reconstruction training, and online adaptation. Each study retains its own experimental units, loss normalization, feasibility restrictions, and inferential scope. Incomplete coverage and adverse results are reported explicitly.

## Anonymous numerical supplement

The [saved-forecast package builder](extensions/tmlr_integration/package_tmlr_review.py) produces `KAHM_TMLR_Saved_Forecasts.zip`, an approximately 80.8 MB anonymous archive. It covers all 192 tasks, all ten procedures, eight numerical tables, and the six primary statistical contrasts. From the repository root, after restoring the mechanism evidence:

```bash
python extensions/tmlr_integration/package_tmlr_review.py --output-dir /tmp/kahm-tmlr-review
```

Use a new output directory. After unpacking the resulting ZIP, install its `requirements.txt` and run `python verify_saved_forecasts.py` from the extracted supplement directory. The [standalone verifier](extensions/tmlr_integration/verify_saved_forecasts.py) checks numerical records and recomputes losses and comparisons from saved forecast arrays. The ZIP is a downloadable review artifact; the repository stores its construction and verification programs.

These checks reproduce the saved-forecast analysis. They do not refit models, repeat feature-bound audits, remeasure computation time, or rerun earlier studies. The public study guides describe those separate reproduction procedures.

## Review format and reproducibility

The source uses the official TMLR style in anonymous review mode. The paper includes an explicit disclosure of AI assistance. Its public repository provides the scientific source and experimental provenance, including the local prospective checkpoint for the principal experiment. That checkpoint is not described as public preregistration. Independent-code checks within the study are distinguished from external replication.

The public repository is author-identifying. An anonymous submission must therefore use the review PDF and an anonymous supporting artifact, without directing reviewers to an identifying version of the work. TMLR's current [author guidelines](https://jmlr.org/tmlr/author-guide.html) and [editorial policies](https://jmlr.org/tmlr/editorial-policies.html) describe the submission requirements. The repository records the research materials; it does not assert submission, acceptance, or journal endorsement.
