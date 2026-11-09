# Reproducibility

Reproduction commands are organized by study and by the inputs they use. Statistical analysis of stored losses, numerical checks of stored forecasts and regeneration of observations and fitted models have different requirements.

## Study commands and evidence

| Task | Instructions | Scope |
| --- | --- | --- |
| Recompute the central feature-augmentation comparisons | [Saved-loss analysis](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment/README.md) | Reconstructs statistical comparisons from the supplied per-run loss table |
| Verify or regenerate the supplementary predictor-capacity experiment | [Predictor-capacity experiment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/capacity_followup/README.md) | Provides source, stored records, verification, analysis and full-rerun commands for that separate experiment |
| Analyze the supporting forecasting studies | [Supporting-study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) | Each study specifies its own observations, experimental protocol and available checks |
| Recompute the prediction-risk experiments | [Prediction-risk reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) | Replays the documented numerical and statistical checks from stored experiment records |

The central feature-augmentation experiment does not supply its original data generator, experiment runner, exact task roster, execution environment, fitted model states or forecast arrays. Its saved-loss checker verifies the analysis of the supplied losses; it cannot regenerate the observations or predictions that produced them. The storage size of numerical archives and the availability of source code, input definitions and environment records are separate constraints.

The supplementary predictor-capacity experiment and the supporting studies have their own data and protocols. Their records do not replace missing inputs from the central experiment. The prediction-risk guide likewise distinguishes a replay of stored results from refitting models or collecting new observations.

## Numerical library on this branch

The [library overview](README.md#numerical-library) describes the reusable reconstruction and forecasting modules. Install their direct dependencies with:

```bash
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
```

The [dependency file](requirements.txt) specifies NumPy, SciPy, scikit-learn, joblib, Numba and tqdm. These pins describe the numerical library dependencies. Use each study's environment instructions for its experiment commands.

Matching package versions alone does not specify an entire execution environment. Record the Python version, operating system, architecture and numerical libraries when running experiments. Where a study checks exact input or output hashes, its verification requires byte-identical arrays and stops if those hashes differ.

## Identifying the source used

Use the exact commit and study protocol associated with each analysis. The [source index](docs/BRANCH_ARCHIVE.json) records source identifiers, and [CITATION.cff](CITATION.cff) contains software citation metadata. The [study index](TMLR.md) links the two research questions and their evidence guides.
