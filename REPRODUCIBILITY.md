# Reproduction commands and requirements

Use the instructions for the relevant study. Statistical analysis of saved losses, numerical checks of stored predictions, and regeneration of fitted models are separate operations.

## Study instructions

| Task | Instructions | Calculation |
| --- | --- | --- |
| Central feature-augmentation comparisons | [Saved-loss analysis](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment/README.md) | Recompute statistical comparisons from per-run losses |
| Supplementary predictor-capacity experiment | [Experiment guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/capacity_followup/README.md) | Verify stored records or rerun this separate experiment |
| Supporting forecasting studies | [Study index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/archive/pre-manuscript-cleanup-2026-11-02/docs/RESEARCH_INDEX.md) | Apply each study's protocol and checks to its data |
| Prediction-risk experiments | [Reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/research/risk-certificates-consolidated-2026-11-08/REPRODUCIBILITY.md) | Recompute six groups of numerical and statistical results from stored observations, features, predictions, and optimization records |

The central feature-augmentation experiment does not supply its original generator, runner, exact task list, execution environment, fitted model states, or forecast arrays. Its checker analyzes supplied losses; it cannot regenerate their underlying observations or predictions. The supplementary capacity experiment and supporting studies have separate inputs and protocols.

The prediction-risk driver verifies source and input hashes before and after execution. It checks population-risk confidence bounds, fixed-dataset empirical-risk bounds, and predictor comparisons. Its deployment group also regenerates predictions from stored inputs and fixed models. These checks do not rerun all model training or collect new timing measurements.

## Library dependencies

The [library overview](README.md#numerical-library) describes the modules on `main`. Install their dependencies with:

```bash
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
```

The [dependency file](requirements.txt) pins NumPy, SciPy, scikit-learn, joblib, Numba, and tqdm. Experiment branches have their own environment requirements. Record the Python version, operating system, architecture, and numerical libraries used. Checks that compare exact input or output hashes require byte-identical arrays and stop on a mismatch.

## Source identification

Cite the exact commit and protocol used for each analysis. The [source index](docs/BRANCH_ARCHIVE.json) records identifiers; [CITATION.cff](CITATION.cff) contains software citation metadata. [TMLR.md](TMLR.md) links the studies and their records.
