# A Hierarchy of Risk Certificates for Representation-Based Forecasting

**Mohit Kumar and Somayeh Kargaran**

## Prediction-risk bounds

This repository contains the experiments for the paper. We evaluate the minimum mean squared error (MSE) over bounded Lipschitz predictors using a fixed feature representation. Each predictor class specifies a feature map, a distance metric, an output interval, and a Lipschitz constant. A fitted predictor is one member of that class; its error need not equal the minimum error over the class.

For blocks of `b >= 2` independent and identically distributed (iid) observations, the method minimizes total squared error subject to pairwise Lipschitz and output constraints and divides the minimum by `b - 1`. The expected values are nondecreasing lower bounds on minimum population risk and converge to that risk on compact metric feature spaces. This monotonicity does not imply that larger blocks give tighter finite-sample confidence bounds.

Population risk is expected squared prediction error; empirical risk is average squared error on a specified dataset. With iid evaluation observations independent of model development, simultaneous confidence bounds can establish that every predictor in a class exceeds an MSE threshold or that an admissible fitted predictor meets it. Deterministic optimization instead bounds minimum empirical risk on fixed observations.

## Experiments

| Directory | Calculation and result |
| --- | --- |
| [controlled_calibration](experiments/controlled_calibration/) | Compares two confidence-bound procedures using independent synthetic episodes and nested feature sets. |
| [ecg_cohort](experiments/ecg_cohort/) | Excludes a specified two-coordinate Lipschitz predictor class from meeting an empirical MSE threshold on 8,192 PTB-XL patients. |
| [class_diagnosis](experiments/class_diagnosis/) | All 54 Bike Sharing predictor classes admit predictions meeting their empirical thresholds; both fitted predictors fail in 39 classes. |
| [deployment_cost](experiments/deployment_cost/) | Two KAHM-based procedures meet the accuracy allowance in all six cases but fail the combined storage and inference-time requirements. |
| [public_screening](experiments/public_screening/) | Risk bounds avoid no additional predictor fits in the Beijing and Bike Sharing experiments. |
| [decoder_diagnostics](experiments/decoder_diagnostics/) | Reports predictor comparisons and affine constraint ablations, including unsuccessful comparisons. |

The Bike Sharing feasible predictions use evaluation targets. They establish minimum empirical-risk upper bounds, not the future accuracy of independently trained predictors. The [experiment guide](docs/EXPERIMENT_GUIDE.md) includes a worked example and identifies each group's inputs and outputs.

## Reproduction

Run from this branch's repository root with Python 3.12 and the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --output reproduction
```

To check one group:

```bash
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
```

The driver verifies `MANIFEST.json`, runs the six groups in separate processes, writes results and logs, and rechecks file integrity. `PASS` means that the numerical checks reproduce the stored results, including unsuccessful method outcomes. These commands check stored records; they do not retrain all models or collect new timing measurements. The deployment group also regenerates predictions from stored raw inputs and model parameters.

Use an output directory outside `docs/`, `experiments/`, `scripts/`, and `validation/`, and not the repository root. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the environment, arithmetic assumptions, and verification scope; [DATA.md](DATA.md) for attribution; and [validation/summary.json](validation/summary.json) for the recorded run.

## Source version and licenses

Cite the exact commit used. `MANIFEST.json` identifies the source, documentation, and numerical inputs by SHA-256 hash. The [manuscript experiment index](docs/MANUSCRIPT_ALIGNMENT.md) maps the six groups to the revised paper. Historical protocols and source records retain their original identifiers.

Source and research-output licenses are specified in [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE). Third-party datasets retain their providers' terms.
