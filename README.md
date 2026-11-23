# Risk certificates for representation-based prediction: experimental archive

**Mohit Kumar and Somayeh Kargaran**

## Prediction-risk bounds

This repository contains the numerical experiments, retained inputs, and verification programs for assessing **prescribed representation–decoder classes**. A class fixes a measurable representation $\Gamma:\mathcal X\to\mathcal U$ of a prediction-time history $X$, a measurable feature pseudometric $d$, an output interval $I$, and a Lipschitz bound $\kappa\geq0$. Its admissible decoders are

$
\mathcal G_\kappa(d)=
\bigl\{g:\mathcal U\to I\text{ measurable}:
|g(u)-g(v)|\leq\kappa d(u,v)\text{ for all }u,v\bigr\}.
$

For a specified evaluation law of $(X,Y)$ and squared loss, **optimal class risk** and **fitted-decoder risk** are distinct:

$
R_\kappa^*=\inf_{g\in\mathcal G_\kappa(d)}
\mathbb E[(Y-g(\Gamma(X)))^2],
\qquad
R(\widehat g)=\mathbb E[(Y-\widehat g(\Gamma(X)))^2].
$

A fitted decoder need not be admissible. Its individual-risk upper bound can establish class feasibility only after admissibility is verified. Some experiments use Kernel Affine Hull Machines (KAHMs) to construct representations; the method is not restricted to KAHM features.

For $b\geq2$ independent and identically distributed (iid) feature–target observations with $Y\in I$ almost surely, the **block score** minimizes total squared error under all pairwise Lipschitz constraints and divides by $b-1$:

$
T_b=\frac{1}{b-1}\min_{v\in\mathbb R^b}
\left\{\sum_{i=1}^b(Y_i-v_i)^2:
|v_i-v_j|\leq\kappa d(U_i,U_j)\ \text{for all }i<j\right\},
\qquad \mathcal W_b=\mathbb E[T_b].
$

The expected scores $\mathcal W_b$ are nondecreasing lower bounds on $R_\kappa^*$ and converge to it on compact metric feature spaces under the stated assumptions. Larger blocks do not necessarily give tighter finite-sample confidence bounds.

**Population risk** is expected squared prediction error under the evaluation law; **empirical risk** is mean squared error on a fixed dataset. Under prespecified calibration and confidence allocations, iid evaluation observations independent of model development support simultaneous one-sided confidence bounds on population risks. By contrast, the PTB-XL and Bike Sharing calculations give deterministic bounds on minimum empirical risk for their fixed observations; those conclusions require neither iid sampling nor targets within $I$, and they do not imply population-risk guarantees.

## Experiments

| Directory | Calculation and result |
| --- | --- |
| [controlled_calibration](experiments/controlled_calibration/) | Compares two confidence-bound procedures using independent synthetic episodes and nested feature sets. |
| [ecg_cohort](experiments/ecg_cohort/) | Excludes a specified two-coordinate Lipschitz predictor class from meeting an empirical MSE threshold on 8,192 PTB-XL patients. |
| [class_diagnosis](experiments/class_diagnosis/) | All 54 Bike Sharing predictor classes admit predictions meeting their empirical thresholds; both fitted predictors fail in 39 classes. |
| [deployment_cost](experiments/deployment_cost/) | Two KAHM-based procedures meet the accuracy allowance in all six cases but fail the combined storage and inference-time requirements. |
| [public_screening](experiments/public_screening/) | Risk bounds avoid no additional predictor fits in the Beijing and Bike Sharing experiments. |
| [decoder_diagnostics](experiments/decoder_diagnostics/) | Supplementary predictor comparisons and affine constraint ablations, including unsuccessful comparisons. |

The Bike Sharing feasible predictions use evaluation targets. They establish minimum empirical-risk upper bounds, not the future accuracy of independently trained predictors. The [experiment guide](docs/EXPERIMENT_GUIDE.md) includes a worked example and identifies each group's inputs and outputs.

## Reproduction

Run from the repository root containing `reproduce.py`, with Python 3.12 and the pinned dependencies:

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

Cite the exact commit used. `MANIFEST.json` identifies the source, documentation, and numerical inputs by SHA-256 hash. The [experiment index](docs/MANUSCRIPT_ALIGNMENT.md) distinguishes the five manuscript experiment groups from the supplementary predictor diagnostics. Protocol and provenance files identify the data, software, and numerical inputs.

Source and research-output licenses are specified in [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE). Third-party datasets retain their providers' terms.
