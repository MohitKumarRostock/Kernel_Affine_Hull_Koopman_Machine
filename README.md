# Kernel Affine Hull Koopman Machine

Scientific software for reconstruction-based state representations, linear prediction in representation space, and the assessment of representation-dependent prediction limits.

Given a collection of observed state transitions, the Kernel Affine Hull Koopman Machine (KAHKM) constructs local kernel affine hull models and represents each state by normalized reconstruction affinities. A linear transition matrix then predicts the successor representation. The associated risk-certificate methods quantify lower bounds on the prediction error achievable from a fixed representation under specified predictor constraints.

## Mathematical formulation

Let $x\in\mathbb R^D$ denote a state and let $T_c(x)\in[0,1]$ be the reconstruction score of reference class $c=1,\ldots,C$. For $\omega,\tau>0$, define

$$
\phi_c(x)=\frac{(1-T_c(x)+\tau)^\omega}{\sum_{j=1}^C(1-T_j(x)+\tau)^\omega}.
$$

The vector $\phi(x)$ belongs to the probability simplex. The one-step forecasting problem is to approximate $\phi(x^+)$ from $\phi(x)$, where $x^+$ denotes the successor state. KAHKM uses the prediction $B^\top\phi(x)$ with a matrix $B\in\mathbb R^{C\times C}$ fitted to training transitions.

A close correspondence between the soft coordinates and a reference partition does not imply that the coordinates preserve the information required for prediction. For a fixed evaluation law and a spectral-norm limit $\kappa$, the prediction-error certificates address the population quantity

$$
R_\kappa=\inf_{\|B\|_2\leq\kappa}
\left(\mathbb E\|\phi(X^+)-B^\top\phi(X)\|_2^2\right)^{1/2}.
$$

A valid lower confidence bound exceeding a prescribed error tolerance excludes that tolerance for the entire constrained predictor class. A zero bound does not establish predictive sufficiency.

## Installation and first calculation

From a Git checkout of the default branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
python kernel_affine_hull_koopman_machines.py
```

The example constructs Duffing oscillator transitions, fits a reconstruction-based representation and linear predictor, and evaluates association-space errors on a later trajectory segment. It is a demonstration of forecasting, not an independent-pair confidence experiment.

[Mathematical specification and Python interface](docs/KAHKM.md) describes the reconstruction, normalization, fitting algorithm, array conventions, example code, and interpretation of diagnostics. The pinned dependencies on the default branch apply to the numerical library; experiment-specific environments are specified in the respective research records.

## Scientific studies

| Research question | Documentation and computational records |
| --- | --- |
| Prediction limits and exact Koopman closure of simplex-valued state abstractions | [Theory and experimental designs](docs/SOFT_STATE_CERTIFICATES.md) · [Reproduction and data access](docs/SOFT_STATE_REPRODUCIBILITY.md) |
| Forecasting with augmented reconstruction-based features | [Study description and records](docs/RESEARCH_STUDIES.md#forecasting-with-augmented-features) |
| Risk bounds for representation-based forecasting with bounded Lipschitz predictors | [Study description and records](docs/RESEARCH_STUDIES.md#risk-certificates-for-representation-based-forecasting) |

The [reproduction guide](REPRODUCIBILITY.md) identifies the available inputs, numerical procedures, and verification criteria for each study. The research code and data are available at the specified Git commit identifiers; the default branch contains the shared numerical library and scientific documentation.

## License and citation

The author-owned software is licensed under the [MIT License](LICENSE). Author-generated data and non-code research outputs are licensed under [CC BY 4.0](DATA_LICENSE), unless specified otherwise. Third-party material retains its applicable license.

Software citation metadata is provided in [CITATION.cff](CITATION.cff). For a numerical result, cite the applicable study, its computational protocol, and the exact source or evidence commit identified in the reproduction guide.
