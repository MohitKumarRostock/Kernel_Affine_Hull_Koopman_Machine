# A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations

**Mohit Kumar and Somayeh Kargaran**

## Why assess a representation class?

A fitted forecaster that misses an accuracy requirement leaves an important question unanswered: could another predictor use the same features successfully? This paper develops bounds on the best mean squared error within a specified predictor class, so an unsuccessful fit can be distinguished from a limitation of that class.

Each predictor class is specified by a feature map, a distance between feature vectors, an output range and a Lipschitz constant. The constant limits how much predictions may change as features move apart. The experiments use Kernel Affine Hull Machine (KAHM) features to study whether changing the representation can help meet a prediction-error requirement.

## What the paper contributes

The theoretical construction minimizes total squared error on groups of observations under the declared sensitivity constraints and divides by one less than the group size. For a fixed prediction problem with bounded scalar targets and outputs and independent and identically distributed (iid) observations, the expected scores form a nondecreasing hierarchy of lower bounds on the best class risk. On a compact metric feature domain, they converge to that risk. Larger groups do not automatically produce tighter finite-sample confidence bounds.

Verified optimization lower bounds and statistical calibration can exclude a class from meeting an error requirement. An upper bound for an admissible fitted predictor can establish acceptance. When neither criterion is met, the decision remains unresolved. These conclusions concern the declared class; a lower bound alone does not separate future uncertainty, information lost by the representation, and the sensitivity restriction.

The [article](manuscript/manuscript.pdf) contains the theory, complete proofs, empirical studies, and detailed appendices. [Editable manuscript sources](manuscript/) accompany it.

## Which evidence answers which question?

| Question | Experiment and records | Interpretation |
| --- | --- | --- |
| Can calibrated bounds identify a useful change in the feature map? | [Controlled calibration](experiments/controlled_calibration/) | Independent-episode population decisions under the stated sampling and numerical assumptions |
| Can a chosen representation class be ruled out on a fixed dataset? | [ECG evaluation dataset](experiments/ecg_cohort/) | A lower bound on the specified evaluation dataset, conditional on the recorded bounds on feature values; no future-patient guarantee |
| Does a failed fitted predictor imply an inadequate class? | [Class diagnosis](experiments/class_diagnosis/) | All 54 examined Bike classes admit predictions meeting their dataset requirements; both tested predictors fail in 39 classes |
| Does selection using risk bounds improve accuracy and cost? | [Deployment cost](experiments/deployment_cost/) and [public screening](experiments/public_screening/) | Evaluation of the specified accuracy, storage and latency criteria, including unsuccessful outcomes |
| Do prediction and feature alternatives improve the tested procedures? | [Predictor diagnostics](experiments/decoder_diagnostics/) | Selected forecast losses, fixed comparisons, and the complete assessed affine-ablation configurations |
| Are specified stored forecasts evaluated accurately? | [Affine forecasting](experiments/affine_forecasting/) | Numerical output checks conditional on stored features and a fixed rational predictor; forecast error and checking cost are separate measurements |

The Bike analysis constructs feasible predictions using the evaluation targets. This establishes what the class can represent on those observations, without establishing future prediction accuracy. Population-risk bounds, capacity calculations on a fixed dataset and numerical output checks answer different questions and require different assumptions.

For a guided route from a research question to its records and commands, start with the [experiment guide](docs/EXPERIMENT_GUIDE.md).

## Recompute the stored numerical checks

Run from the root of this risk-certificate branch. Use Python 3.12 and the recorded dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce.py --output reproduction
```

The driver validates the input manifest and runs the seven experiment groups in separate processes. It writes a summary, group results, and execution logs to the selected output directory, then checks the manifest again. A reproduction `PASS` means agreement with the recorded result, including a negative scientific outcome.

For one group:

```bash
python3 reproduce.py --only ecg_cohort --output reproduction_ecg
```

Choose a separate output directory outside `manuscript/`, `experiments/`, and `validation/`. Use `python3 reproduce.py --help` for the available group names and options.

These commands recompute checks from stored observations, features, forecasts, model parameters, and optimization records. The deployment group also regenerates forecasts from stored raw inputs and model parameters. Model training and runtime benchmarking require separate execution of the supplied source under each study's data and environment requirements. The checks above use the saved timing measurements.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the calculation performed by each group and [DATA.md](DATA.md) for attribution and input provenance. The [recorded verification summary](validation/summary.json) identifies the completed checks for its recorded source state.

## Build the manuscript

With a configured TeX installation containing pdfLaTeX, BibTeX, and the required packages:

```bash
python3 manuscript/build_manuscript.py
```

The command produces `manuscript/manuscript.pdf` and document-build diagnostics. It compiles the manuscript without running the experiment groups. The bundled JMLR style and bibliography retain their original notices.

## Scope and licenses

The numerical checks use the supplied feature values or recorded bounds on those values. They do not verify every step of feature generation. Results on a fixed dataset do not establish population guarantees. The affine forecast checker bounds numerical error relative to a fixed rational predictor evaluated on trusted stored features.

The experiment source is available at [commit 4c09d24a27837d7f165b253f3ab4d3cf28b71c0e](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/4c09d24a27837d7f165b253f3ab4d3cf28b71c0e). File identities are recorded in `MANIFEST.json`.

Original source and research-output license terms appear in [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE). Third-party datasets retain their own terms.
