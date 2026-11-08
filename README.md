# A Hierarchy of Risk Certificates for Forecasting Representations

**Mohit Kumar and Somayeh Kargaran**

## Why assess a representation class?

A fitted forecaster that misses an accuracy requirement leaves an important question unanswered: could another predictor use the same features successfully? This paper develops bounds on the best mean squared error within a specified predictor class, so an unsuccessful fit can be distinguished from a limitation of that class.

The assessed object includes a feature map, a distance, and a Lipschitz sensitivity budget—a limit on how much predictions may change as features move apart. Kernel Affine Hull Machine (KAHM) representations provide the geometric case study. The framework helps assess whether to retain, refine, or reconsider a geometry-induced representation for a common prediction target.

## What the paper contributes

The theoretical construction minimizes total squared error on groups of observations under the declared sensitivity constraints and divides by one less than the group size. For a fixed prediction problem with bounded scalar targets and outputs and independent and identically distributed (iid) observations, the expected scores form a nondecreasing hierarchy of lower bounds on the best class risk. On a compact metric feature domain, they converge to that risk. Larger groups do not automatically produce tighter finite-sample confidence bounds.

Verified optimization lower bounds and statistical calibration can exclude a class from meeting an error requirement. An upper bound for an admissible fitted predictor can establish acceptance. When neither criterion is met, the decision remains unresolved. These conclusions concern the declared class; a lower bound alone does not separate future uncertainty, information lost by the representation, and the sensitivity restriction.

The [article](manuscript/manuscript.pdf) contains the theory, complete proofs, empirical studies, and detailed appendices. [Editable manuscript sources](manuscript/) accompany it.

## Which evidence answers which question?

| Question | Experiment and records | Interpretation |
| --- | --- | --- |
| Can calibrated bounds diagnose a useful geometric refinement? | [Controlled calibration](experiments/controlled_calibration/) | Independent-episode population decisions under the stated sampling and numerical assumptions |
| Can a chosen representation class be ruled out on a fixed dataset? | [ECG cohort](experiments/ecg_cohort/) | A fixed-cohort lower bound, conditional on the declared feature enclosures; no future-patient guarantee |
| Does a failed fitted predictor imply an inadequate class? | [Class diagnosis](experiments/class_diagnosis/) | All 54 examined Bike classes admit predictions meeting their dataset requirements; both tested predictors fail in 39 classes |
| Does certificate-based selection improve accuracy and cost? | [Deployment cost](experiments/deployment_cost/) and [public screening](experiments/public_screening/) | Direct evaluation of the stated operational criteria, including unsuccessful outcomes |
| Do prediction and feature alternatives improve the tested procedures? | [Decoder diagnostics](experiments/decoder_diagnostics/) | Selected forecast losses, fixed comparisons, and the complete assessed affine-ablation configurations |
| Are specified stored forecasts evaluated accurately? | [Affine forecasting](experiments/affine_forecasting/) | Numerical output checks conditional on stored features and a fixed rational predictor; forecast error and checking cost are separate measurements |

The Bike feasible predictions use assessment targets to establish capacity on those observations. They are not newly trained predictors with a guarantee for future data. Population-risk certificates, fixed-dataset capacity calculations, and numerical output checks retain their different assumptions and conclusions.

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

These commands recompute checks from stored observations, features, forecasts, model parameters, and optimization records. The deployment group also regenerates forecasts from stored raw inputs and model parameters. They do not retrain the complete research campaign or remeasure historical timings. Original source modules and protocols are included for inspection and further reproduction under each study's data and environment requirements.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the calculation performed by each group and [DATA.md](DATA.md) for attribution and input provenance. The [recorded verification summary](validation/summary.json) identifies the completed checks for its recorded source state.

## Build the manuscript

With a configured TeX installation containing pdfLaTeX, BibTeX, and the required packages:

```bash
python3 manuscript/build_manuscript.py
```

The command produces `manuscript/manuscript.pdf` and document-build diagnostics. It compiles the manuscript without running the experiment groups. The bundled JMLR style and bibliography retain their original notices.

## Scope and licenses

Fixed-record replay does not recertify the complete native feature pipeline or turn a finite-dataset result into a population guarantee. Native-feature enclosures are inherited only where stated; the numerical forecast checker is conditional on trusted stored features and a fixed rational predictor.

The original experiment snapshot remains accessible at [commit 4c09d24a27837d7f165b253f3ab4d3cf28b71c0e](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/4c09d24a27837d7f165b253f3ab4d3cf28b71c0e). Current file identities are recorded in `MANIFEST.json`.

Original source and research-output license terms appear in [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE). Third-party datasets retain their own terms.
