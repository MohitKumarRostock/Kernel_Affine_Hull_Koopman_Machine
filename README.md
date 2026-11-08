# A Hierarchy of Risk Certificates for Forecasting Representations

**Mohit Kumar and Somayeh Kargaran**

This repository accompanies the manuscript in [manuscript/manuscript.pdf](manuscript/manuscript.pdf). It contains the article's editable sources and the numerical evidence for its empirical claims. Experiments are organized by the scientific question they address.

The paper distinguishes population-risk certificates, deterministic finite-cohort capacity certificates, and numerical prediction-fidelity checks. Their assumptions and conclusions differ. The implementation preserves these distinctions and the unsuccessful predictive, screening, and deployment outcomes.

## Reproduce the numerical evidence

Use Python 3.12 and install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output reproduction
```

The command verifies file integrity and runs all seven experiment groups. It writes a JSON summary, a result for each group, and execution logs to the selected output directory. A successful verification means that the archived numerical claims reproduce; some scientific hypotheses have negative outcomes.

| Experiment | Evidence reproduced | Manuscript location |
| --- | --- | --- |
| [Controlled calibration](experiments/controlled_calibration/) | Block duals, fitted forecasts, calibration endpoints, and necessity/adequacy decisions | Controlled necessity, adequacy, and calibration limits |
| [ECG cohort](experiments/ecg_cohort/) | Exact cohort witnesses, reference threshold, and sensitivity requirement | Natural-data exclusion of a specified class |
| [Class diagnosis](experiments/class_diagnosis/) | Full-graph distances, primal feasibility, weak-dual bounds, and fitted-head excess risk | Adequate classes with unsuccessful fitted heads |
| [Deployment cost](experiments/deployment_cost/) | Raw-input forecasts, exact losses, complete numeric inventories, and recorded timing medians | Predictive and operational limits |
| [Public screening](experiments/public_screening/) | Screening certificates and decisions in the Beijing and Bike studies | Predictive and operational limits |
| [Decoder diagnostics](experiments/decoder_diagnostics/) | Selected forecast losses, primary comparisons, and all assessed affine-ablation configurations | Preserved unsuccessful comparisons |
| [Affine forecasting](experiments/affine_forecasting/) | Exact forecast comparisons, output-certificate challenges, and recorded cost summaries | Conditional numerical output fidelity |

For an individual group:

```bash
python reproduce.py --only ecg_cohort --output reproduction_ecg
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the verification scope and [DATA.md](DATA.md) for data attribution. Original scientific implementation modules and protocols are included with each experiment. Legacy record identifiers in those frozen files identify their sources; the experiment directory names provide the navigation for this article.

The release includes the [numerical verification summary](validation/summary.json) and [manuscript build diagnostics](validation/manuscript_build.json).

## Build the manuscript

With a configured TeX Live installation containing pdfLaTeX and BibTeX:

```bash
python manuscript/build_manuscript.py
```

The build produces `manuscript/manuscript.pdf`. The article uses the bundled JMLR style and includes its proofs, empirical protocols, numerical-fidelity appendix, and bibliography.

## Scope

The numerical replay uses fixed observations, stored feature coordinates, forecasts, model payloads, and optimization witnesses, as specified separately for each study. It does not retrain the entire research campaign, remeasure historical latency, or turn a finite-cohort observation into a population guarantee. Native-feature error enclosures are inherited only where explicitly stated. The forecasting output checker is conditional on trusted stored features and a fixed rational head.

Original source, data, and manuscript license terms are preserved in [LICENSE](LICENSE), [DATA_LICENSE](DATA_LICENSE), and the bundled JMLR style notice. Third-party datasets retain their own terms.
