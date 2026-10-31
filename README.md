# KAHM for time-series forecasting

Research software for **Kernel Affine Hull Machines (KAHMs) as local geometric
forecasting representations**, alongside the preserved Kernel Affine Hull
Koopman Machine (KAHKM) experiments that gave this repository its name.

The current research asks which target-relevant structure a geometric feature
bank retains, how a finite decoder uses it, and whether its prediction gain
justifies the complete fitting and selection cost. The repository retains
negative comparisons and failed historical fits as part of that evidence.

## Current manuscript and study

**Kernel Affine Hull Machines for Time-Series Forecasting: Forecast-Trained
Local Geometry and Matched Evaluation** — an unsubmitted research draft.

- [Current manuscript PDF](extensions/tmlr_integration/review/KAHM_Forecast_Trained_Manuscript.pdf)
- [Editable LaTeX and build guide](extensions/tmlr_integration/README.md)
- [Direct reconstruction study](extensions/direct_reconstruction/README.md)
- [Protocol and scientific freeze](extensions/direct_reconstruction/PROTOCOL.md)
- [Time-series research map](TIME_SERIES_RESEARCH.md)
- [Preserved exploratory gate pilot](extensions/forecast_gate_pilot/INTEGRATION.md)

The new procedure learns horizon-specific KAHM kernel bandwidths and gate
temperatures against actual forecast error. Affine, locality and RBF-local
controls receive matched supervision, parameter counts within each mode, and
optimization ceilings. Raw ridge, raw-history-plus-RBF, histogram gradient
boosting, seasonal naive and a validation-selected non-KAHM policy complete
the 17-method comparison. The method retains the same fitted baseline and
separates head fitting, geometry training, model selection and final evaluation.

The [public checkpoint](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/39174d06dffc7e28fed73dbbf866e539560dd021)
precedes the reserved execution. The planned roster contains 36 fresh
simulation repetitions and ten series from four unused archive collections.
Three selected rideshare series have no complete geometry-training windows;
they remain explicit feasibility failures. The planned four-collection
accuracy aggregate is undefined. Available-case archive results have the
narrower scope of seven series from three collections.

The completed run preserves **731 prediction records across 43 feasible
tasks**, with no additional task failures or numerical fallbacks. Its
18 adjusted synthetic comparisons give **2 improvements, 6 disadvantages
and 10 unresolved differences** for reconstruction-trained KAHM. The clear
within-KAHM gain is **41.43% versus gate-only KAHM on Van der Pol**, while
matched affine and practical baseline comparisons remain stronger in key
regimes. The available natural subset improves ridge by 3.55%, mostly through
one collection, while raw RBF and boosting are both more accurate in aggregate
and faster in the measured implementation. See the
[complete findings](extensions/direct_reconstruction/FINDINGS.md).

The preceding [proxy-metric findings](extensions/target_geometry/FINDINGS.md)
and all historical negative comparisons remain preserved. Establishing an
explicit KAHM forecasting procedure is distinct from demonstrating a
predictive advantage over equally trained alternatives.

## Audit saved evidence without forecast refits

Use Python 3.12. The repository preserves exact prepared inputs, selected
predictions and model states, optimizer traces, validation candidates,
source hashes, unavailable-task records and measured computation nodes.
Task archives are lossless containers of those original files.

```bash
python -m pip install -r extensions/direct_reconstruction/requirements.txt
python extensions/direct_reconstruction/evidence.py unpack \
  --run-dir extensions/direct_reconstruction/results
python extensions/direct_reconstruction/analyze.py \
  extensions/direct_reconstruction/results --verify-live-source
python -m extensions.tmlr_integration.package_direct audit
```

The audit reconstructs windows and losses, checks temporal availability,
selection and costs, and replays every saved selected model. It does not
independently refit every optimization candidate. Read the study guide for
new fits and the distinction between independent repetitions, retrospective
source evaluation and computational replay. No streaming adaptation or
minimum deployment resource claim is made.

## Preserved KAHKM and historical study evidence

The following reference environment and result map concern the original KAHKM
experiments. Later forecasting extensions each declare their own environment,
protocol, freeze and evaluation scope.

## Reproducibility

The authoritative mapping from each reported result to its generator, experiment runner, configuration, seeds, raw or per-run output, and aggregate artifact is:

`reproduction/reported_results.tsv`

Entries marked `PASS` correspond to retained reported results with an identified reproducibility chain. Entries marked `REMOVE` identify legacy outputs that are not part of the canonical reported result set.

### Reference environment

The reference environment is native Apple Silicon (`arm64`) macOS.

Key files:

- `requirements.txt` — direct Python dependencies.
- `requirements-lock-arm64.txt` — frozen reference dependency environment.
- `scripts/bootstrap_arm64.sh` — native Apple Silicon environment bootstrap.
- `scripts/verify_environment.py` — architecture, dependency, and import verification.
- `provenance/reference_arm64_environment/` — captured hardware, operating-system, Python, NumPy, and package metadata.

Create the reference environment with:

```bash
bash scripts/bootstrap_arm64.sh
```

Verify it with:

```bash
.venv-arm64/bin/python scripts/verify_environment.py
```

### Canonical derived-result generators

Composite or newly canonicalized reported artifacts can be regenerated with:

```bash
.venv-arm64/bin/python scripts/generate_selection_table.py
.venv-arm64/bin/python scripts/generate_behavioral_summary.py
.venv-arm64/bin/python scripts/generate_search_protocol.py
.venv-arm64/bin/python scripts/generate_spectral_figures.py
```

Other reported tables and figures use the corresponding `step_*` generators and experiment runners identified in `reproduction/reported_results.tsv`.

## Reproducibility evidence

For a reviewer-facing summary of the current evidence state, see:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence summary and claim boundaries.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical audit and prospective gap-closing campaign are intentionally
kept separate. The historical run-accounting audit preserves the original
Duffing reference-search gap, while the prospective campaign records all 27
scheduled attempts and their exit statuses from a frozen clean commit.

The repository does not claim that every failed attempt from every historical
campaign was preserved when the available historical evidence does not support
that stronger statement.

## Repository organization

- `experiment_*.py` — primary experiment implementations.
- `run_*.py` — experiment and hyperparameter-search runners.
- `step_*.py` — table and figure generation workflows.
- `scripts/` — environment and canonical reproducibility utilities.
- `reproduction/` — result-provenance matrix and canonical generated artifacts.
- `provenance/` — environment and dependency provenance.
- `figures/` — generated figures used by the associated research work.
- `kahkm_*` directories and archives — experiment outputs and derived artifacts.

## Licensing

Original source code in this repository is licensed under the MIT License. See `LICENSE`.

Unless otherwise indicated, original generated research data and non-code research outputs are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See `DATA_LICENSE`.

Third-party dependencies and any separately identified third-party material remain under their respective licenses.

## Citation

Citation metadata is provided in `CITATION.cff`.

A persistent archival identifier such as a DOI should be cited once an immutable archival release is available. Until then, cite the repository and associated research work using the metadata in `CITATION.cff`.

## Repository

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
