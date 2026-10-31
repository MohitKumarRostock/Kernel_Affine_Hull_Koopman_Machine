# KAHM for time-series forecasting

Research software for **Kernel Affine Hull Machines (KAHMs) as local geometric
forecasting representations**, alongside the preserved Kernel Affine Hull
Koopman Machine (KAHKM) experiments that gave this repository its name.

The current research asks which target-relevant structure a geometric feature
bank retains, how a finite decoder uses it, and whether its prediction gain
justifies the complete fitting and selection cost. The repository retains
negative comparisons and failed historical fits as part of that evidence.

## Current manuscript and study

**Kernel Affine Hull Machines for Time-Series Forecasting: Predictive Geometry,
Residual Learning, and Cost** — an unsubmitted research draft.

- [Revised manuscript PDF](extensions/tmlr_integration/review/KAHM_Prediction_Revised.pdf)
- [Editable LaTeX and build guide](extensions/tmlr_integration/README.md)
- [New target-geometry experiment](extensions/target_geometry/README.md)
- [Audited findings](extensions/target_geometry/FINDINGS.md)
- [Time-series research map](TIME_SERIES_RESEARCH.md)
- [Focused next research questions](extensions/target_geometry/RESEARCH_ROADMAP.md)

The latest experiment was publicly frozen before execution and completed all
48 tasks and 768 forecasts. KAHM residual correction improves ridge on the
three synthetic laws and modestly improves the new archive aggregate.
Learning the shared metric and using donor residuals do not establish an
incremental synthetic benefit; simpler local controls remain stronger in
several comparisons. The archive learned-metric KAHM result is **2.18% lower
scaled MSE than ridge**, with **0.162 seconds versus 0.00130 seconds** of charged
pipeline time per task on the recorded host. Fixed-metric in-sample KAHM gains
**2.76%** at lower cost, and simpler corrections have lower aggregate error.
These results support a conditional local-representation research program,
without establishing broad or KAHM-specific forecasting superiority.

## Audit the new evidence without forecast refits

Use Python 3.12. Prepared data, all final predictions, exact source hashes,
recorded validation trials, and measured computation nodes are included.

```bash
python -m pip install -r extensions/target_geometry/requirements.txt
python -m extensions.target_geometry.freeze verify
python -m extensions.target_geometry.analysis \
  --run-dir extensions/target_geometry/results/v1 --output /tmp/kahm-target-audit
```

The study guide documents model refits into a new directory, the 28 engineering
checks, and the distinction between recorded-grid verification and independent
refitting. Source-balanced archive results cover four selected collections;
repeated windows do not create additional independent domains.

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
