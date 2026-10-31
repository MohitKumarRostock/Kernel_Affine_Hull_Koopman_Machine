# Reproducing the KAHM forecasting studies

This repository contains several experiments with distinct source checkpoints and computational environments. The current manuscript draws on the forecasting extensions under `extensions/`; the original Kernel Affine Hull Koopman Machine (KAHKM) experiments have a separate result map under `reproduction/`. Use the environment and entry point for the study being reproduced.

## Principal reconstruction-departure experiment

The [study README](extensions/reconstruction_mechanism/README.md) is the entry point for the ten-procedure, 192-task-suite experiment. Its evidence consists of exact prepared inputs, fitted model states, selected predictions, selection records, statistical summaries, and timing measurements.

Use Python 3.12 and install:

```bash
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
```

From the repository root, restore the lossless evidence and verify the scientific checkpoint:

```bash
python -m extensions.reconstruction_mechanism.evidence unpack
python -m extensions.reconstruction_mechanism.run verify
```

Recompute the analysis in a separate output directory and check it with the independent evaluation implementation:

```bash
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-analysis
python -m extensions.reconstruction_mechanism.audit \
  --analysis-json /tmp/kahm-analysis/decision.json \
  --output /tmp/kahm-audit.json
```

The study guide also describes selected-state replay and refitting. A new execution must use a separate results directory or disposable checkout. The original evidence is retained so that repeated computation can be distinguished from the reported experiment.

### Statistical reproduction and numerical inference portability

The anonymous saved-forecast supplement reconstructs all 192 task suites and the complete primary statistical analysis directly from numeric forecast arrays and A-target variance scales. This operation is separate from evaluating serialized fitted states. Its successful verification does not establish that every processor reproduces each selected-state prediction within the declared pointwise tolerance.

Selected-state replay compares predictions using relative tolerance `1e-10` and absolute tolerance `1e-12`. The reference Linux environment uses Python 3.12.14 and glibc 2.39, with the pinned numerical packages. Processor-dependent vectorized kernels can still affect the comparison. In a targeted test, disabling NumPy AVX512 dispatch changes 49 of 4,992 raw-RBF predictions on `vanderpol__noise0__seed946100` beyond the original tolerance, with maximum absolute difference `1.0663936400590046e-10`; default dispatch reproduces those raw-RBF predictions exactly. The [diagnostic record](extensions/reconstruction_mechanism/publication_validation/NUMERICAL_REPLAY_PORTABILITY.json) specifies the reproducible invocation and identifies the original source arrays and fitted state. The strict replay criterion concerns numerical agreement between model evaluations and archived predictions. The reported statistical analysis uses the archived predictions directly.

## Evidence levels

| Operation | What it checks |
| --- | --- |
| Hash verification | The source, inputs, and stored outputs have the recorded bytes. |
| Analysis reconstruction | The reported losses, aggregates, and statistical contrasts follow from the saved forecasts. |
| Selected-model replay | The saved fitted state reproduces its recorded forecasts in a specified numerical environment. |
| Independent-code evaluation | A separate implementation evaluates saved states or statistics consistently with the original implementation. |
| Refitting | Re-execution of the specified fitting and selection procedure yields reproducible results, subject to numerical and timing variation. |
| External replication | A separate research team reproduces or independently tests the scientific findings. This is not claimed by the repository's internal checks. |

A successful checksum or replay does not, by itself, validate a scientific interpretation. The manuscript supplies the assumptions, comparison design, and inferential scope needed to interpret the verified outputs.

## Other forecasting studies

| Study | Instructions |
| --- | --- |
| Controlled representations and decoder access | [Reproduction guide](extensions/predictive_geometry/REPRODUCING.md) |
| Natural-data evaluation | [Study README](extensions/natural_forecasting/README.md) |
| Input-scaling sensitivity | [Study README](extensions/scaling_sensitivity/README.md) |
| Additive geometric corrections | [Study README](extensions/forecasting_utility/README.md) |
| Supervised metric and donor residuals | [Study README](extensions/target_geometry/README.md) |
| Direct forecast training | [Study README](extensions/direct_reconstruction/README.md) |
| Online adaptation | [Study README](extensions/adaptive_development/REPRODUCING.md) |

The corresponding protocols specify random seeds, chronology, selection rules, and failure handling. Raw forecasts and fitted states are retained where described. Some older audits verify a complete historical repository checkpoint, including its documentation. Those audits require that historical checkout or its corresponding archive; they should not be interpreted as checks of the present manuscript edition. Current scientific-source verification uses the study-specific frozen dependencies.

Numerical package versions alone need not guarantee bitwise agreement. The direct reconstruction study documents CPU-dispatch sensitivity in Duffing generation and one raw-RBF selected-state replay. Such differences are reported against the original tolerances rather than treated as successful exact reproduction. Timing measurements depend on hardware, numerical libraries, and scheduling.

## Original KAHKM experiments

The original architecture's result-to-code mapping is [reproduction/reported_results.tsv](reproduction/reported_results.tsv). The map identifies generators, runners, settings, seeds, retained per-run outputs, and derived artifacts. Entries marked `PASS` identify retained reported results with a reproducibility chain; entries marked `REMOVE` identify legacy outputs excluded from that reported set.

Its reference platform is native Apple Silicon macOS on an Apple M1, using Python 3.14. The direct dependencies are in `requirements.txt`; the resolved environment is in `requirements-lock-arm64.txt`. Platform metadata are recorded under `provenance/reference_arm64_environment/`.

On that platform:

```bash
bash scripts/bootstrap_arm64.sh
.venv-arm64/bin/python scripts/verify_environment.py
```

The full evidence summary is [reproduction/REPRODUCIBILITY_EVIDENCE.md](reproduction/REPRODUCIBILITY_EVIDENCE.md), with a machine-readable counterpart in `reproduction/reproducibility_evidence.json`. The historical Duffing reference search has a documented gap in run accounting. A separate prospective 27-job campaign records all scheduled attempts, exit statuses, logs, and output checksums. These records support complete attempt accounting for that campaign, not for every historical execution in the repository.

## Manuscript reproduction

The manuscript source includes all generated displays. Compilation requires no model fitting:

```bash
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The [manuscript guide](extensions/tmlr_integration/README.md) identifies the source, display-generation scripts, and archived evidence associated with the paper.
