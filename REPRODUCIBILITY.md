# Reproduction and verification

This repository provides a shared Python numerical library and experiment-specific computational records. Select the computational procedure corresponding to the research question; the requirements on the default branch apply only to the shared library.

| Scientific task | Procedure |
| --- | --- |
| Fit a KAHM representation and a linear predictor on supplied transitions | [Mathematical specification and executable example](docs/KAHKM.md) |
| Evaluate label-based, centered, prototype or empirical-risk lower confidence bounds | [Definitions, assumptions and certificate example](docs/SOFT_STATE_CERTIFICATES.md) |
| Recompute all 117,000 controlled count datasets, compare four certificate methods, examine the Van der Pol records, and regenerate forecasting summaries | [Data, source commits, environments, commands and expected values](docs/SOFT_STATE_REPRODUCIBILITY.md) |
| Recompute feature-augmentation or Lipschitz-risk experiments | [Experiment-specific documentation and data](docs/RESEARCH_STUDIES.md) |

## Shared library

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import kernel_affine_hull_koopman_machines"
python kernel_affine_hull_koopman_machines.py
```

The final command fits and evaluates an illustrative Duffing predictor. It does not reproduce the archived certificate or benchmark studies.

## Reproducibility criteria

The experiment-specific guides identify the original inputs or sufficient statistics, fixed configuration, random sampling law where applicable, dependency specification, output schema, and expected numerical or checksum criteria. The supported operations include:

- **Integrity verification:** comparison of file and archive-member SHA-256 digests.
- **Statistical recomputation:** calculation of metrics, certificates and aggregate summaries from distributed observations or sufficient statistics.
- **Numerical reconstruction:** evaluation of a stored representation or predictor on supplied inputs.
- **Experiment regeneration:** generation of new or deterministically reconstructed observations and refitting according to a specified protocol.

The availability of one operation does not establish availability of the others. In particular, the public Van der Pol confirmatory distribution provides sufficient statistics and verification records, not all underlying physical-state and coordinate arrays. The central augmented-feature study provides per-run losses for statistical analysis, not the original complete training and forecasting workflow.

For each calculation use the applicable immutable source identifier. Record the Python version, platform, architecture and numerical-library versions; verify inputs before performing calculations. Do not interpret a successful import or a matching checksum as a statistical validation of the sampling assumptions.

See [research studies](docs/RESEARCH_STUDIES.md) for the additional experiment-specific sources, and [CITATION.cff](CITATION.cff) for software citation metadata.
