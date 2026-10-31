# KAHM for time-series forecasting: target geometry

This separately frozen research extension asks whether **residual-supervised
geometry improves a strong baseline**, and whether the improvement is specific
to Kernel Affine Hull Machines (KAHMs).

The implementation learns a constrained diagonal metric through a training-only
Gaussian kernel residual-regression objective. The same metric is then supplied
to KAHM, affine, locality and RBF-local features. This is supervised metric
transfer, not end-to-end optimization of KAHM, and a lower proxy objective does
not guarantee a better forecast.

The study compares in-sample and honestly held-out donor residuals on identical
correction rows. Raw ridge is the main baseline. The previous geometric utility,
scaling and natural-data studies remain intact, including their negative
findings and failures.

## Research specification

- [Execution protocol](PROTOCOL.md): hypotheses, precise fitting/evaluation
  rules, costs, failure behavior and multiplicity.
- [Configuration](config.json): all numerical settings and primary contrasts.
- [Theory](THEORY.md): squared-risk and residual-target transport identities,
  including the limits of cross-fitted residuals after a baseline changes.
- [Source provenance](sources/README.md): four new archive collections, exact
  selected series, attribution and data licenses.
- [Manuscript source](../tmlr_integration/manuscript/main.tex): the integrated,
  unsubmitted research draft.

The full roster has 36 synthetic tasks (three established laws, twelve fresh
independent repetitions) and 12 archive tasks (three series from each of four
new collections), with 16 forecasting procedures per task. The archive scores
are descriptive and source-balanced. Twelve series do not establish twelve
independent application domains. Bitcoin tasks concern activity indicators,
not price forecasts.

## Reproduce

From the repository root in Python 3.12:

```bash
python -m pip install -r extensions/target_geometry/requirements.txt
python -m unittest extensions.target_geometry.test_models \
  extensions.target_geometry.test_protocol -v
python -m extensions.target_geometry.data verify
python -m extensions.target_geometry.freeze verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m extensions.target_geometry.experiment --output /tmp/kahm-target-rerun
python -m extensions.target_geometry.analysis \
  --run-dir /tmp/kahm-target-rerun --output /tmp/kahm-target-audit
```

Existing outputs require explicit `--resume`; their source/configuration/roster
must match. Failures and unsuccessful work are retained. Prepared arrays are
derived from the pinned official records; `data.py prepare` is for an empty
prepared destination, never a way to select another favorable roster.

The small API primitives are `LearnedDiagonalMetric`,
`RobustLocalFeatureBank`, and `RidgeHead` in [models.py](models.py). The experiment
runner owns donor residual generation, selection, fallbacks and complete cost
accounting. Appending local features or allowing zero correction does not
constitute a forecasting guarantee.

## Evidence and status

The source/protocol snapshot is `freeze.json`; its public checkpoint is recorded
in `PUBLICATION.json`. These are prospective execution records, not an external
registration. Audit-derived findings and the revised manuscript are added after
the fixed execution, without changing the scientific freeze.
