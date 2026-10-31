# KAHM forecasting with directly trained reconstruction

This extension trains local KAHM kernel bandwidths and gate temperatures
against the final forecast and horizon. It evaluates the resulting residual
forecaster against matched affine, locality and RBF-local families, raw
ridge, raw-history-plus-RBF, gradient boosting, seasonal naive and a
validation-selected non-KAHM policy. The complete roster has 17 procedures.

The study is separate from the preceding
[proxy metric experiment](../target_geometry/README.md) and the preserved
[exploratory gate pilot](../forecast_gate_pilot/INTEGRATION.md). The scientific
code, analysis, exact inputs and protocol are frozen before full execution.

## What is trained

All geometric procedures retain the same A-fitted baseline, reference bank,
principal coordinates and correction-head family. A fits heads; B trains
geometry using actual forecast MSE; C selects head penalty and mixing by
horizon; D evaluates the frozen choice. A zero correction reproduces the
baseline. D labels do not enter the model-fitting API.

The reconstruction intervention changes both the KAHM reference kernel and
query kernel through one bounded bandwidth per block and horizon. Each
block also has a bounded gate temperature. Affine projection gain and local
distance anisotropy provide explicitly specified matched interventions for
the controls. All trainable modes have a 256-objective-evaluation ceiling per
penalty and horizon. Equal optimization opportunity is not equal wall time,
memory or functional capacity.

## Evidence roster and feasibility

The planned roster comprises 36 fresh repetitions of three known simulation
laws and ten A-selected series from four previously unused archive
collections. Preparation found that all three selected rideshare series have
no complete B windows. They remain planned feasibility failures, without
replacement or a revised split. Thus the full four-collection accuracy
aggregate is undefined. A seven-series, three-collection available-case
description has a narrower, explicitly stated scope.

The [protocol](PROTOCOL.md) fixes all estimands, methods, failure rules and
cost boundaries. The [design log](PRE_FREEZE_DESIGN_LOG.md) preserves the
A-only rideshare context amendment and subsequent availability finding.
[Data attribution and schema](data/README.md) identify all sources,
checksums, original indices, chronology and licenses.

## Execute and audit

Use Python 3.12 with NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0.
The exact host and environment are captured in the execution records.
The runner sets numerical-library thread counts to one before imports.

```bash
python -m unittest extensions.direct_reconstruction.test_model \
  extensions.direct_reconstruction.test_protocol -v
python extensions/direct_reconstruction/run.py smoke --output /tmp/kahm-direct-smoke
```

Preparation saves inputs and a scientific freeze without fitting a forecaster:

```bash
python extensions/direct_reconstruction/run.py prepare --output /tmp/kahm-direct-new
```

Publish that exact freeze before a new prospective execution, then pass its
full 40-character commit SHA to the `run` subcommand. The preserved completed
run must never be overwritten. A repeated fit is a computational reproduction,
not another unseen confirmation after its outcomes have been inspected.

The independent analysis reconstructs windows, losses, C selection, costs
and frozen hashes, and replays selected serialized models. It does not
independently refit every optimization candidate. Inspect `--help` for the
runner and analysis entry points. Every planned task and method is retained,
including unavailable tasks, optimization budget exits and numerical fallback.

## Resource and claim scope

Warm measured computation covers the fitted procedure's declared dependency
nodes; selected-state packing and serialization are recorded separately and
included in the charged total. Policy cost includes the union of all searched
candidate fits. Downloads, input preparation, imports and evidence-file writes
are outside this measure. Serialized state bytes are not peak resident memory.

This is a static forecasting experiment. It does not evaluate incremental
geometry updates, drift, delayed-label streaming, or deployment resource
ceilings. The manuscript reports whether the observed accuracy justifies a
distinctive predictive claim under this protocol; it does not assume one.
