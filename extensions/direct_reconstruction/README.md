# KAHM forecasting with directly trained reconstruction

This extension trains local KAHM kernel bandwidths and gate temperatures
against the final forecast and horizon. It evaluates the resulting residual
forecaster against matched affine, locality and RBF-local families, raw
ridge, raw-history-plus-RBF, gradient boosting, seasonal naive and a
validation-selected non-KAHM policy. The complete roster has 17 procedures.

The study is separate from the preceding
[proxy metric experiment](../target_geometry/README.md) and the preserved
[exploratory gate pilot](../forecast_gate_pilot/SCIENTIFIC_GUIDE.md). The scientific
code, analysis, exact inputs and protocol are frozen before full execution.

## Completed findings

**43 feasible tasks completed, with all 731 selected task-method forecasts
and model states preserved.** The three anticipated rideshare tasks account
for the other 51 planned outputs. No additional task failure, invalid
objective evaluation or numerical prediction fallback occurred.

The 18 adjusted synthetic comparisons yield **2 positive, 6 negative and
10 unresolved intervals** for reconstruction-trained KAHM. It beats raw RBF
on nonlinear AR by 2.87% and its gate-only KAHM arm on Van der Pol by 41.43%.
However, trained affine and the selected non-KAHM policy win on Van der Pol;
raw RBF, trained affine/locality and the policy win on Duffing. No adjusted
comparison establishes a KAHM advantage over an equally shape-trained
non-KAHM local control.

On the seven-series, three-collection available subset, KAHM's scaled MSE is
0.371935 versus ridge's 0.385625 (3.55% lower). Raw RBF (0.363007), gradient
boosting (0.352201), and trained RBF-local (0.363835) are more accurate in
aggregate. The ridge gain is almost entirely Dominick: excluding it leaves
0.00229% improvement. The complete four-collection aggregate is undefined.
Mean charged natural pipeline time is 7.929 seconds for KAHM, 0.0130 for raw
RBF and 0.6550 for boosting; these are static fitting/selection/query costs.

Read [complete findings](FINDINGS.md), [all numerical reports](results/summary.json),
[the validation record](VALIDATION.json), and [the experimental protocol](PROTOCOL.md).
The study establishes an explicit, tested forecasting construction and a
within-KAHM training benefit in one regime; it does not establish distinctive
general superiority or an incremental-update benefit.

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

**Bitwise replay limitation.** Package-version pins alone do not guarantee
identical Duffing regeneration across CPU dispatch paths. Native local replay
uses NumPy's `AVX512_SKX` float64 power kernel and matches the saved inputs.
Disabling AVX-512 changes values by up to `2.04e-13` in the four batches of
one checked new-study task, which fails the unchanged exact generator check.
The [recorded platform probe](validation/platform_replay_probe.json) preserves
the commands, environment, measured differences, and limited scope. Saved
input and result hashes remain unchanged; numerical closeness does not count
as a passed bitwise audit. Full strict replay requires compatible numerical
kernels in addition to the declared package versions.

A [separate one-state probe](validation/raw_rbf_portability_diagnostic.json)
also reproduces a strict selected-state replay failure for Wikipedia's third
raw-RBF comparator when NumPy AVX-512 is disabled. One-ulp exponential-kernel
differences grow into prediction changes of at most `2.18e-9`, crossing the
unchanged pointwise tolerance near zero; relative MSE changes stay below
`1e-14`. Native replay passes. The [probe script](validation/raw_rbf_cpu_probe.py)
performs no fits. A complete [hosted strict audit and package build](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/actions/runs/41185641611)
also passed, while another worker failed the documented checks. The preserved
forecasts, scientific source, analysis and tolerances are unchanged.


A repository checkout stores completed task evidence in lossless ZIP files.
Restore their exact original bytes and audit all saved outcomes with:

```bash
python -m extensions.direct_reconstruction.evidence unpack \
  --run-dir extensions/direct_reconstruction/results
python extensions/direct_reconstruction/analyze.py \
  extensions/direct_reconstruction/results --verify-live-source
```

The transport manifest verifies every archive and contained file. Unpacking
accepts identical existing files and refuses to overwrite a differing file.
The self-contained reproduction ZIP already contains the unpacked originals.
The task archives are post-execution packaging; the scientific freeze and
outcomes remain byte-identical.

```bash
PYTHONPATH=. python extensions/direct_reconstruction/test_model.py -v
python extensions/direct_reconstruction/test_protocol.py
python extensions/direct_reconstruction/run.py smoke --output /tmp/kahm-direct-smoke
```

Run these commands from the repository root. The frozen model test imports
the repository package; the frozen protocol test imports sibling modules
and therefore runs as a script. These invocation choices preserve the
scientific source hashes.

Preparation saves inputs and a scientific freeze without fitting a forecaster:

```bash
python extensions/direct_reconstruction/run.py prepare --output /tmp/kahm-direct-new
```

The `run` subcommand takes the full 40-character commit identifier containing the prepared freeze. Use a distinct output directory to retain the reported execution separately. A repeated fit provides computational reproduction of the specified procedure; it does not create another independent confirmation after the outcomes have been inspected.

The independent analysis reconstructs windows, losses, C selection, costs
and frozen hashes, and replays selected serialized models. It does not
independently refit every optimization candidate. Inspect `--help` for the
runner and analysis entry points. Every planned task and method is retained,
including unavailable tasks, optimization budget exits and numerical fallback.

## Reuse a selected forecasting state

The model API is `fit_suite(Xa, Ya, groups_a, Xb, Yb, Xc, Yc, Xd, seed, ...)`.
It returns C/D predictions, detailed diagnostics and serialized selected
states for all 17 procedures. Targets have shape `(rows, horizons)`; histories
have shape `(rows, history_length)` and run oldest to newest in original
units. Natural callers must supply `A_inner_train_mask` and
`A_inner_validation_mask` respecting target maturity. `Xd` contains histories
only: the fitting API has no D-target argument.

After unpacking the evidence, a study-generated selected model can be reused:

```python
import zipfile
from extensions.direct_reconstruction.model import predict_serialized

with zipfile.ZipFile(task_directory / "selected_models.zip") as archive:
    state = archive.read("reconstruction_kahm.pkl")
prediction = predict_serialized(state, raw_history_rows)
```

Use the history length and ordered offsets from that task's `task.json`.
The fitted input scaler, local bank, horizon parameters, head, mixing weights
and fallback baseline are already in the state. No caller standardization or
head refitting is needed for replay. This API is a research implementation;
its successful replay does not establish transfer to a different data source.

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
