# When do KAHM features help forecasting?

The study accounts for all 168 scheduled tasks. There are 166 fully completed tasks, 1,336 successful standalone
forecasts, eight geometric fit failures, and four dependent policy failures.

This extension studies the empirical utility of Kernel Affine Hull Machine
(KAHM) features for time-series forecasting. It retains a selected raw-history
forecast, fits a geometric correction to its training residuals, and uses
validation data to choose how much correction to add. Setting the mixing
coefficient to zero returns the raw forecast exactly.

Baseline availability is an architectural property. Whether validation selects
a useful correction is an empirical question; zero remains available without
guaranteeing that the selected predictor improves on a future test set.

## Recorded findings

The complete synthetic factorial finds a positive additive-KAHM versus raw-RBF
contrast in five of the twelve cells under the declared family-adjusted Student
intervals. Simpler geometric corrections have lower mean errors in the Van der
Pol and Duffing cells. Preserving a raw forecast does not by itself establish
a KAHM-specific advantage.

The geometric procedures fail on two capacity settings of one Car Parts series:
23 training windows contain only seven distinct histories, while the fixed
bank requests eight nonempty clusters. Those failures remain in the study.
Full-24-task KAHM and geometry-policy means are **undefined**.

The following table is a **post-hoc common-support diagnostic on 22 tasks and
11 series**, with equal weight to each of four archive collections. It has a
different scope from the full planned roster. Timing is the mean measured
model pipeline, including selection and serialization, on the recorded host.

| Procedure | Scaled test MSE | Model pipeline (ms) |
|---|---:|---:|
| Seasonal naive / persistence | 1.2062 | 1.20 |
| Raw ridge | 0.8466 | 1.52 |
| Raw RBF | 9.1730 | 3.75 |
| Concatenated KAHM | 1.2111 | 69.14 |
| Additive KAHM | 9.1443 | 49.81 |
| Additive affine | 9.3312 | 36.86 |
| Additive locality | 9.5624 | 37.86 |
| Fixed histogram boosting | 10.3174 | 638.04 |
| All-geometry 5% policy | 9.2624 | 86.85 |
| KAHM 5% validation policy | 9.2460 | 46.67 |

On this diagnostic support, additive KAHM reduces the raw-RBF mean error by
about 0.31% while using about 13.29 times its measured model-pipeline time.
Raw ridge has much lower mean error and cost. Source-wise results expose the
large COVID Deaths contribution to the RBF and tree averages. These results
support conditional representation evaluation; they do not justify general
KAHM adoption.

See [the reporting amendment](REPORTING_AMENDMENT.md),
[audited summary](reports/summary.json), and
[synthetic primary contrasts](reports/synthetic_primary_contrasts.csv).
Full-roster baseline results and method coverage are retained separately from
the common-support table. The reported accounting includes those original failed fits.

## Research design

The [protocol](PROTOCOL.md), [configuration](config.json), and
[execution manifest](freeze.json) define the study before its first full run.
The public checkpoint is recorded in `PUBLICATION.json`. This is a prospective
execution freeze, not an external registration. The original controlled,
natural-source, and scaling-sensitivity archives remain unchanged.

| Component | Fixed scope |
|---|---|
| Controlled experiments | Three numerical laws; 32 and 128 training episodes; 16 and 64 requested raw RBF landmarks; 12 fresh complete repetitions per cell; 144 tasks |
| Archive experiments | Weather, Solar Weekly, COVID Deaths, and Car Parts; three training-eligible, hash-selected series each; two requested landmark counts; 24 tasks |
| Matched methods | Raw ridge, raw finite RBF, concatenated KAHM, additive KAHM, additive affine, additive locality, naive, and a fixed histogram-gradient-boosting control |
| Decisions | Validation-selected mixing, a 5% KAHM improvement rule, and the corresponding all-geometry rule |
| Evidence | Saved targets and predictions, every validation trial, independent loss recomputation, paired complete-repetition summaries, and measured cost |

Solar Weekly contains aggregated **simulated NREL solar output**. The four
archive collections are not four purely observational domains, and the 24
configurations are not independent domain replications. Requested RBF landmark
counts can exceed the available training rows; effective landmarks and retained
basis rank are recorded. The fixed tree baseline is a practical control, not an
exhaustive tree-model search or a modern forecasting leaderboard.

## Quick start

From the repository root, with Python 3.12:

```bash
python -m venv .venv-forecasting
. .venv-forecasting/bin/activate
python -m pip install -r extensions/forecasting_utility/requirements.txt
python examples/forecasting_utility.py
```

The quick example uses a small generated series and chronological splits. It
illustrates the API and is separate from the research benchmark.

The additive interface reuses the fitted raw model:

```python
from extensions.forecasting_utility.models import (
    RawForecastRegressor, AdditiveForecastRegressor,
)

raw = RawForecastRegressor(head="rbf", n_landmarks=64, seed=7)
raw.fit_select(X_train, Y_train, X_validation, Y_validation,
               alphas=(0.01, 1.0, 100.0))
correction = AdditiveForecastRegressor(gate="kahm", seed=7)
correction.fit_select(
    X_train, Y_train, X_validation, Y_validation,
    alphas=(0.01, 1.0, 100.0), gammas=(0.0, 0.25, 0.5, 1.0),
    raw_model=raw,
)
prediction = correction.predict(X_test)
```

## Audit the saved study

The reference archive retains every successful final prediction, not only aggregate
tables. Restore it and recompute the recorded errors and summaries without
refitting a forecasting model:

```bash
python -m extensions.forecasting_utility.reference restore
python -m extensions.forecasting_utility.reporting \
  --run-dir extensions/forecasting_utility/results/utility_v1
```

The reporting command verifies complete accounting while preserving the original
analyzer's `complete: false` status for model failures. Running the frozen
`analysis` command directly returns exit code 1 because the full model roster is incomplete. That status is distinct from the checksum and numerical verification of the forecasts that were produced. The audit verifies
the exact task roster, saved targets and predictions,
selected validation settings, split hashes, charged computation nodes, and
gamma-zero equality. Synthetic confidence intervals use 12 complete
repetitions, with a Bonferroni adjustment across the 24 primary contrasts and
an explicit Student approximation. Archive results are descriptive and
source-balanced; correlated windows do not create independent replications.

## Repeat the experiments

The prepared archive inputs and their source metadata are included. Download
URLs, upstream attribution, record licenses, and byte hashes are in
[sources.json](sources.json). Missing values are not interpolated. The roster
was selected using training-prefix eligibility before any model outcomes.

Run the frozen study into a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m extensions.forecasting_utility.experiment \
  --phase all --output-dir /tmp/kahm-utility-rerun
python -m extensions.forecasting_utility.reporting \
  --run-dir /tmp/kahm-utility-rerun
```

Only an explicit `--resume` can reuse a matching run directory, and completed
tasks are preserved. Hash verification rejects changed scientific inputs.
A different design requires a separate source and input record, so its outcomes remain distinguishable from this fixed experiment.

Models are timed one task at a time with one numerical-library thread. Reports
distinguish model-only work from data-inclusive pipeline time. Rejected
corrections still incur their search cost. Warm-library measurements exclude
imports, downloads, and reporting. Model size describes the actual retained
serialized state; synthetic gamma-zero selections may retain the bank fitted
during selection even though prediction bypasses it.

## Manuscript and scope

The [integrated manuscript](../tmlr_integration/manuscript/main.tex) asks:
**When does a forecasting representation contain useful predictive structure,
and when is additional representation complexity justified?**

For this empirical study, useful structure means a correction that improves a
specified fitted predictor on reserved outcomes. Complexity is assessed using
the declared validation rule and measured error-cost tradeoffs. These results
do not estimate unrestricted Bayes risk or certify that every cheaper decoder
class is inadequate. An exclusion claim about every cheaper decoder would require a specified decoder class and additional assumptions.

The analysis distinguishes this additive-correction experiment from the original natural-data confirmation and the post-hoc scaling study. Incomplete coverage and unfavorable results remain part of the empirical conclusion.

## Tests and licensing

```bash
python -m pip install pytest==9.2.0
python -m pytest extensions/forecasting_utility/test_models.py \
  extensions/forecasting_utility/test_protocol.py -q
```

Code inherits the repository's MIT license. Original research outputs inherit
its CC BY 4.0 terms. Third-party data and source files retain their own notices
and attribution. See [LICENSE](../../LICENSE), [DATA_LICENSE](../../DATA_LICENSE),
and the pinned dataset metadata. Repository citation information is provided
in [CITATION.cff](CITATION.cff).
