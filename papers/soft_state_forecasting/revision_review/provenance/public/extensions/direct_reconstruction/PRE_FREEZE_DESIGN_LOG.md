# Pre-freeze feasibility decisions

This log records decisions made before public protocol freezing and before any
forecasting model was fitted to the newly reserved natural sources. These are
design and training-prefix feasibility decisions, not forecasting results.

## 2026-10-31: rideshare history length

The initial plan used an hourly history of 48 observations, seasonal period 24,
and target offsets 1, 3, and 6 for rideshare mean-price series. The pinned
archive is Zenodo record [5122114](https://zenodo.org/records/5122114), DOI
[10.5281/zenodo.5122114](https://doi.org/10.5281/zenodo.5122114). The source
metadata filter is `type=price_mean`, which identifies 144 series, each with
541 raw observations. Eligibility requires at least 96 finite A windows and a
positive finite A-only loss scale. No values are imputed.

With the raw 50/20/15/15 chronological split, the initial A prefix contains
270 observations. Inspection of that prefix alone found 82–85 missing values
per mean-price series. A history of 48 gave 68–91 usable A windows, so no series
could meet the 96-window minimum. This was a source-feasibility failure before
any model training or evaluation.

An A-only diagnostic also counted available windows at shorter histories. A
history of 24 gave 107–130 usable A windows; a history of 12 gave 138–154. The
design was amended to **history length 24**, which retains an observed full
daily seasonal lag. The seasonal period remains 24, target offsets remain
1, 3, and 6, and the eligibility threshold remains 96. The source metadata
filter, identifier-hash selection rule, chronological split, row caps, loss
scale, model budgets, and future-data handling are unchanged.

The amendment was approved before the final source roster and protocol were
frozen. B, C, and D availability and forecast errors did not determine this
choice. Selected series will not be replaced if a later segment lacks usable
windows. All four archives are retained as chosen collections, with three
identifier-selected series each for retail, rideshare, and Wikipedia, and one
sunspot series. These ten tasks do not represent ten independent domains.

## Preserved preparation evidence

`data/sources_manifest.json` records pinned archive checksums, decompressed
member hashes, licensing and attribution, training-prefix eligibility counts,
identifier hashes, selected series, exact raw segment boundaries, loss scales,
and prepared-array hashes. Each task NPZ preserves raw observations and
indices, available and retained origins, explicit history/target indices, and
purged inner-A training/validation masks. Preparation computes availability
and training-only scales; it does not fit predictors or inspect forecast
errors.

## 2026-10-31: later-segment availability audit

After A-only eligibility and identifier selection were complete, the input
audit found that all three selected rideshare series have zero usable B
windows under the declared split and missing-window exclusion rule. The
selected series and preparation settings are retained. This later-segment
availability finding does not trigger a replacement, another history-length
change, imputation, or a revised chronological split. The runner records the
three unavailable tasks before attempting to fit models that require B.

The other seven tasks have nonempty A/B/C/D arrays. The full planned
four-collection aggregate is undefined when the required procedure cannot
be fitted for one collection; a supplemental three-collection summary must
be explicitly labeled availability-conditioned. This documented limitation
precedes model fitting and does not use forecast errors.
