# Reserved natural-source inputs

These files contain ten prepared forecasting tasks from four public source
collections. They were prepared before the direct-reconstruction experiment
was run. Source selection uses metadata, series length, and the initial raw
A prefix only. No forecast error or later-segment outcome determines the
roster. The collections were chosen for diversity and feasibility; they are
not a probability sample of forecasting domains.

## Attribution and data license

Each pinned Zenodo archive declares
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The selected raw observations and derived arrays retain that data license.
The repository's software license does not replace these data licenses.
The archive records credit Rakshitha Godahewa, Christoph Bergmeir, Geoff Webb,
Rob Hyndman, and Pablo Montero-Manso; `sources_manifest.json` preserves each
record's exact creator order, affiliations, ORCIDs, publication date, and
version.

| Collection and pinned record | Selected identifiers, in hash order | History | Target offsets | Seasonal period |
|---|---|---:|---|---:|
| [Dominick, version 2](https://doi.org/10.5281/zenodo.4654802) | T7830, T25851, T108533 | 52 weeks | 1, 2, 4 weeks | 52 |
| [Rideshare with missing values, version 1](https://doi.org/10.5281/zenodo.5122114) | T1809, T1825, T2401 | 24 hours | 1, 3, 6 hours | 24 |
| [Wikipedia web traffic daily with missing values, version 3](https://doi.org/10.5281/zenodo.4656080) | T81727, T132954, T138504 | 28 days | 1, 3, 7 days | 7 |
| [Sunspot daily with missing values, version 4](https://doi.org/10.5281/zenodo.4654773) | T1 | 28 days | 1, 3, 7 days | 1 |

The original sources linked by the archive providers are the
[University of Chicago Kilts Center Dominick dataset](https://www.chicagobooth.edu/research/kilts/datasets/dominicks),
[Uber and Lyft price data](https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices),
[Kaggle's Wikipedia forecasting competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting),
and [SILSO sunspot numbers](https://www.sidc.be/silso/newdataset), respectively.
Licensing statements above describe the pinned archive records. Rideshare
tasks are restricted to metadata `type=price_mean`; no geography-dependent
claim is needed for this experiment.

Changes from the source archives are explicit: identifier-hash selection,
the fixed final 4096-observation sunspot crop, chronological segmentation,
exclusion of windows containing missing history or requested targets, and
deterministic row caps. Raw selected observations remain in the arrays;
missing observations are represented as NaN and are not interpolated. The
Sunspot task is one series and one source, not three replicated domains.

## Reproduce the inputs

From the repository root:

```bash
python -m extensions.direct_reconstruction.prepare_sources --raw-dir /path/to/raw_direct_sources
```

The preparation script downloads any absent pinned archives into the requested
raw cache, checks their published MD5 values, and records SHA-256 hashes for
both each ZIP and its decompressed TSF member. Archives are not committed.
It writes `sources_manifest.json` and the ten deterministic NPZ files here.
The manifest contains the exact selection salt and hash expression, source
eligibility counts, source provenance, and expected NPZ SHA-256 hashes.
The default raw cache is outside the repository in its containing work
directory. Byte-identical array serialization uses fixed ZIP timestamps and
explicit little-endian numeric dtypes.

The initial eligibility requirement is at least 96 finite A windows and a
positive finite A-only seasonal-difference MSE, with a first-difference
fallback. The corresponding eligible-series counts are 17,082 for Dominick,
144 for rideshare mean prices, 130,539 for Wikipedia, and one for Sunspot.
The smallest three salted identifier hashes are selected per eligible
multi-series collection. The single Sunspot series is retained by the same
rule. A documented A-only feasibility amendment reduced rideshare history
from 48 to 24 before freezing; see `../PRE_FREEZE_DESIGN_LOG.md`.

## NPZ schema

All loading can use `numpy.load(path, allow_pickle=False)`. Each file contains:

| Keys | Meaning |
|---|---|
| `raw_values`, `raw_start_index` | Retained raw series and its starting index in the original series. Only Sunspot has a nonzero starting index. |
| `segment_bounds` | Integer boundaries `[0, floor(.50*n), floor(.70*n), floor(.85*n), n]` for A/B/C/D. |
| `L`, `p`, `offsets`, `loss_scale` | History length, seasonal lag, requested target offsets, and A-prefix squared-difference scale. |
| `{A,B,C,D}_X` | Histories, oldest observation first; last column is the observed value at the query origin. |
| `{A,B,C,D}_Y` | Targets at `origin + offsets`, in the stated offset order. |
| `{A,B,C,D}_origins`, `{A,B,C,D}_global_origins` | Query indices relative to `raw_values`, and relative to the complete original series. |
| `{A,B,C,D}_available_origins` | Every finite usable origin in that segment before capping. |
| `{A,B,C,D}_row_indices` | Selected positions in the corresponding available-origin array. |
| `{A,B,C,D}_history_indices`, `{A,B,C,D}_target_indices` | Explicit indices into `raw_values`, permitting independent reconstruction of X and Y. |
| `{A,B,C,D}_groups` | All zeros: each natural task is a single series; rows are not independent simulated episodes. |
| `A_inner_train_mask`, `A_inner_validation_mask`, `A_inner_purged_mask` | Chronological inner-A masks, with earlier training labels required to mature by the first validation query. |

Every requested target lies wholly in its assigned segment, and each preceding
segment's labels have matured by the next segment's first query. Histories can
include earlier observed values. The last 20% of retained A rows form inner
validation. Earlier rows whose farthest label is not yet available at that
first validation query are purged from inner fitting; they remain available
for the final A fit.

Row caps are 512/128/128/128 for A/B/C/D. If a segment has more than its cap,
retained available-row positions are `floor(j*(n-1)/(cap-1))` for
`j=0,...,cap-1`, including both endpoints. Missing-window counts are retained
as availability audits. A selected task is never replaced because its later
segments have few or no available rows.

The natural loss is raw MSE divided by the A-prefix **mean squared** seasonal
difference (no square root). The manifest records the fallback and pair count
for each task. Series are averaged within each source before sources receive
equal weight. This planned source-balanced metric describes the selected
collections, without population confidence intervals.

## Availability audit before model fitting

All three selected rideshare series have zero usable B windows under the
declared chronological split and missing-window rule. Their A segments pass
the eligibility rule; the later B failure did not determine selection and
does not trigger a replacement, new split, imputation, or shorter history.
The three tasks remain in the complete roster as unavailable training tasks.
The other seven tasks have nonempty A/B/C/D arrays.

Consequently the complete-roster, four-collection accuracy aggregate is
undefined for the procedures requiring B. Any descriptive accuracy summary
over the three available collections must be labeled availability-conditioned,
and the fourth collection's failure remains explicit. This is an input
availability finding, obtained before forecasting evaluation, not a forecast
performance result.
