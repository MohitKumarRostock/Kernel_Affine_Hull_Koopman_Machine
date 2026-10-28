# Original PenDigits trajectory benchmark

## Source

Dataset: E. Alpaydin and Fevzi Alimoglu (1996), *Pen-Based Recognition of
Handwritten Digits*, UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
- DOI: https://doi.org/10.24432/C5MG6K
- License shown by UCI: Creative Commons Attribution 4.0 International.
- Download: https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip
- SHA-256 of downloaded archive:
  `1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64`
- Retrieved 28 October 2026.

The archive includes original unnormalized UNIPEN trajectories as
`pendigits-orig.tra.Z` and `pendigits-orig.tes.Z`. The companion
`pendigits-orig.names` describes the original format and protocol. `.Z` is UNIX
compress, which `zcat` here can decompress; it is not a gzip stream for Python's
`gzip` module.

## Why use original records

The familiar 16-dimensional table contains eight locations obtained after
whole-character normalization and spatial resampling. That preprocessing depends
on the completed stroke and is unsuitable for an unqualified causal online
forecasting interpretation. Original coordinates are time sampled and avoid this
issue. Use a fixed division by 500 or a scaling fitted only on development data.

The original documentation specifies a 500-by-500 acquisition box, raw integer
coordinates between 0 and 500, and a nominal 100 ms sample interval. Do not infer
these bounds from observed sample extrema.

The training file has 7,494 digits from 30 writers. The official test file has
3,498 digits from a disjoint set of 14 writers. A `.SEGMENT DIGIT` record describes
a digit and can contain multiple `.PEN_DOWN`/`.PEN_UP` strokes. Forecasts should
not bridge pen-up gaps unless the gap and sampling convention are modeled
explicitly.

## Writer grouping

Every digit record has `.COMMENT a b c` with integer fields. Across the training
file, field b takes precisely the values 1..30, each appearing about 250 times.
Across the test file, it takes 1..14, also about 250 times. Field a matches the
digit label and field c ranges 1..250. This strongly supports interpreting b as
the within-file writer identifier and c as that writer's sample index. The
companion text does not explicitly define these comment fields, so record that
interpretation as an inference. Prefix IDs with the source split: training writer
1 and test writer 1 represent distinct people under the documented split.

Observed ranges (diagnostics only, not assumptions):

| File | Digit records | Writer-like groups | Coordinate min | Coordinate max |
|---|---:|---:|---:|---:|
| original training | 7,494 | 30 | 12 | 449 |
| original test | 3,498 | 14 | 32 | 472 |

## Statistical scope

Do not count all overlapping windows or all strokes as independent draws from a
population of writers. Writer-disjoint prediction results can be reported as a
real-data generalization benchmark. A writer-level population-risk bound would
have only 14 independent test units, even if the independence/exchangeability of
writers is assumed. It is likely to be too wide for stringent tolerances.

One rigorous alternative is to target a *fixed finite benchmark population*:
freeze predictors, uniformly sample calibration units without replacement from
a specified held-out cohort, and certify its finite-population average using a
valid without-replacement bound. This does not certify a new writer or future
handwriting distribution and should be labeled accordingly. It also needs its
own theorem and is not automatically covered by the iid-episode theorem.

Another option is to report empirical error only on this real dataset and retain
the exact finite-sample population certification experiments on independently
generated synthetic episodes. This is scientifically preferable to an unsupported
independence claim.
