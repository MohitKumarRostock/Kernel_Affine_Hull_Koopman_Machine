# Original PenDigits trajectory benchmark: data and statistical scope

## Source and acquisition protocol

The data are the original trajectories from E. Alpaydin and F. Alimoglu (1996),
*Pen-Based Recognition of Handwritten Digits*, UCI Machine Learning Repository.

- [Dataset record](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)
- [DOI: 10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K)
- License reported by UCI: Creative Commons Attribution 4.0 International.
- [Source archive](https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip)
- Archive SHA-256: `1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64`.
- Retrieval date recorded for the original experiment: 28 October 2026.

The archive contains the original, unnormalized UNIPEN files
`pendigits-orig.tra.Z` and `pendigits-orig.tes.Z`, with format and acquisition
information in `pendigits-orig.names`. These `.Z` files use UNIX compress.
`experiments/prepare_pendigits.py` decodes them with `unlzw3` when available
or the `gzip -dc` executable; Python's `gzip` module does not decode this format.
The parser verifies the archive hash, declared record and writer-group counts,
coordinate range, sampling interval, and consistency of digit labels.

The acquisition documentation specifies a 500-by-500 coordinate box, integer
coordinates in `[0,500]`, and a nominal sampling interval of 100 ms. The
forecasting experiment applies the fixed coordinate map `z / 250 - 1`, yielding
values in `[-1,1]`. Scalar predictions are clipped to the same interval. These
bounds come from the acquisition protocol, rather than observed sample extrema.

## Causal inputs and stroke eligibility

The commonly used 16-dimensional PenDigits table contains eight spatially
resampled locations after whole-character normalization. Its preprocessing
depends on the completed character. The present forecasting experiment instead
uses the original time-sampled coordinates, with no per-stroke centering,
alignment, normalization, or resampling.

A `.SEGMENT DIGIT` record can contain several `.PEN_DOWN`/`.PEN_UP` strokes.
The parser retains their boundaries, and forecasts never bridge a pen-up gap.
The experiment uses only the first pen-down stroke of each digit, requires at
least 32 recorded points, and retains its first 32 points. Eligibility depends
on the completed stroke length. The task is therefore forecasting within a
retrospectively defined eligible-stroke population, rather than forecasting
every newly initiated online stroke.

The official development file contains 7,494 digits from 30 writers; the test
file contains 3,498 digits from a disjoint set of 14 writers. Development
writers 1–24 supply training strokes, and writers 25–30 supply validation
strokes. Each fitting seed samples 20 eligible strokes per training writer
without replacement, giving 480 training strokes. Validation uses all 717
eligible strokes from the six validation writers. The test filter leaves
1,751 digits from 13 writers: test writer 5 has no qualifying first stroke.
No loss is imputed for that writer.

## Writer identifiers

Each digit has a `.COMMENT a b c` record containing three integers. In the
development file, the second field takes values 1–30, with approximately 250
records per value; in the test file, it takes values 1–14 with comparable
counts. The first field agrees with the digit label, and the third ranges from
1 to 250. These patterns support interpreting the second field as a within-file
writer identifier and the third as a within-writer sample identifier. The
companion documentation does not explicitly define the fields, so this
interpretation remains an inference.

Writer identifiers include the source-split label. Development writer 1 and
test writer 1 therefore denote distinct groups under the documented split.
The original parsing summary reports the following observed ranges; they are
diagnostics and do not define the population bounds.

| Source file | Digit records | Writer-like groups | Minimum coordinate | Maximum coordinate |
|---|---:|---:|---:|---:|
| Development | 7,494 | 30 | 12 | 449 |
| Test | 3,498 | 14 | 32 | 472 |

The eligible test-writer counts reconstructed from the released writer-loss
records are:

| Test writer | Eligible digits | Test writer | Eligible digits |
|---|---:|---|---:|
| 1 | 239 | 8 | 93 |
| 2 | 142 | 9 | 39 |
| 3 | 151 | 10 | 192 |
| 4 | 120 | 11 | 6 |
| 5 | 0 | 12 | 144 |
| 6 | 222 | 13 | 15 |
| 7 | 175 | 14 | 213 |

## Loss aggregation and interpretation

For each scalar coordinate and forecast horizon, squared errors are averaged
first over forecast origins within a stroke, then over eligible strokes within
a writer, and finally with equal weight across eligible writers. The two
coordinate MSEs are averaged before taking the square root. Reported means and
sample standard deviations are then computed across fitting seeds. This order
of aggregation prevents writers with more eligible digits from receiving more
weight in the cohort error.

The independent-writer assumption required for a population-risk certificate
is not established by the dataset documentation. Test-cohort errors are
descriptive. The additional Hoeffding upper bounds are conditional on treating
the 13 eligible test writers as independent, identically distributed draws
from a common eligible-writer population, independent of development data.
Overlapping windows and multiple strokes from one writer do not increase
that independent sample size. The resulting bounds do not certify the
reported RMSE tolerances of 0.2, 0.35, or 0.5.

Later input-scaling and matched-geometry studies reuse the same held-out
writers. Their new fitting seeds change training-stroke subsampling and
geometric fitting; they do not provide independent test cohorts or establish
uncertainty over a new writer population. PenDigits has no separate independent
calibration-writer cohort for the comparative geometry results.

An alternative guarantee for a fixed finite cohort would require an explicitly
specified without-replacement sampling design and a corresponding bound. Such
a guarantee is not part of the present experiment or the iid-episode theorem.
