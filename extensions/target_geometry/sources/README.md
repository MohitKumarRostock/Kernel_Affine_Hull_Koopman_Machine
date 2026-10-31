# New archive collections and attribution

These four Monash archive collections were selected by provenance, domain and
feasible size before fitting any model in the target-geometry extension. They
do not repeat the twelve collections in the earlier natural-source and additive
utility studies. They can nevertheless overlap those studies in broad subject
matter, such as electricity, transport and environmental monitoring. The
collections are not a random sample of forecasting applications.

All four pinned Zenodo records declare **CC BY 4.0**. `manifest.json` preserves
the complete creator attribution, record DOI, upstream source links, source
description, version, metadata SHA-256, archive SHA-256, upstream checksum and
extracted TSF-member SHA-256. The original compressed archives are retained.

| Collection | Pinned record | Context / direct forecast steps | Training-only selected series |
|---|---|---:|---|
| Australian state electricity demand, half-hourly | [10.5281/zenodo.4659727](https://doi.org/10.5281/zenodo.4659727) | 48 / 12 | T3, T5, T1 |
| Bitcoin indicators, daily, with missing values | [10.5281/zenodo.5121965](https://doi.org/10.5281/zenodo.5121965) | 28 / 7 | sent_addresses, confirmation_time, block_size |
| Melbourne pedestrian counts, hourly | [10.5281/zenodo.4656626](https://doi.org/10.5281/zenodo.4656626) | 48 / 6 | T10, T14, T20 |
| KDD Cup 2018 air quality, hourly, with missing values | [10.5281/zenodo.4656719](https://doi.org/10.5281/zenodo.4656719) | 48 / 6 | T239, T99, T233 |

The selected Bitcoin tasks forecast three **individual activity indicators**,
not bitcoin price or investment returns. The pedestrian archive ends on
2020-04-30. KDD contains individual air-quality measurements from Beijing and
London stations; this experiment supplies each model only the selected
univariate history, without external meteorology or cross-station information.

## Selection and evaluation scope

Every series is split chronologically by observation count at floor(0.6*n) and
floor(0.8*n). Eligibility requires a nonconstant, finite-enough training prefix,
a positive training-only difference scale and at least 96 complete training
windows. Eligible IDs are ranked by SHA-256 of the fixed salt, source ID and
series ID. The first three are retained. Later split availability does not
affect selection, and an unavailable later split is never replaced by another
series. All twelve selected tasks happen to have usable later splits.

The source-level eligible counts are 5/5, 18/18, 66/66 and 258/270 in the table's
order. `../prepared/roster.json` records the training-only exclusions, selection
hashes, exact forecast origins, split boundaries, scales and prepared-array
hashes. The twelve tasks each retain at most 512 evenly spread training windows
and 64 evenly spread validation and test windows. Missing contexts or targets
are omitted, without interpolation. The declared direct horizon's complete
target interval must lie inside its split. At later origins, previously observed
values from that same evaluation segment can enter the history.

The new primary experiment does not refit on validation observations before
testing. Its short-horizon, capped-window protocol differs from the Monash
archive's published leaderboard protocol; those leaderboard scores are not
directly comparable.

## Reproduction

From the repository root:

```bash
python -m extensions.target_geometry.data verify
```

To prepare the exact roster in a clean copy without prepared data:

```bash
python -m extensions.target_geometry.data prepare
```

The command verifies pinned source archives when the manifest exists and only
downloads a missing archive from its recorded URL. It refuses to overwrite an
existing roster. The data module imports the historical, unchanged
`extensions.natural_forecasting.data` window and scaling functions.

Archive citation: Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb,
Rob J. Hyndman and Pablo Montero-Manso (2021), *Monash Time Series Forecasting
Archive*, NeurIPS Datasets and Benchmarks. The dataset records give the original
upstream sources and individual dataset citations.
