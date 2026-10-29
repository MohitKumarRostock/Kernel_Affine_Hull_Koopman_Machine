# Scientific verification of the revised manuscript and implementation

Date: 29 October 2026. Repository baseline:
`531dadef91bd467f9eeb04d5716afbb1d9fd0cbb`.

The review checked the manuscript's derivations, statistical assumptions,
implementation correspondence, retained numerical results, and public
documentation. The revised exposition defines the mathematical objects before
use and distinguishes population results, finite-sample confidence statements,
and descriptive empirical comparisons. No unresolved inconsistency was found
in the core proofs under their stated assumptions. The review identified and
corrected a calibration-pairing edge case, several inaccurate descriptions,
and overbroad interpretations of reused evaluation data.

The verification reconstructs retained evidence and includes targeted numerical
checks. It is not a machine-checked proof, a complete rerun of every production
experiment, or an exhaustive novelty assessment. The detailed records are in
[`verification/readability_scientific_revision/`](../verification/readability_scientific_revision/).

## Scientific corrections

### Calibration reuse and the scope of confidence statements

The original-seed input-scaling study reuses calibration and test streams after
earlier results motivated the preprocessing comparison. Fixing a new protocol
before rerunning those streams does not establish that the protocol is
independent of the previously examined outcomes. The revised manuscript,
study READMEs, protocol documentation, table captions, and report generator
therefore describe these endpoints as nominal calculations and the comparisons
as descriptive. Their numerical values are unchanged.

The synthetic scaling confirmation and matched-geometry study generate new
calibration episodes. Their stated probability guarantees retain their
within-arm or within-task/seed scope, with the documented allocation of failure
probability. The matched study allocates 0.04 to four arm families and 0.01 to
six primary comparisons, giving confidence at least 0.95 within one synthetic
task and fitting repetition. It does not give a simultaneous 0.95 statement
across every task and repetition.

PenDigits follow-ups reuse the same previously examined cohort. Repeated
fitting seeds quantify variation on that cohort; they do not add independently
sampled writers. An iid-writer assumption alone is insufficient for a follow-up
confidence claim when previous evaluation outcomes informed the design.
The revised text states both the sampling and design-independence requirements.

### PenDigits units and preprocessing

The implementation uses the fixed acquisition-coordinate transformation
`x / 250 - 1`. The original files contain 14 held-out writer identifiers, of
which 13 have eligible trajectories under the stated length restriction.
Writer 5 has no eligible trajectory. The retained evaluation contains 1,751
digits and weights the 13 eligible writers equally. Earlier documentation that
described division by 500 or 14 evaluation units has been corrected.

The meaning of the inferred writer field remains a documented assumption.
The retrospective length restriction also remains explicit: the results
concern eligible completed strokes, whose eligibility is not generally known
at the beginning of an online stroke.

### Disjoint pairing in the exact-risk experiment

The original exact-risk runner used alternating slices without first removing
an odd final observation. Most odd sample sizes produced incompatible shapes;
at size three, array broadcasting reused an observation and could create
dependent pair scores. The runner now requires at least four calibration
observations and constructs exactly `floor(n / 2)` disjoint pairs. An odd final
observation is excluded only from the pair statistic; all observations remain
in the fitted-risk and centered statistics.

All retained calibration sizes are even. Their pair scores are bitwise equal
to the historical calculation, so this correction does not change reported
experimental results. Five regression tests cover invalid sample sizes, odd
pairing, the correct sample size in the concentration bound, integration with
the upper-loss calculation, and equality at retained even sizes.

### Confidence metadata and mathematical descriptions

The exact-risk summary now derives its confidence percentage from the
configured failure probability. The oscillator summary derives confidence and
the number of fitting repetitions from its configuration, rather than using
fixed descriptions of the original schedule. These changes affect descriptions,
not numerical estimators. The oscillator table and its generator now label the
reported quantity as the mean of square-root upper endpoints. The matched
dimension figure is described as percentage change in MSE.

Training histories are distinguished from the larger development set, which
also includes validation data. Appending feature blocks preserves earlier
predictors by class inclusion; exact duplication leaves the feasible prediction
class equal. A squared ridge penalty can change under duplication, so the text
does not extend class invariance to the fitted penalized estimator.

## Mathematical and implementation checks

The review covered the pairwise obstruction, conditional root-variance
comparison, two-atom exactness, episode-level lower and upper bounds, model
selection, retained-block nesting, duplicate-block invariance, necessary
distance expansion, residual-correlation improvement, block-ball projection,
and first-order optimization gap. It checked the same conditioning law,
finite-moment assumptions, matrix dimensions, and sensitivity metric in each
result. The [mathematical account](THEORY_AUDIT.md) gives the derivations and
their correspondence with code.

The empirical Bernstein radius agrees with
[Maurer and Pontil (2009), Theorem 4](https://arxiv.org/pdf/0907.3740), after
rescaling to the relevant interval. The implementation uses unbiased sample
variance, `log(2 / alpha)`, and the factor `7 / (3 * (N - 1))`.
Episode losses have interval length `B`, pair obstruction scores have length
`B / 2`, and paired loss differences have length `2 * B`. Upper and lower
directions receive their stated separate allocations. Dependence among
different confidence statistics on the same fresh episodes does not invalidate
their union bound.

Independent finite-law and numerical checks included 300 conditional
vector-target constructions, 300 two-atom attainments, 300 necessary-expansion
comparisons, 300 residual-gain identities, 30 small constrained optima solved
independently, 100 source-formula comparisons, and 336 exact binomial coverage
calculations for the centered comparator. Discrepancies in checked identities
were at floating-point precision. These finite checks supplement the proofs;
they do not establish universal validity by simulation.

The complete repository test suite passes **50 tests with no skips** in the
recorded Linux/Python 3.12 environment. The historical reference commit was
available for the default-prediction regression test. That test compares all
40 default candidate descriptions and predictions with the published runner
and requires exact numerical equality.

## Independent reconstruction of the numerical evidence

The new verification scripts use the Python standard library and do not import
the experimental runners or the implementation of the confidence formulas.

| Evidence | Independent reconstruction | Result |
|---|---|---|
| Original empirical tables | 57 displayed rows across six tables | All entries agree after the documented rounding |
| Original synthetic studies | 1,796 candidate records; 5,388 lower/upper pairs; 792 paired-improvement bounds | 10,776 risk endpoints and all paired bounds agree exactly with saved moments |
| Exact-risk benchmark | 28,800 analytically known minimum risks | Exact agreement; the recorded one pairwise violation and zero centered/upper violations are retained |
| Original PenDigits | 9,360 writer-loss rows; 720 candidate–horizon MSEs | Maximum aggregation discrepancy below `1.4e-17` |
| Original-seed scaling | 8,144 candidate records; 3,264 selected models; 285 taskwise effect rows | Selected identities agree exactly; maximum effect discrepancy below `1.5e-13` |
| Scaling confirmation | 3,840 candidate records; 1,152 selected models; 105 taskwise effect rows | Selected identities agree exactly; maximum effect discrepancy below `1.6e-13` |
| Matched geometry | 13,440 candidate records; 360 primary selections; 42 primary contrasts | Selected identities agree exactly; maximum discrepancy below `4.3e-14` |

The matched reconstruction preserves all 28 favorable and 14 adverse mean
comparisons for KAHM. These are related descriptive contrasts. Selection uses
validation MSE; percentage comparisons are ratios of means across fitting
repetitions. PenDigits pools coordinate losses and weights eligible writers
equally. The separately recorded synthetic comparison procedure yields one
positive improvement endpoint among 180 comparisons; the remaining comparisons
are unresolved.

The original-synthetic confidence reconstruction uses retained sufficient
statistics. It does not recompute those statistics from every original full
episode-score array. Full archives and archived audits retain their original
provenance; their historical audit counts are not presented as newly rerun
checks.

## Reproduction

From the extension directory, using the reference dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -v
python scripts/verify_retained_statistics.py \
  --output verification/retained_statistics_recheck.json
python scripts/verify_scaling_statistics.py \
  --output verification/scaling_statistics_recheck.json
mkdir -p verification/matched_records_recheck
tar -xzf results/matched_geometry_study/records.tar.gz \
  -C verification/matched_records_recheck
python scripts/verify_matched_summary.py \
  --study verification/matched_records_recheck \
  --analysis results/matched_geometry_study/analysis \
  --output verification/matched_statistics_recheck.json
```

The commands require new verification output paths. For the historical parity
test, the checkout also needs commit
`71efa873419e2b58ee89746dafd123967ff007b1`; a shallow clone without that commit
reports a skipped test. The [reproduction guide](REPRODUCIBILITY.md) distinguishes
these scalar-record checks from full-score audits and model refitting.

## Manuscript and evidence integrity

The manuscript now includes a notation guide, dimensions and domains at first
use, expanded proof steps, explicit sampling units, and precise aggregation
definitions. Public documentation uses scientific exposition in place of
internal review instructions. Necessary scientific qualifications and the
disclosure of AI assistance are retained.

All original numerical table entries and supplied figure assets are preserved.
Historical configurations, scalar results, score indices, source snapshots,
and frozen protocols remain unchanged. Caption generators reproduce the
revised scientific interpretation. The LaTeX manuscript builds to 53 pages
without undefined citations or references, overfull boxes, or oversized floats;
all pages were rendered for layout inspection.

The evidence supports the stated conditional mathematical results and the
reported empirical comparisons. It does not establish universal superiority of
KAHM, distribution-free validity after adaptive reuse of evaluation data,
independent sampling of the PenDigits writers, or risk guarantees under an
unspecified deployment shift.
