# Current KAHM time-series manuscript

The current manuscript is **Kernel Affine Hull Machines for Time-Series
Forecasting: Forecast-Trained Local Geometry and Matched Evaluation**.

Read [the revised PDF](extensions/tmlr_integration/review/KAHM_Forecast_Trained_Manuscript.pdf),
[the editable source](extensions/tmlr_integration/manuscript/main.tex), and
[the complete build guide](extensions/tmlr_integration/README.md).
It is an unsubmitted research draft in TMLR preprint style.

## Direct forecast-trained reconstruction

The [current experiment](extensions/direct_reconstruction/README.md) trains
KAHM kernel bandwidths and gate temperatures for the actual forecast and
horizon, with equally supervised affine, locality and RBF-local alternatives.
Its [public freeze](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/39174d06dffc7e28fed73dbbf866e539560dd021)
contains the numerical implementation, independent analysis and all 46
planned input sets before evaluation. Three rideshare tasks are retained as
availability failures; they prevent a complete four-collection aggregate.
The seven-series archive subset is explicitly availability-conditioned.

The manuscript defines the complete forecasting family and separates its
methodological specification from any comparative superiority supported by
actual results. The [new packager](extensions/tmlr_integration/package_direct.py)
produces the complete editable manuscript and a self-contained reproduction
bundle for this extension and its separate exploratory gate pilot.

The completed comparison yields 2 positive, 6 negative and 10 unresolved
adjusted intervals for reconstruction-trained KAHM. It demonstrates a 41.43%
Van der Pol improvement over gate-only KAHM, while matched and practical
comparators remain stronger in important settings. The
[full findings](extensions/direct_reconstruction/FINDINGS.md) report all
procedures, unavailable tasks, source dependence and costs.

## Previous shared proxy-metric study

The [new study](extensions/target_geometry/README.md) compares a ridge baseline,
matched local geometric corrections, in-sample versus excluded-donor residuals,
and a training-only supervised metric shared across KAHM, affine, locality and
RBF-local gates. The public checkpoint
[`0d7ee3c`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/0d7ee3c17515eb1d922cdd302facd2b7874dc9a9)
preceded the 48-task execution. All 768 forecasts are retained and independently
audited, with no fallback forecasts or missing tasks.

Learned-metric KAHM improves ridge by 11.18%, 53.26% and 19.44% on the three
synthetic laws under the 18-comparison analysis. Neither the metric increment
nor the OOF target increment is resolved there, and five matched-control
comparisons favor simpler geometries. On four new archive collections, its
scaled MSE is 1.2721 versus ridge's 1.3005 (2.18% lower), at 0.162 versus
0.00130 seconds of charged pipeline time. Fixed-metric in-sample KAHM reaches
1.2646, and simpler corrections remain stronger. These descriptive archive
results do not establish a KAHM-specific advantage.

See [the complete findings](extensions/target_geometry/FINDINGS.md),
[the residual transport identity](extensions/target_geometry/THEORY.md), and
[the focused research roadmap](extensions/target_geometry/RESEARCH_ROADMAP.md).
The new reproduction bundle covers this extension; it does not replace older
studies or their independently preserved failures. The historical natural study
still reports 4.26% higher error and 2.30 times the cost for its tested KAHM
augmentation, and the additive-utility study retains its eight failed model
fits and post-hoc matched-success diagnostic.

## Historical study context

## New bounded scaling sensitivity

The updated manuscript now includes a fixed, paired comparison of training
z-score geometry scaling and unclipped min-max scaling to `[-1, 1]`, covering
all 208 original natural configurations and all 24 original controlled draws
per law. The public execution checkpoint preceded the new fits, but the data
had already been inspected, so the paper labels this **post-hoc sensitivity**.

On the four formerly reserved natural sources, raw plus KAHM changes from
**0.8419228 to 0.8421913** source-balanced scaled MSE (**+0.0319%**). It remains
**4.29% above raw RBF** and **8.00% above raw ridge**. The main controlled
KAHM-local comparisons also retain their ordering. Min-max improves Duffing
at 256 landmarks by 7.35%, yet KAHM still trails raw and both local controls.
A secondary score-only-versus-affine ordering does reverse on Van der Pol;
the paper explicitly distinguishes that result from the main comparisons.

See the [sensitivity findings](extensions/scaling_sensitivity/FINDINGS.md),
[fixed protocol and reproduction guide](extensions/scaling_sensitivity/README.md),
and [complete audited summary](extensions/scaling_sensitivity/analysis/summary.json).
All 1,664 new natural method runs and 1,440 controlled head fits are retained.
The original frozen benefit diagnostic was not retuned.

## Reproduce and prepare review files

The [integrated guide](extensions/tmlr_integration/README.md) supplies commands to
restore the saved natural run, audit both studies without forecasting fits,
compile the paper, and build deterministic anonymous source and review ZIPs.
The review export preserves scientific bytes and recorded outcomes while
omitting identifying overviews; it explicitly distinguishes its post-study
source inventory from the original execution freezes.

The original TMLR checkpoint remains at
`a67c40190458bab7b3e56b4fac76770798d074b3`. The completed standalone natural
study remains at `750805cef68711e0aeb9c7c14a2d6cd5a785bf30`. The integrated
manuscript before this sensitivity remains at
`ede33f83cb2e6a5eae687282cec93c60cd31d46d`; the new execution freeze is
`c5b8817a9157de685fd1c70a22d83542a7bdf766`. Original scientific source code,
results, and manuscript checkpoints are preserved. The CI workflow restores
both reference artifacts, checks the original and new freezes, compiles the
integrated paper, and verifies and audits the anonymous review export.

## Research status

This is an unpublished draft prepared for substantive author review. The
computational validation does not certify author approval or acceptance by a
journal. The paper includes a first-page disclosure of AI assistance with
protocol development, implementation, verification, analysis, and writing.
The authors must verify the scientific content, settle authorship and required
funding/conflict declarations, and approve any eventual submission.
