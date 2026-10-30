# Current integrated TMLR manuscript

The current manuscript is **Kernel Affine Hull Features for Prediction: Conditional Gains, Costs, and Limits of Transfer**.

Read [the compiled anonymous PDF](extensions/tmlr_integration/review/TMLR_Integrated_Manuscript.pdf).

Start with [the integrated study](extensions/tmlr_integration/README.md) and
[its LaTeX source](extensions/tmlr_integration/manuscript/main.tex).
It combines the controlled predictive-geometry experiments and the completed
natural-source forecasting evaluation in one anonymous TMLR-format paper.

The controlled results retain the Van der Pol gains and the adverse AR, Duffing,
and Airfoil comparisons. The natural confirmation is reported just as clearly:
raw plus KAHM has 4.26% higher source-balanced scaled MSE than raw RBF at
2.30 times the measured warm-library pipeline cost. The geometric diagnostic
changes none of 80 confirmation decisions. Those 80 configurations belong to
four original source families and are not independent domains.

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
