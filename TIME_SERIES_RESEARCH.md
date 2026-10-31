# KAHM for time-series research

**Kernel Affine Hull Machines (KAHMs) provide local reconstruction geometry
that can serve as a feature map or gate for forecasting. This repository
studies when that geometry helps a finite predictor, whether the benefit
survives matched simpler controls, and whether it justifies the full cost.**

The current extension starts from a competitive raw-history ridge forecast.
It compares in-sample and held-out donor residuals on the same correction rows,
then learns a constrained diagonal metric from a kernel residual-regression
proxy. The identical learned metric is supplied to KAHM, affine, locality, and
RBF-local gates. This tests the transfer of a supervised metric to KAHM's
forecasting features; it is not end-to-end optimization of a KAHM forecast.

The code, protocol, source roster, and dependencies were published before the
new full execution. The [publication record](extensions/target_geometry/PUBLICATION.json)
identifies that checkpoint. Favorable findings, unfavorable comparisons,
unresolved differences, and operational failures belong to the evidence.

## Evidence from the completed extension

The [audited results](extensions/target_geometry/reports/summary.json) cover
all **48 tasks and 768 task–method prediction records, with no fallbacks**.
Learned-metric KAHM improves on ridge in each of the three synthetic laws under
the declared 18-contrast interval family. Its improvement over the identity
metric remains unresolved on all three laws, as does the change from in-sample
to held-out donor residuals. Among the nine learned-metric comparisons with
matched geometric controls, five favor the control and four remain unresolved;
none establishes a KAHM advantage.

Across the four selected archive collections, learned-metric KAHM reduces the
source-balanced scaled MSE by **2.18%** relative to ridge. Its mean measured
pipeline time is **0.162 s versus 0.00130 s**, or **124.9×** ridge. Identity-metric
in-sample KAHM has a larger descriptive reduction, **2.76%**, and identity-metric
in-sample RBF-local has the lowest aggregate loss, **3.27%** below ridge. The
learned affine and locality controls also have lower aggregate losses than
learned KAHM. These are descriptive results for the selected sources and one
timing environment.

This evidence supports conditional usefulness of geometric residual features.
It does not establish a predictive advantage specific to KAHM or to the new
metric-learning step. The next priorities are to test geometry trained directly
for the forecast and horizon, and to investigate bounded incremental updates
that could make the representation less expensive. Both are **untested
hypotheses**, requiring fresh evaluation sources and prospectively specified
matched controls; retuning this completed roster would be development work.

## Start here

| Resource | Purpose |
|---|---|
| [Target-geometry study](extensions/target_geometry/README.md) | Current question, model components, frozen evaluation, and reproduction commands |
| [Execution protocol](extensions/target_geometry/PROTOCOL.md) · [configuration](extensions/target_geometry/config.json) | Exact controls, donor boundaries, primary contrasts, fallback rules, and timing scope |
| [Audited summary](extensions/target_geometry/reports/summary.json) · [all primary contrasts](extensions/target_geometry/reports/primary_contrasts.csv) | Completed coverage, favorable and adverse comparisons, uncertainty, and resource results |
| [Manuscript guide](TMLR.md) · [editable source](extensions/tmlr_integration/manuscript/main.tex) | Integrated, unsubmitted research draft and compiled review files |
| [Residual-learning theory](extensions/target_geometry/THEORY.md) | Predictable residual signal, fitted-correction error, and donor-to-final mismatch |
| [Research roadmap](extensions/target_geometry/RESEARCH_ROADMAP.md) | How the declared comparisons guide the next research decision |
| [Earlier forecasting-utility study](extensions/forecasting_utility/README.md) | The preceding additive study, including its mixed results and fit failures |
| [Earlier quick forecasting example](examples/forecasting_utility.py) | A small example of the earlier raw-forecast-plus-correction architecture; not the new benchmark |
| [KAHM classification entry repository](https://github.com/MohitKumarRostock/KAHM) | Installation and the separate classification example with fair baselines |

## What a forecasting claim must specify

A useful comparison names the **history, future target, fitted representation,
decoder, evaluation law, and complete resource scope**. The new study uses
three established synthetic laws with independent new repetitions and four
additional selected archive collections. The archive results describe those
collections; they are not a representative survey of applications or the
Monash leaderboard protocol.

KAHM features are deterministic functions of the observed history. They can
make existing structure easier for a finite learner to use, but do not add
information to a history already retained in full. Better reconstruction,
a lower metric-training objective, and an available zero correction each
answer a different question from improved future prediction.

The matched controls determine whether a benefit is attributable to KAHM's
reconstruction geometry or to the shared local features and supervised metric.
The separate risk-certificate work addresses representation-class questions
under explicit calibration and decoder assumptions. Its class-level bounds
should not be interpreted as evidence of forecasting superiority.

## Preserved Koopman research

KAHM names the local geometric construction. KAHKM names the earlier
Kernel Affine Hull Koopman Machine architecture for soft-state dynamics.
Their targets and decoders differ; results for one architecture do not
automatically transfer to another.

The earlier implementations, declared outcomes, and provenance remain
versioned. The [original result mapping](reproduction/reported_results.tsv)
and [reproducibility evidence](reproduction/REPRODUCIBILITY_EVIDENCE.md)
identify the historical chains. The new extension does not overwrite earlier
negative comparisons or retrospectively replace the earlier failed fits.
