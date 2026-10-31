# KAHM for time-series research

Kernel Affine Hull Machines supply local reconstruction geometry that can
serve as a forecasting feature map. This repository specifies that use and
tests its prediction accuracy, numerical feasibility and measured cost.
A KAHM forecasting method and a demonstrated KAHM-specific performance
advantage are separate scientific claims.

## The current forecasting family

The newest method receives only the observed history available at a forecast
origin. An A-fitted local bank supplies reconstruction gates and principal
coordinates. The forecast retains a raw ridge baseline and adds a separately
regularized residual correction. For each horizon, bounded kernel bandwidths
and gate temperatures are trained using the actual B forecast error. C selects
head penalty and correction weight; D evaluates the frozen result.

The KAHM bandwidth changes both the reference kernel and query kernel.
Reference membership, local principal directions, covariance estimates and
original kernel regularizers remain fixed. The method therefore trains part
of the reconstruction operator, not every KAHM component end to end.

Affine projection gain, locality balance and RBF-local anisotropy provide
explicit matched shape interventions. All four families have the same data
roles, parameter counts within each mode, and optimizer ceilings. Equal
optimization opportunity does not mean equal functional capacity or cost.
Raw ridge, raw-history-plus-RBF, gradient boosting, seasonal naive and a
C-selected non-KAHM policy make the practical comparison explicit.

## Evidence and its boundaries

The [public freeze](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/39174d06dffc7e28fed73dbbf866e539560dd021)
was published before the full reserved execution. It contains numerical code,
independent analysis, protocol, source provenance and all 46 planned inputs.
There are twelve fresh repetitions of each of three known laws and ten
A-selected natural series from four previously unused collections.

All three selected rideshare series lack complete B windows. Their membership
was determined from A only; they remain feasibility failures. The planned
four-collection accuracy aggregate is undefined. A seven-series description
over three available collections is explicitly conditional on feasibility.
No replacement series or revised split is used to complete the aggregate.

The three synthetic laws support a predeclared family of 18 paired contrasts
with Bonferroni t intervals over twelve repetitions per law. The natural
collections were chosen purposively and receive descriptive analysis. Related
windows and multiple horizons do not increase the number of independent
sources. These are single-vintage archive evaluations, without a claim about
historical real-time label releases or later revisions.

The separate [nine-task gate pilot](extensions/forecast_gate_pilot/INTEGRATION.md)
used independently implemented generators and a smaller search budget. It
remains exploratory and is not pooled with the new repetitions. Earlier
studies, including negative comparisons and fitting failures, retain their
original artifacts and statistical families.

## Result of the new intervention

All 43 feasible task suites completed, preserving 731 selected models and
prediction records. The 18 adjusted synthetic intervals give two KAHM gains,
six disadvantages and ten unresolved differences. KAHM beats raw RBF on
nonlinear AR by 2.87% and gate-only KAHM on Van der Pol by 41.43%. It does not
establish superiority over any equally shape-trained local control in this
family; affine wins on Van der Pol and Duffing, and locality also wins on
Duffing. The selected non-KAHM policy wins on both oscillators.

The natural available-case mean is 3.55% below ridge, but raw RBF, boosting and
trained RBF-local are more accurate in aggregate. The ridge gain is concentrated
in Dominick. Reconstruction-trained KAHM costs substantially more than the
strong raw alternatives in this implementation. See the
[full findings](extensions/direct_reconstruction/FINDINGS.md) and
[updated research questions](extensions/direct_reconstruction/RESEARCH_ROADMAP.md).

## Start here

| Resource | Purpose |
|---|---|
| [Direct reconstruction guide](extensions/direct_reconstruction/README.md) | Current method, costs, execution and audit commands |
| [Frozen protocol](extensions/direct_reconstruction/PROTOCOL.md) | Exact objective, controls, source roster, chronology, failure rules and statistical family |
| [Prepared source provenance](extensions/direct_reconstruction/data/README.md) | Licenses, identifiers, selection rules, original indices and availability |
| [Audited summary](extensions/direct_reconstruction/results/summary.json) | Complete coverage, comparisons, unavailable tasks and resource results |
| [Manuscript guide](TMLR.md) | Current integrated unsubmitted research draft |
| [Previous proxy metric study](extensions/target_geometry/README.md) | Shared diagonal metric, donor residuals and their unresolved incremental value |
| [Residual transport theory](extensions/target_geometry/THEORY.md) | Predictable residual signal, fitted-correction error and donor-to-final mismatch |
| [Earlier additive utility study](extensions/forecasting_utility/README.md) | Preserved mixed results, failed fits and explicitly amended successful-subset analysis |
| [KAHM classification repository](https://github.com/MohitKumarRostock/KAHM) | Separate classification implementation and examples |

## What these measurements do and do not establish

KAHM features are deterministic functions of the observed history. They can
make existing structure easier for a finite learner to use, but do not add
information to history already retained in full. Reconstruction accuracy,
a lower training objective and an available zero correction each answer a
different question from improved future prediction.

A distinctive predictive contribution requires a reserved-evidence advantage
over matched alternatives and a practical comparison, with its resource
scope stated. The implementation reports conservative sums of measured
computation dependencies, selected-state packing/serialization and model
bytes. It does not establish minimal production cost, per-method peak
memory, or a deployment acceptance threshold.

Bounded incremental geometry remains a further research direction. The
preserved online module verifies fixed-feature sliding-window ridge algebra
and cached KAHM reconstruction; it does not constitute a delayed-label
streaming experiment. An adaptation study must jointly measure forecast
accuracy, update latency, memory and refresh/fallback behavior on a new
chronological protocol. The present evaluation cannot become unseen
confirmation for a procedure redesigned after inspecting its outcomes.

## Preserved Koopman research

KAHM names the local geometric construction. KAHKM names the earlier Kernel
Affine Hull Koopman Machine architecture for soft-state dynamics. Their
targets and decoders differ, so findings for one do not automatically transfer
to the other. The [historical result map](reproduction/reported_results.tsv)
and [reproducibility evidence](reproduction/REPRODUCIBILITY_EVIDENCE.md)
identify the preserved implementations and output chains.
