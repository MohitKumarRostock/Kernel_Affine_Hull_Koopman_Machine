# Research roadmap: predictive geometry for time series

## Research identity

**KAHM is a local geometric representation and gating mechanism whose value
for forecasting must be established through temporal evaluation, matched
controls, and complete cost accounting.** The research target is to identify
conditions where its geometry improves learning from an available history.
That identity does not require a claim that KAHM is the best forecaster across
unrelated domains.

This roadmap organizes interpretation and possible follow-up work. It does
not change the [frozen protocol](PROTOCOL.md), its primary comparisons, or the
status of earlier evidence. The [publication record](PUBLICATION.json) links
the prospective source checkpoint. New hypotheses suggested by results need
their own development and reserved evaluation.

## Decision from the completed study

The next study should pursue a specific predictive mechanism or a measured
resource improvement. The present results do not justify a broad superiority
claim or an assumption that adding the learned metric and donor residuals
improves KAHM. A bounded research program remains justified by the conditional
gains over ridge and by the ability to test where geometric representations
help; a KAHM-specific advantage still needs evidence.

The [audited summary](reports/summary.json) contains all 48 tasks and 768
task–method prediction records, with zero method fallbacks and zero missing
tasks. The [complete primary contrast table](reports/primary_contrasts.csv)
reports the following outcomes:

| Question | Completed evidence | Research implication |
|---|---|---|
| Does learned-metric KAHM improve the strong ridge baseline? | Favorable intervals on all three synthetic laws; mean MSE reductions of 11.18%, 53.26%, and 19.44% on nonlinear AR, Van der Pol, and Duffing | The complete correction can exploit predictive structure missed by this ridge procedure |
| Does learning the metric improve KAHM? | Learned versus identity-metric OOF KAHM remains unresolved on all three laws | A benefit from the added metric-learning step has not been established |
| Do held-out donor residuals improve KAHM? | Identity-metric OOF versus in-sample KAHM remains unresolved on all three laws | Honest donor generation does not by itself establish better correction targets for the final fitted base |
| Is learned KAHM stronger than the equally trained geometric controls? | Five of nine comparisons favor the control; four remain unresolved; none favors KAHM | The current study does not establish a KAHM-specific predictive advantage |
| Does the archive evidence justify the added procedure? | Learned KAHM reduces source-balanced scaled MSE by 2.18% versus ridge, at 124.9× mean measured pipeline time; fixed in-sample KAHM reduces it by 2.76%, and fixed in-sample RBF-local by 3.27% | Accuracy gains must be assessed against simpler alternatives and their full costs |

The synthetic decisions use the declared family of 18 two-sided Bonferroni
paired Student intervals, with 12 independent repetitions per law and an
explicit normal approximation. Unresolved differences do not establish
equivalence. The five adverse control comparisons are locality and RBF-local
on Van der Pol, and affine, locality, and RBF-local on Duffing.

The archive comparisons are descriptive. Learned affine and locality have
lower aggregate losses than learned KAHM; fixed in-sample RBF-local has the
lowest aggregate loss among the 16 procedures. The observed learned-KAHM and
ridge pipeline means are 0.162 s and 0.00130 s, respectively. The ratio describes
the recorded implementation and timing scope on one host. It does not imply
the same ratio for every deployment, repeated timing experiment, or model size.
The [archive aggregates](reports/natural_aggregate.csv) and
[source-level results](reports/natural_sources.csv) retain every selected source.

These findings make **direct forecast/horizon alignment and bounded incremental
updates** the next research priorities. Their performance benefits remain
untested hypotheses. They should be studied separately so that an accuracy
change can be attributed to the predictive geometry and a resource change to
the update procedure.

## What the declared comparisons measure

| Question | Declared comparison | What the comparison establishes |
|---|---|---|
| Does the complete KAHM correction help a strong reference? | Learned-metric OOF KAHM versus raw ridge | Incremental usefulness relative to the chosen ridge procedure |
| Does predictive metric training help KAHM? | Learned-metric versus identity-metric OOF KAHM | Transfer of the shared kernel-regression proxy to the fitted KAHM decoder |
| Do donor targets help correction training? | Identity-metric OOF versus in-sample KAHM | The effect of the declared donor-residual procedure on the same correction rows |
| Is the result specific to KAHM geometry? | Learned-metric KAHM versus equally trained affine, locality, and RBF-local controls | Whether simpler geometric constructions explain or exceed the benefit |
| Is the method usable under the declared resources? | All declared losses, model coverage, fallbacks, costs, and retained model sizes | An implementation-specific accuracy/resource comparison |

The 18 primary synthetic contrasts form one declared uncertainty family.
Fresh episodes test three already studied numerical laws. The four selected
archive collections supply descriptive source-balanced evidence, including
every selected series. Neither many overlapping windows nor multiple methods
turn four collections into a large sample of independent application domains.

### The metric's exact role

The learned diagonal metric minimizes a training-only Gaussian
leave-group-out residual-regression proxy. It is then supplied identically to
all four geometric gates. The local banks are refitted under that metric and
their ridge corrections are selected on validation data.

This is **supervised metric transfer**. It does not optimize the KAHM forecast
end to end, and it does not establish that a lower proxy loss produces a lower
future loss. The metric-learning lineage is stated explicitly in the
[theory notes](THEORY.md) and manuscript. The shared proxy also provides a
useful experimental control: any claimed KAHM-specific benefit must survive
comparison with the other gates receiving the same metric.

## Interpretation rules for subsequent studies

| Observed pattern | Appropriate interpretation | Decision for subsequent work |
|---|---|---|
| The metric proxy improves but KAHM's held-out forecast does not | The proxy improvement did not transfer to this decoder and evaluation law | Diagnose objective/decoder mismatch before adding complexity |
| Several gates benefit similarly from the learned metric | Predictive metric learning helps the shared local construction | Report the general effect; do not assign it exclusively to KAHM |
| KAHM improves ridge but a simpler geometric control is stronger or cheaper | The tested KAHM procedure has not established a practical advantage over that control | Prefer the simpler procedure for that use case; investigate the remaining mechanism only if justified |
| KAHM improves the strong base and matched controls under useful resources | A conditional geometric advantage is supported within the evaluated scope | Replicate on a prospectively defined application and explain the mechanism |
| Gains are confined to a simulator while new collections remain unfavorable | A controlled effect has limited transfer under the tested procedure | Narrow the claim and deprioritize broad accuracy-driven forecasting extensions |
| Intervals leave the useful effect unresolved | The study does not settle the direction or practical size of the difference | Report uncertainty; do not interpret a nonsignificant result as equivalence |

These rules do not authorize selection of favorable methods or sources from
the completed experiment. A later application study should specify its minimum
useful improvement and resource budget before inspecting confirmation outcomes.
The present protocol does not impose a universal exchange rate between error,
training time, inference latency, and memory.

## Priorities for new research

### 1. Test geometry trained for the final forecast and horizon

**Untested hypothesis:** a metric trained with the final forecasting decoder,
possibly allowing different geometric weights at different horizons, transfers
better than the present shared residual-regression proxy. The current experiment
has not tested end-to-end KAHM training or horizon-specific geometry.

Begin with a small, explicit parameterization and a forecast loss whose horizon
weights are fixed before evaluation. Compare a shared metric against
horizon-specific metrics while keeping the input histories, decoder capacity,
search budget, and validation opportunities controlled. Then compare KAHM
against affine, locality, and RBF-local gates trained with the same predictive
objective. Direct supervision and additional parameters must be available to
the matched controls as well.

The target is a useful held-out improvement over both the strong base and the
best relevant control. A lower training proxy or reconstruction error is not
the outcome. Diagnose unstable local estimates and donor-to-final residual
differences alongside forecast errors. The [transport identity](THEORY.md)
explains the latter issue under one fixed law; it does not eliminate
distribution change between chronological folds.

### 2. Test bounded incremental updates under a real resource budget

**Untested hypothesis:** maintaining a bounded local reference bank and updating
only affected local models preserves useful predictive accuracy at a lower
update cost. The present study fits offline and does not demonstrate that such
an incremental method works.

Profile the current pipeline first, then declare a maximum reference-bank size,
update rule, and compute budget. Evaluate a causal predict-then-update procedure
in which target-dependent updates wait until the relevant horizon has been
observed. Compare against incremental ridge and equally budgeted local controls;
each method must receive the same available histories and delayed labels.

Measure cumulative forecast error, update cost, prediction latency, retained
memory, and fallbacks. Include the cost of unsuccessful attempted updates.
Report deployment prediction latency separately from tuning and initial fitting.
A faster implementation is useful evidence when accuracy and reliability remain
acceptable; it does not establish a geometric accuracy advantage.

### 3. Choose an application with a concrete dynamic mechanism

A subsequent application should explain why local geometry is relevant:
repeated operating states, nonlinear transitions, partial observation, or
predictable errors of a strong global forecaster. Similar observed positions
need not have similar futures. Conversely, reconstruction variation can be
irrelevant to the target.

An intervention on phase, direction, history length, nuisance variables, or
forecast horizon should be applied to every relevant control. Inputs derived
deterministically from existing lags change finite-model access to information;
additional sensors or external covariates introduce a different information
comparison and must also be available to the baselines.

### 4. Make additional complexity respond to predictive value

Adaptive local ranks, reference selection, or dictionary growth are reasonable
only after defining how proposed additions will be evaluated. Repeatedly
selecting on one validation set can overfit its finite evidence. Any sequential
selection claim requires its own search accounting and fresh or otherwise
justified evaluation.

Retaining an earlier feature block preserves feasible forecasts. It does not
guarantee that a fitted regularized model improves, and it does not prove that
every cheaper representation is inadequate. Keep empirical resource comparisons
separate from the stronger class-level claims addressed by risk certificates.

## Reserve new evidence for the next claim

The current source roster and simulation outcomes are now available for
development and diagnosis. Repeated tuning on them cannot supply fresh
confirmation. Before the next confirmation run, preregister or publicly freeze
the primary question, complete methods, matched controls, resource budgets,
source selection, exclusion and failure rules, and multiplicity treatment.

Use fresh application sources or an independently reserved application sample
chosen without inspecting its outcomes. New random episodes from the same
simulator supply additional stochastic evidence on that law; they do not test
transfer to a new domain. Keep all specified sources and methods in accounting,
including adverse results and failed fits. The completed public execution freeze
should continue to be described accurately as an execution freeze, rather than
external preregistration.

## How the research package should communicate its findings

- Lead with the scientific question and the actual result, including an
  unfavorable or unresolved answer.
- Keep the exact forecasting target, available history, baseline, decoder,
  source roster, and loss aggregation visible.
- Report every declared primary contrast and selected source; distinguish
  confirmatory families from descriptive diagnostics and follow-up analyses.
- Preserve earlier failures and negative results. A newly robust procedure
  does not retrospectively complete an earlier failed evaluation.
- Provide an executable small example for understanding the method and a
  separate frozen study for performance claims. The example's result is not
  a benchmark result.
- Link claims to saved predictions, independent audit outputs, configuration,
  source hashes, and the manuscript. Cite immutable repository versions and
  accurately retain the manuscript's unsubmitted status.

KAHM's time-series identity is strongest when it explains **which predictive
structure its geometry makes accessible, which simpler alternatives share that
benefit, and what a complete fitted procedure costs**. The next research decision
should follow that evidence.
