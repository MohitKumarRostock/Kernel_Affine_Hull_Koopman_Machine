# Research roadmap after the direct-reconstruction experiment

The next scientific priority is to identify **what predictive information the KAHM reconstruction adds beyond matched affine, locality and RBF-local representations, after accounting for strong raw-input baselines and cost**. The completed study demonstrates a substantial training effect within KAHM on Van der Pol, but no positive primary interval against an equally shape-trained non-KAHM local control. Further complexity needs a specific, testable reason to improve prediction. [Findings](FINDINGS.md) · [Primary results](results/primary_contrasts.csv)

## Status of the proposed directions

| Direction | Current status | What remains open |
|---|---|---|
| Train geometry for the final forecast and horizon | **Implemented and tested**, for bounded KAHM bandwidths and temperatures with fixed references, covariance, principal directions and original regularization. The 18-comparison family gives two positive, six negative and ten unresolved intervals. | The benefit is conditional. Reference locations, partition memberships, covariance, principal directions and decoder family were held fixed. A distinctive advantage over matched alternatives is unestablished. |
| Reduce useful geometry's cost through bounded incremental updates | **Not executed as a forecasting experiment.** The current study measures a static fitting and prediction pipeline. | Accuracy under changing data, label delays, update latency, incremental memory, reference refreshes and fallback behavior require a new protocol. Fixed-feature ridge identities do not demonstrate incremental KAHM geometry. |
| Establish broad natural-source transfer | **Not established.** Three planned rideshare tasks lack B windows; the full four-collection aggregate is undefined. Seven available series provide descriptive evidence from three collections. | A new evaluation must state its feasibility and missing-data rules before source selection and reserve untouched evaluation evidence. |

These statuses concern the particular procedures and evidence in this repository. They neither prove that all KAHM forecasters fail nor promise that another intervention will succeed. [Frozen scope and interpretation rule](PROTOCOL.md)

## First: test a predictive mechanism

A focused next question is whether a prespecified KAHM reconstruction feature contains forecast-relevant residual information that the matched controls cannot exploit with the same decoder and search opportunity. The present decoder receives affinities and gated principal coordinates. A possible development hypothesis is that directional reconstruction information, rather than only its folding-score gate, matters for a particular horizon or dynamical regime. That is an untested example, not an established explanation or a selected next method.

Before a new confirmation run:

1. Specify one mechanism and its predicted failure cases. Use the now-inspected studies for development and diagnosis only.
2. Change one identified component. Match feature width or declared capacity, decoder family, supervision, selection information, numerical safeguards, and optimization opportunity across the local controls.
3. Retain raw RBF, boosting and an explicitly selected non-KAHM policy. Improvement over ridge or KAHM's own fixed map alone is insufficient.
4. Decide what would count as a useful effect and which resource limits matter for the intended application. Do not choose that threshold after inspecting evaluation losses.

This makes a negative result informative: it can reject the proposed mechanism or identify a decoder or regime limitation without being recast as a positive KAHM contribution.

## Second: evaluate bounded updates if useful geometry warrants them

A subsequent streaming study should compare bounded KAHM updates, full geometry refreshes, and equally optimized simpler procedures under one chronological protocol. It should specify reference limits, eviction or replacement rules, refresh triggers, whether covariance and principal directions change, and which updates are exact or approximate. A bandwidth change alters both reference and query kernels; retaining an old factorization without a stated approximation defines a different procedure.

At every forecast origin, only already available observations and matured labels may influence updates. Predictions should be recorded before future labels are revealed. Measure forecast error together with per-update and prediction latency, peak incremental memory, refresh frequency, and numerical fallback events. Include the work required to select, maintain and occasionally rebuild the procedure. A low-cost fixed-feature update is not evidence that useful KAHM geometry can be maintained at that cost.

## Reserve evidence for either next study

Publish the numerical code, analysis, candidate family, resource rules, roster and input provenance before evaluation. Declare which training or development-period availability information may determine eligibility; specify missing-history, missing-target and delayed-label handling. Preserve unavailable tasks and failed fits instead of replacing them after outcomes are seen.

Use new independent seeds for new known-law claims and genuinely reserved collections or chronological evaluation periods for transfer claims. Existing D outcomes are now inspected and cannot supply another unseen confirmation. Keep simulation inference at the independent repetition level; related origins and series do not create additional independent source collections. Preserve predictions, selected states, optimizer or update logs, costs, and a complete audit trail. Any later change belongs to a documented amendment or a separately frozen study.

The next result should answer a narrower question with stronger attribution, rather than increase the number of components without establishing what they contribute.
