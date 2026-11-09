# Matched local-position and reconstruction experiment

This experiment evaluates four representations for direct prediction of the same future observations. G contains separately transformed locality memberships; P adds transformed gate-weighted local principal coordinates; K adds transformed KAHM reconstruction departures; R adds matched positive-kernel RBF departures. Shared blocks are identical when an enrichment is appended. Raw gates lie in a product of simplices, whereas transformed inputs and signed departures need not be simplex-valued.

The [protocol](PROTOCOL.md) and [configuration](config.json) define 32 independent repetitions, four simulation laws, two added-noise levels, history width 12, and horizons 1, 4, and 12. A fits geometry, transforms, and coefficients. B selects ridge penalties and dimension caps. C supplies diagnostics only. D supplies final evaluation. The primary view uses P's B-selected bank for all arms; independently selected banks are secondary.

## Findings

| Primary comparison | Population | Reduction in mean loss | Criterion established |
| --- | --- | ---: | --- |
| P versus G | Nonlinear | 69.201% | 3% superiority |
| K versus P | Nonlinear | 10.459% | 3% superiority |
| K versus R | Nonlinear | 0.029% | No |
| K versus P | Linear | -1.111% | 5% noninferiority |

The four paired Student bounds use a single Bonferroni family. The failed K/R margin limits a KAHM-specific interpretation. It does not establish equivalence. The linear criterion allows a limited loss increase. K worsens P on nonlinear autoregression, and the aggregate nonlinear improvement is concentrated at the longest tested horizon.

The [complete results](RESULTS.md), [primary table](analysis/primary_contrasts.csv), [panel summary](analysis/panel_summary.csv), and [analysis summary](analysis/analysis_summary.json) retain all declared comparisons. The [audit summary](verification/independent_audit_summary.json) identifies the coverage and limits of the retained numerical verification.

## Reproduction

The scientific source is commit `2b7d19b5b47f46b5b21b28d24960688f7969d96f`. The source freeze and publication receipt identify the implementation before production preparation. Earlier research informed the design; the record does not assert external preregistration or cryptographic target isolation.

The [reporting guide](reporting/README.md) gives commands for compact-record restoration and numerical checks. The protocol and source checkpoint specify full simulation regeneration. Full regeneration requires the inherited dimension-study runtime and policy-study numerical solver, not just this directory. It starts in a clean source-only worktree with the original freeze and genuine publication receipt. Existing completed evidence is not a generation workspace.

Three large records are stored losslessly compressed: B candidate scores, basis dimensions, and the complete audit receipt. The [compact manifest](reporting/compact_release_manifest.json) binds both compressed and original bytes. Prepared episodes, fitted states, forecasts, and execution records are held in three separately indexed numerical archives; their [inventory](reporting/numerical_archive_summary.json) does not assert a public download URL. Source is available to regenerate the study without those author-delivered arrays.

The audit reconstructs feature maps, refits B penalty paths using a different linear-algebra calculation, replays C/D forecasts, and recomputes aggregation and inference. It does not independently reimplement physical-law integration or partition clustering and is not an external scientific replication.

## Scientific scope

The experiment evaluates direct forecasts of observed outputs. It does not establish Markov sufficiency, Koopman invariance, autonomous state propagation, or general KAHM forecasting superiority. Related studies have separate sampling units and comparison families.
