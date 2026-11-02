# Local Position and Reconstruction Features for Forecasting

## A matched study of Kernel Affine Hull Machines

This repository contains the implementation and evidence for a matched study of geometric forecasting features. The paper asks whether local position and reconstruction departures improve a direct affine ridge forecast of the same future observations. It distinguishes predictive usefulness from raw feature-span expressivity and from autonomous state dynamics.

The manuscript source and build instructions are in [papers/local_position_forecasting/](papers/local_position_forecasting/). The [supplementary source](papers/local_position_forecasting/supplement.tex) specifies the complete protocol, all primary and descriptive comparisons, and the computational audit. The [reproducibility guide](REPRODUCIBILITY.md) gives the exact source identity and regeneration commands. The [research index](docs/RESEARCH_INDEX.md) locates related and earlier studies without mixing their inference populations.

## Main result

Four inputs share a local reference construction: G contains membership gates; P adds local principal coordinates; K adds Kernel Affine Hull Machine (KAHM) reconstruction departures; and R adds matched positive-kernel radial-basis-function (RBF) departures. Each input predicts the same observed response with direct affine ridge heads. The primary comparison uses the bank selected by P on validation data. It therefore evaluates enrichment conditional on that selection procedure, not a symmetric comparison of independently optimized forecasting systems.

| Primary comparison | Population | Reduction in mean loss | Prescribed criterion | Established |
| --- | --- | ---: | --- | --- |
| P versus G | Nonlinear | 69.201% | 3% superiority | Yes |
| K versus P | Nonlinear | 10.459% | 3% superiority | Yes |
| K versus R | Nonlinear | 0.029% | 3% superiority | No |
| K versus P | Linear | -1.111% | 5% noninferiority | Yes |

The analysis uses 32 independent repetition aggregates and one four-comparison Bonferroni family. All 256 tasks contribute according to the declared aggregation. Local position substantially improves the membership-only readout, and KAHM departures improve P on average in the nonlinear panel. The results do not establish a KAHM-specific advantage over matched RBF. Failure of the K/R margin is not an equivalence finding. Linear noninferiority permits the observed 1.111% loss increase; it is not superiority.

The reconstruction benefit is heterogeneous: K worsens P in nonlinear autoregression at both noise levels, and its aggregate improvement is concentrated at horizon 12. Independently selected caps slightly favor R over K in mean nonlinear loss. These findings are retained in the [complete study results](extensions/soft_state_ablation_20261102/RESULTS.md) and [compact analysis records](extensions/soft_state_ablation_20261102/analysis/).

## Scope of the mathematical results

The paper gives sufficient conditions for equality of KAHM and RBF augmented raw feature spans and an irregular scalar geometry in which the spans differ. These statements concern exact maps before block transformations and ridge fitting. They do not determine forecasting risk, establish Koopman closure, or prove Markov sufficiency. The evaluated features can be wider than the input history, so the experiment is not evidence of compression.

## Evidence and repository organization

| Location | Purpose |
| --- | --- |
| `papers/local_position_forecasting/` | Main article, focused supplement, editable source, compact evidence, and build/check scripts |
| `extensions/soft_state_ablation_20261102/` | Matched representation experiment: fixed protocol, implementation, analysis, and audit records |
| `docs/RESEARCH_INDEX.md` | Related studies, earlier methods, and retained archival material |
| `REPRODUCIBILITY.md` | Document builds, saved-loss checks, archive restoration, audit replay, and full simulation regeneration |
| `TIME_SERIES_RESEARCH.md` | Interpretation of the central and related evidence |

The central experiment depends on earlier runtime modules. Their original paths and frozen source records are preserved. Earlier paper editions and exploratory studies remain available as archival material; they are not additional independent evidence for the matched experiment. In particular, unfavorable policy and natural-data results are not removed or pooled with the central result.

Large fitted-state and episode archives are separately indexed. An archive index does not establish a public download location. Full source regeneration and replay from author-supplied arrays are different reproducibility routes, as documented in the guide.

## Licensing and citation

Original software is distributed under the [MIT License](LICENSE). Original research outputs are distributed under [CC BY 4.0](DATA_LICENSE), unless another license is stated. Third-party datasets, dependencies, and template files retain their own licenses. Cite the manuscript and the exact study source or evidence commit used. The public repository is an attributed research resource, not an anonymous review archive.
