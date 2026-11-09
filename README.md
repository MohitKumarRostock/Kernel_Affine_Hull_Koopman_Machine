# Reconstruction-based representations for forecasting

This repository studies how reconstruction features affect forecasting accuracy, information retention, and prediction-risk bounds. Kernel Affine Hull Machines (KAHMs) are compared with affine reconstruction, kernel-weighted averaging, and forecasting baselines under matched evaluation designs.

## Research guides

| Topic | Code and evidence |
| --- | --- |
| Feature Augmentation for Forecasting with Reconstruction-Based Representations | [Experiment guide and reproduction bundles](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) |
| A Hierarchy of Lower Bounds on Prediction Risk for Forecasting Representations | [Theory, implementation, and experiment guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) |
| Supporting forecasting experiments | [Research index](docs/RESEARCH_INDEX.md) |

The experiments use separate sampling units, selection procedures, losses, and comparison families. Results are interpreted within their own protocols.

## Reconstruction-affinity features

Normalized reconstruction affinities give soft assignments to local reference regions. Their normalization can discard absolute reconstruction-score information. Log normalization constants and membership-weighted local coordinates supply two ways to augment these features. The experiment tests whether these additions help a fitted predictor.

Protocol `KAHM-SCORE-20261102-v1` specifies 64 repetitions under each of four data-generating distributions, 256 process–repetition combinations, and 40 two-sided paired comparisons in one Bonferroni family. Each primary procedure selects its dimension and predictor settings separately, using the same validation observations within each task.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / kernel-weighted local features | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four distributions and two predictors. Directions follow adjusted intervals. The counts are not an omnibus success criterion. Coverage uses a paired-Student approximation; the protocol specifies no practical or equivalence margin. All four nonlinear-predictor KAHM comparisons with kernel-weighted averaging are unresolved. The [complete primary summary](extensions/score_enrichment_20261102/primary_contrasts.csv) includes favorable, unfavorable, and unresolved comparisons.

On Van der Pol with the nonlinear predictor, KAHM-local has 62.09% lower mean loss than raw input and 2.51% lower loss than affine-local, with resolved adjusted comparisons. Its approximately 0.038% difference from kernel-weighted local features is unresolved. On Duffing, the same KAHM procedure has higher loss than raw and affine-local. These results concern the specified procedures and distributions.

## Information and prediction

Memberships and log normalization constants recover the full reconstruction scores in exact arithmetic. Membership-weighted local coordinates recover the input under positive weights and a full-column-rank condition on the combined coordinate matrices. The inverse can be ill-conditioned. These identities do not order fitted forecasting risks or establish Koopman closure of nonlinear features. This experiment does not fit an autonomous transition model.

## Reproduction

This branch contains the [primary-summary arithmetic checker](extensions/score_enrichment_20261102/README.md) and code and evidence for the supporting studies. The [saved-loss bundle](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment) reconstructs all 40 central comparisons from repetition-level losses. The central generator, experiment runner, complete evaluation roster, fitted states, and forecast arrays are unavailable here, so full model regeneration is not supported for that experiment. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes the available checks and the separate requirements of other studies.

Software follows the [MIT License](LICENSE); research outputs follow [DATA_LICENSE](DATA_LICENSE). Third-party licenses and attribution remain applicable. Cite the relevant protocol and immutable source commit when using the code or results.
