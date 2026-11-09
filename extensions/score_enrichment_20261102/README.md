# Reconstruction-affinity feature augmentation

Protocol `KAHM-SCORE-20261102-v1` evaluates KAHM, affine reconstruction, and kernel-weighted averaging for direct prediction of future observations. Each method produces its own reconstruction scores and normalized affinities. Log normalization constants and membership-weighted local coordinates provide two feature augmentations. Linear ridge and a 64-landmark Gaussian Nyström predictor use the same observed targets.

The evaluation has 64 repetitions under each of four data-generating distributions. A fits candidates, B selects each complete procedure's settings, C supplies diagnostics, and D evaluates fixed predictions. The 40-comparison Bonferroni family contains 16 lower candidate mean losses, seven higher losses, and 17 unresolved comparisons. Local coordinates improve KAHM memberships in six of eight distribution–predictor combinations. All four nonlinear-predictor comparisons between KAHM and kernel-weighted averaging are unresolved. These intervals do not establish equivalence, and their paired-Student approximation is not distribution-free.

| Record | Scope |
| --- | --- |
| [Primary comparisons](primary_contrasts.csv) | Exact 40-row summary, including unfavorable and unresolved outcomes |
| [Protocol](protocol.json) | Recorded experimental configuration |
| [Source identity](PROSPECTIVE_FREEZE.json) | Local source checkpoint; it does not establish public preregistration |
| [Results identity](RESULTS_RELEASE.json) | Recorded output identities; some referenced files are unavailable here |
| [Availability](AVAILABILITY.md) | Available statistical checks and missing execution inputs |

From the repository root, run `python extensions/score_enrichment_20261102/verify_primary_summary.py` with SciPy installed. It checks summary arithmetic and original byte identity. The [saved-loss bundle](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment) reconstructs the comparisons from individual repetition losses and documents its separate scope.

The primary procedures select their settings independently. In a secondary analysis, the affine reconstruction control with linear ridge selects a dimension limit shared across methods. That asymmetric selection rule defines a different comparison. The experiment does not fit autonomous feature dynamics, establish Koopman closure, or demonstrate universal KAHM superiority.
