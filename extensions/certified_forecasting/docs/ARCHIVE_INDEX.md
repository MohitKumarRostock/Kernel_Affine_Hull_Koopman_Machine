# Experimental provenance and verification archive

The scientific descriptions of the method and experiments are in the
[study guide](STUDY_GUIDE.md), [implementation guide](../IMPLEMENTATION_NOTES.md),
[reproduction guide](REPRODUCIBILITY.md), and [manuscript](../paper/).
The records indexed here establish which source versions, protocols, and
numerical checks support particular executions.

An archived record applies to the source version, data, and outputs identified
by its hashes or commit identifiers. Its page counts, test counts, and
byte-equality statements describe that recorded version. They should not be
interpreted as measurements of every subsequent manuscript or software
version. Historical review accounts preserve their contemporaneous language
so that the development and verification history remains traceable.

## Experimental protocols and source records

| Record | Scope |
|---|---|
| [Centered block protocol](../experiments/block_hierarchy/protocol.md) and [frozen evidence](../results/block_hierarchy_v1/) | All eight laws, 800 independent streams, 14,400 endpoints, raw categorical observations, primal/dual evidence, analytic hierarchy formulas, and predraw source snapshot. |
| [Learned KAHM protocol](../experiments/learned_kahm_decision/protocol.json) and [frozen evidence](../results/learned_kahm_decision_v1/) | Forty learned banks, 240 candidate fits, eighty selected heads, all trial outcomes, and one joint confidence allocation over regimes, trials, sample sizes, and bounds. Safe JSON preserves the complete numerical banks for replay. |
| [Witness protocol](../experiments/witness_comparison/protocol.md) and [source record](../results/witness_comparison/SOURCE_AND_AUDIT.md) | Predraw design, source hashes, all simulated trials, and the provenance of derived analyses. The experiment README is included in the frozen source manifest. |
| [Decision protocol](../experiments/decision_value/protocol.json) and [results](../results/decision_value/README.md) | The controlled episode law, confidence allocation, complete outcomes, and source/model freezes. |
| [Speech protocol](../experiments/speech_forecasting/protocol.md) and [evaluation provenance](../experiments/speech_forecasting/README.md#evaluation-order-and-source-provenance) | Group allocation, observable, candidate design, and the original execution and strict-order replay. |
| [Scaling protocols](../results/input_scaling_study/protocols/) and [study records](../results/input_scaling_study/README.md) | Original-seed and confirmation designs, source snapshots, complete records, and score-archive repair provenance. |
| [Matched-geometry frozen protocol](../results/matched_geometry_study/protocol.md) and [design explanation](MATCHED_GEOMETRY_PROTOCOL.md) | Full task and candidate grid, source identities, confidence allocation, and evaluation-cohort limitations. |

Protocols, raw observations, numerical results, manifests, source snapshots,
and verification logs are preserved as execution records. The source hashes in
a manifest continue to identify the files used for that execution. Readers
reproducing an archived execution should restore the corresponding source
version or snapshot as specified in the reproduction guide.

## Current hierarchy and learned-representation verification

The current manuscript is **A Hierarchy of Risk Certificates for Forecasting
Representations**. The [block-hierarchy verification](../experiments/block_hierarchy/VERIFICATION.md)
reconstructs all retained observations, 45,579 empirical primal/dual programs,
14,400 endpoints, and the analytic hierarchy table without importing the
runner or its optimizers. The completed report is
`results/block_hierarchy_v1/independent_verification.json`. Its confidence
scope is eighteen endpoints within each law/repetition, not all 800 streams;
zero observed violations are a finite numerical check.

The learned-KAHM study has both a [portable arithmetic replay](../results/learned_kahm_decision_v1/verification.json)
and a [no-project-import KAHM algebra audit](../verification/acceptance_revision/learned_decision_independent_audit.json).
They verify learned geometry, training-only scalers, selected models, score
reconstruction, all confidence calculations, and every reported decision.
The study's single 0.05 allocation covers both regimes and all trials and
sample sizes. Failed or unresolved decisions and strict optimizer-tolerance
failures remain in the frozen evidence.

The learned study's [pre-calibration repair record](../results/learned_kahm_decision_v1/precalibration_freeze_issue/repair.json)
preserves a zero-byte source archive detected before calibration, initial source
manifests/freezes, and explicit superseding freezes with unchanged models and
design. The [post-run maintenance record](../results/learned_kahm_decision_v1/post_run_maintenance.json)
identifies later packaging-only changes; AST checks show that every data,
fitting, selection, and calibration function is unchanged. These records
supplement the exact execution snapshot and do not replace historical evidence.

## Scaling and portable reproduction checks

The [speech scaling study](../experiments/speech_scaling/) has its own protocol,
development freeze, complete candidate records, compressed group losses,
full-model replay, portable audit, and exact execution-source snapshot in
`results/speech_scaling_v1/`. Its comparisons are descriptive on the reused
speech cohort. The [independent audit](../verification/scaling_complete_review/)
also reconstructs the table without importing the study's calculation code.
That directory records the corrected aggregation scope of the PenDigits
scaling claim, a deterministic KAHM scaling-mechanism check, retained-endpoint
and table reconstructions, and checks of this manuscript edition.

## Version-specific scientific review records

| Record | Scope |
|---|---|
| [Scientific verification](SCIENTIFIC_VERIFICATION.md) | Mathematical, implementation, empirical-arithmetic, and literature checks for the identified versions. |
| [Theory review](THEORY_AUDIT.md) | Proof and notation checks, assumptions, and limitations recorded during theoretical review. |
| [Literature review](LITERATURE_AUDIT.md) | Source retrieval and attribution assessment at the recorded review date. |
| [Manuscript scope review](MANUSCRIPT_SCOPE_AUDIT.md) | Interpretation of the initial forecasting, input-scaling, and matched-geometry studies. |
| [Scientific extension record](SUBSTANTIVE_REVISION.md) | Changes relative to the identified predecessor commit, together with checks and numerical outcomes. |
| [Earlier study overview](MANUSCRIPT_REVISION.md) | A version-specific scientific overview retained for provenance; the current reader-facing overview is the study guide. |
| [Readability verification](../verification/readability_scientific_revision/) | Test, arithmetic, source-identity, and layout checks for the associated edition. |
| [Length and format verification](../verification/length_format_revision/) | Layout and asset checks for the associated edition. |
| [Scientific-extension verification](../verification/substantive_revision/) | Test logs, endpoint reconstructions, empirical summaries, and PDF checks for the associated edition. |

## Interpretation of verification

A source hash identifies bytes; it does not establish mathematical correctness.
An independently reconstructed endpoint checks the numerical calculation
conditional on the stored losses or moments. A deterministic replay additionally
checks a specified data-generation and fitting execution. Monte Carlo coverage
frequencies describe the finite set of simulated conditions and have simulation
uncertainty. Each form of evidence has its stated scope, and none establishes
unverified population sampling assumptions.
