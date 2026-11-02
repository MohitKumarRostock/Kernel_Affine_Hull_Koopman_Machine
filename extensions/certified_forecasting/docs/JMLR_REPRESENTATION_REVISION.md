# Representation-focused scientific revision

This revision starts from repository commit
`dae2a6bff20c74609e034d057fab85ba9beb362a` and its manuscript,
**A Hierarchy of Risk Certificates for Forecasting Representations**.
It makes the representation-design question more explicit while retaining
the general scope of the central theorem. This document records scientific
scope and evidence; it is not a prediction of any journal's editorial decision.

## Scientific question and contribution

The question is whether independent observations can establish that every
decoder in a declared sensitivity class misses a forecasting tolerance, and
whether lower and upper certificates can justify a more costly representation.
Every comparison keeps the observed future target and its evaluation law
fixed. KAHM is the principal learned representation used to study that question.
Its local reconstruction mechanism is inherited from the earlier KAHM work.

| Component | Role in this manuscript | Scope of the claim |
|---|---|---|
| KAHM reconstruction and normalized associations | Concrete learned feature maps | Inherited construction; no claim of a new KAHM reconstruction model |
| Conditional positive-part pair witness | An inexpensive obstruction to low class risk | Related pairwise Lipschitz arguments are established; this paper studies its sharpness, slack, and calibrated use |
| Centered scalar block hierarchy | Main analytical contribution | Exact pair member, average-over-subsets monotonicity, compact-support consistency, and explicit limitations of fixed block sizes |
| Independent bounded-mean calibration and simultaneous selection | Converts population witnesses into confidence statements | Uses established concentration and multiple-comparison principles |
| Retained feature blocks and group-norm sensitivity | A common geometry for representation refinement | Preserves earlier predictors and is invariant to exact block duplication; does not guarantee improved refitting |
| Lower/upper cost decisions | A representation-selection consequence | Minimum cost only among the declared classes, target, law, and budgets |
| Learned KAHM experiments | Tests the practical interpretation of those decisions | Controlled mechanisms and conditional evidence on frozen banks; not a universal KAHM accuracy claim |

The hierarchy does not depend on the KAHM reconstruction formula. Its
distinguishing mathematical content is the centered `b-1` normalization,
the exact size-two obstruction, monotonicity, convergence under the stated
conditions, and the rare-conflict and support-occupancy laws. Empirical-minimum
lower bounds and averaging independent optimization batches have a longer
history in stochastic programming. The manuscript's related-work section
identifies these antecedents rather than presenting general risk certification
as a new idea.

## Relation to the prior KAHM prediction-limit paper

The relevant predecessor is Kumar and Kargaran,
[Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions,
arXiv:2609.32652v3](https://arxiv.org/abs/2609.32652v3).
That version already studies fixed soft-coordinate representations,
norm-bounded linear prediction, class-centered and label-based lower
certificates, empirical-minimum alternatives, independent evaluation, and
fitted-risk upper bounds. It also links KAHM reconstruction-score margins to
agreement with reference labels. Those ideas should not be presented as new
merely because the present study uses additional KAHM dictionaries.

The present work instead keeps the future observed output common across
representations, analyzes pairwise slack through a block hierarchy, and asks
what decisions can be supported within retained-block representation families.
This distinction matters because forecasting each representation's own future
coordinates changes the target when the representation changes. Better
self-coordinate prediction alone does not show better prediction of a fixed
observed signal.

## Complete risk specification

A certificate applies to
`(observed target, evaluation law, frozen representation, feature metric,
sensitivity budget, decoder class)`. Report all six components. A positive
lower bound excludes the specified tolerance for that tuple; it is not an
unrestricted statement about information loss or the best possible forecasting
algorithm. Feature preprocessing belongs to the representation and must be
frozen before calibration or included in a valid simultaneous comparison.

The block-affine class is contained in the full Lipschitz class induced by
the maximum block distance. Consequently,
`optimal Lipschitz risk <= optimal block-affine risk <= fitted predictor risk`.
A lower bound on the first quantity also lower-bounds the second. The
hierarchy's consistency, however, targets the first quantity and does not
imply convergence to the block-affine optimum. The distinction remains even
with unlimited calibration data and exact optimization.

In the independent-future control, positive risk persists even with the full
history. This illustrates irreducible future variation, not evidence of KAHM
discarding an available predictive signal. In the repairable construction,
representation refinement can restore distinctions useful for the fixed output.
Both interpretations are conditional on the stated sensitivity budget.

## Evidence and historical records

The existing studies retain their original protocols, allocations, numerical
records, source snapshots, and adverse outcomes. Their roles remain distinct:

- `results/block_hierarchy_v1/` isolates population slack and finite-sample
  calibration on eight finite-support laws. Its 0.05 allocation applies within
  each law/repetition, not across the entire benchmark.
- `results/learned_kahm_decision_v1/` evaluates stratified pair decisions on
  forty learned KAHM banks. Its single 0.05 allocation covers that complete
  study. It does not establish an advantage for larger blocks on those banks.
- The forecasting, matched-geometry, input-scaling, PenDigits, and speech
  studies retain task-dependent and uninformative outcomes. Reused real-data
  cohorts remain descriptive where their original protocols say so.

The additional
[hierarchy evaluation on learned banks](../experiments/learned_kahm_hierarchy/)
connects the first two questions using new calibration observations and its
own frozen protocol. It reuses every one of the forty archived banks; its
inference is conditional on them and is not a fresh representation-learning
replication. It retains all four block sizes, both learned partitions,
equal-sample full empirical minimization at 256 observations, unresolved
decisions, numerical gaps, and computation diagnostics. Full empirical
minimization is uncomputed at larger sample budgets. All 1,760 endpoints share
one 0.05 failure allocation, separate from earlier studies.

The new block and stratified lower endpoints use KL inversion; the earlier
learned-decision study used empirical Bernstein lower bounds. Only comparisons
within the new study isolate block size under a common calibration rule.

The completed comparison shows a specific benefit and a specific limitation.
At 1,024 observations, blocks of size 8 and 16 exclude both classes in all
twenty irreducible cases, while the pair member and size-four blocks resolve
none. At 256 observations, however, full empirical minimization excludes both
classes in all twenty cases while every block method is unresolved. In the
repairable regime, every block size, including pairs, certifies minimum cost
six in all twenty cases at 1,024 and 4,096 observations. No repairable case
is certified at 256 observations because no fitted upper bound passes.

These outcomes support applying the general hierarchy to learned KAHM
geometry. They do not establish a universal computational or statistical
advantage, an automatic refinement algorithm, or natural-task prevalence.
The independent numerical audit passed. It reconstructs 163,840 observations,
80 selected forecasts, 143,440 optimization programs, 480 mass linear programs,
1,760 confidence endpoints, and 880 decisions. Its 3,626,324 floating-point
scalar checks find a maximum confidence-endpoint discrepancy of approximately
`1.003e-12`. All 379 solver stopping-limit cases remain in the analysis;
the recorded maximum normalized bracket gap is `6.409599374516499e-7`.
The extended-precision audit agrees to approximately `7.4e-18` on that gap.
These are numerical checks, not formal interval certificates.

## Manuscript organization and retained evidence

The main paper keeps the central theory and all its proofs, the learned
representation decisions, and a consolidated account of broader forecasting
behavior. `paper/supplement.tex` compiles the complete secondary experimental
exposition and inherited KAHM reconstruction calculations. The original
source sections and historical experiment archives remain available. This
organization sharpens the contribution without hiding unsuccessful outcomes
or changing the confidence allocation of a previous experiment.

The revised edition has a 186-word abstract, a 50-page main manuscript, and
a 23-page experimental supplement. The standard JMLR font and text dimensions
are unchanged. `scripts/build_manuscript.py` regenerates the supplement from
the retained detailed sources, compiles both PDFs, and checks for unresolved
references, duplicate labels, and overfull boxes.

## Execution provenance exception

The new hierarchy run saved all numerical records before its terminal hash
check detected a changed auxiliary development log. A separately running
dense benchmark had not stopped as initially believed. Its frozen log already
contained twenty rows across both development seeds, and eight more rows were
appended later. The protocol's earlier process-status description was therefore
incorrect; the original text and log are preserved alongside a correction.

An independent comparison confirms that every executable dependency, numerical
configuration, archived bank, and completed output remained unchanged. The
auxiliary log is only enumerated for source inventory; calibration does not
consume its contents. A separate finalizer reconstructed the terminal
inventory from saved files without new observations, endpoint reruns, or
selection. The retained summary records `source_unchanged=false` and
`executable_sources_unchanged=true`. The
[independent provenance audit](../verification/jmlr_representation_revision/independent_provenance_audit.json)
passed its checks of the frozen archive, both preserved logs, code references,
and original output hashes. Wall times may reflect contention from the development
process and support no isolated runtime-superiority claim.

The repository distributes the two largest gzip evidence files as parts of
at most 8 MiB in `results/learned_kahm_hierarchy_v1/archive_parts/`. Run
`python scripts/restore_hierarchy_payloads.py` from the extension root before
auditing the retained records. The helper verifies part hashes and the
reconstructed originals' SHA-256 hashes, preserving their exact frozen bytes.
This packaging step changes no study source, protocol, observation, or result.

## Verification principles

The revision passes all 65 tests in the standard `unittest` suite, seven
mathematical witness-comparison tests, and four additional sparse-solver
function tests: 76 checks in total. The latter check closed-form pairs,
collapsed and nearly collapsed geometries, agreement with an independent
dense program, the maximum feature-block metric, and valid early-stopped
lower values. The [reproduction guide](REPRODUCIBILITY.md#tests) gives all
three commands; the standard discovery command alone runs only 65 tests.
An integrity comparison with the starting manuscript confirms that all six
theorems, six propositions, three corollaries, and fourteen proof environments
are unchanged and remain in the main manuscript assembly.

Primal minimization values alone are unsuitable for a lower certificate.
Recorded dual variables permit an independent weak-duality calculation: the
dense solver uses nonnegative multipliers, and the new solver uses signed-edge
variables. Verification should rebuild distances and constraints from the
stored observations, then reconstruct score ranges, confidence endpoints,
allocations, and final decisions without importing the study's certificate
implementation. Long-double evaluation and feasibility tolerances improve
the numerical check but do not constitute formal interval arithmetic.

A source hash identifies the executed bytes. Recorded freezes establish an
execution sequence but are not independent timestamp attestations. Zero
observed coverage failures on a controlled family do not establish universal
coverage; the mathematical assumptions provide the guarantee. Independent
replay does not establish an unverified real-data sampling assumption.
