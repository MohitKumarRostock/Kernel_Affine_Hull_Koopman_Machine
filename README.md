# Predictive Enrichment of Geometry-Induced Soft-State Abstractions

## Information retention and decoder access

Geometric affinities describe relationships to reference regions, but their interpretation does not establish predictive usefulness. Reconstruction-score normalization can discard absolute affinity information, while local position may be absent from the scores themselves. Even an information-preserving enrichment can be difficult for a finite decoder to use. This research separates those questions using Kernel Affine Hull Machines (KAHMs), matched affine and positive-kernel reconstructions, and common observed-response targets.

The [manuscript directory](papers/score_enrichment/) contains the approved article, supplementary sources, complete static tables, bibliography, and build instructions. The [study guide](extensions/score_enrichment_20261102/README.md) identifies the numerical evidence and its distribution scope. The [research index](docs/RESEARCH_INDEX.md) preserves earlier studies without pooling their sampling units or comparison families.

## Complete primary findings

Protocol `KAHM-SCORE-20261102-v1` has 64 repetitions under each of four laws, 256 task suites, and 40 two-sided paired comparisons in one Bonferroni family. Primary procedures independently select their native dimension and decoder settings on validation data.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / positive-kernel-local | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four laws and two decoders. Directions follow adjusted intervals, not point ordering alone. The counts are not an omnibus success criterion. Coverage relies on a paired-Student approximation; no practical or equivalence margin is assigned. All four nonlinear-decoder KAHM/positive-kernel comparisons are unresolved. The [complete primary summary](extensions/score_enrichment_20261102/primary_contrasts.csv) retains favorable, unfavorable, and unresolved comparisons.

On Van der Pol with the nonlinear decoder, KAHM-local has 62.09% lower mean loss than raw input and 2.51% lower loss than affine-local, with resolved adjusted comparisons. Its approximately 0.038% difference from positive-kernel-local is unresolved. On Duffing, the same KAHM procedure has higher loss than raw and affine-local. These are conditional results for the specified procedures, not general KAHM superiority.

## Mathematical scope

Partition normalizers recover the full score vector from memberships in exact arithmetic. Gated local coordinates preserve the input under positivity and a full-rank basis-stack condition, although the inverse may be ill-conditioned. Matched raw reconstruction spaces can coincide or differ. These statements neither order fitted risks nor establish Koopman closure of nonlinear score coordinates. No autonomous transition matrix is fitted in this study.

## Publication and reproduction scope

This branch publishes the document sources and static tables, the exact 40-row primary summary, and the original protocol and source/release identities. It is not a complete mirror of the separately delivered scientific execution package. That package contains the frozen implementation, repetition-level loss tables, audit receipts, and stage records. The supplied large numerical-archive inventory does not establish a public download service.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) distinguishes document builds, summary-arithmetic checks, saved-loss analysis, and full model replay. The publication manifest identifies the document inputs checked against the approved author package. Earlier frozen experiments and manuscript editions remain unchanged. The public repository is attributed research material, not an anonymous review-code archive.

Original software retains the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). Third-party licenses and attribution are preserved. Cite the manuscript version, study protocol, and immutable repository commit used.
