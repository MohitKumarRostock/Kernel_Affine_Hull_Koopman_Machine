# Predictive Enrichment of Geometry-Induced Soft-State Abstractions

## Information retention and decoder access

Geometric affinities describe relationships to reference regions, but their interpretation does not establish predictive usefulness. Reconstruction-score normalization can discard absolute affinity information, while local position may be absent from the scores themselves. Even an information-preserving enrichment can be difficult for a finite decoder to use. This research separates those questions using Kernel Affine Hull Machines (KAHMs), matched affine and positive-kernel reconstructions, and common observed-response targets.

## Current paper and evidence

| Resource | Location |
| --- | --- |
| Article | [Main PDF](papers/score_enrichment/main.pdf) |
| Supplementary material | [Supplement PDF](papers/score_enrichment/supplement.pdf) |
| Editable sources and static tables | [Paper directory](papers/score_enrichment/) |
| Primary results and protocol identities | [Study directory](extensions/score_enrichment_20261102/) |
| Build and verification commands | [Reproducibility guide](REPRODUCIBILITY.md) |
| Related studies and previous editions | [Historical evidence index](docs/RESEARCH_INDEX.md) |

This checkout contains the current publication materials. Previous paper editions, unrelated experiment outputs, exploratory code, and historical environments are available at the immutable archive identified in the research index rather than alongside the current article. Their removal from this checkout does not remove them from Git history.

## Complete primary findings

Protocol `KAHM-SCORE-20261102-v1` has 64 repetitions under each of four laws, 256 task suites, and 40 two-sided paired comparisons in one Bonferroni family. Primary procedures independently select native dimension and decoder settings on validation data.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / positive-kernel-local | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four laws and two decoders. Directions follow adjusted intervals, not point ordering alone. These counts are not an omnibus success criterion. Coverage relies on a paired-Student approximation; no practical or equivalence margin is assigned. All four nonlinear-decoder KAHM/positive-kernel comparisons are unresolved. The [complete primary table](extensions/score_enrichment_20261102/primary_contrasts.csv) retains all favorable, unfavorable, and unresolved comparisons. [Interpretation](TIME_SERIES_RESEARCH.md) explains the scientific scope.

Partition normalizers recover score information in exact arithmetic. Gated local coordinates preserve the input under positivity and a full-rank basis-stack condition, although recovery can be ill-conditioned. These properties neither order fitted risks nor establish autonomous Koopman closure. No autonomous transition matrix is fitted in this study.

## Reproducibility scope

The checkout supports document compilation and checking arithmetic from the published primary-summary statistics. It is not the complete experimental execution package. Frozen numerical source, repetition-level losses, and full audit records remain in the separately delivered author package; complete model/episode archives have no established public download location. The cleanup does not change that pre-existing distribution limit. See the [availability statement](extensions/score_enrichment_20261102/AVAILABILITY.md).

The article, supplement, all tables, and the published study directory are unchanged from the pre-cleanup snapshot. A read-only integrity workflow checks their file identities, document dependencies, local links, and summary arithmetic. Historical unfavorable results remain described in the supplement and linked through the archive index.

Original software retains the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). Third-party attribution and licenses are preserved. This public repository is attributed research material, not an anonymity-certified review-code archive.
