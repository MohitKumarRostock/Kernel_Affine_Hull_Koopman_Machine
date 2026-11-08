# Predictive Enrichment of Geometry-Induced Soft-State Abstractions

## Why enrich geometric memberships?

Geometric memberships describe how an observed history relates to reference regions. Forecasting also requires distinctions that matter for future observations. Normalization can remove the overall reconstruction-affinity scale, while reconstruction scores can hide position within a region. This study asks what information should accompany memberships, and whether a specified predictive model can use it.

Kernel Affine Hull Machines (KAHMs) provide the main reconstruction method. Matched affine and positive-kernel reconstructions, together with raw histories, help identify which findings concern enrichment and which depend on the reconstruction method.

## What the study contributes

1. **Recovery of reconstruction scores.** Memberships together with log normalizers recover the full score vector in exact arithmetic.
2. **Recovery of input information.** Memberships together with weighted local coordinates can recover the input under explicit positivity and rank conditions. A conditioning bound describes the sensitivity of this recovery.
3. **Matched forecasting evidence.** The experiment compares complete procedures with validation-selected predictor settings; geometric procedures also select their dimension cap. The results show where enrichment improves forecasting and where reconstruction controls perform better or remain unresolved.

Information retention, numerical stability, and predictive performance are separate questions. The experiments fit direct forecasts; they do not fit an autonomous transition matrix for the abstract states.

## Read the paper and inspect the evidence

| Reader's question | Resource |
| --- | --- |
| What is the motivation and mathematical contribution? | [Article](papers/score_enrichment/main.pdf), with [editable sources](papers/score_enrichment/) |
| How are observations, features, prediction rules, and comparisons defined? | [Supplement](papers/score_enrichment/supplement.pdf) and [recorded protocol](extensions/score_enrichment_20261102/protocol.json) |
| What happens in every primary comparison? | [Complete primary table](extensions/score_enrichment_20261102/primary_contrasts.csv) |
| Which checks can I run from this branch? | [Reproduction guide](REPRODUCIBILITY.md) |
| How do the other studies relate to this experiment? | [Scientific interpretation](TIME_SERIES_RESEARCH.md) and [research index](docs/RESEARCH_INDEX.md) |

## Complete primary findings

Protocol `KAHM-SCORE-20261102-v1` uses 64 independent repetitions for each of four laws, giving 256 law–repetition tasks. The primary family contains 40 two-sided paired comparisons with Bonferroni adjustment. Each procedure selects its own settings using the same validation observations within a task; the stored label `independent` denotes separate choices, not independent validation samples.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / positive-kernel-local | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four laws and two predictors. “Local” includes memberships and weighted local coordinates; “normalizer” includes memberships and log normalizers. Directions follow adjusted intervals, with no practical-benefit or equivalence margin. Coverage depends on the adequacy of the paired-Student approximation. An unresolved interval does not establish equivalence. All four nonlinear-predictor KAHM/positive-kernel comparisons remain unresolved.

The findings support local-position enrichment in specified settings. They do not establish a universal advantage for KAHM over the reconstruction controls. Common-cap summaries and related studies retain their separate interpretation and are not pooled with the primary family.

## What can be reproduced here?

From the repository root, with SciPy installed, check the published primary-summary arithmetic:

```bash
python3 extensions/score_enrichment_20261102/verify_primary_summary.py
```

This verifies the 40 comparison identities and recomputes standard errors, intervals, effect ratios, and decisions from the reported means and standard deviations. It does not reconstruct repetition-level losses or model predictions.

Document compilation requires Python and a compatible TeX installation:

```bash
python3 papers/score_enrichment/scripts/build.py
```

The publication branch contains the manuscript sources, static tables, protocol, primary-summary table, and summary checker. The complete simulation and encoding source, exact task roster, environment fingerprint, repetition-level execution release, and model/episode archives are not present. No public download endpoint for the complete archives is established here. The [reproduction guide](REPRODUCIBILITY.md) identifies the inputs required for each further task.

Original software uses the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). The source distribution retains author attribution and third-party notices.
