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

The manuscript files below are retained earlier editions; their availability statements are historical. The revised manuscript and LaTeX package are distributed separately. The guides and reproduction bundles below describe the materials currently available here.

| Reader's question | Resource |
| --- | --- |
| What is the motivation and mathematical contribution? | [Retained article edition](papers/score_enrichment/main.pdf), with [retained sources](papers/score_enrichment/) |
| How are observations, features, prediction rules, and comparisons defined? | [Retained supplement edition](papers/score_enrichment/supplement.pdf) and [recorded protocol](reproduction/score_enrichment/evidence/protocol.json) |
| What happens in every primary comparison? | [Complete primary table](reproduction/score_enrichment/evidence/primary_contrasts.csv) |
| How do I interpret and check a primary comparison? | [Worked examples and experiment guide](docs/EXPERIMENT_GUIDE.md) |
| Can I rebuild the comparisons from individual repetition losses? | [Central evidence and independent checker](reproduction/score_enrichment/) |
| Which other checks can I run from this branch? | [Reproduction guide](REPRODUCIBILITY.md) |
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

## Recompute the central comparisons

The [central evidence bundle](reproduction/score_enrichment/) contains 17 original records, including the complete selected-loss table as `selected_losses.csv.gz`. Its independent checker rebuilds 416 procedure summaries, 2,560 paired differences and all 40 primary comparisons from 26,624 saved loss rows.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The command prints a JSON report. Add `--output /tmp/kahm-score-analysis` to write the rebuilt tables to a new directory. The checker verifies record identities and reconstructs statistics from saved losses; the original model-generation source, exact task roster, fitted models and forecast arrays are still required to regenerate the central experiment.

## Explore the supplementary capacity study

The [decoder-capacity bundle](reproduction/capacity_followup/) contains the original experiment source and archived records for the separate 24-repetition, three-law study. It compares KAHM local-coordinate features with raw-history and affine-coordinate controls at 64 and 256 Nyström landmarks. It supports file verification, statistical analysis of saved records and a full computational rerun. Its MSE outcomes and 12-comparison family remain separate from the central study's SMSE outcomes and 40 comparisons.

The [reproduction directory](reproduction/README.md) explains the two bundles. The [reproduction guide](REPRODUCIBILITY.md) gives commands, dependencies and the meaning of each check.

## Documents and historical records

Compile the retained repository article and supplement with Python and a compatible TeX installation:

```bash
python3 papers/score_enrichment/scripts/build.py
```

The original paper and study directories retain their publication files and historical availability records. Use the reproduction bundles above for the currently accessible numerical evidence. The [historical availability record](extensions/score_enrichment_20261102/AVAILABILITY.md) identifies the missing original central source package and numerical archives; it predates the saved-loss bundle and does not describe that bundle's contents.

Original software uses the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). The source distribution retains author attribution and third-party notices.
