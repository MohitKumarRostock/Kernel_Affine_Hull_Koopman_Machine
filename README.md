# Feature Augmentation for Forecasting with Reconstruction-Based Representations

## Why add features to normalized reconstruction weights?

For each observed history, local reconstruction models produce similarity weights for a set of reference regions. Normalizing these weights gives the representation called memberships in the stored results. Forecasting also requires distinctions that matter for future observations. Normalization can remove the overall reconstruction-affinity scale, while reconstruction scores can hide position within a region. This study asks what information should accompany memberships, and whether a specified predictive model can use it.

Kernel Affine Hull Machines (KAHMs) provide the main reconstruction method. Matched affine and kernel-weighted averaging reconstructions, together with raw histories, help identify which findings concern enrichment and which depend on the reconstruction method.

## What the study contributes

1. **Recovery of reconstruction scores.** Memberships together with log normalizers recover the full score vector in exact arithmetic.
2. **Recovery of input information.** Memberships together with weighted local coordinates can recover the input under explicit positivity and rank conditions. A conditioning bound describes the sensitivity of this recovery.
3. **Matched forecasting evidence.** The experiment compares complete procedures with validation-selected predictor settings; reconstruction-based procedures also select their maximum local dimension. The results show where enrichment improves forecasting and where reconstruction controls perform better or remain unresolved.

Information retention, numerical stability, and predictive performance are separate questions. The experiments fit direct forecasts; they do not fit an autonomous transition matrix for the representation.

## Inspect the documents and evidence

The guides below describe the experiment, its saved numerical results and the available verification commands. The central analysis starts from saved per-run losses; it does not regenerate the observations or fitted predictors.

| Reader's question | Resource |
| --- | --- |
| Where are the document sources and compiled files? | [Repository article PDF](papers/score_enrichment/main.pdf) and [document sources](papers/score_enrichment/) |
| How are observations, features, prediction rules, and comparisons defined? | [Repository supplement PDF](papers/score_enrichment/supplement.pdf) and [recorded protocol](reproduction/score_enrichment/evidence/protocol.json) |
| What happens in every primary comparison? | [Complete primary table](reproduction/score_enrichment/evidence/primary_contrasts.csv) |
| How do I interpret and check a primary comparison? | [Worked examples and experiment guide](docs/EXPERIMENT_GUIDE.md) |
| Can I rebuild the comparisons from individual repetition losses? | [Central evidence and independent checker](reproduction/score_enrichment/) |
| Which other checks can I run from this branch? | [Reproduction guide](REPRODUCIBILITY.md) |
| How do the other studies relate to this experiment? | [Scientific interpretation](TIME_SERIES_RESEARCH.md) and [research index](docs/RESEARCH_INDEX.md) |

## Complete primary findings

Protocol `KAHM-SCORE-20261102-v1` uses 64 independent repetitions for each of four data-generating distributions, giving 256 combinations of distribution and repetition. The primary family contains 40 two-sided paired comparisons with Bonferroni adjustment. Each procedure selects its own settings using the same validation observations within a task; the stored label `independent` denotes separate choices, not independent validation samples.

| Candidate / reference | Lower candidate mean loss | Higher candidate mean loss | Unresolved |
| --- | ---: | ---: | ---: |
| KAHM-local / KAHM-memberships | 6 | 0 | 2 |
| KAHM-normalizer / KAHM-memberships | 3 | 0 | 5 |
| KAHM-local / affine-local | 1 | 4 | 3 |
| KAHM-local / kernel-weighted local features | 3 | 0 | 5 |
| KAHM-local / raw history | 3 | 3 | 2 |

Each row covers four data-generating distributions and two predictors. “Local” includes memberships and weighted local coordinates; “normalizer” includes memberships and log normalizers. Directions follow adjusted intervals, with no practical-benefit or equivalence margin. Coverage depends on the adequacy of the paired-Student approximation. An unresolved interval does not establish equivalence. All four nonlinear-predictor KAHM/kernel-weighted averaging comparisons remain unresolved.

The findings support adding local-coordinate features in specified settings. They do not establish a universal advantage for KAHM over the reconstruction controls. The analysis with a shared dimension limit and the related studies use different comparison rules. Their results are reported separately from the primary comparisons.

## Recompute the central comparisons

The [central evidence bundle](reproduction/score_enrichment/) contains 17 original records, including the complete selected-loss table as `selected_losses.csv.gz`. Its independent checker rebuilds 416 procedure summaries, 2,560 paired differences and all 40 primary comparisons from 26,624 saved loss rows.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The command prints a JSON report. Add `--output /tmp/kahm-score-analysis` to write the rebuilt tables to a new directory. The checker verifies record identities and reconstructs statistics from saved losses; the original model-generation source, exact task roster, fitted models and forecast arrays are still required to regenerate the central experiment.

## Explore the supplementary capacity study

The [predictor-capacity bundle](reproduction/capacity_followup/) contains the original experiment source and archived records for a separate study with 24 repetitions under three data-generating distributions. It compares KAHM local-coordinate features with raw-history and affine-coordinate controls at 64 and 256 Nyström landmarks. It supports file verification, statistical analysis of saved records and a full computational rerun. Its MSE outcomes and 12-comparison family remain separate from the central study's SMSE outcomes and 40 comparisons.

The [reproduction directory](reproduction/README.md) explains the two bundles. The [reproduction guide](REPRODUCIBILITY.md) gives commands, dependencies and the meaning of each check.

## Document build and evidence availability

Compile the repository article and supplement with Python and a compatible TeX installation:

```bash
python3 papers/score_enrichment/scripts/build.py
```

The [availability guide](extensions/score_enrichment_20261102/AVAILABILITY.md) lists the supplied evidence and the missing execution materials. The saved-loss checker verifies statistical calculations conditional on the recorded losses; a document build verifies typesetting.

Original software uses the [MIT License](LICENSE); research outputs retain the terms in [DATA_LICENSE](DATA_LICENSE). The source distribution retains author attribution and third-party notices.
