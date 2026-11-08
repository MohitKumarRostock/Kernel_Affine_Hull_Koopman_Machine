# Explore the predictive-enrichment experiments

Kernel Affine Hull Machines (KAHMs) construct local reconstructions from reference histories. Geometric memberships express the relative reconstruction quality of these local models: each affinity is divided by the total affinity. Log normalizers retain the total affinity removed by that division. Membership-weighted local coordinates describe the input's position along local model directions.

The experiment asks whether retaining this normalization-scale or local-position information improves forecasting. It distinguishes information discarded by an encoding from information that a chosen predictor can use. Each reported comparison evaluates two complete procedures, including their training and validation choices. This guide starts with one comparison, then checks the full primary family using the records available in this branch.

Run the commands below from the repository root. The explanation script uses only the Python standard library. It reads the original [primary comparison table](../extensions/score_enrichment_20261102/primary_contrasts.csv), checks its recorded SHA256, and prints an explanation without changing any file.

## Start with a worked example

```bash
python3 scripts/explain_primary_comparison.py --examples
```

The three examples cover lower loss after local-position enrichment, higher loss against the raw-history control, and an unresolved comparison despite a lower candidate point estimate. They illustrate the different conclusions that the same decision rule can produce.

List all available comparisons or choose one by its recorded identifier:

```bash
python3 scripts/explain_primary_comparison.py --list
python3 scripts/explain_primary_comparison.py linear_ar__linear__local_vs_mem
```

The selected example compares KAHM memberships plus weighted local coordinates with KAHM memberships alone, using the linear predictor under the linear autoregressive law. The stored mean losses are approximately 0.93080 and 0.99825. Their candidate-minus-reference difference is −0.06745, with adjusted interval [−0.07262, −0.06227]. Because the entire interval is below zero, this comparison supports lower candidate mean loss under the experiment's inferential assumptions. Add `--json` for machine-readable output.

### What the numbers mean

**SMSE** is standardized mean squared error: for each of the three forecast horizons, divide the evaluation MSE by that horizon's retained-training target variance, floored at `1e-12`, then average equally across horizons. Each procedure's reported mean averages this quantity over 64 repetitions.

The interval is computed from 64 paired repetition differences. Windows within an episode and the three horizons are not additional independent repetitions. Candidate-minus-reference differences below zero favor the candidate; differences above zero favor the reference. An interval containing zero leaves the direction unresolved and does not establish equivalence.

The relative-gain percentage is `100 * (1 - candidate_mean / reference_mean)`. It is a ratio of mean losses, not an average of repetitionwise percentages or a confidence endpoint. The protocol specifies no minimum practical-benefit margin.

## Match a question to a comparison

| Research question | Candidate and reference in the saved table |
| --- | --- |
| Does local-position information help beyond memberships? | `K__local` versus `K__mem` |
| Does the normalization scale help beyond memberships? | `K__norm` versus `K__mem` |
| Does KAHM outperform an affine reconstruction control? | `K__local` versus `A__local` |
| Does KAHM outperform a positive-kernel reconstruction control? | `K__local` versus `R__local` |
| Does the selected geometric encoding improve on the observed history? | `K__local` versus `raw__raw` |

The full identifiers append the predictor, `linear` or `nonlinear`. `K`, `A`, and `R` denote KAHM, affine, and positive-kernel reconstructions. `mem` means memberships alone; `norm` adds log normalizers; `local` adds membership-weighted local coordinates. The nonlinear predictor appends Gaussian Nyström features before ridge regression; it is a specified predictor family, not an unrestricted nonlinear decoder.

Each procedure makes its own validation choice using the same validation observations within a task. The CSV label `independent` refers to these separate choices. Geometric procedures select a dimension cap as well as predictor settings; raw histories select predictor settings only. There is no refit on the combined training and validation data. Consequently, a comparison does not isolate adding coordinates while holding every fitted choice fixed. See the [recorded protocol](../extensions/score_enrichment_20261102/protocol.json) for the complete design.

## Check all 40 primary comparisons

With SciPy installed, run the existing summary checker:

```bash
python3 extensions/score_enrichment_20261102/verify_primary_summary.py
```

| Command | Input | Output and interpretation |
| --- | --- | --- |
| `scripts/explain_primary_comparison.py` | Original `primary_contrasts.csv` | A readable explanation or JSON for selected rows; no model computation |
| `extensions/score_enrichment_20261102/verify_primary_summary.py` | The same CSV and SciPy's Student quantile | JSON confirming file identity, all 40 comparison identities, standard errors, intervals, relative gains, and decisions |

A successful full check reports `status: passed`, `rows_checked: 40`, and counts of 16 lower-loss, 7 higher-loss, and 17 unresolved comparisons. These comprise five questions, four laws, and two predictor families. The two-sided intervals use Bonferroni adjustment over that entire family. Their coverage depends on the paired-Student approximation; arithmetic agreement does not make the intervals distribution-free.

## Continue to prediction reproduction

The commands above use released summary statistics. Recalculating those statistics from repetition losses requires the selected-loss records and analysis code. Recomputing forecasts additionally requires the original implementation, prepared observations, and fitted models. Those materials, the exact task roster, the complete environment record, and the numerical archives are absent from this branch. No public download endpoint for the complete experiment is established here.

The [reproduction guide](../REPRODUCIBILITY.md) separates supported checks from these additional requirements, and the [availability record](../extensions/score_enrichment_20261102/AVAILABILITY.md) preserves the documented distribution status. The present results concern direct forecasts in the specified simulated settings; they do not test autonomous evolution of the abstract states or a universal advantage for KAHM.
