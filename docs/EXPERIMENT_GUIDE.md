# Experiments on feature augmentation for forecasting

Kernel Affine Hull Machines (KAHMs) construct local reconstructions from reference histories. Membership features express the relative reconstruction quality of these local models: each affinity is divided by the total affinity. These weights are not calibrated probabilities of future outcomes. Log normalization constants retain the total affinity removed by that division. Membership-weighted local coordinates describe the input's position along local model directions.

The experiment asks whether retaining this normalization scale or local-coordinate information improves forecasting. It distinguishes information discarded by an encoding from information that a chosen predictor can use. Each reported comparison evaluates two complete procedures, including their training and validation choices. This guide starts with one comparison, then checks the full primary family using the records available in this branch.

Run the commands below from the repository root. The explanation script uses only the Python standard library. It reads the [primary comparison table](../extensions/score_enrichment_20261102/primary_contrasts.csv), checks its recorded SHA256, and prints an explanation without changing any file.

## Start with a worked example

```bash
python3 scripts/explain_primary_comparison.py --examples
```

The three examples cover lower loss after local-coordinate augmentation, higher loss against the raw-history control, and an unresolved comparison despite a lower candidate point estimate. They illustrate the different conclusions that the same decision rule can produce.

List all available comparisons or choose one by its recorded identifier:

```bash
python3 scripts/explain_primary_comparison.py --list
python3 scripts/explain_primary_comparison.py linear_ar__linear__local_vs_mem
```

The selected example compares KAHM memberships plus weighted local coordinates with KAHM memberships alone, using the linear predictor under the linear autoregressive process. The stored mean losses are approximately 0.93080 and 0.99825. Their candidate-minus-reference difference is −0.06745, with adjusted interval [−0.07262, −0.06227]. Because the entire interval is below zero, this comparison supports lower candidate mean loss under the experiment's inferential assumptions. Add `--json` for machine-readable output.

### What the numbers mean

**SMSE** is standardized mean squared error: for each of the three forecast horizons, divide the evaluation MSE by that horizon's training-set target variance, floored at `1e-12`, then average equally across horizons. Each procedure's reported mean averages this quantity over 64 repetitions.

The interval is computed from 64 paired repetition differences. Windows within an episode and the three horizons are not additional independent repetitions. Candidate-minus-reference differences below zero favor the candidate; differences above zero favor the reference. An interval containing zero leaves the direction unresolved and does not establish equivalence.

The relative-gain percentage is `100 * (1 - candidate_mean / reference_mean)`. It is a ratio of mean losses, not an average of repetitionwise percentages or a confidence endpoint. The protocol specifies no minimum practical-benefit margin.

## Match a question to a comparison

| Research question | Candidate and reference in the saved table |
| --- | --- |
| Does local-coordinate information help beyond memberships? | `K__local` versus `K__mem` |
| Does the normalization scale help beyond memberships? | `K__norm` versus `K__mem` |
| Does KAHM outperform an affine reconstruction control? | `K__local` versus `A__local` |
| Does KAHM outperform a kernel-weighted averaging reconstruction control? | `K__local` versus `R__local` |
| Does the selected reconstruction-based representation improve on the observed history? | `K__local` versus `raw__raw` |

The full identifiers append the predictor, `linear` or `nonlinear`. `K`, `A`, and `R` denote KAHM, affine, and kernel-weighted averaging reconstructions. `mem` means memberships alone; `norm` adds log normalization constants; `local` adds membership-weighted local coordinates. The nonlinear predictor fits ridge regression to the representation concatenated with Gaussian-kernel Nyström features from 64 training landmarks. The results concern this specified predictor family.

Each procedure makes its own validation choice using the same validation observations within a task. The CSV label `independent` refers to these separate choices. Reconstruction-based procedures select a maximum local dimension as well as predictor settings; raw histories select predictor settings only. There is no refit on the combined training and validation data. Consequently, a comparison does not isolate adding coordinates while holding every fitted choice fixed. See the [recorded protocol](../extensions/score_enrichment_20261102/protocol.json) for the complete design.

## Rebuild all 40 primary comparisons from saved losses

The [central bundle](../reproduction/score_enrichment/) contains 17 experiment records, including `selected_losses.csv.gz`. It holds 26,624 selected-loss rows covering four data-generating processes, 64 repetitions, 26 procedures, two selection rules and two evaluation splits.

From the repository root, using Python 3.12:

```bash
python3 -m pip install -r reproduction/score_enrichment/requirements.txt
python3 reproduction/score_enrichment/check_saved_losses.py
```

The independent checker verifies the distributed records and the exact decompressed CSV identity. It reconstructs 416 procedure summaries, the 2,560 paired differences behind the primary family, and all 40 comparison rows. It also checks the set of selected records and consistency of recorded settings. Its JSON report gives `status: PASS` when the reconstructed values agree with the saved results within the declared arithmetic tolerances.

To save the rebuilt tables, use a new directory outside the bundle:

```bash
python3 reproduction/score_enrichment/check_saved_losses.py --output /tmp/kahm-score-analysis
```

Inspect `rebuilt_primary_paired_repetitions.csv` to follow individual paired differences and `rebuilt_primary_contrasts.csv` for their intervals and decisions. `rebuilt_all_procedure_summary.csv` contains the procedure-level means and standard deviations; `verification.json` records the checks. The primary reconstruction uses only test split D losses under the `independent` selection rule.

| Command | Inputs | Scope |
| --- | --- | --- |
| `scripts/explain_primary_comparison.py` | Primary-summary table | Readable explanation or JSON for selected rows; standard library only |
| `reproduction/score_enrichment/check_saved_losses.py` | Complete saved selected-loss rows and saved evidence tables | Independent reconstruction of procedure summaries, paired differences and primary comparisons |
| `extensions/score_enrichment_20261102/verify_primary_summary.py` | Reported means and standard deviations | Summary-arithmetic check using SciPy |

The primary decisions are 16 lower candidate mean losses, 7 higher candidate mean losses and 17 unresolved comparisons. They cover five questions, four data-generating processes and two predictor families. The intervals use Bonferroni adjustment over this complete family and retain the paired-Student approximation; numerical agreement does not make the intervals distribution-free.

The checker starts from saved standardized losses. It can verify each three-horizon average, but the raw predictions, training targets and target variances are not supplied. It therefore does not recompute those losses from forecast errors or demonstrate that each selected setting was the optimum over all validation candidates. The [bundle guide](../reproduction/score_enrichment/README.md) identifies the exact checks and recorded audit results.

## Explore the supplementary capacity question

The separate [predictor-capacity study](../reproduction/capacity_followup/) asks whether KAHM local-coordinate benefits persist at 64 and 256 Nyström landmarks. It provides its experiment source and archived records for 24 repetitions under three data-generating processes. Its 12 comparisons use MSE, not the central experiment's variance-standardized SMSE.

From the repository root, check its files without installing numerical packages:

```bash
cd reproduction/capacity_followup
python3 -m reproduce verify
```

For the saved-record statistical audit, install that bundle's pinned requirements in a suitable Python 3.12 environment and run `python3 -m reproduce audit --output runs/archive-audit` from the same directory. It rebuilds both the supporting first-stage analysis and the capacity comparisons without fitting models. The bundle also supplies a `full` mode for regenerating its observations and fitted predictors; see its README before running it.

**The comparison direction differs between the studies.** Central comparisons use candidate minus reference, so a negative value favors the candidate. The capacity table uses reference minus KAHM, so a positive value favors KAHM. Its practical margins and 12-comparison confidence family belong to that separate protocol. A result at 256 landmarks concerns that tested finite predictor and does not establish superiority over every possible raw-input predictor.

## Central model-generation inputs

The central saved-loss analysis is executable here. Recomputing its features and forecasts requires the complete implementation, exact task list, prepared observations and fitted numerical models. The central generator, runner and complete numerical archives are not supplied, and no public download endpoint for those archives is established.

The [reproduction guide](../REPRODUCIBILITY.md) and [availability guide](../extensions/score_enrichment_20261102/AVAILABILITY.md) identify the inputs needed for each task. The central results concern direct forecasts in the specified simulated settings. They do not test autonomous evolution of the representation or establish a universal KAHM advantage.
