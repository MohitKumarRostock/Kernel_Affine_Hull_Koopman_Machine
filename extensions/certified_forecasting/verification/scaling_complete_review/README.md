# Scaling, scientific scope, and portable reproduction checks

These records describe the manuscript and software verification performed on
29 October 2026 from base commit `ec206876381c50ef91804bbad1e7493dde88cea8`.
The compiled manuscript and source hashes in the records identify this edition.
Historical experiment manifests retain their own execution-source identities.

## Scope

The mathematical review found no counterexample to the principal results under
their stated bounded-loss and independent-episode assumptions. The exposition
now specifies the local KAHM regularization recurrence and explains input
scale sensitivity. Numerical verification is not a formal proof or an exhaustive
assessment of novelty.

The separate speech scaling study is descriptive on the previously examined
recording cohort. No new population confidence statement is made for it.
Its full source and evidence are in `../../experiments/speech_scaling/` and
`../../results/speech_scaling_v1/`.

## Evidence

| Record | Scope and outcome |
|---|---|
| `core_tests.log` | All 56 standard tests pass, with no skips. Includes historical numerical parity and nine preprocessing tests. |
| `witness_tests.log` | All seven mathematical benchmark tests pass. |
| `scaling_mechanism.json` | Fixed reference cloud: invariant Gram matrix to tolerance, changed regularization, reconstructions, and folding scores. No forecasting superiority claim. |
| `retained_statistics.json` | 1,796 candidate records, 5,388 lower/upper pairs, and 792 improvement endpoints reconstruct from saved moments with zero arithmetic discrepancy. |
| `scaling_statistics.json` | Original and confirmation scaling aggregates reconstruct from all selected-candidate records. |
| `table_reaggregation.json` | All 21 matched-geometry rows and eight original speech rows reconstruct, including means, standard deviations, paired counts, and upper endpoints. |
| `pendigits_scaling_scope.json` | Distinguishes improvement for every writer after horizon averaging from five adverse writer/fit/horizon contrasts across the two phases. |
| `speech_scaling_scope.json` | Independently recomputes new scaling means and confirms the conventional baselines remain more accurate in all 108 specified comparisons. |
| `speech_scaling_independent_audit.py` and its JSON outputs | Imports neither experiment runner nor its verifier; reconstructs all 26,460 group/horizon losses, selections, summaries, differences, and displayed table cells. The full audit also checks numeric training arrays without loading pickles. |
| `decision_states/reconstruction.json` | Every reconstructed state archive matches its original SHA-256. Numeric arrays are generated outputs omitted from Git. |
| `decision_audit.json` | All 600 fitted models, 3,000 endpoints, 6,000 mass linear programs, and 600 independent RNG streams verify. |
| `latex_build.log`, `layout.json`, `pdf_fonts.txt`, `release_checks.json` | Successful 55-page build; no undefined references/citations, overfull boxes, or LaTeX warnings; all fonts embedded; all pages visually inspected. |

## Reproduction

Use the extension's `docs/REPRODUCIBILITY.md` for exact commands and the
study's source snapshot for the original execution. Independent audit output
should be directed to a new file, for example:

```bash
python verification/scaling_complete_review/speech_scaling_independent_audit.py \
  --portable --output verification/reproduced_speech_scaling_independent.json
```

The tests and numerical checks do not establish that real recording groups
are independent population draws. They also do not make repeated fitting
seeds into independent cohorts or establish universal KAHM accuracy gains.
