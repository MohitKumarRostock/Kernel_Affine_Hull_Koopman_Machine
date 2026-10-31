# Reconstruction departures for KAHM forecasting

This experiment asks whether a KAHM reconstruction provides useful additional functions to a finite forecasting decoder. It compares a bounded KAHM reconstruction departure with a common gate-and-coordinate basis, a matched radial-basis-function (RBF) departure, raw-history RBF features, boosted trees, and other controls.

The KAHM procedure reduces aggregate nonlinear normalized mean squared error by 12.58% relative to the common basis. Its 1.17% gain over matched RBF departures and 3.92% gain over raw-history RBF features do not establish the prespecified 3% improvement under the simultaneous analysis. Boosted trees and the validation-selected non-KAHM procedure have lower aggregate errors. The linear-control noninferiority criterion and all specified numerical stability checks are satisfied.

Read the [mathematical background](MATHEMATICAL_BACKGROUND.md), [complete findings](FINDINGS.md), [six primary comparisons](analysis/decision.json), and [manuscript](../tmlr_integration/README.md).

## Design and evidence

The design has 24 independent repetitions, four simulation laws, and two added observation-noise levels, giving 192 task suites. Each suite evaluates ten procedures. The three nonlinear laws had been studied during method development; new seeds yield independent realizations under those laws, not new application domains.

Independent base episodes are assigned to four partitions. Partition A fits the representation and predictors; B selects horizon-specific regularization; C selects correction strengths and a non-KAHM forecasting procedure; D evaluates the selected procedures. The added-noise amplitude is estimated from A and shared across the partitions. The primary statistical units are the 24 repetition aggregates, rather than individual windows or forecast values.

The local prospective checkpoint fixes the scientific code, inputs, design, and statistical comparisons before the held-out forecasts were computed. It is not a public preregistration. The [original protocol](PROTOCOL.md), [algebra record](ALGEBRA.md), and `freeze.json` retain the exact bytes covered by that checkpoint. The current mathematical guide explains the same constructions for readers without changing those records.

## Restore and verify saved evidence

Use Python 3.12 from the repository root. Large evidence files are stored in lossless parts. Restoration checks each part, the combined archives, and every extracted file; it accepts identical existing files and refuses conflicting replacements.

```bash
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
python -m extensions.reconstruction_mechanism.evidence unpack
python -m extensions.reconstruction_mechanism.run verify
```

The standalone reproduction ZIP already contains the expanded mechanism evidence and does not require the unpack command. The repository transport contains the mechanism and adaptive studies; the standalone mechanism archive contains the principal study and its numerical dependencies.

Recompute the saved-forecast analysis in a separate directory:

```bash
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-analysis
python -m extensions.reconstruction_mechanism.audit \
  --analysis-json /tmp/kahm-analysis/decision.json \
  --output /tmp/kahm-independent-audit.json
```

The audit uses independent prediction and statistical code. It evaluates saved selected states and verifies recorded selection information; it does not refit all discarded candidate models. Its coverage and limitations are detailed in the [findings](FINDINGS.md).

Replay every selected model and verify the evidence required by the packager:

```bash
python -m extensions.reconstruction_mechanism.run replay
python -m extensions.reconstruction_mechanism.package_results audit
```

These operations verify the recorded computation. They do not add independent experimental observations.

### Numerical portability

Saved-forecast statistical reproduction and selected-state inference are separate checks. The original numeric forecasts and A-target variance scales reproduce the reported losses, repetition aggregates, and six primary confidence bounds directly. The anonymous saved-forecast supplement performs this calculation without loading fitted Python model objects.

Selected-state replay uses relative tolerance `1e-10` and absolute tolerance `1e-12`. It requires compatible numerical kernels as well as matching software versions. A controlled diagnostic under Python 3.12.14, NumPy 2.3.5, and glibc 2.39 reproduces the recorded raw-RBF predictions for Van der Pol task `vanderpol__noise0__seed946100` exactly with default processor dispatch. Disabling NumPy's AVX512 dispatch before import causes 49 of its 4,992 raw-RBF predictions to exceed the original pointwise tolerance; the largest absolute difference is approximately `1.07e-10`. This is a failed strict replay under that dispatch configuration. The diagnostic concerns numerical inference portability; reported losses and confidence bounds are computed from the archived forecast arrays. The [numerical portability record](publication_validation/NUMERICAL_REPLAY_PORTABILITY.json) gives the environment, invocation, source hashes, and complete targeted comparison.

## Numerical checks and manuscript build

The numerical tests use fabricated data or a development seed excluded from the research roster:

```bash
python -m unittest extensions.reconstruction_mechanism.test_models \
  extensions.reconstruction_mechanism.test_inputs \
  extensions.reconstruction_mechanism.test_analysis -v
```

To regenerate the principal displays and compile the paper:

```bash
python extensions/tmlr_integration/build_mechanism_assets.py
python -m extensions.tmlr_integration.format_manuscript_assets
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The bundled manuscript displays also permit compilation without restoring evidence or fitting models. The [continuous-integration workflow](../../.github/workflows/reconstruction-mechanism.yml) describes the automated verification sequence.

## Repeat the fitting experiment

A repeated execution must preserve the reported outputs. In a separate disposable checkout, restore the evidence and move the existing `results/` directory outside the checkout. Then run:

```bash
python -m extensions.reconstruction_mechanism.run run --freeze-commit FULL_CHECKOUT_COMMIT_SHA
```

Replace `FULL_CHECKOUT_COMMIT_SHA` with the full commit identifier of that checkout containing the exact `freeze.json`. The runner checks the frozen scientific files, creates a fresh process for each task, and refuses to overwrite task results. The fitting API receives A, B, and C information and D histories, but no D targets.

Refitting reproduces the fixed procedure; it is not a newly designed confirmatory experiment. Preserve its forecasts and resource measurements separately. New methods or altered designs require their own source and input records so that they cannot be confused with the reported experiment.

## File organization

| File or directory | Purpose |
| --- | --- |
| `config.json`, `PROTOCOL.md`, `freeze.json` | Experimental design and prospective local source/input identities |
| `MATHEMATICAL_BACKGROUND.md` | Definitions, identities, feature-span results, and correction bound |
| `algebra.py`, `ALGEBRA.md`, `algebra_results.json` | Original structural checks and numerical examples |
| `data.py`, `prepared_inputs/manifest.json` | Episode generation, paired noise, causal windows, and input roster |
| `models.py`, `run.py` | Fitting, selection, held-out prediction, saved states, and replay |
| `analyze.py`, `analysis/` | Statistical analysis and complete primary and secondary results |
| `audit.py`, `results/audit.json` | Independent-code evaluation within the study |
| `evidence.py`, `PACKED_EVIDENCE.json`, `evidence_parts/` | Lossless evidence storage and restoration |
| `package_results.py` | Verification and packaging of the manuscript and principal reproduction archive |

Evidence transport, independent verification, and packaging utilities were added after experiment execution. Their provenance is recorded separately from the prospective scientific checkpoint.
