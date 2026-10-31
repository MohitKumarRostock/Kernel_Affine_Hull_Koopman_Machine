# Reconstruction departures: completed KAHM forecasting study

**Completed result: useful value beyond a common local-coordinate basis, but no established KAHM-specific practical advantage under the fixed comparison. The decision is `STOP_FORECASTING_PERFORMANCE_BRANCH`.**

KAHM's bounded departure lowers nonlinear aggregate error by 12.58% against the common basis and by 1.17% against matched bounded RBF. Only the first establishes the chosen 3% useful margin. The 3.92% raw-RBF point gain has an unresolved simultaneous useful-margin bound. KAHM error is 1.77% above boosting and 6.04% above the C-selected non-KAHM policy. All stability screens and linear-control noninferiority pass.

Read the [complete findings](FINDINGS.md), [six-contrast decision](analysis/decision.json), [algebra](ALGEBRA.md), and [integrated manuscript](../tmlr_integration/README.md). The preserved [protocol](PROTOCOL.md) documents the experiment that was executed; it is supporting evidence for the completed results.

## Scope

The fixed study contains 24 independent repetitions × four laws × two added-noise levels = 192 suites, with ten procedures each. The three nonlinear laws are previously investigated simulations; independent fresh draws do not add new application domains. The inferential unit is a complete repetition aggregate, not any of the 9,584,640 scored values. The source/input freeze was local and prospective, not publicly preregistered. Earlier adaptive development remains separate and retains its original and amended attempts.

## Restore and verify saved evidence

Use Python 3.12 from the repository root. The public tree stores bulky exact evidence in lossless parts. Restoration verifies part, archive and per-file hashes and refuses to overwrite conflicting bytes.

```bash
python -m extensions.reconstruction_mechanism.evidence unpack
python -m pip install -r extensions/reconstruction_mechanism/requirements.txt
python -m extensions.reconstruction_mechanism.run verify
python -m extensions.reconstruction_mechanism.analyze --output /tmp/kahm-analysis-recheck
python -m extensions.reconstruction_mechanism.audit \
  --analysis-json /tmp/kahm-analysis-recheck/decision.json \
  --output /tmp/kahm-independent-audit.json
python -m extensions.reconstruction_mechanism.run replay
python -m extensions.reconstruction_mechanism.package_results audit
```

The downloaded reproduction ZIP already contains expanded mechanism evidence; it does not require the transport unpack command and does not include the separate adaptive study. The repository's transport includes both studies. The mechanism audit imports neither the forecasting nor analysis code and does not refit models. It checks selected coefficients, independently evaluates saved predictions and recomputes all six contrasts; its verification limits are stated in [the findings](FINDINGS.md).

## Tests and manuscript

```bash
python -m unittest extensions.reconstruction_mechanism.test_models \
  extensions.reconstruction_mechanism.test_inputs \
  extensions.reconstruction_mechanism.test_analysis -v
python extensions/tmlr_integration/build_mechanism_assets.py
cd extensions/tmlr_integration/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The 38 tests use fabricated data or development seed 975001, excluded from the research roster. Building the paper uses the bundled generated assets and needs no model fitting. [The dedicated workflow](../../.github/workflows/reconstruction-mechanism.yml) verifies the new saved evidence and packages the final manuscript. Historical studies keep their own frozen checks and documented CPU sensitivities.

## Refit the fixed experiment

The study's original inputs, configuration and decision rules are immutable. To repeat the same fitting experiment, use a separate disposable checkout, restore the evidence, move the existing `results/` directory outside that checkout, and run:

```bash
python -m extensions.reconstruction_mechanism.run run --freeze-commit FULL_CHECKOUT_COMMIT_SHA
```

Supply the full commit of that checkout containing the exact `freeze.json`; this is a new execution of the fixed procedure, not a new confirmatory design. The runner verifies all frozen scientific files, creates a fresh process per task and refuses to overwrite task results. A/B/C supply fitting and selection; the model API never receives D labels. Archive the original outputs before any repetition, and keep repeated timing observations and predictions separate. To study another procedure, create and freeze a separate experiment rather than altering this completed one.

## Files

| File or directory | Role |
| --- | --- |
| `config.json`, `PROTOCOL.md`, `freeze.json` | Fixed design and prospective local source/input identities |
| `algebra.py`, `ALGEBRA.md`, `algebra_results.json` | Completed structural checks and proofs |
| `data.py`, `prepared_inputs/manifest.json` | Independent episode generation, paired noise, causal windows, exact input roster |
| `models.py`, `run.py` | A/B/C-only fitting, reserved prediction, preserved states and replay |
| `analyze.py`, `analysis/` | Frozen complete-roster analysis and all primary/secondary results |
| `audit.py`, `results/audit.json` | Internal verification using independent evaluation code |
| `evidence.py`, `PACKED_EVIDENCE.json`, `evidence_parts/` | Post-execution lossless transport and exact restoration |
| `package_results.py` | Completed-result, LaTeX-source and expanded-evidence packaging |

The last three utilities are post-execution verification/packaging tools and are not retroactively represented as prospectively frozen scientific code. Historical numerical dependencies are included unchanged under their original repository paths and licenses.
