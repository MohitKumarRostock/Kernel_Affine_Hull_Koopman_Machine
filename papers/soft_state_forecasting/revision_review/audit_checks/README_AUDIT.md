# Revision audit: KAHM forecasting features

Revision date: **2026-10-31**.

Manuscript: **Kernel Affine Hull Features for Time-Series Forecasting: Feature
Spans, Bounded Corrections, and Matched Evaluation**.

The programs and JSON outputs here document checks performed during revision.
The current manuscript text and PDFs are local revisions. The archived
experiment evidence is pinned separately to commit
`7dda356138675934aab5e327ec667bde0a5eda79` in the public repository:

[Pinned principal experiment](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/7dda356138675934aab5e327ec667bde0a5eda79/extensions/reconstruction_mechanism)

## What was checked

| Check | Scope and limitations |
| --- | --- |
| `audit_method_theory.py` | Newly written floating-point diagnostics for normalized-weight identities, equal-span examples, the stated irregular bank, radial nonexpansiveness, and the intercept-inclusive ridge bound. Includes 240 identity queries, three sampled equal-span banks, 480 radial-map pairs, and 36 ridge fits on diagnostic data. Sampled calculations do not prove global function-space statements. |
| Author-supplied `verify_saved_forecasts.py` | Rerun through recovery of the original compact supplement. Checks the packaged file manifest, saved forecast and target arrays, normalization, eight analysis tables, and six primary contrasts. This is a rerun of the supplied verifier. |
| `independent_compact_forecast_checks.py` | Newly written replay of the six primary contrasts from the compact supplement. Independently checks manifest and source-array hashes, normalization, the task roster, and policy forecast identities. It imports no supplied verifier and does not reconstruct all eight tables. |
| `independent_saved_forecast_checks.py` | Newly written arithmetic audit of the original full numeric evidence. Checks input and prediction hashes, episode/window construction, paired-noise records, forecasting statistics, and the internal arithmetic of recorded selection scores, bounds, and resource measurements. It imports no study fitting or analysis module. |

The principal evidence comprises 192 task suites, 1,920 prediction arrays, and
9,584,640 scalar forecasts. The author-supplied compact replay checks 399 files
covered by its manifest and 115,024 reported table cells. The full independent
audit additionally checks 768 episode-partition/window records, 96 paired-noise
task pairs, and 192 policy forecast identities. Its statistical replay retains
the two established primary criteria: the nonlinear improvement over common
features and the linear-control noninferiority criterion. The compact
independent replay reaches the same decisions; its largest discrepancy among
recomputed primary numeric fields is approximately `1.263e-14`.

No forecasting model was refitted, no selected model object was loaded, and no
new forecasting experiment was run for these revision checks. Selection checks
use recorded candidate scores; they do not independently refit candidates or
recompute their validation forecasts. Bound and resource checks evaluate stored
diagnostic arithmetic; they do not regenerate features or repeat timings.
The recorded approximate statistical intervals retain their original sampling
assumptions. Arithmetic agreement supplies no new experimental observations.

Recorded environment: Python **3.12.14**, NumPy **2.3.5**, SciPy **1.17.0**.
The theory diagnostics require NumPy. The empirical arithmetic checks require
NumPy and SciPy. Retrieval and restoration helpers use the Python standard
library. The unchanged numeric supplement supplies its own `requirements.txt`.

## Package layout and paths

Run the examples below from the extracted source-package directory containing
`main.tex`. The source package places these programs under
`revision_review/audit_checks/` and provenance records under
`revision_review/provenance/`. New results go under `revision_review/reruns/`.
The recorded JSON outputs supplied with the revision remain unchanged.

`KAHM_TMLR_Saved_Forecasts.zip` is delivered separately from the source ZIP.
The examples assume that it is one directory above the extracted source root,
and extract it into the sibling directory `saved_forecasts`. Substitute your
chosen paths when using a different extraction location. Use new extraction
and result directories to preserve earlier evidence and outputs.

The unchanged numeric ZIP has **80,805,506 bytes** and SHA256:

```text
d149ae648b0b380eccf108d8d217566be173182a0ac047982afad4836fb92edd
```

## 1. Compact saved-forecast replay using the supplied verifier

Extract the separately delivered ZIP into a new directory:

```bash
python -m zipfile -e ../KAHM_TMLR_Saved_Forecasts.zip ../saved_forecasts
```

Its top-level folder is `saved_forecast_supplement/`. Install the specified
dependencies in your chosen Python environment, then run the verifier:

```bash
python -m pip install -r ../saved_forecasts/saved_forecast_supplement/requirements.txt
python ../saved_forecasts/saved_forecast_supplement/verify_saved_forecasts.py \
  --evidence-dir ../saved_forecasts/saved_forecast_supplement \
  --write-results revision_review/reruns/author_saved_forecast_replay
```

Dependency installation may need network access. The verifier itself works
offline, uses NumPy loads with `allow_pickle=False`, and imports no fitted
model. It refuses an existing result directory or a result directory inside
the evidence bundle. It writes eight recomputed CSV tables and
`verification.json`; without `--write-results`, it prints the report only.

The revision's successful author-verifier record is the `verification` object
inside `PACKAGE_SUMMARY.json`, retained with the provenance records. It was
produced when the original numeric ZIP was recovered and verified. This
compact bundle suffices for the commands in this section; it contains neither
the full A/B/C/D episode inputs nor the per-task fitting summaries needed for
the expanded audit below.

## 2. Independent primary-contrast replay from the same compact ZIP

Using the extracted supplement from Section 1, run:

```bash
python revision_review/audit_checks/independent_compact_forecast_checks.py \
  --evidence-dir ../saved_forecasts/saved_forecast_supplement \
  --output revision_review/reruns/independent_compact_forecasts.json
```

Both options are required. The script reads the original numeric arrays and
recalculates the six primary contrasts with separately written arithmetic. It
checks all 399 manifest-listed file hashes, 192 task suites, 1,920 prediction
arrays, A-target normalization, and the 192 policy identities. It does not
import or invoke the author verifier. The full collection of eight tables is
checked by the author verifier and the expanded audit, rather than this compact
script.

The output must be outside the immutable evidence directory. Its parent is
created automatically, and an existing output can be replaced; choose a fresh
filename to preserve prior reports. The recorded revision result is
`independent_compact_forecast_checks.json` beside this README. This command
requires only the separately delivered compact ZIP and the audit script; the
larger public transport and per-task fitting summaries are unnecessary.

## 3. Independent mathematical diagnostics

`audit_method_theory.py` has no command-line options. It writes
`method_theory_checks.json` beside its own script and would replace that file
on a repeated run. To preserve the supplied output, first copy the script into
a new directory. For example, create `revision_review/reruns/theory/`, copy
`revision_review/audit_checks/audit_method_theory.py` into it, and run:

```bash
python revision_review/reruns/theory/audit_method_theory.py
```

The program prints and saves its diagnostic report. It uses no manuscript
assets, forecast arrays, or archived study implementation. Its fabricated
diagnostic fits test formulas and are separate from the forecasting experiment.

## 4. Expanded independent audit of the original numeric evidence

`independent_saved_forecast_checks.py` requires the original study directory,
containing `config.json`, `freeze.json`, `prepared_inputs/manifest.json`, the
192 complete prepared-input NPZ files, 192 prediction NPZ files and matching
`summary.json` files, the eight statistical CSVs in `analysis/`, and
`analysis/stability.csv`. The compact saved-forecast ZIP alone is insufficient.

If that original evidence has already been restored into a chosen directory,
run the following, replacing the example path after `--study-dir`:

```bash
python revision_review/audit_checks/independent_saved_forecast_checks.py \
  --study-dir /path/to/pinned_evidence/extensions/reconstruction_mechanism \
  --output revision_review/reruns/independent_full_evidence.json
```

Both options are explicit to avoid the development-workspace default paths.
The program creates the output parent directory and writes the selected JSON
path; choose a fresh filename because it can replace an existing output.
It does not modify the input evidence.

### Optional retrieval and selective restoration

The included helpers can retrieve the original public evidence when the full
audit is needed. This is a separate, larger download than the compact replay;
it requires network access and the supplied pinned Git-tree record. From the
same source-package directory, an example using a new sibling destination is:

```bash
python revision_review/audit_checks/retrieve_pinned_empirical_evidence.py \
  --tree revision_review/provenance/mechanism_tree.json \
  --destination ../kahm_pinned_evidence \
  --parts
python revision_review/audit_checks/restore_numeric_evidence.py \
  --evidence-root ../kahm_pinned_evidence \
  --output revision_review/reruns/numeric_evidence_restoration.json
python revision_review/audit_checks/independent_saved_forecast_checks.py \
  --study-dir ../kahm_pinned_evidence/extensions/reconstruction_mechanism \
  --output revision_review/reruns/independent_full_evidence.json
```

The downloader uses the immutable commit and verifies retrieved bytes against
the Git-tree hashes. The restoration helper checks transport-part and combined
archive SHA256 values, validates membership and paths, and restores selected
numeric inputs and summaries. It does not extract fitted model states or
execute archive scripts. Existing identical restored files are accepted;
conflicting restored files fail verification. The helper writes its combined
transport archive and selected report, so use a separate working destination.

Historical tests, selected-state inference, and complete candidate refitting
described in the public repository are separate operations. None is implied by
successful completion of the compact or expanded arithmetic checks above.
