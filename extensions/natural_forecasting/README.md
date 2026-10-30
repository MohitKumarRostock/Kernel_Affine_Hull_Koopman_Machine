# Natural forecasting: accuracy, computational costs, and prospective selection

This extension studies whether KAHM geometry helps forecasting on a bounded set of natural time series, whether its complete pipeline cost is worthwhile, and whether a diagnostic can identify useful cases before their test outcomes. It preserves the earlier predictive-geometry study and its archived evidence.

The declared design has eight original source families, 26 selected series, and two paired history/horizon settings per series. Two capacity settings and two fitting seeds produce **208 task runs: 128 development and 80 reserved confirmation runs**. The independent descriptive groups remain **four development and four confirmation source families**, not 208 independent datasets.

This guide explains the observed findings, evidence, reproduction commands, and limits on interpreting them. Detailed settings and provenance are in [PROTOCOL.md](PROTOCOL.md), [protocol.json](protocol.json), and [data_sources.json](data_sources.json).

## Observed confirmation results

The reserved confirmation results did not show an average benefit for adding KAHM to the raw-history RBF pipeline. The following are source-balanced means from the [archived method summary](runs/natural_v1/reports/method_summary.csv), available after restoring the reference run:

| Method | Scaled test MSE | Warm pipeline wall time (seconds per task) |
|---|---:|---:|
| Raw-history RBF | 0.8075209 | 0.168791 |
| Raw history + KAHM RBF | 0.8419228 | 0.387412 |
| Raw-history ridge | 0.7798112 | 0.188043 |

Raw history + KAHM had **4.26% higher scaled MSE** and **2.30 times the measured warm pipeline cost** of raw-history RBF in this reference workflow. The time includes the declared preparation, search, refit, prediction, serialization, cleanup, and bookkeeping scopes; it is not a hardware-independent speed ratio. In the predeclared diagnostic comparison, adding geometric features to metadata plus the cheap probe changed **0 of 80 confirmation decisions**. The [confirmation analysis](runs/natural_v1/confirmation_analysis.json) and [derived report](runs/natural_v1/reports/summary.json) retain all methods, diagnostic variants, and source-level details.

These 80 task instances come from **four independent confirmation source families**. Related series, history/horizon settings, capacities, and seeds remain dependent. The study does not validate a general criterion for predicting when KAHM will help; its numerical results and computational costs describe this bounded source collection and implementation.

## Preserve and inspect the evidence first

The original repository checkpoint is `a67c40190458bab7b3e56b4fac76770798d074b3`; [CHECKPOINT.json](CHECKPOINT.json) binds all 217 previously tracked files. The new scientific code, source allocation, task identities, and analysis specification were frozen before development outcomes. The public protocol commit is [`2600e45220b0562b5f67ef0c769fcae6c11e120e`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2600e45220b0562b5f67ef0c769fcae6c11e120e). After development, the diagnostic freeze was published at [`4ab972ba08a4d5678347c1ea124ed6509c806d5a`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/4ab972ba08a4d5678347c1ea124ed6509c806d5a) before confirmation execution. These repository records are not registration with a dedicated preregistration service.

From the repository root, the following check needs only Python's standard library:

```bash
python extensions/natural_forecasting/package.py check-source
```

It verifies all historical files and the registered scientific dependency hashes. Post-study documentation, reporting, and packaging tools are separate additions; they do not redefine the frozen forecasting implementation. The release manifest binds their delivered bytes.

## Inspect the archived results without repeating forecasting fits

Git stores the reference run as ordered 8 MiB binary parts and the manifest `reference/natural_v1.zip.parts.json`. The manifest records each part's filename, size, and SHA-256, together with the complete ZIP's size and SHA-256. This avoids per-request upload limits while preserving the exact archive bytes. In a source checkout whose reference run has not been expanded, verify and restore it with:

```bash
python extensions/natural_forecasting/package.py verify extensions/natural_forecasting/reference/natural_v1.zip.parts.json
python extensions/natural_forecasting/package.py restore-reference extensions/natural_forecasting/reference/natural_v1.zip
```

When the requested full ZIP is absent, restoration automatically reconstructs it from the sibling parts into a temporary directory. It verifies every part, the complete ZIP, all archived payloads, and frozen scientific dependencies, then creates `extensions/natural_forecasting/runs/natural_v1`. The parts manifest can also be passed directly to `restore-reference`. It refuses to overwrite an existing run and does not modify scientific source files or leave a reconstructed ZIP in the checkout. A full reproducibility release already includes the expanded run, so restoration is unnecessary there.

The full reproducibility ZIP retains repository-relative paths. Its top-level `README_ARCHIVE.md` is the entry point for this experiment; historical root documentation is preserved unchanged as part of the source closure.

From an extracted reproducibility root, verify every packaged payload before generating new files:

```bash
python extensions/natural_forecasting/package.py verify .
```

Alternatively, the same script can verify a ZIP directly without extracting or executing its contents:

```bash
python extensions/natural_forecasting/package.py verify /path/to/KAHM_Natural_Forecasting_Reproducibility.zip
```

Verification checks complete inventory, byte lengths, and SHA-256 hashes, and rejects unsafe archive paths, duplicate entries, symlinks, and unlisted payloads. Each internal manifest explicitly excludes its own hash; `RELEASE_MANIFEST.json` binds the complete ZIP bytes including that manifest. These checks establish consistency, not an independent assertion of authenticity.

Linux with Python 3.12 is the reference scientific environment. Install the pinned dependencies in an environment **outside the preserved archive directory**:

```bash
python3.12 -m venv ../natural-review-env
source ../natural-review-env/bin/activate
python -m pip install -r extensions/natural_forecasting/requirements.txt
```

Then audit saved predictions or replay the analysis into a separate empty directory:

```bash
python extensions/natural_forecasting/package.py audit .
python extensions/natural_forecasting/package.py replay . --output ../natural-analysis-replay
```

`audit` verifies frozen source/data/result hashes, recomputes errors from saved prediction arrays, compares targets and origins against prepared windows, checks complete method/task rosters, and evaluates the frozen confirmation policies. It also refits the small diagnostic regressions to check frozen coefficients and archived scientific summaries. Fresh decision/regression timings are excluded from cross-replay equality, while original timing totals and aggregates are checked against their recorded row values. It performs **zero forecasting-model or geometry-bank fits** and is read-only with respect to the source and archived run.

`replay` additionally regenerates development and confirmation analysis JSON in the supplied output directory. Development replay refits only the small diagnostic ridge regressions used in source-family folds. It does not repeat the forecasting benchmark. Newly measured scalar-decision and diagnostic-regression times can differ; archived resource measurements remain unchanged. Offline auditing checks the frozen implementation and numerical evidence without requiring an identical Python build string. A fresh full run records and enforces its own environment freeze.

Both `audit` and `replay` also accept the reproducibility ZIP path. They verify it before extracting into a temporary directory and remove that temporary copy afterward. Use a fresh Python CLI process for each operation.

## Four-stage computational workflow

Full computational reproduction repeats the declared forecasting fits. It does not provide new independent source evidence. Use a fresh run directory so the delivered `runs/natural_v1` remains intact. Start from the repository root in the pinned environment:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
NATURAL_RUN=extensions/natural_forecasting/runs/independent-reproduction
```

### Stage 1 — Prepare, freeze, and run the bounded development study

```bash
python -m unittest discover -s extensions/natural_forecasting/tests -p 'test_*.py' -v
python -m extensions.natural_forecasting.data fetch --role all
python -m extensions.natural_forecasting.run prepare --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run initial-freeze --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run run --role development --run-root "$NATURAL_RUN"
```

The 53 scientific tests use synthetic arrays or source metadata; they do not evaluate natural-data forecasting performance. The fetch command verifies bundled source ZIPs and downloads only absent pinned public snapshots. It needs network access only when a source is missing. `prepare` constructs arrays and the entire 208-run roster without fitting a model. Initial freezing binds source/code hashes, protocol, roster, dependency versions, and environment before development execution.

The source roster is fixed using training-prefix eligibility and a salted hash of source/series identity. All targets must lie entirely within their chronological split. The loader performs no learned normalization or interpolation. Models fit their own preprocessing on allowed training data. Missing-window availability never selects replacement series.

### Stage 2 — Analyze development, fit the diagnostic, and freeze confirmation

```bash
python -m extensions.natural_forecasting.run freeze-confirmation --run-root "$NATURAL_RUN"
```

This command first audits saved development predictions, computes source-family-held-out diagnostic evaluation, and fits all four predeclared diagnostic variants on the complete development families. It writes the frozen coefficients and binds the development records. This is the boundary between development and reserved confirmation, not an opportunity to select a diagnostic using confirmation losses.

### Stage 3 — Execute reserved confirmation and audit it

```bash
python -m extensions.natural_forecasting.run run --role confirmation --run-root "$NATURAL_RUN"
python -m extensions.natural_forecasting.run analyze --run-root "$NATURAL_RUN"
```

Confirmation requires intact scientific code, dependencies, roster, coefficients, and development results. It cannot select an arbitrary subset of tasks. Resumption skips only complete result records whose prediction artifacts match their hashes. Incomplete or mismatched results stop execution and remain available for explicit audit; they are never silently overwritten or removed from the comparison.

Final analysis evaluates the frozen policies, recomputes prediction errors, and checks preservation of all historical tracked files. The run's `VALIDATION.json` records the completed checks. Recorded timing scopes and source-level aggregation are retained in derived results.

### Stage 4 — Report the results, build the manuscript, and package the release

For a fresh computational run, regenerate numerical summaries and paper assets inside that run directory:

```bash
python -m extensions.natural_forecasting.report \
  --run-root "$NATURAL_RUN" \
  --reports-dir "$NATURAL_RUN/reports" \
  --manuscript-dir "$NATURAL_RUN/manuscript-generated"
```

The reporter consumes saved predictions and validated analysis results; it performs no new model fitting or diagnostic selection. The delivered manuscript already includes the reference run's generated figures and tables. Its LaTeX sources are in `manuscript/`. Install TeX Live, `latexmk`, and `lmodern` (the Latin Modern font package). From the repository root, change to the manuscript directory before building:

```bash
cd extensions/natural_forecasting/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ../../..
```

The official JMLR style and its provenance are bundled. Paper compilation performs no model fitting. Checksum verification should precede rebuilding an extracted archive because LaTeX adds auxiliary files.

After the final PDF and complete run are stable, create the release from the repository root:

```bash
python extensions/natural_forecasting/package.py build \
  --run-dir extensions/natural_forecasting/runs/natural_v1 \
  --manuscript-pdf /path/to/Natural_Forecasting_Manuscript.pdf \
  --output /path/to/release \
  --handoff
```

The builder requires all 208 complete records, the full twelve-method portfolio per task, verified prediction artifacts, both freezes, final analyses, a passed final validation, all eight original data ZIPs, and the preserved historical checkpoint. It rejects a PDF older than its manuscript inputs. It produces separate manuscript-source and reproducibility ZIPs, the paper PDF, a release hash manifest, a handoff guide, and the optional encompassing handoff ZIP. Existing named archives are protected unless `--overwrite` is explicitly supplied. A final PDF already at its intended release path is accepted as the build input.

The full archive includes all 217 historical files, the new extension, source metadata/data, prepared arrays, predictions, result JSON, freezes, analyses, logs/reports, manuscript sources/assets, and this verification script. Other ignored experiment runs, temporary manuscript builds, and interpreter caches are excluded. `--extra-log PATH` can include a specifically named external execution log without changing its contents.

For durable storage of the completed reference run in Git, build its compact archive after all final reports are present:

```bash
python extensions/natural_forecasting/package.py build-reference
```

The default destination is `extensions/natural_forecasting/reference/natural_v1.zip`. The builder retains this ordinary full ZIP locally and also writes `natural_v1.zip.part-001`, `natural_v1.zip.part-002`, and as many further parts as needed, plus `natural_v1.zip.parts.json`. **Commit only the parts and their manifest.** The full ZIP and temporary reference-build directories are ignored by Git. The run archive contains only the completed run and its payload manifest, without a second copy of the source repository. An explicit `--overwrite` replaces a previous generated set; stale parts are removed only when the prior manifest identifies them and their bytes still match it. Unrelated files are preserved.

The full release includes expanded run files and excludes the entire reference-storage directory to avoid recursive inclusion. Committing the reference parts is a separate repository operation; the packager does not commit or push files. The manuscript PDF belongs in the release output, not in the Git-backed reference archive.

## Scientific and resource interpretation

The matched primary pair is raw-history RBF versus raw-history-plus-KAHM RBF. Additional affine/locality and raw-history controls distinguish gate choice and access to the original history. The finite RBF/Nyström head uses an ordinary Euclidean ridge penalty. The small and medium settings cap examples and model size; they are **not equal realized wall-time budgets**. The modest declared tuning grids do not establish state-of-the-art performance.

Task MSE is averaged across horizons and recorded origins, then divided by a fixed training-prefix seasonal-difference mean square, with the declared first-difference fallback. Source families receive equal weight, followed by their nested task levels. Related series, overlapping targets, two horizons, capacities, and seeds do not enlarge the number of independent source families. Four reserved families support bounded descriptive evidence and sensitivity analysis, not broad-population significance claims.

Cost records distinguish wall time from process CPU time. The scientific-pipeline accounting includes data/checksum preparation, loading, candidate fitting/validation, selection bookkeeping, final refitting, forecasting, and model serialization. Libraries are initialized before measured child execution; initialization, launch/IPC, and launch-to-result timing are separately reported. Policy costs sum unique required nodes: validation selectors pay all searched methods but only the selected final refit/forecast; diagnostic policies pay their required features/probes even when KAHM is rejected.

The initial investment in a learned diagnostic also includes acquiring its development labels through the matched raw/KAHM forecasting pipelines and its required feature/probe records. That acquisition, the small regression fit, online policy cost, and the full research benchmark's elapsed time are separate scopes. No deployment count is invented to amortize the investment. Summing standalone task costs is conservative when reuse across seeds/capacities could otherwise be implemented.

Model bytes and process high-water RSS are recorded. RSS includes interpreter/inherited arrays and earlier process history; it is not incremental or additive memory and does not reconstruct exact hypothetical policy memory. CPU timing does not measure GPU cost, energy, or monetary expense. A numerical fallback, failed trial, warning, or whole-worker failure remains visible; only actually available timing evidence can be reported for a failed worker.

Short-source and availability limits matter: the longer Hospital/Traffic regimes have 21/29 initial training windows, and NN5 T88's longer regime has seven test origins. Context and horizon vary together, so their separate effects are not identified. The FRED-MD input is a fixed-vintage archive whose metadata describe upstream transformations without a complete per-series processing audit. Train-only guarantees apply to this extension's handling of the supplied snapshot, not unverified upstream transformations, historical revisions, or release delays. See the protocol for the full scope.

## Files and CI

| Path | Purpose |
|---|---|
| `data.py`, `data_sources.json`, `datasets/` | Fixed public snapshots, provenance, selection, and chronological windows |
| `models.py`, `baselines.py` | Standalone geometric pipelines and practical controls |
| `run.py`, `protocol.json` | Frozen orchestration, search, measurements, and execution gates |
| `analysis.py` | Source-balanced diagnostic analysis and required-cost policies |
| `report.py` | Post-study numerical summaries and manuscript assets from saved results |
| `registration/` | Public freeze/index records and validation provenance |
| `runs/natural_v1/` | Delivered prepared arrays, saved predictions, results, and analyses |
| `reference/natural_v1.zip.parts.json`, `reference/natural_v1.zip.part-*` | Git-backed multipart copy of the completed reference run; full ZIP retained only locally |
| `manuscript/` | LaTeX, bibliography, official style, and generated assets |
| `package.py`, `package_tests.py` | Post-study archive builder, verification/replay, and integrity tests |

The new workflow `.github/workflows/natural-forecasting.yml` runs the synthetic scientific suite, packaging integrity tests, source-hash checks, and committed-archive verification. When multipart reference files are present, CI verifies them, restores the run into a source copy under `RUNNER_TEMP`, and performs the offline numerical audit there. It never launches the full forecasting benchmark. It builds the manuscript when LaTeX is already available; a manually requested optional job can install the LaTeX packages and validate the paper build. The historical CI workflow is preserved unchanged.
