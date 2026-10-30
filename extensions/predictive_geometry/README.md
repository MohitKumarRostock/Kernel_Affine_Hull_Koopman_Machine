# When Kernel Affine Hull Geometry Helps Prediction

**Local Coordinates, Decoder Capacity, and Conditional Gains**

This extension accompanies a focused study of when KAHM geometry helps a predictive model. It includes
matched KAHM/affine representations, controlled target laws, forecasting tests,
a fresh capacity follow-up, and a numerical underflow sensitivity check.
The manuscript is prepared in anonymous TMLR submission format. This statement
does not assert submission, acceptance or publication.

The contribution is a testable account of **conditional predictive gains**.
The experiments retain favorable and unfavorable outcomes, use matched
reference observations and PCA bases, and include raw-input baselines. They do
not establish that every task benefits from KAHMs or that better reconstruction
implies better prediction.

Read the [LaTeX source](manuscript/main.tex),
[TMLR packaging guide](TMLR_SUBMISSION.md), and
[paper readiness assessment](PAPER_READINESS.md). Tables, figures, and headline
number macros are generated from archived records; supporting numerical prose
is checked against the same source records. The readiness assessment separates
completed reproduction work from the additional evidence that would strengthen
a journal submission.

## Quick start

Use Python 3.12 and a Linux environment. From the extension directory or an
extracted reproducibility supplement:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m reproduce smoke
```

The smoke command runs provenance and loss-audit tests, the unchanged geometry
tests, numerical arithmetic checks, reduced fits for all three forecasting
systems plus branch and Airfoil data, both full archived primary analyses, and
the paper asset builder. Its reduced fits are execution checks and are never
used as research findings.

**Regenerate every experiment, audit the results, and rebuild the paper assets:**

```bash
python -m reproduce full --workers 4
```

Every invocation creates a new directory under `runs/`. Full execution includes
the 14 pilot jobs, 168 first-stage confirmation jobs, 24 Airfoil fits, 72
capacity-follow-up jobs, and 24 numerical-sensitivity jobs. It retains fitted
banks and predictions in that run directory for auditing. Generated run files
are excluded from Git.

To regenerate tables and figures immediately from the small committed archive:

```bash
python -m reproduce tables
```

To verify archival hashes and independently recompute all 29 first-stage and
12 capacity contrasts:

```bash
python -m reproduce audit
```

To compile the manuscript after rebuilding its assets, install `latexmk` and a
LaTeX distribution with the packages used in `manuscript/main.tex`, then run:

```bash
python -m reproduce paper
```

The historical release check in [VALIDATION.json](VALIDATION.json) records successful runs
of all five routes. The complete local rerun reproduced all 15,234 saved method
MSEs exactly and passed the required prediction audits. It took about 194 seconds
with four workers in the recorded environment; the smoke route took about
35 seconds. These are observed timings for that environment, not a performance
guarantee for another machine. It is historical evidence for the earlier release,
not a validation claim about later edits. The TMLR revision records its actual
checks separately in `TMLR_VALIDATION.json`.

To check the archived paired differences and build the two submission ZIPs while
preserving the manuscript's committed diagnostic assets:

```bash
python -m reproduce review-diagnostics --output runs/review-diagnostics
python -m reproduce submission --output-dir dist/tmlr
```

That diagnostic command checks paired outcomes only. To regenerate the
manuscript's additional saved-bank rank and gate checks, pass the directory
of a completed full run explicitly:

```bash
python -m reproduce review-diagnostics --run-dir runs/full-review --output manuscript/generated
python -m reproduce paper
```

Here `runs/full-review` must first be created with
`python -m reproduce full --workers 4 --run-dir runs/full-review`.

The package command verifies the historical hashes, includes the complete
executable study and compact result archive, and writes deterministic ZIPs with
per-file hashes and content-tree digests. It excludes Git history, prior named
PDFs and first-party author metadata. Required license notices, ordinary
published citations and immutable upstream provenance remain; these narrow
exceptions are listed explicitly in each package manifest. See
[TMLR_SUBMISSION.md](TMLR_SUBMISSION.md) for the review-view contract.

The Python experiment dependencies are pinned in `requirements.txt`. LaTeX is
needed only to compile the manuscript PDF; experiment execution and asset
generation do not require it. This is a source-checkout research bundle, not a
standalone PyPI wheel.

## What is preserved

| Path | Purpose |
|---|---|
| `study/` | Exact historical scientific source snapshot, vendored upstream code, configs, data, and protocol evidence |
| `study/SNAPSHOT.json` | SHA-256 digests of the original uncompressed source and evidence bytes |
| `archive/stage1/` | Full method records, mechanism diagnostics, frozen margins, and 29-comparison analysis |
| `archive/capacity/` | All 648 capacity records, frozen margins, and 12-comparison analysis |
| `archive/numerical_sensitivity/` | All 96 robustness records and numerical audit |
| `archive/MANIFEST.json` | Compressed-file and decompressed-content hashes |
| `reproduce/` | New portability, execution, and auditing code, versioned separately from the scientific snapshot |
| `scripts/build_paper_assets.py` | Source-derived manuscript table and figure generator |
| `scripts/package_tmlr_submission.py` | Deterministic manuscript/reproducibility ZIPs and review-view integrity verification |
| `tests/` | Provenance, run-isolation, completeness, and independent-loss checks |
| `ci/kahm-predictive-geometry.yml` | Source of the workflow installed at the parent repository root |
| `runs/` | New execution manifests, generated states, predictions, audits, and assets; never committed |

Large model states and prediction archives are intentionally regenerated by the
full command. No network access is required after dependencies are installed:
the small Airfoil data file and the exact upstream source files are included.
Data and upstream attribution are in `THIRD_PARTY_NOTICES.md`.

## Reading the evidence

The first stage compares matched reconstruction maps and feature paths. All
geometries share training scaling, clustering, actual local references, and
effective PCA bases. The KAHM and affine contrast uses a common folding score.
Its scope is therefore a controlled reconstruction comparison, not a complete
end-to-end optimization of every model family.

The capacity follow-up uses distinct phase-4 random streams and two specified
Nyström decoder budgets. Its 12 primary comparisons form their own confidence
family; they do not combine with the first stage into a single 95% statement.
The validation selector is a secondary complete-pipeline result.

The numerical robustness check reuses the capacity samples. It rescales each
Gaussian query-kernel row before signed-weight normalization, refits the heads,
and retunes on validation data only. This is descriptive sensitivity evidence,
not a third independent confirmation. The raw implementation is retained for
exact historical reproducibility; the stabilized query routine is available
in `study/numerical_sensitivity.py`.

Re-running published seeds is **computational reproduction** of the existing
evidence. It does not produce additional independent statistical replications.
See [REPRODUCING.md](REPRODUCING.md) for the complete execution and provenance
contract.
