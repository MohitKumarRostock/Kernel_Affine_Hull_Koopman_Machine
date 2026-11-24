# Reproduction and verification

## Environment

The supplied executable studies were generated with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and pandas 2.2.3 (Linux x86-64). The numerical scripts limit common BLAS and OpenMP thread counts to one; the synthetic generator uses four spawned processes. Platform details are recorded in `ablation/run_metadata.json` and `real_data/metadata.json`. The archived saved-loss checker was separately validated with Python 3.12 and its pinned NumPy and SciPy dependencies. Package versions do not identify every platform, compiler or numerical-library detail. Identical floating-point results across arbitrary platforms are not assumed.

From the distribution root, install the pinned dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No external dataset download is required. See [Data provenance](docs/DATA.md) and [Experimental protocols](docs/PROTOCOLS.md).

## Verification levels

| Command | Computation | Interpretation |
| --- | --- | --- |
| `python reproduce.py verify` | Complete distribution inventory, byte counts and SHA-256 identities in `MANIFEST.json` | Integrity only |
| `python reproduce.py check` | Integrity, population formula checks, archive statistical recomputation, capacity-summary arithmetic, synthetic record summaries, observed-series arithmetic and validation-grid selection checks | Numerical consistency without model fitting |
| `python reproduce.py replay --output runs/replay` | Regenerates the synthetic and observed-series experiments, checks their outputs, compares deterministic record hashes and numerical agreement | Computational regeneration of the executable studies in a separate directory |

The root driver runs the checks on temporary copies and leaves distributed evidence unchanged. The replay command requires a new output directory and records subprocess output and a JSON report. It reports exact hash agreement and, if bytes differ, checks CSV schemas, row order, categorical fields, identifiers, counts and ranks exactly. Declared floating-point fields use relative tolerance $10^{-10}$ and absolute tolerance $10^{-12}$. NPZ array names, shapes, types, identifiers, dates, targets and origin levels must agree exactly; predicted changes use the same numerical tolerance. A failure of numerical agreement fails the replay. Environment and metadata equality are reported separately. Exact hash differences alone do not establish scientific disagreement. Use the generated records and independent arithmetic checks to investigate discrepancies. No comparison uses elapsed runtime as an equality requirement.

### Expected checks

- Matched-dimensional formulas: 1,350 cases and 54 optimal-contrast checks.
- Weighted recovery: 300 cases; inverse-stability trade-off: 270 cases.
- Earlier forecast archive: 26,624 saved loss rows, 416 summaries, 2,560 paired differences and all 40 primary comparisons. The original execution workflow is unavailable.
- Supporting capacity evidence: 12 reported summary comparisons (four practical benefits, six harms, two unresolved); no fitting or repetition-level recomputation.
- Fixed-geometry and target-changing simulations: 9,472 and 12,384 loss records; 256 and 512 diagnostic records; all 12 adjusted intervals.
- Atmospheric CO2: 212 model-evaluation records, 390,504 forecast values, 2,544 annual records, 1,050 validation scores and 23 diagnostics.

The replay comparison uses the five deterministic ablation CSV files and five observed-series numerical output files (including the NPZ forecast array). The observed-series metadata is also compared and reported. The synthetic environment metadata records runtime and is not expected to be byte-identical. The original deterministic replay receipts are included in each experiment directory. The archive checker validates 17 evidence records, including compressed and decompressed loss identities; it verifies saved statistics conditional on the recorded losses.

The inventory check rejects additional distributed or tracked files. Local `.venv/`, `runs/`, `.git/`, Python cache directories and pytest caches are excluded from the distribution inventory; none may be tracked as scientific evidence. Git checkout byte identities are preserved by `.gitattributes`.

## Component-level execution

The `scripts/check_matched_dimension.py` and `scripts/check_weighted_recovery.py` scripts calculate the specified finite-law and perturbation examples directly. The latter also reads the archived rank diagnostics. The executable generators and summary scripts are described in [ablation](ablation/) and [real_data](real_data/). Their direct invocations can overwrite generated files in their own directories; run them on a separate copy, or use the root driver. The archived checker supports an explicit new output directory:

```bash
python reproduction/score_enrichment/check_saved_losses.py --output runs/archive-check
```

The separate `python scripts/check_capacity_summary.py` command reconstructs the 12 capacity-study interval endpoints and classifications from reported summary statistics. The [supporting study descriptions](docs/RELATED_EVIDENCE.md) identify additional results and their evidence limitations.

The historical archive contains no complete original model-fitting workflow. Replaying the newer experiments with fixed random streams does not regenerate that earlier cohort, establish independent scientific replication, or convert approximate confidence intervals into distribution-free guarantees.

## Scope of validation

The mathematical checks use ordinary floating-point arithmetic. They are not formal proof verification or verified interval calculations. The real-series check recomputes assessment and annual losses from stored forecasts, reconstructs eligible dates and targets, and checks selection against the recorded validation grid. It does not recalculate the complete fitting process without replay. The synthetic record checker reconstructs its declared interval calculations and verifies rosters and matched-df criteria; the replay additionally performs simulation and fitting. Content hashes identify records but do not establish validity of the assumptions, execution chronology, or correctness of the underlying mathematical claims.
