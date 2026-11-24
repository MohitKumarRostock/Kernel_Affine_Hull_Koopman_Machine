# Predictive Enrichment of Geometry-Induced Soft-State Abstractions: Information Retention, Predictive Accessibility, and Stability

Scientific software, mathematical specification, and computational evidence for reconstruction-based forecasting.

## Scientific scope

Local reconstruction models represent an observed history by relative reconstruction affinities. Normalizing these affinities removes their common scale; weighted local coordinates retain additional position information. The central question is whether two information-equivalent encodings expose the same predictive functions to a restricted decoder. Kernel Affine Hull Machine (KAHM), positive-kernel, affine, and center-only reconstruction scores are compared using fixed reference geometries. The experiments predict future observations directly; they do not assume Markov sufficiency or fit an autonomous Koopman transition model.

The [scientific reference](docs/SCIENTIFIC_REFERENCE.md) defines the reconstruction, normalization, representations, population-risk results, and their assumptions. The [experimental protocols](docs/PROTOCOLS.md) specify the data-generating laws, sample allocations, reference fitting, decoder selection, metrics, and inferential units. [Results](docs/RESULTS.md) presents the principal numerical findings and their scope.

## Computational evidence

| Investigation | Available computation | Evaluation scope |
| --- | --- | --- |
| Matched-dimensional finite-law example and weighted recovery | Deterministic mathematical checks in `scripts/` | Exact population formulas checked numerically; no fitted-forecast inference |
| Fixed-geometry forecasting | [Executable simulation and fitting](ablation/) | 32 independent repetitions per law; 12 adjusted paired comparisons |
| Target-changing alignment | [Executable simulation and fitting](ablation/) | 32 independent repetitions; descriptive excess MSE relative to known conditional means |
| Chronological atmospheric CO2 forecasting | [Executable observed-series study](real_data/) | One series, three horizons, five reference fits; descriptive comparisons |
| Archived forecasting experiment | [Saved-loss analysis](reproduction/score_enrichment/) | 64 repetitions per law; 40 separate adjusted comparisons; original training workflow unavailable |
| Supporting decoder-capacity and related evaluations | [Related evidence](docs/RELATED_EVIDENCE.md) | Distinct protocols; capacity-summary arithmetic and reported contextual findings |

The executable simulations and chronological study are the principal empirical evidence. Their results show that membership weighting is beneficial for some aligned tasks and detrimental for others, even when the weighted and separated encodings preserve identical information. Under the tested fixed slope effective degrees of freedom of eight, weighting has higher observed mean forecasting loss in all 12 simulated law–reconstruction settings. On the historical CO2 series, separated positions have lower aggregate loss in all 20 reconstruction-method/reference-seed combinations under both tested penalty rules. These findings do not establish universal superiority of any reconstruction or encoding.

The [archived forecasting specification](docs/ARCHIVED_FORECASTING.md) describes the earlier cohort, including its numerical conventions and inferential scope. The [related evidence](docs/RELATED_EVIDENCE.md) identifies the supporting studies and their distinct comparison criteria. These are not pooled with the principal executable investigations.

## Installation and verification

Use Python 3.13 with the specified numerical dependencies. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py check
python reproduce.py replay --output runs/replay
```

`verify` checks content identities with the standard library. `check` recomputes mathematical tests, saved-loss statistics, ablation summaries, and observed-series assessment arithmetic without retraining. `replay` refits the executable studies in a new directory, checks their generated records, and compares them with the distributed reference records. The commands do not modify the reference data or require LaTeX, a separate document, an external dataset download, or the shared numerical library. See [Reproduction](REPRODUCIBILITY.md) for precise commands, expected counts, numerical tolerances and reproducibility boundaries.

## Licensing and attribution

Author-owned software uses the [MIT License](LICENSE); original research outputs and non-code material use [CC BY 4.0](DATA_LICENSE), where applicable. The historical CO2 snapshot is third-party public-domain data; [data provenance](docs/DATA.md) identifies its source and terms. Source and data identities are recorded in `MANIFEST.json`. Hashes establish content identity, not scientific validity or execution chronology. [Computational provenance](provenance/README.md) documents execution records and the scope of AI assistance.
