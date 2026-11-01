# Cross-fitted residual study: computational reproduction

This directory reconstructs the same declared simulation cohort using the
published source at `f36e7c797f37d7d073dae42603b3770543d9fb43`. The original
completed numerical records were unavailable when this reconstruction was
prepared. Prior reported aggregate results and the original verifier failure
were known. This execution is a computational reproduction, adds no independent
repetitions, and cannot authenticate the original run's bytes or historical seals.

[PROTOCOL.md](PROTOCOL.md) defines the current execution scope.
[ORIGINAL_SOURCE.json](ORIGINAL_SOURCE.json) records source identities and is
bound by a hash in `config.json`. Executable model, pipeline, numerical solver,
scientific analyzer, and manufactured cross-fit oracle bytes match the original
published source. The full original scientific and numerical-amendment protocols
are preserved under `audit/` as explicitly historical documents.

The only configuration differences are removal of the historical
`numerical_amendment` field and addition of `computational_reproduction`.
The latter preserves the complete removed historical metadata and source hashes.
All scientific settings and seeds are unchanged. The pipeline continues to
install the same checked numerical solver in every fitting worker.
The actual reproduction publication commit is recorded separately after source
freeze and publication. The unchanged `source.publication_commit` field is
historical lineage metadata. Although new execution gates precede regenerated
data and scoring, this reproduction follows knowledge of earlier reports and
is not a new prospective preregistration.

## Execution

Keep this directory beside `dimension_forecasting` and
`preprocessing_dimension`. The published `crossfit_residual_v2` directory is
also required by the independent reproduction provenance audit. Install the
pinned requirements. A new source freeze is required before generating data.
Do not reuse an existing preparation, stage lock, or fitted-state directory.
From this directory the staged commands are:

```bash
python code/pipeline.py freeze
python code/pipeline.py abc
python code/pipeline.py fit --workers 3
python code/pipeline.py seal-fits
python code/pipeline.py d
python code/pipeline.py predict --workers 3
python code/pipeline.py seal-predictions
python code/pipeline.py release
python analysis_code/analyze.py --root .
```

The same 192 task suites are newly fitted. All fits precede D regeneration;
all target-free forecasts precede release and scoring. Unexpected failures
stop execution and remain in the audit trail. New artifacts are never silently
overwritten. The complete independent reproduction audit is run after analysis,
using its separately recorded verifier source and output receipt.

## Interpretation

The retained scientific design contains 24 original simulation repetitions,
four laws, two noise levels, history width 12, and horizons 1/4/12. Each of the
192 task suites retains 42 forecast arrays. Neither 192 tasks nor 8,064 arrays
are independent inferential units. Reusing the same streams does not create a
second independent sample. The original four simultaneous primary decisions
are preserved and reported separately; no favorable comparator can substitute
for another failed primary claim.

The reproduction can establish numerical performance and verification outcomes
for its own preserved execution. It cannot recover unavailable original
manifests, fitted states, forecasts, audit receipts, or execution chronology.
The original unhandled ridge SVD failure and singleton-PCA verifier defect
remain distinct historical events. New solver fallback and audit outcomes are
reported as new execution evidence. The scope remains the declared synthetic
laws and finite candidate menus, with no natural-data adoption or risk-certificate
claim.
