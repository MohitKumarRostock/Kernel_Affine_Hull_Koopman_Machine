# Verification records for experimental evidence

Each record identifies the files, environment, and computation that it checks. A recorded hash binds a claim to specific bytes.

| Record | Verification scope |
| --- | --- |
| `independent_recheck.json` | Independent-code evaluation of saved evidence without refitting a forecasting model |
| `contract_tests.log` | The 38 numerical, data-partition, and statistical-analysis tests |
| `transport_verify.json` | Hashes of all 77 lossless parts and 1,631 archived members |
| `release_check.json` | Recorded archive extraction checks and replay of 1,920 selected prediction arrays |
| `NUMERICAL_REPLAY_PORTABILITY.json` | Targeted inference comparison across numerical-kernel configurations |
| `LOCAL_EXECUTION_PROVENANCE.json` | Execution Git objects and scientific source checkpoint identities |

The numerical portability record reports a failed strict replay when AVX512 dispatch is disabled: 49 of 4,992 raw-RBF predictions exceed the original pointwise tolerance, with a maximum absolute difference of approximately `1.07e-10`. Statistical reanalysis from the recorded forecast arrays and inference from fitted objects therefore have different portability requirements.

Local source identities do not establish public preregistration. Scientific source, predictions, statistical decisions, and audits retain their separate recorded hashes.
