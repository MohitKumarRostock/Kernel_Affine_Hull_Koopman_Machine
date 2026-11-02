# Independent computational verification

`verify_soft_state.py` audits the complete, fixed 256-task joint dimension–scaling cohort. It imports no production model, fitting routine, scientific analyzer, or physical-law generator. `independent_oracle.py` reconstructs numerical quantities from portable fitted states and retained arrays.

## Coverage

- Frozen source identities, the recorded publication receipt, seed exclusions, stage ordering, complete task rosters, and sealed input/state/forecast bytes.
- Every history and response window from retained base episodes, independently regenerated Gaussian noise streams, retained row indices, and the A-derived noise amplitude. Physical-law integration and partition k-means fitting are not independently reimplemented.
- All A-fitted input scaler statistics and all three query transformations: standardization, unclipped min–max, and standardized tanh without post-map rescaling.
- Reference eligibility, native local rank/range semantics, PCA spectral identities, whitened kernels, and KAHM inverse identities.
- Shared gates, position, and reconstruction blocks; their A-only column transforms and separate radial bounds; exact saved prefix identities across G/P/K/R.
- Every candidate's complete B penalty path, using independent Cholesky normal-equation ridge solutions rather than the production SVD. All per-horizon penalty choices and all joint/standard-only candidate choices are recomputed.
- Every selected C and D forecast in the four declared views. C remains diagnostic-only. Joint and standard-only forecasts reuse the same candidate fits.
- Repetition aggregation, the complete six-contrast Bonferroni family, the declared descriptive paired comparisons, and exported selected-scaler/dimension/feature tables.

The report states both numerical tolerances and exact hash checks. Numerical replay agreement is checked with predeclared `rtol=2e-7`, `atol=2e-9`; tighter identity checks are used where algebra is identical. The report records maximum differences, rather than replacing them with a generic success statement.

## Artificial preflight

`../code/preflight.py` tests the implementation using manufactured arrays and a tiny explicitly artificial staged cohort, including all three scalers and both candidate caps 1 and 2. It rejects attempts to generate terminal data before the global fit seal, rejects target release before the global prediction seal, checks changed-source rejection, and independently replays every artificial task after completion. Separate metadata-only publication-receipt tests use visibly artificial Git identities; they do not execute production preparation or claim remote publication.

Artificial preflight outputs are software checks, not study observations. The production audit refuses to label an artificial fixture as completed scientific evidence. Its optional `--fixture-task` mode is permitted only when `manufactured_fixture` is explicitly present in that fixture's configuration.

This is independent computational verification within the same research workflow. It is not independent scientific replication, external preregistration, or cryptographic target isolation. It does not establish distribution-free validity of the Student approximation.
