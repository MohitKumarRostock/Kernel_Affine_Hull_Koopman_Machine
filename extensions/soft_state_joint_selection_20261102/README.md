# Joint input scaling and native dimension selection

This separate follow-up extends the current G/P/K/R representation-only study
with A-fitted input scaling selected jointly with native dimension using B.
It preserves the completed predecessor experiment unchanged.

**Current status: implementation and manufactured-data validation complete;
scientific source frozen; publication and production evaluation pending.**
No new forecasting results are claimed. The final findings will be recorded
here after the fixed cohort and independent computational audit finish.

Ten model checks pass, including bitwise standard-only agreement with the
published predecessor, A-only scaling, independent validation selection,
native cap aliases, and portable-state replay. The staged artificial audit
passes 39,324 checks across eight miniature tasks. The statistical preflight
passes 42 checks. These checks use manufactured data and are not evidence of
forecasting improvement.

The full design is in [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json).
The 15 geometry candidates combine standardization, min-max, and tanh with
native dimension caps 1, 2, 4, 8, 12. Every candidate fits all four arms and the
same five ridge penalties. Shared and independent joint selection are compared
with the corresponding standard-only selections on the same fresh cohort.

The six primary comparisons include the four structural questions and paired
joint-versus-standard KAHM comparisons. No C or D outcome participates in tuning.

Reproduction uses `code/pipeline.py` in the protocol's staged order, followed by
`analysis_code/analyze_soft_state.py` and the independent audit. A complete
publication receipt and source freeze are required before production generation.
Generated numerical records are retained separately from source and indexed
with hashes when the run completes.
