# Frozen dynamical representations: pilot v1

This directory retains the frozen Duffing and Van der Pol KAHM representations used by the dynamical certificate pilot, together with the separate independent verification report produced before certificate evaluation.

The compressed archive contains two top-level directories:

- `evidence/`: provenance-locked representation-build evidence, including the portable representation artifacts and build checks.
- `verification/`: the independently generated verification report and its checksum record.

The representations were generated from execution commit `1add19b94372257d16118f60c69e1e8afa668259`. The later verifier ran from commit `65db148f865c94f988676381c6b5f12222356b31`. These commits have distinct roles and must not be interchanged in provenance claims.

The verifier reported two representations, zero representation refits, zero regenerated random samples, zero certificate-evaluation pairs, and no modification of the retained representation evidence.
