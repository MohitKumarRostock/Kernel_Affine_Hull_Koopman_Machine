# Dynamical certificate pilot: verified evidence

This directory retains the completed frozen-representation dynamical certificate pilot together with its separate independent numerical verification.

The compressed archive contains two top-level directories:

- `evidence/`: the complete provenance-locked pilot campaign, including all 96 sampled/evaluated replicate datasets, retained raw physical-state pairs, seed metadata, association coordinates, sufficient statistics, certificate outputs, frozen model source, preflight checks, and campaign checksum inventory.
- `verification/`: the independently generated verification report and the complete checksum record for the verified campaign evidence.

The pilot executed from commit `4cda98e027cdc6b820e37f13869155e6f35956a8`. The later independent verifier ran from commit `615b017d6d6e480cd6547982851747d8dbc7f432`. These commits have distinct roles and must not be interchanged in provenance claims.

The verifier checked 96 attempts, 86,016 retained independent pairs, 172,032 seed streams, and 86,016 independently reconstructed time-index draws. It recomputed all 96 retained KAHM association datasets, all certificate sufficient statistics, predictor errors, and all certificate bounds from the retained raw pairs and committed frozen representations.

No dynamical pair dataset was regenerated, no KAHM representation was refit, and the retained campaign evidence was not modified by verification.

The campaign's top-level `SHA256SUMS` inventories 487 files. The physical campaign directory contains 489 files because checksum files named `SHA256SUMS` are intentionally excluded from that manifest: the top-level campaign manifest itself and the nested frozen-representation evidence manifest.

Launcher console logs and failed pre-launch diagnostics are not included in this scientific archive; the authoritative successful verification artifacts are the two JSON files under `verification/`.
