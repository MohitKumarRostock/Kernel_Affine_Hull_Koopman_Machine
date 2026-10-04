# Confirmatory dynamical certificate: verified manifest capsule

This directory records the completed confirmatory Van der Pol certificate
campaign and its separate independent numerical verification.

## Important scope

`manifest_capsule.tar.gz` is deliberately a **manifest capsule**, not a copy of
the full raw campaign evidence.

The completed raw campaign contains
`346` physical files and
`649752423` bytes. Duplicating that approximately 620 MiB
payload directly into Git would unnecessarily bloat the repository.

The capsule therefore retains the compact authoritative campaign records, the
campaign's top-level `SHA256SUMS`, and both authoritative independent-
verification JSON files. The verifier checksum record contains the complete
344-file checksum map for the verified
campaign evidence.

This package does **not** claim that omitted raw arrays are present here. A
separate full-data distribution should use the exact raw-evidence identity
recorded below.

## Provenance

- campaign execution commit: `18dc9f3b965e2cc634764e05785ff60d5fbd6422`
- independent numerical-verifier commit: `090eac5fa2a28879378d165c898a875f11573aac`
- confirmatory configuration SHA-256: `1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91`
- confirmatory protocol SHA-256: `daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c`
- full raw-evidence `SHA256SUMS` fingerprint:
  `df9b1b7abd4a4c38b5bf8d0a5f43d7f6cb299ed23c070960b506f52641243221`
- independent verification-report SHA-256:
  `7e36ececa64e0d27b6621814c9516cb85d5abd6d80429692e41543cfe6cd1ab3`
- independent verification-checksums SHA-256:
  `6daedda18d164dcbffed4b1ce6d2aee19db7fac078c23330ea91ee2a2d27a7a3`
- compact capsule SHA-256: `d1c17da83327ab58ad5d31e6ec7b98512f84dbf1fd5205484d9dcae4f161ea20`

The execution commit and later verifier commit have distinct provenance roles
and must not be interchanged.

## Verification result

The independent verifier reported `status = passed`.

It verified all `64` confirmatory attempts,
`1,048,576` retained fresh pairs, and
`2,097,152` seed streams. It independently reconstructed all
`1,048,576` retained time-index draws and recomputed all
`64` KAHM association datasets, hard reference labels,
certificate sufficient statistics, frozen-predictor errors, and all
prespecified certificate bounds.

No random dynamical pair dataset was regenerated, no representation was
refit, and the retained campaign evidence was not modified during
verification.

## Capsule contents

The compressed capsule contains exactly 13 regular files:

- 11 compact authoritative files from the completed campaign, including its
  full `SHA256SUMS` inventory, configuration, protocol, schedule, compact
  sample/result indexes, campaign metadata/summary, representation provenance,
  and final source snapshot;
- 2 authoritative independent-verification JSON files.

The capsule omits `335` large/raw campaign files totaling
`649451837` bytes. Their exact byte identities remain covered by the
campaign `SHA256SUMS` and by
`verification/verification_checksums.json`.

Launcher console logs are intentionally excluded. They are not authoritative
scientific evidence.

`ARCHIVE.json` records every capsule member's byte size and SHA-256 digest and
also records the identity and size of the full raw evidence. This directory's
`SHA256SUMS` covers `ARCHIVE.json`, `README.md`, and
`manifest_capsule.tar.gz`.
