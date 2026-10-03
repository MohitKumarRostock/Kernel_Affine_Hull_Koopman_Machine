# Confirmatory dynamical representation: verified evidence

This directory retains the verified Van der Pol KAHM representation frozen for
the confirmatory dynamical certificate campaign.

The representation was derived prospectively from the previously verified
state-space K-means centers and KAHM/OTFL autoencoders. The association map uses
`omega = 128`, and the NLMS operator was refit from the original training
trajectories only. No confirmatory certificate-evaluation pairs were generated
during representation construction or verification.

## Provenance

- representation-build execution commit: `4a0cedd557a6e06d331aa31b91223629c36d1997`
- independent numerical-verifier commit: `cbfe43808ac54d76fb26a35e5361a634e526745c`
- confirmatory configuration SHA-256: `1b1a6ec9e0d3de92f43c0f78e654f39d44ac6fd1d2d4286b88168027bd475a91`
- confirmatory protocol SHA-256: `daec3dc5bb96e115ce6c0bf82e64e7497318796b9cdd1dc80557b691f46b071c`
- original build-evidence `SHA256SUMS` fingerprint:
  `76b6cd0a2b2369b6cd2e2468fe3d2e06a5ed33e4eb172c84372196c2e4530372`
- verification-report SHA-256: `25bbf7a326e6a9e6f10a459cca4a13aebe166d52335ce12efea17f037f60efbf`
- verification-checksums SHA-256: `10e6698b6c2d3d7995b72e1bc326cddb5f00f80b458947e778d89852b4af7447`
- compressed evidence archive SHA-256: `17fe746da6eddcc194da9fe19dedf7a80d3d887da397b103ae5e5f1f4718bcbc`

The build execution commit and the later verifier/archive commits have distinct
roles and must not be substituted for one another in provenance claims.

## Verification result

The independent verifier reported `status = passed`. It regenerated all three
training trajectories and all 3,600 training snapshot pairs, recomputed both
training association matrices at `omega = 128`, and reran the 20-epoch NLMS
recursion. The recomputed operator matrix and complete NLMS history matched the
retained representation exactly in binary64 arithmetic.

It also verified that:

- the 25 autoencoder shards are byte-identical to the previously verified source
  abstraction;
- both stored cluster-center arrays are unchanged;
- no state-space K-means refit occurred;
- no autoencoder refit occurred;
- no certificate-evaluation pair was generated;
- the retained representation was not modified by verification.

## Archive contents

`evidence.tar.gz` contains exactly 58 regular files:

- `56` files from the retained representation-build evidence,
  including its own `SHA256SUMS`;
- `2` authoritative verification JSON artifacts.

Launcher console logs are intentionally not included in the scientific archive.
The verification console happened to reproduce the verification report on
stdout, but the authoritative verification artifacts are the two JSON files
under `verification/`.

`ARCHIVE.json` records every archived member's byte size and SHA-256 digest.
This directory's `SHA256SUMS` covers `evidence.tar.gz`, `ARCHIVE.json`, and this
README.
