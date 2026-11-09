# Evidence availability and source identity

This directory contains the exact primary-summary CSV, its arithmetic checker, the experimental protocol, and recorded source and output identities. It supports the summary check described in [README.md](README.md).

The [saved-loss bundle](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02/reproduction/score_enrichment) contains 17 original evidence records and an independently implemented statistical checker. It reconstructs 416 procedure summaries, 2,560 paired differences, and all 40 primary comparisons from 26,624 selected-loss rows. The selected-loss table is losslessly compressed; the other records retain their original bytes.

The original generator, experiment runner, complete source snapshot, `cohort.json` evaluation roster and random-stream labels, fitted model states, forecast arrays, and original environment fingerprint are unavailable here. The saved-loss checker cannot verify raw forecast errors, the entire validation search, or the reported model-replay audit. It does not supply full experimental regeneration.

The recorded source-manifest SHA256 is `af4357d382f23a58797cc111f73eca9cb6e8b283415ed31f3cd586af2fe4ba28`. The original records refer to a compact execution package and sixteen numerical archives totaling 9.024 GiB. Their identifiers specify expected materials; they are not public download links. The complete numerical archive set is unavailable for independent size and integrity checks.

The archive volume is a practical distribution constraint, separate from the missing source, evaluation roster, and environment. Complete execution inputs could support regeneration without distributing every fitted model. [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) gives the available computational routes and the recorded software versions.
