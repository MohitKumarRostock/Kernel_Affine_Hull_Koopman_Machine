# Sources, licenses, and data attribution

The vendored KAHM and forecasting files in `study/vendor/upstream/` are unchanged
from MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine, commit
`d6d74d1953a4310b73065a556df816e5ab5e3ebc`. Their MIT license is retained at
`study/vendor/upstream/LICENSE`. `study/upstream_manifest.json` records every
upstream path, source URL, Git blob identifier, and SHA-256 digest.

The Airfoil Self-Noise data are attributed to T. Brooks, D. Pope, and
M. Marcolini (1989), UCI Machine Learning Repository,
<https://doi.org/10.24432/C5VW2C>. UCI identifies the dataset license as
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The preserved numeric file has 1,503 rows and SHA-256
`74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`.
`study/DATA_NOTE.txt` documents the pinned mirrors, byte normalization check,
grouped split, and limitation that the original UCI archive was not independently
downloaded. This is a static regression check on a fixed cohort.

Python dependencies retain their respective licenses. A dependency license is
not replaced by this extension's code license. The manuscript and scientific
authorship metadata are a provisional draft for author review; no journal
acceptance, DOI, or additional author is asserted.

`manuscript/jmlr2e.sty` is the JMLR style file supplied with the original
manuscript. Its source headers and contributor attributions are retained. The
upstream style project is <https://github.com/JmlrOrg/jmlr-style-file>; the
extension's MIT code notice does not replace third-party style terms.
