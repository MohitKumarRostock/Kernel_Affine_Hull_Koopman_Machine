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
not replaced by this extension's code license. No journal acceptance, DOI, or
additional author is asserted by this source distribution.

`manuscript/tmlr.sty`, `manuscript/tmlr.bst`, and `manuscript/fancyhdr.sty`
are the unmodified files from the official TMLR style project,
<https://github.com/JmlrOrg/tmlr>. Their source headers and contributor
attributions are retained, with the supplied project license in
`manuscript/LICENSE_TMLR.txt`. The extension's MIT code notice does not replace
third-party style terms.

The TMLR review package retains these notices and the existing extension and
upstream MIT licenses. Published software attribution, source identifiers and
copyright holders are not removed to suggest that the code has a different
origin. The review manifest lists these narrow attribution/provenance exceptions;
they are not an assertion about the submission's authors. First-party overview
author metadata and current-study branch links are omitted from that view.
