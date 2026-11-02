# Predictive Enrichment of Geometry-Induced Soft-State Abstractions

## Information Retention and Decoder Access

The paper investigates the gap between geometric interpretation and predictive usefulness. KAHM is the principal reconstruction construction, with matched affine and positive-kernel controls. The article separates information retained by an encoding, access through a fitted finite decoder, and the exact invariance of a candidate observable space.

`main.tex` is the article and includes its mathematical proofs. `supplement.tex` contains the numerical specification, all 40 primary intervals, all terminal procedure means under both selection views, related studies, and reproducibility limitations. The bibliography, 14 static tables, and unmodified TMLR style files are included.

```bash
python scripts/build.py
```

The command produces `main.pdf` and `supplement.pdf`. It uses supplied tables and does not refit models or regenerate observations. The branch-specific document workflow uses the same inputs; its generated PDFs and build receipt are distinct from the scientific audit and from the previously supplied PDF binaries.

The [publication manifest](PUBLICATION_MANIFEST.json) identifies approved build inputs. The original author-delivered manuscript PDFs have SHA256 identities `ed66a5b1381d46953e9e966f6a4300e4c94c99ee0c7673d61dc473074a37981a` (main) and `6a5d26eecae8570eb8153edc3d5a5927589f2811b97463876d14f40929fbdca5` (supplement). Recompilation is not expected to preserve binary identity across TeX environments.

The [study directory](../../extensions/score_enrichment_20261102/) contains the exact primary summary and source/release identifiers. The [reproduction guide](../../REPRODUCIBILITY.md) distinguishes this publication subset from the complete author-delivered scientific package. The large episode/model archives are not mirrored here.

The anonymous author fields and review-format header do not assert submission or acceptance. This public directory retains identifying licenses and citations and is not an anonymity-certified code package. Required third-party licenses are preserved.
