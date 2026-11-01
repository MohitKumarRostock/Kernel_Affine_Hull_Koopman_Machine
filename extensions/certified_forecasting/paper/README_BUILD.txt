A Hierarchy of Risk Certificates for Forecasting Representations
Authors: Mohit Kumar and Somayeh Kargaran
Source package date: 2026-11-01

SCIENTIFIC SCOPE

The manuscript develops a normalized scalar Lipschitz block hierarchy. Pairs
are its first member; under the stated sampling, boundedness, and compactness
conditions, expected block scores increase toward the class optimum. Exact
examples describe the limits of fixed block sizes. Independent-block
calibration, stratification, and fitted-risk upper bounds support decisions
over declared representation--decoder classes. The main paper contains all
proofs. The experimental supplement gives implementation details, protocols,
and complete secondary tables.

Local PCA dimension is specified separately from retained feature count.
The native initial request is min(cap, ambient dimension, local reference
count minus one), with default cap 20 and the stated numerical reductions.
The fixed-cap reference forecasting, scaling, matched-geometry, budget,
PenDigits, and speech studies use cap 2. The learned two-lag mechanism is
already ambient dimension 2 and has a separate experimental role.

Supplement Section S9 reports a targeted synthetic comparison of cap 2,
native cap 20, and validation-selected cap. The 26 fitting repetitions use
one selected bank per policy and retain all four prefixes before independent
calibration. One study-wide 0.05 allocation covers 4,680 designated one-sided
statements. Selected cap improves descriptive test error on five laws and
ties on the sixth; the selected forecasts' tolerance decisions are unchanged.
The comparison does not establish general forecasting superiority or improved
class-exclusion resolution. Earlier studies retain their own confidence scope.

PACKAGE CONTENTS

main.tex and supplement.tex are the compilation entry points. Their compiled
outputs are main.pdf and supplement.pdf. sections/ contains the article,
appendices, and detailed study text. supplement_sections/ contains generated
supplement copies. The final source fragment is named 12_dimension_policy.tex
because twelve fragments feed nine numbered supplement sections.

The bibliography, JMLR preprint style, figures, and tables are bundled.
dimension_study/ contains the nine audited analysis CSVs, frozen design and
schedule, aggregate metrics, source-freeze metadata, numerical-archive
identity, and independent audit receipt for the dimension study. Its
provenance.json records each input's SHA-256 and origin. These inputs are
sufficient to regenerate every new manuscript table without loading fitted
models or the large numerical-records archive.

BUILD

From this directory, run:

    python build_manuscript.py

The helper first runs build_dimension_assets.py, which checks all bundled
dimension-study input hashes and generates the new tables and numeric macros.
It then expands the detailed study sources, qualifies references to the main
paper, and compiles main.pdf before supplement.pdf. It rejects undefined
references or citations, duplicate labels, and overfull boxes.

To regenerate tables and supplement sources without compiling:

    python build_manuscript.py --prepare-only

To regenerate only dimension-study assets:

    python build_dimension_assets.py

Both helpers accept --paper PATH. The manuscript builder also works from the
repository's scripts/ directory with its sibling paper/ directory. It does
not require a workspace-specific absolute path. Generated table hashes are
recorded in tables/dimension_policy_assets_manifest.json.

For a manual build after source preparation:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

Dependencies are Python 3.9 or later, latexmk, pdfLaTeX, BibTeX, amsmath,
mathtools, enumitem, microtype, booktabs, longtable, lastpage, xr, needspace,
and the standard PSNFSS times package. The supplied jmlr2e.sty provides the
preprint layout and author formatting. The supplement reads main.aux, which
must remain available between compilation steps.

RESEARCH SOURCE AND NUMERICAL RECORDS

The historical research extension is pinned at commit:

    d6d74d1953a4310b73065a556df816e5ab5e3ebc

https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/d6d74d1953a4310b73065a556df816e5ab5e3ebc/extensions/certified_forecasting

That commit contains the reference forecasting and certificate studies. It
does not contain the new dimension-policy study or the present manuscript
sources. Its docs/STUDY_GUIDE.md and docs/REPRODUCIBILITY.md identify study
commands, public-data sources, checksums, and independent sampling units.
The standalone author archive retains revision_review/, documenting the earlier scoped theory,
calibration, and provenance review; those records keep their original dates
and evidential limits.

The dimension study is supplied separately in the accompanying research
extension extensions/dimension_certificates. Its PROTOCOL.md, config.json,
FINDINGS.md, and audit records define the new comparison. The source freeze
has SHA-256:

    0a92dfb3d0f086541d989d36277620b49c103975d8608f7dc4f956bfe9b3a9ff

The separate KAHM_Risk_Dimension_Numerical_Records.zip has SHA-256:

    f0286e44b6587b4a917dffa73b52356c99a2718c28a6c70fe47617b575b59be1

It contains production records under extensions/dimension_certificates/
runs/20261101 and a file manifest. The complete independent numerical audit
reconstructs all 4,680 designated endpoints, all 26 repetitions' scores and
selection rules, and all nine analysis tables (9,020 rows). Complete forecast
and target arrays for prespecified repetition zero of each law support an
additional 288 prediction-to-loss checks. The full campaign is audited at
score level; full-array checks cover those six jobs. No true population-risk
coverage estimate is inferred from test-error containment.

The historical learned-hierarchy extension distributes its largest evidence
files as exact byte parts in results/learned_kahm_hierarchy_v1/archive_parts.
Its scripts/restore_hierarchy_payloads.py verifies and reconstructs them for
a full historical audit. Neither those files nor the dimension numerical
archive are required to compile this self-contained manuscript package.
