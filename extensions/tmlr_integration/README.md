# KAHM forecasting studies: predictive structure, decoder capacity, and cost

This directory provides numerical reporting and provenance for the controlled
predictive-geometry study, the natural-source forecasting evaluation, the
scaling sensitivity analysis, and the
[forecasting-utility study](../forecasting_utility/README.md).

The studies distinguish reconstruction accuracy, the information supplied to a
predictor, and a finite decoder's ability to use that information. Conditional
synthetic gains are evaluated alongside unfavorable natural-source comparisons.
The natural confirmation includes four original source families. Its 80 task
configurations do not represent 80 independent forecasting domains. The scaling
comparisons reuse those sources and the controlled draws, so they quantify
post-hoc sensitivity. Measured pipeline costs include implementation overhead
and depend on the computational environment.

## Distribution contents

This source distribution contains scientific code, protocols, saved results,
analysis scripts, and generated numerical tables and figures. Manuscript
documents and their typesetting inputs are distributed separately. The
`manuscript/generated/` paths retain their numerical outputs and provenance.
LaTeX files in those paths are numerical tables or macro definitions; PDF files
are experimental figures.

## Forecasting-utility reproduction

The forecasting-utility study evaluates a preserved raw-history forecast plus a
validation-selected geometric residual correction, matched affine/locality
controls, and measured costs. Its execution freeze and complete outcomes define
a scientific record separate from the earlier studies.

The [forecasting-utility release](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/releases/tag/forecasting-utility-v0.1.0)
provides `KAHM_Forecasting_Utility_Reproduction.zip`. That bundle contains the
study's scientific source, prepared inputs, saved predictions, numerical checks,
and a manifest of its payloads. Its 52-file scientific freeze excludes
manuscript documents. The reproduction bundle's README specifies verification
and execution commands and defines the scope of its evidence.

From a repository checkout, use the
[forecasting-utility guide](../forecasting_utility/README.md) to restore the
saved run and recompute its numerical summaries. The reporting procedures use
saved outcomes and perform no new forecasting-model fits. Generated reporting
assets are retained under `manuscript/generated/utility/`.

## Earlier study evidence

The [predictive-geometry guide](../predictive_geometry/README.md) describes the
controlled representations, archived paired comparisons, and numerical
sensitivity checks. Its scientific archive supports independent loss and
paired-statistic reconstruction.

The [natural-forecasting guide](../natural_forecasting/README.md) describes the
208-task study, the source-family split, saved forecasts, and diagnostic
policies. Its original full-distribution audit recomputed 2,496 task-method
errors, checked targets and origins, replayed source-balanced analyses, and
refitted small diagnostic regressions to verify the recorded coefficients.

The [scaling-sensitivity guide](../scaling_sensitivity/README.md) specifies
1,664 natural method runs and 1,440 controlled head fits on reused data,
including the public execution checkpoint and the fixed two-arm design.
Compact controlled replay and recreation of full prediction arrays are
distinct verification tasks.

## Scope of historical validation and export tools

The original complete-repository checkpoint records 217 files, including
manuscript inputs. The `check-source`, `audit`, and `replay` routes in
`package.py` require that complete checkpoint or the explicitly identified
inventory of an original review distribution. Their source checks remain
defined by those recorded inventories. They are not completeness checks for
this source distribution, which excludes manuscript documents.

The original checkpoint, scientific freezes, recorded validation results, and
source hashes retain their original scope. A recorded successful check applies
to the files and environment on which that check ran. The standalone
forecasting-utility reproduction bundle uses its own manifest and scientific
dependencies.

The manuscript exporters `package.py` and `package_utility.py` also require
the separate LaTeX sources and a compiled manuscript. Manuscript compilation
and source/PDF export are therefore outside the execution instructions for
this source distribution.
