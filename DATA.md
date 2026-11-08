# Data and attribution

The experiments retain the original study-specific partitions, normalizations, target definitions, selected model identities, and scientific source records. Dataset citations also appear in `manuscript/references.bib`.

| Data | Source | Used by |
| --- | --- | --- |
| PTB-XL, version 1.0.3 | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45); Wagner et al., *Scientific Data* (2020) | ECG cohort |
| Bike Sharing | [UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894) | Public screening and class diagnosis |
| Beijing PM2.5 | [UCI dataset 381](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) | Public screening |
| Appliances Energy Prediction | [UCI dataset 374](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), DOI [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G) | Deployment cost |
| Seoul Bike Sharing Demand | [UCI dataset 560](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), DOI [10.24432/C5F62R](https://doi.org/10.24432/C5F62R) | Decoder diagnostics |
| Power Consumption of Tetouan City | [UCI dataset 849](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), DOI [10.24432/C5B034](https://doi.org/10.24432/C5B034) | Decoder diagnostics |
| Individual Household Electric Power Consumption | [UCI dataset 235](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54) | Decoder diagnostics and affine forecasting |
| Metro Interstate Traffic Volume | [UCI dataset 492](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), DOI [10.24432/C5X60B](https://doi.org/10.24432/C5X60B) | Decoder diagnostics and affine forecasting |
| Beutenberg weather observations | Max Planck Institute for Biogeochemistry, official weather archives cited in the manuscript and original protocols | Decoder diagnostics and affine forecasting |
| Synthetic controlled episodes and HMM sequences | Generators and fixed protocols in the corresponding experiment source directories | Controlled calibration and deployment cost |

Third-party data remain subject to their providers' terms; the repository's `DATA_LICENSE` applies to original research outputs and excludes third-party material. The source protocols preserve the original acquisition metadata and hashes. For the final Household study, byte identity of the pinned mirror with the official archive was not established; this provenance limit remains stated in the manuscript.

The ECG package concerns the already examined 8,192-patient cohort. The unopened patient reserve is not part of this release. Population, finite-cohort, and conditional numerical assertions are separately identified in the manuscript and experiment documentation.
