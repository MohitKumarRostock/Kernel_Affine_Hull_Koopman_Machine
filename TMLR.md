# Forecasting manuscripts and evidence

The [predictive-enrichment branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/paper/geometry-induced-soft-state-clean-2026-11-02) is the current entry point for **Predictive Enrichment of Geometry-Induced Soft-State Abstractions: Information Retention and Decoder Access**. Its [reproduction guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/paper/geometry-induced-soft-state-clean-2026-11-02/REPRODUCIBILITY.md) covers the central saved-loss analysis and the separate supplementary capacity experiment. The [risk-certificate branch](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/research/risk-certificates-consolidated-2026-11-08) addresses the complementary question of whether a specified decoder class can meet an error requirement.

## Historical four-arm forecasting manuscript

**Geometry-Induced Soft-State Abstractions for Forecasting: Predictive Enrichment with Kernel Affine Hull Machines**

This earlier paper and its supplementary studies are retained at the [immutable manuscript snapshot](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/6c39a003e61ace058851ec6badca3e8dd13ef183/). The links below identify its original PDFs, editable sources and scientific records. Its 32-repetition study is separate from the current paper's 64-repetition central experiment.

- [Main PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/papers/soft_state_forecasting/main.pdf) and [supplementary PDF](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/papers/soft_state_forecasting/supplement.pdf)
- [Editable source directory](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/6c39a003e61ace058851ec6badca3e8dd13ef183/papers/soft_state_forecasting/) and [build instructions](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/papers/soft_state_forecasting/README_BUILD.md)
- [Manuscript scope and evidence guide](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/TMLR.md)
- [Primary results](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/extensions/soft_state_ablation_20261102/analysis/primary_contrasts.csv), [independent audit](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/extensions/soft_state_ablation_20261102/verification/independent_audit_summary.json), and [research index](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/6c39a003e61ace058851ec6badca3e8dd13ef183/docs/RESEARCH_INDEX.md)

The four-arm study completed 32 independent repetitions and 256 tasks. Three of four primary criteria are established: local position and KAHM enrichment improve their designated nonlinear references by 69.201% and 10.459%, and linear noninferiority passes. KAHM's 0.0292% nonlinear gain over matched RBF does not establish the prescribed 3% margin. Linear loss is 1.111% higher than the common-coordinate arm, within the 5% noninferiority tolerance. Independent cap selection is secondary; the comparisons do not establish autonomous state dynamics.

The scientific source is [commit `2b7d19b5`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/2b7d19b5b47f46b5b21b28d24960688f7969d96f), published before production. The completed independent numerical audit covers all 256 tasks; it is not external scientific replication.

TMLR review formatting does not assert submission, acceptance or endorsement. Earlier document editions, scientific procedures and their original outcomes remain preserved.

The completed four-arm results and computational audit were published and hash-verified in [commit `65ecf679`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/65ecf679796cc38b2342e2a4c12fa0226dc9ac78), separately from the frozen scientific source and manuscript edition.

The completed manuscript edition, editable source and document verification are published at [commit `6c39a003`](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/commit/6c39a003e61ace058851ec6badca3e8dd13ef183).
