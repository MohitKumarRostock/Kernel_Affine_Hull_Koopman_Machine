# Reproduction and provenance: four-state ERM comparisons

This directory is an **additive** post-hoc numerical analysis of **117,000 existing four-state evaluation datasets**, not a new experiment, sample generation, trajectory simulation, or re-training campaign.

## Frozen research history

- Historical sample-generation commit (verified as a Git object in the repository): `cde430dbcca09bf911439bdb5bc3f85350e232cd`.
- Publication evidence branch original frozen head: `0bbb2bafa107892e32e221e882326890b14fe36e`.
- Input evidence tarball in the unchanged original campaign folder: `reproduction/prospective_campaigns/four_state_original_iid_final_v1_run001/evidence.tar.gz`.
- Original evidence SHA-256: `4776942edbd659f7a1c02ec554507f7c0259ca410449e2e5fd66f5353bf638d5`.
- Frozen configuration SHA-256: `630b6c30f5dcbda9034b85b2a3416d92ef264953cd05133f3e53fa14e7aae182`.

## Reproducing the data

The documented GitHub Action `.github/workflows/four_state_erm_reanalysis.yml` repackages **existing, unchanged** original files into a temporary ZIP using `scripts/build_source_zip.py`, validates every original archive and sample/certificate record, and runs `scripts/reanalyze_historical_four_state.py`, which imports `scripts/certificate_reanalysis.py`.

The Action verifies the SHA-256 of all four deterministically derived CSVs against the fixed values below, then saves the full 117-cell aggregates and the two replicate-level CSVs compressed with deterministic gzip (mtime=0, no filename in the gzip header) on this *separate branch*. It does not write to the original primary experiment directory or `certificate-validation`.

To run this manually in a Git checkout:

```bash
python -m pip install numpy==2.3.5
python reproduction/posthoc_four_state_erm/scripts/build_source_zip.py --repo-root . --out /tmp/four_state_frozen.zip
python reproduction/posthoc_four_state_erm/scripts/reanalyze_historical_four_state.py \
  --archive /tmp/four_state_frozen.zip \
  --out reproduction/posthoc_four_state_erm/outputs/historical_117k
```

| Output | Expected SHA-256 (uncompressed CSV) |
|---|---|
| `cell_summary_joint.csv` | `6ffea4bfe2e8e1e545ddb8aecf241fc1341a2c3b7a934767907c96d7e2bfac37` |
| `cell_summary_pointwise.csv` | `eefda7da73d7dc9024f36e33664b121e92cedbb6b305cb19c961ad8c298b35a6` |
| `replicates_joint.csv` | `380476b3ce1f01129580ebfa0932828be7e2eae1c495d90bd980335278b859b9` |
| `replicates_pointwise.csv` | `927d48bfd80707abf17ff319f712282c948b0a8162c77a86ab161d5aa4519373` |

## Interpretation of the comparison

Each existing replicate retains its archived state count vector and design. At `kappa=sqrt(2)`, the exact empirical minimum squared risk equals the archived within-reference-class successor variance `s_hat`; no fitted numerical ERM optimization is needed for this example. The fixed-prototype method uses the class coordinates **known from the analytic design**, and is not an estimate learned from the evaluation data. Each of the four confidence bounds uses `delta=0.0125` in the jointly reported comparisons, giving overall level 0.95 **within one prespecified replicate**; this is not a uniform event across all 117,000 replicates or post-hoc selected cells.

| M | Original | Sample centered | Fixed prototype | Generic ERM |
|---:|---:|---:|---:|---:|
| 1,024 | 0.1289 | 0.0000 | 0.1468 | 0.0000 |
| 4,096 | 0.2825 | 0.0951 | 0.3126 | 0.2913 |
| 8,192 | 0.3228 | 0.1635 | 0.3593 | 0.3529 |
| 32,768 | 0.3671 | 0.2616 | 0.4144 | 0.4148 |

These means are computed from 1,000 archived replicates per displayed cell of the conflicting-successor case with `p=31/32`. The true population-optimal RMSE is 0.46875. Zero observed certificate coverage violations are not proof of zero failure probability. Original Van der Pol data and the original experimental tables remain unchanged.

The Action-generated `manifest.json` is run-specific (e.g., timestamp and newly packaged input ZIP SHA may differ), whereas the four **CSV** hashes above are the fixed reproducibility criterion. The compressed CSV SHA-256 hashes can be computed after the verified Action run, and uncompressed contents must match these stored fixed hashes.
