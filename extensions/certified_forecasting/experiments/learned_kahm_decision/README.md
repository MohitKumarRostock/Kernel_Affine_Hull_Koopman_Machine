# Learned KAHM decision experiment

This controlled experiment tests whether lower certificates can exclude an **actually fitted KAHM class** while an upper certificate accepts a retained refinement. The model, history partition, scaler, target, and calibration population are explicit. Its favorable outcomes establish a mechanism on a constructed continuous history law; they are not evidence about the frequency of useful decisions on naturally acquired forecasting data.

## Protocol and findings

`protocol.json` is the authoritative design. `protocol.md` explains it. The smooth two-lag product target is compared with an irreducible control using an independent hidden product. Both use continuous histories, independent episode streams, twenty fresh repetitions, and a fixed MSE tolerance of 0.10. Actual KAHM blocks with two and four clusters are fitted after training-only min-max scaling to [-1,+1]. Retained class costs are two and six, with a common block norm budget of two. Ridge choice uses validation only. The two stratifications are learned-block argmax maps and do not use oracle labels.

The frozen outcome directory is `../../results/learned_kahm_decision_v1/`. At 1,024 calibration episodes the six-coordinate model passes its upper certificate in all twenty repairable trials, while stratified lower certificates additionally establish minimum feasible cost six in 17/20 trials. The unconditional lower certificate gives 0/20 such decisions at this size. At 4,096 episodes both lower constructions certify minimum cost six in all twenty repairable trials. In the irreducible control, stratification excludes both declared classes in 20/20 trials at 4,096 and 16,384 episodes, while unconditional lower bounds certify joint infeasibility in 0/20 even at 16,384. Every trial at 256 episodes is unresolved. All intermediate failures are retained.

The total failure allocation 0.05 covers both regimes, all twenty repetitions, all four sample sizes, both classes, and all upper and lower endpoints jointly. The frequencies are descriptive results of these constructed repetitions. They do not estimate a real-task prevalence of success.

## Reproduce fitting and calibration

From the extension root, with the documented dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/learned_kahm_decision/run_study.py develop \
  --output /tmp/learned-kahm-new-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/learned_kahm_decision/run_study.py calibrate \
  --output /tmp/learned-kahm-new-run
```

These fixed seeds reproduce the published experiment. A genuinely fresh study requires a separately reviewed protocol and an implemented new seed rule before its data are generated; changing a textual seed description alone does not change the runner's fixed numeric generator. The runner intentionally refuses existing development output or an already opened calibration run. All forty feature banks, eighty selected decoders, and their partitions are frozen before the first calibration stream is generated. Calibration checks source and model hashes before doing any prediction.

## Verify the published result without refitting

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/learned_kahm_decision/verify_study.py \
  --results results/learned_kahm_decision_v1 \
  --output /tmp/learned-kahm-verification.json

python experiments/learned_kahm_decision/make_assets.py \
  --results results/learned_kahm_decision_v1
```

The portable verifier reads the complete frozen numerical KAHM banks from `frozen_models.json.gz`, reconstructs all streams and predictions, and audits the selected validation minima. It imports neither the experiment runner nor its certificate functions. It independently computes all confidence endpoints and solves a general linear program for each stratum-mass minimization, rather than using the runner's sorted greedy implementation. No pickle or missing NPZ file is needed.

The retained verification covers 240 candidate fits, 80 selected models, 120 distinct random streams, 1,310,720 reconstructed episode/model losses, 320 upper endpoints, 2,240 unconditional or conditional mean intervals, 640 mass linear programs, and 160 decisions. The largest numerical discrepancy is 1.12e-16. All coefficient norms are feasible. Twenty-two candidate fits and nine selected fits reached the 20,000-iteration limit before the strict 1e-9 objective-gap criterion; the largest gaps are 5.35e-9 across candidates and 2.21e-9 across selected models. These outcomes are retained. Risk validity requires fixed feasible predictors, not a claim of exact optimizer convergence.

## Evidence and reproducibility

- `protocol_freeze.json`, `development_freeze.json`, and `calibration_started.json` establish the execution order. Hashes are provenance records, not an external preregistration service.
- `frozen_models.json.gz` contains complete safe JSON arrays for learned KAHM blocks, scaler extrema, geometry indices, all validation candidates, and selected coefficients. It is a portable numerical artifact, not a hand-designed representation.
- `endpoints.csv`, `decisions.csv`, `summary.csv`, and `interval_details.json.gz` retain every prescribed endpoint and decision.
- `calibration_records.json` gives raw-array and prediction hashes, plus transformed-input range diagnostics.
- `source_snapshot.tar.gz` and `source_snapshot.json` preserve the exact execution sources. `verification.json` records the independent arithmetic and prediction replay.
- The generated `development_arrays.npz` and `calibration_arrays.npz` follow the repository's existing exclusion rule for derived numerical arrays. The verifier reconstructs them from seeds and frozen numerical banks; their hashes remain in the records.
- `pilot_sign.py`, `pilot.py`, and their result JSON files disclose the development-only exploration that preceded the confirmatory protocol.

Before calibration, the initial source archive was found to be zero bytes despite a previously recorded nonempty hash. The calibration preflight stopped before generating any calibration data. The initial manifests and freezes remain in `precalibration_freeze_issue/`. Every actual execution dependency still matched its original hash. One unrelated source module edited in parallel was excluded from the repaired snapshot, and explicit superseding pre-calibration freezes were recorded. Models, seeds, design, tolerance, and sample sizes were unchanged. A subsequent isolated execution of the snapshot-writing routine produced a persistent, valid archive; the original truncation's cause was not identified. This repair is an execution-provenance event, not an experimental revision after calibration.

After calibration, archive maintenance narrowed the source list to the inherited KAHM local-import closure and the four imported extension modules, and changed source-archive creation to validated staged bytes followed by atomic replacement. This prevents unrelated concurrent modules from invalidating future freezes and makes incomplete writes detectable immediately. `post_run_maintenance.json` records the exact before/after runner hashes and a temporary snapshot smoke check. AST comparison confirms that only the two source-packaging functions changed; every data-generation, fitting, selection, and calibration function is unchanged. The exact runner used for the reported experiment remains in its execution snapshot.
