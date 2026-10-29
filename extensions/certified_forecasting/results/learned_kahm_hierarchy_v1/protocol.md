# Frozen learned-KAHM block-hierarchy study

This follow-up closes the gap between the scalar block hierarchy and the
learned KAHM representation study. It conditions on **all 40 archived feature
banks and all 80 selected feasible decoders** from `learned_kahm_decision_v1`.
It does not refit or select banks using the new observations. The fresh
independent streams test the same repairable and irreducible product laws;
they do not constitute a natural-data validation or 40 new training runs.

The machine-readable protocol is `protocol.json`. Freeze it, the archived
model hash, and the complete numerical source closure before drawing any new
calibration episode. Use all banks, both prefixes, n=256/1024/4096, and
unconditional b=2/4/8/16. Evaluate both frozen learned partitions using
stratified pairs with estimated masses. Full-sample scalar Lipschitz ERM is
included at the prespecified feasible size n=256 only; every method uses the
same 256 observations there. Absence of ERM at larger n is a computation
limit, not evidence favoring blocks over full optimization.

All 1,760 endpoints share total delta 0.05 equally. A stratified endpoint
splits its allocation between mass coverage and conditional KL lower means.
Upper bounds concern the archived fitted forecast. Lower bounds concern the
whole scalar kappa-Lipschitz class on each retained representation and hence
its restricted fitted decoder family. The fitted risk and lower class
quantity need not coincide. Minimum cost refers only to the two declared
feature counts 2 and 6, at the inherited study-design MSE tolerance 0.10.

The sparse signed-edge dual uses no matrix with cubic storage. Every finite
iterate supplies a weak-duality lower value. A McShane-envelope repair
supplies a primal comparison; the stopping tolerance is a numerical target,
not a criterion for including a result. Every stopping-limit case, arithmetic
fallback, gap, and unresolved decision remains in the record. Floating-point
guards and independent long-double checks are not interval-arithmetic proofs.

Development-only benchmarks informed computational limits and solver choice.
The dense solver benchmark was terminated while solving n=256 after exposing
wide brackets at some smaller block sizes. Its partial JSON results are
retained. The sparse benchmark evaluated fresh development-only streams on
bank 0 in each regime. Neither development stream is reused for confirmation.
The study makes no claim that any measured method will improve over pairs.

Reproduction:

```bash
python experiments/learned_kahm_hierarchy/run.py freeze --output results/learned_kahm_hierarchy_v1
python experiments/learned_kahm_hierarchy/run.py calibrate --output results/learned_kahm_hierarchy_v1
python experiments/learned_kahm_hierarchy/make_assets.py --results results/learned_kahm_hierarchy_v1
```

The `freeze` and `calibrate` subcommands refuse to overwrite previous stages.
Use a new output directory for a new run. No protocol, source, budget,
preprocessing, model, partition, or tolerance can change after calibration
starts. Result hashes and the source snapshot support independent replay.
