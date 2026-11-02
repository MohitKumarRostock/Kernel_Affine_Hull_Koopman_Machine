# Class exclusion and the minimum feasible representation cost

This package evaluates whether a lower certificate can support a positive-tolerance
class decision beyond upper-only acceptance. It uses explicit bounded
representations and a constructed forecasting law with independent episodes.
This controlled law tests the general representation theorem; it does not
measure KAHM forecasting accuracy.

## Protocol and outcome

The full design is fixed in `protocol.json`. Each episode has observed history
`(Z,S)` and future `Y`. Here `S ~ Bernoulli(0.2)` takes value one with
probability 0.2, and `Z`, `R`, and `N` are mutually independent signs, each
uniform on `{-1,1}` and independent of `S`. The repairable regime has `Y=0.8*S*Z+0.1*N`; the
irreducible regime has `Y=0.8*S*R+0.1*N`. Successive retained simplex blocks
have coordinate costs 1, 3, and 5 and share the coefficient budget 2. Their
exact optimal MSEs are respectively `(0.138,0.138,0.01)` and
`(0.138,0.138,0.138)`. The prescribed RMSE tolerance is 0.30.

The protocol uses 100 independent complete repetitions per regime, 2,048
training episodes, 1,024 validation episodes, and all five calibration sizes
256, 1,024, 4,096, 16,384, and 65,536. All training fits and independent
validation scores are saved and hashed before any calibration stream is
generated. This is an archived analysis specification, not an external
preregistration. The protocol and source hashes preserve the design used for
the retained run.

The stratified method estimates masses for two frozen partitions, `S` and
`(S,Z)`, and solves the probability-interval linear program. Each observation
is one origin from one independent episode. Both partition families, all
prefixes, and all five sample sizes share the stated confidence budget. The
maximum of the two lower endpoints is therefore a declared simultaneous
comparison, not a partition chosen without multiplicity correction.

Upper-only acceptance and acceptance supplemented by either lower bound use
the same fitted model and upper endpoint. Lower calibration can additionally
certify minimum cost among the three prescribed classes, or exclude all three.
It is not a claim of a different selected predictor or a measured saving in
fitting time. The full run includes the unfavorable small-sample comparison:
ordinary pairing excludes the constant class at 4,096 episodes when stratified
pairing still excludes no class.

## Reproduction

From the extension root, with its Python dependencies installed:

```bash
python experiments/decision_value/audit_existing.py --output /tmp/decision_existing_audit
python experiments/decision_value/run.py --output /tmp/decision_fresh_run
python experiments/decision_value/verify.py --results /tmp/decision_fresh_run --output /tmp/decision_verification.json
python experiments/decision_value/make_assets.py
```

The runner refuses to overwrite an existing output directory. The asset
generator reads the retained `results/decision_value/fresh` records and writes
only the manuscript's `decision_` table and figure assets. All simulated
states, fitted coefficients, conditional sufficient statistics, probability
intervals, endpoints, and decisions are retained. The verifier does not import
the runner or the certificate implementation. It reconstructs every endpoint,
checks the mass linear programs by independent vertex enumeration, enumerates
the finite population, and verifies every original random stream.

## Retrospective audit

`audit_existing.py` reads all retained JSON records in the original main,
oscillator, and coefficient-budget campaigns. It reports accepted, excluded,
and unresolved candidates at RMSE tolerances 0.20, 0.35, and 0.50, separately
by horizon and for all horizons together. It also reports lower endpoints
for full-family and KAHM-only validation winners. No models are refitted and
no sample-size curve is inferred from aggregate moments. All campaign
summaries remain descriptive over their original repetition-level events.

## Auditing a fresh checkout

Generated NPZ state archives are omitted from Git. Reconstruct the exact
original streams from their frozen seeds and validate their original hashes:

```bash
python experiments/decision_value/reproduce_states.py \
  --results results/decision_value/fresh --output verification/decision_streams
PYTHONPATH=src python experiments/decision_value/verify.py \
  --results results/decision_value/fresh --state-directory verification/decision_streams \
  --output verification/decision_audit.json
```

Both output paths must be new. These commands reproduce the original streams
and audit the committed evidence without refitting or changing the study.
