# Finite-cohort class-versus-head diagnosis

This study asks whether unsuccessful fitted predictors indicate an inadequate representation–decoder class. It checks 54 fixed classes on the three related Bike Sharing targets: casual, registered, and total rentals. Every class uses the same 172 decision observations for its target, a fixed stored feature map, output range `[0, 1]`, and Euclidean Lipschitz budget 2. Original targets outside that range remain in the loss.

The manuscript reports these results in Section 8.3, Table 4, and Appendix B.3. All 54 classes have constructive adequate witnesses. In 39 classes, both inherited fitted heads are proved to exceed the unchanged tolerance. All 108 heads have a strictly positive certified empirical excess over their class optimum. All 54 classes also have an adequate target-envelope witness without using an optimizer proposal; the constant witnesses pass in none.

## Reproduce the numerical evidence

Use Python 3.10 or newer and NumPy. From the repository root:

```bash
python -m pip install -r experiments/class_diagnosis/requirements.txt
python experiments/class_diagnosis/replay.py --output /tmp/class-diagnosis-replay
```

The command verifies package checksums and the original input bindings, then calls the unchanged independent arithmetic auditor for every class. It reconstructs all 794,124 pairwise distance bounds, checks the three feasible primal candidates per class, evaluates signed weak-dual witnesses, recomputes all class-risk brackets, and recomputes all 108 head-excess intervals. Exact arithmetic can take several minutes. No optimizer or model fit runs.

The output directory contains:

- `summary.json`: verification counts, per-target outcomes, exact class checks, and the numerical-assurance scope.
- `class_risk.csv`: all 54 exact class-risk brackets and tolerances.
- `head_excess.csv`: all 108 inherited head-risk intervals and recomputed excess intervals.

Expected totals are 54 adequate classes, 39 classes with both heads failing, 108 positive excess intervals, 54 adequate target-envelope witnesses, and zero adequate constants. The maximum class-risk bracket width is approximately `2.181e-14` in normalized MSE.

## Interpretation and limits

This is a deterministic diagnosis on already examined observations. The feasible witnesses use their target labels and are not predictors trained before assessment. The three targets are related: total rentals equal casual plus registered rentals. They do not provide three independent application replications.

Stored binary64 feature coordinates are treated as exact inputs. The replay verifies distance, feasibility, dual, risk-bracket, and excess arithmetic under that premise; it does not enclose upstream feature-construction error or establish future-risk coverage. It cannot identify the cause of a fitted head's excess.

The original head membership and achieved-risk proofs are inherited from the fixed public-screening audit. This package verifies their exact hash bindings and independently recomputes the new excess arithmetic; it does not reopen and recalculate the original head prediction payloads. The retained seals and audit records establish consistency among supplied files, not external attestation of historical execution.

## Package contents and data provenance

| Path | Contents |
|---|---|
| `inputs/decision/` | The three cohorts, 54 frozen feature arrays, map identities, unchanged tolerances, and inherited head metadata. |
| `inputs/prior/` | Original public-screening protocols, model seal, numerical audit, and public-source metadata. |
| `records/classes/` | All 54 complete-graph primal/dual witness records. |
| `records/result.json` | Original complete aggregate result with all 108 head rows. |
| `records/original_numerical_audit.json` | Retained independent audit of the original run. |
| `protocol/` | Original scientific protocol, seal, and recorded software requirements. |
| `source/` | Unmodified original scientific producers, input preparation, and independent auditors. |
| `MANIFEST.json` | File sizes and SHA-256 hashes for this package. |

The underlying data are [Bike Sharing, UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing), by Hadi Fanaee-T (2013), DOI `10.24432/C5W894`. This package contains the frozen inputs needed for the finite-cohort diagnosis. The original generation code is retained for scientific inspection and provenance; `replay.py` is the supported offline reproduction entry point. Full regeneration of representations and fitted heads from raw data belongs to the preceding public-screening workflow and is not performed by this replay.
