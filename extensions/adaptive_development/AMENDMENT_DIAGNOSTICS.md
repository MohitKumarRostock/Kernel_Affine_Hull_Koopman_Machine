# A1 numerical diagnosis and focused regression checks

These are post-observation diagnostics of the preserved first attempt. They
separate a correction-solver defect from a genuine adverse fitted-function
behavior. They are not a second efficacy study and do not change the data,
calibration choices or advancement thresholds.

## Reproduction and evidence identity

Run from the repository root:

```sh
python -m extensions.adaptive_development.AMENDMENT_DIAGNOSTICS --check-amendment
```

The script loads the original module, configuration, prepared inputs and
recorded forecasts directly from `archives/initial_attempt.zip`. It validates
the archive SHA-256 before replaying any prefix. It does not extract over the
working source or overwrite original results. Full numerical values are in
`AMENDMENT_DIAGNOSTICS.json`.

* Original archive SHA-256:
  `6ec5bb7cae6336fb53d417b82d907c16b43fc25db2d99af32ce04647c20cdbe3`.
* Archived original `models.py` SHA-256:
  `721f8fe6935cbea0ec43d624b52ada10dee1a0e7eb7ef92ca052f26d3f096bc4`.
* Amended `models.py` checked here:
  `9fe5382814a280f1853f87f94ea69357393453fc7ed9e0fa5369f1b229d7d068`.

All three diagnostic states below use recurring synthetic seed 20261101.
Times are zero-based forecast origins. The fresh solves reconstruct the
current valid replay and its residuals against the current raw baseline,
then solve by independent augmented SVD, both with and without explicit
centering. Neither fresh solve uses accumulated correction moments.

## 1. Numerical corruption of sliding correction moments

At origin 1279, horizon 1, the small fixed-KAHM update fails Cholesky because
its accumulated correction matrix is no longer positive definite.

| Quantity | Value |
|---|---:|
| Smallest fixed feature standard deviation | 5.92577e-12 |
| Largest absolute feature currently in valid replay | 1,122.1014 |
| Maximum absolute accumulated-minus-fresh F-transpose-F entry | 2,696,619,758.0075 |
| Smallest eigenvalue of accumulated ridge matrix | -40,524.0503 |
| Smallest eigenvalue of fresh ridge matrix | 1.0000000001 |
| Fresh ridge matrix condition number | 3,607,567.0262 |
| Fresh centered-SVD correction | 0.0429347035 |

Almost inactive feature directions had received extremely small standard
deviations. Much larger standardized rows subsequently entered and left the
window. Subtracting their outer products left large floating-point errors in
the accumulated matrix. Rebuilding from the actual retained rows removes
those errors. Direct augmented least squares additionally avoids squaring
the design's condition number.

The original head retains its preceding correction coefficients after this
failed solve; their objective on the current replay is 4.1789577474, versus
3.5626890980 for the fresh fit. Those retained coefficients are not a
successful fit at origin 1279. The original harness issued its documented
seasonal fallback, and that recorded forecast remains preserved.

## 2. Large adaptive forecasts persist under a correct fresh solve

| Quantity | Large adaptive KAHM | Large adaptive affine |
|---|---:|---:|
| Origin / horizon | 1554 / 12 | 1555 / 12 |
| Original correction | 175,470.093819 | 40,727.037842 |
| Fresh centered-SVD correction | 175,555.574762 | 40,726.982091 |
| A1 augmented-QR correction | 175,555.574762 | 40,726.982091 |
| Maximum absolute query feature | 3,195,081.870 | 661,282.720 |
| Maximum absolute current replay feature | 391.526 | 23,077.391 |
| Minimum feature standard deviation | 1.61085e-7 | 1.27016e-6 |
| Original objective | 39.2348581383 | 38.310844612237 |
| Fresh objective | 39.2348581355 | 38.310844612235 |

These forecasts are large extrapolations of the fitted function in the
existing standardized feature coordinates. Delayed horizon-12 labels mean
the supervised replay does not yet represent the returned regime in the
same way as the current query. The original objectives are already close
to their fresh optima. Numerical stabilization therefore does not remove
these adverse forecasts. KAHM's minimum relative signed denominator at the
reported peak is 0.35443, well above the existing 1e-10 guard; its query is
valid. Clipping forecasts, changing the feature-scale floor or changing
gamma would be separate modeling interventions and were not performed.

## 3. Exact scope and validation of A1

The correction head now obtains residuals directly from its bounded replay,
centers residuals and features to eliminate the unpenalized intercept, and
solves an augmented ridge system using column equilibration and pivoted QR.
The penalty rows receive the same change of variables as the feature
columns, preserving the original sum-of-squares objective and alpha of 1.
Corrections are evaluated in centered form. Obsolete correction FF, FX and
Fy statistics are removed. Raw baseline statistics and all geometry,
scaling, update and calibration rules remain unchanged. The same routine is
used by every geometric arm, inside the charged update interval.

Eleven numerical tests cover the original checks and three added cases:
huge transient features leaving the replay, preservation of the free
intercept and original penalty under equilibration, and clearing correction
state when no valid rows remain. The transient test also checks bit-identical
raw coefficients against a baseline-only head and agreement with an
independent SVD batch objective.

Targeted A1 replays pass the four formerly failing prefixes and one following
observation: fixed KAHM seeds 20261101/02/03 through origins 1280/1160/1161,
and adaptive KAHM seed 20261103 through 1686. Their components are finite.
The two large adverse forecast states are retained to the independent fresh
solution's numerical precision, as shown above. These focused checks support
the arithmetic repair; the complete amended study must still determine
forecast quality, runtime and its development decision.
