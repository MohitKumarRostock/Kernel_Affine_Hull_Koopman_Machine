"""Independent rational and aggregate-bound checks using manufactured histories."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import common24 as C
import data24 as D
import oracle24 as O


def independent_oracle(history_values, horizon, spec):
    """Normalize rational belief after every symbol, then step future state.

    This uses neither the pinned integer-forward routine nor its matrix-power
    future calculation. Matrix entries are read independently from the JSON.
    """
    transition = [[Fraction(v, spec["transition"]["denominator"]) for v in row]
                  for row in spec["transition"]["numerators"]]
    emission = [[Fraction(v, spec["emission"]["denominator"]) for v in row]
                for row in spec["emission"]["numerators"]]
    values = [Fraction(v, spec["emission"]["symbol_denominator"])
              for v in spec["emission"]["symbol_numerators"]]
    belief = [Fraction(1, 4)] * 4
    for index, value in enumerate(history_values):
        observation = values.index(Fraction.from_float(float(value)))
        if index:
            belief = [sum(belief[i] * transition[i][j] for i in range(4)) for j in range(4)]
        belief = [belief[i] * emission[i][observation] for i in range(4)]
        total = sum(belief)
        belief = [b / total for b in belief]
    for _ in range(horizon):
        belief = [sum(belief[i] * transition[i][j] for i in range(4)) for j in range(4)]
    probabilities = [sum(belief[i] * emission[i][j] for i in range(4)) for j in range(4)]
    assert sum(probabilities) == 1
    mean = sum(p * v for p, v in zip(probabilities, values))
    variance = sum(p * v * v for p, v in zip(probabilities, values)) - mean * mean
    return mean, variance


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return True
    raise AssertionError("An invalid oracle request was accepted")


def check_mean(record, exact_values, nominal_values):
    # The small manufactured panel can sum arbitrary denominators directly;
    # the production evaluator explicitly does not do this across a benchmark.
    truth = sum(exact_values, Fraction()) / len(exact_values)
    lo, hi = C.fraction(record["lower_exact"]), C.fraction(record["upper_exact"])
    assert lo <= truth <= hi
    assert Fraction.from_float(record["lower"]) <= lo <= hi <= Fraction.from_float(record["upper"])
    nominal_mean = sum((Fraction.from_float(float(v)) for v in nominal_values), Fraction()) / len(nominal_values)
    assert nominal_mean == C.fraction(record["mean_of_binary64_representatives_exact"])
    assert record["mean"] == float(nominal_mean)


def main():
    out = PROJECT / "tests/oracle_preflight24.json"
    if out.exists():
        raise FileExistsError("The manufactured oracle receipt is write-once")
    h = D.hmm_module()
    spec = C.read_json(D.STEP23 / "controlled/hmm_spec23.json")
    symbols = np.asarray(h.SYMBOL_VALUES, dtype=np.float64)
    indices = np.asarray([
        [0, 3] * 16,
        [0, 1, 2, 3] * 8,
        [0] * 16 + [3] * 16,
        [3, 2, 1, 0] * 8,
        [1] * 31 + [2],
    ], dtype=np.int64)
    X = symbols[indices]
    Y = np.asarray([[-1., .5, 1.], [.5, -1., -.5], [1., -.5, .5],
                    [-.5, 1., -1.], [1., .5, -.5]])
    horizons = [1, 4, 16]
    predictions = {str(k): {"zero": np.zeros(len(X)),
                             "latest": X[:, -1].copy(),
                             "alternating": np.asarray([-.5, .5, -.5, .5, -.5])}
                   for k in horizons}
    summary, arrays = O.evaluate_oracle(X, Y, horizons, predictions)
    json.dumps(summary, allow_nan=False)
    assert summary["rows"] == len(X) and summary["history_length"] == 32
    assert summary["no_sum_of_arbitrary_posterior_denominators"] is True
    checks = []
    exact_cases, row_brackets, mean_checks = 0, 0, 0
    differences_not_equal_regret = 0
    for column, horizon in enumerate(horizons):
        record = summary["horizons"][str(horizon)]
        means, variances = [], []
        regrets = {cid: [] for cid in predictions[str(horizon)]}
        for row in range(len(X)):
            mean, variance = independent_oracle(X[row], horizon, spec)
            source = h.oracle_from_symbols(h.symbols_from_values(X[row]), horizon)
            assert mean == source.mean and variance == source.variance
            exact_cases += 1
            means.append(mean); variances.append(variance)
            for prefix, exact in [(record["oracle_mean_array_prefix"], mean),
                                  (record["conditional_variance_array_prefix"], variance)]:
                assert Fraction.from_float(float(arrays[prefix + "_lower"][row])) <= exact
                assert exact <= Fraction.from_float(float(arrays[prefix + "_upper"][row]))
                assert arrays[prefix][row] == float(exact)
                row_brackets += 1
            for cid, p in predictions[str(horizon)].items():
                regret = (Fraction.from_float(float(p[row])) - mean) ** 2
                regrets[cid].append(regret)
                prefix = record["models"][cid]["array_prefix"]
                assert Fraction.from_float(float(arrays[prefix + "_lower"][row])) <= regret
                assert regret <= Fraction.from_float(float(arrays[prefix + "_upper"][row]))
                assert arrays[prefix][row] == float(regret)
                row_brackets += 1
        mu_array = arrays[record["oracle_mean_array_prefix"]]
        check_mean(record["oracle_conditional_mean"], means, mu_array)
        check_mean(record["conditional_variance_mean"], variances, arrays[record["conditional_variance_array_prefix"]])
        mean_checks += 2
        oracle_mse = C.exact_mse(Y[:, column], mu_array)
        assert oracle_mse == C.fraction(record["oracle_realized_risk"]["mse_exact"])
        for cid, p in predictions[str(horizon)].items():
            model = record["models"][cid]
            check_mean(model["conditional_regret_mean"], regrets[cid], arrays[model["array_prefix"]])
            mean_checks += 1
            actual = C.exact_mse(Y[:, column], p)
            assert actual == C.fraction(model["realized_risk"]["mse_exact"])
            assert actual - oracle_mse == C.fraction(model["realized_mse_minus_binary64_oracle"]["exact"])
            exact_regret_mean = sum(regrets[cid], Fraction()) / len(X)
            differences_not_equal_regret += int(actual - oracle_mse != exact_regret_mean)
    assert differences_not_equal_regret > 0
    checks.append("Independent stepwise normalized rational filter and repeated transition propagation match all pinned exact means and variances")
    checks.append("Every saved oracle/variance/regret row box contains its independently calculated exact rational value")
    checks.append("Aggregating binary64 lower/upper endpoints exactly encloses direct small-panel rational averages, including negative conditional means")
    checks.append("Realized risks replay exactly and are kept distinct from conditional regret")

    changed_summary, changed_arrays = O.evaluate_oracle(X, -Y, horizons, predictions)
    for key, value in arrays.items():
        assert np.array_equal(value, changed_arrays[key])
    assert any(changed_summary["horizons"][str(k)]["oracle_realized_risk"]["mse"] !=
               summary["horizons"][str(k)]["oracle_realized_risk"]["mse"] for k in horizons)
    checks.append("Changing only future labels leaves all oracle means, variances and conditional regrets unchanged")

    exact_values = [Fraction(1, 3), Fraction(-2, 7), Fraction(5, 11), Fraction(1, 4)]
    boxes = [h.binary64_bracket(v) for v in exact_values]
    nominal = np.asarray([b["nominal"] for b in boxes])
    enclosed = C.jsonable(O._mean_enclosure(nominal,
                       np.asarray([b["lower"] for b in boxes]), np.asarray([b["upper"] for b in boxes])))
    check_mean(enclosed, exact_values, nominal)
    exact_dyadic = np.asarray([.25, .5, -.75])
    enclosed = C.jsonable(O._mean_enclosure(exact_dyadic, exact_dyadic, exact_dyadic))
    assert C.fraction(enclosed["lower_exact"]) == C.fraction(enclosed["upper_exact"]) == 0
    checks.append("Non-dyadic and exact-dyadic manufactured averages verify outward final export and exact zero-width cases")

    invalid = []
    tests = [
        ("empty history matrix", lambda: O.evaluate_oracle(X[:0], Y[:0], horizons, predictions)),
        ("incorrect finite history length", lambda: O.evaluate_oracle(X[:, :-1], Y, horizons, predictions)),
        ("non-HMM observed value", lambda: O.evaluate_oracle(np.zeros_like(X), Y, horizons, predictions)),
        ("non-HMM target value", lambda: O.evaluate_oracle(X, np.zeros_like(Y), horizons, predictions)),
        ("target horizon count", lambda: O.evaluate_oracle(X, Y[:, :2], horizons, predictions)),
        ("unlisted horizon", lambda: O.evaluate_oracle(X, Y, [1, 4, 17], predictions)),
        ("duplicate horizon", lambda: O.evaluate_oracle(X, Y, [1, 1, 16], predictions)),
        ("missing model-horizon mapping", lambda: O.evaluate_oracle(X, Y, horizons, {"1": {}})),
        ("nonfinite forecast", lambda: O.evaluate_oracle(X, Y, horizons, {str(k): {"bad": np.full(len(X), np.nan)} for k in horizons})),
        ("unclipped forecast", lambda: O.evaluate_oracle(X, Y, horizons, {str(k): {"bad": np.full(len(X), 1.01)} for k in horizons})),
        ("nonscalar forecast", lambda: O.evaluate_oracle(X, Y, horizons, {str(k): {"bad": np.ones((len(X), 1))} for k in horizons})),
        ("invalid per-row enclosure", lambda: O._mean_enclosure(np.asarray([0.]), np.asarray([1.]), np.asarray([2.]))),
    ]
    for name, fn in tests:
        assert rejects(fn); invalid.append(name)
    checks.append("Malformed input/target/horizon/forecast and invalid arithmetic-enclosure requests fail closed")
    receipt = {
        "schema": "step24-manufactured-oracle-preflight-v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "check_groups": len(checks),
        "independent_exact_oracle_cases": exact_cases,
        "independent_per_row_bracket_checks": row_brackets,
        "independent_finite_sample_mean_checks": mean_checks + 2,
        "realized_difference_not_equal_conditional_regret_examples": differences_not_equal_regret,
        "rejected_requests": invalid,
        "actual_benchmark_trajectories_generated": 0, "actual_source_csv_rows_parsed": 0,
        "model_fits": 0, "ECG_or_reserve_access": False,
        "source_sha256": {
            str(Path(O.__file__).resolve().relative_to(C.BASE)): C.sha(Path(O.__file__).resolve()),
            str(Path(D.__file__).resolve().relative_to(C.BASE)): C.sha(Path(D.__file__).resolve()),
            str(Path(C.__file__).resolve().relative_to(C.BASE)): C.sha(Path(C.__file__).resolve()),
            str(Path(__file__).resolve().relative_to(C.BASE)): C.sha(Path(__file__).resolve()),
            str((D.STEP23 / "controlled/hmm23.py").relative_to(C.BASE)): C.sha(D.STEP23 / "controlled/hmm23.py"),
            str((D.STEP23 / "controlled/hmm_spec23.json").relative_to(C.BASE)): C.sha(D.STEP23 / "controlled/hmm_spec23.json"),
        },
    }
    with out.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    def forbidden_benchmark_rng(*args, **kwargs):
        raise AssertionError("Manufactured oracle checks cannot construct a benchmark RNG")
    with patch.object(np.random, "PCG64DXSM", forbidden_benchmark_rng):
        main()
