"""Manufactured exact checks only; no benchmark sampling, fitting, or data access."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

import hmm23 as H


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def exact_joint(observations, hidden):
    result = Fraction(1, 4)
    for time, (observation, state) in enumerate(zip(observations, hidden)):
        if time:
            result *= Fraction(H.TRANSITION_NUMERATORS[hidden[time-1]][state], 512)
        result *= Fraction(H.EMISSION_NUMERATORS[state][observation], 32)
    return result


def enumerate_posterior(observations):
    probabilities = [Fraction(0) for _ in range(4)]
    for hidden in product(range(4), repeat=len(observations)):
        probabilities[hidden[-1]] += exact_joint(observations, hidden)
    total = sum(probabilities)
    return tuple(p / total for p in probabilities), total


def independent_future(belief, horizon):
    current = belief
    for _ in range(horizon):
        current = tuple(sum(current[i] * Fraction(H.TRANSITION_NUMERATORS[i][j], 512)
                            for i in range(4)) for j in range(4))
    return tuple(sum(current[i] * Fraction(H.EMISSION_NUMERATORS[i][j], 32)
                     for i in range(4)) for j in range(4))


class ScriptedEndpoints:
    def __init__(self, upper=False):
        self.upper = upper
        self.calls = []

    def integers(self, low, high):
        self.calls.append((low, high))
        return high - 1 if self.upper else low


class ScriptedSequence:
    """Prescribed categorical draws; never constructs a random generator."""

    def __init__(self, draws):
        self.draws = tuple(draws)
        self.calls = []

    def integers(self, low, high):
        index = len(self.calls)
        assert index < len(self.draws), "Simulator requested an extra draw"
        expected_low, expected_high, value = self.draws[index]
        assert (low, high) == (expected_low, expected_high), "Draw order changed"
        assert low <= value < high
        self.calls.append((low, high))
        return value


def nonconstant_script(length):
    """Manufacture a changing state/symbol sequence without sampling it.

    The observations at indices32,35,47 are0,3,2; each also differs from
    its predecessor. Thus unlike constant endpoint fixtures, these values
    expose an off-by-one target index or a changed horizon order.
    """
    states = tuple((time + 1) % 4 for time in range(length))
    observations = tuple((time // 4 + time) % 4 for time in range(length))
    draws = [(0, 4, states[0])]
    for time, (state, observation) in enumerate(zip(states, observations)):
        if time:
            transition_row = H.TRANSITION_NUMERATORS[states[time - 1]]
            draws.append((0, H.TRANSITION_DENOMINATOR, sum(transition_row[:state])))
        emission_row = H.EMISSION_NUMERATORS[state]
        draws.append((0, H.EMISSION_DENOMINATOR, sum(emission_row[:observation])))
    return ScriptedSequence(draws), observations


def check_specification(spec):
    """Match every numerical/structural benchmark field to its code contract.

    Descriptive claims about later fitting and evaluation are protocol
    commitments, not properties this manufactured check can establish.
    """
    expected = {
        "state_order": list(H.STATE_NAMES),
        "initial_state_probabilities": {"numerators": [1, 1, 1, 1], "denominator": 4},
        "transition.numerators": [list(row) for row in H.TRANSITION_NUMERATORS],
        "transition.denominator": H.TRANSITION_DENOMINATOR,
        "emission.numerators": [list(row) for row in H.EMISSION_NUMERATORS],
        "emission.denominator": H.EMISSION_DENOMINATOR,
        "emission.symbol_numerators": list(H.SYMBOL_NUMERATORS),
        "emission.symbol_denominator": H.SYMBOL_DENOMINATOR,
        "observation_order": "Emit at initial time0, then transition before each later emission",
        "history_length": H.HISTORY_LENGTH,
        "history_order": "oldest observation first, newest last",
        "horizons": list(H.HORIZONS),
        "primary_horizon": H.PRIMARY_HORIZON,
        "target_array_indices_zero_based": [H.HISTORY_LENGTH - 1 + h for h in H.HORIZONS],
        "trajectory_length_with_targets": H.HISTORY_LENGTH + max(H.HORIZONS),
        "trajectory_length_support_only": H.HISTORY_LENGTH,
        "replication_ids": list(range(H.REPLICATIONS)),
        "rows_per_replication": H.ROLE_COUNTS,
        "one_example_per_independently_initiated_trajectory": True,
        "role_codes": H.ROLE_CODES,
        "randomness.bit_generator": "NumPy PCG64DXSM",
        "randomness.seed_sequence_entropy": [str(H.SEED_NAMESPACE), "replication_id",
                                              "role_code", str(H.BASE_SEED)],
        "randomness.base_seed": H.BASE_SEED,
        "randomness.namespace": H.SEED_NAMESPACE,
        "randomness.categorical_sampling": "Uniform integer draws with bounds4,512,32 and cumulative integer probabilities",
        "randomness.integer_call_order": "For each row: initialstate; emission0; then transition/emission for times1..length-1",
    }
    for field, value in expected.items():
        actual = spec
        for component in field.split("."):
            actual = actual[component]
        assert actual == value, "Specification/code mismatch: " + field
    assert H.SYMBOL_VALUES == tuple(float(Fraction(v, H.SYMBOL_DENOMINATOR))
                                   for v in H.SYMBOL_NUMERATORS)
    assert H.PRIMARY_HORIZON in H.HORIZONS
    assert H.seed_entropy(0, "support") == (H.SEED_NAMESPACE, 0, H.ROLE_CODES["support"], H.BASE_SEED)
    return sorted(expected)


def bayes_risk(rows, state_key, target_key):
    groups = {}
    for row in rows:
        groups.setdefault(row[state_key], []).append(row)
    risk = Fraction(0)
    for group in groups.values():
        mass = sum(row["probability"] for row in group)
        mean = sum(row["probability"] * row[target_key] for row in group) / mass
        risk += sum(row["probability"] * (row[target_key] - mean) ** 2 for row in group)
    return risk


def serialize_fraction(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def run():
    # Fail loudly if this check accidentally attempts any benchmark split.
    def forbidden_benchmark_generation(*args, **kwargs):
        raise RuntimeError("Step23 manufactured checks cannot generate benchmark trajectories")
    H.generate_role = forbidden_benchmark_generation
    checks = {}
    spec = json.loads((HERE / "hmm_spec23.json").read_text())
    matched_fields = check_specification(spec)
    checks["specification_matches_code"] = True
    checks["specification_matching_fields"] = matched_fields

    for matrix, denominator in [(H.TRANSITION_NUMERATORS, 512), (H.EMISSION_NUMERATORS, 32)]:
        assert all(sum(row) == denominator and all(v > 0 for v in row) for row in matrix)
    assert all(sum(row[j] for row in H.TRANSITION_NUMERATORS) == 512 for j in range(4))
    for i in range(4):
        for j in range(4):
            sign = 31 if i // 2 == j // 2 else 1
            amplitude = 15 if i % 2 == j % 2 else 1
            assert H.TRANSITION_NUMERATORS[i][j] == sign * amplitude
    stationary_obs = tuple(Fraction(sum(row[j] for row in H.EMISSION_NUMERATORS), 128)
                           for j in range(4))
    values = tuple(Fraction(v, 2) for v in H.SYMBOL_NUMERATORS)
    assert stationary_obs == tuple(Fraction(v, 128) for v in (31, 33, 33, 31))
    assert sum(p * v for p, v in zip(stationary_obs, values)) == 0
    second = sum(p * v * v for p, v in zip(stationary_obs, values))
    assert second == Fraction(157, 256)
    stationary_spec = spec["emission"]["stationary_observation_probabilities"]
    assert stationary_obs == tuple(Fraction(v, stationary_spec["denominator"])
                                   for v in stationary_spec["numerators"])
    assert spec["emission"]["stationary_mean"] == 0
    second_spec = spec["emission"]["stationary_second_moment"]
    assert second == Fraction(second_spec["numerator"], second_spec["denominator"])
    checks["rational_rows_stationarity_and_factorization"] = True

    posterior_count = 0
    oracle_count = 0
    loss_identity_count = 0
    bracket_count = 0
    for length in (1, 2, 3):
        total_probability = Fraction(0)
        total_mean = {h: Fraction(0) for h in H.HORIZONS}
        total_second = {h: Fraction(0) for h in H.HORIZONS}
        for observations in product(range(4), repeat=length):
            expected_belief, probability = enumerate_posterior(observations)
            total_probability += probability
            weights = H.forward_integer_weights(observations)
            assert tuple(Fraction(w, sum(weights)) for w in weights) == expected_belief
            posterior_count += 1
            for horizon in H.HORIZONS:
                oracle = H.oracle_from_symbols(observations, horizon)
                expected_future = independent_future(expected_belief, horizon)
                assert oracle.predictive_symbol_probabilities == expected_future
                assert oracle.belief == expected_belief
                assert sum(expected_future) == 1 and all(p > 0 for p in expected_future)
                assert -1 <= oracle.mean <= 1 and 0 <= oracle.variance <= 1
                assert oracle.mean == sum(p * y for p, y in zip(expected_future, values))
                assert oracle.second_moment == sum(p * y * y for p, y in zip(expected_future, values))
                total_mean[horizon] += probability * oracle.mean
                total_second[horizon] += probability * (oracle.variance + oracle.mean ** 2)
                for value in (oracle.mean, oracle.variance, oracle.second_moment):
                    bracket = H.binary64_bracket(value)
                    assert Fraction.from_float(bracket["lower"]) <= value <= Fraction.from_float(bracket["upper"])
                    assert bracket["lower"] <= bracket["nominal"] <= bracket["upper"]
                    bracket_count += 1
                for prediction in (-1.0, -0.25, 0.0, 0.125, 1.0):
                    forecast = Fraction.from_float(prediction)
                    conditional_risk = sum(p * (forecast - y) ** 2 for p, y in zip(expected_future, values))
                    regret = H.exact_conditional_regret(prediction, oracle)
                    assert conditional_risk == oracle.variance + regret
                    assert 0 <= conditional_risk <= 4 and 0 <= regret <= 4
                    loss_identity_count += 1
                oracle_count += 1
        assert total_probability == 1
        assert all(value == 0 for value in total_mean.values())
        assert all(value == second for value in total_second.values())
    checks["exhaustive_short_history_posteriors"] = posterior_count
    checks["independent_exact_future_laws"] = oracle_count
    checks["exact_conditional_loss_decompositions"] = loss_identity_count
    checks["outward_binary64_oracle_brackets"] = bracket_count
    checks["law_of_total_expectation_and_second_moment"] = True

    long_histories = [(0,) * 32, (3,) * 32, tuple(range(4)) * 8, (0, 3) * 16]
    for history in long_histories:
        weights = H.forward_integer_weights(history)
        belief = tuple(Fraction(w, sum(weights)) for w in weights)
        for horizon in H.HORIZONS:
            oracle = H.oracle_from_symbols(history, horizon)
            assert oracle.predictive_symbol_probabilities == independent_future(belief, horizon)
    checks["fixed_length32_manufactured_histories"] = len(long_histories)

    for upper, expected in [(False, 0), (True, 3)]:
        full_rng = ScriptedEndpoints(upper)
        history, targets = H.simulate_one(full_rng)
        assert history == (expected,) * 32
        assert targets == (expected,) * 3
        assert len(full_rng.calls) == 96
        assert full_rng.calls[0] == (0, 4)
        assert full_rng.calls[1] == (0, 32)
        assert full_rng.calls[2:] == [(0, 512), (0, 32)] * 47
        support_rng = ScriptedEndpoints(upper)
        history, targets = H.simulate_one(support_rng, horizons=())
        assert history == (expected,) * 32 and targets == ()
        assert len(support_rng.calls) == 64
    checks["scripted_draw_order_and_support_no_future"] = True

    full_rng, prescribed = nonconstant_script(spec["trajectory_length_with_targets"])
    history, targets = H.simulate_one(full_rng)
    expected_indices = spec["target_array_indices_zero_based"]
    assert expected_indices == [32, 35, 47]
    assert history == prescribed[:32]
    assert targets == tuple(prescribed[index] for index in expected_indices) == (0, 3, 2)
    assert all(prescribed[index] != prescribed[index - 1] for index in expected_indices)
    assert len(full_rng.calls) == len(full_rng.draws) == 96
    permuted_horizons = (16, 1, 4)
    permuted_rng, prescribed_again = nonconstant_script(48)
    permuted_history, permuted_targets = H.simulate_one(permuted_rng, horizons=permuted_horizons)
    assert permuted_history == history
    assert permuted_targets == tuple(prescribed_again[31 + h] for h in permuted_horizons) == (2, 0, 3)
    assert len(permuted_rng.calls) == len(permuted_rng.draws) == 96
    support_rng, prescribed_support = nonconstant_script(32)
    support_history, support_targets = H.simulate_one(support_rng, horizons=())
    assert support_history == prescribed_support == history and support_targets == ()
    assert len(support_rng.calls) == len(support_rng.draws) == 64
    checks["nonconstant_scripted_target_indexing"] = {
        "scripted_trajectories": 3,
        "target_indices": expected_indices,
        "target_symbols": list(targets),
        "permuted_horizons": list(permuted_horizons),
        "permuted_target_symbols": list(permuted_targets),
        "support_has_no_future_draws": True,
        "random_generator_constructed": False,
    }
    all_seeds = [H.seed_entropy(rep, role) for rep in range(5) for role in H.ROLE_CODES]
    assert len(all_seeds) == len(set(all_seeds)) == 20
    assert all(H.seed_entropy(rep, role) == (H.SEED_NAMESPACE, rep, H.ROLE_CODES[role], H.BASE_SEED)
               for rep in range(5) for role in H.ROLE_CODES)
    checks["distinct_replication_role_seed_entropy"] = 20
    assert H.symbols_from_values(H.SYMBOL_VALUES) == (0, 1, 2, 3)
    for invalid, horizon in [([], 1), ([4], 1), ([0.0], 1), ([True], 1), ([0], 0), ([0], True)]:
        try:
            H.oracle_from_symbols(invalid, horizon)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid oracle request was accepted")
    checks["invalid_history_and_horizon_rejection"] = True

    rows = H.normalization_fixture()
    tau = Fraction.from_float(1e-6)
    for row in rows:
        assert all(0 <= c <= 1 for c in row["scores"])
        assert tuple(1 + tau - c for c in row["scores"]) == row["weights"]
        assert sum(row["weights"]) == row["scale"]
        assert row["normalized_state"] == (row["direction"], 1 - row["direction"])
    signal_p = bayes_risk(rows, "normalized_state", "signal_target")
    signal_u = bayes_risk(rows, "weights", "signal_target")
    nuisance_p = bayes_risk(rows, "normalized_state", "nuisance_target")
    nuisance_u = bayes_risk(rows, "weights", "nuisance_target")
    assert (signal_p, signal_u, nuisance_p, nuisance_u) == (1, 0, 0, 0)
    checks["engineered_normalization_control"] = {
        "rows": 4, "exact_signal_risk_normalized": serialize_fraction(signal_p),
        "exact_signal_risk_scale_retained": serialize_fraction(signal_u),
        "exact_nuisance_risk_normalized": serialize_fraction(nuisance_p),
        "exact_nuisance_risk_scale_retained": serialize_fraction(nuisance_u),
        "end_to_end_KAHM_experiment": False,
        "arithmetic": "exact rationals; no rounded near-collision claim"}

    sources = [HERE / "hmm23.py", HERE / "hmm_spec23.json", Path(__file__).resolve()]
    return {"schema": "step23-controlled-manufactured-preflight-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "checks": checks, "benchmark_trajectories_generated": 0, "model_fits": 0,
            "real_dataset_access": False, "ECG_or_reserve_access": False,
            "source_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}}


if __name__ == "__main__":
    receipt = run()
    output = PROJECT / "theory/controlled_preflight23_v2.json"
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))
