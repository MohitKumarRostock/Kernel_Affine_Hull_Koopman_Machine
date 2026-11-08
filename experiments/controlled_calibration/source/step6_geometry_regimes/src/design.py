"""Step 6 observation law, evaluation oracles, and replicate-level inference.

This module has no import-time file access, random draws or fitting.  Learners
must receive only observed X and the specifically revealed training/validation
Y values.  Constants Q/A/phase, clean H, and conditional moments belong to the
generator/evaluation interface, not to predictor inputs or deployment state.

The signal recurrence, initial law and 80-state burn-in match the inherited
bounded nonlinear_ar simulator.  Each pool row is ONE independent episode at
one fixed origin.  Regimes reuse the same H/E rows; lower label budgets slice
the maximal training/validation pools rather than regenerate smaller pools.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy.stats import t as student_t


SEED_DOMAIN = 0x47454F36
REGIMES = ("raw4", "linear20", "curved20")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    return value


def _stored(value):
    value = np.array(value, dtype=np.float64, copy=True, order="C")
    value.setflags(write=False)
    return value


def get_seed(master, domain, *coordinates):
    """The caller supplies the frozen hierarchy; no hidden counter is used."""
    entropy = [_integer(master, "master seed"), _integer(domain, "seed domain")]
    entropy.extend(_integer(value, "seed coordinate") for value in coordinates)
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])


def generator_constants(seed, sigma=.2, rho=.5):
    """Fixed 20D orthogonal mix and label-independent curved graph.

    Draw order: 20x20 standard normals; QR with nonnegative diagonal of R;
    16x4 standard normals normalized to row length three; 16 uniform phases.
    These constants are fixed ACROSS training repetitions.  With sigma>0,
    curved20 is a noisy curved graph, not an intrinsically four-dimensional
    distribution.  Q and the graph must not be supplied to any learner.
    """
    seed = _integer(seed, "generator seed")
    if (not np.isscalar(sigma) or not np.isscalar(rho)
            or not np.isfinite(sigma) or not np.isfinite(rho) or sigma < 0 or rho < 0):
        raise ValueError("Nonnegative finite nuisance/curve amplitudes required")
    rng = np.random.default_rng(seed)
    Q, upper = np.linalg.qr(rng.normal(size=(20, 20)))
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    Q = Q * signs[None, :]
    directions = rng.normal(size=(16, 4))
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise FloatingPointError("Degenerate graph frequency draw")
    A = 3. * directions / lengths[:, None]
    phase = rng.uniform(-np.pi, np.pi, size=16)
    return {"Q": _stored(Q), "A": _stored(A), "phase": _stored(phase),
            "sigma": float(sigma), "rho": float(rho)}


def _constants(constants):
    Q, A, phase = (_finite(constants[key], key) for key in ("Q", "A", "phase"))
    sigma, rho = float(constants["sigma"]), float(constants["rho"])
    if Q.shape != (20, 20) or A.shape != (16, 4) or phase.shape != (16,):
        raise ValueError("Generator constants have incorrect dimensions")
    if (not np.isfinite(sigma) or not np.isfinite(rho) or sigma < 0 or rho < 0
            or np.max(np.abs(Q.T @ Q - np.eye(20))) > 1e-10):
        raise ValueError("Invalid amplitudes or nonorthogonal observation map")
    return Q, A, phase, sigma, rho


def conditional_moments(H, a=.1):
    """Exact one-step oracle from a clean four-state history; evaluation only."""
    H = _finite(H, "clean history")
    if (H.ndim < 1 or H.shape[-1] != 4 or not np.isscalar(a)
            or not np.isfinite(a) or not 0 < a <= 1):
        raise ValueError("Clean four-coordinate histories and 0<a<=1 are required")
    drift = 1.25 * H[..., -1] - .65 * H[..., -2] + .25 * np.sin(3 * H[..., -1])
    if not np.isfinite(drift).all():
        raise FloatingPointError("Nonfinite signal drift")
    t, b = np.tanh(drift), np.tanh(a)
    product = t * b
    mean = np.arctanh(product) / a
    second = 1 - (b / a) * (1 - t * t) / (1 - product * product)
    variance = second - mean * mean
    if np.min(variance) < -1e-12:
        raise FloatingPointError("Conditional variance is negative beyond roundoff")
    return {"drift": drift, "mean": mean, "second_moment": second,
            "variance": np.maximum(variance, 0.)}


def generate_pool(n, seed, nuisance_seed, observed_target=True, origin=59, burnin=80, a=.1):
    """Generate one history/target pair per independent bounded episode.

    For labelled pools the inherited simulator is run to length origin+2;
    unlabelled pools run only to origin+1 and never draw a future innovation
    or compute/store a future target.  Its row-major random draw layout is
    the same as the inherited simulator at the corresponding length.
    Consequently changing n or observed_target changes that random layout;
    nested budgets MUST use prefixes of one already generated maximal pool.

    Return arrays H(n,4), E(n,16), drift/mean/second_moment/variance(n), and
    Y(n) only when observed_target=True.  Oracle arrays are evaluator-owned.
    No complete trajectory or innovation matrix is retained in the result.
    """
    n = _integer(n, "pool size", 1)
    seed, nuisance_seed = _integer(seed, "signal seed"), _integer(nuisance_seed, "nuisance seed")
    origin, burnin = _integer(origin, "origin", 3), _integer(burnin, "burn-in", 0)
    if not isinstance(observed_target, (bool, np.bool_)):
        raise ValueError("observed_target must be boolean")
    if not np.isscalar(a) or not np.isfinite(a) or not 0 < a <= 1:
        raise ValueError("Innovation half-width must satisfy 0<a<=1")
    if seed == nuisance_seed:
        raise ValueError("Signal and nuisance streams must have distinct seeds")
    terminal = burnin + origin
    length = terminal + 1 + int(observed_target)
    rng = np.random.default_rng(seed)
    series = np.empty((n, length), dtype=np.float64)
    series[:, :2] = rng.uniform(-.8, .8, size=(n, 2))
    innovations = rng.uniform(-a, a, size=(n, length - 2))
    for index in range(2, length):
        previous, before = series[:, index - 1], series[:, index - 2]
        series[:, index] = np.tanh(1.25 * previous - .65 * before
            + .25 * np.sin(3 * previous) + innovations[:, index - 2])
    H = _stored(series[:, terminal - 3:terminal + 1])
    E = _stored(np.random.default_rng(nuisance_seed).uniform(-1., 1., size=(n, 16)))
    result = {"H": H, "E": E, **{key: _stored(value) for key, value in conditional_moments(H, a).items()}}
    if observed_target:
        result["Y"] = _stored(series[:, terminal + 1])
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite generated pool")
    return result


def observe(H, E, regime, constants):
    """Map generator rows to the sole permitted learner input, observed X.

    Row convention: X=[H,nuisance_or_curve] @ Q.T.  raw4 deliberately
    exposes H itself, while linear20/curved20 expose only the mixed vector.
    Leading dimensions are preserved; no query-dependent statistics are fit.
    """
    if regime not in REGIMES:
        raise ValueError("Unknown observation regime")
    H, E = _finite(H, "clean history"), _finite(E, "nuisance")
    if H.ndim < 1 or H.shape[-1] != 4 or E.shape != H.shape[:-1] + (16,):
        raise ValueError("Matching history4 and nuisance16 leading dimensions required")
    if regime == "raw4":
        return np.array(H, copy=True)
    Q, A, phase, sigma, rho = _constants(constants)
    tail = sigma * E
    if regime == "curved20":
        tail = tail + rho * np.sin(H @ A.T + phase)
    result = np.concatenate([H, tail], axis=-1) @ Q.T
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite observed input")
    return result


def recover_history(X, regime, constants):
    """Oracle-only inverse projection; never used by a fitted learner."""
    if regime not in REGIMES:
        raise ValueError("Unknown observation regime")
    X = _finite(X, "observed input")
    wanted = 4 if regime == "raw4" else 20
    if X.ndim < 1 or X.shape[-1] != wanted:
        raise ValueError("Observed input dimension differs from the regime")
    if regime == "raw4":
        return np.array(X, copy=True)
    Q, _, _, _, _ = _constants(constants)
    return np.array((X @ Q)[..., :4], copy=True)


def oracle_from_observed(X, regime, constants, a=.1):
    """Exact E[Y|observed X] and variance under the fixed observation law."""
    return conditional_moments(recover_history(X, regime, constants), a)


def point_scores(mean, variance, Y, raw):
    """One bounded one-step response per independent episode, without pooling.

    Every result is a flat per-episode array.  The inference routine below
    receives repetition means, not these individual episode values.
    clipped_count is 0/1 per point and is SUMMED in aggregate reports.
    """
    mean, variance, Y, raw = (_finite(value, name) for value, name in
        ((mean, "conditional mean"), (variance, "conditional variance"),
         (Y, "observed target"), (raw, "raw forecast")))
    if mean.ndim != 1 or any(value.shape != mean.shape for value in (variance, Y, raw)):
        raise ValueError("Matching flat point arrays are required")
    if np.max(np.abs(mean)) > 1 + 1e-12 or np.max(np.abs(Y)) > 1 + 1e-12 or np.min(variance) < -1e-12:
        raise ValueError("Means/targets must be bounded and variances nonnegative")
    prediction = np.clip(raw, -1, 1)
    residual, raw_residual, innovation = mean - prediction, mean - raw, Y - mean
    clipped = raw != prediction
    result = {
        "excess": residual ** 2, "noise": variance,
        "integrated": residual ** 2 + variance, "observed": (Y - prediction) ** 2,
        "cross_term": 2 * residual * innovation,
        "noise_fluctuation": innovation ** 2 - variance,
        "clipped_fraction": clipped.astype(np.float64), "clipped_count": clipped.astype(np.int64),
        "raw_excess": raw_residual ** 2, "raw_observed": (Y - raw) ** 2,
        "raw_cross_term": 2 * raw_residual * innovation,
    }
    result["maximum_loss_identity_error"] = np.abs(result["observed"] - result["integrated"]
        - result["cross_term"] - result["noise_fluctuation"])
    result["maximum_raw_loss_identity_error"] = np.abs(result["raw_observed"] - result["raw_excess"]
        - variance - result["raw_cross_term"] - result["noise_fluctuation"])
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite point score")
    return result


def contrast_interval(values, lower, upper, delta=.00625):
    """Approximate t and bounded EB intervals across independent REPETITIONS.

    These values are paired method contrasts after averaging the distinct
    test stream within each training/support/validation/test repetition.
    Test-pool size is not the sample size of the training-procedure interval.
    """
    values = _finite(values, "independent repetition contrasts")
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(lower) or not np.isfinite(upper)
            or lower >= upper or not np.isfinite(delta) or not 0 < delta < .5
            or np.min(values) < lower - 1e-12 or np.max(values) > upper + 1e-12):
        raise ValueError("Invalid repetition contrasts, population bounds or tail probability")
    n = len(values)
    center, variance = float(values.mean()), float(values.var(ddof=1))
    standard_error = float(np.sqrt(variance / n))
    critical = float(student_t.ppf(1 - delta, n - 1))
    approximate_lower, approximate_upper = center - critical * standard_error, center + critical * standard_error
    radius = float(np.sqrt(2 * variance * np.log(2 / delta) / n)
                   + 7 * (upper - lower) * np.log(2 / delta) / (3 * (n - 1)))
    finite_lower, finite_upper = max(lower, center - radius), min(upper, center + radius)

    def status(lo, hi):
        return "margin_established" if lo > 0 else "margin_ruled_out" if hi < 0 else "inconclusive"

    return {"n_independent_repetitions": n, "degrees_of_freedom": n - 1,
            "inference_unit": "independent_training_validation_support_test_repetition",
            "mean": center, "variance": variance, "standard_error": standard_error,
            "population_lower": float(lower), "population_upper": float(upper),
            "delta_per_tail": float(delta), "t_critical": critical,
            "approximate_lower": approximate_lower, "approximate_upper": approximate_upper,
            "approximate_status": status(approximate_lower, approximate_upper),
            "finite_sample_radius": radius, "finite_sample_lower": finite_lower,
            "finite_sample_upper": finite_upper, "finite_sample_status": status(finite_lower, finite_upper)}


def primary_contrasts(rep_losses, methods, geometry_method="G", baseline_method="B",
                      randomized_method="R", margin=.05, delta=.00625):
    """Four primary-cell directional margins using one risk mean per replicate.

    Rows follow geometry_over_baseline, geometry_over_regrouped,
    baseline_over_geometry, regrouped_over_geometry. Geometry's intended
    result requires positive lower endpoints of BOTH first two margins.
    The reverse margins are alternatives for their respective comparator;
    baseline and regrouped can both improve on geometry simultaneously.
    """
    rep_losses = _finite(rep_losses, "repetition/method risk means")
    methods = list(methods)
    if (rep_losses.ndim != 2 or rep_losses.shape[1] != len(methods)
            or rep_losses.shape[0] < 2 or len(set(methods)) != len(methods)
            or not all(isinstance(method, str) and method for method in methods)
            or np.min(rep_losses) < -1e-12 or np.max(rep_losses) > 4 + 1e-12):
        raise ValueError("Need bounded repetition-by-method primary-cell risk means")
    required = (geometry_method, baseline_method, randomized_method)
    if len(set(required)) != 3 or any(method not in methods for method in required):
        raise ValueError("Three distinct named primary methods are required")
    if not np.isscalar(margin) or not np.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("Relative material margin must lie strictly between zero and one")
    G, B, R = (rep_losses[:, methods.index(method)] for method in required)
    scores = {"geometry_over_baseline": (1 - margin) * B - G,
              "geometry_over_regrouped": (1 - margin) * R - G,
              "baseline_over_geometry": (1 - margin) * G - B,
              "regrouped_over_geometry": (1 - margin) * G - R}
    pairs = ((geometry_method, baseline_method), (geometry_method, randomized_method),
             (baseline_method, geometry_method), (randomized_method, geometry_method))
    rows = [{"contrast": name, "favored_method": favored, "reference_method": reference,
             "material_margin": float(margin),
             **contrast_interval(values, -4., 4 * (1 - margin), delta)}
            for (name, values), (favored, reference) in zip(scores.items(), pairs)]
    return rows, scores
