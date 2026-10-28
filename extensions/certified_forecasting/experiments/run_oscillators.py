#!/usr/bin/env python3
"""Independent-episode nonlinear oscillator follow-up.

This harness explicitly replaces run_forecasting.make_episodes with the
generator defined here, then calls the unchanged run_forecasting.run_one.
The learning, frozen-candidate family, confidence allocation, and selection
code are exactly those of the main campaign.

The certified target is a bounded noisy sensor, not the unbounded physical
position or velocity. The generator is a fixed float64 RK4 numerical law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import sklearn

import run_forecasting as main_runner


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parents[1]
SELECTED_FAMILIES = [
    ("kahm_nested", "KAHM blocks"),
    ("rbf_nested", "RBF blocks"),
    ("affine_nested", "Affine blocks"),
    ("ridge_ar", "Ridge AR"),
    ("polynomial", "Quadratic direct"),
    ("kahm_koopman", "KAHM rollout"),
    ("polynomial_koopman", "Quadratic rollout"),
    ("persistence", "Persistence"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oscillator_episodes(system, n_episodes, length, seed, *, dt=0.025,
                        observation_interval=0.1, burn_time=10.0):
    """Independently initialized, bounded scalar sensor episodes.

    Van der Pol: dx=v; dv=(1-x**2)*v-x.
    Duffing: dx=v; dv=-.2*v+x-x**3+.3*cos(1.2*t+phase).
    Each episode has iid x0,v0~Uniform[-2,2], and an independent phase~U[0,2pi].
    Observation noise is iid Uniform[-.02,.02], generated before integration.
    dt changes therefore retain exactly the same initialization and noise.
    """
    n, length = int(n_episodes), int(length)
    if system not in {"vanderpol", "duffing"} or n < 1 or length < 2:
        raise ValueError("Invalid oscillator name, episode count, or length")
    if dt <= 0 or observation_interval <= 0 or burn_time < 0:
        raise ValueError("Integration times must be positive (burn can be zero)")
    steps_between = int(round(observation_interval / dt))
    burn_steps = int(round(burn_time / dt))
    if (steps_between < 1
            or not np.isclose(steps_between * dt, observation_interval, rtol=0, atol=1e-12)
            or not np.isclose(burn_steps * dt, burn_time, rtol=0, atol=1e-12)):
        raise ValueError("Observation interval and burn time must be integer multiples of dt")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-2.0, 2.0, size=(n, 2))
    x, velocity = initial[:, 0].copy(), initial[:, 1].copy()
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
    noise = rng.uniform(-0.02, 0.02, size=(n, length))
    observed = np.empty((n, length))

    def acceleration(position, speed, t):
        if system == "vanderpol":
            return (1.0 - position * position) * speed - position
        return (-0.2 * speed + position - position**3
                + 0.3 * np.cos(1.2 * t + phase))

    total_steps = burn_steps + (length - 1) * steps_between
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for step in range(total_steps + 1):
            if step >= burn_steps and (step - burn_steps) % steps_between == 0:
                index = (step - burn_steps) // steps_between
                observed[:, index] = 0.95 * np.tanh(x) + noise[:, index]
            if step == total_steps:
                break
            t = step * dt
            k1x = velocity
            k1v = acceleration(x, velocity, t)
            k2x = velocity + 0.5 * dt * k1v
            k2v = acceleration(x + 0.5 * dt * k1x, k2x, t + 0.5 * dt)
            k3x = velocity + 0.5 * dt * k2v
            k3v = acceleration(x + 0.5 * dt * k2x, k3x, t + 0.5 * dt)
            k4x = velocity + dt * k3v
            k4v = acceleration(x + dt * k3x, k4x, t + dt)
            x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            velocity = velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            if not np.isfinite(x).all() or not np.isfinite(velocity).all():
                raise FloatingPointError(f"{system}: non-finite state at integration step {step}")
    if not np.isfinite(observed).all() or np.any(np.abs(observed) > 0.97 + 1e-14):
        raise FloatingPointError("Generated sensor data violate the mathematical range")
    return observed[:, :, None]


def check_integrator(cfg):
    settings = cfg["generator"]
    rows = []
    for system in cfg["systems"]:
        arguments = (system, settings["numerical_check_episodes"],
                     cfg["episode_length"], settings["numerical_check_seed"])
        coarse = oscillator_episodes(*arguments, dt=settings["dt"],
                                     observation_interval=settings["observation_interval"],
                                     burn_time=settings["burn_time"])
        fine = oscillator_episodes(*arguments, dt=settings["dt"] / 2.0,
                                   observation_interval=settings["observation_interval"],
                                   burn_time=settings["burn_time"])
        error = coarse - fine
        maximum = float(np.max(np.abs(error)))
        tolerance = settings["numerical_check_max_sensor_difference"]
        rows.append({
            "system": system, "coarse_dt": settings["dt"], "fine_dt": settings["dt"] / 2,
            "seed": settings["numerical_check_seed"], "n_episodes": len(coarse),
            "same_initial_conditions_phases_and_observation_noise": True,
            "maximum_sensor_absolute_difference": maximum,
            "sensor_root_mean_squared_difference": float(np.sqrt(np.mean(error**2))),
            "predeclared_maximum_difference_tolerance": tolerance,
            "passed": bool(maximum <= tolerance),
        })
    return {
        "checks": rows, "all_passed": all(row["passed"] for row in rows),
        "interpretation": "A fixed matched-randomness half-step consistency check; not a rigorous global ODE error bound.",
    }


def source_hashes():
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    hashes["experiments/run_forecasting.py"] = digest(ROOT / "experiments/run_forecasting.py")
    hashes["experiments/run_oscillators.py"] = digest(__file__)
    for p in sorted(PARENT.glob("*.py")):
        hashes["parent/" + p.name] = digest(p)
    return hashes


def aggregate(cfg, output):
    rows = []
    for system in cfg["systems"]:
        records = [json.loads((output / f"{system}_{seed}" / "results.json").read_text())
                   for seed in cfg["seeds"]]
        for family, label in SELECTED_FAMILIES:
            selected = []
            for record in records:
                candidates = [r for r in record["results"] if r["family"] == family]
                chosen = min(candidates, key=lambda r: (float(np.mean(r["validation_mse"])),
                                                        r["n_features"], r["name"]))
                selected.append(chosen)
            errors = np.sqrt(np.asarray([r["test_mse"] for r in selected]))
            upper = np.asarray([[c["rmse_upper"] for c in r["certificates"]] for r in selected])
            rows.append({
                "system": system, "family": family, "label": label,
                "selected_candidates": [r["name"] for r in selected],
                "mean_test_rmse": errors.mean(axis=0).tolist(),
                "sd_test_rmse_over_training_repetitions": errors.std(axis=0, ddof=1).tolist(),
                "mean_certified_rmse_upper": upper.mean(axis=0).tolist(),
                "mean_selected_features": float(np.mean([r["n_features"] for r in selected])),
                "n_all_horizon_certified_at_0_35": int(np.sum(np.all(upper <= 0.35, axis=1))),
                "n_training_repetitions": len(selected),
            })
    summary = {
        "rows": rows, "horizons": cfg["horizons"],
        "physical_forecast_durations": [h * cfg["generator"]["observation_interval"] for h in cfg["horizons"]],
        "selection_rule": "Within each family, minimize mean validation MSE over horizons; ties use fewer features, then name.",
        "units": "bounded noisy sensor y=0.95*tanh(x)+Uniform[-0.02,0.02]",
        "uncertainty": "Sample SD over three full learning repetitions; the test episodes are independently regenerated for each repetition.",
        "confidence_scope": "At least95% per full frozen40-candidate family and its paired refinements, not jointly across six runs.",
    }
    main_runner.write_json(output / "summary.json", summary)
    return summary


def write_latex_table(summary, destination):
    table = [
        "% Generated by experiments/run_oscillators.py; no manually entered results.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"System & Family & $h=1$ & $h=4$ & $h=12$ & $\overline U_{12}^{1/2}$ \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        system = "Van der Pol" if row["system"] == "vanderpol" else "Duffing"
        values = ["$" + f"{mean:.3f}\\pm{sd:.3f}" + "$" for mean, sd in zip(
            row["mean_test_rmse"], row["sd_test_rmse_over_training_repetitions"])]
        table.append(f"{system} & {row['label']} & " + " & ".join(values)
                     + f" & {row['mean_certified_rmse_upper'][-1]:.3f}" + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/oscillators_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/oscillators_v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty oscillator campaign")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    manifest = {
        "configuration": cfg, "configuration_sha256": digest(args.config),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "source_sha256": source_hashes(),
        "generator_substitution": "Explicitly assign main_runner.make_episodes to this frozen generator; call unchanged main_runner.run_one.",
        "target": "Bounded sensor output, not physical position or velocity.",
        "no_stationarity_assumption": "Calibration and deployment are independent episodes drawn from the identical finite-time initialization and sensor law.",
    }
    main_runner.write_json(args.output / "manifest.json", manifest)
    numerical = check_integrator(cfg)
    main_runner.write_json(args.output / "numerical_check.json", numerical)
    if not numerical["all_passed"]:
        raise RuntimeError("The predeclared matched-randomness numerical consistency check failed")
    settings = cfg["generator"]

    def generator(system, n_episodes, length, seed):
        return oscillator_episodes(
            system, n_episodes, length, seed, dt=settings["dt"],
            observation_interval=settings["observation_interval"],
            burn_time=settings["burn_time"],
        )

    # This explicit harness-level substitution is recorded in the manifest.
    main_runner.make_episodes = generator
    started = time.perf_counter()
    for system in cfg["systems"]:
        for seed in cfg["seeds"]:
            directory = args.output / f"{system}_{seed}"
            directory.mkdir()
            print(f"START {system} seed={seed}", flush=True)
            try:
                record = main_runner.run_one(cfg, system, seed, directory)
            except Exception as error:
                main_runner.write_json(directory / "failure.json",
                                       {"error": repr(error), "traceback": traceback.format_exc()})
                raise
            print(f"DONE {system} seed={seed} candidates={record['n_candidates']} "
                  f"seconds={record['elapsed_seconds']:.2f}", flush=True)
    summary = aggregate(cfg, args.output)
    write_latex_table(summary, ROOT / "paper/tables/oscillators_table.tex")
    main_runner.write_json(args.output / "completion.json", {
        "scheduled_runs": len(cfg["systems"]) * len(cfg["seeds"]),
        "completed_runs": len(list(args.output.glob("*/results.json"))),
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
