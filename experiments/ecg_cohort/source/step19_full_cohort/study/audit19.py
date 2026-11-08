"""Independent Step19 saved-evidence audit and conditional exact postverification.

No producer, optimizer, waveform loader, fit routine or selection routine is
imported. `preflight` uses manufactured inputs only. `verify` requires frozen,
sealed producer evidence. `replay` recomputes saved exact evidence read-only.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import exact19 as E

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BASE = PROJ.parent
RUN = HERE / "run"
PARENT = BASE / "step18_natural_decisions" / "study" / "run"
PREFIXES = (2, 6, 14)
FAMILIES = ("consecutive", "greedy_nearest")
LEVELS = (32, 64, 128)
M = 8192
need = E.need


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def json_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def npz_new(path, **values):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **values)


def near(actual, expected, label, atol=3e-10, rtol=3e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    need(actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label+" shape/nonfinite mismatch")
    error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    need(np.all(np.abs(actual-expected) <= atol + rtol*np.abs(expected)), label+f" discrepancy {error}")
    return error


def slices_for(prefix):
    need(prefix in PREFIXES, "Unknown frozen prefix")
    return tuple((a, b) for a, b in ((0, 2), (2, 6), (6, 14)) if b <= prefix)


def partition_independent(features, family, block_size=32):
    """Independent orchestration of the same frozen float64 tie kernel."""
    f = E.f64_array(features, 2, "partition features")
    need(len(f) % block_size == 0, "Partition cannot cover the full population")
    if family == "consecutive":
        return np.arange(len(f), dtype=np.int64).reshape(-1, block_size)
    need(family == "greedy_nearest", "Unknown base family")
    remaining = set(range(len(f))); groups = []
    while remaining:
        anchor = min(remaining); remaining.remove(anchor)
        candidates = np.array(sorted(remaining), dtype=np.int64)
        difference = f[candidates]-f[anchor]
        # The reduction itself is part of the frozen near-tie rule, as in18.
        distances = np.einsum("ij,ij->i", difference, difference)
        order = np.lexsort((candidates, distances))
        chosen = candidates[order[:block_size-1]].tolist()
        need(len(chosen) == block_size-1, "Incomplete greedy group")
        groups.append([anchor]+chosen); remaining.difference_update(chosen)
    result = np.asarray(groups, dtype=np.int64)
    need(np.array_equal(np.sort(result.ravel()), np.arange(len(f))), "Independent partition not disjoint/full")
    return result


def independent_distances(features, radii, blocks):
    x, radii = np.asarray(features), np.asarray(radii)
    count, b, d = x.shape
    need(radii.shape == (count, b, len(blocks)) and np.all(radii >= 0), "Radius layout/range")
    nominal = np.zeros((count, b, b)); upper = np.zeros_like(nominal)
    extended = x.astype(np.longdouble)
    for k, (a, z) in enumerate(blocks):
        sq = np.zeros_like(nominal); ext_sq = np.zeros_like(nominal, dtype=np.longdouble)
        for q in range(a, z):
            dx = x[:, :, q, None]-x[:, None, :, q]
            dx_ext = extended[:, :, q, None]-extended[:, None, :, q]
            sq += dx*dx; ext_sq += dx_ext*dx_ext
        nominal = np.maximum(nominal, np.sqrt(sq))
        rr = radii[:, :, k].astype(np.longdouble)
        outward = np.nextafter(np.asarray(np.sqrt(ext_sq)+rr[:, :, None]+rr[:, None, :], dtype=float), np.inf)
        upper = np.maximum(upper, outward)
    upper = np.maximum(upper, nominal)
    diagonal = np.arange(b); upper[:, diagonal, diagonal] = 0.; nominal[:, diagonal, diagonal] = 0.
    return nominal, upper


def replay_chunk(chunk, features, targets, radii, blocks, kappa, previous_guarded=None, predictions=None):
    """Audit retained float diagnostics and source choice without rerunning FISTA."""
    index = np.asarray(chunk["indices"])
    need(index.dtype == np.int64 and index.ndim == 2 and 2 <= index.shape[1] <= 128, "Chunk indices")
    need(np.unique(index).size == index.size and np.all(index >= 0) and np.all(index < len(targets)), "Chunk repeats/out-of-range rows")
    x, y, r = features[index], targets[index], radii[index]
    count, b = index.shape; ii, jj = np.triu_indices(b, 1); edges = len(ii)
    nominal, upper = independent_distances(x, r, blocks)
    errors = [near(chunk["nominal_distance"], nominal, "nominal distances"),
              near(chunk["distance_upper"], upper, "corrected distances"),
              near(chunk["feature_block_radius"], r, "inherited radii", 0., 0.)]
    incidence = np.zeros((b, edges))
    incidence[ii, np.arange(edges)] = 1.; incidence[jj, np.arange(edges)] = -1.
    outputs = {}
    for candidate in ("new", "embedded"):
        get = lambda key: chunk[candidate+"_"+key]
        alpha = np.asarray(get("signed_dual")); primal = np.asarray(get("primal"))
        need(alpha.shape == (count, edges) and primal.shape == (count, b), "Candidate alpha/primal shape")
        need(np.all(np.isfinite(alpha)) and np.all(np.isfinite(primal)), "Nonfinite candidate")
        need(np.max(np.abs(primal)) <= 1+1e-12, "Primal outside target range")
        divergence = alpha @ incidence.T
        linear = np.einsum("ij,ij->i", alpha, y[:, ii]-y[:, jj])
        quadratic = .25*np.einsum("ij,ij->i", divergence, divergence)
        nominal_penalty = kappa*np.einsum("ij,ij->i", np.abs(alpha), nominal[:, ii, jj])
        penalty = kappa*np.einsum("ij,ij->i", np.abs(alpha), upper[:, ii, jj])
        nominal_raw = linear-nominal_penalty-quadratic
        nominal_guard = 1e-11*(1+abs(linear)+abs(nominal_penalty)+abs(quadratic))
        nominal_lower = np.maximum(0., nominal_raw-nominal_guard)
        raw = linear-penalty-quadratic
        guard = 1e-11*(1+abs(linear)+abs(penalty)+abs(quadratic))
        lower = np.minimum(nominal_lower, np.maximum(0., raw-guard))
        sse = np.sum((y-primal)**2, axis=1); sse_upper = sse+1e-11*(1+abs(sse))
        violation = np.maximum(0., np.max(abs(primal[:, ii]-primal[:, jj])-kappa*upper[:, ii, jj], axis=1))
        nominal_violation = np.maximum(0., np.max(abs(primal[:, ii]-primal[:, jj])-kappa*nominal[:, ii, jj], axis=1))
        need(np.max(violation) <= 1e-8, "Relaxed primal infeasible")
        need(np.all(lower <= sse_upper+1e-8), "Replayed weak duality failed")
        terms = {"dual_linear": linear, "dual_penalty": penalty, "dual_quadratic": quadratic,
                 "dual_raw": raw, "dual_guard": guard, "nominal_dual_raw": nominal_raw,
                 "nominal_dual_guard": nominal_guard, "nominal_sse_lower": nominal_lower,
                 "nominal_sse_upper": sse_upper, "sse_lower": lower, "sse_upper": sse_upper,
                 "lower": lower/(b-1), "upper": sse_upper/(b-1), "gap": (sse_upper-lower)/(b-1),
                 "nominal_gap": (sse_upper-nominal_lower)/(b-1),
                 "feasibility_residual": violation, "sse_lower_uncertainty_adjustment": nominal_lower-lower,
                 "dual_penalty_inflation": penalty-nominal_penalty}
        for key, value in terms.items():
            errors.append(near(get(key), value, candidate+" "+key))
        errors.append(near(chunk[candidate+"_nominal_primal_feasibility_residual"], nominal_violation, candidate+" nominal primal"))
        need(np.array_equal(get("converged"), get("gap") <= 1e-7), "Corrected convergence flag")
        iterations = np.asarray(get("iterations"))
        if candidate == "new":
            need(np.issubdtype(iterations.dtype, np.integer) and np.all((iterations >= 20) & (iterations <= 4000) & (iterations % 20 == 0)), "Optimizer iterations outside frozen controls")
            need(np.all(iterations[~get("nominal_converged")] == 4000), "Nonconverged optimizer stopped before limit")
        else:
            need(np.all(iterations == 0), "Embedding was counted as a fresh optimization")
        outputs[candidate] = dict(alpha=alpha, lower=np.asarray(get("sse_lower")), upper=np.asarray(get("sse_upper")), primal=primal)
    need(np.array_equal(outputs["new"]["primal"], outputs["embedded"]["primal"]), "Embedded candidate must use full-parent new primal")
    if previous_guarded is None:
        need(np.all(outputs["embedded"]["alpha"] == 0), "Base level inherited witness not zero")
        need(chunk["child_edge_to_parent_edge"].shape == (0, 0), "Base embedding map")
    else:
        child_ids = chunk["child_block_ids"]
        need(child_ids.shape == (count, 2), "Parent must have exactly two children")
        inherited = np.stack([E.embed_children(previous_guarded[int(i)], previous_guarded[int(j)]) for i, j in child_ids])
        need(np.array_equal(outputs["embedded"]["alpha"], inherited), "Guarded child alpha zero extension differs")
        mapping = {(i, j): e for e, (i, j) in enumerate(zip(ii.tolist(), jj.tolist(), strict=True))}
        expected_map = np.asarray([[mapping[(i+offset, j+offset)] for i in range(b//2) for j in range(i+1, b//2)] for offset in (0, b//2)])
        need(np.array_equal(chunk["child_edge_to_parent_edge"], expected_map), "Child edge map corrupted")
    choose = outputs["embedded"]["lower"] > outputs["new"]["lower"]
    need(np.array_equal(chunk["chosen_source"], choose.astype(np.int8)), "Guarded selection/tie rule differs")
    need(np.array_equal(chunk["chosen_signed_dual"], np.where(choose[:, None], outputs["embedded"]["alpha"], outputs["new"]["alpha"])), "Chosen guarded witness differs")
    chosen_lower = np.where(choose, outputs["embedded"]["lower"], outputs["new"]["lower"])
    need(np.array_equal(chunk["chosen_sse_lower"], chosen_lower), "Chosen lower not selected candidate")
    need(np.array_equal(chunk["chosen_sse_upper"], outputs["new"]["upper"]), "Chosen upper/primal source")
    errors.append(near(chunk["chosen_gap"], (chunk["chosen_sse_upper"]-chosen_lower)/(b-1), "Chosen gap"))
    need(np.array_equal(chunk["chosen_converged"], chunk["chosen_gap"] <= 1e-7), "Chosen convergence flag")
    for key in ("dual_linear", "dual_penalty", "dual_quadratic", "dual_raw", "dual_guard", "nominal_sse_lower", "sse_lower_uncertainty_adjustment"):
        need(np.array_equal(chunk["chosen_"+key], np.where(choose, chunk["embedded_"+key], chunk["new_"+key])), "Chosen diagnostic source differs: "+key)
    constant = np.sum((y-y.mean(axis=1)[:, None])**2, axis=1)
    errors.append(near(chunk["constant_sse"], constant, "Constant feasible SSE"))
    need(np.all(chosen_lower <= constant+1e-8), "Chosen lower exceeds constant feasible SSE")
    if predictions is not None:
        p = predictions[index]
        violation = np.maximum(0., np.max(abs(p[:, ii]-p[:, jj])-kappa*upper[:, ii, jj], axis=1))
        errors.append(near(chunk["fitted_sse"], np.sum((y-p)**2, axis=1), "Frozen fitted SSE"))
        errors.append(near(chunk["fitted_relaxed_feasibility_residual"], violation, "Frozen fitted feasibility"))
        need(np.max(violation) <= 1e-8 and np.all(chosen_lower <= chunk["fitted_sse"]+1e-8), "Frozen predictor witness comparison")
    return {"max_float_replay_error": max(errors, default=0.), "blocks": count,
            "edges": count*edges, "new_iteration_limits": int(np.count_nonzero(~chunk["new_nominal_converged"])),
            "chosen_embedded": int(np.count_nonzero(choose))}


def _path_hash_map(mapping):
    for name, digest in mapping.items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts, "Nonportable frozen path")
        need(sha(BASE/path) == digest, "Frozen source/input changed: "+name)


def verify_source_and_producer_seals():
    frozen = read(RUN/"freeze.json"); protocol = read(PROJ/"PROTOCOL.json")
    _path_hash_map(frozen["sources"]); _path_hash_map(frozen["inputs"])
    need(frozen["protocol_sha256"] == sha(PROJ/"PROTOCOL.json"), "Protocol hash mismatch")
    need(protocol["exact_postverification"]["coordinate_grid_power"] == E.GRID_POWER and
         protocol["exact_postverification"]["coordinate_grid_denominator"] == E.Q, "Grid changed")
    need(protocol["population"]["size"] == M and protocol["constraint_schedule"]["number_configurations"] == 18,
         "Population/configuration count changed")
    need(protocol["constraint_schedule"]["block_sizes"] == list(LEVELS) and protocol["constraint_schedule"]["base_families"] == list(FAMILIES), "Constraint schedule changed")
    need(protocol["optimization"]["max_iter"] == 4000 and protocol["optimization"]["tolerance"] == 1e-7 and protocol["optimization"]["check_every"] == 20 and protocol["optimization"]["chunk_blocks"] == 16, "Optimizer controls changed")
    for path in (HERE/"exact19.py", HERE/"audit19.py", HERE/"check_exact19.py"):
        need(frozen["sources"].get(path.relative_to(BASE).as_posix()) == sha(path), "Exact audit source absent from freeze")
    seal = read(RUN/"producer_seal.json")
    names = [RUN/"inputs.npz", RUN/"input_manifest.json", RUN/"summary.json"]
    names += [path for path in (RUN/"results").rglob("*") if path.is_file()]
    actual = {path.relative_to(RUN).as_posix(): sha(path) for path in sorted(names)}
    need(seal["freeze_sha256"] == sha(RUN/"freeze.json") and seal["files"] == actual, "Producer seal mismatch")
    return frozen, protocol, seal


def verify_input_identity():
    data = arrays(RUN/"inputs.npz"); metadata = read(RUN/"input_manifest.json")
    need(metadata["input_arrays_sha256"] == sha(RUN/"inputs.npz") and metadata["freeze_sha256"] == sha(RUN/"freeze.json"), "Input manifest detached")
    observations = {s: arrays(PARENT/s/"observations.npz") for s in ("cal", "terminal")}
    parent_features = {s: arrays(PARENT/s/"features.npz") for s in ("cal", "terminal")}
    parent_predictions = {s: arrays(PARENT/s/"predictions.npz") for s in ("cal", "terminal")}
    selected = read(PARENT/"fit/selected.json"); census = read(PARENT/"census/results.json")
    for key in ("Y", "patient_id", "ecg_id", "origin"):
        expected = np.concatenate([observations[s][key] for s in ("cal", "terminal")])
        need(np.array_equal(data[key], expected) and data[key].dtype == expected.dtype, "Changed parent observation/order: "+key)
    need(data["Y"].shape == (M,) and len(np.unique(data["patient_id"])) == M, "Full cohort identity/count")
    for prefix in PREFIXES:
        p = str(prefix)
        need(float(metadata["selected_kappa"][p]) == float(selected["kahm"][p]["kappa"]) == 1., "Frozen selected kappa differs")
        need(metadata["selected_candidate_ids"][p] == selected["kahm"][p]["id"] and metadata["selected_geometry"][p] == selected["kahm"][p]["geometry"], "Frozen representation identity differs")
        need(metadata["feature_blocks"][p] == [list(pair) for pair in slices_for(prefix)], "Feature block layout differs")
        for field in ("F", "F_lo", "F_hi", "block_radius"):
            key = field+"_"+p; expected = np.concatenate([parent_features[s][key] for s in ("cal", "terminal")])
            need(np.array_equal(data[key], expected), "Changed inherited feature data: "+key)
        for field in ("pred", "prediction_lo", "prediction_hi", "loss_lower", "loss_upper"):
            key = field+"_"+p; expected = np.concatenate([parent_predictions[s][key] for s in ("cal", "terminal")])
            need(np.array_equal(data[key], expected), "Changed frozen predictor data: "+key)
        need(data["F_"+p].shape == (M, prefix) and np.all(data["F_lo_"+p] <= data["F_"+p]) and np.all(data["F_"+p] <= data["F_hi_"+p]), "Canonical box containment of nominal values")
    reference = []
    for stage in ("cal", "terminal"):
        pred = parent_predictions[stage]
        column = pred["baseline_candidate_ids"].tolist().index(selected["reference"]["id"])
        need(pred["baseline_slots"][column] == selected["reference"]["selection_slot"], "Reference identity/slot mismatch")
        reference.append(pred["baseline_predictions"][:, column])
    need(np.array_equal(data["reference_pred"], np.concatenate(reference)), "Frozen reference prediction changed")
    for key in ("threshold", "threshold_lower", "threshold_upper"):
        need(metadata[key] == census[key], "Existing Step18 threshold changed")
    need(metadata["reference_id"] == selected["reference"]["id"] and metadata["reference_slot"] == selected["reference"]["selection_slot"], "Reference selected after prior validation")
    return data, metadata


def unnecessary_kl_diagnostic(lower):
    # Explicitly a counterfactual floating diagnostic; never a decision input.
    q = float(Fraction(lower)/4); radius = math.log(180)/M
    if q == 0: return 0.
    lo, hi = 0., q
    for _ in range(100):
        middle = (lo+hi)/2
        if middle == lo or middle == hi: break
        if middle == 0: divergence = math.inf
        elif q == 1: divergence = -math.log(middle)
        else: divergence = q*math.log(q/middle)+(1-q)*math.log((1-q)/(1-middle))
        if divergence > radius: lo = middle
        else: hi = middle
    return max(0., math.nextafter(4*lo, -math.inf))


def _guarded_mean(values):
    total = float(np.sum(np.asarray(values), dtype=np.longdouble))
    if total == 0: return 0.
    return float(np.nextafter(np.nextafter(total, 0.)/M, 0.))


def necessary_regularity(intercept, penalty_coefficient, population, threshold):
    """Necessary kappa from one fixed exact witness; no sufficiency assertion."""
    intercept, penalty_coefficient, threshold = map(Fraction, (intercept, penalty_coefficient, threshold))
    need(isinstance(population, int) and population > 0 and penalty_coefficient >= 0,
         "Invalid regularity-diagnostic inputs")
    excess = intercept-population*threshold
    if penalty_coefficient == 0:
        return {"status": "all_finite_kappa_excluded" if excess > 0 else "no_positive_requirement",
                "necessary_kappa_lower": None if excess > 0 else E.rational_record(Fraction(0)),
                "strict_exclusion_at_bound": False, "adequacy_above_bound_claimed": False}
    bound = max(Fraction(0), excess/penalty_coefficient)
    need((intercept-penalty_coefficient)/population > threshold if bound > 1
         else (intercept-penalty_coefficient)/population <= threshold,
         "Necessary regularity/primary kappa-one decision mismatch")
    if bound > 0:
        need((intercept-bound*penalty_coefficient)/population == threshold,
             "Exact regularity boundary does not give equality")
    return {"status": "finite_necessary_bound", "necessary_kappa_lower": E.rational_record(bound),
            "strict_exclusion_at_bound": False, "adequacy_above_bound_claimed": False}


def best_regularity(records, key):
    """Take a maximum only over witnesses for one frozen representation."""
    infinite = [record for record in records if record[key]["status"] == "all_finite_kappa_excluded"]
    selected = infinite[0] if infinite else max(records, key=lambda record: E.from_record(record[key]["necessary_kappa_lower"]))
    return {"prefix": selected["prefix"], "family": selected["family"], "block_size": selected["block_size"],
            **selected[key]}


def _compare_exact_saved(path, record, chosen, replay):
    if replay:
        saved = read(path/"blocks.json")
        need(saved == record, "Exact block rational/source/hash records changed")
        actual = arrays(path/"selected_witnesses.npz")
        need(set(actual) == {"indices", "signed_dual", "source_codes"}, "Exact witness artifact fields")
        for key, value in chosen.items():
            need(np.array_equal(actual[key], value), "Exact selected witness differs: "+key)
    else:
        path.mkdir(parents=True, exist_ok=False)
        json_new(path/"blocks.json", record)
        npz_new(path/"selected_witnesses.npz", **chosen)


def run_audit(replay=False):
    started = time.perf_counter()
    frozen, protocol, seal = verify_source_and_producer_seals()
    data, metadata = verify_input_identity()
    exact_root = RUN/"exact"
    if replay:
        old_summary = read(exact_root/"summary.json")
        old_seal = read(RUN/"exact_seal.json")
        actual = {path.relative_to(RUN).as_posix(): sha(path) for path in sorted(exact_root.rglob("*")) if path.is_file()}
        need(old_seal["files"] == actual and old_seal["producer_seal_sha256"] == sha(RUN/"producer_seal.json"), "Exact seal changed")
    else:
        need(not exact_root.exists() and not (RUN/"verification.json").exists(), "Exact verification is write-once")
        exact_root.mkdir()
    reference_mse = E.exact_mse(data["Y"], data["reference_pred"])
    threshold_exact = Fraction(11, 10)*reference_mse
    threshold_upper = Fraction.from_float(float(metadata["threshold_upper"]))
    threshold_lower = Fraction.from_float(float(metadata["threshold_lower"]))
    need(threshold_lower <= threshold_exact <= threshold_upper, "Existing threshold enclosure misses exact reference requirement")
    need(Fraction.from_float(float(metadata["reference_mse_lower"])) <= reference_mse <= Fraction.from_float(float(metadata["reference_mse_upper"])), "Existing reference risk enclosure misses exact dyadic census")
    configurations, quantization, replay_metrics = [], {}, []
    total_programs = total_edges = 0
    producer_summary = read(RUN/"summary.json")
    for prefix in PREFIXES:
        p = str(prefix); blocks = slices_for(prefix); kappa = float(metadata["selected_kappa"][p])
        lo, hi, qmetrics = E.quantize_boxes(data["F_lo_"+p], data["F_hi_"+p]); quantization[p] = qmetrics
        for family in FAMILIES:
            partition_started = time.perf_counter()
            base = partition_independent(data["F_"+p], family)
            folder = RUN/"results"/f"p{prefix}"/family
            need(np.array_equal(arrays(folder/"base_partition.npz")["indices"], base), "Base partition differs from frozen rule")
            partition_info = read(folder/"partition.json")
            need(partition_info["base_partition_sha256"] == sha(folder/"base_partition.npz"), "Base partition receipt")
            partition_seconds = time.perf_counter()-partition_started
            previous_exact = previous_values = previous_l1 = previous_quantization = previous_guarded = previous_float_lower = previous_components = None
            previous_risk = Fraction(0); previous_guarded_retained = 0.
            for b in LEVELS:
                level_started = time.perf_counter(); directory = folder/f"b{b}"
                level_summary = read(directory/"summary.json"); nesting = arrays(directory/"nesting.npz")
                indices = base.reshape(-1, b); count = len(indices); edges = b*(b-1)//2
                base_ids = np.arange(M//32, dtype=np.int64).reshape(-1, b//32)
                child_ids = np.empty((count, 0), dtype=np.int64) if b == 32 else np.arange(M//(b//2), dtype=np.int64).reshape(-1, 2)
                for key, expected in (("indices", indices), ("base_group_ids", base_ids), ("child_block_ids", child_ids)):
                    need(np.array_equal(nesting[key], expected), "Nested partition/map differs: "+key)
                need(level_summary["nesting_sha256"] == sha(directory/"nesting.npz"), "Nesting receipt")
                need(level_summary["blocks"] == count and level_summary["new_block_programs"] == count and level_summary["complete_undirected_edges"] == count*edges, "Level program/edge census")
                chosen = np.empty((count, edges), dtype=np.float64)
                source_codes = np.empty(count, dtype=np.int8); values = []; l1_values = []; quant_values = []; selected_components = []
                block_records = []; current_float = []; guarded_duals = []; chunks = []
                exact_seconds = distance_seconds = float_seconds = 0.; max_error = 0.
                new_active_edges = 0; selected_counts = Counter(); exact_guard_choice_differences = 0
                for chunk_id, start in enumerate(range(0, count, 16)):
                    stop = min(count, start+16); path = directory/f"chunk{chunk_id:03d}.npz"
                    z = arrays(path); info = read(path.with_suffix(".json")); chunks.append(info)
                    need(info["artifact_sha256"] == sha(path) and info["artifact_bytes"] == path.stat().st_size, "Chunk identity")
                    need(info["field_names"] == sorted(z) and info["saved_array_bytes"] == sum(v.nbytes for v in z.values()), "Chunk field/array-byte census")
                    need(info["new_block_programs"] == stop-start and info["complete_undirected_edges"] == (stop-start)*edges, "Chunk program/edge count")
                    for key, expected in (("indices", indices[start:stop]), ("block_ids", np.arange(start, stop)), ("base_group_ids", base_ids[start:stop]), ("child_block_ids", child_ids[start:stop])):
                        need(np.array_equal(z[key], expected), "Chunk nesting/index membership differs: "+key)
                    before = time.perf_counter()
                    checks = replay_chunk(z, data["F_"+p], data["Y"], data["block_radius_"+p], blocks, kappa, previous_guarded, data["pred_"+p])
                    float_seconds += time.perf_counter()-before; max_error = max(max_error, checks["max_float_replay_error"])
                    need(info["new_solver_iteration_limits"] == checks["new_iteration_limits"] and info["chosen_embedded_count"] == checks["chosen_embedded"], "Chunk convergence/selection counts")
                    for kind in ("new", "embedded", "chosen"):
                        need(info[kind+"_signed_witness_bytes"] == z[kind+"_signed_dual"].nbytes, "Signed witness byte census")
                    if b > 32:
                        expected_child = previous_float_lower[child_ids[start:stop]]
                        need(np.array_equal(z["child_guarded_sse_lower"], expected_child), "Inherited guarded child values differ")
                        expected_sum = np.nextafter(np.asarray(np.sum(expected_child.astype(np.longdouble), axis=1), dtype=float), 0.)
                        need(np.array_equal(z["child_guarded_sse_sum"], expected_sum), "Inherited guarded child sum")
                        need(np.array_equal(z["embedded_guarded_minus_child_sum"], z["embedded_sse_lower"]-expected_sum), "Embedded-vs-child diagnostic")
                    before = time.perf_counter()
                    distances, dmetrics = E.edge_upper_numerators(lo, hi, indices[start:stop], blocks)
                    distance_seconds += time.perf_counter()-before
                    before = time.perf_counter()
                    for local, block_id in enumerate(range(start, stop)):
                        raw, components = E.exact_dual(z["new_signed_dual"][local], data["Y"][indices[block_id]], distances[local], kappa)
                        new_active_edges += components["active_edges"]
                        if b == 32:
                            alpha, value, source = E.exact_best(z["new_signed_dual"][local], raw)
                            child_value = None
                        else:
                            left, right = map(int, child_ids[block_id])
                            child_value = previous_values[left]+previous_values[right]
                            alpha, value, source = E.exact_best(z["new_signed_dual"][local], raw,
                                left_alpha=previous_exact[left], right_alpha=previous_exact[right],
                                left_value=previous_values[left], right_value=previous_values[right])
                            need(value >= child_value, "Exact nested block lower decreased")
                        if source == "new":
                            l1, quant = components["alpha_l1"], components["extra_quantization_sse_loss_bound"]
                            selected_terms = tuple(components[name] for name in ("linear", "quadratic", "distance_penalty"))
                        elif source == "zero":
                            l1 = quant = Fraction(0)
                            selected_terms = (Fraction(0),)*3
                        else:
                            l1, quant = previous_l1[left]+previous_l1[right], previous_quantization[left]+previous_quantization[right]
                            selected_terms = tuple(a+b for a, b in zip(previous_components[left], previous_components[right], strict=True))
                        need(value == selected_terms[0]-selected_terms[1]-selected_terms[2], "Exact selected component recursion differs")
                        selected_components.append(selected_terms)
                        chosen[block_id] = alpha; values.append(value); l1_values.append(l1); quant_values.append(quant)
                        source_codes[block_id] = {"zero": 0, "new": 1, "exact_children": 2}[source]; selected_counts[source] += 1
                        if not np.array_equal(alpha, z["chosen_signed_dual"][local]): exact_guard_choice_differences += 1
                        block_records.append({"block_id": block_id, "indices_sha256": E.array_hash(indices[block_id]),
                            "new_signed_dual_sha256": components["signed_dual_sha256"], "chosen_signed_dual_sha256": E.array_hash(alpha),
                            "source": source, "new_raw_sse": E.rational_record(raw), "exact_selected_sse": E.rational_record(value),
                            "exact_child_sum_sse": None if child_value is None else E.rational_record(child_value),
                            "selected_alpha_l1": E.rational_record(l1),
                            "selected_linear": E.rational_record(selected_terms[0]),
                            "selected_quadratic": E.rational_record(selected_terms[1]),
                            "selected_distance_penalty_at_kappa_one": E.rational_record(selected_terms[2]),
                            "selected_extra_quantization_sse_loss_bound": E.rational_record(quant),
                            "new_active_edges": components["active_edges"]})
                    exact_seconds += time.perf_counter()-before
                    current_float.append(z["chosen_sse_lower"]); guarded_duals.append(z["chosen_signed_dual"])
                need(len(list(directory.glob("chunk*.npz"))) == len(chunks) == (count+15)//16, "Extra/missing chunks")
                total = sum(values, Fraction(0)); risk = total/M
                linear_total, quadratic_total, penalty_total = [sum((terms[j] for terms in selected_components), Fraction(0)) for j in range(3)]
                intercept = linear_total-quadratic_total
                need(kappa == 1. and total == intercept-penalty_total, "Selected full-partition A/P decomposition differs")
                need(risk >= previous_risk, "Exact level lower decreased")
                need(Fraction(0) <= risk <= Fraction(4), "Exact class-risk lower outside bounded loss range")
                float_lower = np.concatenate(current_float)
                current_guarded = _guarded_mean(float_lower); retained = max(current_guarded, previous_guarded_retained)
                need(level_summary["current_chosen_witness_empirical_lower"] == current_guarded and level_summary["retained_guarded_empirical_lower"] == retained, "Guarded level aggregation changed")
                need(level_summary["retained_previous_level_empirical_lower"] == previous_guarded_retained, "Previous guarded level identity")
                need(level_summary["saved_signed_witness_bytes"] == sum(v[k+"_signed_witness_bytes"] for v in chunks for k in ("new", "embedded", "chosen")), "Level witness-byte count")
                need(level_summary["saved_array_bytes"] == sum(v["saved_array_bytes"] for v in chunks), "Level saved-array bytes")
                out = exact_root/f"p{prefix}"/family/f"b{b}"
                block_document = {"prefix": prefix, "family": family, "block_size": b, "source_codes": {"0": "zero", "1": "new", "2": "exact_children"},
                                  "blocks": block_records, "exact_sse_sum": E.rational_record(total), "exact_risk_lower": E.rational_record(risk)}
                _compare_exact_saved(out, block_document, {"indices": indices, "signed_dual": chosen, "source_codes": source_codes}, replay)
                record = {"prefix": prefix, "family": family, "block_size": b, "blocks": count, "population": M,
                    "exact_sse_sum": E.rational_record(total), "exact_lower": E.rational_record(risk),
                    "selected_linear_total": E.rational_record(linear_total),
                    "selected_quadratic_total": E.rational_record(quadratic_total),
                    "same_witness_intercept_A": E.rational_record(intercept),
                    "same_witness_distance_coefficient_P": E.rational_record(penalty_total),
                    "necessary_regularity_unchanged_threshold": necessary_regularity(intercept, penalty_total, M, threshold_upper),
                    "necessary_regularity_exact_reference_threshold": necessary_regularity(intercept, penalty_total, M, threshold_exact),
                    "strictly_excludes_unchanged_threshold_upper": risk > threshold_upper,
                    "strictly_excludes_exact_reference_requirement": risk > threshold_exact,
                    "exact_previous_level_lower": E.rational_record(previous_risk), "exact_nesting_verified": True,
                    "guarded_current_lower": current_guarded, "guarded_retained_lower": retained,
                    "exact_minus_guarded_retained": E.rational_record(risk-Fraction.from_float(retained)),
                    "same_census_KL_penalized_lower": unnecessary_kl_diagnostic(risk),
                    "same_census_KL_diagnostic_used_for_decision": False,
                    "extra_quantization_risk_loss_bound": E.rational_record(sum(quant_values, Fraction(0))/M),
                    "quantization_bound_not_subtracted_again": True,
                    "selected_sources": {source: selected_counts[source] for source in ("zero", "new", "exact_children")},
                    "exact_choice_differs_from_guarded_count": exact_guard_choice_differences,
                    "new_active_signed_edges": new_active_edges, "complete_new_edges": count*edges,
                    "exact_selected_signed_witness_bytes": chosen.nbytes,
                    "exact_selected_signed_witness_sha256": E.array_hash(chosen),
                    "exact_artifact_relative_directory": out.relative_to(RUN).as_posix(),
                    "max_independent_float_replay_error": max_error,
                    "timing_seconds": {"integer_distances": distance_seconds, "exact_objectives_and_selection": exact_seconds,
                        "independent_float_replay": float_seconds, "level_total": time.perf_counter()-level_started,
                        "base_partition_reconstruction": partition_seconds if b == 32 else 0.}}
                configurations.append(record); total_programs += count; total_edges += count*edges
                previous_exact, previous_values, previous_l1, previous_quantization = chosen, values, l1_values, quant_values
                previous_components = selected_components
                previous_guarded, previous_float_lower = np.concatenate(guarded_duals), float_lower
                previous_risk, previous_guarded_retained = risk, retained
    need(len(configurations) == 18 and total_programs == 2688 and total_edges == 5431296, "Full frozen computation census")
    need(producer_summary["total_new_block_programs"] == total_programs and producer_summary["total_undirected_edges_with_repetition"] == total_edges, "Producer total cost count mismatch")
    best = {}
    for prefix in PREFIXES:
        candidates = [record for record in configurations if record["prefix"] == prefix]
        winner = max(candidates, key=lambda record: E.from_record(record["exact_lower"]))
        best[str(prefix)] = {key: winner[key] for key in ("prefix", "family", "block_size", "exact_lower", "strictly_excludes_unchanged_threshold_upper", "strictly_excludes_exact_reference_requirement", "same_census_KL_penalized_lower")}
        best[str(prefix)]["necessary_regularity_unchanged_threshold"] = best_regularity(candidates, "necessary_regularity_unchanged_threshold")
        best[str(prefix)]["necessary_regularity_exact_reference_threshold"] = best_regularity(candidates, "necessary_regularity_exact_reference_threshold")
    summary = {"schema": "step19-conditional-exact-results-v1", "population": M,
        "freeze_sha256": sha(RUN/"freeze.json"), "producer_seal_sha256": sha(RUN/"producer_seal.json"),
        "input_arrays_sha256": sha(RUN/"inputs.npz"), "statistical_confidence_events": 0,
        "exact_reference_mse": E.rational_record(reference_mse), "exact_11_over_10_reference_requirement": E.rational_record(threshold_exact),
        "unchanged_step18_threshold_upper": E.rational_record(threshold_upper),
        "step18_threshold_encloses_exact_requirement": True,
        "primary_rule": "Exact class lower strictly exceeds unchanged stored Step18 threshold_upper",
        "configurations": configurations, "best_by_prefix": best, "quantization": quantization,
        "new_block_programs": total_programs, "complete_new_edges_with_repetition": total_edges,
        "canonical_feature_scope": "Conditional on the inherited Step18 box inclusion and audited provenance; exact stored-binary64 targets and predictions; not a formal proof of the upstream engine or measurement process",
        "same_census_KL_comparison_scope": "Unnecessary range-only diagnostic on identical full-cohort lower; never used for decisions",
        "necessary_regularity_scope": "Analytical necessary condition from the same selected signed witnesses, combined only within each frozen geometry; no new optimization and no adequacy assertion above the bound",
        "total_verification_seconds": time.perf_counter()-started}
    if replay:
        # Runtime fields are observations, not mathematical evidence identities.
        def strip_timing(value):
            if isinstance(value, dict):
                return {key: strip_timing(item) for key, item in value.items()
                        if key not in ("timing_seconds", "seconds", "total_verification_seconds", "max_independent_float_replay_error")}
            if isinstance(value, list): return [strip_timing(item) for item in value]
            return value
        need(strip_timing(summary) == strip_timing(old_summary), "Recomputed exact summary differs")
        return {"passed": True, "read_only_replay": True, "configurations": 18, "new_block_programs_recomputed_without_solver": total_programs,
                "exact_new_duals_recomputed": total_programs, "exact_child_choices_reconstructed": True,
                "seconds": time.perf_counter()-started, "original_absolute_paths_required": False}
    json_new(exact_root/"summary.json", summary)
    exact_seal = {"freeze_sha256": sha(RUN/"freeze.json"), "producer_seal_sha256": sha(RUN/"producer_seal.json"),
                  "files": {path.relative_to(RUN).as_posix(): sha(path) for path in sorted(exact_root.rglob("*")) if path.is_file()}}
    json_new(RUN/"exact_seal.json", exact_seal)
    verification = {"passed": True, "schema": "step19-independent-verification-v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha(RUN/"freeze.json"), "producer_seal_sha256": sha(RUN/"producer_seal.json"), "exact_seal_sha256": sha(RUN/"exact_seal.json"),
        "exact_summary_sha256": sha(exact_root/"summary.json"), "source_sha256": {path.relative_to(BASE).as_posix(): sha(path) for path in (HERE/"audit19.py", HERE/"exact19.py")},
        "configurations": 18, "block_programs": total_programs, "edge_constraints_with_repetition": total_edges,
        "all_new_signed_duals_evaluated_exactly": True, "exact_recursive_witness_choice": True,
        "independent_indices_nesting_float_correction_counts_and_seals": True,
        "exact_reference_risk_crosscheck": True, "primary_threshold_unchanged": True,
        "no_new_predictions_or_model_fits": True, "no_raw_or_reserve_access": True,
        "statistical_confidence_events": 0, "conditional_exact_arithmetic_scope": summary["canonical_feature_scope"],
        "seconds": time.perf_counter()-started}
    json_new(RUN/"verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "verify", "run", "replay"))
    args = parser.parse_args()
    if args.stage == "preflight":
        import check_exact19
        result = check_exact19.main()
    else:
        result = run_audit(replay=args.stage == "replay")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
