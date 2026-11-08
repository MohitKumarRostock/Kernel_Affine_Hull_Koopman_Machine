"""Step 24 readouts and compact Nystrom deployments.

Importing this module performs no data access or fitting. Public fit functions
consume only their supplied development arrays. Deployment payloads contain
plain dictionaries, lists, numeric arrays/scalars and identifying strings; no
fitted sklearn estimator or hidden training matrix is retained.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
             "OMP_THREAD_LIMIT"):
    os.environ[_key] = "1"

import hashlib
import math
from numbers import Integral, Real
import time
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


_THREAD_LIMITER = threadpool_limits(limits=1)
RIDGE_ALPHAS = (1e-4, .01, 1.)
HGB_LEAVES = (7, 15, 31)
HGB_SEED = 2026110623
NYSTROM_COUNTS = (32, 128, 512)
GAMMA_MULTIPLIERS = (.1, 1., 10.)
EIGEN_FLOOR_FACTOR = 1e-12
EXPORT_ABS_FACTOR = 1e-9
EXPORT_REL_TOL = 1e-10

HGB_PARAMETERS = dict(
    loss="squared_error", learning_rate=.1, max_iter=200,
    min_samples_leaf=20, l2_regularization=1., max_bins=255,
    early_stopping=False, categorical_features=None, max_depth=None,
    max_features=1., warm_start=False, monotonic_cst=None,
    interaction_cst=None, random_state=HGB_SEED,
)


def _matrix(value, name="features"):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError(name + " must be a nonempty finite two-dimensional matrix")
    return np.ascontiguousarray(a)


def _vector(value, name="values", rows=None):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(name + " must be a nonempty finite vector")
    if rows is not None and a.shape != (int(rows),):
        raise ValueError(name + " row count mismatch")
    return np.ascontiguousarray(a)


def _kind(task_kind):
    if task_kind == "hmm" or task_kind in tuple(f"hmm_{i}" for i in range(5)):
        return "hmm"
    if task_kind == "appliances":
        return "appliances"
    raise ValueError("Unknown fixed task kind")


def output_bounds(task_kind):
    """Numeric output-map parameters retained by every exported fitted head."""
    return np.array([-1., 1.] if _kind(task_kind) == "hmm" else [0.], dtype=np.float64)


def output_map(prediction, task_kind):
    p = _vector(prediction, "prediction")
    if _kind(task_kind) == "hmm":
        return np.clip(p, -1., 1.)
    return np.maximum(p, 0.)


def _apply_stored_output(prediction, head, task_kind, clip=True):
    p = _vector(prediction, "exported physical prediction")
    bounds = np.asarray(head["output_bounds"], dtype=np.float64)
    if not np.array_equal(bounds, output_bounds(task_kind)):
        raise ValueError("Stored output map differs from the fixed task map")
    if not clip:
        return p
    return np.clip(p, bounds[0], bounds[1]) if len(bounds) == 2 else np.maximum(p, bounds[0])


def fit_target_map(y):
    """Fit the shared horizon map using only the supplied head-training labels."""
    y = _vector(y, "training labels")
    constant = bool(np.max(y) == np.min(y))
    offset = float(y[0]) if constant else float(np.mean(y, dtype=np.float64))
    scale = 1. if constant else float(np.std(y, dtype=np.float64, ddof=0))
    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("The training target map is not finite/representable")
    return {"offset": np.float64(offset), "scale": np.float64(scale)}


def _target_map(target_map):
    if not isinstance(target_map, dict) or not {"offset", "scale"} <= set(target_map):
        raise ValueError("Target map requires offset and scale")
    offset, scale = float(target_map["offset"]), float(target_map["scale"])
    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid target map")
    return offset, scale


def fit_feature_scaler(X):
    """Ridge-head standardization, retaining future variation on constant axes."""
    x = _matrix(X)
    constant = np.max(x, axis=0) == np.min(x, axis=0)
    offset = np.mean(x, axis=0, dtype=np.float64)
    offset[constant] = x[0, constant]
    scale = np.std(x, axis=0, dtype=np.float64, ddof=0)
    scale[constant] = 1.
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ArithmeticError("Invalid head feature standardizer")
    return {"offset": offset.copy(), "scale": scale.copy()}, constant


def standardize_features(X, scaler):
    x = _matrix(X)
    offset, scale = np.asarray(scaler["offset"]), np.asarray(scaler["scale"])
    if offset.shape != (x.shape[1],) or scale.shape != offset.shape or np.any(scale <= 0):
        raise ValueError("Head feature standardizer dimension/divisor mismatch")
    z = (x - offset) / scale
    if not np.isfinite(z).all():
        raise ArithmeticError("Nonfinite standardized head feature")
    return np.ascontiguousarray(z)


def audit_export(reference, exported, target_map, label="export"):
    """Fixed numerical fidelity gate; it is not an accuracy/noninferiority test."""
    a, b = _vector(reference, "reference export"), _vector(exported, "compact export")
    if a.shape != b.shape:
        raise ValueError("Export validation shapes differ")
    _, scale = _target_map(target_map)
    absolute_tolerance = EXPORT_ABS_FACTOR * max(1., scale)
    error = np.abs(a - b)
    permitted = absolute_tolerance + EXPORT_REL_TOL * np.abs(a)
    maximum = float(np.max(error))
    if np.any(error > permitted):
        index = int(np.argmax(error - permitted))
        raise ArithmeticError(
            f"{label} changed forecasts above the frozen tolerance: "
            f"max_abs={maximum:.17g}, row={index}, "
            f"error={error[index]:.17g}, allowed={permitted[index]:.17g}")
    return {"export_passed": True, "checked_rows": len(a),
            "max_abs_difference": maximum,
            "max_relative_difference_with_unit_floor": float(np.max(error / np.maximum(1., np.abs(a)))),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": EXPORT_REL_TOL,
            "scope": "Implementation-fidelity gate for frozen operational forecasts"}


def _tree_payload(tree):
    return {"nodes": np.array(tree.nodes, copy=True),
            "binned_left_cat_bitsets": np.array(tree.binned_left_cat_bitsets, copy=True),
            "raw_left_cat_bitsets": np.array(tree.raw_left_cat_bitsets, copy=True)}


def _validate_tree_payload(tree, dimension):
    nodes = tree["nodes"]
    required = {"value", "feature_idx", "num_threshold", "left", "right",
                "is_leaf", "is_categorical", "missing_go_to_left"}
    if not isinstance(nodes, np.ndarray) or nodes.ndim != 1 or not len(nodes):
        raise ValueError("Invalid retained tree nodes")
    if nodes.dtype.names is None or not required <= set(nodes.dtype.names):
        raise ValueError("Incomplete retained TreePredictor node fields")
    if not np.isfinite(nodes["value"]).all() or not np.isfinite(nodes["num_threshold"]).all():
        raise ArithmeticError("Nonfinite tree value or threshold")
    if np.any(nodes["is_categorical"]):
        raise ValueError("Categorical trees are outside the fixed protocol")
    seen, stack = set(), [0]
    while stack:
        i = int(stack.pop())
        if i in seen or not 0 <= i < len(nodes):
            raise ValueError("Invalid tree topology")
        seen.add(i)
        if not bool(nodes["is_leaf"][i]):
            if not 0 <= int(nodes["feature_idx"][i]) < dimension:
                raise ValueError("Tree feature index outside the input")
            stack.extend((int(nodes["left"][i]), int(nodes["right"][i])))
    if len(seen) != len(nodes):
        raise ValueError("Unreachable retained tree nodes")


def predict_head(head, F, task_kind, clip=True):
    """Query an exported head; no fitting, standardizer recovery or data I/O."""
    x = _matrix(F)
    if head["kind"] == "ridge":
        coef = np.asarray(head["coef"], dtype=np.float64)
        if coef.shape != (x.shape[1],):
            raise ValueError("Ridge query dimension mismatch")
        value = x @ coef + head["intercept"]
    elif head["kind"] == "hgb":
        if np.asarray(head["f_idx_map"]).shape != (x.shape[1],):
            raise ValueError("HGB query dimension mismatch")
        value = np.full(len(x), head["baseline"], dtype=np.float64)
        for tree in head["trees"]:
            predictor = TreePredictor(tree["nodes"], tree["binned_left_cat_bitsets"],
                                      tree["raw_left_cat_bitsets"])
            value += predictor.predict(x, head["known_cat_bitsets"], head["f_idx_map"], 1)
        value = head["target_offset"] + head["target_scale"] * value
    else:
        raise ValueError("Unknown exported head kind")
    return _apply_stored_output(value, head, task_kind, clip)


def predict_head_reference(head, F, task_kind, clip=True):
    """Independent unoptimized traversal for exported-head replay.

    Ridge uses an explicit ordered sum rather than BLAS, so last-bit differences
    are measured against the fixed export tolerance, not silently discarded.
    Tree comparisons and accumulation follow the stored binary64 node program.
    """
    x = _matrix(F)
    if head["kind"] == "ridge":
        coef = _vector(head["coef"], rows=x.shape[1])
        value = np.zeros(len(x), dtype=np.float64)
        for j in range(x.shape[1]):
            value += x[:, j] * coef[j]
        value += head["intercept"]
    elif head["kind"] == "hgb":
        if np.asarray(head["f_idx_map"]).shape != (x.shape[1],):
            raise ValueError("HGB reference query dimension mismatch")
        value = np.full(len(x), head["baseline"], dtype=np.float64)
        for tree in head["trees"]:
            nodes = tree["nodes"]
            current = np.zeros(len(x), dtype=np.intp)
            active = np.flatnonzero(~nodes["is_leaf"][current].astype(bool))
            while len(active):
                at = current[active]
                left = x[active, nodes["feature_idx"][at]] <= nodes["num_threshold"][at]
                current[active] = np.where(left, nodes["left"][at], nodes["right"][at])
                active = active[~nodes["is_leaf"][current[active]].astype(bool)]
            value += nodes["value"][current]
        value = head["target_offset"] + head["target_scale"] * value
    else:
        raise ValueError("Unknown exported reference head kind")
    return _apply_stored_output(value, head, task_kind, clip)


def fit_head(spec, X_train, y_train, X_val, target_map, task_kind):
    """Fit one frozen menu head, audit its export, and return validation output."""
    start = time.perf_counter()
    x, xv = _matrix(X_train, "training features"), _matrix(X_val, "validation features")
    if x.shape[1] != xv.shape[1]:
        raise ValueError("Training/validation head dimensions differ")
    y = _vector(y_train, "head-training labels", len(x))
    offset, target_scale = _target_map(target_map)
    ys = (y - offset) / target_scale
    if not np.isfinite(ys).all():
        raise ArithmeticError("Nonfinite standardized training targets")
    diagnostics = {"family": spec["family"], "training_rows": len(x),
                   "feature_dimension": x.shape[1],
                   "target_offset": offset, "target_scale": target_scale}
    if spec["family"] == "ridge":
        alpha = float(spec["alpha"])
        if alpha not in RIDGE_ALPHAS:
            raise ValueError("Ridge alpha outside the frozen menu")
        scaler, constant = fit_feature_scaler(x)
        z = standardize_features(x, scaler)
        zv = standardize_features(xv, scaler)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Ridge(alpha=len(x) * alpha, fit_intercept=True,
                           solver="cholesky", copy_X=True).fit(z, ys)
        scaled_coef = np.asarray(fitted.coef_, dtype=np.float64) / scaler["scale"]
        coef = np.ascontiguousarray(target_scale * scaled_coef)
        intercept = offset + target_scale * (float(fitted.intercept_) - float(scaler["offset"] @ scaled_coef))
        head = {"kind": "ridge", "coef": coef, "intercept": np.float64(intercept),
                "output_bounds": output_bounds(task_kind)}
        reference_raw = offset + target_scale * fitted.predict(zv)
        diagnostics.update({"alpha": alpha, "sklearn_alpha": len(x) * alpha,
            "training_objective": float(np.mean((fitted.predict(z) - ys) ** 2)
                                         + alpha * np.dot(fitted.coef_, fitted.coef_)),
            "head_scaler_evidence": {"offset": scaler["offset"].tolist(),
                                     "scale": scaler["scale"].tolist(),
                                     "training_constant_mask": constant.tolist()},
            "warnings": [str(w.message) for w in caught]})
    elif spec["family"] == "hgb":
        leaves = int(spec["max_leaf_nodes"])
        if leaves not in HGB_LEAVES:
            raise ValueError("HGB leaf setting outside the frozen menu")
        for key, value in HGB_PARAMETERS.items():
            if key in spec and spec[key] != value:
                raise ValueError("HGB parameter changed from the fixed protocol: " + key)
        parameters = dict(HGB_PARAMETERS, max_leaf_nodes=leaves)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = HistGradientBoostingRegressor(**parameters).fit(x, ys)
        bitsets, fmap = fitted._bin_mapper.make_known_categories_bitsets()
        trees = [_tree_payload(trees[0]) for trees in fitted._predictors]
        if int(fitted.n_iter_) != 200 or len(trees) != 200:
            raise ArithmeticError("HGB did not execute its fixed 200 iterations")
        for tree in trees:
            _validate_tree_payload(tree, x.shape[1])
        head = {"kind": "hgb", "trees": trees,
                "baseline": np.float64(fitted._baseline_prediction[0, 0]),
                "known_cat_bitsets": np.array(bitsets, copy=True),
                "f_idx_map": np.array(fmap, copy=True),
                "target_offset": np.float64(offset), "target_scale": np.float64(target_scale),
                "output_bounds": output_bounds(task_kind)}
        reference_raw = offset + target_scale * fitted.predict(xv)
        diagnostics.update({"parameters": parameters,
            "tree_count": len(trees), "node_count": sum(len(t["nodes"]) for t in trees),
            "retained_node_fields": list(trees[0]["nodes"].dtype.names),
            "node_dtype_itemsize": int(trees[0]["nodes"].dtype.itemsize),
            "warnings": [str(w.message) for w in caught]})
    else:
        raise ValueError("Unknown frozen head family")
    exported_raw = predict_head(head, xv, task_kind, clip=False)
    diagnostics["unclipped_export"] = audit_export(reference_raw, exported_raw, target_map,
                                                    spec["family"] + " raw export")
    reference = output_map(reference_raw, task_kind)
    exported = predict_head(head, xv, task_kind)
    diagnostics["clipped_export"] = audit_export(reference, exported, target_map,
                                                  spec["family"] + " output export")
    diagnostics["numeric_bytes"] = numeric_inventory(head)["numeric_bytes"]
    diagnostics["fit_and_export_seconds"] = time.perf_counter() - start
    return head, exported, diagnostics


def fit_input_scaler(U, scaling):
    """Support-only input scaler matching the inherited geometry convention."""
    u = _matrix(U, "reference features")
    if scaling not in ("minmax01", "standard"):
        raise ValueError("Unknown support input scaler")
    lower, upper = np.min(u, axis=0), np.max(u, axis=0)
    constant = upper == lower
    if scaling == "minmax01":
        offset, divisor = lower, upper - lower
    else:
        offset, divisor = np.mean(u, axis=0), np.std(u, axis=0, ddof=0)
    divisor = np.where(constant, 1., divisor)
    if not np.isfinite(offset).all() or not np.isfinite(divisor).all() or np.any(divisor <= 0):
        raise ArithmeticError("Invalid support input scaler")
    return {"offset": np.array(offset, copy=True), "divisor": np.array(divisor, copy=True),
            "constant_mask": np.array(constant, dtype=bool, copy=True)}


def apply_input_scaler(X, scaler):
    x = _matrix(X)
    offset, divisor = np.asarray(scaler["offset"]), np.asarray(scaler["divisor"])
    constant = np.asarray(scaler["constant_mask"], dtype=bool)
    if offset.shape != (x.shape[1],) or divisor.shape != offset.shape or constant.shape != offset.shape:
        raise ValueError("Input scaler query dimension mismatch")
    z = (x - offset) / divisor
    z[:, constant] = 0.
    if not np.isfinite(z).all():
        raise ArithmeticError("Nonfinite support-scaled query")
    return np.ascontiguousarray(z)


def rbf_kernel(scaled_X, scaled_landmarks, gamma):
    x, z = _matrix(scaled_X), _matrix(scaled_landmarks)
    if x.shape[1] != z.shape[1] or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Invalid RBF kernel inputs")
    distances = cdist(x, z, metric="sqeuclidean")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ArithmeticError("Nonfinite/negative RBF squared distance")
    kernel = np.exp(-float(gamma) * distances)
    if not np.isfinite(kernel).all():
        raise ArithmeticError("Nonfinite RBF kernel")
    return np.ascontiguousarray(kernel)


def fit_nystrom_base(U, scaling, m, gamma_multiplier):
    """Fit one support-only basis; retain W separately as training evidence."""
    start = time.perf_counter()
    u = _matrix(U, "Nystrom reference bank")
    if len(u) != 512:
        raise ValueError("Nystrom requires the fixed 512-row reference bank")
    if isinstance(m, bool) or not isinstance(m, Integral) or int(m) not in NYSTROM_COUNTS:
        raise ValueError("Nystrom landmark count outside the frozen menu")
    m = int(m)
    gamma_multiplier = float(gamma_multiplier)
    if gamma_multiplier not in GAMMA_MULTIPLIERS:
        raise ValueError("Nystrom gamma multiplier outside the frozen menu")
    scaler = fit_input_scaler(u, scaling)
    us = apply_input_scaler(u, scaler)
    distances = pdist(us, metric="sqeuclidean")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ArithmeticError("Nonfinite/negative reference pairwise distance")
    positive = distances[distances > 0]
    if not len(positive):
        raise ArithmeticError("No positive reference distance; no bandwidth fallback")
    median = float(np.median(positive))
    gamma = gamma_multiplier / median
    landmarks = np.array(us[:m], dtype=np.float64, order="C", copy=True)
    W = rbf_kernel(landmarks, landmarks, gamma)
    W = np.ascontiguousarray((W + W.T) * .5)
    eigenvalues, eigenvectors = eigh(W, driver="evd", overwrite_a=False, check_finite=True)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0:
        raise ArithmeticError("Invalid Nystrom largest eigenvalue")
    floor = EIGEN_FLOOR_FACTOR * largest
    clipped_eigenvalues = np.maximum(eigenvalues, floor)
    normalization = (eigenvectors * (1. / np.sqrt(clipped_eigenvalues))) @ eigenvectors.T
    normalization = np.ascontiguousarray((normalization + normalization.T) * .5)
    if not np.isfinite(normalization).all():
        raise ArithmeticError("Nonfinite Nystrom inverse square root")
    base = {"kind": "nystrom_training_base", "scaler": scaler,
            "landmarks": landmarks, "gamma": np.float64(gamma),
            "normalization": normalization}
    evidence = {"summary": {
        "scaling": scaling, "landmark_count": m, "gamma_multiplier": gamma_multiplier,
        "gamma": gamma, "positive_pair_count": len(positive),
        "reference_pairwise_squared_distance_median": median,
        "eigenvalue_floor": floor, "largest_eigenvalue": largest,
        "smallest_eigenvalue_before_floor": float(eigenvalues[0]),
        "floored_eigenvalue_count": int(np.count_nonzero(eigenvalues < floor)),
        "normalization_max_asymmetry": float(np.max(np.abs(normalization - normalization.T))),
        "eigensolver": "scipy.linalg.eigh(driver='evd')",
        "fit_seconds": time.perf_counter() - start,
        "W_is_training_evidence_not_deployment": True,
    }, "arrays": {"W": W, "eigenvalues": eigenvalues,
                  "floored_eigenvalues": clipped_eigenvalues}}
    return base, evidence


def nystrom_features(base, X):
    if base["kind"] != "nystrom_training_base":
        raise ValueError("Expected a training Nystrom basis")
    scaled = apply_input_scaler(X, base["scaler"])
    kernel = rbf_kernel(scaled, base["landmarks"], base["gamma"])
    features = kernel @ base["normalization"]
    if not np.isfinite(features).all():
        raise ArithmeticError("Nonfinite Nystrom feature")
    return np.ascontiguousarray(features)


def collapse_nystrom(base, ridge_head):
    """Collapse the symmetric feature map into physical-unit kernel weights.

    The caller must audit both unclipped/clipped validation predictions against
    the fitted feature head, and select using the actual compact predictions.
    """
    if base["kind"] != "nystrom_training_base" or ridge_head["kind"] != "ridge":
        raise ValueError("Only a Nystrom ridge head can be collapsed")
    normalization = np.asarray(base["normalization"], dtype=np.float64)
    coef = _vector(ridge_head["coef"])
    if normalization.shape != (len(coef), len(coef)):
        raise ValueError("Nystrom collapse dimension mismatch")
    weights = np.ascontiguousarray(normalization @ coef)
    if not np.isfinite(weights).all():
        raise ArithmeticError("Nonfinite collapsed Nystrom weights")
    return {"kind": "nystrom", "scaler": {k:np.array(v,copy=True) for k,v in base["scaler"].items()},
            "landmarks": np.array(base["landmarks"], dtype=np.float64, order="C", copy=True),
            "gamma": np.float64(base["gamma"]), "coef": weights,
            "intercept": np.float64(ridge_head["intercept"]),
            "output_bounds": np.array(ridge_head["output_bounds"], copy=True)}


def predict_nystrom(deployed, X, task_kind, clip=True):
    if deployed["kind"] != "nystrom":
        raise ValueError("Expected a compact Nystrom deployment")
    scaled = apply_input_scaler(X, deployed["scaler"])
    kernel = rbf_kernel(scaled, deployed["landmarks"], deployed["gamma"])
    value = kernel @ deployed["coef"] + deployed["intercept"]
    return _apply_stored_output(value, deployed, task_kind, clip)


def numeric_inventory(payload):
    """Count retained numeric payload, without Python/object/runtime overhead.

    Full ndarray storage owners are counted once, including structured padding
    and storage retained by views. Explicit numeric scalar parameter slots use
    their NumPy dtype width (Python int/float: int64/float64); scalar interning is
    not credited as a deployment optimization. Non-numeric strings/None carry
    no numeric bytes. Unknown objects and object arrays fail rather than hide
    uncounted estimator state. Serialized bytes/RSS must be reported separately.
    """
    components, owners, containers = [], {}, set()
    def walk(value, path):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or (value.dtype.names is None and value.dtype.kind not in "biufc"):
                raise TypeError("Uncountable nonnumeric ndarray at " + path)
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if owner.base is None:
                key = ("ndarray", id(owner))
                retained = int(owner.nbytes)
            else:
                buffer = owner.base
                try:
                    retained = int(memoryview(buffer).nbytes)
                except TypeError as exc:
                    raise TypeError("Unknown retained array owner at " + path) from exc
                key = ("buffer", id(buffer))
            record = {"path":path,"dtype":str(value.dtype),"shape":list(value.shape),
                      "logical_bytes":int(value.nbytes)}
            if key in owners:
                record.update(retained_bytes=0,alias_of=owners[key])
            else:
                owners[key]=path
                record["retained_bytes"]=retained
            components.append(record)
        elif isinstance(value, dict):
            if id(value) in containers:return
            containers.add(id(value))
            for key in sorted(value):
                walk(value[key],path+"."+str(key))
        elif isinstance(value,(list,tuple)):
            if id(value) in containers:return
            containers.add(id(value))
            for index,part in enumerate(value):walk(part,f"{path}[{index}]")
        elif isinstance(value,(np.generic,Real,Integral)):
            a=np.asarray(value)
            if a.dtype.kind not in "biufc" or a.ndim != 0:
                raise TypeError("Uncountable numeric scalar at " + path)
            if a.dtype.kind in "fc" and not np.isfinite(a):
                raise ValueError("Nonfinite retained scalar at " + path)
            components.append({"path":path,"dtype":str(a.dtype),"shape":[],
                               "logical_bytes":int(a.nbytes),"retained_bytes":int(a.nbytes)})
        elif value is None or isinstance(value,str):
            return
        else:
            raise TypeError("Uncounted deployment object " + type(value).__name__ + " at " + path)
    walk(payload,"model")
    total=sum(record["retained_bytes"] for record in components)
    return {"numeric_bytes":int(total),"components":components,
            "scope":"Complete retained numeric arrays/scalars for one supplied deployment; array-owner aliases counted once; Python/runtime overhead separate"}


def source_hash():
    """Local implementation identity; imports alone never call this helper."""
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
