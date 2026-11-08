"""Frozen Step 18 baseline menu and compact, training-data-free predictors.

Importing this module does not draw data or fit a model. All input transforms,
centers, and kernel bandwidths are fitted on the unlabeled support pool.
Ridge alpha means mean squared error plus alpha times coefficient norm squared.
The MLP alpha follows sklearn's documented sum-of-squares/sample-size convention.
"""
from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

SEED = 2026110618
SCALINGS = ('minmax01', 'standard')
RIDGE_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, .1)
RBF_COUNTS = (2, 6, 14)
GAMMA_MULTIPLIERS = (.25, 1., 4.)
RBF_ALPHAS = (1e-6, 1e-4, 1e-2, .1)
NYSTROM_COUNTS = (64, 128)
NYSTROM_ALPHAS = (1e-4, 1e-2, .1)
MLP_HIDDEN = (32, 64)
MLP_ALPHAS = (1e-4, 1e-2)
HGB_LEAVES = (15, 31)
BOUNDS = (-1., 1.)
TARGET_SCALE_MV = 2.
NYSTROM_EXPORT_TOLERANCE = 1e-8


def source_files():
    return [Path(__file__).resolve()]


def grid():
    result = []
    def add(family, slot, **parameters):
        result.append(dict(id=f'b{len(result):03d}', index=len(result),
                           family=family, selection_slot=slot, **parameters))
    add('mean', 'mean')
    add('persistence', 'persistence', target_scale_mV=TARGET_SCALE_MV)
    for family in ('ridge', 'quadratic'):
        for scaling in SCALINGS:
            for alpha in RIDGE_ALPHAS:
                add(family, family, scaling=scaling, alpha=alpha,
                    objective='mean_squared_error + alpha * squared_coefficient_norm')
    for scaling in SCALINGS:
        for count in RBF_COUNTS:
            for gamma in GAMMA_MULTIPLIERS:
                for alpha in RBF_ALPHAS:
                    add('compact_rbf', f'compact_rbf_{count}', scaling=scaling,
                        centers=count, gamma_multiplier=gamma, alpha=alpha,
                        objective='mean_squared_error + alpha * squared_coefficient_norm')
    for scaling in SCALINGS:
        for count in NYSTROM_COUNTS:
            for gamma in GAMMA_MULTIPLIERS:
                for alpha in NYSTROM_ALPHAS:
                    add('nystrom', f'nystrom_{count}', scaling=scaling,
                        centers=count, gamma_multiplier=gamma, alpha=alpha,
                        normalization_fold_tolerance=NYSTROM_EXPORT_TOLERANCE,
                        objective='mean_squared_error + alpha * squared_coefficient_norm')
    for scaling in SCALINGS:
        for hidden in MLP_HIDDEN:
            for alpha in MLP_ALPHAS:
                add('mlp', f'mlp_{hidden}', scaling=scaling, hidden=hidden,
                    alpha=alpha, activation='tanh', solver='lbfgs', max_iter=400,
                    tolerance=1e-4, max_fun=15000, seed=SEED)
    for leaves in HGB_LEAVES:
        add('hgb', f'hgb_{leaves}', max_leaf_nodes=leaves, learning_rate=.1,
            max_iter=200, l2_regularization=1e-2, min_samples_leaf=20,
            max_bins=255, early_stopping=False, seed=SEED)
    if len(result) != 140 or len({s['selection_slot'] for s in result}) != 13:
        raise ArithmeticError('Baseline roster mismatch')
    return result


def scaler_fit(x, mode):
    x = matrix(x)
    constant = np.max(x,axis=0) == np.min(x,axis=0)
    if mode == 'minmax01':
        offset = np.min(x, axis=0)
        divisor = np.max(x, axis=0) - offset
    elif mode == 'standard':
        offset, divisor = np.mean(x, axis=0), np.std(x, axis=0, ddof=0)
    else:
        raise ValueError('Unknown frozen scaling')
    return dict(mode=mode, offset=offset.copy(), divisor=np.where(constant, 1., divisor),
                constant_mask=constant)


def matrix(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty matrix')
    return x


def scale(x, scaler):
    x = matrix(x)
    z = (x - scaler['offset']) / scaler['divisor']
    z[:, scaler['constant_mask']] = 0.
    return z


def quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.concatenate((x, x[:, i] * x[:, j]), axis=1)


def squared_distance(x, centers):
    # Bounded memory for 8192 by128 by32; each center is processed once.
    return np.column_stack([np.sum((x - c) ** 2, axis=1) for c in centers])


def kernel(x, centers, gamma, normalized=False):
    logits = -float(gamma) * squared_distance(x, centers)
    if normalized:
        logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def fit_ridge(f, y, alpha):
    model = Ridge(alpha=float(alpha) * len(f), fit_intercept=True,
                  solver='cholesky').fit(f, y)
    return np.ascontiguousarray(model.coef_, dtype=np.float64), float(model.intercept_)


def predict(model, x):
    x = matrix(x)
    kind = model['kind']
    if kind == 'mean':
        p = np.full(len(x), model['constant'])
    elif kind == 'persistence':
        p = x[:, -1] / model['target_scale_mV']
    elif kind == 'hgb':
        p = np.full(len(x), model['baseline'], dtype=np.float64)
        for tree in model['trees']:
            p += tree.predict(x, model['known_cat_bitsets'], model['f_idx_map'], 1)
    else:
        z = scale(x, model['scaler'])
        if kind == 'quadratic':
            z = quadratic(z)
        elif kind in ('compact_rbf', 'nystrom'):
            z = kernel(z, model['centers'], model['gamma'], kind == 'compact_rbf')
        if kind == 'mlp':
            z = np.tanh(z @ model['hidden_weights'] + model['hidden_bias'])
            p = (z @ model['output_weights'] + model['output_bias']).reshape(-1)
        else:
            p = z @ model['coef'] + model['intercept']
    p = np.clip(p, *model['bounds'])
    if p.shape != (len(x),) or not np.isfinite(p).all():
        raise ArithmeticError('Nonfinite exported baseline prediction')
    return p


def numeric_inventory(value, name='model'):
    """Count actually retained query parameters, including encoders/scalers."""
    rows = []
    if isinstance(value, np.ndarray):
        rows.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                         bytes=int(value.nbytes)))
    elif isinstance(value, (float, int, np.number)) and not isinstance(value, bool):
        rows.append(dict(name=name, shape=[], dtype='scalar64', bytes=8))
    elif isinstance(value, bool):
        rows.append(dict(name=name, shape=[], dtype='bool', bytes=1))
    elif isinstance(value, dict):
        for k, v in value.items():
            rows.extend(numeric_inventory(v, name + '.' + str(k)))
    elif isinstance(value, (tuple, list)):
        for k, v in enumerate(value):
            rows.extend(numeric_inventory(v, name + '.' + str(k)))
    elif hasattr(value, 'nodes'):
        # TreePredictor query storage; retain its full structured node arrays.
        for k in ('nodes', 'raw_left_cat_bitsets', 'binned_left_cat_bitsets'):
            rows.extend(numeric_inventory(getattr(value, k), name + '.' + k))
    elif value is not None and not isinstance(value, str):
        raise TypeError(f'Uninventoried deployment type {type(value).__name__}')
    return rows


def payload_cost(model):
    components = numeric_inventory(model)
    return dict(numeric_bytes=sum(c['bytes'] for c in components),
                pickle_bytes=len(pickle.dumps(model, protocol=5)), components=components,
                scope='retained prediction parameters including scaler and encoder; Python objects/runtime/working memory excluded')


def fit_candidates(support_x, train_x, train_y, validation_x, validation_y,
                   *, specs=None):
    """Fit the frozen menu; all predictions use the compact exported predictor."""
    specs = grid() if specs is None else list(specs)
    u, x, xv = map(matrix, (support_x, train_x, validation_x))
    y, yv = np.asarray(train_y, float), np.asarray(validation_y, float)
    if y.shape != (len(x),) or yv.shape != (len(xv),):
        raise ValueError('Target shape mismatch')
    if not np.isfinite(y).all() or not np.isfinite(yv).all():
        raise ValueError('Nonfinite target')
    scalers, transformed, medians, centers, nystrom = {}, {}, {}, {}, {}
    cache_events = []
    for mode in SCALINGS:
        start = time.perf_counter()
        scalers[mode] = scaler_fit(u, mode)
        zu, z, zv = (scale(a, scalers[mode]) for a in (u, x, xv))
        transformed[mode] = (zu, z, zv)
        ds = pdist(zu, metric='sqeuclidean')
        positive = ds[ds > 0]
        medians[mode] = float(np.median(positive)) if len(positive) else 1.
        cache_events.append(dict(kind='support_scaler_and_median', scaling=mode,
                                 seconds=time.perf_counter() - start,
                                 median_positive_squared_distance=medians[mode]))
    models, records, predictions = {}, [], []
    for spec in specs:
        start = time.perf_counter()
        kind = spec['family']
        model = dict(kind=kind, bounds=np.asarray(BOUNDS))
        fit_diagnostics = {}
        if kind == 'mean':
            model['constant'] = float(np.mean(y))
        elif kind == 'persistence':
            model['target_scale_mV'] = TARGET_SCALE_MV
        elif kind == 'hgb':
            fitted = HistGradientBoostingRegressor(loss='squared_error',
                learning_rate=spec['learning_rate'], max_iter=spec['max_iter'],
                max_leaf_nodes=spec['max_leaf_nodes'],
                l2_regularization=spec['l2_regularization'],
                min_samples_leaf=spec['min_samples_leaf'], max_bins=spec['max_bins'],
                early_stopping=False, random_state=SEED).fit(x, y)
            bits, mapping = fitted._bin_mapper.make_known_categories_bitsets()
            model.update(baseline=float(fitted._baseline_prediction[0, 0]),
                         trees=[p[0] for p in fitted._predictors],
                         known_cat_bitsets=bits.copy(), f_idx_map=mapping.copy())
            replay = np.clip(fitted.predict(xv), *BOUNDS)
            fit_diagnostics['iterations'] = int(fitted.n_iter_)
        else:
            mode = spec['scaling']
            zu, z, zv = transformed[mode]
            model['scaler'] = scalers[mode]
            if kind in ('ridge', 'quadratic'):
                f = z if kind == 'ridge' else quadratic(z)
                model['coef'], model['intercept'] = fit_ridge(f, y, spec['alpha'])
            elif kind == 'compact_rbf':
                key = (mode, spec['centers'])
                if key not in centers:
                    before = time.perf_counter()
                    centers[key] = KMeans(n_clusters=spec['centers'], random_state=SEED,
                        n_init=5, max_iter=150).fit(zu).cluster_centers_.copy()
                    cache_events.append(dict(kind='compact_rbf_support_centers',
                        scaling=mode, centers=spec['centers'], seconds=time.perf_counter()-before))
                gamma = spec['gamma_multiplier'] / medians[mode]
                f = kernel(z, centers[key], gamma, True)
                coef, intercept = fit_ridge(f, y, spec['alpha'])
                model.update(centers=centers[key], gamma=gamma, coef=coef, intercept=intercept)
            elif kind == 'nystrom':
                key = (mode, spec['centers'], spec['gamma_multiplier'])
                if key not in nystrom:
                    before = time.perf_counter()
                    gamma = spec['gamma_multiplier'] / medians[mode]
                    feature = Nystroem(kernel='rbf', gamma=gamma,
                        n_components=spec['centers'], random_state=SEED).fit(zu)
                    nystrom[key] = (feature, feature.transform(z))
                    cache_events.append(dict(kind='nystrom_support_landmarks',
                        scaling=mode, centers=spec['centers'], gamma=gamma,
                        seconds=time.perf_counter()-before))
                feature, f = nystrom[key]
                coef, intercept = fit_ridge(f, y, spec['alpha'])
                # The normalization can be composed into the linear head, so
                # the exported query model need not retain an m-by-m matrix.
                folded = np.ascontiguousarray(feature.normalization_.T @ coef)
                model.update(centers=feature.components_.copy(), gamma=float(feature.gamma),
                             coef=folded, intercept=intercept)
                original = np.clip(feature.transform(zv) @ coef + intercept, *BOUNDS)
                fit_diagnostics['normalization_fold_max_prediction_error'] = float(
                    np.max(np.abs(original - predict(model, xv))))
                if fit_diagnostics['normalization_fold_max_prediction_error'] > NYSTROM_EXPORT_TOLERANCE:
                    raise ArithmeticError('Nystrom compact export exceeded its frozen replay tolerance')
            elif kind == 'mlp':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    fitted = MLPRegressor(hidden_layer_sizes=(spec['hidden'],),
                        activation='tanh', solver='lbfgs', alpha=spec['alpha'],
                        max_iter=400, max_fun=15000, tol=1e-4,
                        random_state=SEED, early_stopping=False).fit(z, y)
                model.update(hidden_weights=fitted.coefs_[0].copy(),
                    hidden_bias=fitted.intercepts_[0].copy(),
                    output_weights=fitted.coefs_[1].copy(),
                    output_bias=fitted.intercepts_[1].copy())
                fit_diagnostics.update(iterations=int(fitted.n_iter_),
                    objective=float(fitted.loss_),
                    convergence_warnings=[str(w.message) for w in caught
                                          if issubclass(w.category, ConvergenceWarning)])
                replay = np.clip(fitted.predict(zv), *BOUNDS)
            else:
                raise ValueError('Unknown baseline family')
        pred = predict(model, xv)
        if kind in ('hgb', 'mlp'):
            error = float(np.max(abs(replay - pred)))
            fit_diagnostics['export_max_prediction_error'] = error
            if error > 1e-10:
                raise ArithmeticError('Compact exported model changed validation predictions')
        cost = payload_cost(model)
        record = dict(**spec, validation_mse=float(np.mean((pred - yv) ** 2)),
            fit_and_validation_seconds=time.perf_counter()-start,
            numeric_bytes=cost['numeric_bytes'], cost=cost,
            diagnostics=fit_diagnostics,
            nonconvergence_retained=True, finite_prediction_verified=True)
        models[spec['id']] = model
        records.append(record)
        predictions.append(pred)
    selected = {}
    for record in records:
        slot = record['selection_slot']
        if slot not in selected or (record['validation_mse'], record['index']) < (
                selected[slot]['validation_mse'], selected[slot]['index']):
            selected[slot] = record
    reference = min(selected.values(), key=lambda r: (r['validation_mse'], r['index']))
    return dict(models=models, records=records,
                validation_predictions=np.column_stack(predictions),
                selected=selected, reference=reference, cache_events=cache_events,
                support_only_input_transforms=True,
                training_targets_only_for_fitting=True,
                candidate_count=len(specs), seed=SEED)


def frontier(rows, risk_key='validation_mse', cost_key='numeric_bytes'):
    """Nondominated records, preserving all ties instead of hiding equal-cost arms."""
    return [r for r in rows if not any(
        q[cost_key] <= r[cost_key] and q[risk_key] <= r[risk_key]
        and (q[cost_key] < r[cost_key] or q[risk_key] < r[risk_key]) for q in rows)]

