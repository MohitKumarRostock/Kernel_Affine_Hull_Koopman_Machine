"""Prospective Step 17: selected nested KAHM classes and matched dual reuse.

All scientific data and stages are write-once.  Geometry and ridge choices
are frozen before independent calibration; calibration cannot fit, choose,
replace, or discard a predictor.  Importing this module performs no fitting
or scientific random draw.  Every pool has one independent episode per row.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_THREAD_LIMIT',
              'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BASE = PROJ.parent
RUN = HERE / 'run'
for _folder in ('step6_geometry_regimes/src', 'step11_output_accuracy',
                'step11_output_accuracy/engine',
                'risk_source/extensions/certified_forecasting/src'):
    sys.path.insert(0, str(BASE / _folder))
import numpy as np
import scipy
import sklearn
import design as D
import geometry as G
import certified as C
import soft_state as PS
import certified_forecasting.models as HEAD
import data17 as J
import certificate17 as Q

NREP = 12
WORKERS = 4
REGIMES = ('repairable', 'irreducible')
MASTER = 2026110617
DOMAIN = 1195724617
SIZE = dict(support=256, train=512, validation=256, cal=4096, test=1024)
POOL_ORDER = ('support', 'train', 'validation', 'cal', 'test', 'geometry')
PREFIXES = (2, 6)
BLOCKS = (2, 4)
SCALINGS = ('minmax01', 'standard')
CAPS = (1, 2)
CONCENTRATIONS = (.5, 4., 16., 64.)
RIDGES = (0., 1e-4, 1e-2)
KAPPA = 2.
BOUNDS = (-.85, .85)
EPSILON = .1
SAMPLE_SIZES = (256, 1024, 4096)
PRIMARY_N = 1024
CERTIFICATE_BLOCK_SIZES = (2, 4, 8, 16, 32)
DELTA = .05
EVENTS = 1008
ALPHA = DELTA / EVENTS
HEAD_TOLERANCE = 1e-7
HEAD_MAX_ITER = 20000
HEAD_INWARD_MARGIN = 1e-12
SOLVER_CONFIG = dict(max_iter=4000, tolerance=1e-7, check_every=20)
GEOMETRY_CONFIG = dict(scalings=SCALINGS, caps=CAPS, cluster_counts=BLOCKS,
                       max_fit=256, max_references=24, tau=1e-6)
SLICE = dict(support=(0, 256), train=(256, 768), validation=(768, 1024))
MANUFACTURED_SEEDS = tuple(range(2026110617000, 2026110617100))
PINNED_16 = {
    'freeze.json': '8141dfadf7bfee3a27f4e691a2f6dedaeec42bbba46719db79943b891e4a48bc',
    'fit_seal.json': 'c74c390ef20b29c4e7451a69b913d8e650b895789f01b56916a428154e28028c',
    'fit_validation.json': 'b30d790e947d8b8c0bc46981c5d5fc49be9e9cdf8fd7c449d32209bfd189073e',
    'cal_seal.json': '51408d3b6361154f98c44cf7c3efbf0c26593fc0aacaea057b2be1e156021e67',
    'cal_validation.json': '43750b6fc1740df7d4063303cbea279efc7d088fac17f7c022feff79d7405d52',
    'test_seal.json': '6b26f2899055dd878e19c5052b226614d5bfbe89a7c5898eed867a72d2346168',
    'verification.json': '74e93dae599fd2ef901f05afa8a715ef50ff5ad0f0ba23f1a8d1ed721aa93622',
}
PRIOR_FREEZES = (
    'step11_output_accuracy/prediction/run/freeze.json',
    'step12_predictive_geometry/study/run/freeze.json',
    'step13_temporal_geometry/study/run/freeze.json',
    'step14_linear_rollout/study/run/freeze.json',
    'step15_external_dynamics/study/run/freeze.json',
    'step16_observable_adequacy/study/run/freeze.json',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dumpj(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=json_default)
        stream.write('\n')


def dump(path, value):
    with Path(path).open('xb') as stream:
        pickle.dump(value, stream, protocol=5)


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def cases():
    return [dict(case=ri * NREP + rep, regime=regime, regime_index=ri, repetition=rep)
            for ri, regime in enumerate(REGIMES) for rep in range(NREP)]


def case_record(case):
    if isinstance(case, bool) or not isinstance(case, (int, np.integer)):
        raise ValueError('Case must be an integer')
    if not 0 <= case < len(cases()):
        raise ValueError('Case outside the frozen roster')
    return cases()[int(case)]


def geometry_specs():
    result = []
    for scaling in SCALINGS:
        for cap in CAPS:
            for concentration in CONCENTRATIONS:
                i = len(result)
                result.append(dict(id='g' + str(i), index=i, scaling=scaling,
                    cap=cap, concentration=concentration, blocks=BLOCKS,
                    maximum_prefix=6))
    return result


def feature_blocks(prefix):
    if prefix == 2:
        return ((0, 2),)
    if prefix == 6:
        return ((0, 2), (2, 6))
    raise ValueError('Prefix must be two or six')


def section(array, name):
    a, b = SLICE[name]
    return array[a:b]


def prior_seeds():
    old = read(BASE / 'step10_stable_normalization/results/run_20261105/freeze.json')
    result = (set(old['excluded_prior_scalar_streams'])
              | set(old['predictive_streams'].values())
              | set(old['numerical_streams'].values()))
    historical = old.get('historical_streams_retained_only_for_paired_diagnostics', {})
    result.update(historical.values() if isinstance(historical, dict) else historical)
    for filename in PRIOR_FREEZES:
        record = read(BASE / filename)
        result.update(record['streams'].values())
        result.update(record.get('manufactured_seeds', ()))
    return result


def streams():
    result = {}
    for item in cases():
        for pi, pool in enumerate(POOL_ORDER):
            key = f"r{item['case']}_{pool}"
            result[key] = D.get_seed(MASTER, DOMAIN, item['regime_index'],
                                     item['repetition'], pi)
    if len(result) != 144 or len(set(result.values())) != 144:
        raise ArithmeticError('Repeated Step 17 stream')
    if prior_seeds().intersection(result.values()):
        raise ArithmeticError('Step 17 stream reuses a reserved predecessor stream')
    if set(MANUFACTURED_SEEDS).intersection(result.values()):
        raise ArithmeticError('Scientific and manufactured streams overlap')
    if len({result[f'r{i}_geometry'] % 2**31 for i in range(24)}) != 24:
        raise ArithmeticError('Native geometry seeds collide after reduction')
    return result


def pool_seed(frozen, case, pool):
    return frozen['streams'][f'r{case}_{pool}']


def sources():
    paths = set(HERE.glob('*.py'))
    paths.update((HERE / 'PROTOCOL.md', HERE / 'protocol17.json'))
    for pattern in ('*.json', '*.npz', '*.pkl'):
        paths.update((PROJ / 'tests').glob(pattern))
    paths.update((Path(D.__file__), Path(G.__file__), Path(C.__file__), Path(PS.__file__),
                  Path(HEAD.__file__), Path(HEAD.__file__).with_name('__init__.py'),
                  BASE / 'step11_output_accuracy/engine/interval.cpp',
                  BASE / 'step11_output_accuracy/softsrc/power_box.cpp',
                  BASE / 'step10_stable_normalization/results/run_20261105/freeze.json',
                  BASE / 'risk_source/extensions/certified_forecasting/experiments/learned_kahm_decision/protocol.json',
                  BASE / 'risk_source/extensions/certified_forecasting/experiments/learned_kahm_decision/run_study.py',
                  BASE / 'risk_source/extensions/certified_forecasting/src/certified_forecasting/sparse_block_witness.py'))
    paths.update(BASE / filename for filename in PRIOR_FREEZES)
    paths.update(BASE / 'step16_observable_adequacy/study/run' / name for name in PINNED_16)
    return {**G.source_hashes(), **{p.relative_to(BASE).as_posix(): sha(p) for p in sorted(paths)}}


def numerical_configuration():
    return dict(score_atol=1e-10, coordinate_atol=1e-9, power_atol=1e-9,
        simplex_residual_atol=2e-13, precision_bits=(80, 128, 192),
        head_tolerance=HEAD_TOLERANCE, head_max_iter=HEAD_MAX_ITER,
        coefficient_inward_margin=HEAD_INWARD_MARGIN,
        dual_solver=SOLVER_CONFIG,
        endpoint_scope='guarded float64 coefficients, duals, and KL roots; no end-to-end interval certificate')


def freeze():
    if RUN.exists():
        raise FileExistsError('The scientific run is write-once')
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE', '0') not in ('', '0'):
        raise RuntimeError('Assertions must remain enabled')
    for name, expected in PINNED_16.items():
        if sha(BASE / 'step16_observable_adequacy/study/run' / name) != expected:
            raise ArithmeticError('Predecessor evidence changed: ' + name)
    assert read(BASE / 'step16_observable_adequacy/study/run/verification.json')['passed'] is True
    for stem in ('runner', 'certificate', 'auditor'):
        gate = read(PROJ / 'tests' / (stem + '_preflight.json'))
        assert gate['passed'] is True and gate['no_new_scientific_data'] is True
    own = read(PROJ / 'tests/runner_preflight.json')
    for field, filename in (('runner_source_sha256', 'runner17.py'),
                             ('controller_source_sha256', 'controller17.py'),
                             ('data_source_sha256', 'data17.py'),
                             ('test_source_sha256', 'check_runner17.py')):
        assert own[field] == sha(HERE / filename)
    assert own['fixture_sha256'] == sha(PROJ / 'tests/runner_fixture.pkl')
    cert = read(PROJ / 'tests/certificate_preflight.json')
    assert cert['certificate_source_sha256'] == sha(HERE / 'certificate17.py')
    assert cert['solver_source_sha256'] == sha(HERE / 'block_solver17.py')
    assert cert['test_source_sha256'] == sha(HERE / 'check_certificate17.py')
    auditor = read(PROJ / 'tests/auditor_preflight.json')
    assert auditor['verifier_source_sha256'] == sha(HERE / 'verify17.py')
    assert auditor['test_source_sha256'] == sha(HERE / 'check_audit17.py')
    config = read(HERE / 'protocol17.json')
    assert config['n_repetitions_per_regime'] == NREP and tuple(config['regimes']) == REGIMES
    assert config['sizes'] == SIZE and tuple(config['prefixes']) == PREFIXES
    assert config['confidence_events'] == EVENTS and config['alpha'] == ALPHA
    assert tuple(config['sample_sizes']) == SAMPLE_SIZES and config['primary_n'] == PRIMARY_N
    assert tuple(config['block_sizes']) == CERTIFICATE_BLOCK_SIZES
    assert tuple(config['target_bounds']) == BOUNDS and config['kappa'] == KAPPA
    assert config['mse_tolerance'] == EPSILON
    assert EVENTS == len(cases()) * len(PREFIXES) * len(SAMPLE_SIZES) * (len(CERTIFICATE_BLOCK_SIZES) + 2)
    source, seeds = sources(), streams()
    RUN.mkdir()
    dumpj(RUN / 'freeze.json', dict(utc=utc(), sources=source, streams=seeds,
        excluded_prior_seeds=sorted(prior_seeds()), manufactured_seeds=MANUFACTURED_SEEDS,
        predecessor_sha256=PINNED_16, protocol_sha256=sha(HERE / 'protocol17.json'),
        protocol=config, cases=cases(), NREP=NREP, NCASES=len(cases()), sizes=SIZE,
        prefixes=PREFIXES, sample_sizes=SAMPLE_SIZES, primary_n=PRIMARY_N,
        certificate_block_sizes=CERTIFICATE_BLOCK_SIZES, alpha=ALPHA,
        confidence_events=EVENTS, delta=DELTA, kappa=KAPPA, target_bounds=BOUNDS,
        mse_tolerance=EPSILON, geometry_roster=geometry_specs(), geometry_config=GEOMETRY_CONFIG,
        ridges=RIDGES, feature_slices=SLICE, numerical=numerical_configuration(),
        workers=WORKERS, numerical_threads_per_worker=1,
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, platform=platform.platform(), engine=C.metadata()))
    print('Frozen 24 cases, 144 fresh streams, 96 candidate heads per case, and 1008 confidence events.', flush=True)


def linked():
    record = read(RUN / 'freeze.json')
    if record['sources'] != sources() or record['streams'] != streams():
        raise ArithmeticError('Frozen scientific sources or stream roster changed')
    return record


def stage_files(stage):
    return {p.relative_to(RUN).as_posix(): sha(p)
            for p in sorted((RUN / stage).rglob('*')) if p.is_file()}


def seal(stage):
    dumpj(RUN / (stage + '_seal.json'), dict(utc=utc(), freeze_sha256=sha(RUN / 'freeze.json'),
                                           files=stage_files(stage)))


def checkseal(stage):
    record = read(RUN / (stage + '_seal.json'))
    if record['freeze_sha256'] != sha(RUN / 'freeze.json') or record['files'] != stage_files(stage):
        raise ArithmeticError('Sealed stage changed: ' + stage)


def require_fit_gate():
    gate = read(RUN / 'fit_validation.json')
    assert gate['passed'] is True and gate['checked_before_calibration_generation'] is True
    assert gate['checked_before_terminal_generation'] is True
    assert gate['freeze_sha256'] == sha(RUN / 'freeze.json')
    assert gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')
    return gate


def require_cal_gate():
    gate = read(RUN / 'cal_validation.json')
    assert gate['passed'] is True and gate['checked_before_terminal_generation'] is True
    assert gate['freeze_sha256'] == sha(RUN / 'freeze.json')
    assert gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')
    assert gate['cal_seal_sha256'] == sha(RUN / 'cal_seal.json')
    assert gate['precalibration_fit_gate_sha256'] == sha(RUN / 'fit_validation.json')
    return gate


def encode_bank(bank, x, concentrations, prefix=6, tag=''):
    start = time.perf_counter()
    scored = C.certified_bank(bank, x, prefix=prefix, precision=(80, 128, 192),
                              score_atol=1e-10, F_atol=1e-9)
    score_seconds = time.perf_counter() - start
    if not np.all(scored['accepted']):
        raise ArithmeticError('An interval feature query failed; no row may be dropped')
    precision, counts = np.unique(scored['precision_bits'], return_counts=True)
    events = [dict(tag=tag, kind='folding_scores', prefix=prefix, n=len(x),
        seconds=score_seconds, max_score_error=float(scored['score_error_bound'].max()),
        max_F_error=float(scored['F_error_bound'].max()),
        precision_counts={str(int(p)): int(n) for p, n in zip(precision, counts)})]
    block_sizes = BLOCKS if prefix == 6 else BLOCKS[:1]
    result = {}
    for omega in concentrations:
        start = time.perf_counter()
        nominal, low, high = PS.power_state(scored['score_lo'], scored['score_hi'],
                                          block_sizes, omega, tau=1e-6, atol=1e-9)
        elapsed = time.perf_counter() - start
        maximum_error = float(PS.radius(nominal, low, high).max())
        simplex_error = max(float(np.max(abs(nominal[:, a:b].sum(axis=1) - 1.)))
                            for a, b in feature_blocks(prefix))
        if maximum_error > 1e-9 or simplex_error > 2e-13:
            raise ArithmeticError('Power-state accuracy or simplex gate failed')
        result[float(omega)] = dict(F=nominal, F_lo=low, F_hi=high,
            score_lo=scored['score_lo'], score_hi=scored['score_hi'])
        events.append(dict(tag=tag, kind='power_state', prefix=prefix, n=len(x),
            concentration=float(omega), seconds=elapsed, max_P_error=maximum_error,
            max_block_sum_residual=simplex_error))
    return result, events


def encode_candidates(banks, x, tag='development'):
    outputs, events = {}, []
    for scaling in SCALINGS:
        for cap in CAPS:
            bank = banks[(scaling, cap)]
            if bank.block_sizes != BLOCKS or bank.scaler.mode != scaling or bank.cap != cap:
                raise ArithmeticError('Native bank does not match the fixed geometry roster')
            encoded, local_events = encode_bank(bank, x, CONCENTRATIONS, tag=tag)
            events += [dict(scaling=scaling, cap=cap, **event) for event in local_events]
            for spec in geometry_specs():
                if spec['scaling'] == scaling and spec['cap'] == cap:
                    outputs[spec['id']] = encoded[spec['concentration']]['F']
    if set(outputs) != {s['id'] for s in geometry_specs()}:
        raise ArithmeticError('Incomplete feature candidate roster')
    return outputs, events


def _head_quantities(x, y, coef, alpha, blocks):
    xm, ym = np.mean(x, axis=0), float(np.mean(y))
    xc, yc = x - xm, y - ym
    residual = xc @ coef - yc
    objective = float(np.mean(residual ** 2) + alpha * np.dot(coef, coef))
    gradient = 2. * (xc.T @ residual / len(x) + alpha * coef)
    norm_sum = float(sum(np.linalg.norm(coef[a:b]) for a, b in blocks))
    support = KAPPA * max(np.linalg.norm(gradient[a:b]) for a, b in blocks)
    gap = max(0., float(np.dot(gradient, coef) + support))
    return xm, ym, objective, gap, norm_sum


def fit_head(x, y, prefix, alpha):
    blocks = feature_blocks(prefix)
    fitted = HEAD.BlockBoundedRidge(alpha=alpha, kappa=KAPPA, bounds=BOUNDS,
        block_slices=[slice(a, b) for a, b in blocks], tol=HEAD_TOLERANCE,
        max_iter=HEAD_MAX_ITER).fit(x, y)
    raw = np.ascontiguousarray(fitted.coef_[:, 0])
    raw_norm = float(sum(np.sqrt(np.sum(np.asarray(raw[a:b], np.longdouble) ** 2))
                         for a, b in blocks))
    factor = 1. if raw_norm == 0 else min(1., (KAPPA - HEAD_INWARD_MARGIN) / raw_norm)
    coef = np.ascontiguousarray(raw * factor)
    xm, ym, objective, gap, norm_sum = _head_quantities(x, y, coef, alpha, blocks)
    intercept = float(ym - xm @ coef)
    if not np.isfinite(coef).all() or not np.isfinite(intercept) or norm_sum > KAPPA:
        raise ArithmeticError('Stored head violates finite values or the declared norm budget')
    return dict(coef=coef, intercept=intercept, alpha=float(alpha), prefix=prefix,
        blocks=blocks, kappa=KAPPA, bounds=BOUNDS, x_mean=xm, y_mean=ym,
        coef_before_guard=raw, inward_factor=factor, raw_norm_sum=raw_norm,
        norm_sum=norm_sum, objective=objective, gap=gap,
        original_optimizer_gap=float(fitted.optimization_gap_),
        original_optimizer_objective=float(fitted.objective_),
        iterations=int(fitted.n_iter_), tolerance=HEAD_TOLERANCE,
        max_iter=HEAD_MAX_ITER, inward_margin=HEAD_INWARD_MARGIN,
        converged=bool(gap <= HEAD_TOLERANCE * max(1., abs(objective))))


def predict_head(head, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != head['prefix'] or not np.isfinite(x).all():
        raise ValueError('Feature matrix does not match the stored head')
    out = np.clip(x @ head['coef'] + head['intercept'], *BOUNDS)
    if not np.isfinite(out).all():
        raise ArithmeticError('Nonfinite stored-head prediction')
    return out


def fit_paths(features, train_y, valid_y, *, train_slice=None, validation_slice=None):
    train_slice = slice(*SLICE['train']) if train_slice is None else train_slice
    validation_slice = slice(*SLICE['validation']) if validation_slice is None else validation_slice
    paths, grid = {}, np.empty((16, 2, 3))
    for spec in geometry_specs():
        f = features[spec['id']]
        heads, elapsed = {}, np.empty((2, 3))
        for pi, prefix in enumerate(PREFIXES):
            local = []
            for ri, ridge in enumerate(RIDGES):
                start = time.perf_counter()
                head = fit_head(f[train_slice, :prefix], train_y, prefix, ridge)
                elapsed[pi, ri] = time.perf_counter() - start
                pred = predict_head(head, f[validation_slice, :prefix])
                grid[spec['index'], pi, ri] = np.mean((valid_y - pred) ** 2)
                local.append(head)
            heads[str(prefix)] = local
        paths[spec['id']] = dict(spec=spec, heads=heads,
            validation_losses=grid[spec['index']].copy(), fit_seconds=elapsed)
    if not np.isfinite(grid).all():
        raise ArithmeticError('Nonfinite candidate selection score')
    return paths, grid


def select_models(paths, grid):
    # NumPy's first minimum implements the frozen geometry-index/ridge-index tie rule.
    geometry_index, largest_ridge = np.unravel_index(np.argmin(grid[:, -1, :]), (16, 3))
    spec = geometry_specs()[int(geometry_index)]
    models, selected = {}, {}
    for pi, prefix in enumerate(PREFIXES):
        ri = int(np.argmin(grid[geometry_index, pi]))
        models[str(prefix)] = paths[spec['id']]['heads'][str(prefix)][ri]
        selected[str(prefix)] = dict(prefix=prefix, ridge_index=ri, ridge=RIDGES[ri],
            validation_mse=float(grid[geometry_index, pi, ri]),
            coefficient_norm_sum=models[str(prefix)]['norm_sum'],
            optimization_gap=models[str(prefix)]['gap'],
            converged=models[str(prefix)]['converged'])
    return models, dict(geometry_id=spec['id'], geometry_index=int(geometry_index),
        geometry_spec=spec, largest_prefix_ridge_index=int(largest_ridge),
        selected_heads=selected,
        selection_rule='minimum largest-prefix validation MSE; geometry index then ridge index; conditional per-prefix ridge minimum',
        nested_geometry_shared=True, validation_selection_only=True)


def payload_cost(bank, concentration, head):
    compact = G.compact(bank, 1 if head['prefix'] == 2 else 2)
    components = dict(scaler_offset=compact.scaler.offset.nbytes,
        scaler_divisor=compact.scaler.divisor.nbytes,
        scaler_constant_mask=compact.scaler.constant_mask.nbytes,
        retained_centers=sum(block['cluster_centers'].nbytes for block in compact.blocks),
        native_encoder_arrays=sum(encoder[key].nbytes for block in compact.blocks
            for group in block['classifier'] for encoder in group
            for key in ('y_data_n_subspace', 'PC', 'B', 'y_data', 'sqrt_weights_matrix')),
        head_coefficient=head['coef'].nbytes,
        scalar_parameters=5 * 8)
    # Scalar payload is intercept, omega, tau and the two fixed clipping endpoints.
    payload = dict(bank=compact, concentration=concentration, coef=head['coef'],
                   intercept=head['intercept'], bounds=BOUNDS)
    return dict(prefix=head['prefix'], numeric_bytes=sum(components.values()),
        components=components, pickle_bytes=len(pickle.dumps(payload, protocol=5)),
        feature_blocks=len(compact.blocks), native_states=sum(compact.block_sizes),
        native_local_models=sum(len(group) for block in compact.blocks for group in block['classifier']),
        local_reference_columns=sum(encoder['y_data'].shape[1] for block in compact.blocks
            for group in block['classifier'] for encoder in group),
        scope='retained numeric deployment payload including centers; Python objects and runtime libraries excluded')


def fit_case(case):
    record, item = read(RUN / 'freeze.json'), case_record(case)
    if (RUN / 'cal').exists() or (RUN / 'test').exists():
        raise RuntimeError('All selections must precede calibration and testing')
    out = RUN / 'fit' / f'r{case}'
    out.mkdir()
    start, started = time.perf_counter(), utc()
    G.initialize_runtime()
    support = J.draw_support(SIZE['support'], pool_seed(record, case, 'support'))
    train = J.draw_episodes(SIZE['train'], pool_seed(record, case, 'train'), item['regime'])
    validation = J.draw_episodes(SIZE['validation'], pool_seed(record, case, 'validation'), item['regime'])
    arrays = {pool + '_' + key: value for pool, values in
              (('support', support), ('train', train), ('validation', validation))
              for key, value in values.items()}
    npz(out / 'pools.npz', **arrays)
    dumpj(out / 'data_receipt.json', dict(**item, started=started, finished=utc(),
        freeze_sha256=sha(RUN / 'freeze.json'),
        calibration_directory_absent=True, terminal_directory_absent=True,
        pool_seeds={pool: pool_seed(record, case, pool) for pool in ('support', 'train', 'validation')},
        geometry_seed=pool_seed(record, case, 'geometry'),
        support_targets_generated=False, shapes={key: value.shape for key, value in arrays.items()}))
    started_geometry = time.perf_counter()
    family = G.fit_family(support['X'], GEOMETRY_CONFIG, pool_seed(record, case, 'geometry'))
    geometry_seconds = time.perf_counter() - started_geometry
    banks = family['learned']
    if len(banks) != 4 or len(family['regrouped']) != 4:
        raise ArithmeticError('Incomplete geometry family')
    dump(out / 'banks.pkl', banks)
    dumpj(out / 'geometry_metadata.json', family['metadata'])
    del family['regrouped']
    x = np.vstack((support['X'], train['X'], validation['X']))
    encoding_start = time.perf_counter()
    features, events = encode_candidates(banks, x, tag='development_1024')
    encoding_seconds = time.perf_counter() - encoding_start
    dump(out / 'allfeaturecandidates.pkl', features)
    head_start = time.perf_counter()
    paths, grid = fit_paths(features, train['Y'], validation['Y'])
    head_seconds = time.perf_counter() - head_start
    dump(out / 'allheadpaths.pkl', paths)
    npz(out / 'grid.npz', validation_losses=grid)
    models, selected = select_models(paths, grid)
    spec = selected['geometry_spec']
    bank = G.compact(banks[(spec['scaling'], spec['cap'])], 2)
    dump(out / 'selected_bank.pkl', bank)
    dump(out / 'models.pkl', models)
    selected.update(item)
    dumpj(out / 'selected.json', selected)
    dumpj(out / 'events.json', events)
    costs = {str(prefix): payload_cost(bank, spec['concentration'], models[str(prefix)])
             for prefix in PREFIXES}
    if not costs['2']['numeric_bytes'] < costs['6']['numeric_bytes']:
        raise ArithmeticError('Declared prefix order does not increase deployment payload')
    dumpj(out / 'costs.json', costs)
    summary = dict(**item, started=started, finished=utc(), seconds=time.perf_counter()-start,
        geometry_fit_seconds=geometry_seconds, shared_encoding_seconds=encoding_seconds,
        heads_and_validation_seconds=head_seconds,
        summed_head_fit_seconds=float(sum(np.sum(v['fit_seconds']) for v in paths.values())),
        fit_candidate_heads=96, feature_configurations=16, native_learned_banks=4,
        native_regrouped_banks_fitted_and_charged=4,
        nonconverged_heads=sum(not h['converged'] for path in paths.values()
            for local in path['heads'].values() for h in local),
        maximum_head_gap=max(h['gap'] for path in paths.values() for local in path['heads'].values() for h in local),
        maximum_norm_sum=max(h['norm_sum'] for path in paths.values() for local in path['heads'].values() for h in local),
        selected=selected, costs=costs,
        no_calibration_reselection=True, no_postselection_refit=True)
    dumpj(out / 'summary.json', summary)
    print(f"fit case={case} {item['regime']} geometry={spec['id']} seconds={summary['seconds']:.3f}", flush=True)
    return summary


def summarize_fit():
    rows = [read(RUN / 'fit' / f'r{i}' / 'summary.json') for i in range(len(cases()))]
    dumpj(RUN / 'fit/summary.json', dict(by_case=rows, cases=len(rows),
        candidate_heads=sum(row['fit_candidate_heads'] for row in rows),
        feature_configurations=sum(row['feature_configurations'] for row in rows),
        all_selections_completed_before_calibration=not (RUN / 'cal').exists(),
        all_selections_completed_before_terminal=not (RUN / 'test').exists()))


def data_receipt(stage, case, started, data, frozen):
    item = case_record(case)
    record = dict(**item, started=started, finished=utc(),
        freeze_sha256=sha(RUN / 'freeze.json'), fit_seal_sha256=sha(RUN / 'fit_seal.json'),
        precalibration_fit_gate_sha256=sha(RUN / 'fit_validation.json'),
        selected_models_sha256=sha(RUN / 'fit' / f'r{case}' / 'models.pkl'),
        selected_bank_sha256=sha(RUN / 'fit' / f'r{case}' / 'selected_bank.pkl'),
        pool_seed=pool_seed(frozen, case, stage),
        shapes={key: value.shape for key, value in data.items()},
        frozen_models_unchanged=True)
    if stage == 'cal':
        record['terminal_directory_absent'] = not (RUN / 'test').exists()
    else:
        record.update(cal_seal_sha256=sha(RUN / 'cal_seal.json'),
            preterminal_cal_gate_sha256=sha(RUN / 'cal_validation.json'))
    return record


def calibrated_decisions(records, costs, regime, n):
    lookup = {prefix: next(row for row in records[str(prefix)] if row['n'] == n)
              for prefix in PREFIXES}
    uppers = {prefix: lookup[prefix]['upper']['upper'] for prefix in PREFIXES}
    result = {}
    for method in ('hierarchy', 'recycled', 'combined'):
        lows = {prefix: (lookup[prefix]['recycled']['lower'] if method == 'recycled'
                         else lookup[prefix][method + '_lower']) for prefix in PREFIXES}
        accepted = [prefix for prefix in PREFIXES if uppers[prefix] <= EPSILON]
        compatible = [prefix for prefix in PREFIXES if lows[prefix] <= EPSILON]
        low_prefix = min(compatible) if compatible else None
        high_prefix = min(accepted) if accepted else None
        refine = bool(lows[2] > EPSILON and uppers[6] <= EPSILON)
        excluded_all = bool(all(lows[prefix] > EPSILON for prefix in PREFIXES))
        resolved = refine if regime == 'repairable' else excluded_all
        result[method] = dict(lower_by_prefix=lows, upper_by_prefix=uppers,
            excluded_by_prefix={prefix: bool(lows[prefix] > EPSILON) for prefix in PREFIXES},
            accepted_by_prefix={prefix: bool(uppers[prefix] <= EPSILON) for prefix in PREFIXES},
            compatible_minimum_prefix=low_prefix, adequate_minimum_prefix=high_prefix,
            cost_lower_numeric_bytes=None if low_prefix is None else costs[str(low_prefix)]['numeric_bytes'],
            cost_upper_numeric_bytes=None if high_prefix is None else costs[str(high_prefix)]['numeric_bytes'],
            minimum_prefix_certified=bool(high_prefix is not None and low_prefix == high_prefix),
            refinement_certified=refine, all_classes_excluded=excluded_all,
            resolved_primary_regime_question=bool(resolved))
    h, r = (result[name]['resolved_primary_regime_question'] for name in ('hierarchy', 'recycled'))
    pair = 'both' if h and r else 'hierarchy_only' if h else 'recycled_only' if r else 'neither'
    result['comparison'] = dict(paired_resolution=pair,
        same_fitted_upper_endpoints=True, no_additional_optimization_for_recycled=True,
        primary_n=n == PRIMARY_N, regime_question='refinement' if regime == 'repairable' else 'joint_class_exclusion')
    result['upper_only'] = dict(adequate_minimum_prefix=result['hierarchy']['adequate_minimum_prefix'],
        class_exclusions_available=False, resolution='fitted adequacy only; does not exclude an entire class')
    return result


def calibrate_case(case):
    frozen, item = read(RUN / 'freeze.json'), case_record(case)
    require_fit_gate()
    if (RUN / 'test').exists():
        raise RuntimeError('Calibration must complete before terminal generation')
    fit = RUN / 'fit' / f'r{case}'
    out = RUN / 'cal' / f'r{case}'
    out.mkdir()
    start, started = time.perf_counter(), utc()
    data = J.draw_episodes(SIZE['cal'], pool_seed(frozen, case, 'cal'), item['regime'])
    npz(out / 'observations.npz', **data)
    dumpj(out / 'data_receipt.json', data_receipt('cal', case, started, data, frozen))
    selected, models, bank = read(fit / 'selected.json'), load(fit / 'models.pkl'), load(fit / 'selected_bank.pkl')
    omega = selected['geometry_spec']['concentration']
    encode_start = time.perf_counter()
    encodings, events = encode_bank(bank, data['X'], (omega,), tag='calibration_4096')
    encoding_seconds = time.perf_counter() - encode_start
    features = encodings[omega]
    npz(out / 'features.npz', **features)
    pred = np.column_stack([predict_head(models[str(prefix)], features['F'][:, :prefix]) for prefix in PREFIXES])
    loss = (data['Y'][:, None] - pred) ** 2
    npz(out / 'predictions.npz', pred=pred, loss=loss, prefixes=np.asarray(PREFIXES))
    evidence, records, certificate_seconds = {}, {}, {}
    for pi, prefix in enumerate(PREFIXES):
        before = time.perf_counter()
        answer = Q.evaluate_calibration(features['F'][:, :prefix], data['Y'], pred[:, pi],
            sample_sizes=SAMPLE_SIZES, block_sizes=CERTIFICATE_BLOCK_SIZES,
            block_slices=feature_blocks(prefix), kappa=KAPPA, target_bounds=BOUNDS,
            alpha=ALPHA, solver_config=SOLVER_CONFIG)
        certificate_seconds[str(prefix)] = time.perf_counter() - before
        evidence[str(prefix)] = answer
        records[str(prefix)] = answer['records']
    costs = read(fit / 'costs.json')
    decisions = {str(n): calibrated_decisions(records, costs, item['regime'], n) for n in SAMPLE_SIZES}
    dump(out / 'dual_records.pkl', evidence)
    dumpj(out / 'calibration.json', dict(**item, by_prefix=records, decisions=decisions,
        alpha=ALPHA, confidence_events_per_case=42, same_solved_duals=True,
        primary_n=PRIMARY_N, selected_models_sha256=sha(fit / 'models.pkl')))
    dumpj(out / 'events.json', events)
    diagnostics = {str(prefix): {str(b): dict(
        programs=len(evidence[str(prefix)]['block_evidence'][str(b)]['sse_lower']),
        unconverged=int(np.sum(~evidence[str(prefix)]['block_evidence'][str(b)]['converged'])),
        max_gap=float(np.max(evidence[str(prefix)]['block_evidence'][str(b)]['gap'])),
        max_primal_feasibility_residual=float(np.max(evidence[str(prefix)]['block_evidence'][str(b)]['feasibility_residual'])),
        max_iterations=int(np.max(evidence[str(prefix)]['block_evidence'][str(b)]['iterations'])))
        for b in CERTIFICATE_BLOCK_SIZES} for prefix in PREFIXES}
    summary = dict(**item, started=started, finished=utc(), seconds=time.perf_counter()-start,
        shared_encoding_seconds=encoding_seconds, certificate_seconds=certificate_seconds,
        block_solve_seconds={prefix: answer['block_solve_seconds'] for prefix, answer in evidence.items()},
        solver_diagnostics=diagnostics, decisions=decisions, by_prefix=records,
        models_unchanged_sha256=sha(fit / 'models.pkl'),
        geometry_unchanged_sha256=sha(fit / 'selected_bank.pkl'))
    dumpj(out / 'summary.json', summary)
    print(f"cal case={case} {item['regime']} seconds={summary['seconds']:.3f}", flush=True)
    return summary


def summarize_cal():
    rows = [read(RUN / 'cal' / f'r{i}' / 'summary.json') for i in range(len(cases()))]
    paired = []
    for regime in REGIMES:
        for n in SAMPLE_SIZES:
            local = [r for r in rows if r['regime'] == regime]
            counts = {name: sum(r['decisions'][str(n)]['comparison']['paired_resolution'] == name for r in local)
                      for name in ('both', 'hierarchy_only', 'recycled_only', 'neither')}
            paired.append(dict(regime=regime, n=n, primary_n=n == PRIMARY_N,
                repetitions=len(local), paired_counts=counts,
                resolved_by_method={method: sum(r['decisions'][str(n)][method]['resolved_primary_regime_question'] for r in local)
                    for method in ('hierarchy', 'recycled', 'combined')},
                scope='descriptive paired procedure counts, no population superiority inference'))
    dumpj(RUN / 'cal/summary.json', dict(by_case=rows, paired_resolution=paired,
        cases=len(rows), selected_classes=len(rows)*2, class_sample_records=len(rows)*2*3,
        confidence_events=EVENTS, alpha=ALPHA, delta=DELTA, primary_n=PRIMARY_N,
        numerical_block_programs=sum(v['programs'] for r in rows for p in r['solver_diagnostics'].values() for v in p.values()),
        no_new_optimization_for_recycled=True,
        same_per_endpoint_alpha=True, block_family_events_per_class_sample=5,
        recycled_events_per_class_sample=1,
        calibration_completed_before_terminal=not (RUN / 'test').exists()))


def isolated_timing(bank, omega, head, x):
    prefix = head['prefix']
    compact = G.compact(bank, 1 if prefix == 2 else 2)
    start = time.perf_counter()
    encoded, events = encode_bank(compact, x, (omega,), prefix=prefix,
                                  tag='standalone_deployment_' + str(prefix))
    encoding_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = predict_head(head, encoded[omega]['F'])
    head_seconds = time.perf_counter() - start
    return dict(prefix=prefix, rows=len(x), encoder_seconds=encoding_seconds,
        head_seconds=head_seconds, total_seconds=encoding_seconds+head_seconds,
        events=events, warm_process_uncached_features=True,
        accuracy_scope='interval-assured forward features and float64 head'), pred


def test_case(case):
    frozen, item = read(RUN / 'freeze.json'), case_record(case)
    require_fit_gate()
    require_cal_gate()
    fit, cal = RUN / 'fit' / f'r{case}', RUN / 'cal' / f'r{case}'
    out = RUN / 'test' / f'r{case}'
    out.mkdir()
    start, started = time.perf_counter(), utc()
    data = J.draw_episodes(SIZE['test'], pool_seed(frozen, case, 'test'), item['regime'])
    npz(out / 'observations.npz', **data)
    dumpj(out / 'data_receipt.json', data_receipt('test', case, started, data, frozen))
    selected, models, bank = read(fit / 'selected.json'), load(fit / 'models.pkl'), load(fit / 'selected_bank.pkl')
    omega = selected['geometry_spec']['concentration']
    before = time.perf_counter()
    encoded, events = encode_bank(bank, data['X'], (omega,), tag='terminal_1024')
    shared_encoding_seconds = time.perf_counter() - before
    features = encoded[omega]
    npz(out / 'features.npz', **features)
    pred = np.column_stack([predict_head(models[str(prefix)], features['F'][:, :prefix]) for prefix in PREFIXES])
    loss = (data['Y'][:, None] - pred)**2
    mean, variance = J.conditional_moments(data['X'], item['regime'])
    expected_loss = (mean[:, None] - pred)**2 + variance[:, None]
    npz(out / 'predictions.npz', pred=pred, loss=loss, prefixes=np.asarray(PREFIXES),
        conditional_mean=mean, conditional_variance=variance,
        conditional_expected_loss=expected_loss)
    timing, max_timing_replay_error = {}, 0.
    for pi, prefix in enumerate(PREFIXES):
        timing[str(prefix)], timed_pred = isolated_timing(bank, omega, models[str(prefix)], data['X'])
        error = float(np.max(abs(timed_pred - pred[:, pi])))
        max_timing_replay_error = max(max_timing_replay_error, error)
        if error > 1e-11:
            raise ArithmeticError('Independent prefix timing changed stored predictions')
    dumpj(out / 'events.json', events)
    dumpj(out / 'deployment_timing.json', timing)
    results = [dict(**item, prefix=prefix, observed_mse=float(np.mean(loss[:, pi])),
        mean_conditional_expected_loss=float(np.mean(expected_loss[:, pi])),
        population_bayes_risk=J.bayes_risk(item['regime']),
        population_class_minimum_known=item['regime'] == 'irreducible',
        known_population_class_minimum=J.bayes_risk(item['regime']) if item['regime'] == 'irreducible' else None,
        conditional_expected_loss_is_population_risk=False,
        selected=selected['selected_heads'][str(prefix)], geometry=selected['geometry_spec'],
        costs=read(fit / 'costs.json')[str(prefix)]) for pi, prefix in enumerate(PREFIXES)]
    dumpj(out / 'results.json', results)
    summary = dict(**item, started=started, finished=utc(), seconds=time.perf_counter()-start,
        rows=SIZE['test'], forecast_values=SIZE['test']*2, shared_encoding_seconds=shared_encoding_seconds,
        standalone_deployment=timing, max_standalone_prediction_error=max_timing_replay_error,
        results=results, frozen_calibration_decisions=read(cal / 'calibration.json')['decisions'],
        models_unchanged_sha256=sha(fit / 'models.pkl'), geometry_unchanged_sha256=sha(fit / 'selected_bank.pkl'))
    dumpj(out / 'summary.json', summary)
    print(f"test case={case} {item['regime']} seconds={summary['seconds']:.3f}", flush=True)
    return summary


def summarize_test():
    rows = [read(RUN / 'test' / f'r{i}' / 'summary.json') for i in range(len(cases()))]
    aggregates = []
    for regime in REGIMES:
        for prefix in PREFIXES:
            local = [next(r for r in row['results'] if r['prefix'] == prefix)
                     for row in rows if row['regime'] == regime]
            aggregates.append(dict(regime=regime, prefix=prefix, repetitions=len(local),
                mean_observed_mse=float(np.mean([r['observed_mse'] for r in local])),
                mean_conditional_expected_loss=float(np.mean([r['mean_conditional_expected_loss'] for r in local])),
                population_bayes_risk=J.bayes_risk(regime),
                mean_numeric_payload_bytes=float(np.mean([r['costs']['numeric_bytes'] for r in local]))))
    dumpj(RUN / 'test/summary.json', dict(by_case=rows, aggregates=aggregates, cases=len(rows),
        terminal_episodes=len(rows)*SIZE['test'], terminal_scalar_forecasts=len(rows)*SIZE['test']*len(PREFIXES),
        terminal_targets=len(rows)*SIZE['test'], inferential_unit='complete independent learning repetition',
        precalibration_fit_gate_sha256=sha(RUN / 'fit_validation.json'),
        preterminal_cal_gate_sha256=sha(RUN / 'cal_validation.json'),
        scope='independent terminal description; calibration decisions are frozen and not changed by these results'))


def run_stage(stage):
    linked()
    if stage == 'fit':
        if (RUN / 'cal').exists() or (RUN / 'test').exists():
            raise RuntimeError('Fit must precede later data')
        function, summarize = fit_case, summarize_fit
    elif stage == 'cal':
        checkseal('fit')
        require_fit_gate()
        if (RUN / 'test').exists():
            raise RuntimeError('Calibration must precede terminal data')
        function, summarize = calibrate_case, summarize_cal
    elif stage == 'test':
        checkseal('fit')
        checkseal('cal')
        require_fit_gate()
        require_cal_gate()
        function, summarize = test_case, summarize_test
    else:
        raise ValueError('Unknown scientific stage')
    folder = RUN / stage
    folder.mkdir()
    start, started = time.perf_counter(), utc()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(function, item['case']): item for item in cases()}
        for future in as_completed(futures):
            future.result()
    summarize()
    dumpj(folder / 'stage_receipt.json', dict(stage=stage, started=started, finished=utc(),
        seconds=time.perf_counter()-start, cases=len(cases()), completed_first_run=True,
        freeze_sha256=sha(RUN / 'freeze.json')))
    seal(stage)
    print('Sealed ' + stage, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'fit', 'cal', 'test'))
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        run_stage(args.stage)


if __name__ == '__main__':
    main()
