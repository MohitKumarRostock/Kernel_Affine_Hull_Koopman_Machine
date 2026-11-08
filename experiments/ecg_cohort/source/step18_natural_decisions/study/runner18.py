"""Write-once natural-task Step 18 experiment. No scientific work on import.

Stages: preflight, freeze, fit, cal, terminal, census. Independent audit gates
are written by the root verifier between fit/cal and cal/terminal. Evaluation
opens one row per patient and retains all finite feasible heads, irrespective
of optimization convergence. No calibration-driven re-selection or refit.
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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BASE = PROJ.parent
RUN = HERE / 'run'
TESTS = PROJ / 'tests'
for _folder in ('step6_geometry_regimes/src', 'step11_output_accuracy',
                'step11_output_accuracy/engine',
                'risk_source/extensions/certified_forecasting/src'):
    sys.path.insert(0, str(BASE / _folder))
import numpy as np
import scipy
import sklearn
import geometry as G
import certified as C
import soft_state as PS
import certified_forecasting.models as HEAD
import baseline18 as B

SIZE = dict(support=512, train=3072, validation=1536, calibration=4096, terminal=4096)
PREFIXES = (2, 6, 14)
BLOCKS = (2, 4, 8)
SCALINGS = ('minmax01', 'standard')
CAPS = (2, 8, 20)
CONCENTRATIONS = (.5, 4., 16., 64.)
RIDGES = (0., 1e-4, 1e-2)
KAPPAS = (1., 2., 4.)
BOUNDS = (-1., 1.)
GEOMETRY_SEED = 2026110618
MANUFACTURED_SEED = 2026110618001
SAMPLE_SIZES = (512, 2048, 4096)
ALPHA = .05 / 9
POPULATION_SIZE = 8192
BLOCK_SIZE = 32
HEAD_TOLERANCE = 1e-7
HEAD_MAX_ITER = 20000
HEAD_INWARD_MARGIN = 1e-12
SOLVER_CONFIG = dict(max_iter=4000, tolerance=1e-7, check_every=20)
GEOMETRY_CONFIG = dict(scalings=SCALINGS, caps=CAPS, cluster_counts=BLOCKS,
                       max_fit=512, max_references=24, tau=1e-6)
SCORE_ATOL = 1e-12
FEATURE_ATOL = 1e-9
PRECISION = (80, 128, 192)
TIMING_ROWS = 512
TIMING_REPEATS = 3


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
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=json_default)
        f.write('\n')


def dump(path, value):
    with Path(path).open('xb') as f:
        pickle.dump(value, f, protocol=5)


def load(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def data_module():
    import data18
    return data18


def certificate_module():
    import certificate18
    return certificate18


def feature_blocks(prefix):
    if prefix not in PREFIXES:
        raise ValueError('Prefix must end at a declared complete block')
    edges = np.cumsum((0,) + BLOCKS)
    return tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b <= prefix)


def geometry_specs():
    return [dict(id=f'g{i:02d}', index=i, scaling=s, cap=q, concentration=w)
            for i, (s, q, w) in enumerate((s, q, w) for s in SCALINGS
                for q in CAPS for w in CONCENTRATIONS)]


def menu():
    return dict(schema='step18-model-menu-v1', sizes=SIZE, input_dimension=32,
        blocks=BLOCKS, prefixes=PREFIXES, geometry=geometry_specs(),
        geometry_configuration=GEOMETRY_CONFIG, ridges=RIDGES, kappas=KAPPAS,
        head_candidates=648, baseline_candidates=B.grid(), baseline_count=140,
        baseline_winner_slots=13, geometry_seed=GEOMETRY_SEED,
        baseline_seed=B.SEED, manufactured_seed=MANUFACTURED_SEED,
        bounds=BOUNDS, score_atol=SCORE_ATOL, feature_atol=FEATURE_ATOL,
        precision=PRECISION, head_tolerance=HEAD_TOLERANCE,
        head_max_iter=HEAD_MAX_ITER, head_inward_margin=HEAD_INWARD_MARGIN,
        sample_sizes=SAMPLE_SIZES, alpha=ALPHA, population_size=POPULATION_SIZE,
        certificate_block_size=BLOCK_SIZE, solver_config=SOLVER_CONFIG,
        selection='independent minimum validation MSE per KAHM prefix; candidate index breaks ties',
        reference_selection='minimum validation MSE among frozen non-KAHM winner slots; baseline candidate index breaks ties',
        success='canonical KAHM census MSE upper <= outward-lower(1.10 * reference census MSE lower) AND KAHM numeric payload strictly smaller',
        class_exclusion='class lower > outward-upper(1.10 * reference census MSE upper)',
        census_arithmetic='binary64 outward loss bounds; long-double outward accumulation/division and outward binary64 conversion; guarded arithmetic, not MPFI proof',
        timing_rows=TIMING_ROWS, timing_repeats=TIMING_REPEATS,
        numerical_assurance='KAHM MPFI/MPFR forward enclosures plus outward head arithmetic; baselines float64')


def sources():
    paths = {Path(__file__), Path(B.__file__), Path(G.__file__), Path(C.__file__),
             Path(PS.__file__), Path(HEAD.__file__),
             Path(HEAD.__file__).with_name('__init__.py'),
             BASE / 'step11_output_accuracy/engine/interval.cpp',
             BASE / 'step11_output_accuracy/softsrc/power_box.cpp',
             PROJ / 'PROTOCOL.txt', HERE / 'model_menu18.json',
             TESTS / 'runner_preflight18.json', TESTS / 'runner_manufactured18.npz',
             TESTS / 'certificate_preflight.json', HERE / 'check_certificate18.py',
             PROJ / 'references/ptbxl_v103/manufactured_data18_tests.json'}
    j, q = data_module(), certificate_module()
    paths.update(j.source_files())
    paths.add(Path(q.__file__))
    if hasattr(q, 'source_files'):
        paths.update(q.source_files())
    return {**G.source_hashes(), **{str(Path(p).resolve().relative_to(BASE)): sha(p)
                                  for p in sorted(paths)}}


def files_in(path):
    return {str(p.relative_to(RUN)): sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def seal(stage):
    dumpj(RUN / f'{stage}_seal.json', dict(freeze_sha256=sha(RUN / 'freeze.json'),
        files=files_in(RUN / stage), sealed=utc()))


def check_seal(stage):
    record = read(RUN / f'{stage}_seal.json')
    if record['freeze_sha256'] != sha(RUN / 'freeze.json') or record['files'] != files_in(RUN / stage):
        raise ArithmeticError(f'{stage} stage seal mismatch')


def linked():
    frozen = read(RUN / 'freeze.json')
    if frozen['sources'] != sources():
        raise ArithmeticError('Frozen scientific source closure changed')
    if frozen['model_menu'] != json.loads(json.dumps(menu())):
        raise ArithmeticError('Frozen scientific menu changed')
    return frozen


def require_fit_gate():
    linked()
    check_seal('fit')
    gate = read(RUN / 'fit_validation.json')
    if not (gate['passed'] is True and gate['freeze_sha256'] == sha(RUN / 'freeze.json')
            and gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')):
        raise ArithmeticError('Independent fit gate missing or mismatched')
    return gate


def require_cal_gate():
    require_fit_gate()
    check_seal('cal')
    gate = read(RUN / 'cal_validation.json')
    if not (gate['passed'] is True and gate['freeze_sha256'] == sha(RUN / 'freeze.json')
            and gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')
            and gate['cal_seal_sha256'] == sha(RUN / 'cal_seal.json')
            and gate['precalibration_fit_gate_sha256'] == sha(RUN / 'fit_validation.json')):
        raise ArithmeticError('Independent calibration gate missing or mismatched')
    return gate


def stage_data(stage):
    j = data_module()
    j.extract_stage(stage, freeze_path=RUN / 'freeze.json', output_root=RUN / 'data')
    data = j.load_stage(stage, output_root=RUN / 'data')
    if data['X'].shape != (SIZE[stage], 32) or not np.isfinite(data['X']).all():
        raise ArithmeticError('Data stage shape/finite gate failed')
    if stage == 'support':
        if 'Y' in data:
            raise ArithmeticError('Support targets must not be generated or exposed')
    elif data['Y'].shape != (SIZE[stage],) or not np.isfinite(data['Y']).all() or np.max(abs(data['Y'])) > 1:
        raise ArithmeticError('Data target shape/bounds gate failed')
    if len(set(np.asarray(data['patient_id']).tolist())) != SIZE[stage]:
        raise ArithmeticError('More than one row per patient')
    return data


def upper_l2(x):
    """Outward upper Euclidean norm along the last axis in binary64."""
    x = np.asarray(x, dtype=np.float64)
    acc = np.zeros(x.shape[:-1], dtype=np.float64)
    for i in range(x.shape[-1]):
        term = np.nextafter(x[..., i] * x[..., i], np.inf)
        acc = np.nextafter(acc + term, np.inf)
    return np.nextafter(np.sqrt(acc), np.inf)


def block_radii(nominal, low, high, prefix):
    coordinate = np.nextafter(np.maximum(np.abs(nominal-low), np.abs(high-nominal)), np.inf)
    return np.column_stack([upper_l2(coordinate[:, a:b]) for a, b in feature_blocks(prefix)])


def prediction_interval(head, low, high):
    """Enclose the real stored-coefficient head using outward multiply/add."""
    coef = np.asarray(head['coef'], dtype=float)
    lo = np.full(len(low), float(head['intercept']))
    hi = lo.copy()
    for i, c in enumerate(coef):
        a, b = c * low[:, i], c * high[:, i]
        lo = np.nextafter(lo + np.nextafter(np.minimum(a, b), -np.inf), -np.inf)
        hi = np.nextafter(hi + np.nextafter(np.maximum(a, b), np.inf), np.inf)
    return np.clip(lo, *BOUNDS), np.clip(hi, *BOUNDS)


def loss_upper(y, lo, hi):
    left = np.nextafter(np.abs(y-lo), np.inf)
    right = np.nextafter(np.abs(y-hi), np.inf)
    maximum = np.maximum(left, right)
    return np.minimum(4., np.nextafter(maximum*maximum, np.inf))


def loss_lower(y, lo, hi):
    distance = np.maximum(np.maximum(lo-y, y-hi), 0.)
    distance = np.maximum(0., np.nextafter(distance, -np.inf))
    return np.maximum(0., np.nextafter(distance*distance, -np.inf))


def mean_bound(values, direction):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Mean bound requires a nonempty finite vector')
    target = np.longdouble(np.inf if direction == 'upper' else -np.inf)
    total = np.longdouble(0.)
    for value in values:
        total = np.nextafter(total+np.longdouble(value), target)
    value = np.nextafter(total/np.longdouble(len(values)), target)
    answer = float(np.nextafter(float(value), float(target)))
    return max(0., answer) if np.all(values >= 0) else answer


def mean_upper(values):
    return mean_bound(values, 'upper')


def threshold_bound(reference_mse, direction):
    target=np.longdouble(np.inf if direction=='upper' else -np.inf)
    value=np.nextafter(np.longdouble(11)*np.longdouble(reference_mse),target)
    value=np.nextafter(value/np.longdouble(10),target)
    return max(0.,float(np.nextafter(float(value),float(target))))


def encode_bank(bank, x, concentrations, prefix=14, tag='', adapter=None):
    start = time.perf_counter()
    scored = (C.certified_bank(bank, x, prefix=prefix, precision=PRECISION,
                score_atol=SCORE_ATOL, F_atol=FEATURE_ATOL) if adapter is None
              else adapter.run(x))
    score_seconds = time.perf_counter()-start
    if not np.all(scored['accepted']):
        raise ArithmeticError('Interval query failed; required rows may not be dropped')
    precision, counts = np.unique(scored['precision_bits'], return_counts=True)
    events = [dict(tag=tag, kind='folding_scores', prefix=prefix, rows=len(x),
        seconds=score_seconds, max_score_error=float(np.max(scored['score_error_bound'])),
        max_F_error=float(np.max(scored['F_error_bound'])),
        precision_counts={str(int(p)):int(n) for p,n in zip(precision,counts)})]
    out = {}
    block_sizes = BLOCKS[:len(feature_blocks(prefix))]
    for omega in concentrations:
        start = time.perf_counter()
        f, low, high = PS.power_state(scored['score_lo'], scored['score_hi'],
            block_sizes, omega, tau=1e-6, atol=FEATURE_ATOL)
        radius = block_radii(f, low, high, prefix)
        error = float(PS.radius(f, low, high).max())
        simplex = max(float(np.max(abs(f[:,a:b].sum(axis=1)-1))) for a,b in feature_blocks(prefix))
        if error > FEATURE_ATOL or simplex > 2e-13 or not np.isfinite(radius).all():
            raise ArithmeticError('Power-state accuracy/simplex gate failed')
        out[float(omega)] = dict(F=f, F_lo=low, F_hi=high, block_radius=radius,
                                score_lo=scored['score_lo'], score_hi=scored['score_hi'])
        events.append(dict(tag=tag, kind='power_state', prefix=prefix, rows=len(x),
            concentration=float(omega), seconds=time.perf_counter()-start,
            max_coordinate_error=error, max_block_radius=float(radius.max()),
            max_simplex_residual=simplex))
    return out, events


def fit_head(x, y, prefix, alpha, kappa):
    blocks = feature_blocks(prefix)
    fitted = HEAD.BlockBoundedRidge(alpha=alpha, kappa=kappa, bounds=BOUNDS,
        block_slices=[slice(a,b) for a,b in blocks], tol=HEAD_TOLERANCE,
        max_iter=HEAD_MAX_ITER).fit(x,y)
    raw = np.ascontiguousarray(fitted.coef_[:,0])
    raw_norm = float(sum(np.sqrt(np.sum(np.asarray(raw[a:b],np.longdouble)**2)) for a,b in blocks))
    factor = 1. if raw_norm == 0 else min(1., (kappa-HEAD_INWARD_MARGIN)/raw_norm)
    coef = np.ascontiguousarray(raw*factor)
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x-xm, y-ym
    residual = xc@coef-yc
    objective = float(np.mean(residual**2) + alpha*np.dot(coef,coef))
    gradient = 2*(xc.T@residual/len(x) + alpha*coef)
    norm_sum = float(sum(np.linalg.norm(coef[a:b]) for a,b in blocks))
    gap = max(0.,float(np.dot(gradient,coef)+kappa*max(np.linalg.norm(gradient[a:b]) for a,b in blocks)))
    intercept = float(ym-xm@coef)
    if not np.isfinite(coef).all() or not np.isfinite(intercept) or norm_sum > kappa:
        raise ArithmeticError('Stored head is not finite/feasible')
    return dict(coef=coef, intercept=intercept, alpha=float(alpha), prefix=prefix,
        blocks=blocks, kappa=float(kappa), bounds=BOUNDS, x_mean=xm, y_mean=ym,
        coef_before_guard=raw, inward_factor=factor, raw_norm_sum=raw_norm,
        norm_sum=norm_sum, objective=objective, gap=gap,
        original_optimizer_gap=float(fitted.optimization_gap_),
        original_optimizer_objective=float(fitted.objective_), iterations=int(fitted.n_iter_),
        tolerance=HEAD_TOLERANCE, max_iter=HEAD_MAX_ITER,
        inward_margin=HEAD_INWARD_MARGIN,
        converged=bool(gap <= HEAD_TOLERANCE*max(1.,abs(objective))))


def predict_head(head, f):
    result = np.clip(np.asarray(f)@head['coef']+head['intercept'], *BOUNDS)
    if result.shape != (len(f),) or not np.isfinite(result).all():
        raise ArithmeticError('Invalid stored head prediction')
    return result


def payload_cost(bank, concentration, head):
    compact = G.compact(bank, len(feature_blocks(head['prefix'])))
    components = dict(scaler_offset=compact.scaler.offset.nbytes,
        scaler_divisor=compact.scaler.divisor.nbytes,
        scaler_constant_mask=compact.scaler.constant_mask.nbytes,
        retained_centers=sum(b['cluster_centers'].nbytes for b in compact.blocks),
        native_encoder_arrays=sum(a[k].nbytes for b in compact.blocks for group in b['classifier']
            for a in group for k in ('y_data_n_subspace','PC','B','y_data','sqrt_weights_matrix')),
        head_coefficient=head['coef'].nbytes, scalar_parameters=5*8)
    payload = dict(bank=compact, concentration=concentration, coef=head['coef'],
                   intercept=head['intercept'], bounds=BOUNDS)
    return dict(numeric_bytes=int(sum(components.values())), components=components,
        pickle_bytes=len(pickle.dumps(payload,protocol=5)), prefix=head['prefix'],
        feature_blocks=len(compact.blocks), native_states=sum(compact.block_sizes),
        native_local_models=sum(len(g) for b in compact.blocks for g in b['classifier']),
        local_reference_columns=sum(a['y_data'].shape[1] for b in compact.blocks for g in b['classifier'] for a in g),
        scope='retained numeric query payload including scaler/centers/encoder/head; training diagnostics,Python/runtime/working memory excluded')


def freeze():
    preflight = read(TESTS/'runner_preflight18.json')
    if preflight['passed'] is not True:
        raise ArithmeticError('Manufactured preflight did not pass')
    for name, digest in preflight['producer_sha256'].items():
        if sha(BASE/name) != digest:
            raise ArithmeticError('Runner/baseline producer changed since manufactured preflight')
    data_gate_path=PROJ/'references/ptbxl_v103/manufactured_data18_tests.json'
    data_gate=read(data_gate_path)
    if not (data_gate['passed'] is True and data_gate['data_module_sha256']==sha(HERE/'data18.py')
            and data_gate['source_sha256']==sha(HERE/'manufactured_data18_tests.py')):
        raise ArithmeticError('Data manufactured preflight source/gate mismatch')
    cert_gate=read(TESTS/'certificate_preflight.json')
    if not (cert_gate['passed'] is True and cert_gate['certificate_source_sha256']==sha(HERE/'certificate18.py')
            and cert_gate['solver_source_sha256']==sha(HERE/'block_solver18.py')
            and cert_gate['test_source_sha256']==sha(HERE/'check_certificate18.py')):
        raise ArithmeticError('Certificate manufactured preflight source/gate mismatch')
    if read(HERE/'model_menu18.json') != json.loads(json.dumps(menu())):
        raise ArithmeticError('Declared model menu differs from code')
    RUN.mkdir()
    dumpj(RUN/'freeze.json', dict(schema='step18-freeze-v1', frozen=utc(),
        sources=sources(), model_menu=menu(), data_protocol=data_module().DATA_PROTOCOL,
        manufactured_preflight_sha256=sha(TESTS/'runner_preflight18.json'),
        data_preflight_sha256=sha(data_gate_path),
        certificate_preflight_sha256=sha(TESTS/'certificate_preflight.json'),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, architecture=platform.machine(),
        engine_environment=C.metadata(), scientific_stage_started=False))


def fit():
    linked()
    if (RUN/'cal').exists() or (RUN/'terminal').exists():
        raise RuntimeError('All development must precede calibration and terminal')
    out=RUN/'fit';out.mkdir()
    started=utc();start=time.perf_counter()
    support,train,val=(stage_data(s) for s in ('support','train','validation'))
    patient_sets=[set(v['patient_id'].tolist()) for v in (support,train,val)]
    if any(patient_sets[i]&patient_sets[j] for i in range(3) for j in range(i)):
        raise ArithmeticError('Development patient pools overlap')
    npz(out/'observations.npz',support_X=support['X'],train_X=train['X'],train_Y=train['Y'],
        validation_X=val['X'],validation_Y=val['Y'],support_patient_id=support['patient_id'],
        train_patient_id=train['patient_id'],validation_patient_id=val['patient_id'])
    G.initialize_runtime()
    before=time.perf_counter()
    family=G.fit_family(support['X'],GEOMETRY_CONFIG,GEOMETRY_SEED)
    geometry_seconds=time.perf_counter()-before
    banks=family['learned']
    if len(banks)!=6 or len(family['regrouped'])!=6:
        raise ArithmeticError('Incomplete native geometry roster')
    dump(out/'banks.pkl',banks);dumpj(out/'geometry_metadata.json',family['metadata'])
    del family['regrouped']
    x=np.concatenate((train['X'],val['X']))
    features,events={},[]
    before=time.perf_counter()
    for scaling in SCALINGS:
        for cap in CAPS:
            encoded,local=encode_bank(banks[(scaling,cap)],x,CONCENTRATIONS,tag='development')
            events.extend(dict(scaling=scaling,cap=cap,**e) for e in local)
            for spec in geometry_specs():
                if (spec['scaling'],spec['cap'])==(scaling,cap):
                    features[spec['id']]=encoded[spec['concentration']]['F']
    encoding_seconds=time.perf_counter()-before
    npz(out/'feature_candidates.npz',**features)
    heads,rows,preds={},[],[]
    for spec in geometry_specs():
        f=features[spec['id']]
        for prefix in PREFIXES:
            for kappa in KAPPAS:
                for alpha in RIDGES:
                    before=time.perf_counter()
                    head=fit_head(f[:SIZE['train'],:prefix],train['Y'],prefix,alpha,kappa)
                    pred=predict_head(head,f[SIZE['train']:,:prefix])
                    cid=f'k{len(rows):03d}'
                    cost=payload_cost(banks[(spec['scaling'],spec['cap'])],spec['concentration'],head)
                    row=dict(id=cid,index=len(rows),family='kahm',prefix=prefix,geometry=spec,
                        kappa=kappa,alpha=alpha,validation_mse=float(np.mean((pred-val['Y'])**2)),
                        gap=head['gap'],norm_sum=head['norm_sum'],converged=head['converged'],
                        fit_and_validation_seconds=time.perf_counter()-before,
                        numeric_bytes=cost['numeric_bytes'],cost=cost)
                    rows.append(row);heads[cid]=head;preds.append(pred)
    if len(rows)!=648:
        raise ArithmeticError('KAHM head roster mismatch')
    chosen={str(p):min((r for r in rows if r['prefix']==p),key=lambda r:(r['validation_mse'],r['index'])) for p in PREFIXES}
    selected_models={str(p):dict(head=heads[chosen[str(p)]['id']],
        bank=G.compact(banks[(chosen[str(p)]['geometry']['scaling'],chosen[str(p)]['geometry']['cap'])],len(feature_blocks(p))),
        concentration=chosen[str(p)]['geometry']['concentration'],candidate_id=chosen[str(p)]['id'],
        geometry=chosen[str(p)]['geometry']) for p in PREFIXES}
    dump(out/'candidate_heads.pkl',heads);dumpj(out/'kahm_candidates.json',rows)
    npz(out/'kahm_validation_predictions.npz',predictions=np.column_stack(preds),
        candidate_ids=np.asarray([r['id'] for r in rows]))
    dump(out/'selected_models.pkl',selected_models)
    before=time.perf_counter()
    baseline=B.fit_candidates(support['X'],train['X'],train['Y'],val['X'],val['Y'])
    baseline_seconds=time.perf_counter()-before
    dump(out/'baseline_candidates.pkl',baseline['models'])
    baseline_selected={slot:dict(candidate_id=record['id'],model=baseline['models'][record['id']])
                       for slot,record in baseline['selected'].items()}
    dump(out/'baseline_selected.pkl',baseline_selected)
    dumpj(out/'baseline_candidates.json',baseline['records'])
    dumpj(out/'baseline_cache_events.json',baseline['cache_events'])
    npz(out/'baseline_validation_predictions.npz',predictions=baseline['validation_predictions'],
        candidate_ids=np.asarray([r['id'] for r in baseline['records']]))
    selected=dict(kahm=chosen,baseline=baseline['selected'],reference=baseline['reference'],
        validation_only_selection=True,no_refit=True,calibration_unread=True,terminal_unread=True,
        selection_rule=menu()['selection'],reference_rule=menu()['reference_selection'])
    dumpj(out/'selected.json',selected)
    costs=dict(kahm={p:r['cost'] for p,r in chosen.items()},
        baseline={s:r['cost'] for s,r in baseline['selected'].items()})
    dumpj(out/'costs.json',costs)
    all_rows=rows+baseline['records']
    dumpj(out/'validation_frontier.json',dict(all_candidates=all_rows,
        nondominated=B.frontier(all_rows),scope='validation-only candidate frontier; terminal evaluates frozen winners only'))
    dumpj(out/'events.json',events)
    dumpj(out/'summary.json',dict(started=started,finished=utc(),seconds=time.perf_counter()-start,
        geometry_fit_seconds=geometry_seconds,shared_encoding_seconds=encoding_seconds,
        baseline_fit_seconds=baseline_seconds,head_candidates=len(rows),baseline_candidates=len(baseline['records']),
        nonconverged_heads=sum(not r['converged'] for r in rows),max_head_gap=max(r['gap'] for r in rows),
        learned_banks=6,regrouped_banks_fitted_and_charged=6,
        selected_kahm={p:r['id'] for p,r in chosen.items()},reference=baseline['reference']['id'],
        support_targets_read=False,all_feasible_finite_candidates_retained=True))
    seal('fit')


def encode_selected(models,x,tag):
    result,events={},[]
    groups={}
    for p,model in models.items():
        key=(model['geometry']['scaling'],model['geometry']['cap'])
        groups.setdefault(key,[]).append(p)
    for key,prefixes in groups.items():
        pmax=max(prefixes,key=int)
        bank=models[pmax]['bank']
        concentrations=tuple(sorted({models[p]['concentration'] for p in prefixes}))
        encoded,local=encode_bank(bank,x,concentrations,prefix=int(pmax),tag=tag)
        events.extend(dict(scaling=key[0],cap=key[1],**e) for e in local)
        for p in prefixes:
            e=encoded[models[p]['concentration']];q=int(p);nb=len(feature_blocks(q))
            result[p]=dict(F=e['F'][:,:q],F_lo=e['F_lo'][:,:q],F_hi=e['F_hi'][:,:q],
                           block_radius=e['block_radius'][:,:nb])
    return result,events


def evaluate_models(models,baselines,data,tag):
    features,events=encode_selected(models,data['X'],tag)
    arrays={};saved_features={}
    for p,m in models.items():
        f=features[p]
        pred=predict_head(m['head'],f['F'])
        lo,hi=prediction_interval(m['head'],f['F_lo'],f['F_hi'])
        arrays.update({f'pred_{p}':pred,f'prediction_lo_{p}':lo,f'prediction_hi_{p}':hi,
            f'loss_nominal_{p}':(data['Y']-pred)**2,
            f'loss_lower_{p}':loss_lower(data['Y'],lo,hi),
            f'loss_upper_{p}':loss_upper(data['Y'],lo,hi)})
        saved_features.update({f'{name}_{p}':value for name,value in f.items()})
    slots=sorted(baselines)
    bp=np.column_stack([B.predict(baselines[s]['model'],data['X']) for s in slots])
    arrays.update(baseline_predictions=bp,baseline_losses=(data['Y'][:,None]-bp)**2,
                  baseline_loss_lower=loss_lower(data['Y'][:,None],bp,bp),
                  baseline_loss_upper=loss_upper(data['Y'][:,None],bp,bp),
                  baseline_slots=np.asarray(slots),baseline_candidate_ids=np.asarray([baselines[s]['candidate_id'] for s in slots]))
    return features,saved_features,arrays,events


def stage_receipt(stage,data):
    return dict(stage=stage,time=utc(),rows=len(data['X']),freeze_sha256=sha(RUN/'freeze.json'),
        fit_seal_sha256=sha(RUN/'fit_seal.json'),precalibration_fit_gate_sha256=sha(RUN/'fit_validation.json'),
        selected_models_sha256=sha(RUN/'fit/selected_models.pkl'),
        baseline_selected_sha256=sha(RUN/'fit/baseline_selected.pkl'),
        data_receipt=data.get('receipt'),patient_rows_unique=True)


def calibrate():
    require_fit_gate()
    if (RUN/'terminal').exists():
        raise RuntimeError('Calibration precedes terminal access')
    out=RUN/'cal';out.mkdir();start=time.perf_counter()
    data=stage_data('calibration')
    npz(out/'observations.npz',**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    dumpj(out/'data_receipt.json',stage_receipt('cal',data))
    models=load(RUN/'fit/selected_models.pkl');baselines=load(RUN/'fit/baseline_selected.pkl')
    features,stored,arrays,events=evaluate_models(models,baselines,data,'calibration')
    npz(out/'features.npz',**stored);npz(out/'predictions.npz',**arrays)
    q=certificate_module();evidence={};records={};timings={}
    for p in map(str,PREFIXES):
        before=time.perf_counter()
        answer=q.evaluate_calibration(features[p]['F'],data['Y'],arrays[f'pred_{p}'],
            sample_sizes=SAMPLE_SIZES,block_size=BLOCK_SIZE,block_slices=feature_blocks(int(p)),
            kappa=models[p]['head']['kappa'],target_bounds=BOUNDS,alpha=ALPHA,
            population_size=POPULATION_SIZE,feature_radius=features[p]['block_radius'],
            solver_config=SOLVER_CONFIG)
        timings[p]=time.perf_counter()-before;evidence[p]=answer;records[p]=answer['records']
    dump(out/'certificate_evidence.pkl',evidence)
    dumpj(out/'calibration.json',dict(by_prefix=records,alpha=ALPHA,delta=.05,
        population_size=POPULATION_SIZE,reference_tolerance_uncomputed_until_census=True,
        lower_endpoint_scope='selected canonical KAHM decoder class on fixed8192patient evaluation population',
        no_sample_upper=True))
    dumpj(out/'events.json',events)
    dumpj(out/'summary.json',dict(seconds=time.perf_counter()-start,rows=len(data['X']),
        certificate_seconds=timings,selected_classes=3,class_sample_records=9,
        models_unchanged_sha256=sha(RUN/'fit/selected_models.pkl'),terminal_unread=True))
    seal('cal')


def deployment_timing(models,baselines,x,expected):
    rows={}
    for p,m in models.items():
        packed=pickle.dumps(m,protocol=5)
        before=time.perf_counter();local=pickle.loads(packed);deserialize=time.perf_counter()-before
        before=time.perf_counter()
        adapter=C.bank_adapter(local['bank'],prefix=int(p),precision=PRECISION,
            score_atol=SCORE_ATOL,F_atol=FEATURE_ATOL)
        prepared=time.perf_counter()-before
        try:
            before=time.perf_counter()
            encoded,_=encode_bank(local['bank'],x,(local['concentration'],),prefix=int(p),adapter=adapter)
            pred=predict_head(local['head'],encoded[local['concentration']]['F'])
            first=time.perf_counter()-before;warm=[]
            for _ in range(TIMING_REPEATS):
                before=time.perf_counter()
                encoded,_=encode_bank(local['bank'],x,(local['concentration'],),prefix=int(p),adapter=adapter)
                replay=predict_head(local['head'],encoded[local['concentration']]['F'])
                warm.append(time.perf_counter()-before)
                if np.max(abs(replay-pred))>1e-11:
                    raise ArithmeticError('Repeated KAHM timing changed predictions')
        finally:
            adapter.close()
        error=float(np.max(abs(pred-expected[f'pred_{p}'][:len(x)])))
        if error>1e-9:
            raise ArithmeticError('Isolated KAHM timing changed frozen predictions')
        rows['kahm_'+p]=dict(rows=len(x),deserialize_seconds=deserialize,
            preparation_seconds=prepared,cold_query_seconds=prepared+first,
            warm_query_seconds=warm,warm_median_seconds=float(np.median(warm)),
            max_replay_error=error,assurance='MPFI/MPFR interval forward features; floating head',
            process_and_runtime_already_initialized=True,cold_scope='new model adapter/cache, not fresh process or native library compilation')
    slots=expected['baseline_slots'].tolist()
    for slot,m in baselines.items():
        packed=pickle.dumps(m['model'],protocol=5)
        before=time.perf_counter();model=pickle.loads(packed);deserialize=time.perf_counter()-before
        before=time.perf_counter();p=B.predict(model,x);cold=time.perf_counter()-before
        warm=[]
        for _ in range(TIMING_REPEATS):
            before=time.perf_counter();replay=B.predict(model,x);warm.append(time.perf_counter()-before)
            if np.max(abs(replay-p))>1e-11:
                raise ArithmeticError('Repeated baseline timing changed predictions')
        error=float(np.max(abs(p-expected['baseline_predictions'][:len(x),slots.index(slot)])))
        if error>1e-10:
            raise ArithmeticError('Isolated baseline timing changed predictions')
        rows[slot]=dict(rows=len(x),deserialize_seconds=deserialize,cold_query_seconds=cold,
            warm_query_seconds=warm,warm_median_seconds=float(np.median(warm)),
            max_replay_error=error,assurance='ordinary float64 exported predictor',
            process_and_runtime_already_initialized=True,cold_scope='new deserialized predictor in warm process')
    return rows


def terminal():
    require_cal_gate()
    out=RUN/'terminal';out.mkdir();start=time.perf_counter()
    data=stage_data('terminal')
    npz(out/'observations.npz',**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    receipt=stage_receipt('terminal',data)
    receipt.update(cal_seal_sha256=sha(RUN/'cal_seal.json'),preterminal_cal_gate_sha256=sha(RUN/'cal_validation.json'))
    dumpj(out/'data_receipt.json',receipt)
    models=load(RUN/'fit/selected_models.pkl');baselines=load(RUN/'fit/baseline_selected.pkl')
    _,stored,arrays,events=evaluate_models(models,baselines,data,'terminal')
    npz(out/'features.npz',**stored);npz(out/'predictions.npz',**arrays)
    dumpj(out/'events.json',events)
    timing=deployment_timing(models,baselines,data['X'][:TIMING_ROWS],arrays)
    dumpj(out/'deployment_timing.json',timing)
    dumpj(out/'summary.json',dict(rows=len(data['X']),seconds=time.perf_counter()-start,
        kahm_mse={p:float(np.mean(arrays[f'loss_nominal_{p}'])) for p in map(str,PREFIXES)},
        baseline_mse={s:float(np.mean(arrays['baseline_losses'][:,i])) for i,s in enumerate(arrays['baseline_slots'].tolist())},
        scope='descriptive untouched terminal cohort; no new model selection or inference'))
    seal('terminal')


def census():
    require_cal_gate();check_seal('terminal')
    out=RUN/'census';out.mkdir()
    c,t=(load_npz(RUN/s/'predictions.npz') for s in ('cal','terminal'))
    selected=read(RUN/'fit/selected.json');costs=read(RUN/'fit/costs.json')
    if c['baseline_slots'].tolist()!=t['baseline_slots'].tolist():
        raise ArithmeticError('Different frozen baseline rosters')
    rows=[]
    for p in map(str,PREFIXES):
        nominal=np.concatenate((c[f'loss_nominal_{p}'],t[f'loss_nominal_{p}']))
        lower=np.concatenate((c[f'loss_lower_{p}'],t[f'loss_lower_{p}']))
        upper=np.concatenate((c[f'loss_upper_{p}'],t[f'loss_upper_{p}']))
        mse=float(np.mean(nominal));up=mean_upper(upper)
        rows.append(dict(id='kahm_'+p,candidate_id=selected['kahm'][p]['id'],family='kahm',prefix=int(p),
            mse=mse,mse_lower=mean_bound(lower,'lower'),mse_upper=up,mse_upper_minus_nominal=up-mse,
            numeric_bytes=costs['kahm'][p]['numeric_bytes'],cost=costs['kahm'][p],
            evaluation_rows=len(nominal),scope='canonical-feature head with outward arithmetic loss upper'))
    slots=c['baseline_slots'].tolist()
    for i,slot in enumerate(slots):
        loss=np.concatenate((c['baseline_losses'][:,i],t['baseline_losses'][:,i]))
        lower=np.concatenate((c['baseline_loss_lower'][:,i],t['baseline_loss_lower'][:,i]))
        upper=np.concatenate((c['baseline_loss_upper'][:,i],t['baseline_loss_upper'][:,i]))
        mse=float(np.mean(loss))
        rows.append(dict(id=slot,candidate_id=selected['baseline'][slot]['id'],family=selected['baseline'][slot]['family'],
            mse=mse,mse_lower=mean_bound(lower,'lower'),mse_upper=mean_upper(upper),
            numeric_bytes=costs['baseline'][slot]['numeric_bytes'],
            cost=costs['baseline'][slot],evaluation_rows=len(loss),scope='fixed floating-point predictor census'))
    ref=next(r for r in rows if r['candidate_id']==selected['reference']['id'])
    threshold=1.10*ref['mse']
    threshold_lower=threshold_bound(ref['mse_lower'],'lower')
    threshold_upper=threshold_bound(ref['mse_upper'],'upper')
    for r in rows:
        r['mse_ratio_to_reference']=r['mse']/ref['mse'] if ref['mse']>0 else None
        r['numeric_payload_smaller_than_reference']=r['numeric_bytes']<ref['numeric_bytes']
        r['within_ten_percent_reference_mse']=r['mse_upper']<=threshold_lower
        r['meets_joint_criterion']=r['numeric_payload_smaller_than_reference'] and r['within_ten_percent_reference_mse']
        r['nominal_joint_criterion']=r['numeric_payload_smaller_than_reference'] and r['mse']<=threshold
        r['rounding_changed_joint_decision']=r['meets_joint_criterion']!=r['nominal_joint_criterion']
    calibration=read(RUN/'cal/calibration.json')['by_prefix']
    exclusions={}
    for p,records in calibration.items():
        exclusions[p]={str(record['n']):dict(class_lower=record['lower'],
            insufficient=record['lower']>threshold_upper,
            nominal_insufficient=record['lower']>threshold,
            rounding_changed_decision=(record['lower']>threshold_upper)!=(record['lower']>threshold))
            for record in records}
    answer=dict(schema='step18-census-v1',population_size=POPULATION_SIZE,
        reference=ref,threshold=threshold,threshold_lower=threshold_lower,
        threshold_upper=threshold_upper,threshold_multiplier=1.10,rows=rows,
        conservative_frontier=B.frontier(rows,risk_key='mse_upper'),
        nominal_frontier=B.frontier(rows,risk_key='mse'),
        selected_kahm_joint_success=any(r['meets_joint_criterion'] for r in rows if r['family']=='kahm'),
        calibration_lower_endpoints=calibration,class_exclusions=exclusions,
        numerical_guard_changed_any_joint_decision=any(r['rounding_changed_joint_decision'] for r in rows),
        census_arithmetic=menu()['census_arithmetic'],
        models_selected_before_all8192evaluation_targets=True,
        all_evaluation_rows_counted_once=True,
        no_claim_of_independent_hospital_or_deployment_population=True,
        scope='finite fixed8192patient evaluation pool; exact census of frozen floating predictors and conservative canonical KAHM loss enclosure')
    dumpj(out/'results.json',answer)
    seal('census')


def preflight():
    """Manufactured arrays only: all native caps/scalers and extreme concentration."""
    TESTS.mkdir(parents=True,exist_ok=True)
    if (TESTS/'runner_preflight18.json').exists():
        raise FileExistsError('Preflight is write-once')
    rng=np.random.default_rng(MANUFACTURED_SEED)
    u=rng.normal(size=(512,32))
    x=rng.normal(size=(48,32));x[-8:]*=5.
    y=np.clip(.35*x[:,0]-.2*x[:,1]+.05*np.sin(x[:,2]),-1,1)
    G.initialize_runtime()
    family=G.fit_family(u,GEOMETRY_CONFIG,GEOMETRY_SEED)
    events=[];heads=[];stored={};maximum_head_interval_violation=0.
    for scaling in SCALINGS:
        for cap in CAPS:
            bank=family['learned'][(scaling,cap)]
            encoded,local=encode_bank(bank,x,CONCENTRATIONS,tag='manufactured_only')
            events.extend(dict(scaling=scaling,cap=cap,**e) for e in local)
            for omega in CONCENTRATIONS:
                f=encoded[omega]
                for prefix in PREFIXES:
                    head=fit_head(f['F'][:32,:prefix],y[:32],prefix,1e-4,2.)
                    pred=predict_head(head,f['F'][:,:prefix])
                    lo,hi=prediction_interval(head,f['F_lo'][:,:prefix],f['F_hi'][:,:prefix])
                    # The floating nominal head can differ from its real-value
                    # target by a rounding ulp; interval evaluation includes it.
                    violation=max(float(np.max(lo-pred)),float(np.max(pred-hi)),0.)
                    maximum_head_interval_violation=max(maximum_head_interval_violation,violation)
                    if violation>1e-14 or head['norm_sum']>2.:
                        raise ArithmeticError('Manufactured head interval/feasibility failure')
                    heads.append(dict(scaling=scaling,cap=cap,omega=omega,prefix=prefix,
                        gap=head['gap'],norm_sum=head['norm_sum'],max_interval_width=float(np.max(hi-lo))))
                if omega==64.:
                    key=f'{scaling}_{cap}'
                    stored.update({key+'_'+k:v for k,v in f.items()})
    baseline_specs=B.grid()
    baseline=B.fit_candidates(u,x[:32],y[:32],x[32:],y[32:],specs=baseline_specs)
    for model in baseline['models'].values():
        if not np.array_equal(B.predict(model,x),B.predict(pickle.loads(pickle.dumps(model)),x)):
            raise ArithmeticError('Export serialization replay changed baseline predictions')
    constant_fixture=np.full((192,32),.37)
    for mode in SCALINGS:
        sc=B.scaler_fit(constant_fixture,mode)
        if not np.all(sc['constant_mask']) or np.any(B.scale(np.full((2,32),2.1),sc)!=0.):
            raise ArithmeticError('Constant baseline support coordinate did not stay zero')
    from fractions import Fraction
    manufactured_y=np.array([-1.,-.3,0.,.125,.9,1.])
    manufactured_p=np.array([1.,-.30000000000000004,.1,.125,.7,-1.])
    exact=sum((Fraction(float(a))-Fraction(float(b)))**2 for a,b in zip(manufactured_y,manufactured_p))/len(manufactured_y)
    loss_lo=loss_lower(manufactured_y,manufactured_p,manufactured_p)
    loss_hi=loss_upper(manufactured_y,manufactured_p,manufactured_p)
    mean_lo,mean_hi=mean_bound(loss_lo,'lower'),mean_upper(loss_hi)
    if not Fraction(mean_lo)<=exact<=Fraction(mean_hi):
        raise ArithmeticError('Manufactured exact rational census mean escaped outward interval')
    threshold_exact=exact*Fraction(11,10)
    if not Fraction(threshold_bound(mean_lo,'lower'))<=threshold_exact<=Fraction(threshold_bound(mean_hi,'upper')):
        raise ArithmeticError('Manufactured rational tolerance escaped outward interval')
    arithmetic_y=rng.uniform(-1,1,8192);arithmetic_p=rng.uniform(-1,1,8192)
    long_exact=sum((Fraction(float(a))-Fraction(float(b)))**2 for a,b in zip(arithmetic_y,arithmetic_p))/8192
    long_lower=mean_bound(loss_lower(arithmetic_y,arithmetic_p,arithmetic_p),'lower')
    long_upper=mean_upper(loss_upper(arithmetic_y,arithmetic_p,arithmetic_p))
    if not Fraction(long_lower)<=long_exact<=Fraction(long_upper):
        raise ArithmeticError('8192-row manufactured rational mean escaped enclosure')
    manufactured_path=TESTS/'runner_manufactured18.npz'
    npz(manufactured_path,support_X=u,query_X=x,query_Y=y,**stored)
    dumpj(HERE/'model_menu18.json',menu())
    outside=np.any((x<np.min(u,axis=0))|(x>np.max(u,axis=0)),axis=1)
    result=dict(schema='step18-runner-preflight-v2',passed=True,manufactured_only=True,seed=MANUFACTURED_SEED,
        support_shape=list(u.shape),query_shape=list(x.shape),
        outside_support_minmax_rows=int(outside.sum()),caps=list(CAPS),scalings=list(SCALINGS),
        concentrations=list(CONCENTRATIONS),heads_checked=len(heads),
        baseline_families_checked=sorted({s['family'] for s in baseline_specs}),
        baseline_candidates_checked=len(baseline_specs),
        baseline_constant_axis_check=True,
        max_head_interval_violation=maximum_head_interval_violation,
        events=events,head_diagnostics=heads,
        baseline_diagnostics=baseline['records'],
        census_arithmetic_exact_rational_check=True,
        census_arithmetic_example=dict(lower=mean_lo,upper=mean_hi,exact_float=float(exact)),
        census8192_exact_rational_check=dict(lower=long_lower,upper=long_upper,exact_float=float(long_exact)),
        manufactured_arrays_sha256=sha(manufactured_path),
        producer_sha256={str(Path(p).relative_to(BASE)):sha(p) for p in (Path(__file__),Path(B.__file__))},
        superseded_v1_receipt_sha256=sha(TESTS/'runner_preflight18_v1.json'),
        no_natural_waveforms_or_targets_read=True)
    dumpj(TESTS/'runner_preflight18.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','head_diagnostics','baseline_diagnostics')},default=json_default),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('preflight','freeze','fit','cal','terminal','census'))
    args=parser.parse_args()
    {'preflight':preflight,'freeze':freeze,'fit':fit,'cal':calibrate,
     'terminal':terminal,'census':census}[args.stage]()


if __name__=='__main__':
    main()

