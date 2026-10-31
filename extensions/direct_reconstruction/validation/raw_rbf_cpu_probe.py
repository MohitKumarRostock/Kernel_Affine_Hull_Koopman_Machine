"""One-state CPU-dispatch replay diagnostic. No fitting or source mutation."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import pickle
import sys
import zipfile

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info

repo, prefix = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from extensions.direct_reconstruction.model import predict_serialized

root = repo/'extensions/direct_reconstruction/results'
task = root/'tasks/wikipedia_daily__03'
with zipfile.ZipFile(task/'selected_models.zip') as archive:
    blob = archive.read('raw_rbf.pkl')
state = pickle.loads(blob)
raw, head = state['model'], state['model']['head']
assert state['kind']=='safe_raw' and raw['kind']=='raw_rbf'
arrays = dict(np.load(root/'prepared_inputs/wikipedia_daily__03.npz', allow_pickle=False))
saved = dict(np.load(task/'predictions.npz', allow_pickle=False))
trace, metrics = {}, {}
for split in ('C','D'):
    x, y = arrays[split+'_X'], arrays[split+'_Y']
    reference = saved[split+'__raw_rbf']
    prediction = predict_serialized(blob,x)
    z = (x-raw['mean'])/raw['scale']
    distance = cdist(z,raw['landmarks'],metric='sqeuclidean')
    kernel = np.exp(-distance/(2*raw['bandwidth_squared']))
    projected = kernel@raw['transform']
    features = np.column_stack((z,projected))
    standardized = (features-head['mean'])/head['scale']
    scalar_readout = np.column_stack([head['ymean'][h]+standardized@head['coef'][:,h] for h in range(3)])
    assert np.array_equal(prediction,scalar_readout)
    delta = prediction-reference
    absolute = np.abs(delta)
    tolerance = 1e-10+1e-10*np.abs(reference)
    worst = np.unravel_index(np.argmax(absolute/tolerance),absolute.shape)
    reference_loss = float(np.mean((reference-y)**2))
    replay_loss = float(np.mean((prediction-y)**2))
    nonzero = reference != 0
    metrics[split] = {
        'shape':list(reference.shape),'exact_match':bool(np.array_equal(prediction,reference)),
        'strict_replay_pass':bool(np.allclose(prediction,reference,rtol=1e-10,atol=1e-10)),
        'violating_entries':int(np.count_nonzero(absolute>tolerance)),
        'max_absolute_prediction_difference':float(np.max(absolute)),
        'max_relative_prediction_difference_nonzero_reference':float(np.max(absolute[nonzero]/np.abs(reference[nonzero]))),
        'max_difference_to_audit_tolerance_ratio':float(np.max(absolute/tolerance)),
        'worst_tolerance_location':{'row':int(worst[0]),'horizon_index':int(worst[1]),
            'saved':float(reference[worst]),'replayed':float(prediction[worst]),
            'difference':float(delta[worst]),'allowed_tolerance':float(tolerance[worst])},
        'saved_mse':reference_loss,'replayed_mse':replay_loss,
        'mse_difference':replay_loss-reference_loss,
        'relative_mse_difference':(replay_loss-reference_loss)/reference_loss,
        'scaled_mse_difference':(replay_loss-reference_loss)/float(arrays['loss_scale']),
        'saved_mse_by_horizon':np.mean((reference-y)**2,axis=0).tolist(),
        'replayed_mse_by_horizon':np.mean((prediction-y)**2,axis=0).tolist(),
    }
    for name,value in [('z',z),('distance',distance),('kernel',kernel),('projected',projected),
                       ('standardized',standardized),('prediction',prediction)]:
        trace[split+'_'+name]=value

singular=np.linalg.svd(raw['transform'],compute_uv=False)
runtime_stream=io.StringIO()
with contextlib.redirect_stdout(runtime_stream):
    np.show_runtime()
report={
    'task':'wikipedia_daily__03','method':'raw_rbf','selected_state_sha256':hashlib.sha256(blob).hexdigest(),
    'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'environment':{name:os.environ.get(name) for name in ['NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE',
        'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
    'numpy_version':np.__version__,'threadpools':threadpool_info(),'numpy_runtime':runtime_stream.getvalue(),
    'state':{'bandwidth_squared':float(raw['bandwidth_squared']),'ridge_alpha':float(head['alpha']),
        'transform_shape':list(raw['transform'].shape),
        'transform_operator_norm':float(singular.max()),'transform_condition_number':float(singular.max()/singular.min()),
        'feature_scale_min':float(head['scale'].min()),
        'coefficient_max_abs':float(np.abs(head['coef']).max()),
        'coefficient_l1_per_horizon':np.abs(head['coef']).sum(axis=0).tolist(),
        'A_loss_scale':float(arrays['loss_scale'])},
    'splits':metrics,'scope':'Only selected-state replay and arithmetic tracing on saved C/D inputs; no fits or generator calls',
}
prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
np.savez_compressed(prefix.with_suffix('.npz'),**trace)
print(json.dumps({'configuration':prefix.name,'splits':metrics},indent=2))
