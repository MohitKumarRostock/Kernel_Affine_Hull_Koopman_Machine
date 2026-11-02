#!/usr/bin/env python3
"""Independently verify the fixed speaker-separated speech forecasting study.

Reconstruct the bounded energy observable from the provider archive, check
identifier splits and frozen validation selections, recompute all selected
predictors' per-identifier losses, and invert the 72 KL bounds independently.
An optional deterministic replay is compared by fitted values and losses,
allowing the different CSV ordering implied by its stricter scoring schedule.
The study runner is never imported. The source root supplies only the fitted
feature/model classes needed to decode the archived predictors.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import pickle
import sys
import wave
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import brentq

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(actual, expected):
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-13), (actual, expected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Official mini_speech_commands.zip provider archive.')
    parser.add_argument('--results', type=Path, required=True, help='Original speech study directory with its frozen source and models.')
    parser.add_argument('--replay-results', type=Path, help='Optional strict-order replay directory to compare numerically.')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2], help='Extension root containing src/certified_forecasting.')
    parser.add_argument('--protocol-source', type=Path, help='Frozen original protocol (defaults to RESULTS/source/protocol.md).')
    parser.add_argument('--output', type=Path, required=True, help='Destination JSON verification report.')
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    BASE = args.results.resolve()
    OUT = args.output.resolve()
    archive = args.archive.resolve()
    replay = args.replay_results.resolve() if args.replay_results is not None else None
    protocol_source = args.protocol_source.resolve() if args.protocol_source is not None else BASE / 'source/protocol.md'
    sys.path.insert(0, str(ROOT / 'src'))
    metadata = pd.read_csv(BASE / 'clips.csv', dtype={'speaker': str})
    speakers = pd.read_csv(BASE / 'speakers.csv', dtype={'speaker': str})
    losses = pd.read_csv(BASE / 'speaker_losses.csv.gz', dtype={'speaker': str})
    results = json.loads((BASE / 'results.json').read_text())
    provenance = json.loads((BASE / 'data_provenance.json').read_text())
    freeze = json.loads((BASE / 'development_freeze.json').read_text())
    completion = json.loads((BASE / 'completion.json').read_text())
    assert digest(BASE / 'development_freeze.json') == completion['development_freeze_sha256']
    assert digest(BASE / 'results.json') == completion['result_sha256']
    assert digest(BASE / 'speaker_losses.csv.gz') == completion['loss_sha256']
    assert digest(protocol_source) == freeze['protocol_sha256']
    assert metadata.path.nunique() == len(metadata) == 7178
    assert metadata.speaker.nunique() == len(speakers) == 1630
    assert metadata.groupby('speaker').split.nunique().max() == 1
    assert set(metadata.speaker) == set(speakers.speaker)
    assert np.array_equal(metadata.speaker.to_numpy(), metadata.path.str.split('/').str[-1].str.split('_nohash_').str[0].to_numpy())
    assert metadata.groupby('split').speaker.nunique().to_dict() == provenance['groups']
    assert metadata.groupby('split').size().to_dict() == provenance['clips']
    order = np.random.default_rng(8602026).permutation(sorted(speakers.speaker))
    n = len(order)
    counts = [int(0.15 * n), int(0.1 * n), int(0.6 * n)]
    counts += [n - sum(counts)]
    cursor = 0
    for split, count in zip(('train', 'validation', 'calibration', 'test'), counts):
        assert set(order[cursor:cursor + count]) == set(speakers.loc[speakers.split == split, 'speaker'])
        cursor += count
    assert digest(archive) == provenance['archive_sha256']
    energy = np.load(BASE / 'energy_observations.npz', allow_pickle=False)['energy']
    assert energy.shape == (7178, 100) and np.isfinite(energy).all() and (0 <= energy.min()) and (energy.max() <= 1)
    with zipfile.ZipFile(archive) as z:
        files = sorted((p for p in z.namelist() if p.startswith('mini_speech_commands/') and p.endswith('.wav')))
        assert len(files) == 8000 and len(provenance['excluded']) == 822
        excluded = {row['path'] for row in provenance['excluded']}
        assert set(metadata.path) | excluded == set(files) and (not set(metadata.path) & excluded)
        for index, path in enumerate(metadata.path):
            with wave.open(io.BytesIO(z.read(path))) as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 16000)
                pcm = np.frombuffer(w.readframes(16000), dtype='<i2').astype(float) / 32768
            expected = np.minimum(1, np.maximum(0, (10 * np.log10(np.maximum((pcm.reshape(100, 160) ** 2).mean(axis=1), 1e-06)) + 60) / 60))
            check(energy[index], expected)
    origins = np.arange(11, 88, 4)
    horizons = np.array([1, 4, 12])
    assert len(origins) == 20 and origins.min() - 11 == 0 and (origins.max() + 12 == 99)
    assert len(results['results']) == 24 and results['n_upper_statements'] == 72
    check(results['upper_tail_alpha'], 0.05 / 72)
    assert len(losses) == 3 * 8 * 3 * (978 + 245)
    assert not losses.duplicated(['seed', 'family', 'split', 'speaker', 'horizon']).any()
    assert losses.mse.min() >= 0 and losses.mse.max() <= 1

    class ForecastRecord:
        pass

    class SavedModelUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if module == '__main__' and name == 'Forecast':
                return ForecastRecord
            return super().find_class(module, name)

    def same_model_values(a, b, seen=None):
        seen = set() if seen is None else seen
        if (id(a), id(b)) in seen:
            return True
        seen.add((id(a), id(b)))
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all((same_model_values(x, y, seen) for x, y in zip(a, b)))
        if isinstance(a, dict):
            return a.keys() == b.keys() and all((same_model_values(a[k], b[k], seen) for k in a))
        if hasattr(a, '__dict__'):
            return same_model_values(vars(a), vars(b), seen)
        return bool(a == b)
    selected = {}
    for seed in (8601, 8602, 8603):
        development = json.loads((BASE / f'development_{seed}.json').read_text())
        assert digest(BASE / f'development_{seed}.json') == freeze['development'][str(seed)]
        assert digest(BASE / f'selected_models_{seed}.pkl') == freeze[f'{seed}_model_sha256']
        assert len(development['candidates']) == 486
        with (BASE / f'selected_models_{seed}.pkl').open('rb') as f:
            selected[seed] = SavedModelUnpickler(f).load()
        for row in development['selected']:
            candidates = [r for r in development['candidates'] if r['family'] == row['family']]
            winner = min(candidates, key=lambda r: (r['validation_mean'], r['features'], r['history'], r['candidate']))
            assert row == winner
        train = np.load(BASE / f'training_rows_{seed}.npz', allow_pickle=False)
        assert len(train['clip_index']) == 4096 and set(metadata.loc[train['clip_index'], 'split']) == {'train'}
        assert train['origin_index'].min() >= 0 and train['origin_index'].max() < 20
    recomputed = 0
    table = []
    for entry in results['results']:
        seed, family = (entry['seed'], entry['family'])
        _, forecast, frozen = selected[seed][family]
        assert entry['candidate'] == frozen['candidate']
        if family in ('kahm', 'rbf', 'affine'):
            model = forecast.model
            norm = sum((np.linalg.norm(model.coef_[sl]) for sl in model.block_slices_))
            check(norm, entry['coefficient_norm'])
            assert norm <= entry['parameters']['kappa'] + 1e-10
        for split in ('calibration', 'test'):
            ids = metadata.index[metadata.split == split].to_numpy()
            m = entry['history']
            history_idx = origins[:, None] - np.arange(m - 1, -1, -1)[None, :]
            x = energy[ids][:, history_idx].reshape(-1, m)
            target = energy[ids][:, origins[:, None] + horizons].reshape(-1, 3)
            if family == 'persistence':
                prediction = np.repeat(x[:, -1:], 3, axis=1)
            elif family == 'constant':
                prediction = np.tile(forecast.mean, (len(x), 1))
            else:
                if family in ('kahm', 'rbf', 'affine'):
                    features = forecast.feature.transform(x, n_blocks=forecast.stage)
                else:
                    features = forecast.scaler.transform(x)
                    if forecast.feature is not None:
                        features = forecast.feature.transform(features)
                prediction = np.clip(features @ forecast.model.coef_ + forecast.model.intercept_, 0, 1)
            clip_loss = np.mean((prediction - target).reshape(len(ids), len(origins), 3) ** 2, axis=1)
            grouped = pd.DataFrame(clip_loss, columns=[1, 4, 12])
            grouped['speaker'] = metadata.loc[ids, 'speaker'].to_numpy()
            group = grouped.groupby('speaker', sort=True).mean()
            sub = losses[(losses.seed == seed) & (losses.family == family) & (losses.split == split)]
            actual = sub.pivot(index='speaker', columns='horizon', values='mse').sort_index()
            assert set(actual.index) == set(speakers.loc[speakers.split == split, 'speaker'])
            check(actual.to_numpy(), group.to_numpy())
            check(group.mean(axis=0).to_numpy(), entry[split + '_mse'])
            assert len(group) == entry[split + '_speakers']
            recomputed += len(group) * 3
            if split == 'calibration':
                for h, mean, upper in zip((1, 4, 12), group.mean(axis=0), entry['mse_upper']):
                    alpha = 0.05 / 72
                    n = len(group)
                    constant = math.log(1 / alpha) / n
                    fn = lambda p: mean * math.log(mean / p) + (1 - mean) * math.log((1 - mean) / (1 - p)) - constant
                    expected = brentq(fn, max(mean, 1e-16), 1 - 1e-15, xtol=1e-14)
                    check(expected, upper)
        for tolerance in (0.1, 0.2, 0.3):
            assert entry['all_horizon_pass'][str(tolerance)] == (max(entry['mse_upper']) <= tolerance * tolerance)
        table.append({'seed': seed, 'family': family, 'test_rmse': np.sqrt(entry['test_mse']).tolist(), 'upper_rmse': np.sqrt(entry['mse_upper']).tolist()})
    replay_equal = None
    if replay is not None:
        replay_loss = pd.read_csv(replay / 'speaker_losses.csv.gz', dtype={'speaker': str})
        key = ['seed', 'family', 'split', 'speaker', 'horizon']
        assert losses.sort_values(key).reset_index(drop=True).equals(replay_loss.sort_values(key).reset_index(drop=True))
        other = json.loads((replay / 'results.json').read_text())
        remove_timing = lambda r: {k: v for k, v in r.items() if not k.endswith('_seconds')}
        assert [remove_timing(r) for r in results['results']] == [remove_timing(r) for r in other['results']]
        replay_freeze = json.loads((replay / 'development_freeze.json').read_text())
        for seed in (8601, 8602, 8603):
            assert digest(replay / f'selected_models_{seed}.pkl') == replay_freeze[f'{seed}_model_sha256']
            with (replay / f'selected_models_{seed}.pkl').open('rb') as f:
                replayed = SavedModelUnpickler(f).load()
            for family in selected[seed]:
                a, b = (selected[seed][family][1], replayed[family][1])
                assert a.history == b.history and a.family == b.family and (a.stage == b.stage)
                if a.model is not None:
                    assert np.array_equal(a.model.coef_, b.model.coef_) and np.array_equal(a.model.intercept_, b.model.intercept_)
                if a.mean is not None:
                    assert np.array_equal(a.mean, b.mean)
                if a.scaler is not None:
                    assert np.array_equal(a.scaler.mean_, b.scaler.mean_) and np.array_equal(a.scaler.scale_, b.scaler.scale_)
                assert same_model_values(a, b)
        replay_equal = True
    report = {'status': 'passed', 'raw_audio_clips_recomputed': 7178, 'excluded_clips': 822, 'unique_recording_identifiers': 1630, 'split_group_counts': provenance['groups'], 'selected_models_verified': 24, 'candidates_per_seed': 486, 'per_identifier_horizon_losses_recomputed': recomputed, 'KL_upper_endpoints_verified': 72, 'all_model_selections_match_frozen_validation_minima': True, 'all_geometric_norm_constraints_feasible': True, 'same_identifier_never_crosses_split': True, 'bounded_causal_frame_observable_verified': True, 'replay_losses_and_non_timing_results_identical': replay_equal, 'replay_all_saved_predictor_values_identical': replay_equal, 'original_chronology_deviation': 'Calibration/test evaluated sequentially within each already-frozen model, before all calibration calculations were complete. Model independence remains valid; replay freezes all calibration calculations before any test scoring.', 'scope': 'Released provider-trimmed exact-one-second clips; identifiers are recording-group/session proxies, not biometric identity proof. IID common-population recording bundles remain a modeling assumption.', 'selected_model_metrics': table}
    report['audit_source'] = {'path': 'experiments/speech_forecasting/verify_study.py', 'sha256': digest(Path(__file__))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'selected_model_metrics'}, indent=2))
if __name__ == '__main__':
    main()
