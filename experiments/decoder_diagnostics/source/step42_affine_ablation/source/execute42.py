"""Run the sealed producer/auditor sequence once, retaining every phase log."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import subprocess
import sys
import time
import traceback
import common42 as c


def commands():
    rows = [('preflight', ['source/audit42.py', 'preflight'])]
    for task in c.TASKS:
        for stage in ('f1', 'f2', 'f3'):
            rows.append((f'produce_{task}_{stage}', ['source/run42.py', 'stage', '--task', task, '--stage', stage]))
            rows.append((f'audit_{task}_{stage}', ['source/audit42.py', 'stage', '--task', task, '--stage', stage]))
        rows.append((f'select_{task}', ['source/run42.py', 'selection', '--task', task]))
        rows.append((f'audit_selection_{task}', ['source/audit42.py', 'selection', '--task', task]))
        rows.append((f'produce_{task}_final', ['source/run42.py', 'stage', '--task', task, '--stage', 'final']))
        rows.append((f'audit_{task}_final', ['source/audit42.py', 'stage', '--task', task, '--stage', 'final']))
        rows.append((f'bridge_{task}', ['source/run42.py', 'bridge', '--task', task]))
        rows.append((f'audit_bridge_{task}', ['source/audit42.py', 'bridge', '--task', task]))
    rows.append(('aggregate', ['source/aggregate42.py']))
    rows.append(('audit_aggregate', ['source/audit42.py', 'aggregate']))
    assert len(rows) == 27 and len({label for label, _ in rows}) == 27
    return rows


def main():
    c.assert_frozen()
    destination = c.ROOT / 'output'
    started = time.perf_counter()
    specs = commands()
    protocol = c.read(c.ROOT / 'protocol42.json')
    c.save(destination / 'execution_started42.json',
           {'schema': 'step42-controller-start-v1', 'utc': c.utc(), 'producer_passes': 1,
            'protocol_seal_sha256': c.sha(c.ROOT / 'protocol_seal42.json'),
            'command_count': len(specs), 'commands': [{'label': label, 'arguments': args} for label, args in specs],
            'verification_solve_replays_are_not_new_candidate_searches': True})
    logdir = destination / 'logs'; logdir.mkdir(parents=True, exist_ok=True)
    events = destination / 'execution_events42.jsonl'
    with events.open('xb'):
        pass
    previous = None; event_index = 0; completed = []

    def event(kind, **fields):
        nonlocal previous, event_index
        value = {'index': event_index, 'utc': c.utc(), 'kind': kind,
                 'previous_event_sha256': previous, **fields}
        raw = c.canonical_bytes(value)
        with events.open('ab') as stream:
            stream.write(raw)
        previous = c.digest(value); event_index += 1

    event('execution_started', protocol_seal_sha256=c.sha(c.ROOT / 'protocol_seal42.json'))
    try:
        for index, (label, arguments) in enumerate(specs):
            out = logdir / f'{index:02d}_{label}.stdout.txt'
            err = logdir / f'{index:02d}_{label}.stderr.txt'
            event('command_started', command_index=index, label=label, arguments=arguments)
            print(c.canonical_bytes({'phase': 'START', 'command': index + 1, 'total': len(specs), 'label': label}).decode(), end='', flush=True)
            t0 = time.perf_counter()
            with out.open('xb') as stdout, err.open('xb') as stderr:
                timeout_seconds = protocol['execution']['aggregate_budget_seconds'] if label == 'aggregate' else 2400
                result = subprocess.run([sys.executable, *arguments], cwd=c.ROOT,
                                        env=os.environ.copy(), stdout=stdout, stderr=stderr,
                                        timeout=timeout_seconds, check=False)
            record = {'command_index': index, 'label': label, 'arguments': arguments,
                      'returncode': result.returncode, 'wall_seconds': time.perf_counter() - t0,
                      'stdout': {'path': c.relative(out), 'sha256': c.sha(out)},
                      'stderr': {'path': c.relative(err), 'sha256': c.sha(err)}}
            completed.append(record); event('command_completed', **record)
            print(c.canonical_bytes({'phase': 'END', 'command': index + 1, 'label': label,
                                     'returncode': result.returncode, 'wall_seconds': record['wall_seconds']}).decode(), end='', flush=True)
            if result.returncode:
                raise RuntimeError('Frozen command failed: ' + label)
        event('execution_completed', commands=len(completed))
        receipt = {'schema': 'step42-controller-complete-v1', 'status': 'PASS',
                   'utc': c.utc(), 'wall_seconds': time.perf_counter() - started,
                   'command_count': len(completed), 'commands': completed,
                   'producer_grid_runs': 1, 'outcome_based_reruns': 0,
                   'event_count': event_index, 'event_log_sha256': c.sha(events),
                   'protocol_seal_sha256': c.sha(c.ROOT / 'protocol_seal42.json'),
                   'scientific_audit_sha256': c.sha(c.ROOT / 'audit/postrun_audit42.json'),
                   'result_sha256': c.sha(c.ROOT / 'output/result42.json')}
        c.save(destination / 'execution_completed42.json', receipt)
        print(c.canonical_bytes({'status': 'PASS', 'commands': len(completed),
                                 'wall_seconds': receipt['wall_seconds'], 'producer_grid_runs': 1}).decode(), end='', flush=True)
    except Exception as exc:
        event('execution_failed', error_type=type(exc).__name__, error=str(exc))
        c.save(destination / 'execution_failed42.json',
               {'status': 'FAILED_RETAINED', 'utc': c.utc(), 'error_type': type(exc).__name__,
                'error': str(exc), 'commands': completed, 'event_log_sha256': c.sha(events),
                'traceback': traceback.format_exc(), 'automatic_rerun': False})
        raise


if __name__ == '__main__':
    main()
