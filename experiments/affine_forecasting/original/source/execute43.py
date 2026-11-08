"""One fixed producer execution, with independent gates and retained command logs."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'

from pathlib import Path
import subprocess
import sys
import time
import traceback

import common43 as c


def command_plan():
    commands = [('preflight', ['source/audit43.py', 'preflight'])]
    for task in c.TASKS:
        for stage in c.STAGES[:-1]:
            commands.append((f'development_{task}_{stage}',
                             ['source/run43.py', 'stage', '--task', task, '--stage', stage]))
    commands.append(('selection', ['source/run43.py', 'select']))
    for task in c.TASKS:
        commands.append((f'final_fit_{task}',
                         ['source/run43.py', 'stage', '--task', task, '--stage', 'final']))
    commands += [('selection_seal', ['source/run43.py', 'seal-selection']),
                 ('preassessment_audit', ['source/audit43.py', 'preassessment'])]
    for task in c.TASKS:
        commands.append((f'assessment_{task}', ['source/run43.py', 'assess', '--task', task]))
    for task in c.TASKS:
        commands.append((f'misbinding_{task}', ['source/run43.py', 'misbinding', '--task', task]))
    for task in c.TASKS:
        commands.append((f'cost_{task}', ['source/cost43.py', '--task', task]))
    commands += [('aggregate', ['source/run43.py', 'aggregate']),
                 ('final_audit', ['source/audit43.py', 'final'])]
    if len(commands) != 27:
        raise RuntimeError('Declared command plan changed')
    return commands


def main():
    root = c.ROOT
    protocol_hash = c.protocol_identity(deep=True)
    protocol = c.read_json(root / 'protocol43.json')
    commands = command_plan()
    started_path = root / 'output/execution_started43.json'
    if started_path.exists() or (root / 'output/execution_completed43.json').exists():
        raise FileExistsError('The frozen scientific grid is single pass; retain existing evidence')
    if (root / 'output/data_access43.jsonl').exists():
        raise RuntimeError('Natural measurement access preceded the controller start')
    c.write_json(started_path, {'schema': 'step43-controller-start-v1', 'utc': c.utc_now(),
                                'protocol_seal_sha256': protocol_hash, 'command_count': len(commands),
                                'producer_grid_runs': 1, 'outcome_based_reruns': 0,
                                'commands': [{'label': label, 'arguments': args} for label, args in commands]})
    c.event('execution_started', log='output/execution_events43.jsonl',
            protocol_seal_sha256=protocol_hash, command_count=len(commands))
    records = []
    started = time.perf_counter()
    environment = dict(os.environ)
    environment['PYTHONHASHSEED'] = '4303'
    try:
        for index, (label, arguments) in enumerate(commands):
            stdout = root / f'output/logs/{index:02d}_{label}.stdout.txt'
            stderr = root / f'output/logs/{index:02d}_{label}.stderr.txt'
            stdout.parent.mkdir(parents=True, exist_ok=True)
            c.event('command_started', log='output/execution_events43.jsonl',
                    command_index=index, label=label, arguments=arguments)
            print(f'[{index+1:02d}/{len(commands)}] START {label}', flush=True)
            stamp = time.perf_counter()
            with stdout.open('xb') as out, stderr.open('xb') as err:
                process = subprocess.run([sys.executable, '-u', *arguments], cwd=root,
                                         env=environment, stdout=out, stderr=err,
                                         timeout=protocol['execution']['command_budget_seconds'])
            record = {'command_index': index, 'label': label, 'arguments': arguments,
                      'returncode': process.returncode, 'wall_seconds': time.perf_counter()-stamp,
                      'stdout': c.file_ref(stdout), 'stderr': c.file_ref(stderr)}
            records.append(record)
            c.event('command_completed', log='output/execution_events43.jsonl', **record)
            print(f'[{index+1:02d}/{len(commands)}] EXIT {process.returncode} {label} '
                  f'({record["wall_seconds"]:.3f}s)', flush=True)
            if process.returncode:
                raise RuntimeError('Failed frozen command: ' + label)
        audit = c.read_json(root / 'audit/postrun_audit43.json')
        if audit.get('status') != 'PASS':
            raise RuntimeError('Final independent scientific audit did not pass')
        c.event('execution_completed', log='output/execution_events43.jsonl', commands=len(commands))
        events = root / 'output/execution_events43.jsonl'
        result = {'schema': 'step43-controller-complete-v1', 'status': 'PASS', 'utc': c.utc_now(),
                  'protocol_seal_sha256': protocol_hash, 'command_count': len(commands),
                  'commands': records, 'producer_grid_runs': 1, 'outcome_based_reruns': 0,
                  'wall_seconds': time.perf_counter()-started,
                  'result_sha256': c.sha_file(root / 'output/result43.json'),
                  'scientific_audit_sha256': c.sha_file(root / 'audit/postrun_audit43.json'),
                  'event_log_sha256': c.sha_file(events),
                  'event_count': len(events.read_bytes().splitlines())}
        c.write_json(root / 'output/execution_completed43.json', result)
        print(c.canonical_bytes({'status': 'PASS', 'commands': len(commands),
                                 'wall_seconds': result['wall_seconds'],
                                 'result_sha256': result['result_sha256']}).decode(), end='', flush=True)
    except Exception as failure:
        record = {'schema': 'step43-controller-failed-v1', 'status': 'STOPPED_RETAINED_FAILURE',
                  'utc': c.utc_now(), 'type': type(failure).__name__, 'reason': str(failure),
                  'traceback': traceback.format_exc(), 'completed_commands': records,
                  'protocol_seal_sha256': protocol_hash, 'producer_grid_runs': 1,
                  'outcome_based_reruns': 0, 'wall_seconds': time.perf_counter()-started}
        path = root / 'output/execution_failed43.json'
        if not path.exists():
            c.write_json(path, record)
        c.event('execution_failed', log='output/execution_events43.jsonl',
                reason=str(failure), failed_receipt=c.file_ref(path))
        print(c.canonical_bytes({'status': 'STOPPED_RETAINED_FAILURE', 'reason': str(failure)}).decode(),
              end='', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
