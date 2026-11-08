"""Write-once controller for the seven ordered Step 17 scientific stages."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
OUT = HERE.parent / 'output'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    if RUN.exists():
        raise FileExistsError('The scientific run is write-once: ' + str(RUN))
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE', '0') not in ('', '0'):
        raise RuntimeError('Scientific assertions must remain enabled')
    OUT.mkdir(exist_ok=True)
    stages = (
        ('runner17.py', 'freeze', 'freeze.json'),
        ('runner17.py', 'fit', 'fit_seal.json'),
        ('verify17.py', 'fit', 'fit_validation.json'),
        ('runner17.py', 'cal', 'cal_seal.json'),
        ('verify17.py', 'cal', 'cal_validation.json'),
        ('runner17.py', 'test', 'test_seal.json'),
        ('verify17.py', 'terminal', 'verification.json'),
    )
    with (OUT / 'controller_sequence.jsonl').open('x') as chronology:
        for script, stage, receipt in stages:
            command = [sys.executable, '-B', str(HERE / script), stage]
            if script == 'verify17.py':
                command += ['--out', str(RUN / receipt)]
            row = dict(script=script, stage=stage, started=utc(), argv=command,
                calibration_directory_absent_before_call=not (RUN / 'cal').exists(),
                terminal_directory_absent_before_call=not (RUN / 'test').exists())
            if stage != 'terminal':
                assert row['terminal_directory_absent_before_call'] is True
            if stage in ('freeze', 'fit'):
                assert row['calibration_directory_absent_before_call'] is True
            if script == 'runner17.py' and stage in ('cal', 'test'):
                gate = json.loads((RUN / 'fit_validation.json').read_text())
                assert gate['passed'] is True
                assert gate['checked_before_calibration_generation'] is True
                assert gate['checked_before_terminal_generation'] is True
                assert gate['freeze_sha256'] == sha(RUN / 'freeze.json')
                assert gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')
                row['used_fit_validation_sha256'] = sha(RUN / 'fit_validation.json')
            if stage == 'test':
                gate = json.loads((RUN / 'cal_validation.json').read_text())
                assert gate['passed'] is True and gate['checked_before_terminal_generation'] is True
                assert gate['freeze_sha256'] == sha(RUN / 'freeze.json')
                assert gate['fit_seal_sha256'] == sha(RUN / 'fit_seal.json')
                assert gate['cal_seal_sha256'] == sha(RUN / 'cal_seal.json')
                assert gate['precalibration_fit_gate_sha256'] == sha(RUN / 'fit_validation.json')
                row['used_cal_validation_sha256'] = sha(RUN / 'cal_validation.json')
            if (RUN / receipt).exists():
                raise FileExistsError('Stage receipt already exists: ' + receipt)
            chronology.write(json.dumps(dict(event='start', **row)) + '\n')
            chronology.flush()
            print('Starting', script, stage, flush=True)
            console_path = OUT / (script.removesuffix('.py') + '_' + stage + '.log')
            with console_path.open('x') as console:
                process = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT)
            row.update(finished=utc(), exit_code=process.returncode,
                console_sha256=sha(console_path),
                receipt_sha256=sha(RUN / receipt) if (RUN / receipt).is_file() else None)
            chronology.write(json.dumps(dict(event='exit', **row)) + '\n')
            chronology.flush()
            print('Finished', script, stage, process.returncode, flush=True)
            if process.returncode:
                raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
