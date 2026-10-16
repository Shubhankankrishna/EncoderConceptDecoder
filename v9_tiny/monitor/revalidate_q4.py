"""Revalidate Q4's stale lock with its existing contracts and smoke test on CPU."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'
STATE = MON / 'q4_validation.json'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(state):
    state['updated_utc'] = utc()
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def files_hash(paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def main():
    with (MON / 'q4_validation.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Q4 validation already running', flush=True)
            return
        os.nice(10)
        status = json.loads((V9 / 'runs/status_Q4.json').read_text())
        if status.get('status') == 'running':
            raise RuntimeError('Q4 production is already running; do not replace its lock')
        backup = MON / ('q4_validation_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        backup.mkdir()
        for name in ('spec/LOCK_Q4.json', 'tests/q4_contracts.json', 'tests/q4_smoke.json'):
            shutil.copy2(V9 / name, backup / Path(name).name)
        env = dict(os.environ, V8_DEVICE='cpu', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONUNBUFFERED='1')
        # Keep Torch and compiled CPU kernels on the same OpenMP runtime.
        # Loading Conda's libomp alongside Torch's copy deadlocks at a barrier.
        torch_dir = Path(sys.executable).resolve().parents[1] / 'lib/python3.12/site-packages/torch'
        if (torch_dir / 'include/omp.h').exists() and (torch_dir / 'lib/libomp.dylib').exists():
            env.update(OMP_PREFIX=str(torch_dir), DYLD_LIBRARY_PATH=str(torch_dir / 'lib'),
                       TORCHINDUCTOR_CACHE_DIR=str(MON / 'q4_compiler_cache'))
        for key in ('V9_RUNS', 'V9_SMOKE', 'V8_DEADLINE_EPOCH'):
            env.pop(key, None)
        state = dict(status='running', pid=os.getpid(), started_utc=utc(),
                     device='cpu', backup=str(backup), completed_stages=[])
        write(state)
        try:
            for stage in ('contracts', 'smoke'):
                state.update(stage=stage, child_pid=None)
                logfile = backup / (stage + '.log')
                state['log'] = str(logfile)
                with logfile.open('a') as log:
                    child = subprocess.Popen([sys.executable, '-B', str(V9 / f'code/q4_{stage}.py')],
                                             cwd=V9, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    write(state)
                    while child.poll() is None:
                        time.sleep(10)
                        write(state)
                state['child_pid'] = None
                if child.returncode:
                    raise RuntimeError(f'Q4 {stage} exited {child.returncode}; see {logfile}')
                test = json.loads((V9 / f'tests/q4_{stage}.json').read_text())
                lock = json.loads((V9 / 'spec/LOCK_Q4.json').read_text())
                if test.get('status') != 'passed' or test.get('lock_sha256') != lock['lock_sha256']:
                    raise RuntimeError(f'Q4 {stage} did not pass under the current lock')
                state['completed_stages'].append(stage)
                state['lock_sha256'] = lock['lock_sha256']
                write(state)
            lock = json.loads((V9 / 'spec/LOCK_Q4.json').read_text())
            checks = dict(
                code=files_hash([V9 / 'code' / name for name in lock['files']]) == lock['files_sha256'],
                frozen=files_hash(list((V9 / 'frozen/code').glob('*.py'))) == lock['frozen_implementation_sha256'])
            for label, filename, key in (
                ('manifest', 'manifest.json', 'data_manifest_sha256'),
                ('exclusions', 'evaluation_symbolic_hashes.txt', 'exclusion_list_sha256')):
                checks[label] = hashlib.sha256((V9 / 'frozen/data' / filename).read_bytes()).hexdigest() == lock[key]
            if not all(checks.values()):
                raise RuntimeError(f'Q4 inputs changed during validation: {checks}')
            state.update(status='completed', stage='finished', checks=checks, completed_utc=utc())
        except Exception:
            state.update(status='failed', error=traceback.format_exc())
        write(state)
        print(json.dumps(state), flush=True)
        if state['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
