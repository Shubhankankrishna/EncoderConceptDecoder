"""Validate a scoped Python startup repair, then resume the existing V9 queue.

No scientific source, lock, official test result, or completed run is edited.
The existing Q3 contracts and full smoke run on CPU with isolated outputs.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'
STATE = MON / 'python_startup_recovery.json'
PY = MON / 'v9_python'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(state):
    state['updated_utc'] = utc()
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def processes():
    out = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    rows = {}
    for line in out.splitlines():
        a = line.strip().split(None, 2)
        if len(a) == 3:
            rows[int(a[0])] = (int(a[1]), a[2])
    return rows


def check_cancel():
    if (MON / 'STOP').exists() or (MON / 'PYTHON_STARTUP_CANCEL').exists():
        raise RuntimeError('Explicit monitor/cancellation marker present')


def run(args, env=None, timeout=120):
    check_cancel()
    r = subprocess.run([str(PY), *map(str, args)], cwd=V9, env=env,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


def validation(state, name, folder, env):
    check_cancel()
    # Redirect only validation outputs, retaining the actual scientific source
    # and original locks/preflight machinery used by the isolated smoke driver.
    code = (f'import sys;from pathlib import Path;sys.path.insert(0,{str(V9 / "code")!r});'
            f'import {name} as m;m.TESTS=Path({str(folder / "tests")!r});')
    if name.endswith('contracts'):
        code += f'm.SPEC=Path({str(folder / "spec")!r});'
    code += 'm.main()'
    log_path = folder / (name + '.log')
    state.update(stage=name, log=str(log_path))
    with log_path.open('w') as log:
        child = subprocess.Popen([str(PY), '-B', '-c', code], cwd=V9,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        write(state)
        try:
            deadline = time.monotonic() + 14400
            while child.poll() is None:
                check_cancel()
                if time.monotonic() > deadline:
                    raise RuntimeError(name + ' exceeded validation timeout')
                time.sleep(5)
                write(state)
        except BaseException:
            child.terminate()
            raise
    state['child_pid'] = None
    result = read(folder / 'tests' / (name + '.json'))
    expected = read(V9 / 'spec/LOCK_Q3.json')['lock_sha256']
    assert child.returncode == 0 and result['status'] == 'passed', (name, result)
    assert result['lock_sha256'] == expected, 'Validation lock mismatch'
    state['completed_stages'].append(name)
    write(state)


def stop_waiter(state, block):
    check_cancel()
    st = read(V9 / f'runs/status_{block}.json')
    ps = processes()
    assert st['status'] == 'waiting' and not st.get('child_pid'), st
    pid = st['driver_pid']
    assert f'/code/drive_{block.lower()}.py' in ps.get(pid, (0, ''))[1]
    state.setdefault('operationally_stopped', []).append(block)
    write(state)
    os.kill(pid, signal.SIGTERM)
    until = time.monotonic() + 90
    while pid in processes() and time.monotonic() < until:
        time.sleep(1)
    assert pid not in processes(), block + ' failed to stop'


def resume(state, block, env):
    check_cancel()
    ps = processes()
    assert not any(f'/code/drive_{block.lower()}.py' in cmd for _, cmd in ps.values())
    if block == 'Q3':
        assert not any('/code/train_s2.py ' in cmd or '/code/measure_s2.py ' in cmd
                       for _, cmd in ps.values()), 'Trainer/evaluator still exists'
    launch = json.loads(run([V9 / 'code/launch_block.py', block], env))
    pid = launch['pid']
    state['resumed'][block] = launch
    write(state)
    until = time.monotonic() + 180
    while time.monotonic() < until:
        check_cancel()
        st = read(V9 / f'runs/status_{block}.json')
        ps = processes()
        if pid not in ps:
            raise RuntimeError(block + ' exited after launch')
        if st.get('driver_pid') == pid and st['status'] in ('running', 'waiting'):
            child = st.get('child_pid')
            if block != 'Q3' or child and ps.get(child, (0, ''))[0] == pid:
                state['resumed'][block].update(stage=st['stage'], heartbeat=st['updated_utc'], child_pid=child)
                write(state)
                return
        time.sleep(2)
    raise RuntimeError(block + ' heartbeat did not resume')


def main():
    with (MON / 'python_startup_recovery.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(STATE)
        state.update(status='validating', pid=os.getpid(), completed_stages=[], resumed={}, child_pid=None)
        folder = Path(tempfile.mkdtemp(prefix='v9_python_startup_validation_'))
        (folder / 'tests').mkdir()
        (folder / 'spec').mkdir()
        state['validation_dir'] = str(folder)
        write(state)
        try:
            check_cancel()
            assert read(V9 / 'runs/status_Q3.json')['status'] == 'stopped_by_user'
            protected = set((V9 / 'runs').glob('**/checkpoint.pt'))
            for pattern in ('**/system_checkpoint.pt', '**/measure.json', '**/train_summary.json'):
                protected.update((V9 / 'runs').glob(pattern))
            protected.update((V9 / 'spec').glob('LOCK_*.json'))
            protected.update((V9 / 'tests').glob('*.json'))
            protected.update(V9 / f'runs/{b}_RESULT.json' for b in ('Q1', 'Q2'))
            hashes = {str(p): digest(p) for p in sorted(protected)}
            (folder / 'protected_sha256.json').write_text(json.dumps(hashes, indent=2))
            env = dict(os.environ, V8_DEVICE='cpu', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONUNBUFFERED='1',
                       OMP_PREFIX='/opt/anaconda3/lib/python3.12/site-packages/torch',
                       DYLD_LIBRARY_PATH='/opt/anaconda3/lib/python3.12/site-packages/torch/lib',
                       TORCHINDUCTOR_CACHE_DIR=str(MON / 'q4_compiler_cache'))
            for k in ('V9_RUNS', 'V9_SMOKE', 'V8_DEADLINE_EPOCH'):
                env.pop(k, None)
            validation(state, 'q3_contracts', folder, env)
            validation(state, 'q3_smoke', folder, env)
            assert all(digest(Path(p)) == h for p, h in hashes.items()), 'Protected artifacts changed'
            state.update(status='recovering', completed_work_preserved=True, stage='resuming_queue')
            write(state)
            prod_env = dict(os.environ, V8_DEVICE='mps', PYTHONUNBUFFERED='1')
            for k in ('V9_RUNS', 'V9_SMOKE', 'V8_DEADLINE_EPOCH'):
                prod_env.pop(k, None)
            resume(state, 'Q3', prod_env)
            for block in ('Q4', 'Q5', 'Q6', 'QF'):
                stop_waiter(state, block)
                resume(state, block, prod_env)
            check_cancel()
            old_watch = read(MON / 'watch.pid')['pid']
            ps = processes()
            assert '/monitor/watch.py' in ps.get(old_watch, (0, ''))[1]
            os.kill(old_watch, signal.SIGTERM)
            until = time.monotonic() + 90
            while old_watch in processes() and time.monotonic() < until:
                time.sleep(1)
            assert old_watch not in processes(), 'Old watchdog did not stop'
            run([MON / 'watch.py', '--detach'], prod_env)
            state['watchdog_pid'] = read(MON / 'watch.pid')['pid']
            state['stage'] = 'confirming_K4_progress'
            write(state)
            until = time.monotonic() + 600
            while time.monotonic() < until:
                check_cancel()
                st = read(V9 / 'runs/status_Q3.json')
                assert st['status'] == 'running', st
                live_path = V9 / 'runs/S2_K4_dense_s911/live.json'
                if live_path.exists() and read(live_path).get('step', 0) >= 100:
                    state.update(status='completed', stage='finished', completed_utc=utc(),
                                 confirmed_progress=read(live_path), confirmed_heartbeat=st['updated_utc'])
                    write(state)
                    break
                time.sleep(5)
                write(state)
            else:
                raise RuntimeError('Recovered K4 did not reach 100 steps within 10 minutes')
        except BaseException:
            state.update(status='failed', error=traceback.format_exc())
            write(state)
        with (MON / 'recovery_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(state, utc=utc(), event='python_startup_recovery')) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
