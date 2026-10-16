"""Validate the recurrent evaluation memory repair, then resume the existing queue.

Waits for the isolated full-size MPS probe before editing evaluation batching.
Each affected pending block resumes only after its existing contracts and smoke
pass under its current hashes. Completed Q2's original lock/results are archived
and retained; no completed training or measurement is recalculated.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'
STATE = MON / 'i22_validation.json'
OLD = 'return 64 if eng.K == 1 else 16'
NEW = "return 64 if eng.K == 1 and eng.hs.core_kind == 'dense' else 16"


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
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
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
    if (MON / 'STOP').exists() or (MON / 'I22_CANCEL').exists():
        raise RuntimeError('Monitor stop/cancellation marker present; no further launch')


def stop_waiter(block):
    check_cancel()
    st = read(V9 / f'runs/status_{block}.json')
    assert st['status'] == 'waiting' and not st.get('child_pid'), st
    pid = st['driver_pid']; ps = processes()
    assert f'/code/drive_{block.lower()}.py' in ps.get(pid, (None, ''))[1], (block, pid)
    os.kill(pid, signal.SIGTERM)
    until = time.time() + 90
    while pid in processes() and time.time() < until:
        time.sleep(1)
    assert pid not in processes(), f'{block} driver failed to stop'
    assert read(V9 / f'runs/status_{block}.json')['status'] == 'stopped_by_user'


def completed_snapshot():
    paths = set((V9 / 'runs').glob('**/checkpoint.pt'))
    paths.update((V9 / 'runs').glob('**/system_checkpoint.pt'))
    paths.update((V9 / 'runs').glob('**/measure.json'))
    paths.update((V9 / 'runs').glob('**/train_summary.json'))
    paths.update(V9 / f'runs/{b}_RESULT.json' for b in ('Q1', 'Q2'))
    return {str(p): digest(p) for p in sorted(paths) if p.exists()}


def verify_inputs(block):
    lock = read(V9 / f'spec/LOCK_{block}.json')
    h = hashlib.sha256()
    for p in sorted(V9 / 'code' / n for n in lock['files']):
        h.update(p.name.encode()); h.update(p.read_bytes())
    assert h.hexdigest() == lock['files_sha256'], f'{block}: code hash changed'
    h = hashlib.sha256()
    for p in sorted((V9 / 'frozen/code').glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    assert h.hexdigest() == lock['frozen_implementation_sha256']
    assert digest(V9 / 'frozen/data/manifest.json') == lock['data_manifest_sha256']
    assert digest(V9 / 'frozen/data/evaluation_symbolic_hashes.txt') == lock['exclusion_list_sha256']
    identity = {k: lock[k] for k in ('block', 'frozen_implementation_sha256', 'files_sha256', 'data_manifest_sha256', 'exclusion_list_sha256')}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == lock['lock_sha256']
    for kind in ('contracts', 'smoke'):
        t = read(V9 / f'tests/{block.lower()}_{kind}.json')
        assert t['status'] == 'passed' and t['lock_sha256'] == lock['lock_sha256'], (block, kind)
    return lock['lock_sha256']


def run_stage(state, stage, env, backup):
    check_cancel()
    state.update(stage=stage, child_pid=None, log=str(backup / (stage + '.log')))
    with Path(state['log']).open('a') as log:
        child = subprocess.Popen([sys.executable, '-B', str(V9 / f'code/{stage}.py')],
                                 cwd=V9, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid; write(state)
        while child.poll() is None:
            time.sleep(10)
            if (MON / 'STOP').exists() or (MON / 'I22_CANCEL').exists():
                child.terminate()
                raise RuntimeError('Validation cancelled; no production launch')
            write(state)
    state['child_pid'] = None
    assert child.returncode == 0, f'{stage} failed ({child.returncode}); see {state["log"]}'
    state['completed_stages'].append(stage); write(state)


def resume(state, block, prod_env):
    check_cancel()
    lock = verify_inputs(block)
    ps = processes()
    assert not any(f'/code/drive_{block.lower()}.py' in cmd for _, cmd in ps.values()), f'{block} driver already exists'
    if block == 'Q3':
        assert not any('/code/measure_s2.py ' in cmd and '/v9_tiny/runs/' in cmd for _, cmd in ps.values()), 'Orphan production evaluator exists'
    r = subprocess.run([sys.executable, '-B', str(V9 / 'code/launch_block.py'), block],
                       cwd=V9, env=prod_env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    launch = json.loads(r.stdout)
    pid = launch['pid']
    until = time.time() + 120
    while time.time() < until:
        st = read(V9 / f'runs/status_{block}.json')
        ps = processes()
        ready = st['status'] in ('running', 'waiting')
        if block == 'Q3':
            ready = st['status'] == 'running' and bool(st.get('child_pid'))
        if st['driver_pid'] == pid and ready and pid in ps:
            if st.get('child_pid'):
                assert ps[st['child_pid']][0] == pid
            state['resumed'][block] = dict(pid=pid, child_pid=st.get('child_pid'), lock_sha256=lock,
                                          utc=utc(), stage=st['stage'], heartbeat=st['updated_utc'])
            write(state)
            if block == 'Q3':
                first = st['updated_utc']; state['stage'] = 'confirming_Q3_recovery'
                for _ in range(12):
                    time.sleep(10); check_cancel(); ps = processes()
                    st = read(V9 / 'runs/status_Q3.json')
                    assert pid in ps and st['status'] == 'running', 'Q3 stopped after recovery'
                    child = st.get('child_pid')
                    assert child in ps and ps[child][0] == pid, 'Q3 recovery child is absent/orphaned'
                    write(state)
                    if st['updated_utc'] != first:
                        state['resumed'][block].update(confirmed_utc=utc(), heartbeat=st['updated_utc'])
                        write(state); return
                raise RuntimeError('Q3 recovered driver did not refresh its heartbeat')
            return
        if pid not in ps:
            raise RuntimeError(f'{block} driver exited after launch; see driver log')
        time.sleep(2)
    raise RuntimeError(f'{block} launch heartbeat did not appear')


def main():
    with (MON / 'i22_validation.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(10)
        state = dict(status='running', pid=os.getpid(), started_utc=utc(),
                     stage='waiting_for_scratch_probe', completed_stages=[], resumed={}, device='cpu',
                     helper_sha256=digest(Path(__file__)), operationally_stopped=['Q3'])
        write(state)
        try:
            check_cancel()
            while True:
                probe = read(MON / 'i22_probe.json')
                if probe['status'] != 'running':
                    break
                assert probe['pid'] in processes(), 'Scratch probe died without an outcome'
                time.sleep(10); check_cancel(); write(state)
            assert probe['status'] == 'completed' and probe['batch'] == 16 and probe['determinism'], probe
            result = read(Path(probe['output']))
            assert result['checkpoint_sha256'] == digest(V9 / 'runs/S2_K1_rec_s911/checkpoint.pt')
            assert result['teacher_sha256'] == digest(V9 / 'runs/S1_A_delayed_s902/checkpoint.pt')
            assert all(len(result['greedy'][name]['success']) == n for name, n in [('standard', 2048), ('hard10', 512), ('hard14', 256)])
            parity = read(MON / 'i22_batch_parity.json')
            assert parity['status'] == 'passed' and parity['checkpoint_sha256'] == result['checkpoint_sha256']
            assert read(V9 / 'runs/status_Q3.json')['status'] == 'stopped_by_user'
            for block in ('Q4', 'Q5', 'Q6'):
                stop_waiter(block)
                state['operationally_stopped'].append(block); write(state)
            backup = MON / ('i22_validation_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
            backup.mkdir(); state['backup'] = str(backup)
            for name in ['code/measure_s2.py', 'spec/PRE_REGISTRATION.md'] + [f'{folder}/{prefix}{b}{suffix}.json' for b in ('Q2', 'Q3', 'Q4', 'Q5', 'Q6') for folder, prefix, suffix in [('spec', 'LOCK_', ''), ('tests', '', '_contracts'), ('tests', '', '_smoke')]]:
                # Test names are lowercase; lock names are uppercase.
                p = V9 / name
                if name.startswith('tests/'):
                    p = p.with_name(p.name.lower())
                shutil.copy2(p, backup / p.name)
            protected = completed_snapshot()
            (backup / 'completed_work_sha256.json').write_text(json.dumps(protected, indent=2))
            for name in ('i22_probe.json', 'i22_batch_parity.json', 'probe_i22.py', 'revalidate_i22.py'):
                shutil.copy2(MON / name, backup / name)
            target = V9 / 'code/measure_s2.py'; source = target.read_text()
            assert source.count(OLD) == 1, 'Evaluation code changed; inspect before applying repair'
            source = source.replace(OLD, NEW)
            source = source.replace('where K = 1 keeps 64 (I19)', 'where dense K = 1 keeps 64 (I19); recurrent K = 1 also uses 16 (I22)')
            tmp = target.with_suffix('.py.tmp'); tmp.write_text(source); os.replace(tmp, target)
            peak = max(e['driver_gb'] for e in probe['events'])
            note = f'''\n\n## I22. Recurrent S2 evaluation batch 64 → 16 ({utc()})\n\n- **Failure.** The first recurrent K=1 S2 measurement, S2_K1_rec_s911_measure, failed during D_self with MPS out of memory (2.46 GiB tensors plus 17.65 GiB other allocations, cap 20.13 GiB). Its 12,000 training updates had completed. The watchdog restarted once; that retry was stopped operationally for diagnosis, without changing the completed checkpoint.\n- **Validation before the edit.** A fresh full production-size scratch measurement of that checkpoint with recurrent batch 16 completed in {probe['seconds']:.1f} seconds, including all states and drift. Peak observed MPS driver allocation was {peak:.3f} GB. Scratch output stays outside runs/ and is not used as a scientific result. This extends the shape-memory mitigation documented in I19/I20 to recurrent K=1.\n- **Change.** eval_batch keeps dense K=1 at 64 and all K>1 at 16; recurrent K=1 now uses 16. Architectures, training, seeds, exposures, data items, suite sizes, thresholds and gates stay unchanged. MPS rounding may differ at rare ties when changing batches; no recurrent S2 measurement had completed.\n- **Locks and recovery.** The affected pending Q3, Q4, Q5 and Q6 blocks are stopped operationally during their own revalidation. Their existing contracts and full smoke workflows run on CPU with isolated temporary runs, then each resumes through launch_block only when physical code/data hashes and test identities match. Q3 resumes as soon as its own validation passes while later validations continue. Completed Q2 retains its original historical lock and results, with the original shared source archived under {backup}. QF's source list is unchanged. Checkpoints and all completed measurements are preserved and checked by SHA256 before Q3 resumes.\n'''
            with (V9 / 'spec/PRE_REGISTRATION.md').open('a') as f:
                f.write(note)
            state.update(stage='q3_contracts', scratch_probe_peak_driver_gb=peak,
                         code_sha256=digest(target), completed_work_count=len(protected)); write(state)
            prod_env = dict(os.environ, V8_DEVICE='mps')
            env = dict(os.environ, V8_DEVICE='cpu', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONUNBUFFERED='1')
            torch_dir = Path(sys.executable).resolve().parents[1] / 'lib/python3.12/site-packages/torch'
            env.update(OMP_PREFIX=str(torch_dir), DYLD_LIBRARY_PATH=str(torch_dir / 'lib'),
                       TORCHINDUCTOR_CACHE_DIR=str(MON / 'q4_compiler_cache'))
            for e in (env, prod_env):
                for key in ('V9_RUNS', 'V9_SMOKE', 'V8_DEADLINE_EPOCH'):
                    e.pop(key, None)
            for block in ('Q3', 'Q4'):
                for kind in ('contracts', 'smoke'):
                    run_stage(state, f'{block.lower()}_{kind}', env, backup)
                if block == 'Q3':
                    assert all(digest(Path(p)) == h for p, h in protected.items()), 'Completed work changed during validation'
                    state['completed_work_preserved'] = True
                resume(state, block, prod_env)
            for block in ('Q5', 'Q6'):
                run_stage(state, f'{block.lower()}_contracts', env, backup)
            run_stage(state, 'q56_smoke', env, backup)
            for block in ('Q5', 'Q6'):
                resume(state, block, prod_env)
            state.update(status='completed', stage='finished', completed_utc=utc())
        except Exception:
            state.update(status='failed', error=traceback.format_exc())
        write(state)
        with (MON / 'recovery_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(utc=utc(), action='I22_revalidation_chain', **state)) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
