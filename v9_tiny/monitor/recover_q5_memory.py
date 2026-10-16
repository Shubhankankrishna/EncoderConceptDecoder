"""Validate the allocation repair in isolation, then publish locks and resume once.

The original scientific exposures, pilot sizing, and completed outputs are retained.
This helper is protected by a file lock and keeps a durable stage/child heartbeat.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'
STATE = MON / 'q5_memory_recovery.json'
PY = MON / 'v9_python'


def read(p):
    return json.loads(p.read_text())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(s):
    s['updated_utc'] = utc()
    t = STATE.with_suffix('.json.tmp')
    t.write_text(json.dumps(s, indent=2))
    os.replace(t, STATE)


def processes():
    rows = {}
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True).splitlines():
        a = line.strip().split(None, 2)
        if len(a) == 3:
            rows[int(a[0])] = (int(a[1]), a[2])
    return rows


def check_cancel():
    if (MON / 'STOP').exists() or (MON / 'Q5_MEMORY_CANCEL').exists():
        raise RuntimeError('Explicit cancellation marker present')


def run_stage(s, name, args, env, python=None, timeout=21600):
    check_cancel()
    log = Path(s['validation_dir']) / (name + '.log')
    s.update(stage=name, log=str(log), child_pid=None)
    with log.open('w') as out:
        p = subprocess.Popen([str(python or PY), *map(str, args)], cwd=V9, env=env,
                             stdout=out, stderr=subprocess.STDOUT)
        s['child_pid'] = p.pid
        write(s)
        end = time.monotonic() + timeout
        try:
            while p.poll() is None:
                check_cancel()
                if time.monotonic() > end:
                    raise RuntimeError(name + ' timed out')
                time.sleep(5)
                write(s)
        except BaseException:
            p.terminate()
            raise
    s['child_pid'] = None
    if p.returncode:
        raise RuntimeError(f'{name} exited {p.returncode}; see {log}')
    s.setdefault('completed_stages', []).append(name)
    write(s)


def resume(s, block, env):
    check_cancel()
    ps = processes()
    assert not any(f'/code/drive_{block.lower()}.py' in c for _, c in ps.values())
    if block == 'Q5':
        assert not any('/code/train_s4.py' in c for _, c in ps.values()), 'orphan Q5 trainer'
    p = subprocess.run([str(PY), str(V9 / 'code/launch_block.py'), block], cwd=V9,
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    launch = json.loads(p.stdout)
    s.setdefault('resumed', {})[block] = launch
    write(s)
    end = time.monotonic() + 180
    while time.monotonic() < end:
        check_cancel()
        st = read(V9 / f'runs/status_{block}.json')
        ps = processes()
        assert launch['pid'] in ps, block + ' exited after launch'
        child = st.get('child_pid')
        if st.get('driver_pid') == launch['pid'] and st['status'] in ('running', 'waiting'):
            if block != 'Q5' or child and ps.get(child, (0, ''))[0] == launch['pid']:
                s['resumed'][block].update(stage=st['stage'], heartbeat=st['updated_utc'], child_pid=child)
                write(s)
                return
        time.sleep(2)
    raise RuntimeError(block + ' heartbeat did not resume')


def main():
    with (MON / 'q5_memory_recovery.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = read(STATE)
        folder = Path(s['validation_dir'])
        s.update(status='validating', pid=os.getpid(), started_validation_utc=utc(),
                 completed_stages=[], resumed={}, child_pid=None)
        write(s)
        gpu_probe = None
        gpu_log = None
        try:
            check_cancel()
            ps = processes()
            assert not any('/monitor/watch.py' in c or '/code/drive_q5.py' in c or '/code/drive_q6.py' in c or '/code/train_s4.py' in c for _, c in ps.values())
            backup = MON / ('q5_memory_repair_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
            backup.mkdir()
            s['backup'] = str(backup)
            shutil.copy2(folder / 'code/csearch.py', backup / 'csearch.before.py')
            shutil.copy2(folder / 'code/train_s4.py', backup / 'train_s4.before.py')
            for b in ('Q5', 'Q6'):
                for p in (V9 / f'spec/LOCK_{b}.json', V9 / f'tests/{b.lower()}_contracts.json', V9 / f'tests/{b.lower()}_smoke.json'):
                    shutil.copy2(p, backup / p.name)
            paths = set()
            for pattern in ('**/checkpoint.pt', '**/system_checkpoint.pt', '**/train_summary.json', '**/measure.json', '*RESULT.json'):
                paths.update((V9 / 'runs').glob(pattern))
            maps = [read(MON / 'q3_completion_notification.json')['artifacts_sha256'],
                    read(MON / 'q4_completed_artifacts.json'), read(MON / 'q5_completed_artifacts.json')]
            for mapping in maps:
                assert all(digest(V9.parent / p) == h for p, h in mapping.items()), 'Prior completed result changed'
                paths.update(V9.parent / p for p in mapping)
            protected = {str(p): digest(p) for p in sorted(paths)}
            (backup / 'completed_sha256.json').write_text(json.dumps(protected, indent=2))
            s['protected_completed_count'] = len(protected)
            s['repaired_source_sha256'] = digest(V9 / 'code/csearch.py')
            s['repaired_train_s4_sha256'] = digest(V9 / 'code/train_s4.py')
            bootstrap = folder / 'bootstrap.py'
            bootstrap.write_text(
                'import os,sys,runpy\nfrom pathlib import Path\n'
                f'sys.path.insert(0,{str(V9 / "code")!r})\nimport v9paths\n'
                f'v9paths.SPEC=Path({str(folder / "spec")!r})\n'
                f'v9paths.TESTS=Path({str(folder / "tests")!r})\n'
                f'v9paths.RUNS=Path(os.environ.get("V9_RUNS",{str(folder / "runs")!r}))\n'
                f'sys.executable={str(folder / "validation_python")!r}\n'
                'sys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n')
            launcher = folder / 'validation_python'
            import shlex
            launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(str(PY)) + ' ' + shlex.quote(str(bootstrap)) + ' "$@"\n')
            launcher.chmod(0o755)
            cpu = dict(os.environ, V8_DEVICE='cpu', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONUNBUFFERED='1',
                       OMP_PREFIX='/opt/anaconda3/lib/python3.12/site-packages/torch',
                       DYLD_LIBRARY_PATH='/opt/anaconda3/lib/python3.12/site-packages/torch/lib',
                       TORCHINDUCTOR_CACHE_DIR=str(MON / 'q4_compiler_cache'))
            for k in ('V9_RUNS', 'V9_SMOKE', 'V8_DEADLINE_EPOCH'):
                cpu.pop(k, None)
            gpu_log_path = folder / 'repaired_mps_probe.log'
            gpu_log = gpu_log_path.open('w')
            gpu_probe = subprocess.Popen([str(PY), str(MON / 'q5_memory_probe.py'), 'mps_workers'],
                                         cwd=V9, env=dict(cpu, V8_DEVICE='mps'), stdout=gpu_log, stderr=subprocess.STDOUT)
            s.update(probe_pid=gpu_probe.pid, probe_log=str(gpu_log_path))
            write(s)
            run_stage(s, 'exact_cpu_parity', [MON / 'q5_memory_parity.py', folder], cpu)
            for stage in ('q5_contracts', 'q6_contracts', 'q56_smoke'):
                run_stage(s, stage, [V9 / f'code/{stage}.py'], cpu, launcher)
            for b in ('Q5', 'Q6'):
                lock = read(folder / f'spec/LOCK_{b}.json')
                for kind in ('contracts', 'smoke'):
                    test = read(folder / f'tests/{b.lower()}_{kind}.json')
                    assert test['status'] == 'passed' and test['lock_sha256'] == lock['lock_sha256']
            # CPU validations and the isolated GPU full-frontier probe run independently.
            s['stage'] = 'awaiting_mps_stability'
            end = time.monotonic() + 14400
            while gpu_probe.poll() is None:
                check_cancel()
                assert time.monotonic() < end, 'GPU stability probe timed out'
                time.sleep(5); write(s)
            gpu_log.close(); gpu_log = None
            assert gpu_probe.returncode == 0, 'GPU stability probe failed; see ' + str(gpu_log_path)
            probe = [json.loads(l) for l in gpu_log_path.read_text().splitlines() if l.startswith('{')]
            endrow = next(r for r in probe if r.get('label') == 'workers_complete')
            resets = [r for r in probe if r.get('event') == 'search_progress']
            assert endrow['all_done'] and resets and len(resets) > 1, endrow
            peak = max(r['worker_driver_bytes'] for r in resets)
            assert peak < 12_000_000_000, peak
            s.update(mps_full_frontier=endrow, mps_peak_boundary_bytes=peak, mps_worker_handoffs=len(resets), probe_pid=None)
            s['completed_stages'].append('repaired_mps_probe')
            run_stage(s, 'physical_input_hashes', [MON / 'audit_q5_memory_inputs.py', folder / 'spec'], cpu)
            assert digest(V9 / 'code/csearch.py') == s['repaired_source_sha256'], 'Code changed during validation'
            assert digest(V9 / 'code/train_s4.py') == s['repaired_train_s4_sha256'], 'S4 code changed during validation'
            assert all(digest(Path(p)) == h for p, h in protected.items()), 'Completed outputs changed'
            s.update(status='publishing_validated_locks', completed_work_preserved=True)
            write(s)
            for b in ('Q5', 'Q6'):
                for rel in (f'spec/LOCK_{b}.json', f'tests/{b.lower()}_contracts.json', f'tests/{b.lower()}_smoke.json'):
                    p = V9 / rel
                    tmp = p.with_suffix(p.suffix + '.memory.tmp')
                    shutil.copyfile(folder / rel, tmp)
                    os.replace(tmp, p)
            s['new_locks'] = {b: read(V9 / f'spec/LOCK_{b}.json')['lock_sha256'] for b in ('Q5', 'Q6')}
            prod = dict(cpu, V8_DEVICE='mps')
            for b in ('Q5', 'Q6'):
                resume(s, b, prod)
            run_stage(s, 'restore_watchdog', [MON / 'watch.py', '--detach'], prod, timeout=120)
            s['watchdog_pid'] = read(MON / 'watch.pid')['pid']
            s.update(status='confirming_recovered_progress', stage='Q5_search_progress')
            write(s)
            # A fresh driver heartbeat plus advancing child CPU confirms active search;
            # search has no training-step log until the full iteration is saved.
            child = s['resumed']['Q5']['child_pid']
            def search_cpu():
                ps = processes()
                pids = [child] + [pid for pid, (parent, cmd) in ps.items()
                                   if parent == child and '/code/train_s4.py' in cmd]
                return {pid: subprocess.check_output(['ps', '-p', str(pid), '-o', 'time='], text=True).strip()
                        for pid in pids if pid in ps}
            before_cpu = search_cpu()
            end = time.monotonic() + 300
            while time.monotonic() < end:
                check_cancel()
                time.sleep(5)
                st = read(V9 / 'runs/status_Q5.json')
                assert st['status'] == 'running' and st['child_pid'] == child, st
                assert child in processes() and s['watchdog_pid'] in processes()
                now_cpu = search_cpu()
                write(s)
                advancing = any(pid in before_cpu and value != before_cpu[pid] for pid, value in now_cpu.items())
                for pid, value in now_cpu.items():
                    before_cpu.setdefault(pid, value)
                if advancing and st['updated_utc'] != s['resumed']['Q5']['heartbeat']:
                    s.update(status='completed', stage='finished', completed_utc=utc(),
                             confirmed_child_cpu=now_cpu, confirmed_heartbeat=st['updated_utc'])
                    write(s)
                    break
            else:
                raise RuntimeError('Q5 heartbeat or child CPU did not advance')
        except BaseException:
            if gpu_probe is not None and gpu_probe.poll() is None:
                gpu_probe.terminate(); gpu_probe.wait(timeout=90)
            if gpu_log is not None:
                gpu_log.close()
            s.update(status='failed', error=traceback.format_exc())
            write(s)
        with (MON / 'recovery_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(s, event='q5_memory_recovery', utc=utc())) + '\n')
        print(json.dumps(s), flush=True)
        if s['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
