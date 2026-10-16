"""End-to-end smoke of the Q1 driver (20-update runs, 64-item evaluations, a temporary runs directory)."""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import V9, CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_q1_smoke_')); t0 = time.time(); status = 'failed'; info = {}
    lock = read_json(SPEC / 'LOCK_Q1.json')
    try:
        env = dict(os.environ, V9_RUNS=str(tmp / 'runs'), V9_SMOKE='1')
        # Interrupted-run scenario: start the first run with the driver's exact arguments, SIGTERM it mid-training.
        args = subprocess.run([sys.executable, str(CODE / 'drive_q1.py'), '--dry-run'], cwd=tmp, env=env, capture_output=True, text=True).stdout.splitlines()[0].split()
        first = Path(args[args.index('--out') + 1]); first.mkdir(parents=True)
        child = subprocess.Popen([sys.executable, str(v9paths.FROZEN_CODE / 'train.py'), *args], cwd=tmp, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not (first / 'live.json').exists():
            assert child.poll() is None, 'smoke child ended before its first logged update'
            time.sleep(.05)
        child.terminate(); child.wait(timeout=300)
        pre = read_json(first / 'train_summary.json'); assert pre['status'] == 'interrupted_by_signal' and pre['completed_steps'] < 60, pre['status']
        r = subprocess.run([sys.executable, str(CODE / 'drive_q1.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=7200)
        if r.returncode:
            raise AssertionError('driver failed: ' + r.stdout[-3000:] + r.stderr[-3000:])
        runs = tmp / 'runs'; q = read_json(runs / 'Q1_RESULT.json')
        names = [f'S1_{a}_s{s}' for a in ('A_delayed', 'A_alpha0', 'A_flat') for s in (901, 902)]
        missing = [n for n in names if not (runs / n / 'train_summary.json').exists() or not (runs / (n + '_dev') / 'summary.json').exists()
                   or not (runs / (n + '_sealed') / 'summary.json').exists() or read_json(runs / (n + '_trained_contracts') / 'summary.json')['status'] != 'passed']
        assert not missing, missing
        assert all((runs / f'S1_{a}_s811_sealed' / 'summary.json').exists() for a in ('A_delayed', 'A_alpha0', 'A_flat'))
        ev0 = [json.loads(l) for l in (runs / 'queue.jsonl').read_text().splitlines()]
        assert any(e['event'] == 'resuming' and e['stage'] == first.name for e in ev0), 'driver did not resume the interrupted run'
        assert read_json(first / 'train_summary.json')['completed_steps'] == 60 and read_json(first / 'train_summary.json')['status'] == 'completed'
        info['interrupted_at'] = pre['completed_steps']
        res = read_json(runs / 'S1_RESULTS.json'); assert set(res['holm']) == {'H3', 'H4', 'H5'}
        assert read_json(runs / 'S1_A_delayed_s901_sealed' / 'summary.json')['split'] == 'dev', 'smoke must never read the sealed split'
        st = read_json(runs / 'status.json'); assert st['status'] == 'completed'
        # Restart on a completed queue must verify and skip everything, not retrain.
        r2 = subprocess.run([sys.executable, str(CODE / 'drive_q1.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=3600)
        assert r2.returncode == 0
        events = [json.loads(l)['event'] for l in (runs / 'queue.jsonl').read_text().splitlines()]
        info = dict(q1_result=q, teachers=read_json(runs / 'S2_TEACHERS.json'), restart_events=events[-30:], driver_seconds=time.time() - t0)
        assert events.count('verified_completed') >= 6 + 6 + 6 + 9, events
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        assert not changed, changed[:20]
        status = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    write_json(TESTS / 'q1_smoke.json', dict(status=status, lock_sha256=lock['lock_sha256'], temp_dir=str(tmp), seconds=time.time() - t0, **info))
    print(status, round(time.time() - t0)); print(info.get('traceback', ''))
    sys.exit(status != 'passed')


if __name__ == '__main__':
    main()
