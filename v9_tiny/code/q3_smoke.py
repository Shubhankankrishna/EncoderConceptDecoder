"""End-to-end smoke of the Q3 driver: 40-update S2 arms, 20-update replays, quick measurements, register, promotion,
an interrupted-and-resumed arm, and a restart that verifies everything without retraining. Temporary runs directory."""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_q3_smoke_')); t0 = time.time(); status = 'failed'; info = {}
    lock = read_json(SPEC / 'LOCK_Q3.json')
    try:
        runs = tmp / 'runs'; runs.mkdir()
        t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']
        write_json(runs / 'S2_TEACHERS.json', dict(teachers=['A', 'B'], flag='smoke', checkpoints={'A': t['checkpoint'], 'B': t['checkpoint']}, sha256={'A': t['checkpoint_sha256'], 'B': t['checkpoint_sha256']}))
        env = dict(os.environ, V9_RUNS=str(runs), V9_SMOKE='1')
        # Interrupt the first arm mid-training with the driver's exact arguments.
        first = runs / 'S2_V8repro_dense_s911'; first.mkdir()
        args = ['--teacher', t['checkpoint'], '--out', str(first), '--core', 'dense', '--k', '1', '--m', '2', '--p-cl', '0.0', '--steps', '40', '--seed', '911', '--compiled']
        child = subprocess.Popen([sys.executable, str(CODE / 'train_s2.py'), *args], cwd=tmp, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not (first / 'live.json').exists():
            assert child.poll() is None, 'child ended early'; time.sleep(.05)
        child.terminate(); child.wait(timeout=300)
        pre = read_json(first / 'train_summary.json'); assert pre['status'] == 'interrupted_by_signal' and pre['completed_steps'] < 40, pre['status']
        r = subprocess.run([sys.executable, str(CODE / 'drive_q3.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=14400)
        if r.returncode:
            raise AssertionError('driver failed: ' + r.stdout[-3000:] + r.stderr[-3000:] + (runs / 'failure_Q3.json').read_text()[-3000:] if (runs / 'failure_Q3.json').exists() else '')
        q = read_json(runs / 'Q3_RESULT.json'); assert q['status'] == 'completed' and read_json(runs / 'Q3A_RESULT.json')['status'] == 'completed'
        ev = [json.loads(l) for l in (runs / 'queue.jsonl').read_text().splitlines()]
        assert any(e['event'] == 'resuming' and e['stage'] == first.name for e in ev), 'no resume of the interrupted arm'
        assert read_json(first / 'train_summary.json')['completed_steps'] == 40
        for core in ('dense', 'recurrent'):
            reg = read_json(runs / f'S2_REGISTER_{core}.json'); assert len(reg['per_seed']) == 2 and 'H_star' in reg['consequences']
        assert len(read_json(runs / 'S2_PROMOTION.json')['models']) == 2
        # Same seed -> same data order across arms (K1 vs K8 dense at seed 911).
        import torch
        m1 = torch.load(runs / 'S2_K1_dense_s911' / 'checkpoint.pt', map_location='cpu', weights_only=False)['mixer']['seen_symbols']
        m8 = torch.load(runs / 'S2_K8_dense_s911' / 'checkpoint.pt', map_location='cpu', weights_only=False)['mixer']['seen_symbols']
        assert m1 == m8, 'arms at one seed must see identical batches'
        r2 = subprocess.run([sys.executable, str(CODE / 'drive_q3.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=3600)
        assert r2.returncode == 0, r2.stderr[-2000:]
        ev2 = [json.loads(l)['event'] for l in (runs / 'queue.jsonl').read_text().splitlines()][len(ev):]
        assert 'started' not in [e for e in ev2 if e != 'started'] and ev2.count('verified_completed') >= 10 + 10 + 2 + 2, ev2
        info = dict(promotion=read_json(runs / 'S2_PROMOTION.json')['k8_promoted'], replay=q['replay'], driver_seconds=time.time() - t0, interrupted_at=pre['completed_steps'])
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        assert not changed, changed[:20]
        status = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    write_json(TESTS / 'q3_smoke.json', dict(status=status, lock_sha256=lock['lock_sha256'], temp_dir=str(tmp), seconds=time.time() - t0, **info))
    print(status, round(time.time() - t0)); print(info.get('traceback', ''))
    sys.exit(status != 'passed')


if __name__ == '__main__':
    main()
