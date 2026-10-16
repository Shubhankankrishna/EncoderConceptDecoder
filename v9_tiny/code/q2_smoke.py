"""End-to-end smoke of the Q2 driver (10-update phases, capped suites, 4-batch GMM fits) in a temporary runs directory,
plus a restart that verifies all completed work without retraining."""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_q2_smoke_')); t0 = time.time(); status = 'failed'; info = {}
    lock = read_json(SPEC / 'LOCK_Q2.json')
    try:
        runs = tmp / 'runs'; runs.mkdir()
        t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']
        write_json(runs / 'S2_TEACHERS.json', dict(teachers=['A', 'B'], flag='smoke', checkpoints={'A': t['checkpoint'], 'B': t['checkpoint']}, sha256={'A': t['checkpoint_sha256'], 'B': t['checkpoint_sha256']}))
        env = dict(os.environ, V9_RUNS=str(runs), V9_SMOKE='1')
        r = subprocess.run([sys.executable, str(CODE / 'drive_q2.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=14400)
        if r.returncode:
            fail = (runs / 'failure_Q2.json').read_text()[-4000:] if (runs / 'failure_Q2.json').exists() else ''
            raise AssertionError('driver failed: ' + r.stderr[-3000:] + fail)
        q = read_json(runs / 'Q2_RESULT.json'); assert q['status'] == 'completed'
        for seed, sd in q['seeds'].items():
            ph = sd['phases']; assert set(ph) == {'1', '2', '3'} or set(ph) == {1, 2, 3}, ph.keys()
            p3 = ph.get('3', ph.get(3)); assert 'H12' in p3 and p3['mode'] == 'bind'
            for k in ('1', '2', '3'):
                row = ph.get(k, ph.get(int(k))); assert all(g in row['lsl'] for g in ('G6.1', 'G6.2', 'G6.3', 'G6.4', 'G6.5', 'lsl_credit')), row['lsl'].keys()
                assert row['telemetry'] and all(v is not None for v in row['function_kl'].values())
        ev = [json.loads(l) for l in (runs / 'queue.jsonl').read_text().splitlines()]
        r2 = subprocess.run([sys.executable, str(CODE / 'drive_q2.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=3600)
        assert r2.returncode == 0, r2.stderr[-2000:]
        ev2 = [json.loads(l)['event'] for l in (runs / 'queue.jsonl').read_text().splitlines()][len(ev):]
        assert ev2.count('started') == 0 and ev2.count('verified_completed') >= 18, ev2
        info = dict(seconds_driver=time.time() - t0, chosen_P2={s: d['g60']['2']['chosen'] if '2' in d['g60'] else d['g60'][2]['chosen'] for s, d in q['seeds'].items()})
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        assert not changed, changed[:20]
        status = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    write_json(TESTS / 'q2_smoke.json', dict(status=status, lock_sha256=lock['lock_sha256'], temp_dir=str(tmp), seconds=time.time() - t0, **info))
    print(status, round(time.time() - t0)); print(info.get('traceback', ''))
    sys.exit(status != 'passed')


if __name__ == '__main__':
    main()
