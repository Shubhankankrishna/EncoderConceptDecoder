"""End-to-end smoke of the Q4 driver on stand-in S2 outputs (a K = 8 fixture promoted on two seeds, fixture K1
measurements), 20-update S3 arms, quick measurements, G3 and the forward decision; then a restart that verifies all
completed work without retraining. Temporary runs directory."""
import json, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot


def stand_in(runs):
    import hncp
    t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']; sha = digest(t); models = []; q3runs = {}
    for i, seed in enumerate((911, 912)):
        torch.manual_seed(seed); hs, _ = hncp.build(t, 'dense', 8, 'cpu')
        with torch.no_grad():
            hs.codes.W_z.weight.normal_(std=.3); hs.codes.selector.weight.normal_(std=.1)
        name = f'S2_K8_dense_s{seed}'; (runs / name).mkdir(parents=True)
        cfg = dict(teacher=t, teacher_sha256=sha, k=8, m=2, core='dense', seed=seed, alpha=.1, eps=.05, lam_sel=.1)
        hncp.save(runs / name / 'checkpoint.pt', hs, dict(config=cfg)); write_json(runs / name / 'config.json', cfg)
        shutil.copy(TESTS / 'fixtures' / 'm_k8.json', runs / name / 'measure.json')
        k1 = runs / f'S2_K1_dense_s{seed}'; k1.mkdir(); shutil.copy(TESTS / 'fixtures' / 'm_k1.json', k1 / 'measure.json')
        src = dict(name=name, checkpoint=str(runs / name / 'checkpoint.pt'), measure=str(runs / name / 'measure.json'), teacher='A', teacher_sha256=sha, seed=seed, core='dense', k=8, m=2, p_cl=.25)
        models.append(dict(seed=seed, s3_seed=921 + i, source=src, flag=None))
        q3runs[f'K1/dense/{seed}'] = dict(measure=str(k1 / 'measure.json')); q3runs[f'K8/dense/{seed}'] = src
    write_json(runs / 'S2_PROMOTION.json', dict(k8_promoted=True, models=models, consequences={}, g22d_per_seed=['pass', 'pass']))
    write_json(runs / 'Q3A_RESULT.json', dict(status='completed', runs=q3runs, consequences=dict(H_star=1)))


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_q4_smoke_')); t0 = time.time(); status = 'failed'; info = {}
    lock = read_json(SPEC / 'LOCK_Q4.json')
    try:
        runs = tmp / 'runs'; runs.mkdir(); stand_in(runs)
        env = dict(os.environ, V9_RUNS=str(runs), V9_SMOKE='1')
        r = subprocess.run([sys.executable, str(CODE / 'drive_q4.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=14400)
        if r.returncode:
            fail = (runs / 'failure_Q4.json').read_text()[-4000:] if (runs / 'failure_Q4.json').exists() else ''
            raise AssertionError('driver failed: ' + r.stderr[-3000:] + fail)
        q = read_json(runs / 'Q4_RESULT.json'); assert q['status'] == 'completed' and len(q['seeds']) == 2
        for row in q['seeds']:
            assert set(row['arms']) == {'dec', 'joint'} and all('G3' in a['G3'] for a in row['arms'].values())
            lam = row['lambda_curve']['rows']; assert set(lam) == {'0.0', '0.3', '0.6', '1.0'}
        d = read_json(runs / 'S3_DECISION.json'); assert len(d['forward']) == 2
        for s in (921, 922):
            for arm in ('dec', 'joint'):
                j = read_json(runs / f'S3_s{s}_{arm}' / 'train_summary.json'); assert j['status'] == 'completed' and j['completed_steps'] == 20
        ev = [json.loads(l) for l in (runs / 'queue.jsonl').read_text().splitlines()]
        r2 = subprocess.run([sys.executable, str(CODE / 'drive_q4.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=3600)
        assert r2.returncode == 0, r2.stderr[-2000:]
        ev2 = [json.loads(l)['event'] for l in (runs / 'queue.jsonl').read_text().splitlines()][len(ev):]
        assert ev2.count('started') == 0 and ev2.count('verified_completed') >= 2 + 4 + 4, ev2
        info = dict(decision=d['decision'], lambda_rows={r['seed']: r['lambda_curve']['rows'] for r in q['seeds']})
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        assert not changed, changed[:20]
        status = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    write_json(TESTS / 'q4_smoke.json', dict(status=status, lock_sha256=lock['lock_sha256'], temp_dir=str(tmp), seconds=time.time() - t0, **info))
    print(status, round(time.time() - t0)); print(info.get('traceback', ''))
    sys.exit(status != 'passed')


if __name__ == '__main__':
    main()
