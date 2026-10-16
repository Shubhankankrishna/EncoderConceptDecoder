"""End-to-end smoke of the Q5 and Q6 drivers on stand-in upstream results (one promoted K = 8 fixture seed), tiny sizes,
in one temporary runs directory; then a restart of both that verifies every completed stage without recomputation.
Writes tests/q5_smoke.json and tests/q6_smoke.json (each under its own lock). Smoke runs never read sealed items."""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json, digest
from q1_contracts import runs_snapshot


def stand_in(runs):
    import hncp
    t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']; torch.manual_seed(911); hs, _ = hncp.build(t, 'dense', 8, 'cpu')
    with torch.no_grad():
        hs.codes.W_z.weight.normal_(std=.3); hs.codes.selector.weight.normal_(std=.1)
    d = runs / 'S2_K8_dense_s911'; d.mkdir(parents=True); hncp.save(d / 'checkpoint.pt', hs, dict(config=dict(teacher=t, k=8, m=2, core='dense', seed=911)))
    write_json(runs / 'S3_DECISION.json', dict(decision=dict(replaces_S2=None), forward=[dict(seed=911, checkpoint=str(d / 'checkpoint.pt'), stage='S2', flag=None)]))
    write_json(runs / 'S2_PROMOTION.json', dict(k8_promoted=True, models=[]))
    write_json(runs / 'Q3A_RESULT.json', dict(status='completed', runs={}, consequences=dict(H_star=1, RH1_allowed=True, RH1_exploratory=False, concept_branches_run=True, IMAG='flagged')))
    write_json(runs / 'S2_REGISTER_dense.json', dict(combined={'G2.3_H2': dict(zone='partial'), 'G2.3_H4': dict(zone='fail')}))


def run_driver(name, runs, env, tmp):
    r = subprocess.run([sys.executable, str(CODE / name)], cwd=tmp, env=env, capture_output=True, text=True, timeout=21600)
    if r.returncode:
        blk = name.split('_')[1].split('.')[0].upper(); f = runs / f'failure_{blk}.json'
        raise AssertionError(f'{name} failed: ' + r.stderr[-3000:] + (f.read_text()[-4000:] if f.exists() else ''))


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_q56_smoke_')); t0 = time.time(); info = {}; status = {'Q5': 'failed', 'Q6': 'failed'}
    locks = {b: read_json(SPEC / f'LOCK_{b}.json') for b in ('Q5', 'Q6')}
    try:
        runs = tmp / 'runs'; runs.mkdir(); stand_in(runs); env = dict(os.environ, V9_RUNS=str(runs), V9_SMOKE='1')
        run_driver('drive_q5.py', runs, env, tmp)
        q5 = read_json(runs / 'Q5_RESULT.json'); assert q5['status'] == 'completed' and q5['concept_branch']
        row = q5['seeds'][0]; assert set(row['arms']) == {'reference', 'EI', 'rft', 'grpo', 'reinforce'}, row['arms'].keys()
        ei = read_json(runs / 'S4_s911_EI' / 'train_summary.json'); assert ei['status'] == 'completed' and ei['iterations'] == [0, 1]
        for m in ('rft', 'grpo', 'reinforce'):
            rr = read_json(runs / f'S4_s911_{m}' / 'rounds.json'); assert all(r['verifier_calls'] >= r['target_calls'] for r in rr), (m, rr)
        assert (runs / 'S4_s911_EI' / 'iter1' / 'traj.pt').exists() and read_json(runs / 'S4_SIZING.json')['rule'] == 'I15'
        status['Q5'] = 'passed'
        run_driver('drive_q6.py', runs, env, tmp)
        q6 = read_json(runs / 'Q6_RESULT.json'); assert q6['status'] == 'completed'
        cells = q6['seeds'][0]['cells']; assert len(cells) == 2 * 2 * 4, cells.keys()
        proc = runs / 'S5_SEALED_PROCEDURE_s911.json'; first_sealed = min((runs / 'S5').glob('s911_sealed_*.json'), key=lambda p: p.stat().st_mtime)
        assert proc.stat().st_mtime < first_sealed.stat().st_mtime, 'procedure must be written before the first sealed cell'
        for k in ('answerloop_gradcuit', 'answerloop_random', 'RH1grounded', 'RH1nodematched'):
            assert list((runs / 'S5').glob(f's911_sealed_{k}_*')), k
        assert list((runs / 'S5').glob('s911_dev_G2.json')) and list((runs / 'S5').glob('s911_dev_IMAG.json')) and not list((runs / 'S5').glob('s911_dev_G4.json'))
        assert not (SPEC / 'S5_SEALED_PROCEDURE_s911.json').exists()
        n_ev = len((runs / 'queue.jsonl').read_text().splitlines())
        run_driver('drive_q5.py', runs, env, tmp); run_driver('drive_q6.py', runs, env, tmp)
        ev2 = [json.loads(l)['event'] for l in (runs / 'queue.jsonl').read_text().splitlines()][n_ev:]
        assert ev2.count('started') == 0 and ev2.count('verified_completed') >= 20, ev2
        info = dict(q5=dict(sizing=q5['sizing']), q6_cells=cells, seconds=time.time() - t0)
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        assert not changed, changed[:20]
        status['Q6'] = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    for b in ('Q5', 'Q6'):
        write_json(TESTS / f'{b.lower()}_smoke.json', dict(status=status[b], lock_sha256=locks[b]['lock_sha256'], temp_dir=str(tmp), **info))
    print(status, round(time.time() - t0)); print(info.get('traceback', ''))
    sys.exit(0 if all(v == 'passed' for v in status.values()) else 1)


if __name__ == '__main__':
    main()
