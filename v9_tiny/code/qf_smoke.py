"""End-to-end smoke of the QF driver on stand-in results (a K = 8 fixture as every S4 arm, a zero LoRA adapter, synthetic
S5 cells, synthetic S6 measures); sealed reads are replaced by dev reads in smoke mode."""
import json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import CODE, TESTS, SPEC, read_json, write_json
from q1_contracts import runs_snapshot


def stand_in(runs):
    import hncp, baselines
    from runtime import atomic_save
    t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']; torch.manual_seed(1); hs, _ = hncp.build(t, 'dense', 8, 'cpu')
    d = runs / 'S2_K8_dense_s911'; d.mkdir(parents=True); hncp.save(d / 'checkpoint.pt', hs, dict(config={}))
    baselines.add_lora(hs); atomic_save(runs / 'rft_adapter.pt', dict(method='rft', state=baselines.lora_state(hs)))
    write_json(runs / 'Q5_RESULT.json', dict(status='completed', concept_branch=True, rh_allowed=True, seeds=[dict(seed=911, source=str(d / 'checkpoint.pt'),
               arms=dict(reference=dict(greedy='x'), EI=dict(checkpoint=str(d / 'checkpoint.pt')), rft=dict(adapter=str(runs / 'rft_adapter.pt'))))]))
    (runs / 'S5').mkdir(); rng = np.random.RandomState(0)
    for c in ('sealed_RH1_hard10_16x', 'sealed_token_tree_hard10_16x', 'sealed_answerloop_gradcuit_hard10_16x', 'sealed_answerloop_random_hard10_16x', 'sealed_RH1grounded_hard10_16x'):
        write_json(runs / 'S5' / f's911_{c}.json', dict(success=[bool(v) for v in rng.rand(64) < .5]))
    for arm in ('lsl', 'open'):
        p = runs / f'S6_s931_{arm}_P3'; p.mkdir(); write_json(p / 'measure.json', dict(symbolic=dict(standard=dict(success=[bool(v) for v in rng.rand(64) < .4]))))


def main():
    before = runs_snapshot(); tmp = Path(tempfile.mkdtemp(prefix='v9_qf_smoke_')); t0 = time.time(); status = 'failed'; info = {}; lock = read_json(SPEC / 'LOCK_QF.json')
    try:
        runs = tmp / 'runs'; runs.mkdir(); stand_in(runs); env = dict(os.environ, V9_RUNS=str(runs), V9_SMOKE='1')
        r = subprocess.run([sys.executable, str(CODE / 'drive_qf.py')], cwd=tmp, env=env, capture_output=True, text=True, timeout=7200)
        if r.returncode:
            f = runs / 'failure_QF.json'; raise AssertionError(r.stderr[-3000:] + (f.read_text()[-3000:] if f.exists() else ''))
        res = read_json(runs / 'FINAL_RESULTS.json'); s5 = res['families']['S5']
        assert all('p_holm' in s5[k] for k in ('L1', 'H14', 'H16')) and 'p' in res['families']['S4']['L3'] and 'p' in res['families']['S6']['H12']
        assert res['families']['S3']['H11'].get('status', '').startswith('missing') and (runs / 'FINAL_REPORT.md').exists()
        assert (runs / 'S4_SEALED_PROCEDURE.json').stat().st_mtime < min(p.stat().st_mtime for p in (runs / 'S4F').glob('*.json'))
        assert not (SPEC / 'S4_SEALED_PROCEDURE.json').exists()
        info = dict(report=(runs / 'FINAL_REPORT.md').read_text()[:1500])
        after = runs_snapshot(); assert not [k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k]
        status = 'passed'
    except Exception:
        info['traceback'] = traceback.format_exc()
    write_json(TESTS / 'qf_smoke.json', dict(status=status, lock_sha256=lock['lock_sha256'], temp_dir=str(tmp), seconds=time.time() - t0, **info))
    print(status); print(info.get('traceback', '')); sys.exit(status != 'passed')


if __name__ == '__main__':
    main()
