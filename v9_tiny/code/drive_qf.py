"""QF (final): sealed greedy evaluation of the S4 arms under a procedure fixed before any sealed read (L3), then the
final analysis: stage-wise Holm families, per-seed and pooled paired results, FINAL_REPORT.md."""
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, SPEC, read_json, write_json, digest, DATA
from queue_base import Queue

QF_FILES = [CODE / n for n in ('drive_qf.py', 'queue_base.py', 's4_sealed.py', 'final_analysis.py', 's4_eval.py', 's5_run.py', 'baselines.py', 'stats.py', 'register_s2.py', 'hncp.py', 'hgen.py', 'v9paths.py')]


class QF(Queue):
    block = 'QF'; files = QF_FILES

    def main(self):
        if not SMOKE:
            self.wait_for(RUNS / 'Q6_RESULT.json', 'Q6')
        lock = self.preflight(extra_tests=('qf_smoke.json',)); self.log(stage='QF', event='preflight_passed', lock=lock['lock_sha256'])
        q5 = read_json(RUNS / 'Q5_RESULT.json'); (RUNS / 'S4F').mkdir(exist_ok=True)
        proc = (RUNS if SMOKE else SPEC) / 'S4_SEALED_PROCEDURE.json'
        if not proc.exists():
            write_json(proc, dict(arms={r['seed']: {k: v for k, v in r['arms'].items()} for r in q5['seeds']}, decoding='greedy; selector-argmax (EI, REINFORCE, reference) or code-free (RFT, GRPO)',
                                  suites=dict(hard10=512, hard14=256), sealed=dict(hard10=digest(DATA / 'v9test_hard10.npz'), hard14=digest(DATA / 'v9test_hard14.npz')),
                                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        for r in q5['seeds']:
            for arm, v in r['arms'].items():
                src = v.get('checkpoint') or r['source']; mode = 'free' if arm in ('rft', 'grpo') or (arm == 'reference' and not q5['concept_branch']) else 'codes'
                out = RUNS / 'S4F' / f's{r["seed"]}_{arm}.json'
                self.once(f'S4F_s{r["seed"]}_{arm}', CODE / 's4_sealed.py', ['--source', src, '--mode', mode, '--out', out] + (['--adapter', v['adapter']] if v.get('adapter') else []), out)
        self.run('final_analysis', CODE / 'final_analysis.py', [])
        write_json(RUNS / 'QF_RESULT.json', dict(status='completed', results=str(RUNS / 'FINAL_RESULTS.json'), report=str(RUNS / 'FINAL_REPORT.md'), lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='QF', event='completed')


if __name__ == '__main__':
    QF().execute()
