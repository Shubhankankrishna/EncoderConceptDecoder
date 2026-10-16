"""Q4 queue (draft §7, §14): S3 on the promoted S2 dense model, per promoted seed: the lambda transplant curve,
S3a-dec and S3b (6,000 updates, same data order), S2 measurements recomputed, G3, and the model that goes forward.
If K8 was not promoted (G2.2(d)), S3a-dec only, on V9-K1, without G3's authority conditions."""
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, digest, read_json, write_json
from queue_base import Queue

Q4_FILES = [CODE / n for n in ('drive_q4.py', 'queue_base.py', 'train_s3.py', 's3_eval.py', 'register_s3.py', 'register_s2.py', 'train_s2.py', 'hncp.py', 'hgen.py',
                               'measure_s2.py', 'ceiling.py', 's6.py', 'lsl.py', 'v9data.py', 'v9paths.py')]
STEPS = 20 if SMOKE else 6000


class Q4(Queue):
    block = 'Q4'; files = Q4_FILES

    def main(self):
        if not SMOKE:
            self.wait_for(RUNS / 'Q3_RESULT.json', 'Q3')
        lock = self.preflight(extra_tests=('q4_smoke.json',)); self.log(stage='Q4', event='preflight_passed', lock=lock['lock_sha256'])
        import register_s3 as R
        promo = read_json(RUNS / 'S2_PROMOTION.json'); q3a = read_json(RUNS / 'Q3A_RESULT.json'); runs = q3a['runs']
        depth = max(1, q3a['consequences'].get('H_star', 0))  # the lookahead depth S5 will use (H*)
        arms = ('dec', 'joint') if promo['k8_promoted'] else ('dec',)
        per_arm = {a: [] for a in arms}; seeds = []
        for mdl in promo['models']:
            src = mdl['source']; ck = src['checkpoint']; s2seed = src['seed']; s3seed = mdl['s3_seed']
            teacher = read_json(RUNS / src['name'] / 'config.json')['teacher']
            if digest(teacher) != src['teacher_sha256']:
                raise RuntimeError('teacher changed (contract 9)')
            tag = f'S3_s{s3seed}'
            lam = RUNS / f'{tag}_lambda' / 'lambda.json'
            self.once(f'{tag}_lambda', CODE / 's3_eval.py', ['--source', ck, '--teacher', teacher, '--out', lam], lam)
            k1 = runs[f'K1/dense/{s2seed}']['measure']; s2m = read_json(src['measure'])
            row = dict(seed=s3seed, s2=src['name'], lambda_curve=read_json(lam), arms={})
            for arm in arms:
                name = f'{tag}_{arm}'
                out = self.train(name, CODE / 'train_s3.py', ['--source', ck, '--teacher', teacher, '--out', RUNS / name, '--arm', arm, '--steps', STEPS, '--seed', s3seed, '--compiled'], STEPS)
                meas = RUNS / name / 'measure.json'
                self.once(name + '_measure', CODE / 'measure_s2.py', ['--source', out, '--teacher', teacher, '--out', meas] + (['--quick'] if SMOKE else []), meas)
                g = R.g3(read_json(meas), s2m, read_json(k1), depth)
                per_arm[arm].append(g); row['arms'][arm] = dict(checkpoint=str(out), measure=str(meas), G3=g)
            seeds.append(row); write_json(RUNS / f'{tag}_G3.json', row)
        dec = R.decide(per_arm)
        forward = [dict(seed=r['seed'], checkpoint=r['arms'][dec['replaces_S2']]['checkpoint'] if dec['replaces_S2'] else m['source']['checkpoint'],
                        stage='S3 ' + dec['replaces_S2'] if dec['replaces_S2'] else 'S2', s2_source=m['source'], flag=m['flag']) for r, m in zip(seeds, promo['models'])]
        write_json(RUNS / 'S3_DECISION.json', dict(decision=dec, forward=forward, lookahead_depth=depth))
        write_json(RUNS / 'Q4_RESULT.json', dict(status='completed', seeds=seeds, decision=dec, forward=forward, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q4', event='completed')


if __name__ == '__main__':
    Q4().execute()
