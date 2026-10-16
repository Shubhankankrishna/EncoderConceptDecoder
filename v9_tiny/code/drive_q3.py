"""Q3 queue (draft §6.7, §14): S2 Tier A (dense) and Tier B (recurrent), two seeds, the two V8-schedule replays,
all S2 measurements, the register decisions and promotion. Waits for Q1 (the S2 teachers are the new A_delayed seeds)."""
import json, sys, time
import numpy as np
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, V8, digest, read_json, write_json
from queue_base import Queue, Stop

Q3_FILES = [CODE / n for n in ('drive_q3.py', 'queue_base.py', 'train_s2.py', 'hncp.py', 'hgen.py', 'measure_s2.py', 'register_s2.py', 'ceiling.py', 'export_k1.py', 'v9paths.py')]
STEPS = 40 if SMOKE else 12000
REPLAY_STEPS = 20 if SMOKE else 6000
SEEDS = (911, 912)
# (name, core, K, m, p_cl): Tier A dense, Tier B recurrent (draft §6.7).
ARMS = [('V8repro', 'dense', 1, 2, 0.), ('K1', 'dense', 1, 2, .25), ('K8', 'dense', 8, 2, .25), ('K1', 'recurrent', 1, 2, .25), ('K8', 'recurrent', 8, 2, .25)]
CONTINGENCY = [('K8m1', 'dense', 8, 1, .25), ('K8mK', 'dense', 8, 8, .25), ('K4', 'dense', 4, 2, .25)]  # move in if G2.2 fails while G2.1 passes (§6.8)


def run_name(arm, core, seed):
    return f'S2_{arm}_{"dense" if core == "dense" else "rec"}_s{seed}'


class Q3(Queue):
    block = 'Q3'; files = Q3_FILES

    def arm(self, arm, core, k, m, p, seed, teacher, steps=None):
        name = run_name(arm, core, seed); steps = steps or STEPS
        args = ['--teacher', teacher['checkpoint'], '--out', RUNS / name, '--core', core, '--k', k, '--m', m, '--p-cl', p, '--steps', steps, '--seed', seed, '--compiled']
        ck = self.train(name, CODE / 'train_s2.py', args, steps)
        cfg = read_json(RUNS / name / 'config.json')
        if cfg['teacher_sha256'] != teacher['sha256']:
            raise RuntimeError(f'{name}: teacher SHA mismatch (contract 9)')
        meas = RUNS / name / 'measure.json'
        self.once(name + '_measure', CODE / 'measure_s2.py', ['--source', ck, '--teacher', teacher['checkpoint'], '--out', meas] + (['--quick'] if SMOKE else []), meas)
        return dict(name=name, checkpoint=str(ck), measure=str(meas), teacher=teacher['name'], teacher_sha256=teacher['sha256'], seed=seed, core=core, k=k, m=m, p_cl=p)

    def replay(self, core):
        """V8-schedule replay: K = 1, closed loop off, 6,000 updates, seed 811, V8's own teacher (implementation check)."""
        v8t = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']; name = f'S2_replay_{core}'
        args = ['--teacher', v8t['checkpoint'], '--out', RUNS / name, '--core', core, '--k', 1, '--p-cl', 0., '--steps', REPLAY_STEPS, '--seed', 811, '--compiled']
        ck = self.train(name, CODE / 'train_s2.py', args, REPLAY_STEPS)
        sysck = RUNS / name / 'system_checkpoint.pt'
        if not sysck.exists():
            self.run(name + '_export', CODE / 'export_k1.py', ['--source', ck, '--out', sysck])
        dev = RUNS / (name + '_dev') / 'summary.json'
        self.once(name + '_dev', FROZEN_CODE / 'evaluate.py', ['--source', sysck, '--out', dev, '--n', 64 if SMOKE else 2048, '--names', 'standard,lookup,compose,pointer,hard10', '--split', 'dev', '--fast'], dev)
        saved = read_json(V8 / 'runs' / f'B_refit_{core}_dev' / 'summary.json'); now = read_json(dev)
        r = dict(v9={k: v['rate'] for k, v in now['symbolic'].items()}, v8_saved={k: v['rate'] for k, v in saved['symbolic'].items()},
                 text_v9={k: v['nll'] for k, v in now['text'].items()}, text_v8_saved={k: v['nll'] for k, v in saved['text'].items()},
                 note='implementation check, expected within MPS nondeterminism; not a gate')
        write_json(RUNS / f'S2_REPLAY_{core}.json', r); return r

    def tier(self, core, teachers):
        runs = {}
        for i, seed in enumerate(SEEDS):
            for arm, c, k, m, p in ARMS:
                if c == core:
                    runs[(arm, core, seed)] = self.arm(arm, core, k, m, p, seed, teachers[i])
        rep = self.replay(core)
        import register_s2
        pairs = [(runs[('K8', core, s)]['measure'], runs[('K1', core, s)]['measure']) for s in SEEDS]
        self.stage = f'S2_register_{core}'; self.status()
        reg = register_s2.decide(pairs, RUNS / f'S2_REGISTER_{core}.json')
        self.log(stage=self.stage, event='decided', consequences=reg['consequences'], combined={k: v['zone'] for k, v in reg['combined'].items()})
        return runs, rep, reg

    def main(self):
        """Tier A first (user decision 2026-10-07, I13): Q3a = dense tier, replay, register, promotion -> Q3A_RESULT.json;
        then Q2 runs; Q3b = recurrent tier, its replay and register, and any §6.8 contingency arms -> Q3_RESULT.json."""
        if not SMOKE:
            self.wait_for(RUNS / 'Q1_RESULT.json', 'Q1')
        lock = self.preflight(extra_tests=('q3_smoke.json',)); self.log(stage='Q3', event='preflight_passed', lock=lock['lock_sha256'])
        tj = read_json(RUNS / 'S2_TEACHERS.json')
        teachers = []
        for t in tj['teachers']:
            path = tj['checkpoints'][t]
            if digest(path) != tj['sha256'][t]:
                raise RuntimeError('teacher changed: ' + t)
            teachers.append(dict(name=t, checkpoint=path, sha256=tj['sha256'][t]))
        dense, rep_d, reg_d = self.tier('dense', teachers)
        promo = self.promote(dense, reg_d)
        write_json(RUNS / 'S2_PROMOTION.json', promo)
        write_json(RUNS / 'Q3A_RESULT.json', dict(status='completed', tier='A (dense)', teacher_flag=tj['flag'], runs={'/'.join(map(str, k)): v for k, v in dense.items()}, replay=rep_d,
                                                  consequences=reg_d['consequences'], promotion=promo, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.log(stage='Q3', event='tier_A_completed')
        if not SMOKE:
            self.wait_for(RUNS / 'Q2_RESULT.json', 'Q2')
        rec, rep_r, reg_r = self.tier('recurrent', teachers)
        extra = []
        if reg_d['consequences']['prior_not_deployable']:
            self.log(stage='Q3', event='contingency', reason='G2.2 fails while G2.1 passes: m-arms and K = 4 move into the mandatory queue (§6.8)')
            for arm, c, k, m, p in CONTINGENCY:
                extra.append(self.arm(arm, c, k, m, p, SEEDS[0], teachers[0]))
        runs = dict(dense); runs.update(rec)
        write_json(RUNS / 'Q3_RESULT.json', dict(status='completed', teacher_flag=tj['flag'], runs={'/'.join(map(str, k)): v for k, v in runs.items()}, contingency=extra,
                                                 replay=dict(dense=rep_d, recurrent=rep_r), consequences=dict(dense=reg_d['consequences'], recurrent=reg_r['consequences']),
                                                 promotion=promo, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q3', event='completed')

    def promote(self, runs, reg):
        """Dense model to S3-S5 (§6.7). G2.2(d) decides K8 vs K1 per seed; a seed split proceeds with the passing seed's model."""
        per = [ps['G2.2d']['zone'] for ps in reg['per_seed']]; cons = reg['consequences']
        models = []
        passing = [s for s, z in zip(SEEDS, per) if z == 'pass']
        for i, s in enumerate(SEEDS):
            if len(passing) == 2:
                src = runs[('K8', 'dense', s)]; flag = None
            elif len(passing) == 1:
                src = runs[('K8', 'dense', passing[0])]; flag = 'seed-dependent G2.2(d): both S3 seeds start from the passing seed\'s K8 model'
            else:
                src = runs[('K1', 'dense', s)]; flag = 'K8 not promoted (G2.2(d)): V9-K1, S3a-dec only, token methods only in S4/S5'
            models.append(dict(seed=s, s3_seed=921 + i, source=src, flag=flag))
        return dict(k8_promoted=len(passing) > 0, models=models, consequences=cons, g22d_per_seed=per)


if __name__ == '__main__':
    Q3().execute()
