"""Q5 queue (draft §8, §14): S4 per promoted seed: the nodes/s and iteration-time pilot (I15 sizing rule), expert
iteration over codes (iteration 0 + 4), the three baselines matched in problems and verifier calls, development greedy
evaluation of every arm, and the GradCuit lr / norm-bound pilot. Concept branches run only if G2.1 did not fail and
K8 was promoted (§6.8, G2.2(d)); otherwise S4 runs the token baselines only."""
import time
import v9paths
from v9paths import RUNS, SMOKE, CODE, read_json, write_json
from queue_base import Queue

Q5_FILES = [CODE / n for n in ('drive_q5.py', 'queue_base.py', 'csearch.py', 'train_s4.py', 'baselines.py', 'kernels.py', 's4_eval.py', 'hncp.py', 'hgen.py',
                               'measure_s2.py', 'ceiling.py', 'train_s2.py', 'v9paths.py')]


class Q5(Queue):
    block = 'Q5'; files = Q5_FILES

    def main(self):
        if not SMOKE:
            self.wait_for(RUNS / 'Q4_RESULT.json', 'Q4')
        lock = self.preflight(extra_tests=('q5_smoke.json',)); self.log(stage='Q5', event='preflight_passed', lock=lock['lock_sha256'])
        s3 = read_json(RUNS / 'S3_DECISION.json'); q3a = read_json(RUNS / 'Q3A_RESULT.json'); cons = q3a['consequences']; promo = read_json(RUNS / 'S2_PROMOTION.json')
        concept = bool(cons.get('concept_branches_run', True)) and promo['k8_promoted']
        H = max(1, cons.get('H_star', 0)); rh = int(bool(cons.get('RH1_allowed', False)))
        result = dict(concept_branch=concept, H=H, rh_allowed=bool(rh), rh_label='primary' if rh and not cons.get('RH1_exploratory') else 'exploratory', seeds=[])
        sizing = None
        for fw in s3['forward']:
            src = fw['checkpoint']; tag = f'S4_s{fw["seed"]}'; row = dict(seed=fw['seed'], source=src, stage=fw['stage'], arms={})
            if sizing is None:
                p = RUNS / 'S4_pilot.json'; pil = self.once('S4_pilot', CODE / 's4_eval.py', ['pilot', '--source', src, '--H', H, '--out', p], p)
                sizing = dict(pil['sizing']); write_json(RUNS / 'S4_SIZING.json', dict(sizing, pilot=str(p), rule='I15'))
                if SMOKE:
                    sizing = dict(B_train=4, n=8, n0=8)
            ref = RUNS / f'{tag}_reference_greedy.json'
            self.once(f'{tag}_reference_greedy', CODE / 's4_eval.py', ['greedy', '--source', src, '--mode', 'codes' if promo['k8_promoted'] else 'free', '--out', ref], ref)
            row['arms']['reference'] = dict(greedy=str(ref))
            iters = 1 if SMOKE else 4
            ei = RUNS / f'{tag}_EI'
            if concept:
                ck = self.train(f'{tag}_EI', CODE / 'train_s4.py', ['--source', src, '--out', ei, '--seed', fw['seed'], '--H', H, '--budget', sizing['B_train'], '--iterations', iters,
                                                                    '--n0', sizing['n0'], '--n', sizing['n'], '--rh-allowed', rh], iters + 1)
                g = RUNS / f'{tag}_EI_greedy.json'; self.once(f'{tag}_EI_greedy', CODE / 's4_eval.py', ['greedy', '--source', ck, '--mode', 'codes', '--out', g], g)
                row['arms']['EI'] = dict(checkpoint=str(ck), greedy=str(g))
            for method in ('rft', 'grpo') + (('reinforce',) if concept else ()):
                name = f'{tag}_{method}'; sched = ei if concept else ('default-smoke' if SMOKE else 'default')   # I16 default schedule without EI
                self.train(name, CODE / 'baselines.py', ['--source', src, '--out', RUNS / name, '--method', method, '--ei', sched, '--seed', fw['seed']], iters + 1 if concept else (2 if SMOKE else 5))
                g = RUNS / f'{name}_greedy.json'
                self.once(f'{name}_greedy', CODE / 's4_eval.py', ['greedy', '--source', src, '--mode', 'codes' if method == 'reinforce' else 'free', '--adapter', RUNS / name / 'adapter.pt', '--out', g], g)
                row['arms'][method] = dict(adapter=str(RUNS / name / 'adapter.pt'), greedy=str(g))
            if concept:
                gp = RUNS / f'{tag}_gradcuit_pilot.json'; self.once(f'{tag}_gradcuit_pilot', CODE / 's4_eval.py', ['gradcuit', '--source', str(ei / 'checkpoint.pt'), '--out', gp], gp)
                row['gradcuit_pilot'] = str(gp)
            result['seeds'].append(row); write_json(RUNS / f'{tag}_summary.json', row)
        write_json(RUNS / 'Q5_RESULT.json', dict(status='completed', **result, sizing=sizing, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q5', event='completed')


if __name__ == '__main__':
    Q5().execute()
