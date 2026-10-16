"""Q2 queue (draft §10, §14): S6-base on the S1 anchors: LSL-scaled, always-open and VQ-only, 3 phases, 2 seeds.
G6.0 before each phase fixes the phase label; P2 is colour-count or shifted (whichever passes G6.0; if both, the lower
mean old-task hit rate, colour-count on a tie; if neither, colour-count, labelled non-separable)."""
import json, time
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, CODE, digest, read_json, write_json
from queue_base import Queue

Q2_FILES = [CODE / n for n in ('drive_q2.py', 'queue_base.py', 's6.py', 's6_pilot.py', 'lsl.py', 'v9data.py', 'register_s6.py', 'register_s2.py', 'measure_s2.py', 'v9paths.py')]
STEPS = 10 if SMOKE else 2000
SEEDS = (931, 932)
ARMS = ('lsl', 'open', 'vq')
ALL_MODES = 'text,colourcount,shifted,bind'


class Q2(Queue):
    block = 'Q2'; files = Q2_FILES

    def measure(self, name, source, pre=None, telemetry=None, earlier=()):
        out = RUNS / name / 'measure.json'
        args = ['measure', '--source', source, '--modes', ALL_MODES, '--out', out]
        if pre:
            args += ['--pre', pre]
        if telemetry:
            args += ['--telemetry-mode', telemetry, '--earlier', ','.join(earlier)]
        return self.once(name + '_measure', CODE / 's6.py', args, out)

    def g60(self, name, source, mode, earlier, seed):
        out = RUNS / name / 'g60.json'
        return self.once(name, CODE / 's6.py', ['g60', '--source', source, '--mode', mode, '--earlier', ','.join(earlier), '--seed', seed, '--out', out], out)

    def main(self):
        if not SMOKE:
            self.wait_for(RUNS / 'Q3A_RESULT.json', 'Q3 tier A')  # one GPU: S2 dense tier, then Q2, then the S2 recurrent tier (I13)
        lock = self.preflight(extra_tests=('q2_smoke.json',)); self.log(stage='Q2', event='preflight_passed', lock=lock['lock_sha256'])
        import register_s6 as R
        # §10.7 timing pilot as Q2's first stage (the GPU is occupied by Q1/Q3a until here; I12). Timing only, no decision.
        pilot = (RUNS / 'S6_pilot.json') if SMOKE else (v9paths.TESTS / 's6_pilot.json')
        self.once('S6_pilot', CODE / 's6_pilot.py', ['--out', pilot], pilot)
        tj = read_json(RUNS / 'S2_TEACHERS.json'); anchors = [tj['checkpoints'][t] for t in tj['teachers']]
        result = dict(seeds={})
        for i, seed in enumerate(SEEDS):
            anchor = anchors[i]; tag = f'S6_s{seed}'
            base = self.measure(f'{tag}_anchor', anchor)
            prev = {arm: anchor for arm in ARMS}; prev_eval = {arm: base for arm in ARMS}; earlier = []; rows = {}; g60s = {}
            for phase in (1, 2, 3):
                if phase == 1:
                    mode = 'text'; g60s[1] = self.g60(f'{tag}_P1_g60', anchor, 'text', [], seed)
                elif phase == 2:
                    cands = {m: self.g60(f'{tag}_P2_g60_{m}', prev['lsl'], m, earlier, seed) for m in ('colourcount', 'shifted')}
                    passing = [m for m, g in cands.items() if g['passed']]
                    mode = min(passing, key=lambda m: (cands[m]['mean'], m != 'colourcount')) if passing else 'colourcount'
                    g60s[2] = dict(candidates=cands, chosen=mode, positive_test_p1_only=not passing)
                else:
                    mode = 'bind'; g60s[3] = self.g60(f'{tag}_P3_g60', prev['lsl'], 'bind', earlier, seed)
                post = {}
                for arm in ARMS:
                    name = f'{tag}_{arm}_P{phase}'
                    ck = self.train(name, CODE / 's6.py', ['train', '--source', prev[arm], '--out', RUNS / name, '--arm', arm, '--mode', mode, '--phase', phase, '--steps', STEPS, '--seed', seed], STEPS)
                    post[arm] = self.measure(name, ck, pre=prev[arm], telemetry=mode if arm == 'lsl' else None, earlier=earlier)
                    prev[arm] = str(ck)
                lsl_rows = R.phase_rows(mode, earlier, prev_eval['lsl'], post['lsl'], dict(open=post['open'], vq=post['vq']), post['lsl'].get('forced_open_new'))
                other = {arm: R.phase_rows(mode, earlier, prev_eval[arm], post[arm], {}) for arm in ('open', 'vq')}
                rows[phase] = dict(mode=mode, lsl=lsl_rows, open=other['open'], vq=other['vq'], function_kl={arm: post[arm].get('function_kl') for arm in ARMS},
                                   telemetry=post['lsl'].get('telemetry'))
                if phase == 3:
                    rows[phase]['H12'] = R.h12(post['lsl'], post['open'])
                prev_eval = post; earlier = earlier + [mode]
                write_json(RUNS / f'{tag}_decisions.json', dict(g60=g60s, phases=rows))
            result['seeds'][seed] = dict(g60=g60s, phases=rows)
        write_json(RUNS / 'Q2_RESULT.json', dict(status='completed', **result, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q2', event='completed')


if __name__ == '__main__':
    Q2().execute()
