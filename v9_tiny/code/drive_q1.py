"""Q1 queue (draft §5.2, §14): S1 foundations, two new seeds of V8's exact recipe, dev evaluation, G1, sealed S1 tests.

Detached and resumable: completed runs are verified and skipped; an interrupted run resumes from its checkpoint
with identical arguments (frozen train.py enforces this). Only frozen V8 code trains or evaluates.
"""
import argparse, json, os, signal, subprocess, sys, time, traceback
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json

PY = sys.executable
SEEDS = (901, 902)
ARMS = (('A_delayed', 0.1, False), ('A_alpha0', 0.0, False), ('A_flat', 0.0, True))
SEED811 = {'A_delayed': 'A_delayed_ncp', 'A_alpha0': 'A_alpha0', 'A_flat': 'A_flat'}
Q1_FILES = [CODE / n for n in ('drive_q1.py', 'v9paths.py', 'stats.py', 's1_eval.py')]
STEPS = 60 if SMOKE else 24000  # smoke runs use a temporary V9_RUNS directory
EVAL_N = 64 if SMOKE else 2048
SECONDS = 43200  # per-run wall ceiling passed to train.py; fixed so --resume sees identical arguments


def train_args(arm, seed, compiled):
    _, alpha, flat = next(a for a in ARMS if a[0] == arm)
    # V8 runs/A_*/config.json: stage A, 24000 steps, batch 64, context 192, mode foundation, core dense, --development-curves, compiled.
    args = ['--stage', 'A', '--out', RUNS / f'S1_{arm}_s{seed}', '--steps', STEPS, '--seconds', SECONDS, '--batch', 64, '--seed', seed,
            '--alpha', alpha, '--mode', 'foundation', '--context', 192, '--core', 'dense', '--development-curves']
    if flat:
        args += ['--flat']
    if compiled:
        args += ['--compiled']
    return [str(v) for v in args]


class Stop(Exception):
    pass


class Queue:
    def __init__(self):
        self.child = None; self.stopped = False; self.stage = 'preflight'; self.start = time.time()
        self.log_path = RUNS / 'queue.jsonl'
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop)

    def stop(self, *_):
        self.stopped = True
        if self.child and self.child.poll() is None:
            self.child.terminate()

    def log(self, **obj):
        obj['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with self.log_path.open('a') as f:
            f.write(json.dumps(obj) + '\n')

    def status(self, status='running', **extra):
        live = {}
        if self.child is not None:
            p = RUNS / self.stage / 'live.json'
            if p.exists():
                try:
                    j = read_json(p); live = {k: j.get(k) for k in ('step', 'planned_steps', 'loss', 'elapsed_s', 'text', 'symbolic', 'ncp')}
                except Exception:
                    pass
        write_json(RUNS / 'status.json', dict(block='Q1', status=status, stage=self.stage, driver_pid=os.getpid(),
                                               child_pid=self.child.pid if self.child and self.child.poll() is None else None,
                                               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start)), elapsed_s=time.time() - self.start,
                                               updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), live=live, **extra))

    def preflight(self):
        lock = read_json(SPEC / 'LOCK_Q1.json')
        import runtime
        checks = dict(frozen=runtime.implementation_hash() == lock['frozen_implementation_sha256'],
                      q1_files=files_hash(Q1_FILES) == lock['q1_files_sha256'],
                      manifest=digest(DATA / 'manifest.json') == lock['data_manifest_sha256'],
                      exclusion=digest(DATA / 'evaluation_symbolic_hashes.txt') == lock['exclusion_list_sha256'])
        contracts = read_json(TESTS / 'q1_contracts.json')
        checks['contracts'] = contracts['status'] == 'passed' and contracts['lock_sha256'] == lock['lock_sha256']
        if not SMOKE:  # the end-to-end smoke of this exact driver must have passed under the same lock
            smoke = read_json(TESTS / 'q1_smoke.json') if (TESTS / 'q1_smoke.json').exists() else {}
            checks['smoke'] = smoke.get('status') == 'passed' and smoke.get('lock_sha256') == lock['lock_sha256']
        if not all(checks.values()):
            raise RuntimeError('Q1 preflight failed: ' + json.dumps(checks))
        self.lock = lock
        return lock

    def run(self, name, script, args):
        if self.stopped:
            raise Stop()
        self.stage = name; folder = RUNS / name; folder.mkdir(parents=True, exist_ok=True)
        command = [PY, str(script), *map(str, args)]
        self.log(stage=name, event='started', command=command); self.status()
        with open(folder / 'process.log', 'a') as f:
            self.child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=V9)
            last = 0
            while self.child.poll() is None:
                time.sleep(2)
                if time.time() - last > 60:
                    self.status(); last = time.time()
            rc = self.child.returncode
        self.child = None; self.status()
        if self.stopped:
            self.log(stage=name, event='stopped_by_signal', exit_code=rc); raise Stop()
        if rc != 0:
            self.log(stage=name, event='failed', exit_code=rc)
            raise RuntimeError(f'{name} exited {rc}; see {folder}/process.log')
        self.log(stage=name, event='finished'); return True

    def train(self, arm, seed):
        name = f'S1_{arm}_s{seed}'; folder = RUNS / name; args = train_args(arm, seed, self.lock['compiled'])
        summary = folder / 'train_summary.json'
        if summary.exists():
            j = read_json(summary)
            if j['status'] == 'completed' and j['completed_steps'] == STEPS and j['seed'] == seed:
                self.log(stage=name, event='verified_completed', checkpoint_sha256=digest(folder / 'checkpoint.pt')); return folder / 'checkpoint.pt'
            if j['status'] in ('interrupted_by_signal', 'partial_time_budget') and (folder / 'checkpoint.pt').exists():
                self.log(stage=name, event='resuming', prior_status=j['status'], from_step=j['completed_steps'])
            else:
                raise RuntimeError(f'{name} has summary status {j["status"]} at {j["completed_steps"]} updates; inspect before rerunning')
        if (folder / 'checkpoint.pt').exists():
            args = args + ['--resume']
        self.run(name, FROZEN_CODE / 'train.py', args)
        j = read_json(summary)
        if j['status'] != 'completed' or j['completed_steps'] != STEPS:
            raise RuntimeError(f'{name} ended {j["status"]} at {j["completed_steps"]} updates; the declared exposure was not completed')
        write_json(folder / 'checkpoint_sha256.json', dict(sha256=digest(folder / 'checkpoint.pt')))
        return folder / 'checkpoint.pt'

    def once(self, name, script, args, out):
        if out.exists() and read_json(out).get('status', 'completed') in ('completed', 'passed'):
            self.log(stage=name, event='verified_completed'); return read_json(out)
        self.run(name, script, args); return read_json(out)

    def main(self):
        lock = self.preflight(); self.log(stage='Q1', event='preflight_passed', lock=lock['lock_sha256'])
        sources = {}
        for arm, _, _ in ARMS:
            for seed in SEEDS:
                name = f'S1_{arm}_s{seed}'
                src = self.train(arm, seed); sources[(arm, seed)] = src
                self.once(name + '_trained_contracts', FROZEN_CODE / 'revision_checks.py', ['--source', src, '--out', RUNS / (name + '_trained_contracts') / 'summary.json'],
                          RUNS / (name + '_trained_contracts') / 'summary.json')
                self.once(name + '_dev', FROZEN_CODE / 'evaluate.py', ['--source', src, '--out', RUNS / (name + '_dev') / 'summary.json', '--n', EVAL_N,
                                                                          '--names', 'standard,lookup,compose,pointer,hard10', '--split', 'dev', '--fast'], RUNS / (name + '_dev') / 'summary.json')
                if arm == 'A_delayed':
                    self.g1()
        self.g1(final=True)
        self.stage = 'S1_sealed'; self.status()
        seed811 = read_json(SPEC / 'SEED811.json')
        for arm, _, _ in ARMS:
            for seed in SEEDS:
                name = f'S1_{arm}_s{seed}_sealed'
                self.once(name, CODE / 's1_eval.py', ['--source', sources[(arm, seed)], '--out', RUNS / name / 'summary.json'], RUNS / name / 'summary.json')
            v8 = seed811[SEED811[arm]]
            if digest(v8['checkpoint']) != v8['checkpoint_sha256']:
                raise RuntimeError('seed-811 checkpoint changed: ' + arm)
            name = f'S1_{arm}_s811_sealed'
            self.once(name, CODE / 's1_eval.py', ['--source', v8['checkpoint'], '--out', RUNS / name / 'summary.json'], RUNS / name / 'summary.json')
        self.run('S1_analysis', CODE / 's1_eval.py', ['--analyze'])
        res = read_json(RUNS / 'S1_RESULTS.json')
        write_json(RUNS / 'Q1_RESULT.json', dict(status='completed', G1=read_json(RUNS / 'G1.json'), teachers=read_json(RUNS / 'S2_TEACHERS.json'),
                                                 H3=res['H3']['supported'], H4=res['H4']['supported'], H5=res['H5']['supported'], holm=res['holm'],
                                                 elapsed_s=time.time() - self.start, lock=lock['lock_sha256']))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q1', event='completed')

    def g1(self, final=False):
        """G1 (register §12.3): point estimates on dev, >= 80% standard and >= 95% lookup; not evaluate.competence()."""
        rows = {}
        for seed in SEEDS:
            p = RUNS / f'S1_A_delayed_s{seed}_dev' / 'summary.json'
            if p.exists():
                e = read_json(p); st = e['symbolic']['standard']; lk = e['symbolic']['lookup']
                rows[seed] = dict(standard=st['rate'], standard_successes=st['successes'], standard_n=st['n'], lookup=lk['rate'], lookup_n=lk['n'],
                                  passed=bool(st['rate'] >= .80 and lk['rate'] >= .95))
        write_json(RUNS / 'G1.json', dict(rule='dev point estimates: standard >= .80 and lookup >= .95 (register G1)', seeds=rows, final=final))
        if final:
            passing = [s for s in SEEDS if rows.get(s, {}).get('passed')]
            if len(passing) == 2:
                teachers = [f'S1_A_delayed_s{s}' for s in passing]; flag = 'independent teachers'
            elif len(passing) == 1:
                teachers = [f'S1_A_delayed_s{passing[0]}', 'A_delayed_ncp_s811']; flag = 'independent teachers (one new seed failed G1; second teacher is seed 811)'
            else:
                teachers = ['A_delayed_ncp_s811', 'A_delayed_ncp_s811']; flag = 'non-independent teachers (both new seeds failed G1); S1 reproduction gap reported'
            paths = {t: (str(RUNS / t / 'checkpoint.pt') if t.startswith('S1_') else read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']) for t in teachers}
            write_json(RUNS / 'S2_TEACHERS.json', dict(teachers=teachers, flag=flag, checkpoints=paths, sha256={t: digest(p) for t, p in paths.items()}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    if a.dry_run:
        for arm, _, _ in ARMS:
            for seed in SEEDS:
                print(' '.join(train_args(arm, seed, True)))
        return
    if SMOKE == (RUNS.resolve() == (V9 / 'runs').resolve()):
        raise SystemExit('smoke mode requires a temporary V9_RUNS directory, and production requires the default runs directory')
    write_json(RUNS / 'driver.json', dict(block='Q1', pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), python=PY))
    q = Queue()
    try:
        q.main()
    except Stop:
        q.status('stopped_by_user'); q.log(stage=q.stage, event='stopped')
    except Exception as exc:
        write_json(RUNS / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), stage=q.stage))
        q.status('failed', error=repr(exc)); q.log(stage=q.stage, event='failed', error=repr(exc)); raise


if __name__ == '__main__':
    main()
