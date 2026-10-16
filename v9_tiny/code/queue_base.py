"""Shared detached-queue machinery for V9 blocks after Q1 (Q1's own driver is locked and unchanged)."""
import hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import v9paths
from v9paths import V9, RUNS, SMOKE, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json

PY = sys.executable


class Stop(Exception):
    pass


def lock_identity(block, files):
    import runtime
    ident = dict(block=block, frozen_implementation_sha256=runtime.implementation_hash(), files_sha256=files_hash(files),
                 data_manifest_sha256=digest(DATA / 'manifest.json'), exclusion_list_sha256=digest(DATA / 'evaluation_symbolic_hashes.txt'))
    ident['lock_sha256'] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident


class Queue:
    block = 'Q?'
    files = []

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
        obj['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); obj['block'] = self.block
        with self.log_path.open('a') as f:
            f.write(json.dumps(obj, default=str) + '\n')

    def status(self, status='running', **extra):
        live = {}
        if self.child is not None:
            p = RUNS / self.stage / 'live.json'
            if p.exists():
                try:
                    j = read_json(p); live = {k: j.get(k) for k in ('step', 'planned_steps', 'loss', 'elapsed_s', 'text', 'symbolic', 'ncp', 'selector')}
                except Exception:
                    pass
        write_json(RUNS / f'status_{self.block}.json', dict(block=self.block, status=status, stage=self.stage, driver_pid=os.getpid(),
                                               child_pid=self.child.pid if self.child and self.child.poll() is None else None,
                                               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start)), elapsed_s=time.time() - self.start,
                                               updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), live=live, **extra))

    def preflight(self, extra_tests=()):
        lock = read_json(SPEC / f'LOCK_{self.block}.json'); now = lock_identity(self.block, self.files)
        checks = {k: now[k] == lock[k] for k in ('frozen_implementation_sha256', 'files_sha256', 'data_manifest_sha256', 'exclusion_list_sha256', 'lock_sha256')}
        for name in (f'{self.block.lower()}_contracts.json',) + tuple(extra_tests if not SMOKE else ()):
            t = read_json(TESTS / name) if (TESTS / name).exists() else {}
            checks[name] = t.get('status') == 'passed' and t.get('lock_sha256') == lock['lock_sha256']
        if not all(checks.values()):
            raise RuntimeError(f'{self.block} preflight failed: ' + json.dumps(checks))
        self.lock = lock; return lock

    def wait_for(self, path, what):
        """Block until a prerequisite block has written its completed result (Q-blocks run in order on one GPU)."""
        self.stage = 'waiting_for_' + what; self.status('waiting')
        while True:
            if self.stopped:
                raise Stop()
            if Path(path).exists() and read_json(path).get('status') == 'completed':
                return read_json(path)
            time.sleep(60); self.status('waiting')

    def run(self, name, script, args, env=None):
        if self.stopped:
            raise Stop()
        self.stage = name; folder = RUNS / name; folder.mkdir(parents=True, exist_ok=True)
        command = [PY, str(script), *map(str, args)]
        self.log(stage=name, event='started', command=command); self.status()
        with open(folder / 'process.log', 'a') as f:
            self.child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=V9, env=dict(os.environ, **(env or {})))
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

    def train(self, name, script, args, steps):
        """Resumable training: completed runs are verified and skipped; interrupted or partial runs resume."""
        folder = RUNS / name; summary = folder / 'train_summary.json'
        if summary.exists():
            j = read_json(summary)
            if j['status'] == 'completed' and j['completed_steps'] == steps:
                self.log(stage=name, event='verified_completed'); return folder / 'checkpoint.pt'
            if j['status'] in ('interrupted_by_signal', 'partial_time_budget') and (folder / 'checkpoint.pt').exists():
                self.log(stage=name, event='resuming', prior_status=j['status'], from_step=j['completed_steps'])
            else:
                raise RuntimeError(f'{name} has summary status {j["status"]}; inspect before rerunning')
        if (folder / 'checkpoint.pt').exists():
            args = list(args) + ['--resume']
        self.run(name, script, args)
        j = read_json(summary)
        if j['status'] != 'completed' or j['completed_steps'] != steps:
            raise RuntimeError(f'{name} ended {j["status"]} at {j["completed_steps"]} updates; declared exposure not completed')
        return folder / 'checkpoint.pt'

    def once(self, name, script, args, out):
        if Path(out).exists():
            self.log(stage=name, event='verified_completed'); return read_json(out)
        self.run(name, script, args); return read_json(out)

    def execute(self):
        if SMOKE == (RUNS.resolve() == (V9 / 'runs').resolve()):
            raise SystemExit('smoke mode requires a temporary V9_RUNS directory, and production requires the default runs directory')
        write_json(RUNS / f'driver_{self.block}.json', dict(block=self.block, pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), python=PY))
        try:
            self.main()
        except Stop:
            self.status('stopped_by_user'); self.log(stage=self.stage, event='stopped')
        except Exception as exc:
            write_json(RUNS / f'failure_{self.block}.json', dict(error=repr(exc), traceback=traceback.format_exc(), stage=self.stage))
            self.status('failed', error=repr(exc)); self.log(stage=self.stage, event='failed', error=repr(exc)); raise
