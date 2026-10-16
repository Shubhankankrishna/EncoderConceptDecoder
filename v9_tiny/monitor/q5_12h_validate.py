"""CPU-only contracts and isolated end-to-end smoke for the reduced Q5 queue."""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

os.environ['V8_DEVICE'] = 'cpu'
os.environ['V9_SMOKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import q5_12h as q
import numpy as np
import torch
import baselines
import hncp
import hgen
import train_s4
import q5_contracts as old
from runtime import atomic_save
from s4_eval import greedy_eval

TESTS = []


def check(name, cond, **kw):
    assert cond, (name, kw)
    TESTS.append(dict(name=name, passed=True, **kw))
    print(json.dumps(TESTS[-1]), flush=True)


def model_diff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def contracts(hs):
    old.baselines_check(hs); old.metrics()
    TESTS.extend(old.tests)
    items = train_s4.fresh_problems(2, 123456)
    answers = []
    for j, it in enumerate(items):
        for k in range(8):
            # Controlled complete groups with nonzero gradient for an optimizer parity test.
            response = it.response.copy()
            if k % 2:
                response[-4] = 99
            answers.append((np.r_[it.prompt, response], np.full(64, -1), not bool(k % 2), it.P, j))
    data = q.fit_data(answers)
    check('Complete groups use the original standardized within-group advantages', len(data) == 16 and [a[4] for a in data[:8]] == [1., -1.] * 4)
    check('Constant-reward groups have no learning signal', not q.fit_data([(*a[:2], True, *a[3:]) for a in answers]))
    try:
        q.fit_data(answers[:-1]); good = False
    except AssertionError:
        good = True
    check('Incomplete groups are rejected', good)
    torch.manual_seed(921); base = copy.deepcopy(hs); baselines.add_lora(base)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    np.random.seed(101)
    ra = baselines.train_round(a, 'grpo', baselines.add_lora(a), answers, bs=8)
    np.random.seed(101); progress = {}
    rb = q.fit_resumable(b, baselines.add_lora(b), answers, progress, lambda opt: None, time.time() + 600, bs=8, micro=2)
    delta = model_diff(baselines.lora_state(a), baselines.lora_state(b))
    check('Gradient accumulation preserves the original GRPO objective and Adam updates', delta < 5e-5 and abs(ra['loss'] - rb['loss']) < 1e-4,
          maximum_parameter_difference=delta, loss_difference=abs(ra['loss'] - rb['loss']))
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    np.random.seed(107)
    q.fit_resumable(a, baselines.add_lora(a), answers, {}, lambda opt: None, time.time() + 600, bs=8, micro=2)
    saved = {}; progress = {}; np.random.seed(107)

    class Interrupted(Exception):
        pass

    def save(opt):
        saved.update(progress=copy.deepcopy(progress), optimizer=copy.deepcopy(opt.state_dict()), numpy=np.random.get_state())
        raise Interrupted()

    try:
        q.fit_resumable(b, baselines.add_lora(b), answers, progress, save, time.time() + 600, bs=8, micro=2)
    except Interrupted:
        pass
    progress = saved['progress']; progress['optimizer'] = saved['optimizer']; np.random.set_state(saved['numpy'])
    q.fit_resumable(b, baselines.add_lora(b), answers, progress, lambda opt: None, time.time() + 600, bs=8, micro=2)
    delta = model_diff(baselines.lora_state(a), baselines.lora_state(b))
    check('Optimizer checkpoint and resume reproduce uninterrupted fitting', delta == 0, maximum_parameter_difference=delta)
    b = copy.deepcopy(base); state0 = baselines.lora_state(b); saved = []
    r = q.fit_resumable(b, baselines.add_lora(b), answers, {}, lambda opt: saved.append(True), time.time() - 1, bs=8, micro=2)
    check('Expired fitting deadline takes no optimizer step and saves a usable boundary', not r['completed'] and saved and model_diff(state0, baselines.lora_state(b)) == 0)
    full = train_s4.fresh_problems(512, 178100); short = train_s4.fresh_problems(128, 178100)
    check('Reduced training exposure is a predetermined prefix, independent of outcomes', all(np.array_equal(a.prompt, b.prompt) for a, b in zip(full, short)))


def smoke(hs, tmp, prodroot):
    q.ROOT = tmp / 'runs'; q.ROOT.mkdir()
    source = tmp / 'fixture.pt'; hncp.save(source, hs, dict(config=dict(fixture=True)))
    p = copy.deepcopy(q.read(prodroot / 'PROTOCOL.json')); p['source'] = str(source)
    p.update(optimizer_batch=8, gradient_microbatch=2, sampling_problem_batch=1)
    deadline = time.time() + 900
    row = q.train_worker(p, q.ROOT / 'grpo/round0', 0, 2, deadline)
    check('Isolated CPU smoke samples, fits and saves one complete GRPO round', row['status'] == 'completed' and row['verifier_calls'] == 16)
    ck = torch.load(q.ROOT / 'grpo/round0/adapter.pt', map_location='cpu', weights_only=False)
    check('Checkpoint contains adapter, generator and framework RNG state', len(ck['state']) == 12 and ck['generator'].numel() and ck['torch_rng'].numel())
    files = {str(f): q.digest(f) for f in (q.ROOT / 'grpo/round0').iterdir()}
    again = q.train_worker(p, q.ROOT / 'grpo/round0', 0, 2, deadline)
    check('A completed round is reused without rewriting its outputs', again == row and all(q.digest(f) == h for f, h in files.items()))
    row2 = q.train_worker(p, q.ROOT / 'grpo/round1', 1, 2, deadline)
    check('Second round resumes the prior adapter and completes the two-round workflow', row2['status'] == 'completed')
    ev = greedy_eval(str(source), 'free', str(q.ROOT / 'grpo/round1/adapter.pt'), n=2)
    check('Saved adapter reloads and evaluates on development suites', set(ev) == {'hard10', 'standard'} and all(r['n'] == 2 for r in ev.values()))
    # Simulate interruption immediately after one complete sampling group was persisted.
    uninterrupted = q.ROOT / 'sample_full'; interrupted = q.ROOT / 'sample_resume'
    q.train_worker(p, uninterrupted, 0, 2, deadline)
    orig = baselines.sample_answers; calls = [0]

    class Interrupted(Exception):
        pass

    def sample(*a, **kw):
        calls[0] += 1
        if calls[0] == 2:
            raise Interrupted()
        return orig(*a, **kw)

    baselines.sample_answers = sample
    try:
        q.train_worker(p, interrupted, 0, 2, deadline)
    except Interrupted:
        pass
    finally:
        baselines.sample_answers = orig
    q.train_worker(p, interrupted, 0, 2, deadline)
    a = torch.load(uninterrupted / 'adapter.pt', map_location='cpu', weights_only=False)
    b = torch.load(interrupted / 'adapter.pt', map_location='cpu', weights_only=False)
    check('Sampling interruption resumes without duplicate groups and yields the same adapter/RNG',
          model_diff(a['state'], b['state']) == 0 and torch.equal(a['generator'], b['generator']))
    q.ROOT = prodroot


def queue_contracts(tmp, prodroot):
    q.ROOT = tmp / 'control'; q.ROOT.mkdir()
    p = copy.deepcopy(q.read(prodroot / 'PROTOCOL.json')); q.write(q.ROOT / 'PROTOCOL.json', p)
    s = q.Supervisor(); calls = []
    # Only the explicit saved-memory code is resumable without a new diagnosis.
    def once(name, args, deadline):
        calls.append(name)
        q.write(q.ROOT / 'grpo/round0/live.json', dict(next_problem=4, phase='sampling'))
        return 75 if len(calls) == 1 else True
    s._run_worker_once = once
    check('Supervisor reuses saved memory handoffs sequentially', s.run('grpo_round0', ['train', '--round', '0'], time.time() + 10) is True and len(calls) == 2)
    calls.clear()
    def failed(*args):
        calls.append('failed'); raise RuntimeError('concrete worker failure')
    s._run_worker_once = failed
    try:
        s.run('grpo_round0', ['train', '--round', '0'], time.time() + 10); rejected = False
    except RuntimeError:
        rejected = True
    check('Deterministic worker failures are not blindly retried', rejected and len(calls) == 1)
    check('An expired stage deadline launches no worker', s.run('expired', ['train'], time.time() - 1) is False and len(calls) == 1)
    q.ROOT = prodroot


def main():
    prodroot = q.ROOT; tmp = Path(tempfile.mkdtemp(prefix='v9_q5_12h_cpu_'))
    lockfile = (prodroot / 'validation.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); start = time.time(); ident = q.identity(); status = 'failed'
    q.write(prodroot / 'validation.json', dict(status='running', pid=os.getpid(), started_utc=q.utc(), temporary_runs=str(tmp)))
    try:
        hs, eng = old.fixture(); contracts(hs); smoke(hs, tmp, prodroot); queue_contracts(tmp, prodroot)
        check('Production code, actual data hashes and protocol are unchanged during validation', q.identity() == ident)
        preserved = q.read(prodroot / 'preserved.json')
        check('Previously completed scientific outputs remain unchanged', all(q.digest(f) == h for f, h in preserved.items()), files=len(preserved))
        status = 'passed'
    except Exception:
        TESTS.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
        print(traceback.format_exc(), flush=True)
    finally:
        q.ROOT = prodroot
        result = dict(status=status, lock_sha256=ident['lock_sha256'], tests=TESTS, temporary_runs=str(tmp),
                      device='cpu', seconds=time.time() - start, finished_utc=q.utc())
        if status == 'passed':
            q.write(prodroot / 'LOCK.json', ident)
        q.write(prodroot / 'validation.json', result)
        print(json.dumps(dict(status=status, tests=len(TESTS), seconds=result['seconds'])), flush=True)
    raise SystemExit(0 if status == 'passed' else 1)


if __name__ == '__main__':
    main()
