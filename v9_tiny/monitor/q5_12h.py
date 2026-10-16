"""Isolated, deadline-bounded Q5 amendment. No original scientific output is modified.

The stdlib supervisor waits for the running EI round, holds the original queue,
then launches only validated, serial GPU workers. ML imports occur in workers.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON, CODE = V9 / 'monitor', V9 / 'code'
ROOT = V9 / 'runs_q5_12h'
PY = MON / 'v9_python'
SOURCE = V9 / 'runs/S2_K8_dense_s911/checkpoint.pt'
EI = V9 / 'runs/S4_s921_EI'
sys.path.insert(0, str(CODE))


def utc(t=None):
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).isoformat()


def read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def event(name, **kw):
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'queue.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=utc(), event=name, **kw)) + '\n')


def procs():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    return {int(a): (int(b), c) for line in rows.splitlines() for a, b, c in [line.strip().split(None, 2)]}


def protocol():
    return read(ROOT / 'PROTOCOL.json')


def identity():
    p = protocol()
    def files_hash(paths):
        h = hashlib.sha256()
        for f in sorted(paths):
            h.update(f.name.encode()); h.update(f.read_bytes())
        return h.hexdigest()
    prior = read(V9 / 'spec/LOCK_Q5.json')
    files = [CODE / name for name in prior['files']]
    frozen = files_hash(list((V9 / 'frozen/code').glob('*.py')))
    assert frozen == prior['frozen_implementation_sha256']
    pending = {}
    for block in ['Q5', 'Q6', 'QF']:
        lockpath = V9 / f'spec/LOCK_{block}.json'; old = read(lockpath)
        assert files_hash([CODE / name for name in old['files']]) == old['files_sha256'], block
        assert frozen == old['frozen_implementation_sha256'], block
        assert digest(V9 / 'frozen/data/manifest.json') == old['data_manifest_sha256'], block
        assert digest(V9 / 'frozen/data/evaluation_symbolic_hashes.txt') == old['exclusion_list_sha256'], block
        pending[block] = digest(lockpath)
    ident = dict(block='Q5_12H', frozen_implementation_sha256=frozen,
                 files_sha256=files_hash(files + [Path(__file__), MON / 'q5_12h_validate.py']),
                 data_manifest_sha256=digest(V9 / 'frozen/data/manifest.json'),
                 exclusion_list_sha256=digest(V9 / 'frozen/data/evaluation_symbolic_hashes.txt'),
                 original_pending_locks_sha256=pending)
    ident['protocol_sha256'] = digest(ROOT / 'PROTOCOL.json')
    ident['source_sha256'] = digest(Path(p['source']))
    ident['reference_sha256'] = digest(Path(p['reference']))
    # Actual bytes of every manifest-listed data file, including sealed bytes,
    # are checked without decoding or using sealed evaluation items.
    data = {}
    fd = V9 / 'frozen/data'; m = read(fd / 'manifest.json')
    for group, suffix in [('text', '.npy'), ('symbolic', '.npz')]:
        for name, row in m[group].items():
            f = fd / (name + suffix); assert digest(f) == row['sha256'], str(f); data[str(f)] = row['sha256']
    f = fd / 'tokenizer.json'; assert digest(f) == m['tokenizer_sha256']; data[str(f)] = m['tokenizer_sha256']
    f = V9.parent / 'v5_tiny/data/ts_train.npy'; assert digest(f) == m['tiny_stories']['sha256']; data[str(f)] = m['tiny_stories']['sha256']
    for name, row in m['tiny_stories']['partitions'].items():
        for suffix, key in [('ids.npy', 'ids_sha256'), ('hashes.txt', 'content_hashes_sha256')]:
            f = fd / f'ts_{name}_{suffix}'; assert digest(f) == row[key]; data[str(f)] = row[key]
    m = read(V9 / 'data_v9/manifest.json')
    for group in ('colourcount', 'calibration'):
        for name, row in m[group].items():
            f = V9 / f'data_v9/{group}_{name}.npz'; assert digest(f) == row['sha256']; data[str(f)] = row['sha256']
    f = V9 / 'data_v9/colourcount_eval_hashes.txt'; assert digest(f) == m['eval_hashes_sha256']; data[str(f)] = m['eval_hashes_sha256']
    ident['actual_data_sha256'] = data
    ident['lock_sha256'] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident


def preflight():
    now = identity(); lock = read(ROOT / 'LOCK.json', {})
    test = read(ROOT / 'validation.json', {})
    assert now == lock, 'Reduced Q5 code/data/protocol lock mismatch'
    assert test.get('status') == 'passed' and test.get('lock_sha256') == lock['lock_sha256'], 'Reduced Q5 validation missing/stale'
    preserved = read(ROOT / 'preserved.json', {})
    assert preserved and all(digest(p) == h for p, h in preserved.items()), 'Previously completed artifacts changed'
    return lock


def init():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'PROTOCOL.json').exists():
        return protocol()
    now = time.time(); done = read(EI / 'state.json')['done']
    p = dict(version=1, status='active_amendment', activated_utc=utc(now), activated_epoch=now,
             deadline_utc=utc(now + 12 * 3600), deadline_epoch=now + 12 * 3600,
             training_deadline_epoch=now + 9 * 3600, evaluation_deadline_epoch=now + 11.5 * 3600,
             seed=921, seeds=[921], source=str(SOURCE), original_ei=str(EI),
             reference=str(V9 / 'runs/S4_s921_reference_greedy.json'),
             finish_ei_iteration=len(done), target_ei_rounds=len(done) + 1,
             grpo_rounds=2, exposure_candidates=[512, 256, 128], group=8,
             sampling_problem_batch=4, optimizer_batch=64, gradient_microbatch=8,
             epochs=2, learning_rate=0.001, lora_rank=16, temperature=1., hop_limit=20,
             calibration_problems=16, reserve_fraction=0.25, evaluation_items_per_suite=512,
             evaluation_suites=['standard', 'hard10'], sealed_evaluation=False,
             removed=['seed922', 'rft', 'reinforce', 'GradCuit pilot'],
             interpretation='Exploratory single-seed comparison with unequal training budgets. No equal-compute MCTS superiority claim.',
             boundary_policy='Finish current MCTS round, freeze its verified heads, then hold original Q5/Q6/QF. Do not resume full queue automatically.')
    preserved = {}
    recovery = read(MON / 'q5_memory_recovery.json')
    for f, h in read(Path(recovery['backup']) / 'completed_sha256.json').items():
        assert digest(f) == h, f; preserved[f] = h
    for name in ['q3_completion_notification.json', 'q4_completed_artifacts.json', 'q5_completed_artifacts.json']:
        d = read(MON / name); d = d.get('artifacts_sha256', d)
        for rel, h in d.items():
            f = V9.parent / rel; assert digest(f) == h, str(f); preserved[str(f)] = h
    for row in read(MON / 'q5_completed_iterations.json', {}).get('iterations', {}).values():
        for rel, h in row.get('files', {}).items():
            f = V9.parent / rel; assert digest(f) == h; preserved[str(f)] = h
    assert digest(SOURCE) == '3a42a0a6399d8382658f8dac23fa184d75cdd4534704d91db8b40064fbef22f5'
    write(ROOT / 'preserved.json', preserved); write(ROOT / 'PROTOCOL.json', p)
    event('activated', protocol=p)
    return p


def snapshot_ei(p):
    i = p['finish_ei_iteration']; folder = EI / f'iter{i}'
    assert i in read(EI / 'state.json')['done']
    hashes = {str(folder / n): digest(folder / n) for n in ['heads.pt', 'traj.pt', 'summary.json']}
    prev = read(ROOT / 'preserved.json'); prev.update(hashes); write(ROOT / 'preserved.json', prev)
    write(ROOT / 'EI_SNAPSHOT.json', dict(iteration=i, rounds=i + 1, files=hashes,
          policy=read(folder / 'summary.json')['next_policy'], original_config=read(EI / 'config.json')))


def hold_original(p):
    # Commands and ancestry, not PID files alone, establish ownership.
    ps = procs(); targets = []
    for block in ['Q5', 'Q6', 'QF']:
        st = read(V9 / f'runs/status_{block}.json', {})
        pid = st.get('driver_pid')
        if pid in ps and f'/code/drive_{block.lower()}.py' in ps[pid][1]:
            if block == 'Q5':
                assert st.get('stage') == 'S4_s921_EI', st
            else:
                assert st.get('status') in ['waiting', 'stopped_by_user'], st
            targets.append(pid)
    write(ROOT / 'ORIGINAL_QUEUE_HOLD.json', dict(utc=utc(), reason='User-approved 12-hour, one-seed GRPO-only Q5 amendment', driver_pids=targets))
    for pid in targets:
        os.kill(pid, signal.SIGTERM)
    end = time.time() + 90
    while time.time() < end:
        ps = procs()
        active = [pid for pid, (_, cmd) in ps.items() if str(CODE / 'train_s4.py') in cmd or
                  any(str(CODE / f'drive_{b}.py') in cmd for b in ['q5', 'q6', 'qf'])]
        if not active:
            event('original_queue_held', drivers=targets); return
        time.sleep(1)
    raise RuntimeError(f'Original queue has live children; refusing duplicate GPU work: {active}')


def checkpoint_state(hs, gen, progress, opt=None):
    import torch
    import numpy as np
    import baselines
    return dict(method='grpo', state=baselines.lora_state(hs), progress=progress,
                generator=gen.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                optimizer=opt.state_dict() if opt is not None else None,
                mps_rng=torch.mps.get_rng_state() if next(hs.parameters()).device.type == 'mps' else None)


def clear_cache():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available() and os.environ.get('V8_DEVICE') != 'cpu':
        torch.mps.synchronize(); torch.mps.empty_cache()


class RecycleWorker(Exception):
    """Saved operational handoff, not a failed scientific update."""
    pass


def fit_data(answers):
    import numpy as np
    groups = {}
    for a in answers:
        groups.setdefault(a[4], []).append(a)
    data = []
    for rows in groups.values():
        assert len(rows) == 8, 'Incomplete GRPO group'
        r = np.array([float(a[2]) for a in rows]); sd = r.std()
        if sd:
            data += [(*a[:4], float(adv)) for a, adv in zip(rows, (r - r.mean()) / sd)]
    return data


def fit_resumable(hs, params, answers, progress, save, deadline, bs=64, micro=8):
    import numpy as np
    import torch
    import baselines
    data = fit_data(answers); dev = next(hs.parameters()).device
    opt = torch.optim.Adam(params, lr=0.001)
    if progress.get('optimizer'):
        opt.load_state_dict(progress.pop('optimizer'))
    progress.setdefault('fit_epoch', 0); progress.setdefault('fit_offset', 0)
    progress.setdefault('fit_losses', [])
    if not data:
        progress['fit_epoch'] = 2
        return dict(examples=0, loss=None, completed=True)
    while progress['fit_epoch'] < 2:
        if 'fit_order' not in progress:
            progress['fit_order'] = np.random.permutation(len(data)).tolist()
        order = progress['fit_order']
        for start in range(progress['fit_offset'], len(data), bs):
            if time.time() >= deadline:
                save(opt); return dict(examples=len(data), loss=None, completed=False)
            rows = [data[i] for i in order[start:start + bs]]; opt.zero_grad()
            loss_sum = 0.
            for off in range(0, len(rows), micro):
                batch = rows[off:off + micro]
                x, docs, m = baselines.seq_batch([a[:4] for a in batch], dev)
                w = torch.tensor([a[4] for a in batch], dtype=torch.float32, device=dev)
                with torch.enable_grad():
                    loss = -(w * baselines.token_logprob(hs, x, docs, m)).sum() / len(rows)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite GRPO loss')
                    loss.backward()
                loss_sum += float(loss.detach()); del loss, x, docs, m, w
            torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
            progress['fit_losses'].append(loss_sum); progress['fit_offset'] = start + len(rows)
            save(opt); clear_cache()
        progress['fit_epoch'] += 1; progress['fit_offset'] = 0; progress.pop('fit_order')
        save(opt)
    return dict(examples=len(data), loss=float(np.mean(progress['fit_losses'])), completed=True)


def train_worker(p, out, round_i, n, deadline, source=None, calibration=False):
    import numpy as np
    import torch
    import baselines, hncp, hgen, train_s4
    from runtime import DEVICE, atomic_save
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    result = out / 'summary.json'
    if result.exists() and read(result).get('status') == 'completed':
        return read(result)
    torch.manual_seed(p['seed']); np.random.seed(p['seed']); torch.set_num_threads(4)
    hs, _ = hncp.load(source or p['source'], DEVICE)
    params = baselines.add_lora(hs); gen = torch.Generator().manual_seed(88000 + p['seed'])
    statepath = out / 'resume.pt'; answerspath = out / 'answers.pt'
    progress = dict(round=round_i, next_problem=0, phase='sampling', active_seconds=0.)
    answers = []
    if statepath.exists():
        ck = torch.load(statepath, map_location='cpu', weights_only=False)
        baselines.load_lora(hs, ck['state']); gen.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng'])
        if DEVICE == 'mps' and ck.get('mps_rng') is not None:
            torch.mps.set_rng_state(ck['mps_rng'])
        progress = ck['progress']; progress['optimizer'] = ck.get('optimizer')
        answers = torch.load(answerspath, map_location='cpu', weights_only=False)[:progress['next_problem'] * 8]
    elif round_i > 0 and not calibration:
        ck = torch.load(ROOT / f'grpo/round{round_i - 1}/adapter.pt', map_location='cpu', weights_only=False)
        baselines.load_lora(hs, ck['state']); gen.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng'])
    eng = hgen.Engine(hs)
    # Generate the full original fresh draw before slicing: 256 is the exact prefix of 512.
    items = train_s4.fresh_problems(512 if not calibration else n, 86000 + 100 * p['seed'] + round_i)[:n]
    run_started = time.time(); initial_active = progress['active_seconds']

    def save(opt=None):
        progress['active_seconds'] = initial_active + time.time() - run_started
        atomic_save(statepath, checkpoint_state(hs, gen, progress, opt))
        write(out / 'live.json', dict(utc=utc(), pid=os.getpid(), **{k: v for k, v in progress.items() if k not in ['fit_order', 'optimizer', 'fit_losses']},
                                    problems=n, verifier_calls=progress['next_problem'] * 8,
                                    last_loss=progress.get('fit_losses', [None])[-1] if progress.get('fit_losses') else None))
        if DEVICE == 'mps':
            clear_cache()
            if torch.mps.driver_allocated_memory() >= 4_000_000_000:
                print(json.dumps(dict(event='saved_memory_handoff', driver_bytes=torch.mps.driver_allocated_memory(),
                                      round=round_i, next_problem=progress['next_problem'], phase=progress['phase'])), flush=True)
                raise RecycleWorker()

    if progress['phase'] == 'sampling':
        # Reserve 40% of the remaining training slot for fitting if sampling is unexpectedly slow.
        sampling_deadline = p.get('sampling_deadline_epoch', deadline)
        for j in range(progress['next_problem'], n, p['sampling_problem_batch']):
            if time.time() >= sampling_deadline:
                save(); write(result, dict(status='partial_time_budget', round=round_i, problems=progress['next_problem'], verifier_calls=len(answers))); return read(result)
            batch = items[j:j + p['sampling_problem_batch']]
            group_items = [it for it in batch for _ in range(8)]
            res = baselines.sample_answers(eng, group_items, 1., 'free', hop_limit=20, gen=gen)
            answers += [(t, c, ok, it.P, j + k // 8) for k, (it, (t, c, ok)) in enumerate(zip(group_items, res))]
            progress['next_problem'] = j + len(batch)
            atomic_save(answerspath, answers); save(); clear_cache()
            print(json.dumps(dict(event='sampled', round=round_i, problems=progress['next_problem'], calls=len(answers), utc=utc())), flush=True)
        progress['phase'] = 'fitting'; save()
    fit = fit_resumable(hs, params, answers, progress, save, deadline, p['optimizer_batch'], p['gradient_microbatch'])
    save()
    complete = fit['completed'] and progress['next_problem'] == n
    ck = checkpoint_state(hs, gen, progress)
    ck.update(source=p['source'], trainable=sum(q.numel() for q in params))
    atomic_save(out / 'adapter.pt', ck)
    row = dict(status='completed' if complete else 'partial_time_budget', round=round_i, problems=n,
               verifier_calls=len(answers), accepted=sum(int(a[2]) for a in answers), fit=fit,
               active_seconds=progress['active_seconds'], adapter_sha256=digest(out / 'adapter.pt'))
    write(result, row); print(json.dumps(row), flush=True); return row


def worker(args):
    p = protocol()
    import torch
    torch.set_num_threads(4)
    if args.worker == 'calibrate':
        out = ROOT / 'calibration'
        row = train_worker(p, out, 0, p['calibration_problems'], min(time.time() + 1800, p['training_deadline_epoch']), calibration=True)
        assert row['status'] == 'completed', 'Calibration incomplete'
        # Time a conservative fit even if actual calibration groups have equal
        # rewards and therefore no gradient. Synthetic labels are discarded and
        # never enter scientific training, checkpoints or reported accuracy.
        import baselines, hncp
        from runtime import DEVICE
        bench = read(out / 'fit_benchmark.json')
        if bench is None:
            answers = torch.load(out / 'answers.pt', map_location='cpu', weights_only=False)
            synthetic = [(*a[:2], bool(k % 2), *a[3:]) for k, a in enumerate(answers)]
            hs, _ = hncp.load(p['source'], DEVICE); params = baselines.add_lora(hs)
            t = time.time()
            fit = fit_resumable(hs, params, synthetic, {}, lambda opt: None,
                                min(t + 900, p['training_deadline_epoch']), p['optimizer_batch'], p['gradient_microbatch'])
            assert fit['completed'] and fit['examples'] == len(answers)
            bench = dict(seconds=time.time() - t, examples=fit['examples'], synthetic_labels=True, discarded=True)
            write(out / 'fit_benchmark.json', bench); del hs, params; clear_cache()
        available = max(0., p['training_deadline_epoch'] - time.time())
        measured = row['active_seconds'] + bench['seconds']
        choice = next((n for n in p['exposure_candidates'] if measured / p['calibration_problems'] * n * p['grpo_rounds'] * 1.25 <= available), None)
        write(ROOT / 'EXPOSURE.json', dict(problems_per_round=choice, rounds=2, measured_calibration_seconds=row['active_seconds'],
              conservative_fit_benchmark=bench,
              available_training_seconds=available, projection_factor=1.25, chosen_utc=utc(), calibration_is_not_scientific_result=True,
              projected_training_seconds=measured / p['calibration_problems'] * choice * 2 if choice else None))
    elif args.worker == 'train':
        n = read(ROOT / 'EXPOSURE.json')['problems_per_round']; assert n
        p['sampling_deadline_epoch'] = p['training_deadline_epoch'] - 1800
        train_worker(p, ROOT / f'grpo/round{args.round}', args.round, n, p['training_deadline_epoch'])
    elif args.worker == 'freeze':
        import hncp
        hs, _ = hncp.load(p['source'], 'cpu'); snap = read(ROOT / 'EI_SNAPSHOT.json')
        heads = torch.load(EI / f'iter{snap["iteration"]}/heads.pt', map_location='cpu', weights_only=False)
        hs.codes.selector.load_state_dict(heads['selector'])
        hncp.save(ROOT / 'EI/checkpoint.pt', hs, dict(config=dict(stage='S4_12H_EI', source=p['source'], completed_rounds=snap['rounds']), heads=heads['heads'], policy=snap['policy']))
        assert all(bool(torch.isfinite(t).all()) for t in hs.state_dict().values() if t.is_floating_point())
    elif args.worker == 'eval':
        from s4_eval import greedy_eval
        name = args.arm
        source = str(ROOT / 'EI/checkpoint.pt') if name == 'EI' else p['source']
        adapter = str(ROOT / f'grpo/round{args.round}/adapter.pt') if name == 'grpo' else None
        write(ROOT / f'eval_{name}.json', greedy_eval(source, 'free' if name == 'grpo' else 'codes', adapter, n=512))


def paired(a, b):
    av, bv = a['success'], b['success']; assert len(av) == len(bv) == 512
    wins = sum(x and not y for x, y in zip(av, bv)); losses = sum(y and not x for x, y in zip(av, bv))
    return dict(delta_percentage_points=100 * (sum(av) - sum(bv)) / len(av), a_only=wins, b_only=losses, n=len(av))


def report(reason=None):
    p = protocol(); results = {}
    for arm in ['reference', 'EI', 'grpo']:
        f = Path(p['reference']) if arm == 'reference' else ROOT / f'eval_{arm}.json'
        if f.exists():
            results[arm] = read(f)
    rounds = [read(ROOT / f'grpo/round{i}/summary.json', {}) for i in range(2)]
    complete = all(r.get('status') == 'completed' for r in rounds) and all(a in results for a in ['reference', 'EI', 'grpo'])
    pairs = {}
    if 'EI' in results and 'grpo' in results:
        pairs = {suite: paired(results['EI'][suite], results['grpo'][suite]) for suite in p['evaluation_suites']}
    ei_rounds = [read(EI / f'iter{i}/summary.json') for i in range(p['target_ei_rounds']) if (EI / f'iter{i}/summary.json').exists()]
    obj = dict(status='completed' if complete else 'incomplete', created_utc=utc(), reason=reason,
               deadline_utc=p['deadline_utc'], protocol=p, exposure=read(ROOT / 'EXPOSURE.json'),
               arms=results, grpo_rounds=rounds, ei_rounds=ei_rounds, paired_EI_minus_grpo=pairs,
               interpretation=p['interpretation'], original_full_Q5_completed=False)
    write(ROOT / 'Q5_12H_RESULT.json', obj)
    text = ['# Reduced Q5 development comparison', '', f'Status: {obj["status"]}. Created: {obj["created_utc"]}.', '', p['interpretation'], '',
            'Seed 921 only. The existing MCTS round was allowed to finish. GRPO is the only baseline. Sealed tests were not used.', '',
            '| Arm | Standard | Hard10 |', '| --- | ---: | ---: |']
    for arm, rows in results.items():
        text.append(f'| {arm} | {100 * rows["standard"]["rate"]:.2f}% | {100 * rows["hard10"]["rate"]:.2f}% |')
    text += ['', f'GRPO exposure and completed rounds: {json.dumps(obj["exposure"])}; {json.dumps(rounds)}', '',
             f'Paired EI minus GRPO: {json.dumps(pairs)}', '', f'Limitation/status reason: {reason or "none"}.', '',
             'Original Q6/QF remain held. This artifact does not mark the original full Q5 completed.']
    (ROOT / 'REPORT.md').write_text('\n'.join(text) + '\n')
    event('report_written', status=obj['status'], reason=reason)
    return obj


class Supervisor:
    def __init__(self):
        self.child = None; self.stopped = False; self.stage = 'waiting_current_MCTS_round'
        self.p = protocol(); self.started = time.time()
        signal.signal(signal.SIGTERM, self.stop); signal.signal(signal.SIGINT, self.stop)

    def stop(self, *_):
        self.stopped = True
        if self.child and self.child.poll() is None:
            self.child.terminate()

    def status(self, status='running', **kw):
        write(ROOT / 'status.json', dict(status=status, stage=self.stage, driver_pid=os.getpid(),
              child_pid=self.child.pid if self.child and self.child.poll() is None else None,
              updated_utc=utc(), deadline_utc=self.p['deadline_utc'], **kw))

    def run(self, name, args, deadline):
        last_progress = None; repeats = 0
        while not self.stopped and time.time() < deadline:
            result = self._run_worker_once(name, args, deadline)
            if result != 75:
                return result
            live = ROOT / ('calibration' if name == 'calibrate' else f'grpo/round{args[-1]}') / 'live.json'
            j = read(live, {})
            current = tuple(j.get(k) for k in ['next_problem', 'phase', 'fit_epoch', 'fit_offset'])
            repeats = repeats + 1 if current == last_progress else 0
            assert repeats < 2, 'Memory handoffs are not advancing; inspect the concrete cause'
            last_progress = current
            event('memory_worker_recycled', stage=name, saved_progress=j)
        return False

    def _run_worker_once(self, name, args, deadline):
        preflight(); self.stage = name
        command = [str(PY), str(Path(__file__).resolve()), '--worker', *args]
        event('started', stage=name, command=command)
        with (ROOT / f'{name}.log').open('a') as f:
            self.child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=V9)
            self.status(); last = 0
            while self.child.poll() is None:
                if time.time() > deadline or self.stopped:
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.child.kill(); self.child.wait()
                    event('bounded_stop', stage=name); self.child = None; return False
                if time.time() - last > 30:
                    self.status(); last = time.time()
                time.sleep(1)
            rc = self.child.returncode; self.child = None
        event('finished' if rc == 0 else 'saved_memory_handoff' if rc == 75 else 'failed', stage=name, exit_code=rc)
        self.status()
        if rc == 75:
            return 75
        if rc:
            raise RuntimeError(f'{name} exited {rc}; inspect {name}.log before any retry')
        return True

    def main(self):
        p = self.p; held = (ROOT / 'EI_SNAPSHOT.json').exists() and (ROOT / 'ORIGINAL_QUEUE_HOLD.json').exists()
        while not held:
            if self.stopped:
                self.status('stopped_by_user'); return
            if time.time() >= p['deadline_epoch'] - 60:
                return self.finish('MCTS did not reach its completed-round boundary within the twelve-hour deadline')
            done = read(EI / 'state.json', {}).get('done', [])
            if len(done) >= p['target_ei_rounds']:
                snapshot_ei(p); hold_original(p); held = True; break
            # Never disturb MCTS before its completed-round boundary.
            self.status('waiting', completed_ei_rounds=len(done), target_ei_rounds=p['target_ei_rounds'])
            time.sleep(1)
        self.stage = 'waiting_validation'
        while read(ROOT / 'validation.json', {}).get('status') != 'passed':
            if self.stopped:
                self.status('stopped_by_user'); return
            if time.time() > p['activated_epoch'] + 2 * 3600:
                raise RuntimeError('Validation did not pass within transition allocation')
            self.status('waiting'); time.sleep(5)
        preflight()
        # Refuse any orphan trainer/evaluator even if the original driver has exited.
        active = [pid for pid, (_, cmd) in procs().items() if str(CODE) + '/' in cmd and '.py' in cmd]
        assert not active, f'GPU may still be owned by original project processes: {active}'
        if not (ROOT / 'EI/checkpoint.pt').exists():
            (ROOT / 'EI').mkdir(exist_ok=True)
            assert self.run('freeze', ['freeze'], p['training_deadline_epoch'])
        if not (ROOT / 'EXPOSURE.json').exists():
            if not self.run('calibrate', ['calibrate'], min(time.time() + 1800, p['training_deadline_epoch'])):
                return self.finish('Calibration deadline exceeded')
        if not read(ROOT / 'EXPOSURE.json')['problems_per_round']:
            return self.finish('Smallest declared GRPO exposure cannot fit the measured deadline')
        for i in range(2):
            row = read(ROOT / f'grpo/round{i}/summary.json', {})
            if row.get('status') == 'completed':
                assert digest(ROOT / f'grpo/round{i}/adapter.pt') == row['adapter_sha256']
                event('verified_completed', stage=f'grpo_round{i}'); continue
            if not self.run(f'grpo_round{i}', ['train', '--round', str(i)], p['training_deadline_epoch']):
                break
            if read(ROOT / f'grpo/round{i}/summary.json')['status'] != 'completed':
                break
        for arm in ['EI', 'grpo']:
            if (ROOT / f'eval_{arm}.json').exists():
                continue
            trained = [i for i in range(2) if (ROOT / f'grpo/round{i}/adapter.pt').exists()]
            if arm == 'grpo' and not trained:
                continue
            if not self.run(f'eval_{arm}', ['eval', '--arm', arm, '--round', str(max(trained, default=0))], p['evaluation_deadline_epoch']):
                break
        self.finish()

    def finish(self, reason=None):
        if self.stopped:
            self.status('stopped_by_user'); return
        obj = report(reason); self.stage = 'report'; self.status(obj['status'], report=str(ROOT / 'REPORT.md'))


def detach():
    p = init(); ps = procs()
    pid = read(ROOT / 'status.json', {}).get('driver_pid')
    if pid in ps and str(Path(__file__).resolve()) in ps[pid][1]:
        print(json.dumps(dict(status='already_running', pid=pid, deadline=p['deadline_utc']))); return
    orphan = [pid for pid, (_, cmd) in ps.items() if str(Path(__file__).resolve()) in cmd and '--worker' in cmd]
    assert not orphan, f'A reduced Q5 worker is still alive; refusing a duplicate: {orphan}'
    if read(ROOT / 'status.json', {}).get('status') == 'completed' and (ROOT / 'Q5_12H_RESULT.json').exists():
        print(json.dumps(dict(status='already_completed', report=str(ROOT / 'REPORT.md')))); return
    with (ROOT / 'supervisor.log').open('a') as f:
        proc = subprocess.Popen(['/opt/anaconda3/bin/python3', '-S', '-B', str(Path(__file__).resolve()), '--supervise'],
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True, cwd=V9)
    subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(json.dumps(dict(status='started', pid=proc.pid, deadline=p['deadline_utc'])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--init', action='store_true'); ap.add_argument('--detach', action='store_true')
    ap.add_argument('--supervise', action='store_true'); ap.add_argument('--worker', choices=['calibrate', 'train', 'freeze', 'eval'])
    ap.add_argument('--round', type=int, default=0); ap.add_argument('--arm', choices=['EI', 'grpo'])
    args = ap.parse_args()
    if args.init:
        print(json.dumps(init())); return
    if args.detach:
        detach(); return
    if args.worker:
        try:
            worker(args)
        except RecycleWorker:
            raise SystemExit(75)
        return
    lockfile = (ROOT / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another reduced Q5 supervisor owns the queue')
    s = Supervisor()
    try:
        s.main()
    except Exception as exc:
        write(ROOT / 'failure.json', dict(utc=utc(), error=repr(exc), traceback=traceback.format_exc()))
        s.status('failed', error=repr(exc)); event('failed', error=repr(exc)); raise


if __name__ == '__main__':
    main()
