"""Serial, bounded two-arm continuation with immutable source snapshots."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from v15_astra_ncm_v4.common import (ROOT, WORKSPACE, atomic_json, heavy_job, resource_issue,
    seed_all, sha)
from .train import DEFAULT_RUN, PARENT, sources

STOP = False
CHILD = None


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None: CHILD.terminate()


def final_evaluation(run):
    import torch
    from v15_astra_ncm_v4.cache import load_cache
    from .train import build, evaluate
    torch.set_num_threads(2); seed_all(1251)
    cache = load_cache(); net, encoder, _, _, inference = build()
    result = {'timestamp': time.time(), 'split': 'test', 'rows': '24:96',
        'selection': 'Validation free exact route then edit similarity then NLL, subject to retention gate',
        'limitations': ['One continuation seed; exploratory comparison.',
                       'Update budgets match; actual elapsed compute differs.',
                       'Old test rows0:24 excluded; no new broad-language claim.']}
    result['parent'] = evaluate(net, encoder, cache, inference, 0, 'test', 72)
    for arm in ('control', 'on_policy'):
        path = run / arm / 'best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        net.load_adapters(saved['adapters']); encoder.load_state_dict(saved['conditioner'])
        result[arm] = evaluate(net, encoder, cache, inference, saved['step'], 'test', 72)
        result[arm]['checkpoint_sha'] = sha(path)
        atomic_json(run / 'final_results.partial.json', result)
    atomic_json(run / 'final_results.json', result)


def main(args):
    global CHILD
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    deadline = parent_plan['deadline']
    if time.time() >= deadline - 600: raise RuntimeError('Original experiment time budget expired')
    if not json.loads((ROOT / 'reports/followup_preflight.json').read_text())['passed']:
        raise RuntimeError('Follow-up preflight did not pass')
    args.run.mkdir(parents=True, exist_ok=True)
    plan = {'version': 'renderer-residual-recovery-1', 'parent': str(PARENT),
        'parent_checkpoint_sha': sha(PARENT / 'renderer/latest.pt'), 'sources': sources(),
        'deadline': deadline, 'deadline_ist': parent_plan['deadline_ist'], 'training_deadline': deadline - 600,
        'steps_per_arm': 4000, 'arms': ['on_policy', 'control'], 'chunk_steps': 500,
        'seed': 119, 'comparison': 'Original prefix-noise continuation versus generated prefixes with consistent residual targets',
        'preflight_sha': sha(ROOT / 'reports/followup_preflight.json'),
        'diagnosis_sha': sha(ROOT / 'reports/renderer_diagnosis.json'),
        'cache_manifest_sha': sha(ROOT / 'data/ncm_cache/manifest.json')}
    plan_path = args.run / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan: raise RuntimeError('Supervisor identity changed')
    else:
        with heavy_job():
            atomic_json(plan_path, plan)
            for name in plan['sources']:
                destination = args.run / 'source' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(WORKSPACE / name, destination)
    atomic_json(args.run / 'process.json', {'pid': os.getpid(), 'started': time.time()})
    for chunk in range(500, 4001, 500):
        for arm in plan['arms']:
            if STOP or (args.run / 'STOP').exists():
                atomic_json(args.run / 'status.json', {'status': 'stopped', 'timestamp': time.time()}); return
            if time.time() >= deadline - 600:
                atomic_json(args.run / 'status.json', {'status': 'deadline', 'timestamp': time.time()}); return
            issue = resource_issue()
            if issue: raise RuntimeError(issue)
            state_path = args.run / arm / 'status.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get('step', 0) >= chunk: continue
            atomic_json(args.run / 'status.json', {'status': 'training', 'active_stage': arm,
                'chunk_target': chunk, 'timestamp': time.time(), 'deadline': deadline})
            command = [sys.executable, '-m', 'v15_astra_ncm_v4.followup.train', '--run', str(args.run / arm),
                '--arm', arm, '--steps', '4000', '--until-step', str(chunk), '--seed', '119',
                '--deadline', str(deadline - 600)]
            with (args.run / f'{arm}.log').open('a') as log:
                CHILD = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
                CHILD.wait()
            code = CHILD.returncode; CHILD = None
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if code != 0: raise RuntimeError(f'{arm} failed with return code {code}: {state}')
            if state.get('status') not in ('chunk_complete', 'complete'):
                atomic_json(args.run / 'status.json', {**state, 'active_stage': arm}); return
    atomic_json(args.run / 'status.json', {'status': 'final_evaluation', 'timestamp': time.time()})
    with heavy_job(): final_evaluation(args.run)
    atomic_json(args.run / 'status.json', {'status': 'complete', 'timestamp': time.time(),
        'quality_claim': 'Inspect held-out free route fidelity; lower teacher loss alone does not establish improvement.'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    try:
        args.run.mkdir(parents=True, exist_ok=True)
        with (args.run / 'supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(args)
    except Exception as error:
        atomic_json(args.run / 'status.json', {'status': 'failed', 'error': repr(error), 'timestamp': time.time()})
        raise
