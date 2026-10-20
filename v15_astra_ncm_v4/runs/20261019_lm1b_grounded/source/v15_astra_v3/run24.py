"""Detached, single-owner GRAM P1 experiment with an absolute 24-hour cap."""
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
from datetime import datetime
from zoneinfo import ZoneInfo
import psutil
from .isolation import ROOT,output_path,atomic_json,require_legacy_idle,sha256
from .train import source_identity

CHILD=None
STOP=False


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()


def run_child(command,logfile,deadline,run):
    global CHILD
    with logfile.open('a',buffering=1) as log:
        CHILD=subprocess.Popen(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(run/'child.json',dict(pid=CHILD.pid,command=command,timestamp=time.time()))
        while CHILD.poll() is None:
            if STOP or (run/'STOP').exists() or time.time()>=deadline:
                CHILD.terminate()
                try:CHILD.wait(timeout=45)
                except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
                return CHILD.returncode
            time.sleep(5)
        return CHILD.returncode


def main(args):
    require_legacy_idle();run=output_path(args.run);run.mkdir(parents=True,exist_ok=True)
    if args.hours<=0 or args.hours>24:raise ValueError('This authorization allows at most 24 hours')
    lock=(ROOT/'reports/supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another v3 supervisor is active') from None
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    preflight=json.loads(args.preflight.read_text());manifest=json.loads((args.data/'manifest.json').read_text())
    if not preflight['passed'] or preflight['dataset_sha']!=manifest['episodes_sha']:raise ValueError('Preflight or dataset identity mismatch')
    if not manifest['codec_state_unchanged'] or manifest['decode_audit']['nonpad_token_accuracy']<.8:raise ValueError('Codec acceptance failed')
    path=run/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['sources']!=source_identity() or plan['dataset_sha']!=manifest['episodes_sha']:
            raise ValueError('Registered source or data changed')
    else:
        now=time.time();budget=args.hours*3600;deadline=now+budget;reserve=min(600,budget*.05)
        schedule=max(1000,int((budget-reserve)/preflight['estimated_accumulated_step_seconds']))
        plan=dict(version='v3-raw-gram-24h',start=now,deadline=deadline,training_deadline=deadline-reserve,
            final_evaluation_reserve_seconds=reserve,hours=args.hours,first_arm='code_gram',arms=['code_gram'],
            phase='P1 raw-text next-concept pretraining',initialization='fresh; preflight and old action checkpoint are not reused',
            source_corpus=str(ROOT/'artifacts/fineweb_p0_20261018/documents.jsonl'),data=str(args.data.resolve()),
            dataset_sha=manifest['episodes_sha'],codec_id=manifest['codec']['codec_id'],sources=source_identity(),
            config_sha=sha256(args.config),preflight_sha=sha256(args.preflight),seed=151803,
            device='mps',precision='float32',batch=4,accumulation=2,lr=.0003,schedule_steps=schedule,
            started_ist=datetime.fromtimestamp(now,ZoneInfo('Asia/Kolkata')).isoformat(),
            deadline_ist=datetime.fromtimestamp(deadline,ZoneInfo('Asia/Kolkata')).isoformat(),
            test_selection='Validation only during training; fixed-budget latest and best validation snapshots preserved. Sealed test evaluated once at end.',
            prior_experiment='Original 48h GRAM/HRM pilot cancelled by user; checkpointed at 75878; no automatic HRM continuation',
            disabled=['grounded action encoder','candidate critic','global strategy z','search','continual learning'],
            limitation='One-seed short raw-text pilot; not an action-accuracy comparison or validation of the full architecture')
        atomic_json(path,plan);snapshot=run/'source';snapshot.mkdir(exist_ok=True)
        for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    if plan['config_sha']!=sha256(args.config):raise ValueError('Run configuration changed')
    atomic_json(run/'process.json',dict(pid=os.getpid(),timestamp=time.time(),deadline=plan['deadline']))
    training=run/'gram';command=[sys.executable,'-u','-m','v15_astra_v3.train','--task','raw','--arm','code_gram',
        '--run',str(training),'--data',str(args.data.resolve()),'--config',str(args.config.resolve()),
        '--steps','100000000','--schedule-steps',str(plan['schedule_steps']),'--seconds',str(plan['hours']*3600),
        '--deadline',str(plan['training_deadline']),'--device','mps','--batch','4','--accum','2',
        '--seed',str(plan['seed']),'--save-every','300','--eval-every','1800','--eval-n','64']
    for attempt in range(3):
        if STOP or (run/'STOP').exists() or time.time()>=plan['training_deadline']:break
        atomic_json(run/'status.json',dict(status='training',phase='raw_gram',pid=os.getpid(),deadline=plan['deadline'],timestamp=time.time()))
        rc=run_child(command,run/'training.log',plan['training_deadline'],run)
        if STOP or (run/'STOP').exists():break
        status=json.loads((training/'status.json').read_text()) if (training/'status.json').exists() else {}
        if rc==0:
            if status.get('status') in ('low_memory','low_disk','low_battery'):
                atomic_json(run/'status.json',dict(status='paused_resource',reason=status['status'],timestamp=time.time()));return
            break
        atomic_json(run/'retry.json',dict(attempt=attempt+1,exit_code=rc,timestamp=time.time()))
        if attempt==2:raise RuntimeError(f'GRAM failed after bounded retries: {rc}')
        time.sleep(20)
    if STOP or (run/'STOP').exists():
        atomic_json(run/'status.json',dict(status='stopped',timestamp=time.time()));return
    checkpoint=training/'best_nll_inference.pt'
    if not checkpoint.exists():checkpoint=training/'latest_inference.pt'
    if not checkpoint.exists():raise RuntimeError('No successful training checkpoint')
    atomic_json(run/'status.json',dict(status='final_evaluation',timestamp=time.time(),deadline=plan['deadline']))
    for split in ('valid','test'):
        if time.time()>=plan['deadline']-20 or STOP:break
        out=run/f'final_{split}.json'
        if out.exists():continue
        rc=run_child([sys.executable,'-u','-m','v15_astra_v3.evaluate_raw',str(checkpoint),
            '--data',str(args.data.resolve()),'--out',str(out),'--device','mps','--split',split,'--n','64'],
            run/'evaluation.log',plan['deadline'],run)
        if rc:atomic_json(run/'evaluation_error.json',dict(split=split,exit_code=rc,timestamp=time.time()))
    complete=(run/'final_test.json').exists() and (run/'final_valid.json').exists()
    atomic_json(run/'status.json',dict(status='complete' if complete else 'budget_finished_evaluation_incomplete',
        timestamp=time.time(),checkpoint=str(checkpoint),deadline=plan['deadline']))
    (run/'FINAL_REPORT.md').write_text('# V3 GRAM raw-text pilot\n\n'+
        f'24-hour experiment. Final evaluation complete: {complete}. Checkpoint: {checkpoint.name}.\n\n'+
        'Raw next-code likelihood and freely decoded continuations are in final_valid.json and final_test.json. '
        'This run has no grounded action, strategy, HRM comparison, search, or continual-learning claim.\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--preflight',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--hours',type=float,default=24)
    args=ap.parse_args()
    try:main(args)
    except Exception as error:
        atomic_json(output_path(args.run)/'status.json',dict(status='failed',error=repr(error),timestamp=time.time()))
        raise
