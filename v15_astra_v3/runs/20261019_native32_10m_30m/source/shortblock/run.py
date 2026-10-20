"""Single-owner preparation and alternating 30M/10M jobs under one 24h cap."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from ..isolation import ROOT, atomic_json, sha256, require_legacy_idle
from .data import DATA, REPORT

RUN=ROOT/'runs/20261019_native32_10m_30m'
CHILD=None
STOP=False


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()


def child(command, log, deadline, root, arm=None):
    global CHILD
    with log.open('a',buffering=1) as output:
        CHILD=subprocess.Popen(command,cwd=ROOT.parent,stdout=output,stderr=subprocess.STDOUT)
        info=dict(pid=CHILD.pid,command=command,timestamp=time.time())
        atomic_json(root/'child.json',info)
        if arm:atomic_json(root/arm/'child.json',info)
        while CHILD.poll() is None:
            if STOP or (root/'STOP').exists() or time.time()>=deadline:
                CHILD.terminate()
                try:CHILD.wait(timeout=45)
                except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
                break
            time.sleep(2)
    return CHILD.returncode


def initialize_plan(root=RUN):
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'plan.json').exists():
        start=time.time()
        predecessor=ROOT/'runs/20261018_24h_raw_gram/stop_request.json'
        if predecessor.exists():start=json.loads(predecessor.read_text())['timestamp']
        deadline=start+24*3600
        atomic_json(root/'plan.json',dict(version='native32-paired-v1',start=start,deadline=deadline,
            phase='Native32 codec acceptance, cached dataset, then matched 30M/10M GRAM',
            hours=24,data=str(DATA),target_ids=32,context_ids=256,arms=['30m','10m'],
            started_ist=datetime.fromtimestamp(start,ZoneInfo('Asia/Kolkata')).isoformat(),
            deadline_ist=datetime.fromtimestamp(deadline,ZoneInfo('Asia/Kolkata')).isoformat(),
            budget_scope='One shared 24-hour wall clock, including preparation and final evaluation',
            predecessor='Stopped by user at step 17085; preserved, never resumed by this supervisor',
            supervisor_module='v15_astra_v3.shortblock.run'))
    plan=json.loads((root/'plan.json').read_text())
    for name in ('30m','10m'):
        arm=root/name;arm.mkdir(exist_ok=True)
        if not (arm/'plan.json').exists():
            atomic_json(arm/'plan.json',{**plan,'phase':f'{name.upper()} GRAM; native32 preparation'})
            atomic_json(arm/'status.json',dict(status='codec_acceptance',timestamp=time.time()))
    return plan


def main(root=RUN):
    require_legacy_idle()
    root.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'reports/supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another supervisor is active') from None
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    plan=initialize_plan(root)
    deadline=plan['deadline'];training_deadline=deadline-600
    atomic_json(root/'process.json',dict(pid=os.getpid(),timestamp=time.time()))
    for name in ('30m','10m'):
        atomic_json(root/name/'process.json',dict(pid=os.getpid(),timestamp=time.time()))
    stages=(('codec_acceptance','v15_astra_v3.shortblock.adapt',[]),
        ('encoding','v15_astra_v3.shortblock.data',['--adaptation',str(REPORT/'decoder32.pt')]),
        ('preflight','v15_astra_v3.shortblock.preflight',[]))
    for stage,module,extra in stages:
        if STOP or (root/'STOP').exists():break
        # Completed acceptance/preflight artifacts are validated by their own
        # stages. Cache construction resumes from a hashed prefix.
        if stage=='preflight' and (REPORT/'preflight.json').exists():
            check=json.loads((REPORT/'preflight.json').read_text())
            manifest=json.loads((DATA/'manifest.json').read_text())
            if check['passed'] and check['dataset_sha']==manifest['episodes_sha']:continue
        atomic_json(root/'status.json',dict(status=stage,timestamp=time.time(),deadline=deadline))
        for name in ('30m','10m'):
            atomic_json(root/name/'status.json',dict(status=stage,timestamp=time.time()))
        rc=child([sys.executable,'-u','-m',module,*extra],root/f'{stage}.log',training_deadline,root)
        if rc:
            if STOP or (root/'STOP').exists():break
            raise RuntimeError(f'{stage} failed with exit {rc}; later stages blocked')
    if STOP or (root/'STOP').exists():
        atomic_json(root/'status.json',dict(status='stopped',timestamp=time.time()));return
    from .train import sources
    preflight=json.loads((REPORT/'preflight.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    if not preflight['passed'] or preflight['dataset_sha']!=manifest['episodes_sha']:
        raise ValueError('Unaccepted preflight')
    registered=root/'training_plan.json'
    if registered.exists():
        training=json.loads(registered.read_text())
        if training['sources']!=sources() or training['dataset_sha']!=manifest['episodes_sha']:
            raise ValueError('Registered experiment source or data changed')
    else:
        # Reserve 600s beyond final-evaluation reserve for switching overhead.
        seconds=min(42600,max(0,(training_deadline-time.time()-600)/2))
        if seconds<1800:raise RuntimeError('Too little budget left for the matched training comparison')
        common_schedule=max(1000,int(seconds*.95/max(
            preflight['arms'][name]['median_accumulated_update_seconds'] for name in ('30m','10m'))))
        training=dict(sources=sources(),dataset_sha=manifest['episodes_sha'],codec_id=manifest['codec']['codec_id'],
            per_arm_seconds=seconds,deadline=training_deadline,slice_seconds=1800,
            configs={name:sha256(REPORT/f'configs/{name}.json') for name in ('30m','10m')},
            schedules={name:common_schedule for name in ('30m','10m')},
            schedule_policy='Same learning rate at the same update index; common schedule fits the slower arm budget',
            comparison='Alternating exclusive jobs; same data/codec/context/target/recurrence/batch; equal arm wall-time budget and equal-step milestones')
        atomic_json(registered,training)
        snapshot=root/'source'
        for name in training['sources']:
            destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
    for name in ('30m','10m'):
        arm=root/name;arm.mkdir(exist_ok=True)
        atomic_json(arm/'plan.json',{**plan,'phase':f'{name.upper()} GRAM; 32-ID target, 256-ID observed context',
            'dataset_sha':manifest['episodes_sha'],'codec_id':manifest['codec']['codec_id'],
            'training_deadline':training_deadline,'per_arm_seconds':training['per_arm_seconds']})
        atomic_json(arm/'process.json',dict(pid=os.getpid(),timestamp=time.time()))
        atomic_json(arm/'status.json',dict(status='waiting_turn',timestamp=time.time()))
    while time.time()<training_deadline and not STOP and not (root/'STOP').exists():
        elapsed={}
        for name in ('30m','10m'):
            path=root/name/'gram/status.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            elapsed[name]=state.get('elapsed_seconds',0.)
        remaining=[name for name in ('30m','10m') if elapsed[name]<training['per_arm_seconds']]
        if not remaining:break
        name=min(remaining,key=lambda n:elapsed[n])
        if sha256(REPORT/f'configs/{name}.json')!=training['configs'][name]:
            raise ValueError('Registered configuration changed')
        atomic_json(root/'status.json',dict(status='training',active_arm=name,arm_elapsed_seconds=elapsed,timestamp=time.time()))
        atomic_json(root/name/'status.json',dict(status='training',timestamp=time.time()))
        command=[sys.executable,'-u','-m','v15_astra_v3.shortblock.train','--run',str(root/name/'gram'),
            '--config',str(REPORT/f'configs/{name}.json'),'--data',str(DATA),
            '--schedule-steps',str(training['schedules'][name]),'--total-seconds',str(training['per_arm_seconds']),
            '--slice-seconds',str(training['slice_seconds']),'--deadline',str(training_deadline)]
        rc=child(command,root/name/'training.log',training_deadline,root,name)
        if STOP or (root/'STOP').exists():break
        state=json.loads((root/name/'gram/status.json').read_text())
        if rc or state['status'] in ('low_memory','low_disk','low_battery','error','stop_requested'):
            raise RuntimeError(f'{name} stopped: {state}; exit={rc}')
        atomic_json(root/name/'status.json',dict(status='waiting_turn',timestamp=time.time()))
    if STOP or (root/'STOP').exists():
        atomic_json(root/'status.json',dict(status='stopped',timestamp=time.time()))
        for name in ('30m','10m'):atomic_json(root/name/'status.json',dict(status='stopped',timestamp=time.time()))
        return
    atomic_json(root/'status.json',dict(status='final_evaluation',timestamp=time.time()))
    rc=child([sys.executable,'-u','-m','v15_astra_v3.shortblock.finalize','--run',str(root)],
        root/'final_evaluation.log',deadline,root)
    complete=rc==0 and (root/'comparison.json').exists()
    status='complete' if complete else 'budget_finished_evaluation_incomplete'
    atomic_json(root/'status.json',dict(status=status,timestamp=time.time()))
    for name in ('30m','10m'):atomic_json(root/name/'status.json',dict(status=status,timestamp=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=RUN)
    args=ap.parse_args()
    try:main(args.run)
    except Exception as error:
        atomic_json(args.run/'status.json',dict(status='failed',error=repr(error),timestamp=time.time()))
        for name in ('30m','10m'):
            if (args.run/name).exists():atomic_json(args.run/name/'status.json',dict(status='failed',error=repr(error),timestamp=time.time()))
        raise
