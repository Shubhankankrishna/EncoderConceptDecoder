"""Single-owner 48-hour experiment supervisor; deadline survives all retries."""
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

from .codec import ROOT,sha256
from .data import DATA,atomic_json

CHILD=None
STOP=False
ARMS=('gram','hrm')


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()


def run_child(command,logpath,deadline,run):
    global CHILD
    with open(logpath,'a',buffering=1) as log:
        CHILD=subprocess.Popen(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        while CHILD.poll() is None:
            if STOP or (run/'STOP').exists() or time.time()>=deadline:
                CHILD.terminate()
                try:CHILD.wait(timeout=90)
                except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
                return CHILD.returncode
            battery=psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent<20:
                CHILD.terminate();CHILD.wait(timeout=90)
                atomic_json(run/'status.json',dict(status='paused_low_battery',timestamp=time.time()))
                return 75
            time.sleep(15)
        return CHILD.returncode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--hours',type=float,default=48);ap.add_argument('--data-device',default='cpu');ap.add_argument('--data-batch',type=int,default=32)
    args=ap.parse_args();run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'artifacts/experiment.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another Astra experiment supervisor is active')
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    planfile=run/'plan.json'
    if planfile.exists():plan=json.loads(planfile.read_text())
    else:
        start=time.time();plan=dict(start=start,deadline=start+min(args.hours,48)*3600,hours=min(args.hours,48),
            timezone='Asia/Kolkata',started_local=datetime.fromtimestamp(start,ZoneInfo('Asia/Kolkata')).isoformat(),
            final_eval_reserve_seconds=3600,arms=list(ARMS),seed=1518,
            comparison='GRAM 10,000,333 vs HRM 10,001,292 parameters; same data and seed, equal allocated wall-time; one seed pilot',
            model_stage=2,source_files={p.name:sha256(p) for p in ROOT.glob('*.py')})
        atomic_json(planfile,plan)
        snapshot=run/'source';snapshot.mkdir(exist_ok=True)
        for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    atomic_json(run/'process.json',dict(pid=os.getpid(),started=time.time(),deadline=plan['deadline']))
    preflight=json.loads((ROOT/'reports/preflight.json').read_text())
    gram_preflight=json.loads((ROOT/'reports/gram_preflight.json').read_text())
    audit=json.loads((ROOT/'reports/codec_audit.json').read_text())
    assert preflight['overfit_pass'] and preflight['codec_state_unchanged'] and audit['decision']['action_codec_accepted']
    assert gram_preflight['overfit_pass'] and gram_preflight['codec_state_unchanged'] and gram_preflight['posterior_removed_inference_equal']
    mps_check=json.loads((ROOT/'reports/gram_mps_verification.json').read_text())
    data_check=json.loads((ROOT/'reports/data_audit.json').read_text())
    assert mps_check['mps_stochastic_resume_max_abs_error']<1e-6
    assert data_check['verified_transitions']==json.loads((DATA/'source_manifest.json').read_text())['records']
    assert not data_check['group_overlap'] and data_check['exact_current_state_reconstruction']
    assert plan['arms']==list(ARMS),'Run plan differs from the requested GRAM-first comparison'
    assert plan['source_files']=={p.name:sha256(p) for p in ROOT.glob('*.py')},'Sources changed; explicitly version the run before resuming'
    py=sys.executable
    if not (DATA/'manifest.json').exists():
        atomic_json(run/'status.json',dict(status='encoding_dataset',timestamp=time.time(),deadline=plan['deadline']))
        rc=run_child([py,'-u','-m','v15_astra_m2.data','--device',args.data_device,'--batch',str(args.data_batch)],run/'data.log',plan['deadline'],run)
        if rc:raise RuntimeError(f'Dataset encoding failed: {rc}')
    if 'stages' not in plan:
        # Real production-loader / optimizer / checkpoint integration before the
        # timed comparison. These runs never initialize the main experiment.
        for arm,steps in [('gram',12),('gram',3),('hrm',12)]:
            smoke=run/('smoke_'+arm)
            rc=run_child([py,'-u','-m','v15_astra_m2.train','--run',str(smoke),'--arm',arm,
                          '--deadline',str(min(plan['deadline'],time.time()+600)),
                          '--steps',str(steps),'--save-every','5','--eval-every','0'],
                         run/'smoke.log',min(plan['deadline'],time.time()+600),run)
            if rc:raise RuntimeError(f'Production-loader smoke failed: {arm} ({rc})')
        import torch
        ck=torch.load(run/'smoke_gram/latest.pt',map_location='cpu',weights_only=True)
        assert ck['step']>=15 and ck['data_sha']==json.loads((DATA/'manifest.json').read_text())['code_sha']
        inference=torch.load(run/'smoke_gram/latest_inference.pt',map_location='cpu',weights_only=True)
        assert not any(k.startswith(('posterior.','target_encoder.')) for k in inference['model'])
        rc=run_child([py,'-u','-m','v15_astra_m2.evaluate',str(run/'smoke_gram/latest_inference.pt'),
                      '--split','valid','--n','8','--out',str(run/'smoke_evaluation.json')],run/'smoke.log',min(plan['deadline'],time.time()+600),run)
        if rc:raise RuntimeError('Prior-only deployed checkpoint evaluation smoke failed')
        atomic_json(run/'smoke_verified.json',dict(gram_resumed_step=ck['step'],checkpoint_readable=True,finite_updates=True,posterior_absent_from_inference=True,evaluation_pipeline=True))
        del ck
        start=time.time();available=max(0,plan['deadline']-start-plan['final_eval_reserve_seconds']);half=available/2
        plan['stages']=[dict(arm=ARMS[0],start=start,deadline=start+half),dict(arm=ARMS[1],start=start+half,deadline=start+available)]
        atomic_json(planfile,plan)
    for stage in plan['stages']:
        if STOP or (run/'STOP').exists():break
        out=run/stage['arm'];done=out/'DONE.json'
        if done.exists():continue
        if time.time()>=stage['deadline']:continue
        atomic_json(run/'status.json',dict(status='training',arm=stage['arm'],deadline=stage['deadline'],global_deadline=plan['deadline'],timestamp=time.time()))
        cmd=[py,'-u','-m','v15_astra_m2.train','--run',str(out),'--arm',stage['arm'],
             '--deadline',str(stage['deadline']),'--batch','8','--accum','2','--seed','1518']
        rc=1
        for attempt in range(3):
            rc=run_child(cmd,run/f'{stage["arm"]}.log',stage['deadline']+90,run)
            if rc==0 or STOP or (run/'STOP').exists() or time.time()>=stage['deadline']:break
            atomic_json(run/'restart.json',dict(arm=stage['arm'],attempt=attempt+1,exit_code=rc,timestamp=time.time()))
            if rc==75:break
            time.sleep(20)
        if STOP or (run/'STOP').exists():break
        if rc:raise RuntimeError(f'{stage["arm"]} stopped with exit {rc}; inspect logs; no unbounded restart')
        atomic_json(done,dict(timestamp=time.time()))
    if STOP or (run/'STOP').exists():
        atomic_json(run/'status.json',dict(status='stopped',timestamp=time.time()));return
    atomic_json(run/'status.json',dict(status='final_evaluation',timestamp=time.time()))
    for arm in ARMS:
        path=run/arm/'best_policy.pt'
        if not path.exists():path=run/arm/'latest.pt'
        if not path.exists():continue
        if arm=='gram':path=path.with_name(path.stem+'_inference.pt')
        for split in ('test','ood'):
            if time.time()>plan['deadline']-60:break
            out=run/f'{arm}_{split}.json'
            if out.exists():continue
            rc=run_child([py,'-u','-m','v15_astra_m2.evaluate',str(path),'--split',split,'--n','512','--out',str(out)],run/'final_eval.log',plan['deadline'],run)
            if rc:atomic_json(run/'evaluation_error.json',dict(arm=arm,split=split,exit_code=rc))
    results={p.stem:json.loads(p.read_text()) for p in run.glob('*_test.json')}
    lines=['# V15 Astra M2 final experiment report','',
           'Single seed pilot. Frozen official LM1B NCM tokenizer; current-state-conditioned action policy.',
           'GRAM: 10,000,333 trainable parameters including posterior; HRM control: 10,001,292.',
           'GRAM runs first, as requested. Diagonal-Gaussian full-BPTT Astra adaptation; no search, distillation, or continual learning.',
           'GRAM NLL is a four-sample prior Monte Carlo estimate; HRM NLL is exact conditional code NLL.','',
           '| Arm | Test code NLL | Parsed actions | Optimal actions | Majority baseline |',
           '|---|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f'| {name} | {r["nll_per_code"]:.4f} | {r["parse_rate"]:.1%} | {r["optimal_action_rate"]:.1%} | {r["majority_baseline"]:.1%} |')
    complete=len(results)==2 and all((run/f'{a}_ood.json').exists() for a in ARMS)
    lines+=['',f'Final evaluation complete: {complete}. See JSON files for group confidence intervals and closed-loop results.']
    (run/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    from .plot_metrics import plot
    plot(run)
    atomic_json(run/'status.json',dict(status='complete' if complete else 'budget_finished_evaluation_incomplete',timestamp=time.time(),report=str(run/'FINAL_REPORT.md')))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('SUPERVISOR FAILED:',repr(e),flush=True)
        if '--run' in sys.argv:
            failure_dir=Path(sys.argv[sys.argv.index('--run')+1])
            atomic_json(failure_dir/'status.json',dict(status='failed',error=repr(e),timestamp=time.time()))
        raise
