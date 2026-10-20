"""Single supervisor, frozen experiment identity, sequential staged jobs, 24h maximum."""
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

from .common import ROOT,WORKSPACE,atomic_json,sha,source_hashes,runtime_identity

CHILD=None
STOP=False


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()


def launch(command,root,name,deadline):
    global CHILD
    registered=json.loads((root/'plan.json').read_text())
    if source_hashes()!=registered['sources'] or runtime_identity()!=registered['runtime']:
        raise ValueError('Source changed before stage launch')
    atomic_json(root/'status.json',{'status':'training' if name!='finalize' else 'final_evaluation',
        'active_stage':name,'timestamp':time.time(),'deadline':deadline})
    with (root/f'{name}.log').open('a',buffering=1) as log:
        CHILD=subprocess.Popen(command,cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(root/'child.json',{'pid':CHILD.pid,'command':command,'timestamp':time.time()})
        while CHILD.poll() is None:
            if STOP or (root/'STOP').exists() or time.time()>=deadline:
                CHILD.terminate()
                try:CHILD.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    # One extra minute for an in-flight evaluation and atomic checkpoint.
                    try:CHILD.wait(timeout=60)
                    except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
                break
            time.sleep(2)
    return CHILD.returncode


def summarize_gates(root):
    arms={}
    for folder in root.iterdir():
        p=folder/'latest_eval.json'
        if p.exists() and folder.name!='renderer':
            result=json.loads(p.read_text())
            arms[folder.name]={z:{'success':r['rollouts']['greedy']['success_rate'],
                'random_success':r['rollouts']['random']['success_rate'],'value_mse':r['discounted_value_mse']}
                for z,r in result['strategies'].items()}
    r2=[arms[f'r2_seed{s}']['0'] for s in (42,43) if f'r2_seed{s}' in arms]
    grounded=len(r2)==2 and all(r['success']>=.7 and r['success']>=r['random_success']+.1 for r in r2)
    gains=[];strongest_gains=[]
    for seed in (42,43):
        if f'r0_seed{seed}' in arms and f'r2_seed{seed}' in arms:
            gains.append(arms[f'r2_seed{seed}']['0']['success']-arms[f'r0_seed{seed}']['0']['success'])
        controls=[arms[f'{name}_seed{seed}']['0']['success'] for name in ('r0','r1','untied2','untied_compute2')
                  if f'{name}_seed{seed}' in arms]
        if len(controls)==4 and f'r2_seed{seed}' in arms:
            strongest_gains.append(arms[f'r2_seed{seed}']['0']['success']-max(controls))
    result={'timestamp':time.time(),'arms':arms,'grounded_interface_gate':grounded,
        'r2_minus_r0_success_by_seed':gains,'r2_minus_strongest_control_by_seed':strongest_gains,
        'recurrence_benefit_demonstrated':len(strongest_gains)==2 and min(strongest_gains)>.02,
        'recurrence_gate_scope':'Pilot threshold on64 validation graphs; two seeds; not a broad statistical claim',
        'search_promoted':False,'search_reason':'MCTS is evaluation-only; equal-cost stronger-baseline evidence still required',
        'z_semantics':'Explicit supervised preferences, not an unsupervised Gaussian strategy discovery claim',
        'ncm_scope':'Verified-plan rendering; cannot establish pretrained transfer to graph planning'}
    atomic_json(root/'gates.json',result)
    return result


def main(root):
    root.mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another v4 supervisor is active') from None
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        for report in ('upstream_audit','preflight','adapter_preflight'):
            if not json.loads((ROOT/f'reports/{report}.json').read_text()).get('passed'):
                raise RuntimeError(f'Acceptance gate failed: {report}')
        identity=source_hashes()
        acceptance=json.loads((ROOT/'reports/launch_acceptance.json').read_text())
        if not acceptance.get('passed') or acceptance['sources']!=identity:
            raise ValueError('Final launch acceptance does not match current sources')
        stages=[]
        for seed in (42,43):
            for name,r,mode,z in [('r0',0,'shared',False),('r1',1,'shared',False),('r2',2,'shared',False),
                                  ('untied2',2,'untied',False),('untied_compute2',2,'untied_compute',False),
                                  ('strategy2',2,'shared',True)]:
                stages.append({'name':f'{name}_seed{seed}','seed':seed,'recurrence':r,'mode':mode,'strategies':z})
        if not (root/'plan.json').exists():
            start=json.loads((WORKSPACE/'v15_astra_v3/runs/20261019_native32_10m_30m/replacement_request.json').read_text())['timestamp']
            deadline=start+24*3600
            plan={'version':'astra-v4-staged-1','sources':identity,'runtime':runtime_identity(),'start':start,'deadline':deadline,
                'deadline_ist':datetime.fromtimestamp(deadline,ZoneInfo('Asia/Kolkata')).isoformat(),
                'budget_scope':'Up to24h including replacement setup; finish earlier on completed stages or plateau',
                'data_sha':sha(ROOT/'data/graphs.json'),'cache_manifest_sha':sha(ROOT/'data/ncm_cache/manifest.json'),
                'stages':stages,'structured_updates':4000,'renderer_updates_max':12000,'seed_order':'42 then43',
                'structured_selection':'best validation greedy success per arm; last-step comparisons also retained',
                'test_policy':'test/OOD read only by finalizer after validation selection'}
            atomic_json(root/'plan.json',plan)
            for file in identity:
                dest=root/'source'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WORKSPACE/file,dest)
        plan=json.loads((root/'plan.json').read_text())
        if plan['sources']!=identity or plan['runtime']!=runtime_identity() or plan['data_sha']!=sha(ROOT/'data/graphs.json') or plan['cache_manifest_sha']!=sha(ROOT/'data/ncm_cache/manifest.json'):
            raise ValueError('Registered run identity changed')
        deadline=plan['deadline'];train_deadline=deadline-1800
        atomic_json(root/'process.json',{'pid':os.getpid(),'timestamp':time.time()})
        subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for stage in stages:
            if STOP or (root/'STOP').exists() or time.time()>=train_deadline:break
            if source_hashes()!=identity:raise ValueError('Source changed during registered training')
            arm=root/stage['name'];state=json.loads((arm/'status.json').read_text()) if (arm/'status.json').exists() else {}
            if state.get('status')=='complete':continue
            command=[sys.executable,'-u','-m','v15_astra_ncm_v4.train','--run',str(arm),
                '--seed',str(stage['seed']),'--recurrence',str(stage['recurrence']),'--mode',stage['mode'],
                '--steps',str(plan['structured_updates']),'--deadline',str(train_deadline)]
            if stage['strategies']:command.append('--strategies')
            rc=launch(command,root,stage['name'],train_deadline)
            if STOP or (root/'STOP').exists():break
            if rc:raise RuntimeError(f"Stage {stage['name']} failed: exit{rc}")
            summarize_gates(root)
        if STOP or (root/'STOP').exists():
            atomic_json(root/'status.json',{'status':'stopped','timestamp':time.time()});return
        gates=summarize_gates(root)
        if gates['grounded_interface_gate'] and time.time()<train_deadline:
            state=json.loads((root/'renderer/status.json').read_text()) if (root/'renderer/status.json').exists() else {}
            if state.get('status') not in ('complete','validation_plateau'):
                rc=launch([sys.executable,'-u','-m','v15_astra_ncm_v4.adapt','--run',str(root/'renderer'),
                    '--steps',str(plan['renderer_updates_max']),'--deadline',str(train_deadline)],root,'renderer',train_deadline)
                if rc and not STOP:raise RuntimeError(f'Renderer failed: exit{rc}')
        else:
            atomic_json(root/'renderer_gate.json',{'enabled':False,'reason':'Grounded interface gate failed or time budget exhausted'})
        if STOP or (root/'STOP').exists():
            atomic_json(root/'status.json',{'status':'stopped','timestamp':time.time()});return
        if time.time()<deadline:
            rc=launch([sys.executable,'-u','-m','v15_astra_ncm_v4.finalize','--run',str(root),'--deadline',str(deadline)],root,'finalize',deadline)
            if rc:raise RuntimeError(f'Final evaluation incomplete: exit{rc}')
        complete=(root/'final_results.json').exists() and json.loads((root/'final_results.json').read_text()).get('all_structured_models_evaluated',False)
        atomic_json(root/'status.json',{'status':'complete' if complete else 'budget_exhausted','timestamp':time.time(),
            'quality_claim':'Read final_results and quality gates; completion is not proof of improved reasoning/language'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'runs/20261019_lm1b_grounded')
    args=ap.parse_args()
    try:main(args.run)
    except Exception as error:
        atomic_json(args.run/'status.json',{'status':'failed','error':repr(error),'timestamp':time.time()});raise
