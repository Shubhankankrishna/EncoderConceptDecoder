"""Serial six-hour experiment with seven-hour ceiling and an evaluation reserve."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from runtime import ROOT,atomic_json,Log,implementation_hash


def now():return datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=6);ap.add_argument('--max-hours',type=float,default=7);a=ap.parse_args()
    runs=ROOT/'runs';tests=json.loads((ROOT/'tests'/'optimized_checks.json').read_text())
    if (runs/'status.json').exists():
        previous=json.loads((runs/'status.json').read_text())
        if previous.get('status')=='running':
            try:os.kill(previous['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('V6 driver is already running')
        if list(runs.glob('*/train_summary.json')):raise RuntimeError('Existing experiment results: resume recorded stages rather than overwrite them')
    if not tests['passed']:raise RuntimeError('Prelaunch correctness failed')
    bench=json.loads((runs/'calibration'/'benchmark.json').read_text());ta=bench['training']['A']['median_step_s'];tb=bench['training']['B0']['median_step_s'];tg=bench['training']['B1']['median_step_s']
    begin=time.time();deadline=begin+a.max_hours*3600;train_deadline=deadline-90*60
    phase_specs=[('A_ncp_s0','A',18000,None,{}),('B0_det_s0','B0',6000,'A_ncp_s0',{}),
                 ('B1_gated_b025_s0','B1',6000,'B0_det_s0',dict(kind='gated',beta=.25)),
                 ('B1_direct_b025_s0','B1',6000,'B0_det_s0',dict(kind='direct',beta=.25)),
                 ('B1_gated_b0_s0','B1',6000,'B0_det_s0',dict(kind='gated',beta=0)),
                 ('B0_extra_compute_s0','det_extra',int(math.ceil(6000*tg/tb)),'B0_det_s0',{}),
                 ('A_no_forecast_prefix_s0','A',18000,None,dict(alpha=0,stop_after=9000))]
    estimates={'A':ta,'B0':tb,'B1':tg,'det_extra':tb}
    estimate=sum((opts.get('stop_after',steps))*estimates[stage] for _,stage,steps,_,opts in phase_specs)
    plan=dict(started=now(),started_epoch=begin,target_hours=a.hours,maximum_hours=a.max_hours,deadline_epoch=deadline,
              expected_training_h=estimate*1.2/3600,evaluation_reserve_min=90,
              expected_finish=datetime.fromtimestamp(begin+estimate*1.2+75*60,ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds'),
              latest_finish=datetime.fromtimestamp(deadline,ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds'),
              foundation_seed=0,training_batch=64,input_slots_per_update=12288,precision='FP32',active_parameters=2063904,
              control_parameters=2048,scope='One full foundation/core seed plus matched Z/objective/extra-training controls; independent foundation seeds deferred',
              implementation_sha256=implementation_hash(),draft_sha256=hashlib.sha256((ROOT.parent/'v6_draft.md').read_bytes()).hexdigest(),
              stages=[dict(name=n,stage=s,steps=t,source=src,options=o) for n,s,t,src,o in phase_specs],
              benchmark=bench,primary='Preselected gated beta=.25 / GradCuit KL=.05 versus strongest development retry anchor at a locked development batch-time budget')
    atomic_json(ROOT/'PLAN.json',plan)
    (ROOT/'spec').mkdir(exist_ok=True);shutil.copyfile(ROOT.parent/'v6_draft.md',ROOT/'spec'/'v6_draft.md')
    queue=Log(runs/'queue.jsonl');state=dict(status='running',pid=os.getpid(),started=plan['started'],stage='initializing',finished_stages=[],skipped_stages=[])
    anchor_name='B0_det_s0';anchor_ok=True
    atomic_json(runs/'status.json',state)
    def report():subprocess.run([sys.executable,str(ROOT/'code'/'report.py')],cwd=ROOT,check=True)
    def run(name,argv):
        if implementation_hash()!=plan['implementation_sha256']:raise RuntimeError('Source changed during the experiment; refusing mixed implementations')
        state.update(stage=name,updated=now());atomic_json(runs/'status.json',state);queue(event='start',stage=name,arguments=argv)
        print(f"{now()} START {name}",flush=True)
        with open(runs/(name+'.out'),'a') as output:
            child=subprocess.Popen([sys.executable]+argv,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;atomic_json(runs/'status.json',state);rc=child.wait()
        if rc:raise RuntimeError(f'{name} exited {rc}; see runs/{name}.out')
        state['finished_stages'].append(name);state.pop('child_pid',None);queue(event='completed',stage=name)
        atomic_json(runs/'status.json',state);report();print(f"{now()} COMPLETE {name}",flush=True)
    try:
        report()
        for name,stage,steps,source,options in phase_specs:
            remaining=train_deadline-time.time()
            if remaining<300:
                state['skipped_stages'].append(dict(stage=name,reason='preserve evaluation reserve'));continue
            argv=[str(ROOT/'code'/'train.py'),'--stage',stage,'--out',str(runs/name),'--steps',str(steps),'--seed','0',
                  '--seconds',str(min(remaining,estimates[stage]*(options.get('stop_after',steps))*1.6+180))]
            if source:
                if source=='B0_det_s0':
                    if not anchor_ok:
                        state['skipped_stages'].append(dict(stage=name,reason='anchor competence screen failed after fixed recovery'));continue
                    source=anchor_name
                src=runs/source/'checkpoint.pt'
                if not src.exists():state['skipped_stages'].append(dict(stage=name,reason='source unavailable'));continue
                argv+=['--source',str(src)]
            for key,value in options.items():argv+=['--'+key.replace('_','-'),str(value)]
            run(name,argv)
            if name=='B0_det_s0':
                gate_path=runs/'ANCHOR_GATE.json'
                def screen():
                    run('anchor_competence_screen',[str(ROOT/'code'/'anchor_gate.py'),'--foundation',str(runs/'A_ncp_s0'/'checkpoint.pt'),
                        '--anchor',str(runs/anchor_name/'checkpoint.pt'),'--out',str(gate_path)])
                    return json.loads(gate_path.read_text())['passed_mean_screen']
                anchor_ok=screen()
                if not anchor_ok and train_deadline-time.time()>600:
                    recovery='B0_guard_recovery_s0'
                    run(recovery,[str(ROOT/'code'/'train.py'),'--stage','det_extra','--source',str(runs/anchor_name/'checkpoint.pt'),
                        '--out',str(runs/recovery),'--steps','1500','--seconds','600','--seed','0'])
                    anchor_name=recovery;anchor_ok=screen()
                state.update(anchor_checkpoint=str(runs/anchor_name/'checkpoint.pt'),anchor_mean_screen_passed=anchor_ok)
                atomic_json(runs/'status.json',state)
        # Record every generated training graph, and enforce disjoint evaluation headers/topologies.
        seen=set()
        for checkpoint in runs.glob('*/checkpoint.pt'):
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False);seen.update(ck['mixer'].get('seen_graphs',[]));del ck
        (ROOT/'data'/'training_graph_hashes.txt').write_text('\n'.join(sorted(seen))+'\n')
        state['unique_training_graphs']=len(seen);atomic_json(runs/'status.json',state)
        anchor=runs/anchor_name/'checkpoint.pt';main_model=runs/'B1_gated_b025_s0'/'checkpoint.pt'
        for model in ('B1_gated_b025_s0','B1_direct_b025_s0','B1_gated_b0_s0'):
            ck=runs/model/'checkpoint.pt'
            if ck.exists():run('audit_'+model,[str(ROOT/'code'/'audit.py'),'--checkpoint',str(ck),'--anchor',str(anchor),'--out',str(runs/('audit_'+model+'.json'))])
        def evaluate(name,checkpoint,split='dev',standard=64,hard=128,quick=False,budget=None,methods=None):
            if time.time()+60>deadline or not Path(checkpoint).exists():state['skipped_stages'].append(dict(stage=name,reason='time budget or missing checkpoint'));return None
            argv=[str(ROOT/'code'/'evaluate.py'),'--checkpoint',str(checkpoint),'--out',str(runs/name),'--split',split,'--standard',str(standard),'--hard',str(hard),
                  '--bs','16','--deadline',str(deadline)]
            if quick:argv+=['--quick']
            if budget is not None:argv+=['--budget',str(budget)]
            if methods:argv+=['--methods',','.join(methods)]
            run(name,argv)
            summary=runs/name/'summary.json'
            return json.loads(summary.read_text()) if summary.exists() else None
        evaluate('eval_foundation_dev',runs/'A_ncp_s0'/'checkpoint.pt',quick=True)
        det=evaluate('eval_anchor_dev',anchor,quick=True)
        extra_path=runs/'B0_extra_compute_s0'/'checkpoint.pt';extra=evaluate('eval_extra_anchor_dev',extra_path,quick=True)
        evaluate('eval_main_dev',main_model)
        dev_hashes=set()
        for manifest in runs.glob('eval_*_dev/split_manifest.json'):
            for section in json.loads(manifest.read_text()).values():dev_hashes.update(section['canonical_hashes'])
        (ROOT/'data'/'development_graph_hashes.txt').write_text('\n'.join(sorted(dev_hashes))+'\n')
        candidates=[(anchor,det),(extra_path,extra)];valid=[(p,r) for p,r in candidates if r and 'hard' in r and 'retry' in r['hard']]
        if not valid:raise RuntimeError('No complete development retry reference; cannot lock primary cost')
        selected,reference=max(valid,key=lambda pair:pair[1]['hard']['retry']['success'])
        budget=reference['hard']['retry']['seconds_per_problem']*16
        primary=dict(selected_retry_checkpoint=str(selected),development_success=reference['hard']['retry']['success'],locked_batch_time_budget_s=budget,batch=16,
                     method='gated beta=.25, GradCuit KL=.05, LR=.1; selected before test',locked_at=now())
        atomic_json(runs/'PRIMARY_COMPARISON.json',primary)
        base_test=evaluate('eval_primary_retry_test',selected,'test',256,512,True,budget,['greedy','retry'])
        main_test=evaluate('eval_primary_gradcuit_test',main_model,'test',256,512,False,budget,['greedy','gradcuit_kl'])
        if base_test and main_test and 'hard' in base_test and 'hard' in main_test:
            from evaluate import bootstrap
            paired={setname:bootstrap(main_test[setname]['gradcuit_kl']['flags'],base_test[setname]['retry']['flags']) for setname in ('standard','hard') if setname in main_test and setname in base_test}
            atomic_json(runs/'PRIMARY_RESULT.json',dict(**primary,paired=paired,independent_foundation_seeds=1,
                        promotion='Unresolved: independent foundation-seed confirmation and specified confidence gates remain required'))
        evaluate('eval_main_attempt_count_test',main_model,'test',256,512)
        evaluate('eval_direct_dev',runs/'B1_direct_b025_s0'/'checkpoint.pt')
        evaluate('eval_beta0_dev',runs/'B1_gated_b0_s0'/'checkpoint.pt')
        prefix_summary=runs/'A_no_forecast_prefix_s0'/'train_summary.json'
        if prefix_summary.exists() and json.loads(prefix_summary.read_text())['completed_steps']==9000:
            evaluate('eval_forecast_prefix_dev',runs/'A_ncp_s0'/'checkpoint_step_9000.pt',quick=True)
            evaluate('eval_no_forecast_prefix_dev',runs/'A_no_forecast_prefix_s0'/'checkpoint.pt',quick=True)
        else:state['skipped_stages'].append(dict(stage='matched_forecast_prefix_comparison',reason='forecast control did not reach exactly 9,000 updates'))
        partial=[str(p.parent.name) for p in runs.glob('eval_*/summary.json') if json.loads(p.read_text()).get('partial_reason')]
        state.update(status='completed' if not state['skipped_stages'] and not partial else 'completed_with_limits',stage='finished',finished=now(),elapsed_h=(time.time()-begin)/3600,partial_evaluations=partial)
        atomic_json(runs/'status.json',state);queue(event='experiment_finished',status=state['status']);report()
    except Exception as exc:
        state.update(status='failed',error=repr(exc),failed_at=now());atomic_json(runs/'status.json',state);queue(event='failed',error=repr(exc),stage=state['stage']);report();raise


if __name__=='__main__':main()
