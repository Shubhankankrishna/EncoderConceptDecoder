"""Detached serial, competence-first V8 pilot; preserves exposure before arm breadth."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from runtime import *
from model import *
from data import DATA,Mixer,digest
from evaluate import evaluate,retention,symbolic_eval,text_eval,paired,cluster_ci
from continual import gate_rates
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';CODE=ROOT/'code';PY=sys.executable
class BudgetEnd(Exception):pass
class GateEnd(Exception):pass
class Queue:
    def __init__(self,hours=12):
        self.start=time.time();self.deadline=self.start+hours*3600;self.reserve=5400;self.child=None;self.stage='initializing';self.entries=[];self.log=Log(RUNS/'queue.jsonl');self.stopped=False
        self.bench=json.loads((ROOT/'tests/throughput.json').read_text());self.compiled=self.bench['chosen']=='compiled';self.step_s=self.bench[self.bench['chosen']]['step_s']*1.3;self.foundations={};self.completed=[];self.skipped=[]
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,self.stop)
    def stop(self,sig,frame):
        self.stopped=True
        if self.child and self.child.poll() is None:self.child.terminate()
    def status(self,status='running',**extra):
        atomic_json(RUNS/'status.json',dict(status=status,stage=self.stage,driver_pid=os.getpid(),child_pid=self.child.pid if self.child and self.child.poll() is None else None,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.start)),deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.deadline)),final_testing_reserve_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.deadline-self.reserve)),elapsed_s=time.time()-self.start,remaining_s=max(0,self.deadline-time.time()),**extra))
    def remaining(self,reserve=True):return self.deadline-time.time()-(self.reserve if reserve else 0)
    def skip(self,name,reason):self.skipped.append(dict(stage=name,reason=reason));self.log(stage=name,event='skipped',reason=reason)
    def run(self,name,script,args,estimated=0,required=False,reserve=True):
        if (ROOT/'spec/LOCK.json').exists() and json.loads((ROOT/'spec/LOCK.json').read_text())['implementation_sha256']!=implementation_hash():raise RuntimeError('Locked experiment source changed')
        if self.stopped:raise KeyboardInterrupt('User stopped queue')
        remaining=self.remaining(reserve)
        if remaining<max(estimated,60):
            self.skip(name,'Insufficient reserved wall time for complete planned exposure')
            if required:raise BudgetEnd(name)
            return False
        self.stage=name;folder=RUNS/name;folder.mkdir(parents=True,exist_ok=True);command=[PY,str(CODE/script),*map(str,args)];self.log(stage=name,event='started',command=command);self.status()
        with open(folder/'process.log','a') as f:
            self.child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parent);self.status()
            while self.child.poll() is None:
                time.sleep(2)
                if self.stopped:self.child.terminate()
            rc=self.child.returncode
        self.child=None;self.status()
        if self.stopped:raise KeyboardInterrupt('User stopped queue')
        if rc!=0:
            raise RuntimeError(f'{name} exited {rc}; inspect {folder}/process.log and failure.json')
        self.completed.append(name);self.log(stage=name,event='finished');return True
    def train(self,name,stage,steps,source=None,alpha=0.,flat=False,mode='foundation',context=192,core='dense',no_distill=False,always_open=False,required=False,batch=64,learning=False,fixed=False,estimate_multiplier=1.):
        estimate=steps*self.step_s*estimate_multiplier+120
        if self.remaining()<estimate:
            self.skip(name,'Complete update budget will not fit; foundation exposure never silently reduced')
            if required:raise BudgetEnd(name)
            return None
        args=['--stage',stage,'--out',RUNS/name,'--steps',steps,'--seconds',max(1,self.remaining()-60),'--batch',batch,'--seed',811,'--alpha',alpha,'--mode',mode,'--context',context,'--core',core]
        if source:args+=['--source',source]
        if flat:args+=['--flat']
        execution_key=('A_FLAT' if flat and stage=='A' else stage)+'/'+core
        if source:
            metadata=torch.load(source,map_location='cpu',weights_only=False).get('metadata',{})
            if metadata.get('core')=='recurrent' and stage in ('Z','CL'):execution_key=stage+'/recurrent'
        checks=json.loads((ROOT/'tests/stage_contracts.json').read_text())['checks']
        eligible_compile=next((v.get('compiled_safe',False) for v in checks if v['stage']==execution_key),stage in ('SIMPLE','MEM','HARD','EXTRA'))
        if self.compiled and eligible_compile:args+=['--compiled']
        if stage!='A':args+=['--constant-alpha']
        if no_distill:args+=['--no-distill']
        if always_open:args+=['--always-open']
        if learning:args+=['--learning-check']
        if fixed:args+=['--fixed']
        if not self.run(name,'train.py',args,estimate,required):return None
        summary=json.loads((RUNS/name/'train_summary.json').read_text())
        if summary['status'] not in ('completed','learning_gate_passed'):
            self.skip(name+'_dependent','Partial training cannot be a confirmed anchor');return None
        return RUNS/name/'checkpoint.pt'
    def eval(self,name,source,n=2048,names='standard,lookup,compose,hard10',split='dev',reserve=True):
        out=RUNS/name/'summary.json';self.run(name,'evaluate.py',['--source',source,'--out',out,'--n',n,'--names',names,'--split',split,'--fast'],estimated=90,required=True,reserve=reserve);return json.loads(out.read_text())
    def audit(self,name,source):self.run(name,'checks.py',['--source',source,'--out',RUNS/name/'summary.json','--fast'],90,True)
    def inline(self,name):self.stage=name;self.status();self.log(stage=name,event='started_inline')
    def continual(self,label,anchor,base_eval,flat=False,always=False):
        accepted=anchor;first=anchor;history=[];accepteds={};initial=None
        self.inline(label+'_headroom');s,_=load_system(anchor);freeze(s);s.fast_generation=not isinstance(s.core,Core)
        baseline=dict(text=text_eval(s,family='continual',n=256),bind=symbolic_eval(s,'bind',n=512),shifted=symbolic_eval(s,'shifted',n=512));atomic_json(RUNS/(label+'_headroom.json'),baseline)
        for phase,mode in enumerate(('text','bind','shifted'),1):
            if self.remaining()<2000*self.step_s+900:self.skip(label+f'_phase{phase}','Time reserved for final evaluation');break
            name=label+f'_phase{phase}';source=self.train(name,'CL',2000,accepted,mode=mode,always_open=always)
            if source is None:break
            # Same weights forced-open evaluation before gate fitting; old gates unchanged.
            self.inline(name+'_forced_open');s,_=load_system(source);freeze(s);s.fast_generation=not isinstance(s.core,Core);open_new=text_eval(s,family='continual') if mode=='text' else symbolic_eval(s,mode,n=512)
            before,_=load_system(accepted);freeze(before);before.fast_generation=not isinstance(before.core,Core);previous_new=text_eval(before,family='continual') if mode=='text' else symbolic_eval(before,mode,n=512)
            atomic_json(RUNS/name/'same_weights_open_new.json',open_new)
            if mode=='text':
                open_ci=cluster_ci(np.array(open_new['numerators'])-np.array(previous_new['numerators']),open_new['denominators'],open_new['document_ids']);open_learned=open_ci[1]<0
            else:
                open_change=paired(open_new['success_vector'],previous_new['success_vector']);open_learned=open_change['delta']>=.05 and open_change['interval'][0]>0
            if not open_learned and not always:
                atomic_json(RUNS/name/'decision.json',dict(phase=phase,mode=mode,accepted=False,reason='Always-open new bank did not establish skill gain; support gate not blamed',open_new=open_new,previous_new=previous_new))
                self.skip(label+'_later_phases','Adapter learning gate failed before positive support fitting');break
            if not always:
                if phase==1:
                    # Negative support was fitted once at P0, before any phase training (see caller).
                    pass
                fit_name=name+'_positive';ok=self.run(fit_name,'continual.py',['--source',source,'--out',RUNS/fit_name,'--positive','--mode',mode,'--tokens',1000000],estimated=500)
                if not ok:break
                candidate=RUNS/fit_name/'checkpoint.pt'
            else:candidate=source
            cand_eval=self.eval(name+'_retention',candidate);previous_eval=base_eval if phase==1 else self.eval(name+'_previous_retention',accepted);adjacent=retention(cand_eval,previous_eval);cumulative=retention(cand_eval,base_eval)
            self.inline(name+'_new_skill');s,_=load_system(candidate);freeze(s);s.fast_generation=not isinstance(s.core,Core);new=text_eval(s,family='continual') if mode=='text' else symbolic_eval(s,mode,n=512)
            if mode=='text':
                gain_ci=cluster_ci(np.array(new['numerators'])-np.array(previous_new['numerators']),new['denominators'],new['document_ids']);gain=new['nll']<previous_new['nll'] and gain_ci[1]<0
                installation_ci=cluster_ci(np.array(new['numerators'])-np.array(open_new['numerators']),new['denominators'],new['document_ids']);install=installation_ci[1]<=.02;change=dict(gain_nll=new['nll']-previous_new['nll'],gain_interval=gain_ci,installation_interval=installation_ci)
            else:
                change=paired(new['success_vector'],previous_new['success_vector']);installation=paired(new['success_vector'],open_new['success_vector']);gain=change['delta']>=.05 and change['interval'][0]>0;install=installation['interval'][0]>=-.02;change['installation']=installation
            old_tasks={};old_ok=True
            for old_mode,reference in accepteds.items():
                old_now=text_eval(s,family='continual') if old_mode=='text' else symbolic_eval(s,old_mode,n=512)
                if old_mode=='text':
                    ci=cluster_ci(np.array(old_now['numerators'])-np.array(reference['numerators']),old_now['denominators'],old_now['document_ids']);test=dict(nll_change=old_now['nll']-reference['nll'],interval=ci);old_ok&=ci[1]<=.02
                else:test=paired(old_now['success_vector'],reference['success_vector']);old_ok&=test['interval'][0]>=-.02
                old_tasks[old_mode]=test
            accepted_phase=bool(gain and install and adjacent['pass'] and cumulative['pass'] and old_ok)
            gates=gate_rates(s,Mixer(89400+phase,64,mode).batch(DEVICE)) if not always else []
            row=dict(phase=phase,mode=mode,source=str(candidate),accepted=accepted_phase,new_skill=new,change=change,adjacent_retention=adjacent,cumulative_retention=cumulative,old_task_retention=old_tasks,gates_teacher_forced=gates,interpretation='research-only sequential control' if always else 'operational promotion gate; unresolved retention is rejection')
            atomic_json(RUNS/name/'decision.json',row);history.append(row)
            if always:accepted=candidate;accepteds[mode]=new
            elif accepted_phase:accepted=candidate;accepteds[mode]=new
            else:self.skip(label+'_later_phases','New skill/gate/retention evidence insufficient; failed candidate preserved');break
        atomic_json(RUNS/(label+'_summary.json'),dict(history=history,accepted_anchor=str(accepted),baseline=baseline,partial=len(history)<3,research_only=always));return accepted
    def finish(self,status,message):
        self.stage='reporting';self.status();summary=dict(status=status,message=message,foundations=self.foundations,completed_stages=self.completed,skipped=self.skipped,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.start)),deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.deadline)),elapsed_s=time.time()-self.start,training_seed_count=1,confirmatory=False,implementation_sha256=implementation_hash())
        atomic_json(ROOT/'PRIMARY_RESULT.json',summary)
        from report import write_report
        write_report(summary);self.stage='finished';self.status(status,message=message,report=str(ROOT/'REPORT.md'));self.log(stage='finished',event=status,message=message)

    def main(self):
        for p in ('contracts.json','bridge_contracts.json','stage_contracts.json','throughput.json'):
            j=json.loads((ROOT/'tests'/p).read_text())
            if p!='throughput.json' and j['status']!='passed':raise RuntimeError('Preflight failed '+p)
            if p=='throughput.json' and not j['gradient_parity']:self.compiled=False
        lock=json.loads((ROOT/'spec/LOCK.json').read_text())
        if lock['implementation_sha256']!=implementation_hash():raise RuntimeError('Locked source changed before queue launch')
        self.inline('learning_gates')
        mem=self.train('engineering_mem128','MEM',2000,mode='simple',batch=128,learning=True,fixed=True,required=True,estimate_multiplier=2)
        if mem is None or not json.loads((RUNS/'engineering_mem128/learning_gate.json').read_text())['pass_gate']:raise GateEnd('128-instance memorization gate failed')
        for flat in (False,True):
            name='engineering_simple_flat' if flat else 'engineering_simple_hierarchy';source=self.train(name,'SIMPLE',6000,flat=flat,mode='simple',learning=True,required=True)
            if source is None or not json.loads((RUNS/name/'learning_gate.json').read_text())['pass_gate']:raise GateEnd(name+' fresh lookup/two-hop gate failed')
        # New initializations, full per-arm 24,000-update horizons; no transfer from engineering models.
        for name,flat,alpha in [('A_alpha0',False,0.),('A_delayed_ncp',False,.1),('A_flat',True,0.)]:
            source=self.train(name,'A',24000,flat=flat,alpha=alpha,required=True)
            if source is None:raise BudgetEnd(name+' complete foundation not obtained')
            self.audit(name+'_trained_contracts',source);e=self.eval(name+'_dev',source);self.foundations[name]=dict(source=str(source),competence_pass=e['competence_pass'],standard=e['symbolic']['standard']['rate'],lookup=e['symbolic']['lookup']['rate'],text={k:v['nll'] for k,v in e['text'].items()},active_parameters=e['active_parameters']);atomic_json(RUNS/'foundation_results.json',self.foundations)
        candidates=[name for name in ('A_delayed_ncp','A_alpha0') if self.foundations[name]['competence_pass']]
        if not candidates:raise GateEnd('No competent hierarchy; dependent refit/Z/search/continual stages blocked')
        # Prefer competent delayed NCP; disclose fallback. Selection is development-only.
        selected=candidates[0];anchor=Path(self.foundations[selected]['source']);base_eval=json.loads((RUNS/(selected+'_dev')/'summary.json').read_text());alpha=.1 if selected=='A_delayed_ncp' else 0.;flat=Path(self.foundations['A_flat']['source'])
        self.inline('anchor_selection');atomic_json(RUNS/'ANCHOR.json',dict(selected=selected,source=str(anchor),reason='Prefer NCP if competent; otherwise alpha-zero component study; no NCP success claim on fallback'))
        # Optional hardness continuation; competence/retention control prevents sacrificing mastered simple tasks.
        hard=self.train('A_hard_continuation','HARD',8000,anchor,alpha,mode='hard',context=384,estimate_multiplier=2)
        if hard:
            e=self.eval('A_hard_dev',hard);r=retention(e,base_eval);atomic_json(RUNS/'A_hard_continuation/acceptance.json',r)
            if r['pass']:anchor=hard;base_eval=e
        # Compare dense and recurrent replacement, retaining the original anchor if preservation fails.
        accepted_refits=[]
        for core in ('dense','recurrent'):
            source=self.train('B_refit_'+core,'REFIT',6000,anchor,alpha,core=core)
            if source:
                self.run('B_refit_'+core+'_contracts','checks.py',['--source',source,'--out',RUNS/('B_refit_'+core+'_contracts')/'summary.json'],90,True)
                e=self.eval('B_refit_'+core+'_dev',source);r=retention(e,base_eval);atomic_json(RUNS/('B_refit_'+core)/'acceptance.json',r)
                if r['pass']:accepted_refits.append((core,source,e))
        control=self.train('B_no_teacher_control','REFIT',6000,anchor,alpha,no_distill=True)
        if control:self.eval('B_no_teacher_control_dev',control)
        if accepted_refits:
            core,anchor,base_eval=accepted_refits[0] # Dense first avoids architecture selection on test data.
        z=self.train('C_exploration','Z',6000,anchor,alpha,mode='z')
        search_ready=False;z_anchor=anchor
        if z:
            self.inline('C_exploration_coverage');s,_=load_system(z);freeze(s);s.fast_generation=not isinstance(s.core,Core);coverage=symbolic_eval(s,'hard10',n=512,latent_draws=8);e=self.eval('C_exploration_retention',z);r=retention(e,base_eval);atomic_json(RUNS/'C_exploration/coverage.json',coverage);atomic_json(RUNS/'C_exploration/acceptance.json',r)
            if r['pass']:z_anchor=z;base_eval=e
            search_ready=bool(r['pass'] and coverage['coverage_gate'])
        # Main continual protocol starts at a competent accepted version, not at a failed exploratory candidate.
        neg=self.run('D_negative_support','continual.py',['--source',z_anchor,'--out',RUNS/'D_negative_support','--tokens',1000000],estimated=500)
        if neg:
            clbase=RUNS/'D_negative_support/checkpoint.pt';self.continual('D_lsl',clbase,base_eval)
            self.continual('D_always_open',z_anchor,base_eval,always=True)
        if self.foundations['A_flat']['competence_pass'] and self.remaining()>5000:
            flat_eval=json.loads((RUNS/'A_flat_dev/summary.json').read_text());ok=self.run('D_flat_negative','continual.py',['--source',flat,'--out',RUNS/'D_flat_negative','--tokens',1000000],estimated=500)
            if ok:self.continual('D_flat_lsl',RUNS/'D_flat_negative/checkpoint.pt',flat_eval,flat=True)
        else:self.skip('D_flat_lsl','Baseline competence or remaining full-trajectory budget insufficient')
        # Search runs only when the actual zero-anchor and independent prior reach passed fixed development gates.
        if search_ready:
            seconds=min(3600,max(0,self.remaining()-60))
            if seconds>=900:self.run('E_search','search_experiment.py',['--source',z_anchor,'--flat-source',flat,'--out',RUNS/'E_search','--seconds',seconds],estimated=seconds)
            else:self.skip('E_search','Insufficient paired-comparison wall-time reserve')
        else:self.skip('E_search','Exploration coverage/retention gate not passed; no expensive zero-success study')
        self.final_tests();self.finish('completed_pilot','Single-seed V8 pilot completed; see stage decisions and explicitly skipped controls')
    def final_tests(self):
        self.inline('final_testing')
        for name,v in self.foundations.items():
            if self.remaining(False)<180:self.skip(name+'_sealed_test','Budget exhausted');continue
            self.eval(name+'_sealed_test',v['source'],n=2048,names='standard,lookup,compose,hard10,hard14,long18,pointer,bind,shifted',split='test',reserve=False)
        # Every promoted continual version has sealed new-task and retention measurement, without selecting again.
        for p in sorted(RUNS.glob('D_*_summary.json')):
            j=json.loads(p.read_text());source=j['accepted_anchor']
            if self.remaining(False)<180:self.skip(p.stem+'_sealed_test','Budget exhausted');continue
            e=self.eval(p.stem+'_sealed_test',source,n=2048,names='standard,lookup,compose,hard10,bind,shifted',split='test',reserve=False);self.inline(p.stem+'_text_test');s,_=load_system(source);freeze(s);atomic_json(RUNS/(p.stem+'_continual_text_test.json'),text_eval(s,split='test',family='continual'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=12);args=ap.parse_args();q=Queue(args.hours)
    try:q.main()
    except GateEnd as exc:
        q.log(stage=q.stage,event='gate_failure',reason=str(exc));q.final_tests();q.finish('completed_gate_failed',str(exc))
    except BudgetEnd as exc:q.final_tests();q.finish('completed_partial_budget',str(exc))
    except KeyboardInterrupt:q.finish('stopped_by_user','Queue interrupted by user; finite atomic checkpoints preserved')
    except Exception as exc:
        atomic_json(RUNS/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),stage=q.stage,driver_pid=os.getpid()));q.finish('failed',str(exc));raise
