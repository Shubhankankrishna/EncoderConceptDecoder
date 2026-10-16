"""User-authorized full-exposure pilot: score thresholds annotate, never abort it."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from drive import Queue as OriginalQueue, BudgetEnd
from runtime import *
from model import *
from data import Mixer,digest,DATA
from evaluate import retention,text_eval,symbolic_eval,paired,cluster_ci
from continual import gate_rates
RUNS=ROOT/'runs';CODE=ROOT/'code'

class Queue(OriginalQueue):
    def __init__(self,hours=12,continue_saved=False):
        super().__init__(hours);self.reserve=5400;self.endpoints={};self.decisions=[];self.partial=[];self.policy='diagnostic thresholds; no competence-based stopping'
        self.continue_saved=continue_saved;self.recovery=None
        if continue_saved:
            self.recovery=json.loads((ROOT/'spec/CONTINUATION.json').read_text())
            self.start=float(self.recovery['original_start_epoch']);self.deadline=float(self.recovery['original_deadline_epoch'])
            if time.time()>=self.deadline:raise RuntimeError('Original experiment budget expired; continuation cannot reset it')
            self.completed=['A_alpha0']
            self.log(stage='recovery',event='authorized_continuation',original_deadline_epoch=self.deadline,recovery=self.recovery)
        elif (RUNS/'status.json').exists():raise RuntimeError('Existing experiment state; use the authorized continuation instead of overwriting it')
        os.environ['V8_DEADLINE_EPOCH']=str(self.deadline-self.reserve-30)
    def status(self,status='running',**extra):
        if getattr(self,'continue_saved',False):extra['recovery']='eager decoding; reuse completed alpha-zero; original budget retained'
        return super().status(status,**extra)
    def run(self,name,script,args,estimated=0,required=False,reserve=True):
        lock=json.loads((ROOT/'spec/LOCK.json').read_text())
        if lock['implementation_sha256']!=implementation_hash():raise RuntimeError('Locked revision source changed')
        if lock['data_manifest_sha256']!=digest(DATA/'manifest.json'):raise RuntimeError('Locked dataset manifest changed')
        if self.stopped:raise KeyboardInterrupt('User stopped queue')
        cutoff=self.deadline-(self.reserve if reserve else 30)
        if script=='search_experiment.py':cutoff=min(cutoff,time.time()+float(args[args.index('--seconds')+1]))
        if cutoff-time.time()<max(estimated,30):
            self.skip(name,'Insufficient time in declared budget, unrelated to score')
            if required:raise BudgetEnd(name)
            return False
        if script=='train.py' and args[args.index('--stage')+1]=='A':args=[*args,'--development-curves']
        if script=='train.py' and args[args.index('--stage')+1]=='CL':
            metadata=torch.load(args[args.index('--source')+1],map_location='cpu',weights_only=False)['metadata']
            key='CL_FLAT/dense' if metadata['flat'] else 'CL_GATED/dense' if metadata['phases'] else 'CL/dense'
            safe=next(v['compiled_safe'] for v in json.loads((ROOT/'tests/stage_contracts.json').read_text())['checks'] if v['stage']==key)
            args=[v for v in args if v!='--compiled']
            if safe and self.compiled:args+=[ '--compiled' ]
        self.stage=name;folder=RUNS/name;folder.mkdir(parents=True,exist_ok=True);command=[sys.executable,str(CODE/script),*map(str,args)]
        env=dict(os.environ,V8_DEADLINE_EPOCH=str(cutoff-20));self.log(stage=name,event='started',command=command,stage_deadline_epoch=cutoff);self.status(policy=self.policy)
        expired=False;termination_at=None
        with open(folder/'process.log','a') as f:
            self.child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parent,env=env);self.status(policy=self.policy)
            while self.child.poll() is None:
                time.sleep(1)
                if self.stopped: self.child.terminate()
                if time.time()>=cutoff-5:
                    expired=True
                    if termination_at is None:self.child.terminate();termination_at=time.time()
                    elif time.time()-termination_at>15:self.child.kill()
            rc=self.child.returncode
        self.child=None;self.status(policy=self.policy)
        if self.stopped:raise KeyboardInterrupt('User stopped queue')
        if expired or rc==75:
            self.partial.append(dict(stage=name,reason='Wall-time boundary',exit_code=rc));self.log(stage=name,event='partial_budget',exit_code=rc)
            if required:raise BudgetEnd(name)
            return False
        if rc!=0:raise RuntimeError(f'{name} exited {rc}; inspect {folder}/process.log')
        self.completed.append(name);self.log(stage=name,event='finished');return True
    def train(self,name,stage,steps,source=None,alpha=0.,flat=False,**kwargs):
        if self.continue_saved and name=='A_alpha0':
            path=RUNS/name/'checkpoint.pt';summary=json.loads((RUNS/name/'train_summary.json').read_text())
            expected=dict(stage='A',steps=24000,completed_steps=24000,batch=64,context=192,seed=811,alpha=0.,flat=False,status='completed',data_sha256=digest(DATA/'manifest.json'))
            if stage!='A' or steps!=24000 or source is not None or flat or alpha!=0.:raise RuntimeError('Completed-foundation reuse setting mismatch')
            if any(summary.get(k)!=v for k,v in expected.items()):raise RuntimeError('Completed-foundation summary does not match continuation')
            if digest(path)!=self.recovery['completed_checkpoint_sha256']:raise RuntimeError('Completed checkpoint changed before reuse')
            for relative,sha in self.recovery['unchanged_training_files_sha256'].items():
                if digest(ROOT/relative)!=sha:raise RuntimeError('Training recipe changed before reuse: '+relative)
            if summary['implementation_sha256']!=self.recovery['original_implementation_sha256']:raise RuntimeError('Completed checkpoint provenance mismatch')
            self.log(stage=name,event='reused_completed_checkpoint',source=str(path),updates=24000,graph_examples=summary['symbolic_exposure']['graph'],original_implementation_sha256=summary['implementation_sha256'],current_implementation_sha256=implementation_hash())
            return path
        # Dedicated production path measurements select execution; exposure remains fixed.
        multiplier=1.6 if flat else 1.
        if stage=='Z':multiplier=1.5
        if kwargs.get('core')=='recurrent':multiplier=1.3
        return super().train(name,stage,steps,source,alpha,flat,estimate_multiplier=multiplier,**kwargs)
    def eval(self,name,source,n=2048,names='standard,lookup,compose,pointer,hard10',split='dev',reserve=True):
        out=RUNS/name/'summary.json'
        ok=self.run(name,'evaluate.py',['--source',source,'--out',out,'--n',n,'--names',names,'--split',split,'--fast'],estimated=60,required=False,reserve=reserve)
        if not ok or not out.exists():return None
        result=json.loads(out.read_text())
        if result.get('status')!='completed':self.partial.append(dict(stage=name,reason='Incomplete evaluation'));return None
        return result
    def audit(self,name,source):
        if self.continue_saved:name+='_recovery2'
        self.run(name,'revision_checks.py',['--source',source,'--out',RUNS/name/'summary.json'],estimated=90,required=True)
    def endpoint(self,name,source,development=None,**extra):
        self.endpoints[name]=dict(source=str(source),development=development,**extra);atomic_json(RUNS/'endpoints.json',self.endpoints)
    def decision(self,name,row):
        self.decisions.append(dict(stage=name,**row));atomic_json(RUNS/name/'decision.json',row)
    def score(self,e):
        return (e['symbolic']['standard']['rate'],e['symbolic']['hard10']['rate'],e['symbolic']['lookup']['rate'],-e['text']['edu']['nll']-e['text']['ts']['nll'])
    def continual(self,label,anchor,base_eval,flat=False,always=False):
        current=anchor;promoted=anchor;history=[];earlier={};previous_eval=base_eval
        for phase,mode in enumerate(('text','bind','shifted'),1):
            name=f'{label}_phase{phase}'
            if self.remaining()<2000*self.step_s*(1.6 if flat else 1.)+800:self.skip(name,'Final-test reserve; no score-based stop');break
            source=self.train(name,'CL',2000,current,mode=mode,always_open=always)
            if not source:break
            self.inline(name+'_same_weights_open')
            s,_=load_system(source);freeze(s);s.fast_generation=not isinstance(s.core,Core)
            before,_=load_system(current);freeze(before);before.fast_generation=not isinstance(before.core,Core)
            open_new=text_eval(s,family='continual') if mode=='text' else symbolic_eval(s,mode,n=512)
            previous_new=text_eval(before,family='continual') if mode=='text' else symbolic_eval(before,mode,n=512)
            del s,before
            candidate=source
            if not always:
                fit_name=name+'_positive'
                if not self.run(fit_name,'continual.py',['--source',source,'--out',RUNS/fit_name,'--positive','--mode',mode,'--tokens',1000000],estimated=300):break
                candidate=RUNS/fit_name/'checkpoint.pt'
            self.audit(name+'_trained_contracts',candidate)
            e=self.eval(name+'_dev',candidate)
            if e is None:break
            adjacent=retention(e,previous_eval);cumulative=retention(e,base_eval)
            self.inline(name+'_new_skill');s,_=load_system(candidate);freeze(s);s.fast_generation=not isinstance(s.core,Core)
            new=text_eval(s,family='continual') if mode=='text' else symbolic_eval(s,mode,n=512)
            if mode=='text':
                ci=cluster_ci(np.array(new['numerators'])-np.array(previous_new['numerators']),new['denominators'],new['document_ids']);installation=cluster_ci(np.array(new['numerators'])-np.array(open_new['numerators']),new['denominators'],new['document_ids']);gain=ci[1]<0;install=installation[1]<=.02;change=dict(nll_delta=new['nll']-previous_new['nll'],interval=ci,installation_interval=installation)
            else:
                change=paired(new['success_vector'],previous_new['success_vector']);installation=paired(new['success_vector'],open_new['success_vector']);gain=change['delta']>=.05 and change['interval'][0]>0;install=installation['interval'][0]>=-.02;change['installation']=installation
            old_tests={};old_ok=True
            for old_mode,reference in earlier.items():
                now=text_eval(s,family='continual') if old_mode=='text' else symbolic_eval(s,old_mode,n=512)
                if old_mode=='text':
                    ci=cluster_ci(np.array(now['numerators'])-np.array(reference['numerators']),now['denominators'],now['document_ids']);test=dict(nll_delta=now['nll']-reference['nll'],interval=ci);old_ok&=ci[1]<=.02
                else:test=paired(now['success_vector'],reference['success_vector']);old_ok&=test['interval'][0]>=-.02
                old_tests[old_mode]=test
            qualified=bool(gain and install and adjacent['pass'] and cumulative['pass'] and old_ok)
            row=dict(phase=phase,mode=mode,source=str(candidate),evidence_qualified=qualified,continued_despite_threshold=not qualified,new_skill=new,open_new_skill=open_new,previous_new_skill=previous_new,change=change,adjacent_retention=adjacent,cumulative_retention=cumulative,old_task_retention=old_tests,gates_teacher_forced=gate_rates(s,Mixer(89400+phase,64,mode).batch(DEVICE)) if not always else [],interpretation='Exploratory trajectory; numerical scores never abort later phases')
            del s
            self.decision(name,row);history.append(row);self.endpoint(name,candidate,e,continual=True,qualified=qualified)
            # Keep the research trajectory intact. Promotion additionally requires every prior phase.
            if qualified and len(history)==phase and all(v['evidence_qualified'] for v in history):promoted=candidate
            current=candidate;previous_eval=e;earlier[mode]=new
        atomic_json(RUNS/(label+'_summary.json'),dict(history=history,research_anchor=str(current),promoted_anchor=str(promoted),partial=len(history)<3,always_open_control=always))
        return current
    def main(self):
        lock=json.loads((ROOT/'spec/LOCK.json').read_text())
        if lock['implementation_sha256']!=implementation_hash():raise RuntimeError('Preflight revision mismatch')
        for name in ('contracts.json','stage_contracts.json','revision_contracts.json','throughput.json'):
            j=json.loads((ROOT/'tests'/name).read_text())
            if j.get('implementation_sha256')!=implementation_hash():raise RuntimeError('Stale preflight '+name)
            if name!='throughput.json' and j['status']!='passed':raise RuntimeError('Preflight failure '+name)
        # User explicitly replaced early ability gates with full-exposure diagnostics.
        self.log(stage='policy',event='user_override',reason='Run for12h; early competence and coverage scores do not stop research',prior_run=str(ROOT.parent/'v8_tiny'))
        for name,flat,alpha in [('A_alpha0',False,0.),('A_delayed_ncp',False,.1),('A_flat',True,0.)]:
            source=self.train(name,'A',24000,flat=flat,alpha=alpha,required=True)
            if source is None:raise BudgetEnd('Incomplete foundation '+name)
            self.audit(name+'_trained_contracts',source);e=self.eval(name+'_dev',source)
            if e is None:raise BudgetEnd('Incomplete foundation development '+name)
            self.foundations[name]=dict(source=str(source),competence_pass=e['competence_pass'],standard=e['symbolic']['standard']['rate'],lookup=e['symbolic']['lookup']['rate'],text={k:v['nll'] for k,v in e['text'].items()},active_parameters=e['active_parameters'],completed_steps=24000,graph_examples=576000)
            self.endpoint(name,source,e,foundation=True,qualified=e['competence_pass']);atomic_json(RUNS/'foundation_results.json',self.foundations)
        selected=max(('A_delayed_ncp','A_alpha0'),key=lambda name:self.score(self.endpoints[name]['development']))
        anchor=Path(self.foundations[selected]['source']);base=self.endpoints[selected]['development'];alpha=.1 if selected=='A_delayed_ncp' else 0.;flat=Path(self.foundations['A_flat']['source'])
        atomic_json(RUNS/'ANCHOR.json',dict(selected=selected,source=str(anchor),competence_pass=base['competence_pass'],selection='Development standard/hard/lookup rates then text NLL; research lineage, not confirmation'))
        refits=[]
        for core in ('dense','recurrent'):
            name='B_refit_'+core;source=self.train(name,'REFIT',6000,anchor,alpha,core=core)
            if source:
                self.audit(name+'_trained_contracts',source);e=self.eval(name+'_dev',source)
                if e:
                    r=retention(e,base);self.decision(name,dict(retention=r,competence_pass=e['competence_pass'],research_only=not r['pass']));self.endpoint(name,source,e,qualified=r['pass']);refits.append((name,source,e))
        if refits:
            # A failed preservation threshold is recorded rather than terminating this requested study.
            chosen=max(refits,key=lambda v:self.score(v[2]));anchor=chosen[1];base=chosen[2]
            atomic_json(RUNS/'REFIT_LINEAGE.json',dict(selected=chosen[0],source=str(anchor),research_only=not self.endpoints[chosen[0]]['qualified']))
        z=self.train('C_exploration','Z',6000,anchor,alpha,mode='z')
        z_anchor=anchor
        if z:
            self.audit('C_exploration_trained_contracts',z);e=self.eval('C_exploration_dev',z)
            if e:
                self.inline('C_prior_coverage');s,_=load_system(z);freeze(s);s.fast_generation=not isinstance(s.core,Core);coverage=symbolic_eval(s,'hard10',n=512,latent_draws=8);del s
                atomic_json(RUNS/'C_exploration/coverage.json',coverage);r=retention(e,base);self.decision('C_exploration',dict(retention=r,coverage_pass=coverage['coverage_gate'],research_only=not(r['pass'] and coverage['coverage_gate'])));self.endpoint('C_exploration',z,e,qualified=r['pass'] and coverage['coverage_gate']);z_anchor=z;base=e
            if self.remaining()>1200:
                seconds=min(1800,self.remaining()-300)
                self.run('E_search','search_experiment.py',['--source',z_anchor,'--flat-source',flat,'--out',RUNS/'E_search','--seconds',seconds],estimated=60)
            else:self.skip('E_search','Wall-time reserve; no coverage-based stop')
        else:self.skip('E_search','No completed trained Z checkpoint within budget')
        neg=self.run('D_negative_support','continual.py',['--source',z_anchor,'--out',RUNS/'D_negative_support','--tokens',1000000],estimated=300)
        if neg:self.continual('D_lsl',RUNS/'D_negative_support/checkpoint.pt',base)
        self.continual('D_always_open',z_anchor,base,always=True)
        if self.remaining()>4000:
            ok=self.run('D_flat_negative','continual.py',['--source',flat,'--out',RUNS/'D_flat_negative','--tokens',1000000],estimated=300)
            if ok:self.continual('D_flat_lsl',RUNS/'D_flat_negative/checkpoint.pt',self.endpoints['A_flat']['development'],flat=True)
        else:self.skip('D_flat_lsl','No full-trajectory time allocation remains')
        control=self.train('B_no_teacher_control','REFIT',6000,Path(self.foundations[selected]['source']),alpha,no_distill=True)
        if control:
            e=self.eval('B_no_teacher_control_dev',control)
            if e:self.endpoint('B_no_teacher_control',control,e,control=True)
        # Use spare learning time for symmetric exposure extensions, never for lowering a threshold.
        latest={name:Path(row['source']) for name,row in self.foundations.items()};round_number=0
        while self.remaining()>2000*self.step_s*4.2+600:
            round_number+=1
            for foundation,source in list(latest.items()):
                name=f'F_extension{round_number}_{foundation}';is_flat=foundation=='A_flat';a=.1 if foundation=='A_delayed_ncp' else 0.
                extended=self.train(name,'HARD',2000,source,a,flat=is_flat)
                if extended is None:break
                latest[foundation]=extended;e=self.eval(name+'_dev',extended,n=512)
                self.endpoint(name,extended,e,foundation_extension=True,foundation_parent=foundation,total_updates=24000+round_number*2000)
            else:continue
            break
        self.final_tests();self.finish('completed_extended_pilot','Full-exposure diagnostic pilot; low score thresholds did not stop training')
    def final_tests(self):
        self.inline('final_testing');os.environ['V8_DEADLINE_EPOCH']=str(self.deadline-30)
        for name,row in self.endpoints.items():
            if self.remaining(False)<120:self.skip(name+'_sealed_test','Final deadline');continue
            e=self.eval(name+'_sealed_test',row['source'],n=2048,names='standard,lookup,compose,pointer,hard10,hard14,long18,bind,shifted',split='test',reserve=False)
            if e is not None and row.get('continual'):
                self.inline(name+'_continual_text_test');s,_=load_system(row['source']);freeze(s)
                try:atomic_json(RUNS/(name+'_continual_text_test.json'),text_eval(s,split='test',family='continual'))
                except TimeoutError:self.partial.append(dict(stage=name+'_continual_text_test',reason='Final deadline'))
                del s
    def finish(self,status,message):
        self.stage='reporting';self.status(policy=self.policy)
        result=dict(status=status,message=message,policy=self.policy,foundations=self.foundations,endpoints=self.endpoints,decisions=self.decisions,completed_stages=self.completed,skipped=self.skipped,partial=self.partial,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.start)),deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.deadline)),elapsed_s=time.time()-self.start,training_seed_count=1,confirmatory=False,implementation_sha256=implementation_hash(),recovery=self.recovery)
        atomic_json(ROOT/'PRIMARY_RESULT.json',result)
        from report import write_report
        write_report(result);self.stage='finished';self.status(status,message=message,report=str(ROOT/'REPORT.md'),policy=self.policy);self.log(stage='finished',event=status,message=message)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=12);ap.add_argument('--continue-saved',action='store_true');a=ap.parse_args();q=Queue(a.hours,a.continue_saved)
    try:q.main()
    except (BudgetEnd,TimeoutError) as exc:
        q.partial.append(dict(stage=q.stage,reason=str(exc)))
        try:q.final_tests()
        except TimeoutError:pass
        q.finish('completed_partial_budget',str(exc))
    except KeyboardInterrupt:q.finish('stopped_by_user','User interrupted the revised queue; checkpoints retained')
    except Exception as exc:
        atomic_json(RUNS/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),stage=q.stage));q.finish('failed',str(exc));raise
