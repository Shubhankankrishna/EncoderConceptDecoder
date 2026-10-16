"""One serial, budget-limited experiment queue; monitors never own training jobs."""
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from data import ROOT,DATA,T,K,digest
from runtime import atomic_json,Log,implementation_hash,DEVICE
from evaluate import evaluate,load_eval,search_eval,paired,weighted_nll_pair,symbolic_metrics
from generation import generate
from audit import audit

def score(dev):return .5*(dev['text']['nll']+dev['symbolic']['standard']['nll'])
def release():
    gc.collect()
    if DEVICE=='mps':torch.mps.empty_cache()

class Queue:
    def __init__(self,smoke=False):
        if not smoke and (ROOT/'runs/status.json').exists():raise RuntimeError('Existing queue status: refusing duplicate launch or overwritten experiment')
        self.smoke=smoke;self.root=ROOT/'tests/pipeline' if smoke else ROOT/'runs';self.root.mkdir(parents=True,exist_ok=True)
        self.log=Log(self.root/'queue.jsonl');self.start=time.time();self.deadline=self.start+(3600 if smoke else 24*3600);self.training_end=self.deadline-(600 if smoke else 6*3600)
        self.sha=implementation_hash();self.skipped=[];self.records=[];self.stage='initializing';self.pid=None
        self.benchmark=json.loads((ROOT/'runs/calibration/benchmark.json').read_text()) if not smoke else dict(batch=4,compiled=False,compiled_static_generation=False)
        self.batch=4 if smoke else self.benchmark['batch'];self.fast=not smoke and self.benchmark['compiled_static_generation']
        rates=[r['scored_targets_per_s'] for r in self.benchmark.get('training',[]) if r['stage'] in ('A','flat') and r['compiled']==self.benchmark['compiled']]
        slow=min(rates) if rates else 1000
        # Freeze a conservative common target budget before any pilot outcome.
        self.foundation_tokens=4000 if smoke else int(min(40000000,max(16000000,slow*3600*1.25))//1000000*1000000)
        self.pilot_tokens=4000 if smoke else 3000000;self.refit_tokens=4000 if smoke else 8000000;self.adapter_tokens=4000 if smoke else 6000000
        self.plan=dict(started_epoch=self.start,deadline_epoch=self.deadline,training_end_epoch=self.training_end,seconds_ceiling=self.deadline-self.start,
            implementation_sha256=self.sha,data_manifest_sha256=digest(DATA/'manifest.json'),spec_sha256=digest(ROOT/'spec/draft_v7.md'),benchmark=self.benchmark,
            common_foundation_target_tokens=self.foundation_tokens,pilot_target_tokens=self.pilot_tokens,refit_target_tokens=self.refit_tokens,adapter_target_tokens=self.adapter_tokens,
            batch=self.batch,foundation_seeds=[171,272,373] if not smoke else [171],source_scope='independent V7 implementation; ordinary parameter-matched Transformer stack',
            test_policy='lock all training/inference choices on development before opening test outcomes',smoke=smoke)
        self.data_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in DATA.iterdir() if p.is_file()}
        legacy=ROOT.parent/'v5_tiny/data/ts_train.npy';self.data_stats[str(legacy)]=(legacy.stat().st_size,legacy.stat().st_mtime_ns)
        self.spec_hash=self.plan['spec_sha256'];self.manifest_hash=self.plan['data_manifest_sha256']
        manifest=json.loads((DATA/'manifest.json').read_text())
        for split,meta in manifest['splits'].items():
            if digest(DATA/f'edu_{split}.npy')!=meta['sha256']:raise RuntimeError('Text split hash mismatch: '+split)
        for name,meta in manifest['symbolic'].items():
            if digest(DATA/(name+'.npz'))!=meta['sha256']:raise RuntimeError('Symbolic split hash mismatch: '+name)
        if digest(legacy)!=manifest['legacy_train_sha256']:raise RuntimeError('Legacy training corpus hash mismatch')
        atomic_json(self.root/'plan.json',self.plan);self.status('running')
    def status(self,status='running',**kw):
        atomic_json(self.root/'status.json',dict(status=status,stage=self.stage,driver_pid=os.getpid(),child_pid=self.pid,started_epoch=self.start,deadline_epoch=self.deadline,
            elapsed_s=time.time()-self.start,remaining_s=max(0,self.deadline-time.time()),implementation_sha256=self.sha,skipped=self.skipped,**kw))
    def check_source(self):
        if implementation_hash()!=self.sha:raise RuntimeError('Source changed after experiment lock; refusing mixed implementations')
        if digest(DATA/'manifest.json')!=self.manifest_hash or digest(ROOT/'spec/draft_v7.md')!=self.spec_hash:raise RuntimeError('Frozen manifest or specification changed')
        for path,expected in self.data_stats.items():
            p=Path(path)
            if not p.exists() or (p.stat().st_size,p.stat().st_mtime_ns)!=expected:raise RuntimeError('Frozen data changed: '+path)
    def skip(self,name,reason):self.skipped.append(dict(name=name,reason=reason));self.log(event='skipped',name=name,reason=reason);self.status()
    def checkpoint(self,name):return self.root/name/'checkpoint.pt'
    def train(self,name,stage='A',source=None,seed=171,alpha=.1,lr=.001,tokens=None,beta=.25,flat=False,plain=False,pilot=False):
        self.check_source();tokens=tokens or self.foundation_tokens
        if time.time()>=self.training_end:self.skip(name,'Training reserve exhausted; final testing retained');return None
        self.stage=name;out=self.root/name;out.mkdir(parents=True,exist_ok=True)
        batch=self.batch if stage!='B1' else max(4,self.batch//2)
        estimated=tokens/3000+120;seconds=min(self.training_end-time.time(),max(600,estimated))
        args=[sys.executable,str(ROOT/'code/train.py'),'--stage',stage,'--out',str(out),'--seed',str(seed),'--tokens',str(tokens),'--batch',str(batch),'--seconds',str(seconds),'--alpha',str(alpha),'--lr',str(lr),'--beta',str(beta)]
        if source:args+=['--source',str(source)]
        if flat:args+=['--flat']
        if plain:args+=['--plain']
        if self.smoke:args+=['--mini','--max-steps','2']
        if not self.benchmark['compiled']:args+=['--eager']
        started=time.time();self.log(event='started',name=name,stage=stage,command=args)
        with open(out/'stdout.log','w') as log:
            p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);self.pid=p.pid;self.status()
            while p.poll() is None:
                time.sleep(5);self.status()
            self.pid=None
        if p.returncode:
            f=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else dict(error='child exit '+str(p.returncode),log=str(out/'stdout.log'))
            if pilot and 'FloatingPointError' in f.get('error',''):
                self.skip(name,'Numerically rejected pilot: '+f['error']);return None
            raise RuntimeError(json.dumps(f))
        summary=json.loads((out/'train_summary.json').read_text());self.records.append(dict(name=name,train=summary));self.log(event='finished',name=name,status=summary['status'],seconds=time.time()-started)
        self.status();return out/'checkpoint.pt'
    def eval(self,name,path,test=False,quick=False,latent=0):
        self.check_source();self.stage=name+('_test' if test else '_dev');self.status();out=Path(path).parent/('test.json' if test else 'dev.json')
        if out.exists():return json.loads(out.read_text())
        if time.time()>=self.deadline:self.skip(self.stage,'Experiment deadline reached');return None
        n=4 if self.smoke else (64 if quick else (512 if test else 128));nt=4 if self.smoke else (64 if quick else (1024 if test else 256))
        suites=('standard',) if quick else ('standard','hard','long','pointer','pointer_long','lookup','layout','compose')
        r=evaluate(path,out,'test' if test else 'dev',n,nt,self.batch,self.fast,suites,latent);release();self.log(event='evaluation',name=name,split=r['split'],score=score(r))
        if not self.smoke:
            from report import build_report
            build_report(self.root,ROOT/'REPORT.md')
        return r
    def audit(self,name,path):
        self.stage=name+'_audit';self.status();r=audit(path,Path(path).parent/'audit.json');release();return r

def run(q):
    seeds=q.plan['foundation_seeds'];pilots=[]
    grid=[('ncp',.1,lr,False,False) for lr in (.0005,.001,.002)]+[('flat',0.,lr,True,False) for lr in (.0005,.001,.002)]
    if q.smoke:grid=[grid[1],grid[4]]
    for family,alpha,lr,flat,plain in grid:
        name=f'pilot_{family}_lr{lr:g}';p=q.train(name,seed=970,alpha=alpha,lr=lr,tokens=q.pilot_tokens,flat=flat,plain=plain,pilot=True)
        if p:
            dev=q.eval(name,p,quick=True);pilots.append(dict(family=family,alpha=alpha,lr=lr,score=score(dev),checkpoint=str(p),plain=plain))
    if not pilots:raise RuntimeError('No finite foundation pilot')
    chosen={f:min([r for r in pilots if r['family']==f],key=lambda r:r['score']) for f in ('ncp','flat')}
    if not q.smoke:
        for name,alpha,plain in [('alpha001',.01,False),('alpha1',1.,False),('alpha0',0.,False),('plain_ncp',.1,True),('plain_alpha0',0.,True)]:
            p=q.train('pilot_'+name,seed=970,alpha=alpha,lr=chosen['ncp']['lr'],tokens=q.pilot_tokens,plain=plain,pilot=True)
            if p:
                dev=q.eval('pilot_'+name,p,quick=True);pilots.append(dict(family='ncp' if not plain and alpha>0 else name,alpha=alpha,lr=chosen['ncp']['lr'],score=score(dev),checkpoint=str(p),plain=plain))
        chosen['ncp']=min([r for r in pilots if r['family']=='ncp'],key=lambda r:r['score'])
    atomic_json(q.root/'training_lock.json',dict(pilots=pilots,selected=chosen,selection='lowest balanced fresh-text and standard-symbolic development NLL; no test access',common_tokens=q.foundation_tokens))
    foundations={};devs={};lineages={};controls={}
    for seed in seeds:
        # All three arms consume the same data RNG stream and actual target budget.
        for family in ('flat','alpha0','ncp'):
            sel=chosen['flat'] if family=='flat' else chosen['ncp'];name=f'A_{family}_s{seed}'
            p=q.train(name,seed=seed,alpha=sel['alpha'] if family=='ncp' else 0.,lr=sel['lr'],flat=family=='flat')
            if p:foundations[(family,seed)]=p;devs[(family,seed)]=q.eval(name,p);q.audit(name,p)
        p=foundations.get(('ncp',seed))
        if p:
            b0=q.train(f'B0_ncp_s{seed}','B0',p,seed,chosen['ncp']['alpha'],chosen['ncp']['lr'],q.refit_tokens)
            if b0:
                d=q.eval(f'B0_ncp_s{seed}',b0);q.audit(f'B0_ncp_s{seed}',b0);base=devs[('ncp',seed)]
                txt=weighted_nll_pair(d['text'],base['text']);sym_nll=(sum(d['symbolic']['standard']['sums'])/sum(d['symbolic']['standard']['counts'])-base['symbolic']['standard']['nll']);success=paired(d['symbolic']['standard']['success'],base['symbolic']['standard']['success'])
                accepted=txt['ci95'][1]<=.02 and sym_nll<=.02 and success['ci95'][0]>=-.02
                atomic_json(b0.parent/'acceptance.json',dict(accepted=accepted,text=txt,symbolic_nll_difference=sym_nll,standard_success=success,
                    action='advance refit' if accepted else 'retain foundation as anchor; refit failure still published'))
                anchor=b0 if accepted else p
                b1=q.train(f'B1_ncp_s{seed}','B1',anchor,seed,chosen['ncp']['alpha'],chosen['ncp']['lr'],q.adapter_tokens)
                if b1:lineages[seed]=dict(foundation=str(p),refit=str(b0),refit_accepted=accepted,anchor=str(anchor),adapters=str(b1));q.eval(f'B1_ncp_s{seed}',b1,latent=2);q.audit(f'B1_ncp_s{seed}',b1)
    first=seeds[0]
    if first in lineages:
        l=lineages[first];anchor=l['anchor'];p=l['foundation'];common=q.adapter_tokens
        matrix=[('direct_Z','B1',p,.25),('mean_Z','B1',anchor,1.),('best_Z','B1',anchor,0.),('det_extra','det_extra',anchor,.25),('full_extra','full_extra',p,.25)]
        for name,stage,src,beta in matrix:
            if q.smoke and name!='mean_Z':continue
            cp=q.train(name,stage,src,first,chosen['ncp']['alpha'],chosen['ncp']['lr'],common,beta)
            if cp:controls[name]=str(cp);q.eval(name,cp,latent=2 if stage=='B1' else 0);q.audit(name,cp)
        a0=foundations.get(('alpha0',first))
        if a0 and not q.smoke:
            b0=q.train('B0_alpha0','B0',a0,first,0.,chosen['ncp']['lr'],q.refit_tokens)
            if b0:
                d=q.eval('B0_alpha0',b0);q.audit('B0_alpha0',b0);base=devs[('alpha0',first)];txt=weighted_nll_pair(d['text'],base['text']);suc=paired(d['symbolic']['standard']['success'],base['symbolic']['standard']['success'])
                accepted=txt['ci95'][1]<=.02 and d['symbolic']['standard']['nll']-base['symbolic']['standard']['nll']<=.02 and suc['ci95'][0]>=-.02
                atomic_json(b0.parent/'acceptance.json',dict(accepted=accepted,text=txt,standard_success=suc));a=b0 if accepted else a0
                cp=q.train('B1_alpha0','B1',a,first,0.,chosen['ncp']['lr'],common)
                if cp:controls['alpha0_Z']=str(cp);q.eval('B1_alpha0',cp,latent=2);q.audit('B1_alpha0',cp)
    atomic_json(q.root/'lineages.json',dict(lineages=lineages,controls=controls,foundations={f'{f}_s{s}':str(p) for (f,s),p in foundations.items()}))
    # Inference calibration uses development only, and every model's own frozen anchor.
    if first in lineages:
        anchor=lineages[first]['adapters'];s,_=load_eval(anchor,q.fast);ds=np.load(DATA/'dev_hard.npz');times=[]
        q.stage='development_control_reach';q.status();reach={}
        for length in (1,2,4) if not q.smoke else (1,):reach[str(length)]=symbolic_metrics(s,'dev',4 if q.smoke else 64,q.batch,('standard','hard'),8 if not q.smoke else 1,length,seed=50001)
        atomic_json(q.root/'control_reach.json',reach)
        if not q.smoke:
            from reachability import privileged_reach
            privileged_reach(s,q.root/'privileged_reach.json',n=16)
        window=max(reach,key=lambda key:reach[key]['hard']['success_rate']);window=int(window)
        for i in range(4):
            r=generate(s,ds['x'][i:i+1,:int(ds['prompt_len'][i])]);times.append(r['seconds'])
        del s;release();tau=float(np.median(times[1:]));budgets=[round(max(.25,tau*mult),3) for mult in (4,8,16)]
        if q.smoke:budgets=[30.]
        pilot_n=4 if q.smoke else 16;best_retry=None;best_grad=None
        for temp,top in ((.6,0),(.8,128),(1.,0)) if not q.smoke else ((.8,128),):
            params=dict(temperature=temp,top_k=top,candidate_limit=16,length=window,lr=.03,kl=.05)
            r=search_eval(anchor,q.root/f'dev_retry_t{temp:g}_k{top}','dev',['retry'],[budgets[1] if len(budgets)>1 else budgets[0]],pilot_n,params=params,fast=q.fast,seed=51001,deadline=q.deadline)
            rate=next(iter(r['arms'].values()))['success_rate'];cand=(rate or 0.,-temp,params)
            if best_retry is None or cand[:2]>best_retry[:2]:best_retry=cand
            release()
        for lr,kl in ((.01,0.),(.03,0.),(.1,0.),(.03,.05)) if not q.smoke else ((.03,0.),):
            params={**best_retry[2],'lr':lr,'kl':kl};method='gradcuit_kl' if kl else 'gradcuit'
            r=search_eval(anchor,q.root/f'dev_grad_lr{lr:g}_kl{kl:g}','dev',[method],[budgets[1] if len(budgets)>1 else budgets[0]],pilot_n,params=params,fast=q.fast,seed=51001,deadline=q.deadline)
            rate=next(iter(r['arms'].values()))['success_rate'];cand=(rate or 0.,-lr,params,method)
            if best_grad is None or cand[:2]>best_grad[:2]:best_grad=cand
            release()
        params={**best_retry[2],'lr':best_grad[2]['lr'],'kl':best_grad[2]['kl']};grad=best_grad[3]
        methods=['retry','latent','combined',grad,'ab_prior','random_chain','ab_hybrid','hybrid_random','hybrid_chain'] if not q.smoke else ['retry','gradcuit','ab_hybrid']
        dev=search_eval(anchor,q.root/'dev_search','dev',methods,budgets,4 if q.smoke else 32,params=params,fast=q.fast,seed=52001,deadline=q.deadline);release()
        middle=budgets[1] if len(budgets)>1 else budgets[0];ncp_dev=devs[('ncp',first)];competent=ncp_dev['symbolic']['standard']['success_rate']>=.8
        flat_params=params
        flat_path=foundations.get(('flat',first))
        if flat_path:
            best_flat=None
            for temp,top in ((.6,0),(.8,128),(1.,0)) if not q.smoke else ((.8,128),):
                fp={**params,'temperature':temp,'top_k':top}
                r=search_eval(flat_path,q.root/f'dev_flat_retry_t{temp:g}_k{top}','dev',['retry'],[middle],pilot_n,params=fp,fast=q.fast,seed=51001,deadline=q.deadline)
                rate=next(iter(r['arms'].values()))['success_rate'];cand=(rate or 0.,-temp,fp)
                if best_flat is None or cand[:2]>best_flat[:2]:best_flat=cand
                release()
            flat_params=best_flat[2]
        lock=dict(parameters=params,grad_method=grad,methods=methods,budgets_seconds=budgets,primary_budget_s=middle,selection_dataset='development only',competence_gate=competent,
            interpretation='primary repair benchmark' if competent else 'diagnostic search only: standard foundation competence below 80%',test_methods=['retry',grad,'ab_hybrid','hybrid_random','hybrid_chain'],
            uncertainty='paired problem intervals per trained seed; three separately trained seeds shown individually; incomplete runs cannot confirm repeatability',flat_retry_parameters=flat_params,
            sealed_hard_items=512 if competent else 64,scope='full search confirmation' if competent else 'bounded diagnosis of a weak foundation')
        atomic_json(q.root/'inference_lock.json',lock)
    else:lock=None;q.skip('search','No finite exploration lineage')
    # All choices are now frozen. No subsequent tuning is allowed.
    q.stage='sealed_evaluation';q.status()
    for (family,seed),path in foundations.items():q.eval(f'A_{family}_s{seed}',path,test=True)
    for seed,l in lineages.items():
        q.eval(f'B0_ncp_s{seed}',l['refit'],test=True);q.eval(f'B1_ncp_s{seed}',l['adapters'],test=True)
    for name,path in controls.items():q.eval(name,path,test=True)
    if lock:
        for seed,l in lineages.items():
            q.stage=f'sealed_search_s{seed}';q.status()
            if time.time()>=q.deadline:q.skip(q.stage,'Experiment deadline reached');continue
            search_eval(l['adapters'],q.root/f'test_search_s{seed}','test',lock['test_methods'],[lock['primary_budget_s']],4 if q.smoke else lock['sealed_hard_items'],params=lock['parameters'],fast=q.fast,seed=53001,deadline=q.deadline);release()
            flat=foundations.get(('flat',seed))
            if flat and time.time()<q.deadline:
                q.stage=f'sealed_transformer_retry_s{seed}';q.status();search_eval(flat,q.root/f'test_flat_retry_s{seed}','test',['retry'],[lock['primary_budget_s']],4 if q.smoke else lock['sealed_hard_items'],params=lock['flat_retry_parameters'],fast=q.fast,seed=53001,deadline=q.deadline);release()
        if not q.smoke and time.time()<q.deadline and first in lineages:
            q.stage='sealed_cost_curve';q.status();search_eval(lineages[first]['adapters'],q.root/'test_cost_curve','test',lock['methods'],lock['budgets_seconds'],128 if lock['competence_gate'] else 32,params=lock['parameters'],fast=q.fast,seed=54001,deadline=q.deadline);release()
            q.stage='sealed_standard_search';q.status();search_eval(lineages[first]['adapters'],q.root/'test_standard_search','test',lock['test_methods'],[lock['primary_budget_s']],256 if lock['competence_gate'] else 64,suite='standard',params=lock['parameters'],fast=q.fast,seed=55001,deadline=q.deadline);release()
            if 'alpha0_Z' in controls:search_eval(controls['alpha0_Z'],q.root/'test_alpha0_search','test',lock['test_methods'],[lock['primary_budget_s']],128,params=lock['parameters'],fast=q.fast,seed=56001,deadline=q.deadline);release()
    q.skip('larger-scale/distributed/BF16/context>384 training','Requires a separate experiment; not validated by this tiny run')
    q.skip('full V5/V6 causal bridge and retrained port-only ablations','Repository evidence reviewed; targeted 24-hour programme tests V7 matched foundations and staged controls, not the entire research programme')
    q.skip('matched token-control site, control-dimension sweep, binding-language tasks and fractional-feedback search','Not included in the initial 24-hour scope; two-hop composition is tested, but general variable binding is not')
    status='completed_with_limits' if q.skipped or any(r['train']['status']!='completed' for r in q.records) else 'completed'
    q.stage='reporting';q.status();from report import build_report;build_report(q.root,ROOT/'tests/pipeline/REPORT.md' if q.smoke else ROOT/'REPORT.md',final_status=status)
    q.status(status,completed_epoch=time.time());q.log(event=status,seconds=time.time()-q.start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();torch.set_num_threads(4);q=Queue(args.smoke)
    try:run(q)
    except Exception as exc:
        err=dict(error=repr(exc),traceback=traceback.format_exc(),stage=q.stage);atomic_json(q.root/'failure.json',err);q.status('failed',failure=err);q.log(event='failed',**err)
        try:
            from report import build_report
            build_report(q.root,ROOT/'tests/pipeline/REPORT.md' if q.smoke else ROOT/'REPORT.md')
        except Exception:pass
        raise

if __name__=='__main__':main()
