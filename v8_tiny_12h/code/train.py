"""Update/exposure-budget training. Dependent stages require explicit accepted anchors."""
import argparse,copy,json,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from model import *
from data import Mixer,DATA,digest,NODE0,COL0,N_COL
from runtime import *

def configure(stage,source=None,flat=False,core='dense',kind='gated',device=DEVICE,rho=1.):
    if source:s,_=load_system(source,device)
    else:s=System(Flat() if flat else Foundation()).to(device)
    if stage=='REFIT':
        freeze(s);s.adapters=None;s.core=(DenseCore(s.foundation) if core=='dense' else Core()).to(device)
        for p in s.core.parameters():p.requires_grad_(True)
    elif stage=='EXTRA':
        freeze(s);s.adapters=None
        if s.is_flat:
            for p in s.parameters():p.requires_grad_(True)
        else:
            if s.core is None:s.core=DenseCore(s.foundation).to(device)
            for p in s.core.parameters():p.requires_grad_(True)
    elif stage=='Z':
        freeze(s);s.adapters=Adapters(kind).to(device)
    elif stage=='CL':
        freeze(s);s.add_phase(rho,True)
        for block in s.active_blocks():
            for p in block.banks[-1].parameters():p.requires_grad_(True)
    elif stage=='HARD':
        for p in s.parameters():p.requires_grad_(True)
    return s

def forward_function(s,stage,compiled=False):
    if stage in ('REFIT','EXTRA','Z') and not s.is_flat:
        def fn(h,c,cd,z,docs):
            q,a,u=s.plans(c,cd,z);return s.foundation.decode(h,q,a,docs),c,q,u,a
    else:
        def fn(x,docs):return s(x,docs,details=not s.is_flat)
    return torch.compile(fn) if compiled else fn

def objective(s,stage,b,fn,alpha,delta,diagnostics=False):
    zero=b['x'].new_zeros((),dtype=torch.float32);old=None
    if stage in ('REFIT','EXTRA') and not s.is_flat:
        with torch.no_grad():
            h,c,cd=s.context(b['x'],b['docs'])
            if delta:
                u0=s.foundation.concept(c,cd);old=(u0,s.foundation.port(u0),s.foundation.pq(u0))
        result=fn(h,c,cd,None,b['docs'])
    else:result=fn(b['x'],b['docs'])
    if s.is_flat:logits=result;c=q=valid=None;vq=ncp=dist=zero;code=None
    else:
        logits,c,q,u,a=result;valid=chunk_info(b['docs'])[1]
        vq,code=s.foundation.vq(c,valid) if stage in ('A','HARD','SIMPLE','MEM') else (zero,None)
        cm=ncp_weights(b);ncp=balanced((q[:,:-1]-c[:,1:].detach()).square().mean(-1)*cm,cm,b['domain'])[0]
        dist=balanced(((u-old[0]).square().mean(-1)+(a-old[1]).square().mean(-1)+(q-old[2]).square().mean(-1))/3*valid,valid,b['domain'])[0] if old else zero
    tok=nll(logits,b['x'],b['m']);ntp,dom,_=balanced(tok,b['m'][:,:-1],b['domain'])
    if stage=='CL':ncp=vq=dist=zero
    logs=dict(text=dom[0],symbolic=dom[1],ncp=ncp,vq=vq,distillation=dist,alpha=alpha)
    if diagnostics:
        with torch.no_grad():
            target=b['x'][:,1:];correct=logits[:,:-1].argmax(-1)==target
            for task in sorted({v['task'] for v in b['metas']} - {'text'}):
                rows=torch.tensor([v['task']==task for v in b['metas']],device=target.device)[:,None]
                scored=b['m'][:,:-1]*rows;node=scored*((target>=NODE0)&(target<COL0));formatting=scored*((target<NODE0)|(target>=COL0+N_COL))
                for kind,mask in [('node',node),('format',formatting)]:
                    logs[task+'_'+kind+'_ce']=(tok*mask).sum()/mask.sum().clamp_min(1)
                    logs[task+'_'+kind+'_accuracy']=(correct*mask).sum()/mask.sum().clamp_min(1)
                    logs[task+'_'+kind+'_targets']=mask.sum()
    return ntp+alpha*ncp+vq+delta*dist,logs,c,code,valid

def candidate_backward(s,b,fn,g,beta):
    z,mask,index,enabled,lengths=controls(b,g);bank=ball(torch.randn(*z.shape,generator=g)).to(z.device);other=torch.gather(bank,1,index[...,None].expand_as(z))*enabled[...,None];schedules=[z,other]
    with torch.no_grad():h,c,cd=s.context(b['x'],b['docs'])
    def rows(v):return nll(fn(h,c,cd,v,b['docs'])[0],b['x'],mask).sum(1)/mask[:,:-1].sum(1).clamp_min(1)
    eligible=mask[:,:-1].sum(1)>0
    with torch.no_grad():scores=torch.stack([rows(v) for v in schedules]);winner=scores.argmin(0);value=balanced(((1-beta)*scores.min(0).values+beta*scores.mean(0))*eligible,eligible,b['domain'])[0]
    dens=torch.stack([((b['domain']==i)&eligible).sum() for i in (0,1)]);present=(dens>0).sum().clamp_min(1);w=eligible/dens[b['domain']].clamp_min(1)/present
    for i,v in enumerate(schedules):(rows(v)*((1-beta)*(winner==i)+beta/2)*w).sum().backward()
    return value

def frozen_hash(s):return tensor_hash_tensors({n:p for n,p in s.named_parameters() if not p.requires_grad})
def tensor_hash_tensors(values):
    h=hashlib.sha256()
    for n,p in sorted(values.items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--stage',default='A',choices=['A','SIMPLE','MEM','HARD','REFIT','EXTRA','Z','CL']);ap.add_argument('--source');ap.add_argument('--flat',action='store_true');ap.add_argument('--core',default='dense');ap.add_argument('--kind',default='gated');ap.add_argument('--mode',default='foundation');ap.add_argument('--steps',type=int,default=24000);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--context',type=int,default=192);ap.add_argument('--seed',type=int,default=811);ap.add_argument('--seconds',type=float,default=43200);ap.add_argument('--alpha',type=float,default=.1);ap.add_argument('--lr',type=float);ap.add_argument('--beta',type=float,default=.25);ap.add_argument('--constant-alpha',action='store_true');ap.add_argument('--compiled',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--no-distill',action='store_true');ap.add_argument('--always-open',action='store_true');ap.add_argument('--fixed',action='store_true');ap.add_argument('--learning-check',action='store_true');ap.add_argument('--development-curves',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);seed_all(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);s=configure(args.stage,args.source,args.flat,args.core,args.kind);mix=Mixer(81000+args.seed,args.batch,args.mode,args.context);g=torch.Generator().manual_seed(82000+args.seed)
    if args.always_open:
        for block in s.active_blocks():
            for bank in block.banks:bank.force=True
    lr=args.lr or (.002 if args.stage in ('A','SIMPLE','MEM') else .0005 if args.stage=='HARD' else .0003 if args.stage=='CL' else .001)
    opt=optimizer(s,lr,args.stage);start_step=0
    config={**vars(args),'lr':lr,'device':DEVICE,'precision':'float32','implementation_sha256':implementation_hash(),'data_sha256':digest(DATA/'manifest.json'),'active_parameters':active_count(s),'resident_parameters':count(s),'buffer_scalars':sum(v.numel() for v in s.buffers()),'trainable_parameters':sum(p.numel() for p in s.parameters() if p.requires_grad),'metadata':s.metadata(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if args.resume:
        restored,ck=load_system(out/'checkpoint.pt');s.load_state_dict(restored.state_dict());opt.load_state_dict(ck['optimizer']);mix.restore(ck['mixer']);g.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);start_step=ck['step']
        if ck['config']['implementation_sha256']!=config['implementation_sha256']:raise RuntimeError('Resume source mismatch')
        for k in vars(args):
            if k!='resume' and ck['config'].get(k)!=config.get(k):raise RuntimeError('Resume setting mismatch '+k)
    elif args.stage=='HARD':
        _,ck=load_system(args.source)
        if ck.get('optimizer'):opt.load_state_dict(ck['optimizer'])
        if ck.get('mixer'):mix.restore(ck['mixer'])
    elif args.stage in ('A','SIMPLE','MEM') and not s.is_flat:
        # Initialization probe does not consume/charge a training batch.
        probe=Mixer(89999,args.batch,args.mode,args.context).batch(DEVICE)
        with torch.no_grad():h,c,cd=s.context(probe['x'],probe['docs']);s.foundation.restart(c,chunk_info(probe['docs'])[1],generator=g)
    initial_tasks=dict(mix.seen_tasks);initial_targets=dict(mix.targets);initial_slots=mix.slots;initial_nonpad=mix.nonpad
    fn=forward_function(s,args.stage,args.compiled);atomic_json(out/'config.json',config);log=Log(out/'log.jsonl');begin=time.monotonic();last_save=begin;last=None;step=start_step;learning_pass=False;protected=frozen_hash(s);fixed=mix.batch(DEVICE) if args.fixed else None
    save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,dict(status='initialized'))
    interrupted=[False]
    def request_stop(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        for index in range(start_step,args.steps):
            if interrupted[0] or (index>start_step and time.monotonic()-begin>=args.seconds):break
            try:check_deadline(20)
            except TimeoutError:break
            b=fixed if fixed is not None else mix.batch(DEVICE,single_text=args.stage=='Z')
            if fixed is not None and index>start_step:
                mix.slots+=b['x'].numel();mix.nonpad+=int((b['docs']>=0).sum())
                for row,meta in enumerate(b['metas']):
                    task=meta['task'];mix.seen_tasks[task]=mix.seen_tasks.get(task,0)+1;mix.targets[task]=mix.targets.get(task,0)+int(b['m'][row].sum())
            for group in opt.param_groups:group['lr']=lr_at(index,args.steps,lr,300 if args.stage in ('A','SIMPLE','MEM') else 100 if args.stage=='CL' else 200)*group.get('lr_mult',1)
            alpha=args.alpha if args.constant_alpha or args.stage not in ('A',) else args.alpha*min(1.,max(0.,(index+1-6000)/2000))
            if args.stage in ('SIMPLE','MEM','CL'):alpha=0.
            frac=index/args.steps;delta=(1. if frac<.1 else max(0.,1-(frac-.1)/.2)) if args.stage=='REFIT' and not args.no_distill else 0.
            opt.zero_grad(set_to_none=True)
            if args.stage=='Z':loss=candidate_backward(s,b,fn,g,args.beta);logs=dict(selection=float(loss));c=code=valid=None
            else:
                loss,logs,c,code,valid=objective(s,args.stage,b,fn,alpha,delta,diagnostics=index==start_step or (index+1)%100==0)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective; update prevented')
                loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in s.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();step=index+1
            if code is not None:
                s.foundation.count_usage(code,valid)
                if args.stage in ('A','SIMPLE','MEM') and step%250==0 and step<.8*args.steps:s.foundation.restart(c,valid,opt,g)
            if step==1 or step%100==0 or step==args.steps:
                sync();dt=time.monotonic()-begin;logs={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in logs.items()}
                new_targets=sum(mix.targets.values())-sum(initial_targets.values());exposure={k:v-initial_tasks.get(k,0) for k,v in mix.seen_tasks.items()}
                last=dict(step=step,planned_steps=args.steps,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=dt,slots_per_s=(mix.slots-initial_slots)/dt,scored_targets_per_s=new_targets/dt,memory_bytes=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory()) if DEVICE=='mps' else {},scored_targets=new_targets,exposure=exposure,lifetime_exposure=mix.seen_tasks,unique_symbols=len(mix.seen_symbols),unique_documents=len(mix.seen_docs),source_targets=mix.source_targets,lr=opt.param_groups[0]['lr'],**logs);log(**last);atomic_json(out/'live.json',last);print(json.dumps(last),flush=True)
            if args.learning_check and step%500==0:
                from generation import generate
                from evaluate import symbolic_eval
                if fixed is not None:
                    p=int(fixed['metas'][0]['prompt_len']);result=generate(s,fixed['x'][:,:p].cpu().numpy(),max_new=4);rate=sum(v['success'] for v in result['verification'])/len(result['verification']);learning_pass=rate>=.99;screen=dict(memorization_rate=rate)
                else:
                    s.fast_generation=True;lookup=symbolic_eval(s,'lookup',n=512);compose=symbolic_eval(s,'compose',n=512);learning_pass=lookup['rate']>=.95 and compose['rate']>=.90;screen=dict(lookup=lookup['rate'],compose=compose['rate'])
                atomic_json(out/'learning_gate.json',dict(step=step,pass_gate=bool(learning_pass),**screen))
                if learning_pass:break
            if args.development_curves and step in (2000,4000,6000,8000,12000,18000,24000):
                from evaluate import symbolic_eval
                s.fast_generation=not isinstance(s.core,Core)
                try:
                    screen={name:symbolic_eval(s,name,n=128) for name in ('standard','lookup','compose','hard10')}
                    atomic_json(out/f'development_step_{step}.json',dict(step=step,diagnostic_only=True,never_stops_training=True,symbolic=screen))
                except TimeoutError:break
            if step%2000==0 or time.monotonic()-last_save>=600:save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,dict(status='training',last=last));last_save=time.monotonic()
        if frozen_hash(s)!=protected:raise RuntimeError('Frozen weights changed')
        status='interrupted_by_signal' if interrupted[0] else 'learning_gate_passed' if learning_pass else 'completed' if step==args.steps else 'partial_time_budget'
        save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,dict(status=status,last=last));summary=dict(**config,status=status,completed_steps=step,wall_s=time.monotonic()-begin,last=last,input_slots=mix.slots-initial_slots,nonpadding_tokens=mix.nonpad-initial_nonpad,scored_targets_by_task={k:v-initial_targets.get(k,0) for k,v in mix.targets.items()},symbolic_exposure={k:v-initial_tasks.get(k,0) for k,v in mix.seen_tasks.items()},lifetime_symbolic_exposure=mix.seen_tasks,unique_symbols=len(mix.seen_symbols),unique_documents=len(mix.seen_docs),source_targets=mix.source_targets,frozen_weights_preserved=True);atomic_json(out/'train_summary.json',summary)
    except Exception as exc:atomic_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),step=step,last_finite_checkpoint=str(out/'checkpoint.pt')));raise
if __name__=='__main__':main()
