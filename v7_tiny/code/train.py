"""Target-token-budget trainer; all stages keep finite atomic recovery checkpoints."""
import argparse,copy,json,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from model import Config,System,freeze,count,scale
from data import Mixer,DATA,ROOT
from runtime import *

def configure(stage,source=None,cfg=None,device=DEVICE):
    if source:s,ck=load_system(source,device)
    else:s=System(cfg).to(device)
    if stage=='B0':s.make_refit()
    elif stage=='B1':s.make_adapters()
    elif stage=='det_extra':
        freeze(s);s.adapters=None
        for p in (s.core or s.foundation.con).parameters():p.requires_grad_(True)
    elif stage=='full_extra':
        s.adapters=None
        for p in s.parameters():p.requires_grad_(True)
    return s

def forward_function(s,stage,compiled=False):
    if stage=='B1':fn=lambda ctx,docs,z:s.from_context(ctx,docs,z)
    elif stage in ('B0','det_extra'):fn=lambda ctx,docs:s.from_context(ctx,docs)
    else:fn=lambda x,docs:s(x,docs,details=s.cfg.kind!='flat')
    return torch.compile(fn) if compiled else fn

def train_loss(s,stage,batch,fn,alpha=.1,delta=0.):
    if s.cfg.kind=='flat':logits=fn(batch['x'],batch['docs']);c=q=valid=None;ntp,dom,_=balanced(nll(logits,batch['x'],batch['m']),batch['m'][:,:-1],batch['domain']);return ntp,dict(text=dom[0],symbolic=dom[1],ncp=ntp*0,vq=ntp*0,distillation=ntp*0),c,q,valid
    if stage in ('B0','det_extra'):
        with torch.no_grad():ctx=s.context(batch['x'],batch['docs']);old=s.foundation.plans(ctx[1],ctx[2],ctx[4]) if delta else None
        logits,c,forecast,u,ports,valid=fn(ctx,batch['docs'])
    else:logits,c,forecast,u,ports,valid=fn(batch['x'],batch['docs']);old=None
    ntp,dom,_=balanced(nll(logits,batch['x'],batch['m']),batch['m'][:,:-1],batch['domain'])
    cm=ncp_weights(batch);errors=(forecast[:,:-1]-c[:,1:].detach()).square().mean(-1)
    ncp=balanced(errors*cm,cm,batch['domain'])[0]
    if stage in ('A','full_extra'):vq,q,_=s.foundation.vq(c,valid,batch['domain'])
    else:vq=ntp*0;q=None
    dist=ntp*0
    if old is not None:
        per=((forecast-old[0]).square().mean(-1)+sum((a-b).square().mean(-1) for a,b in zip(ports,old[1])))/3
        dist=balanced(per*valid,valid,batch['domain'])[0]
    return ntp+alpha*ncp+vq+delta*dist,dict(text=dom[0],symbolic=dom[1],ncp=ncp,vq=vq,distillation=dist),c,q,valid

def update(s,opt,stage,batch,fn,g,progress,alpha=.1,beta=.25,replay=False,ramp=True,diagnostic=False):
    opt.zero_grad(set_to_none=True);grad_diag={}
    if stage=='B1':
        batch['dz']=s.cfg.dz;z,mask,index,enabled,lengths=controls(batch,g)
        bank=ball(torch.randn(*z.shape,generator=g)).to(z.device);z2=torch.gather(bank,1,index[:,:,None].expand_as(z))*enabled[:,:,None]
        value=candidate_backward(s,batch,[z,z2],mask,beta,fn,replay);loss=torch.tensor(value,device=z.device);logs={'selection':value};c=q=valid=None
    else:
        delta=(1. if progress<.1 else max(0.,1-(progress-.1)/.2)) if stage=='B0' else 0.
        coefficient=alpha_at(progress,alpha,ramp) if stage in ('A','full_extra') else alpha
        loss,logs,c,q,valid=train_loss(s,stage,batch,fn,coefficient,delta)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective prevented update')
        if diagnostic and s.cfg.kind!='flat' and stage=='A':
            params=[s.foundation.enc.blocks[0].qkv.weight,s.foundation.con.blocks[0].qkv.weight,s.foundation.head.weight]
            tok=.5*(logs['text']+logs['symbolic']);g1=torch.autograd.grad(tok,params,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(logs['ncp'],params,retain_graph=True,allow_unused=True)
            for name,a,b in zip(('encoder','concept','head'),g1,g2):
                if a is not None and b is not None:grad_diag[name]=dict(token_norm=float(a.norm()),ncp_norm=float(b.norm()),cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12)))
        loss.backward();logs['alpha']=coefficient
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite selection objective prevented update')
    gn=torch.nn.utils.clip_grad_norm_([p for p in s.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
    if q is not None:s.foundation.record_usage(q,valid)
    return float(loss.detach()),float(gn),{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in logs.items()},c,valid,grad_diag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A','B0','B1','det_extra','full_extra'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--source');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--tokens',type=int,required=True)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--seconds',type=float,default=86400);ap.add_argument('--alpha',type=float,default=.1)
    ap.add_argument('--beta',type=float,default=.25);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--flat',action='store_true');ap.add_argument('--plain',action='store_true')
    ap.add_argument('--constant-alpha',action='store_true');ap.add_argument('--eager',action='store_true');ap.add_argument('--replay',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--max-steps',type=int);ap.add_argument('--mini',action='store_true')
    args=ap.parse_args();torch.set_num_threads(4);seed_all(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=Config(kind='flat' if args.flat else 'hierarchy',routing=not args.plain)
    if args.mini:cfg=Config(d=32,heads=4,ff=96,dz=8,rank=8,segments=4,codes=8,flat_ff=64,kind=cfg.kind,routing=cfg.routing)
    s=configure(args.stage,args.source,cfg)
    config={**vars(args),'cfg':asdict(s.cfg),'device':DEVICE,'precision':'float32','implementation_sha256':implementation_hash(),'data_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        'active_parameters':s.active_count(),'resident_parameters':count(s),'trainable_parameters':sum(p.numel() for p in s.parameters() if p.requires_grad),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    mix=Mixer(70000+args.seed,batch=args.batch);g=torch.Generator().manual_seed(80000+args.seed);opt=optimizer(s,args.lr);step=0;targets=0
    if args.resume:
        restored,ck=load_system(out/'checkpoint.pt');s.load_state_dict(restored.state_dict());opt.load_state_dict(ck['optimizer']);mix.restore(ck['mixer']);g.set_state(ck['generator'])
        if ck['config']['implementation_sha256']!=config['implementation_sha256']:raise RuntimeError('Resume source differs')
        for k in ('stage','tokens','batch','alpha','beta','lr','flat','plain','constant_alpha','eager','replay','source','seed'):
            if ck['config'].get(k)!=config.get(k):raise RuntimeError('Resume mismatch: '+k)
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);__import__('random').setstate(ck['python_rng']);step=ck['step'];targets=ck['extra']['targets'];config=ck['config']
    elif args.stage=='A' and not args.flat:
        probe=Mixer(79999,batch=args.batch).batch(DEVICE)
        with torch.no_grad():ctx=s.context(probe['x'],probe['docs']);s.foundation.restart(ctx[1],ctx[-1],generator=g)
    elif args.stage=='B1':
        probe=Mixer(79998,batch=args.batch).batch(DEVICE,single_text=True);probe['dz']=s.cfg.dz
        with torch.no_grad():ctx=s.context(probe['x'],probe['docs']);z,*_=controls(probe,g)
        # Calibrate both independent sites using actual site inputs, not arbitrary hidden features.
        captured=[];hooks=[b.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach())) for b in (s.core or s.foundation.con).blocks if b in ((s.core or s.foundation.con).blocks[0],(s.core or s.foundation.con).blocks[s.cfg.lc//2])]
        with torch.no_grad():s.from_context(ctx,probe['docs'],torch.zeros_like(z))
        for h in hooks:h.remove()
        factors=[]
        with torch.no_grad():
            for site,v in enumerate(captured):
                w=s.adapters.write(v,z,site);ratio=float((w.float().square().mean(-1).sqrt()/scale(v).squeeze(-1)).mean());factor=.01/max(ratio,1e-8);s.adapters.U[site].weight.mul_(min(factor,100));factors.append(factor)
        config['adapter_calibration_factors']=factors
    atomic_json(out/'config.json',config);log=Log(out/'log.jsonl');fn=forward_function(s,args.stage,not args.eager)
    started=time.monotonic();last_save=started;last=None;restart_interval=max(1,args.tokens//100);next_restart=((targets//restart_interval)+1)*restart_interval
    frozen_before=tensor_hash(s.foundation) if args.stage in ('B0','B1','det_extra') else None
    core_before=tensor_hash(s.core) if args.stage=='B1' and s.core is not None else None
    save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,dict(targets=targets,status='initialized'))
    try:
        while targets<args.tokens and (args.max_steps is None or step<args.max_steps):
            if step and time.monotonic()-started>=args.seconds:break
            progress=targets/args.tokens;b=mix.batch(DEVICE,progress,single_text=args.stage=='B1')
            for group in opt.param_groups:group['lr']=lr_at(max(progress,1/max(args.tokens,1)),args.lr)
            value,gn,logs,c,valid,gd=update(s,opt,args.stage,b,fn,g,progress,args.alpha,args.beta,args.replay,not args.constant_alpha,step%1000==0)
            targets+=int(b['m'][:,:-1].sum());step+=1
            if c is not None and args.stage=='A' and not args.flat and targets>=next_restart and progress<.2:
                restarted=s.foundation.restart(c,valid,opt,g);log(step=step,restarted_codes=restarted);next_restart+=restart_interval
            if step==1 or step%100==0 or targets>=args.tokens:
                sync();elapsed=time.monotonic()-started;memory=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory()) if DEVICE=='mps' else {}
                last=dict(step=step,targets=targets,progress=min(1.,targets/args.tokens),loss=value,gradient_norm=gn,lr=opt.param_groups[0]['lr'],elapsed_s=elapsed,
                    scored_targets_per_s=(targets-(ck['extra']['targets'] if args.resume else 0))/elapsed,input_slots_per_s=step*args.batch*T/elapsed,memory=memory,gradient_alignment=gd,**logs)
                log(**last);atomic_json(out/'live.json',last);print(json.dumps(last),flush=True)
            if time.monotonic()-last_save>600:
                save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,dict(targets=targets,last=last));last_save=time.monotonic()
        status='completed' if targets>=args.tokens else 'partial_token_budget'
        if args.max_steps is not None and step>=args.max_steps and targets<args.tokens:status='pilot_prefix_completed'
        if frozen_before is not None and tensor_hash(s.foundation)!=frozen_before:raise RuntimeError('Frozen surrounding model mutated')
        if core_before is not None and tensor_hash(s.core)!=core_before:raise RuntimeError('Frozen refit core mutated')
        extra=dict(targets=targets,status=status,last=last)
        save_checkpoint(out/'checkpoint.pt',s,opt,step,mix,g,config,extra)
        summary=dict(**config,status=status,completed_steps=step,completed_targets=targets,target_budget=args.tokens,wall_s=time.monotonic()-started,
            last=last,frozen_hash_preserved=frozen_before is not None,unique_text_documents=len(mix.seen_docs),unique_symbolic_instances=len(mix.seen_symbols),symbolic_exposure=mix.seen_tasks)
        atomic_json(out/'train_summary.json',summary);print(json.dumps(summary),flush=True)
    except Exception as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),step=step,targets=targets,last_finite_checkpoint=str(out/'checkpoint.pt')));raise

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        if '--out' in sys.argv:
            failure_out=Path(sys.argv[sys.argv.index('--out')+1]);failure_out.mkdir(parents=True,exist_ok=True)
            if not (failure_out/'failure.json').exists():atomic_json(failure_out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),last_finite_checkpoint=str(failure_out/'checkpoint.pt') if (failure_out/'checkpoint.pt').exists() else None))
        raise
