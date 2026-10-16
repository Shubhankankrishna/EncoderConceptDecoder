"""Independent stage trainer. Every completed or time-limited stage is recoverable."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import Foundation,Core,Adapters,System,count,active_count,freeze,chunk_info
from data import Mixer,DATA
from runtime import (DEVICE,ROOT,Log,atomic_json,seed_all,load_system,save_checkpoint,optimizer,lr_at,
                     nll,balanced_sums,ncp_weights,controls,candidate_backward,sync,implementation_hash)


def configure(stage,source=None,kind='gated',device=DEVICE):
    if stage=='A':system=System(Foundation()).to(device)
    else:
        system,_=load_system(source,device)
        freeze(system.foundation)
        if stage=='B0':system.core=Core().to(device);system.adapters=None
        elif stage=='B1':
            freeze(system.core);system.adapters=Adapters(kind).to(device)
        elif stage=='det_extra':
            system.adapters=None
            for p in system.core.parameters():p.requires_grad_(True)
        else:raise ValueError(stage)
    return system


def tensors_forward(system,stage,compiled=False):
    f=system.foundation
    if compiled and stage!='A':
        system.optimized_context=torch.compile(system.context)
        system.optimized_teacher_concept=torch.compile(f.concept)
    if stage=='A':
        def fn(x,docs):
            logits,c,chat,u,a=f(x,docs)
            return logits,c,chat
    else:
        def fn(h,c,cd,z,docs):
            chat,a,u=system.plans(c,cd,z)
            return f.decode(h,chat,a,docs),chat,u,a
    return torch.compile(fn) if compiled else fn


def train_loss(system,stage,batch,forward,delta=0,alpha=1):
    f=system.foundation;cm=ncp_weights(batch)
    if stage=='A':
        logits,c,chat=forward(batch['x'],batch['docs'])
        vq,q=f.vq(c,chunk_info(batch['docs'])[1])
        dist=torch.zeros((),device=batch['x'].device)
    else:
        with torch.no_grad():
            h,c,cd=getattr(system,'optimized_context',system.context)(batch['x'],batch['docs'])
            old=getattr(system,'optimized_teacher_concept',f.concept)(c,cd) if delta else None
            old_a=f.port(old) if delta else None
        logits,chat,u,a=forward(h,c,cd,None,batch['docs'])
        valid=chunk_info(batch['docs'])[1].float()
        dist=balanced_sums(((u-old).square().mean(-1)+(a-old_a).square().mean(-1))*valid,valid,batch['domain'])[0] if delta else torch.zeros((),device=c.device)
        vq=torch.zeros((),device=c.device);q=None
    tok=nll(logits,batch['x'],batch['m'])
    ntp,domains,_=balanced_sums(tok,batch['m'][:,:-1],batch['domain'])
    err=(chat[:,:-1]-c[:,1:].detach()).square().mean(-1)
    ncp=balanced_sums(err*cm,cm,batch['domain'])[0]
    loss=ntp+alpha*ncp+vq+delta*dist
    return loss,dict(text=domains[0],graph=domains[1],ncp=ncp,vq=vq,distillation=dist),c,q


def do_update(system,opt,stage,batch,forward,generator,step,total,beta=0.25,alpha=1,replay=False):
    opt.zero_grad(set_to_none=True)
    if stage=='B1':
        z1,mask,index,enabled,lengths=controls(batch,generator)
        from model import ball
        bank2=ball(torch.randn(*z1.shape,generator=generator)).to(z1.device)
        z2=torch.gather(bank2,1,index[:,:,None].expand_as(z1))*enabled[:,:,None]
        def adapter_forward(h,c,cd,z,docs):return forward(h,c,cd,z,docs)[0]
        value=candidate_backward(system,batch,[z1,z2],mask,beta,adapter_forward,replay)
        loss=torch.tensor(value,device=z1.device);logs=dict(selection=value)
        c=q=None
    else:
        frac=step/max(1,total);delta=1.0 if frac<0.1 else max(0.,1-(frac-0.1)/0.2)
        if stage!='B0':delta=0
        loss,logs,c,q=train_loss(system,stage,batch,forward,delta,alpha)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss: optimizer update prevented')
        loss.backward()
    trainable=[p for p in system.parameters() if p.requires_grad]
    gn=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite selection loss: optimizer update prevented')
    opt.step()
    if stage=='A':system.foundation.count_usage(q,chunk_info(batch['docs'])[1])
    return float(loss.detach()),float(gn),{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in logs.items()},c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A','B0','B1','det_extra'],required=True)
    ap.add_argument('--source');ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--seconds',type=float,default=20000)
    ap.add_argument('--kind',default='gated');ap.add_argument('--beta',type=float,default=.25)
    ap.add_argument('--alpha',type=float,default=1);ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--eager',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--stop-after',type=int,help='Predeclared prefix control; preserves the original LR/restart horizon')
    a=ap.parse_args();torch.set_num_threads(4);seed_all(a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);log=Log(out/'log.jsonl')
    config={**vars(a),'device':DEVICE,'precision':'float32','implementation_sha256':implementation_hash(),
            'data_manifest':json.loads((DATA/'manifest.json').read_text()),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    system=configure(a.stage,a.source,a.kind);mix=Mixer(10000+a.seed,batch=a.batch);g=torch.Generator().manual_seed(20000+a.seed)
    lr=.001 if a.stage=='B1' else .002
    opt=optimizer(system.parameters(),lr,wd=0 if a.stage=='B1' else .1)
    if a.stage=='A':
        # Avoid the duplicate Q group: build the non-Q groups first.
        opt=optimizer([p for n,p in system.foundation.named_parameters() if n!='Q'],lr)
        opt.add_param_group(dict(params=[system.foundation.Q],weight_decay=0.,lr_mult=10.))
    start_step=0
    if a.resume and (out/'checkpoint.pt').exists():
        restored,ck=load_system(out/'checkpoint.pt')
        if ck['config']['implementation_sha256']!=config['implementation_sha256']:raise ValueError('Resume source hash mismatch')
        for key in ('stage','source','seed','steps','kind','beta','alpha','batch','eager','stop_after','data_manifest'):
            if ck['config'].get(key)!=config.get(key):raise ValueError('Resume configuration mismatch: '+key)
        system.load_state_dict(restored.state_dict());opt.load_state_dict(ck['optimizer']);mix.restore(ck['mixer']);g.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng'])
        np.random.set_state(ck['numpy_rng']);import random;random.setstate(ck['python_rng']);start_step=ck['step'];config=ck['config']
    elif a.stage=='A':
        batch=mix.batch(DEVICE)
        with torch.no_grad():c=system.foundation.pool(system.foundation.encode(batch['x'],batch['docs']))
        system.foundation.restart(c,chunk_info(batch['docs'])[1],generator=g)
    elif a.stage=='B1':
        batch=mix.batch(DEVICE);calibration_state=g.get_state()
        with torch.no_grad():_,c,_=system.context(batch['x'],batch['docs']);e=system.core.in_proj(c)
        schedules=[controls(batch,g)[0] for _ in range(2)]
        config['adapter_initialization_factors']=system.adapters.calibrate(e,schedules)
        config['calibration_rng']=calibration_state.tolist()
    config['active_parameters']=active_count(system);config['resident_parameters']=count(system)
    config['trainable_parameters']=sum(p.numel() for p in system.parameters() if p.requires_grad)
    atomic_json(out/'config.json',config);print(json.dumps({'stage':a.stage,'params':config['active_parameters'],'trainable':config['trainable_parameters'],'steps':a.steps}),flush=True)
    forward=tensors_forward(system,a.stage,not a.eager);begin=time.monotonic();last_save=begin;tokens=0;target_totals=torch.zeros(2,device=DEVICE,dtype=torch.long);valid_tokens=torch.zeros((),device=DEVICE,dtype=torch.long);final_step=start_step;last=None
    save_checkpoint(out/'checkpoint.pt',system,opt,start_step,mix,g,config,dict(status='initialized'))
    try:
        for step in range(start_step,min(a.steps,a.stop_after or a.steps)):
            if step>start_step and time.monotonic()-begin>=a.seconds:break
            for group in opt.param_groups:group['lr']=lr_at(step,a.steps,lr,300 if a.stage=='A' else 200)*group.get('lr_mult',1)
            batch=mix.batch(DEVICE)
            loss,gn,logs,c=do_update(system,opt,a.stage,batch,forward,g,step,a.steps,a.beta,a.alpha)
            final_step=step+1;tokens+=batch['x'].numel()
            target_totals+=torch.stack([batch['m'][batch['domain']==i,:-1].sum() for i in (0,1)]).long()
            valid_tokens+=(batch['docs']>=0).sum()
            if a.stage=='A' and final_step%250==0 and final_step<.8*a.steps:
                restarted=system.foundation.restart(c,chunk_info(batch['docs'])[1],opt,g);log(step=final_step,restarted_codes=restarted)
            if step%50==0 or final_step==a.steps:
                sync();dt=time.monotonic()-begin
                target_counts=[float(batch['m'][batch['domain']==i,:-1].sum()) for i in (0,1)]
                memory=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory()) if DEVICE=='mps' else {}
                last=dict(step=final_step,loss=loss,gradient_norm=gn,lr=opt.param_groups[0]['lr'],elapsed_s=dt,input_slots=tokens,
                          slots_per_s=tokens/max(dt,.001),targets_this_batch=target_counts,unique_text_documents=len(mix.seen_docs),symbolic_instances=mix.graphs,memory_bytes=memory,**logs)
                log(**last);print(json.dumps(last),flush=True)
                atomic_json(out/'live.json',{**last,'status':'training','planned_steps':a.steps})
            if a.stage=='A' and final_step==9000:
                save_checkpoint(out/'checkpoint_step_9000.pt',system,opt,final_step,mix,g,config,dict(status='matched_prefix_control'))
            if final_step%1000==0 or time.monotonic()-last_save>300:
                save_checkpoint(out/'checkpoint.pt',system,opt,final_step,mix,g,config,dict(status='training'));last_save=time.monotonic()
        status='completed' if final_step==a.steps else ('planned_prefix_completed' if a.stop_after and final_step==a.stop_after else 'time_budget_reached')
        save_checkpoint(out/'checkpoint.pt',system,opt,final_step,mix,g,config,dict(status=status))
        summary=dict(status=status,planned_steps=a.steps,completed_steps=final_step,wall_s=time.monotonic()-begin,active_parameters=active_count(system),
                     input_slots=tokens,nonpadding_input_tokens=int(valid_tokens),loss_bearing_targets_by_domain=target_totals.cpu().tolist(),
                     unique_text_documents=len(mix.seen_docs),symbolic_instances=mix.graphs,last=last)
        atomic_json(out/'train_summary.json',summary);atomic_json(out/'live.json',summary);print(json.dumps(summary),flush=True)
    except Exception as exc:
        atomic_json(out/'failure.json',dict(status='failed',step=final_step,error=repr(exc),last_finite_checkpoint=str(out/'checkpoint.pt')))
        raise


if __name__=='__main__':main()
