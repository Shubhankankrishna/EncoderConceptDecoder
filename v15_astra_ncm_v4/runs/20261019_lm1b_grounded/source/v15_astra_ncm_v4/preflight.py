"""Meaningful numerical, learning, and optimizer/RNG resume checks before launch."""
import copy
from dataclasses import asdict
import json
import time

import torch

from .common import ROOT,atomic_json,heavy_job,restore_rng,rng_state,save_torch,seed_all,source_hashes
from .evaluate import evaluate
from .graphs import batch_states,load_dataset
from .model import Agent,AgentConfig,objective
from .train import sample_batch


def update(model,opt,batch):
    opt.zero_grad(set_to_none=True)
    loss,_=objective(model(batch),batch)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()),float(grad)


def main():
    torch.set_num_threads(2);seed_all(412)
    started=time.time()
    graphs=[g for g in load_dataset()['graphs'] if g['split']=='train']
    report={'status':'running','started':started,'variants':{}}
    atomic_json(ROOT/'reports/preflight.json',report)
    for name,cfg in [('r0',AgentConfig(steps=0)),('r1',AgentConfig(steps=1)),('r2',AgentConfig()),
                     ('untied2',AgentConfig(mode='untied')),('untied_compute2',AgentConfig(mode='untied_compute')),
                     ('strategy2',AgentConfig(strategies=True))]:
        seed_all(412)
        cpu=Agent(cfg).eval()
        batch=batch_states([(graphs[i],graphs[i]['start'],i%2 if cfg.strategies else 0) for i in range(8)])
        with torch.no_grad():original=cpu(batch)
        model=copy.deepcopy(cpu).to('mps')
        gpu_batch={k:v.to('mps') for k,v in batch.items()}
        with torch.no_grad():gpu=model(gpu_batch)
        delta=max(float((gpu[k].cpu()-original[k]).abs().max()) for k in ('logits','value','q'))
        assert delta<2e-4,(name,delta)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        losses=[];timings=[]
        for _ in range(12):
            tick=time.perf_counter()
            loss,grad=update(model,opt,gpu_batch);torch.mps.synchronize()
            losses.append(loss);timings.append(time.perf_counter()-tick)
        saved={'model':copy.deepcopy(model.state_dict()),'opt':copy.deepcopy(opt.state_dict()),'rng':rng_state()}
        save_torch(ROOT/'reports/resume_probe.pt',saved)
        next_batch=sample_batch(graphs,8,cfg.strategies,'mps')
        loss1,_=update(model,opt,next_batch)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        saved=torch.load(ROOT/'reports/resume_probe.pt',map_location='cpu',weights_only=True)
        resumed=Agent(cfg).to('mps');resumed.load_state_dict(saved['model'])
        opt2=torch.optim.AdamW(resumed.parameters(),lr=3e-4,weight_decay=.01);opt2.load_state_dict(saved['opt'])
        restore_rng(saved['rng'])
        replay=sample_batch(graphs,8,cfg.strategies,'mps')
        for key in replay:assert torch.equal(replay[key],next_batch[key]),key
        loss2,_=update(resumed,opt2,replay)
        resume_delta=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in resumed.state_dict().items())
        assert resume_delta==0,(name,resume_delta)
        assert losses[-1]<losses[0],(name,losses)
        report['variants'][name]={'parameters':sum(p.numel() for p in model.parameters()),
            'recurrent_parameters':sum(p.numel() for p in model.cells.parameters()),'cpu_mps_max_abs':delta,
            'first_loss':losses[0],'last_loss':losses[-1],'resume_weight_max_abs':resume_delta,
            'resume_loss_diff':abs(loss1-loss2),'median_step_seconds':sorted(timings[2:])[len(timings[2:])//2]}
        atomic_json(ROOT/'reports/preflight.json',report)
    # Memorization is a debugging control, explicitly not held-out generalization.
    seed_all(43)
    model=Agent().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tiny=graphs[:8]
    rows=[(g,c,0) for g in tiny for c in range(g['n'])]
    batch=batch_states(rows,'mps')
    initial=float(objective(model(batch),batch)[0].detach())
    for _ in range(250):update(model,opt,batch)
    final=float(objective(model(batch),batch)[0].detach())
    result=evaluate(model,tiny,'mps')
    assert final<initial*.5,(initial,final)
    assert result['rollouts']['greedy']['success_rate']>=.875,result['rollouts']['greedy']['success_rate']
    report['memorization']={'graphs':8,'steps':250,'initial_loss':initial,'final_loss':final,
        'optimal_top1':result['optimal_top1'],'training_graph_success':result['rollouts']['greedy']['success_rate'],
        'meaning':'debugging only; not generalization evidence'}
    report['passed']=True;report['status']='complete';report['seconds']=time.time()-started
    report['sources']=source_hashes()
    atomic_json(ROOT/'reports/preflight.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with heavy_job():main()
