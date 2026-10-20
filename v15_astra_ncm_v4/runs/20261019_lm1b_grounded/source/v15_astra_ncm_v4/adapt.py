"""Bounded NCM rank4 LoRA/FiLM training on verified plans with frozen-base retention."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from .cache import load_cache,WORDS
from .common import (ROOT,atomic_json,append_json,heavy_job,resource_issue,restore_rng,
                     rng_state,save_torch,seed_all,sha,source_hashes)
from .model import PlanConditioner
from .ncm import load_ncm,decoded_text,GENERATOR_SHA

STOP=False


def stop(*_):
    global STOP
    STOP=True


def tensor(array,device='cpu'):
    return torch.from_numpy(np.array(array,copy=True)).to(device)


def condition_for(encoder,records,indices,device):
    return encoder([records[i]['path'] for i in indices],[records[i]['strategy'] for i in indices],device)


def parse_path(text):
    return [WORDS.index(x) if x in WORDS else -1 for x in re.findall(r'node\s+([a-z]+|\d+)',text.lower())]


@torch.no_grad()
def evaluate_renderer(net,encoder,cache,inference,step,split='valid',count=24):
    net.eval();encoder.eval()
    records=cache['records'];device=net.device
    indices=[i for i,r in enumerate(records) if r['kind']=='plan' and r['split']==split][:count]
    retention=[i for i,r in enumerate(records) if r['kind']=='retention' and r['split']=='valid'][:8]
    if not indices:raise ValueError('No renderer evaluation examples')
    level_sums=[0.]*12;baseline=shuffled=0.
    paired=[]
    for j,i in enumerate(indices):
        condition=condition_for(encoder,records,[i],device)
        wrong=condition_for(encoder,records,[indices[(j+1)%len(indices)]],device)
        row_loss=row_shuffle=row_base=0.
        for level in range(12):
            lo,hi=net.offsets[level:level+2]
            raw=tensor(cache['inputs'][i:i+1,lo:hi])
            target=tensor(cache['codes'][i:i+1,lo:hi],device).long()
            pred=net.level_logits(raw,level,condition)
            ce=F.cross_entropy(pred.flatten(0,1),target.flatten())
            mismatch=F.cross_entropy(net.level_logits(raw,level,wrong).flatten(0,1),target.flatten())
            with net.frozen_reference():
                base=F.cross_entropy(net.level_logits(raw,level,None).flatten(0,1),target.flatten())
            level_sums[level]+=float(ce)
            row_loss+=float(ce)*(hi-lo)/641;row_shuffle+=float(mismatch)*(hi-lo)/641;row_base+=float(base)*(hi-lo)/641
        baseline+=row_base;shuffled+=row_shuffle
        paired.append({'index':i,'nll':row_loss,'shuffled_nll':row_shuffle,'base_nll':row_base})
    kl=[]
    for k,i in enumerate(retention):
        level=(k*5)%12;lo,hi=net.offsets[level:level+2]
        raw=tensor(cache['inputs'][i:i+1,lo:hi])
        with net.frozen_reference():teacher=net.level_logits(raw,level,None).softmax(-1)
        student=net.level_logits(raw,level,None).log_softmax(-1)
        kl.append(float(F.kl_div(student,teacher,reduction='none').sum(-1).mean()))
    samples=[]
    # Both deterministic and sampled text; exact structured route remains authoritative.
    for i in indices[:4]:
        cond=condition_for(encoder,records,[i],device)
        for temperature in (0.,.8):
            codes=net.generate_codes(cond,temperature=temperature)
            plain=decoded_text(net.codec[0],codes,inference,False)[0]
            penalty=decoded_text(net.codec[0],codes,inference,True,True)[0]
            path=parse_path(plain)
            samples.append({'graph_id':records[i]['graph_id'],'verified_path':records[i]['path'],
                'reference':records[i]['text'],'temperature':temperature,'plain_text':plain,
                'penalty_text':penalty,'parsed_plain_path':path,'exact_plain_path':path==records[i]['path']})
    n=len(indices);mean=sum(r['nll'] for r in paired)/n
    return {'step':step,'timestamp':time.time(),'split':split,'count':n,'code_nll':mean,
            'per_level_nll':[x/n for x in level_sums],'frozen_base_code_nll':baseline/n,
            'shuffled_minus_correct_nll':shuffled/n-mean,'retention_kl_probe':sum(kl)/max(1,len(kl)),
            'retention_source':'FineWeb probe; not LM1B benchmark','paired':paired,'samples':samples,
            'greedy_path_exact':sum(s['exact_plain_path'] for s in samples if s['temperature']==0)/4,
            'sampled_path_exact':sum(s['exact_plain_path'] for s in samples if s['temperature']!=0)/4}


def main(args):
    global STOP
    STOP=False
    torch.set_num_threads(2);seed_all(args.seed)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    args.run.mkdir(parents=True,exist_ok=True)
    cache=load_cache();records=cache['records']
    training=[i for i,r in enumerate(records) if r['kind']=='plan' and r['split']=='train']
    retention=[i for i,r in enumerate(records) if r['kind']=='retention' and r['split']=='train']
    net,codec,inference=load_ncm('mps')
    if net.codec[0].identity['codec_id']!=cache['manifest']['codec']['codec_id']:raise ValueError('Codec/cache changed')
    encoder=PlanConditioner().to('mps')
    params=[p for p in net.parameters() if p.requires_grad]+list(encoder.parameters())
    opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.requires_grad],'lr':1e-4},
                          {'params':encoder.parameters(),'lr':3e-4}],weight_decay=.01)
    identity={'sources':source_hashes(),'cache_manifest_sha':sha(ROOT/'data/ncm_cache/manifest.json'),
        'generator_sha':GENERATOR_SHA,'seed':args.seed,'deadline':args.deadline,'steps':args.steps,
        'rank':4,'accumulation':4,'loss':'0.5 position-average CE +0.5 level-average CE; .1 frozen-base KL every update',
        'prefix_corruption':'after step200:20% of microbatches, uniform replacements at rate.05 on earlier levels'}
    atomic_json(args.run/'plan.json',{**identity,'trainable_parameters':sum(p.numel() for p in params),
        'frozen_generator_parameters':97065600,'frozen_codec_parameters':111404672})
    atomic_json(args.run/'process.json',{'pid':os.getpid(),'timestamp':time.time()})
    step=0;best=float('inf');bad=0;elapsed=0.
    if (args.run/'latest.pt').exists():
        saved=torch.load(args.run/'latest.pt',map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Renderer resume identity changed')
        net.load_adapters(saved['adapters']);encoder.load_state_dict(saved['conditioner']);opt.load_state_dict(saved['optimizer'])
        step,best,bad,elapsed=saved['step'],saved['best'],saved['bad'],saved['elapsed'];restore_rng(saved['rng'])
    began=time.monotonic()
    def save():
        save_torch(args.run/'latest.pt',{'adapters':net.adapter_state(),'conditioner':encoder.state_dict(),
            'optimizer':opt.state_dict(),'rng':rng_state(),'identity':identity,'step':step,
            'best':best,'bad':bad,'elapsed':elapsed+time.monotonic()-began})
    def validate():
        nonlocal best,bad
        state=rng_state()
        seed_all(947)
        result=evaluate_renderer(net,encoder,cache,inference,step)
        restore_rng(state)
        atomic_json(args.run/f'eval_{step:07d}.json',result);atomic_json(args.run/'latest_eval.json',result)
        if result['code_nll']<best-.002:
            best=result['code_nll'];bad=0
            save_torch(args.run/'best.pt',{'adapters':net.adapter_state(),'conditioner':encoder.state_dict(),
                'step':step,'identity':identity,'score':best})
        else:bad+=1
        # These are quality gates, not prerequisites for writing a checkpoint.
        atomic_json(args.run/'quality_gates.json',{'step':step,'condition_used':result['shuffled_minus_correct_nll']>.01,
            'retention_probe_acceptable':result['retention_kl_probe']<.15,
            'tiny_free_generation_probe_acceptable':result['greedy_path_exact']>=.5,
            'limitation':'Four free generations and24 validation cases are exploratory, not broad quality evidence'})
    if step==0:validate();save()
    reason='complete';last_save=time.monotonic()
    while step<args.steps:
        if STOP or (args.run.parent/'STOP').exists():reason='stopped';break
        if time.time()>=args.deadline:reason='deadline';break
        issue=resource_issue()
        if issue:reason=issue;break
        net.train();encoder.train();opt.zero_grad(set_to_none=True)
        total=0.
        for micro in range(4):
            i=random.choice(training);level=random.randrange(12)
            lo,hi=net.offsets[level:level+2]
            raw=tensor(cache['inputs'][i:i+1,lo:hi])
            if step>=200 and level>0 and random.random()<.2:
                altered=tensor(cache['codes'][i:i+1]).long()
                noise=torch.rand(altered[:,:lo].shape)<.05
                replacements=torch.randint(8192,altered[:,:lo].shape)
                altered[:,:lo]=torch.where(noise,replacements,altered[:,:lo])
                raw=net.teacher_inputs(altered)[:,lo:hi]
            target=tensor(cache['codes'][i:i+1,lo:hi],'mps').long()
            cond=condition_for(encoder,records,[i],'mps')
            logits=net.level_logits(raw,level,cond)
            weight=.5+.5*12*(hi-lo)/641
            loss=weight*F.cross_entropy(logits.flatten(0,1),target.flatten())/4
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite NCM loss')
            loss.backward();total+=float(loss.detach())
        i=random.choice(retention);level=random.randrange(12);lo,hi=net.offsets[level:level+2]
        raw=tensor(cache['inputs'][i:i+1,lo:hi])
        with torch.no_grad(),net.frozen_reference():teacher=net.level_logits(raw,level,None).softmax(-1)
        logp=net.level_logits(raw,level,None).log_softmax(-1)
        retain=F.kl_div(logp,teacher,reduction='none').sum(-1).mean()
        if not torch.isfinite(retain):raise RuntimeError('Nonfinite retention KL')
        (.1*retain).backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        warm=min(1.,(step+1)/100)
        opt.param_groups[0]['lr']=1e-4*warm;opt.param_groups[1]['lr']=3e-4*warm
        opt.step();step+=1
        if step%10==0:
            metrics={'step':step,'timestamp':time.time(),'weighted_code_loss':total,'retention_kl':float(retain.detach()),
                'grad_norm':float(grad),'elapsed_seconds':elapsed+time.monotonic()-began}
            append_json(args.run/'metrics.jsonl',metrics)
            atomic_json(args.run/'status.json',{'status':'training',**metrics,'target_steps':args.steps,'pid':os.getpid()})
        if step%args.eval_every==0:
            validate();save();last_save=time.monotonic()
            if bad>=6:reason='validation_plateau';break
        elif time.monotonic()-last_save>=120:save();last_save=time.monotonic()
    save()
    if reason in ('complete','validation_plateau') and step%args.eval_every:validate();save()
    atomic_json(args.run/'status.json',{'status':reason,'step':step,'timestamp':time.time(),
        'elapsed_seconds':elapsed+time.monotonic()-began,'best_validation_nll':best})
    if reason in ('low_memory','low_disk','low_battery'):raise RuntimeError(reason)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=44);ap.add_argument('--steps',type=int,default=12000)
    ap.add_argument('--eval-every',type=int,default=500);ap.add_argument('--deadline',type=float,required=True)
    args=ap.parse_args()
    try:
        with heavy_job():main(args)
    except Exception as error:
        atomic_json(args.run/'status.json',{'status':'failed','error':repr(error),'timestamp':time.time()});raise
