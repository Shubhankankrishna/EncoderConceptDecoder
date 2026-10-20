"""Actual cached-data adapter optimization and exact optimizer/RNG resume probe."""
import copy
import json
import time

import torch
import torch.nn.functional as F

from .adapt import condition_for,tensor
from .cache import load_cache
from .common import ROOT,atomic_json,heavy_job,restore_rng,rng_state,save_torch,seed_all
from .model import PlanConditioner
from .ncm import load_ncm


def main():
    torch.set_num_threads(2);seed_all(72)
    cache=load_cache();net,codec,_=load_ncm('mps');enc=PlanConditioner().to('mps')
    params=[p for p in net.parameters() if p.requires_grad]+list(enc.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4)
    selected=[0,1];level=4;lo,hi=net.offsets[level:level+2]
    raw=tensor(cache['inputs'][selected,lo:hi]);target=tensor(cache['codes'][selected,lo:hi],'mps').long()
    source_codes=tensor(cache['codes'][selected]).long()
    rebuilt=net.teacher_inputs(source_codes)
    torch.testing.assert_close(rebuilt,tensor(cache['inputs'][selected]),atol=0,rtol=0)
    altered=source_codes.clone()
    altered[:,net.offsets[6]:]=torch.randint(8192,altered[:,net.offsets[6]:].shape)
    changed=net.teacher_inputs(altered)
    torch.testing.assert_close(rebuilt[:,:net.offsets[7]],changed[:,:net.offsets[7]],atol=0,rtol=0)
    def update():
        opt.zero_grad(set_to_none=True)
        cond=condition_for(enc,cache['records'],selected,'mps')
        logits=net.level_logits(raw,level,cond)
        loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        return float(loss.detach())
    losses=[];times=[]
    for _ in range(16):
        tick=time.perf_counter();losses.append(update());torch.mps.synchronize();times.append(time.perf_counter()-tick)
    assert losses[-1]<losses[0],losses
    payload={'adapters':net.adapter_state(),'conditioner':enc.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state()}
    path=ROOT/'reports/adapter_resume_probe.pt';save_torch(path,payload)
    expected_loss=update();expected=net.adapter_state();expected_enc=copy.deepcopy(enc.state_dict())
    saved=torch.load(path,map_location='cpu',weights_only=True)
    net.load_adapters(saved['adapters']);enc.load_state_dict(saved['conditioner']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    replay_loss=update()
    delta=max(float((v-expected[n]).abs().max()) for n,v in net.adapter_state().items())
    enc_delta=max(float((v-expected_enc[n]).abs().max()) for n,v in enc.state_dict().items())
    assert delta==0 and enc_delta==0,(delta,enc_delta)
    with torch.no_grad():
        cond=condition_for(enc,cache['records'],selected,'mps')
        actual=net.level_logits(raw,level,cond)
        wrong=net.level_logits(raw,level,cond.flip(0))
        sensitivity=float((actual-wrong).abs().mean())
        with net.frozen_reference():reference=net.level_logits(raw,level,None)
        adapted=net.level_logits(raw,level,None)
        drift=float(F.kl_div(adapted.log_softmax(-1),reference.softmax(-1),reduction='none').sum(-1).mean())
    assert sensitivity>1e-6,sensitivity
    assert any(p.grad is not None and float(p.grad.norm())>0 for p in enc.parameters())
    report={'passed':True,'losses':losses,'adapter_resume_weight_max_abs':delta,'conditioner_resume_weight_max_abs':enc_delta,
            'cache_float32_exact_parity':True,'no_target_or_future_level_leakage':True,
            'resume_loss_difference':abs(expected_loss-replay_loss),'conditioning_logit_sensitivity':sensitivity,
            'retention_kl_after_probe':drift,'median_batch2_sparse_update_seconds':sorted(times[2:])[7],
            'trainable_renderer_parameters':sum(p.numel() for p in params),
            'mps_driver_bytes':torch.mps.driver_allocated_memory(),'samples_from_train_only':selected}
    atomic_json(ROOT/'reports/adapter_preflight.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    with heavy_job():main()
