"""One bounded default-size Metal optimizer update under a temporary memory cap."""
import json
import time
from pathlib import Path
import psutil
import torch
from .model import V101,losses
from .validate_training import load_examples,pack,clip_and_log
from .provenance import provenance
from .accumulate import backward_effective_batch
from v10_10M.train import canonical_ids,build_optimizers


def main(microbatch_size=None):
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.15);torch.manual_seed(311)
    result=dict(**provenance(),device='mps',context=384,batch_size=8,candidate_update=True,memory_fraction_cap=.15,
        limitation='One shared-device update, not a sustained throughput or thermal benchmark',
        microbatch_size=microbatch_size or 8,
        available_gib_before=psutil.virtual_memory().available/2**30)
    try:
        if result['available_gib_before']<1.5:
            raise RuntimeError('Insufficient current host headroom for a concurrent default-batch check')
        samples,_,tokenizer=load_examples('train',context=384,count=8);batch=pack(samples['text'])
        batch={k:v.to('mps') for k,v in batch.items()}
        model=V101();model.engram.canonical.copy_(canonical_ids(tokenizer,8192));model.place('mps').train()
        opt,table,dense=build_optimizers(model);start=time.perf_counter()
        if microbatch_size is None:
            result['stage']='main_forward';out=model(batch['ids'],batch['lengths'])
            result['stage']='candidate_and_objectives';parts=losses(model,out,**batch,candidate_update=True,generator=torch.Generator().manual_seed(198))
            result['stage']='backward';parts['total'].backward()
            loss_values={k:float(parts[k].detach().cpu()) for k in ['total','token','latent','vq','chunk','explore','selector']}
        else:
            result['stage']='accumulated_forward_backward'
            loss_values=backward_effective_batch(model,batch,microbatch_size,candidate_update=True,seed=198)
        result['stage']='optimizer';grads=clip_and_log(model,dense);opt.step();table.step();torch.mps.synchronize()
        result['stage']='complete'
        result.update(status='passed',parameters=model.counts(),seconds=time.perf_counter()-start,
            loss=loss_values,
            **grads,input_tokens=int(batch['lengths'].sum()),mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,
            mps_driver_gib=torch.mps.driver_allocated_memory()/2**30,rss_gib=psutil.Process().memory_info().rss/2**30)
    except RuntimeError as error:
        result.update(status='failed_or_resource_limited',error=str(error))
    name='default_batch_mps' if microbatch_size is None else f'effective_batch8_micro{microbatch_size}_mps'
    Path(f'v10_1/validation/{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--microbatch-size',type=int);a=p.parse_args();main(a.microbatch_size)
