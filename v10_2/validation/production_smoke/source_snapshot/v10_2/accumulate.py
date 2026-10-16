"""CPU boundary/hash preparation; objective-correct effective-batch accumulation."""
import time
import torch
from .model import losses,chunk_batch
from v10_1.optimization.fast import gradient_metrics


def backward_effective_batch(model,raw,microbatch_size=8,candidate_update=False,seed=0,diagnostics=None):
    if microbatch_size<1:raise ValueError('Positive microbatch size required')
    prepare_start=time.monotonic()
    device=model.embedding.weight.device
    cpu={k:raw[k].detach().cpu() for k in ('ids','lengths','targets','target_mask')}
    ids,lens,targets,mask=[cpu[k] for k in cpu]
    valid=torch.arange(ids.shape[1]//8)[None,:]<lens[:,None]//8
    chunk=chunk_batch(ids,lens,targets,mask,valid,generator=torch.Generator().manual_seed(seed))
    docs=0 if chunk is None else len(chunk['rows']);nt=int(mask.sum())
    indices=model.engram.indices(ids)
    prepare_seconds=time.monotonic()-prepare_start
    keys=['token','future','explore','selector'];weights=[1.,.1,.1,.05]
    reported=torch.zeros(4,device=device);visit_sum=torch.zeros(2,device=device)
    candidate_stats=torch.zeros(4,device=device);off_num=torch.zeros(8,device=device);off_den=torch.zeros(8,device=device)
    code_counts=torch.zeros(model.c.codes,dtype=torch.long);winner_counts=torch.zeros_like(code_counts)
    for lo in range(0,len(ids),microbatch_size):
        hi=min(lo+microbatch_size,len(ids));part={k:v[lo:hi].to(device) for k,v in cpu.items()}
        chosen=None;n=0
        if chunk is not None:
            selected=(chunk['rows']>=lo)&(chunk['rows']<hi);n=int(selected.sum())
            if n:
                chosen={k:chunk[k][selected].to(device) for k in ('rows','boundaries','labels','mask')};chosen['rows']-=lo
        out=model(part['ids'],part['lengths'],indices=indices[lo:hi])
        values=losses(model,out,**part,candidate_update=candidate_update,prepared_chunk=chosen,chunk_prepared=True)
        ratios=[int(mask[lo:hi].sum())/max(1,nt)]+[n/max(1,docs)]*3
        objective=sum(values[k]*r*w for k,r,w in zip(keys,ratios,weights))
        if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite effective-batch objective')
        objective.backward()
        reported+=torch.stack([values[k].detach()*r for k,r in zip(keys,ratios)])
        visit_sum+=torch.stack([v.detach() for v in values['future_visits']])*ratios[1]
        candidate_stats+=torch.stack([values[k].detach() for k in ('selector_regret','utility_gap','utility_spread','selector_agreement')])*ratios[1]
        off_num+=values['future_offset_numerators'];off_den+=values['future_offset_counts']
        code_counts+=torch.bincount(out['codes'].detach().cpu()[valid[lo:hi]],minlength=model.c.codes)
        if values['winners'] is not None:winner_counts+=torch.bincount(values['winners'].detach().cpu(),minlength=model.c.codes)
    # Consolidated scalar transfer rather than one synchronization per loss/component.
    scalars=torch.cat((reported,visit_sum,candidate_stats,off_num,off_den)).cpu().tolist()
    result=dict(zip(keys,scalars[:4]));result['total']=sum(result[k]*w for k,w in zip(keys,weights))
    if diagnostics is not None:
        nc=docs if candidate_update else 0
        den=dict(token=nt,future=docs,explore=nc,selector=nc)
        diagnostics.update(batch_prepare_seconds=prepare_seconds,candidate_documents=nc,future_documents=docs,
            sampled_boundaries=[] if chunk is None else [{'row':int(r),'group':int(g)} for r,g in zip(chunk['rows'],chunk['boundaries'])],
            weighted_losses={k:result[k]*w for k,w in zip(keys,weights)},
            future_supervised_targets=0 if chunk is None else chunk['supervised_targets'],
            selector_regret_sum=scalars[6]*nc,utility_gap=scalars[7] if nc else None,
            utility_spread=scalars[8] if nc else None,selector_agreement=scalars[9] if nc else None,
            code_counts=code_counts.tolist(),winner_counts=winner_counts.tolist(),future_visits=scalars[4:6],
            future_offset_numerators=scalars[10:18],future_offset_counts=scalars[18:26],
            objective_denominators=den,objective_numerators={k:result[k]*v for k,v in den.items()})
    return result
