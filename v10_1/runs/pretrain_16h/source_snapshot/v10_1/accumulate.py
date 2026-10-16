"""Memory-bounded effective batches with each objective's correct denominator."""
import torch
from .model import losses


def backward_effective_batch(model,batch,microbatch_size=1,candidate_update=False,seed=0,diagnostics=None):
    if microbatch_size<1:raise ValueError('Positive microbatch size required')
    if candidate_update and model.architecture.chunk_objective and not model.architecture.utility_exploration:
        raise ValueError('H-arm accumulation requires independently preselected chunk/candidate boundaries; use S or disable candidates')
    ids,lengths,targets,target_mask=[batch[k] for k in ['ids','lengths','targets','target_mask']]
    pair_counts=(lengths//8-1).clamp_min(0)
    legal=torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None]
    chunk_docs=((target_mask & legal)[:,7:].any(-1) & (lengths>=8))
    candidate_docs=chunk_docs if model.architecture.utility_exploration else (lengths//8>1)
    totals=dict(token=max(1,int(target_mask.sum())),pairs=max(1,int(pair_counts.sum())),
                chunk=max(1,int(chunk_docs.sum())),candidate=max(1,int(candidate_docs.sum())))
    generator=torch.Generator().manual_seed(seed);reported={k:0. for k in ['total','token','latent','vq','chunk','explore','selector']}
    if diagnostics is not None:
        diagnostics.update(candidate_documents=0,chunk_supervised_targets=0,selector_regret_sum=0.,
                           code_counts=[0]*model.c.codes,latent_visits=[0.,0.])
    for lo in range(0,len(ids),microbatch_size):
        hi=min(len(ids),lo+microbatch_size)
        part={k:v[lo:hi] for k,v in batch.items()}
        out=model(part['ids'],part['lengths'])
        values=losses(model,out,**part,candidate_update=candidate_update,generator=generator)
        ratios=dict(token=int(part['target_mask'].sum())/totals['token'],
            latent=int(pair_counts[lo:hi].sum())/totals['pairs'],vq=int(pair_counts[lo:hi].sum())/totals['pairs'],
            chunk=int(chunk_docs[lo:hi].sum())/totals['chunk'],
            explore=int(candidate_docs[lo:hi].sum())/totals['candidate'],selector=int(candidate_docs[lo:hi].sum())/totals['candidate'])
        weights=dict(token=1.,latent=.1,vq=.01,chunk=.1,explore=.1,selector=.05)
        total=sum(values[k]*ratios[k]*weights[k] for k in weights)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite microbatch objective')
        total.backward()
        reported['total']+=float(total.detach())
        for k in weights:reported[k]+=float(values[k].detach())*ratios[k]
        if diagnostics is not None:
            n=int(candidate_docs[lo:hi].sum()) if values['candidate_executed'] else 0
            diagnostics['candidate_documents']+=n
            diagnostics['selector_regret_sum']+=(values['selector_regret'] or 0.)*n
            diagnostics['chunk_supervised_targets']+=values['chunk_supervised_targets']
            counts=torch.bincount(out['codes'][out['valid_groups']].detach().cpu(),minlength=model.c.codes).tolist()
            diagnostics['code_counts']=[a+b for a,b in zip(diagnostics['code_counts'],counts)]
            for i,v in enumerate(values['latent_visits']):diagnostics['latent_visits'][i]+=float(v.detach())*ratios['latent']
        del out,values,total
    if diagnostics is not None:
        den=dict(token=int(target_mask.sum()),latent=int(pair_counts.sum()),vq=int(pair_counts.sum()),
                 chunk=int(chunk_docs.sum()),explore=diagnostics['candidate_documents'],selector=diagnostics['candidate_documents'])
        diagnostics['objective_denominators']=den
        diagnostics['objective_numerators']={k:reported[k]*n for k,n in den.items()}
    return reported
