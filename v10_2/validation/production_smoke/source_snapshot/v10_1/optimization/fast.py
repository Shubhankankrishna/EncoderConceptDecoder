"""Keep batch bookkeeping on CPU and combine device-to-host scalar transfers.

The model, losses, sampled boundary RNG, optimizer groups and update frequency
are unchanged. This path is intentionally restricted to full-batch arm S.
"""
import math
import torch
from v10_1.model import losses,chunk_batch


def backward_full_batch(model,raw,device,candidate_update=False,seed=0,diagnostics=None):
    if not all(vars(model.architecture).values()):raise ValueError('Fast path requires full S architecture')
    cpu={k:raw[k] for k in ('ids','lengths','targets','target_mask')}
    if any(v.device.type!='cpu' for v in cpu.values()):raise ValueError('Pre-transfer CPU batch required')
    ids,lens,targets,mask=[cpu[k] for k in ('ids','lengths','targets','target_mask')]
    g=ids.shape[1]//8
    valid=torch.arange(g)[None,:]<lens[:,None]//8
    generator=torch.Generator().manual_seed(seed)
    chosen=chunk_batch(ids,lens,targets,mask,valid,generator=generator)
    # Exact same CPU generator and torch.randint calls as the reference helper.
    boundaries=chosen['boundaries'].to(device) if chosen is not None else None
    indices=model.engram.indices(ids)
    batch={k:v.to(device) for k,v in cpu.items()}
    out=model(batch['ids'],batch['lengths'],indices=indices)
    values=losses(model,out,**batch,candidate_update=candidate_update,generator=generator,boundaries=boundaries)
    if not bool(torch.isfinite(values['total'])):raise FloatingPointError('Nonfinite full-batch objective')
    values['total'].backward()
    names=['total','token','latent','vq','chunk','explore','selector']
    scalars=torch.stack([values[k].detach() for k in names]+[v.detach() for v in values['latent_visits']]).cpu().tolist()
    reported=dict(zip(names,scalars[:7]))
    if diagnostics is not None:
        n=0 if chosen is None else len(chosen['rows'])
        nc=n if values['candidate_executed'] else 0
        # Transfer all codes once; boolean filtering and bincount run on CPU.
        codes=out['codes'].detach().cpu()[valid]
        den=dict(token=int(mask.sum()),latent=int((lens//8-1).clamp_min(0).sum()),vq=int((lens//8-1).clamp_min(0).sum()),
                 chunk=n,explore=nc,selector=nc)
        diagnostics.update(candidate_documents=nc,chunk_supervised_targets=0 if chosen is None else chosen['supervised_targets'],
            selector_regret_sum=(values['selector_regret'] or 0.)*nc,
            code_counts=torch.bincount(codes,minlength=model.c.codes).tolist(),latent_visits=scalars[7:],
            objective_denominators=den,objective_numerators={k:reported[k]*v for k,v in den.items()})
    return reported


def gradient_metrics(model,dense,detailed=False):
    # The old detailed telemetry synchronizes once for EVERY parameter. Aggregate
    # each component on-device, then make one transfer for the complete report.
    components={}
    if detailed:
        for name,p in model.named_parameters():
            if p.grad is not None and name!='engram.table.weight':
                components.setdefault(name.split('.')[0],[]).append(p.grad.detach().square().sum())
        names=list(components)
        sums=torch.stack([torch.stack(components[k]).sum() for k in names]).cpu().tolist()
        components={k:math.sqrt(v) for k,v in zip(names,sums)}
    norm=float(torch.nn.utils.clip_grad_norm_(dense,1.,error_if_nonfinite=True))
    g=model.engram.table.weight.grad
    if g is None:raise ValueError('Missing Engram gradient')
    g=g.coalesce();model.engram.table.weight.grad=g;v=g.values();sn=float(v.norm())
    if not math.isfinite(sn):raise FloatingPointError('Nonfinite Engram gradient')
    v.mul_(min(1.,1./max(sn,1e-12)))
    return dict(dense_grad_norm=norm,dense_grad_norm_post=min(norm,1.),dense_clip_factor=min(1.,1./max(norm,1e-12)),
        engram_grad_norm=sn,engram_grad_norm_post=min(sn,1.),engram_clip_factor=min(1.,1./max(sn,1e-12)),
        engram_rows_touched=g._nnz(),component_grad_norms=components)
