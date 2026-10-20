"""Noise interventions report sensitivity, not causal benefit of a trained module."""
import torch
from .gram import AstraGRAM


@torch.inference_mode()
def prior_sensitivity(model,x,state,reference_prefix,noise):
    """Compare fixed prior paths under identical teacher-forced code prefixes.

    reference_prefix is used ONLY in the output conditional diagnostic. Neither
    prior trajectory reads it. Behavioral benefit still needs trained controls.
    """
    if not isinstance(model,AstraGRAM):raise ValueError('GRAM required')
    mem=model.encode_context(x,state)
    a,_=model._trajectory(mem,noise=noise);b,_=model._trajectory(mem,noise=-noise)
    mean,_=model._trajectory(mem,mean=True);per_level=[]
    for i in range(len(model.config.levels)):
        la=model.logits(model.decode_hidden(mem,a,reference_prefix,i),i)
        lb=model.logits(model.decode_hidden(mem,b,reference_prefix,i),i)
        per_level.append(float((la-lb).square().mean().sqrt()))
    return dict(prior_state_rms_difference=float((a-b).square().mean().sqrt()),
                prior_vs_mean_state_rms=float((a-mean).square().mean().sqrt()),
                conditional_logit_rms_by_level=per_level,
                interpretation='Fixed-prefix sensitivity intervention; no claim of improved decisions')
