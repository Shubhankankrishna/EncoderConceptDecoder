"""Explicit objective units and measures; no noisy-mixture bound shortcuts."""
import torch
import torch.nn.functional as F


def gaussian_kl(mu_q, sd_q, mu_p, sd_p, *, validate=True):
    """Stable coordinate KL(q||p), no clamping of loss or gradients.

    log1p(relative delta) preserves near-equality scale information. Around
    zero, a polynomial evaluates expm1(x)-x without subtractive cancellation.
    """
    if validate and any(not torch.isfinite(v).all() for v in (mu_q,sd_q,mu_p,sd_p)):
        raise ValueError('Gaussian parameters must be finite')
    if validate and ((sd_q<=0).any() or (sd_p<=0).any()):raise ValueError('Positive scales required')
    ratio=(sd_q-sd_p)/sd_p
    near=ratio.abs()<.1
    # Safe unused branch: log1p(-1) must not poison torch.where backward.
    t=torch.where(near,torch.log1p(torch.where(near,ratio,torch.zeros_like(ratio))),
                  sd_q.log()-sd_p.log())
    x=2*t
    poly=x.square()*(.5+x*(1/6+x*(1/24+x/120)))
    variance=torch.where(x.abs()<.01,poly,torch.expm1(x)-x)
    return .5*(variance+((mu_q-mu_p)/sd_p).square())


def joint_mode_elbo(mode_bounds, router_logits, assignment_logits):
    """[B,K] variational bound with explicit q(k|X,Y).

    mode_bounds may be unbiased MC estimates. assignment_logits must come
    from an explicit router or independent pilot samples, not those same
    noisy estimates; callers record that provenance in the manifest.
    """
    if mode_bounds.ndim!=2 or not(mode_bounds.shape==router_logits.shape==assignment_logits.shape):
        raise ValueError('Expected matching [batch,modes] tensors')
    log_r=assignment_logits.log_softmax(-1);log_pi=router_logits.log_softmax(-1)
    return (log_r.exp()*(mode_bounds+log_pi-log_r)).sum(-1)


def episode_mode_elbo(step_bounds,valid_steps,router_logits,assignment_logits):
    """One mode per episode: sum [B,T,K] bounds, pay router KL once.

    The caller must construct each step at that mode's teacher-history path.
    This reduction cannot turn arbitrary per-step numbers into a valid bound.
    """
    if step_bounds.ndim!=3 or valid_steps.shape!=step_bounds.shape[:2] or valid_steps.dtype!=torch.bool:
        raise ValueError('Expected [batch,time,mode] and boolean step mask')
    if not valid_steps.any(1).all():raise ValueError('Empty episode')
    totals=step_bounds.masked_fill(~valid_steps[:,:,None],0).sum(1)
    return joint_mode_elbo(totals,router_logits,assignment_logits)


def conditional_distillation(teacher_logits,student_logits,*,prefix_measure):
    """Return per-example sum over positions, plus an honest objective label.

    Full hierarchy KL additionally requires teacher-marginal conditionals
    and prefixes sampled from the same teacher hierarchy distribution.
    """
    labels={'data':'data_prefix_conditional_KL_surrogate',
            'teacher_sample':'teacher_prefix_conditional_KL_estimate',
            'student_sample':'student_prefix_conditional_KL_surrogate'}
    if prefix_measure not in labels:raise ValueError('Explicit prefix sampling measure required')
    t=teacher_logits.detach().log_softmax(-1);s=student_logits.log_softmax(-1)
    terms=(t.exp()*(t-s)).sum(-1)
    return terms.reshape(len(terms),-1).sum(-1),labels[prefix_measure]


def reward_score_proxy(logps,rewards):
    """LOO estimator for independent on-policy trajectories [samples,batch]."""
    if logps.shape!=rewards.shape or logps.ndim!=2 or len(logps)<2:
        raise ValueError('At least two independent draws per example are required')
    rewards=rewards.detach()
    baseline=(rewards.sum(0,keepdim=True)-rewards)/(len(rewards)-1)
    return ((rewards-baseline)*logps).mean()
