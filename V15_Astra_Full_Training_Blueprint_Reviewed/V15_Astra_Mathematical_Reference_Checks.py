"""Independent tiny mathematical invariants for V15 engineering spec.
This is NOT a pretrained NCM implementation or model training. It tests discrete
factorization, Gaussian KL, mixture bounds, score-gradient identity, root visits,
terminal backup, bounded support intuition, and zero-effect LoRA initialization.
"""
import json, math, torch
from collections import defaultdict
from dataclasses import dataclass

LEVELS=(1,8,16,24,32,40,52,64,76,92,108,128)
OFFSETS=(0,1,9,25,49,81,121,173,237,313,405,513,641)


def test_code_layout():
    assert len(LEVELS)==12 and sum(LEVELS)==641
    assert all(OFFSETS[i+1]-OFFSETS[i]==LEVELS[i] for i in range(12))
    ids=torch.arange(641)
    assert torch.equal(torch.cat([ids[OFFSETS[i]:OFFSETS[i+1]] for i in range(12)]),ids)
    assert 641*4096 == 2625536
    assert 641*4096*4 == 10502144 # FP32 bytes per block logits
    assert 1000000*641*2 == 1282000000 # uint16 storage
    return '12 level offsets, 641 code positions, logit/cache bytes pass'


def test_toy_hierarchy_likelihood():
    # Toy: one code at coarse level, two independent given coarse at fine level
    p=torch.tensor([0.8,0.5,0.25],dtype=torch.float64)
    nll=-p.log().sum()
    assert abs(nll.item()-math.log(10))<1e-12
    assert abs((nll/3).item()-0.7675283643313485)<1e-10
    level_balanced=(-p[0].log()+(-p[1].log()-p[2].log())/2)/2
    assert abs(level_balanced.item()-0.6314321618767297)<1e-8
    return {'summed_nll':nll.item(),'per_code_nll':(nll/3).item(),'level_surrogate':level_balanced.item()}


def test_gaussian_kl():
    torch.manual_seed(25)
    mu_q=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    mu_p=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    sd_q=torch.exp(torch.randn(2,3,dtype=torch.float64)*0.2).requires_grad_(True)
    sd_p=torch.exp(torch.randn(2,3,dtype=torch.float64)*0.2).requires_grad_(True)
    kl=0.5*(2*torch.log(sd_p/sd_q)+(sd_q**2+(mu_q-mu_p)**2)/sd_p**2-1)
    ref=torch.distributions.kl_divergence(torch.distributions.Normal(mu_q,sd_q),torch.distributions.Normal(mu_p,sd_p))
    assert torch.allclose(kl,ref,rtol=1e-11,atol=1e-11)
    assert torch.all(kl>=-1e-11)
    kl.sum().backward()
    assert torch.isfinite(mu_q.grad).all() and torch.isfinite(mu_p.grad).all()
    ident=torch.distributions.kl_divergence(torch.distributions.Normal(mu_q.detach(),sd_q.detach()),torch.distributions.Normal(mu_q.detach(),sd_q.detach())).abs().max()
    assert ident<1e-11
    return {'max_abs_drift_vs_torch':(kl-ref).abs().max().item(),'identical_KL':ident.item()}


def test_categorical_mixture():
    # Both p(y=0) and p(y=1) properly normalized under each strategy.
    pi=torch.tensor([0.5,0.5],dtype=torch.float64)
    p=torch.tensor([[0.8,0.2],[0.3,0.7]],dtype=torch.float64)
    mix=(pi[:,None]*p).sum(0)
    assert torch.isclose(mix.sum(),torch.tensor(1.,dtype=torch.float64))
    for y in range(2):
        l=torch.logsumexp(pi.log()+p[:,y].log(),dim=0)
        assert torch.isclose(l,mix[y].log())
    B=torch.tensor([-2.,-4.],dtype=torch.float64)
    r=torch.softmax(pi.log()+B,dim=0)
    assert abs(r[0].item()-0.8807970779778823)<1e-12
    lower=torch.logsumexp(pi.log()+B,dim=0)
    # Full likelihood chosen to be larger than both individual ELBO bounds.
    true_log=torch.logsumexp(pi.log()+torch.tensor([-1.,-3.],dtype=torch.float64),0)
    assert lower<true_log
    return {'mixture_p_y0':mix[0].item(),'responsibility_first_mode':r[0].item(),'bound_gap':(true_log-lower).item()}


def test_score_gradient_with_discrete_actions():
    # For binary action A∈{0,1}, p(A=1|z)=sigmoid(z), R(A)=A.
    z=torch.tensor(0.4,dtype=torch.float64,requires_grad=True)
    p=torch.sigmoid(z)
    # Enumerating the score function precisely instead of noisy Monte Carlo:
    rewards=torch.tensor([0.,1.],dtype=torch.float64)
    log_probs=torch.stack([torch.log1p(-p),torch.log(p)])
    baseline=torch.tensor(0.3,dtype=torch.float64)
    # weighted by probabilities; score-gradient identity
    distribution=torch.stack([1-p,p])
    surrogate=(distribution.detach()*(rewards-baseline)*log_probs).sum()
    g=torch.autograd.grad(surrogate,z)[0]
    actual=p.detach()*(1-p.detach())
    assert torch.allclose(g,actual,rtol=1e-11,atol=1e-11)
    # Finite difference of expected reward sigmoid(z)
    delta=1e-5
    fd=(torch.sigmoid(z.detach()+delta)-torch.sigmoid(z.detach()-delta))/(2*delta)
    assert abs(g.item()-fd.item())<1e-10
    return {'score_grad':g.item(),'finite_diff':fd.item(),'true_grad':actual.item()}


def test_terminal_backup():
    # 5 -> 6 -> 0 under add1 in mod7: rewards [0,1]
    path_rewards=[0.,1.]
    ret=0. # terminal bootstrap ZERO since reward already delivered on incoming edge
    for reward in path_rewards[::-1]:
        ret=reward+1.*ret
    assert ret==1.0
    assert ret != 2.0
    return {'root_return':ret,'terminal_reward_counted_once':True}


def test_lora_zero_initialization():
    torch.manual_seed(7)
    din,dout,r=5,3,2
    W=torch.randn(dout,din,dtype=torch.float64)
    A=torch.randn(r,din,dtype=torch.float64)*.01
    B=torch.zeros(dout,r,dtype=torch.float64)
    x=torch.randn(din,dtype=torch.float64)
    for gate in [0.,0.5,1.]:
        y=(W@x)+gate*(B@A@x)
        assert torch.equal(y,W@x)
    return 'zero-initialized LoRA B leaves base output unchanged under any gate'


def test_gmm_support_direction():
    def normal_logpdf(x,mu,sd):
        return -0.5*math.log(2*math.pi)-math.log(sd)-0.5*((x-mu)/sd)**2
    def gate(x):
        lp=normal_logpdf(x,4,.5)
        ln=normal_logpdf(x,0,2)
        return 1/(1+math.exp(-(lp-ln)))
    assert gate(4.)>.95 and gate(0.)<.01
    return {'new_support_gate':gate(4.),'old_support_gate':gate(0.)}


def test_puct_root_accounting():
    # Flat deterministic graph. Root expanded ahead of count.
    N={'add1':0,'mul2':0};W={'add1':0.,'mul2':0.}
    prior={'add1':.6,'mul2':.4}
    for _ in range(20):
        scores={a:(W[a]/N[a] if N[a]>0 else 0.) + 1.2*prior[a]*math.sqrt(1+sum(N.values()))/(1+N[a]) for a in N}
        a=max(scores,key=scores.get)
        # exact toy rollout to terminal from child using greedy plan: MUL2 slower, reward=0 in budget
        result=1. if a=='add1' else 0.
        N[a]+=1;W[a]+=result
    assert sum(N.values())==20 and N['add1']>N['mul2']
    assert W['add1']/N['add1']==1.0
    return {'total_root_visits':sum(N.values()),'add1_visits':N['add1'],'mul2_visits':N['mul2']}


def main():
    funcs=[test_code_layout,test_toy_hierarchy_likelihood,test_gaussian_kl,test_categorical_mixture,
           test_score_gradient_with_discrete_actions,test_terminal_backup,test_lora_zero_initialization,
           test_gmm_support_direction,test_puct_root_accounting]
    results={}
    for f in funcs:
        results[f.__name__]=f()
    with open('/mnt/data/V15_Astra_Math_Checks_Result.json','w') as out:
        json.dump({'pass':len(results),'failed':0,'tests':results,'scope':'small mathematical invariants only; no official NCM checkpoint loaded or model trained'},out,indent=2)
    print(json.dumps({'pass':len(results),'results':results},indent=2))

if __name__=='__main__':
    main()
