import copy
from dataclasses import asdict
import unittest

import torch
from torch.distributions import Normal,kl_divergence

from .gram import AstraGRAM,GramConfig,gaussian_kl


def small():
    return GramConfig(width=24,code_dim=8,vocab=17,levels=(1,2,4),heads=3,ff=32,
                      ctx_layers=1,head_layers=2,slots=3,state_length=8,gaussian_hidden=16)


class TestGRAM(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(37)
        self.m=AstraGRAM(small());self.x=torch.randint(17,(2,7));self.y=torch.randint(17,(2,7));self.s=torch.randint(20,(2,8))
        self.noise=torch.randn(2,2,3,24)

    def test_gaussian_kl_matches_distribution_and_gradients(self):
        mq=torch.randn(3,4,dtype=torch.float64,requires_grad=True);mp=torch.randn_like(mq,requires_grad=True)
        sq=torch.rand(3,4,dtype=torch.float64,requires_grad=True)+.01;sp=torch.rand_like(sq,requires_grad=True)+.01
        actual=gaussian_kl(mq,sq,mp,sp);ref=kl_divergence(Normal(mq,sq),Normal(mp,sp))
        torch.testing.assert_close(actual,ref,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(gaussian_kl(mp,sp,mp,sp),torch.zeros_like(mp),atol=0,rtol=0)
        for g in torch.autograd.grad(actual.sum(),(mq,mp,sq,sp)):self.assertTrue(torch.isfinite(g).all())
        self.assertGreaterEqual(float(actual.min()),0.)

    def test_exact_elbo_units(self):
        # Make q differ so this checks a nonzero KL, not only CE + zero.
        with torch.no_grad():self.m.posterior.out.bias[:24]=.01
        mem=self.m.encode_context(self.x,self.s);h,d=self.m.full_bptt(mem,self.y,self.noise)
        expected=(self.m.conditional_nll(mem,h,self.y).sum(1)+d['kl']).mean()/7
        actual,stats=self.m(self.x,self.s,self.y,noise=self.noise)
        torch.testing.assert_close(actual,expected)
        self.assertGreater(float(stats['kl_sum']),0.)

    def test_full_and_truncated_gradient_lifecycles(self):
        for truncated in (False,True):
            m=copy.deepcopy(self.m);mem=m.encode_context(self.x,self.s)
            fn=m.gram_truncated_last_transition if truncated else m.full_bptt
            h,d=fn(mem,self.y,self.noise,retain_states=True)
            (m.conditional_nll(mem,h,self.y).sum()+d['kl'].sum()).backward()
            if truncated:self.assertFalse(d['trace'][0].requires_grad)
            else:self.assertGreater(float(d['trace'][0].grad.norm()),0.)
            self.assertGreater(float(m.slow.ffout.weight.grad.norm()),0.)

    def test_posterior_target_dependency_and_prior_separation(self):
        # Gaussian output starts at zero; perturb to exercise its target path.
        with torch.no_grad():self.m.posterior.out.weight.normal_(std=.02)
        mem=self.m.encode_context(self.x,self.s);y2=(self.y+1)%17
        a,_=self.m.full_bptt(mem,self.y,self.noise);b,_=self.m.full_bptt(mem,y2,self.noise)
        self.assertGreater(float((a-b).abs().max()),1e-6)
        p1=self.m.generate(self.x,self.s,noise=self.noise)
        self.m.full_bptt(mem,y2,self.noise)
        p2=self.m.generate(self.x,self.s,noise=self.noise)
        self.assertTrue(torch.equal(p1,p2))

    def test_physical_posterior_removal_and_rng_reproduction(self):
        self.m.eval();deployment=AstraGRAM(small(),include_posterior=False).eval()
        state=self.m.inference_state_dict();deployment.load_state_dict(state,strict=True)
        self.assertFalse(hasattr(deployment,'posterior'));self.assertFalse(hasattr(deployment,'target_encoder'))
        self.assertFalse(any(k.startswith(('posterior.','target_encoder.')) for k in state))
        torch.manual_seed(841);a=self.m.generate(self.x,self.s)
        torch.manual_seed(841);b=deployment.generate(self.x,self.s)
        self.assertTrue(torch.equal(a,b))
        mem=deployment.encode_context(self.x,self.s)
        h1,_=deployment._trajectory(mem,noise=self.noise);h2,_=deployment._trajectory(mem,noise=-self.noise)
        self.assertGreater(float((h1-h2).abs().max()),.001)

    def test_prior_network_conditioning_uses_same_posterior_path(self):
        calls=[]
        handle=self.m.prior.register_forward_pre_hook(lambda mod,args:calls.append(args[0].detach().clone()))
        with torch.no_grad():self.m.posterior.out.bias[:24]=.5
        mem=self.m.encode_context(self.x,self.s)
        self.m.full_bptt(mem,self.y,self.noise);q_calls=calls[:];calls.clear()
        self.m._trajectory(mem,noise=self.noise);p_calls=calls[:];handle.remove()
        torch.testing.assert_close(q_calls[0],p_calls[0])
        self.assertGreater(float((q_calls[1]-p_calls[1]).abs().max()),.1)

    def test_parameter_budget_includes_posterior(self):
        m=AstraGRAM();self.assertEqual(m.parameter_count(),10000333)
        self.assertEqual(sum(p.numel() for n,p in m.named_parameters() if n.startswith(('posterior.','target_encoder.'))),237505)
        self.assertEqual(sum(v.numel() for v in m.inference_state_dict().values()),9762828)

    def test_prior_mc_likelihood_sums_before_logmeanexp(self):
        m=self.m.eval();mem=m.encode_context(self.x,self.s)
        torch.manual_seed(903);logps=[]
        for _ in range(4):
            h,_=m._trajectory(mem)
            logps.append(-m.conditional_nll(mem,h,self.y).sum(1))
        expected=-(torch.logsumexp(torch.stack(logps),0)-__import__('math').log(4)).mean()/7
        torch.manual_seed(903);actual=m.score_prior(self.x,self.s,self.y,4)['nll']
        torch.testing.assert_close(actual,expected)

    def test_gaussian_finite_difference(self):
        # Fixed reparameterization noise is essential for a meaningful derivative.
        m=self.m.double();noise=self.noise.double()
        param=m.posterior.out.bias;coordinate=3
        with torch.no_grad():param[coordinate]=.01
        loss,_=m(self.x,self.s,self.y,noise=noise);loss.backward()
        analytic=float(param.grad[coordinate]);eps=1e-5
        with torch.no_grad():
            original=float(param[coordinate]);param[coordinate]=original+eps
            plus=float(m(self.x,self.s,self.y,noise=noise)[0]);param[coordinate]=original-eps
            minus=float(m(self.x,self.s,self.y,noise=noise)[0]);param[coordinate]=original
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=6)

    def test_slot_isotropic_ablation_and_sigma_floor(self):
        c=small();c.variance='slot_isotropic';m=AstraGRAM(c)
        mem=m.encode_context(self.x,self.s);h,d=m.full_bptt(mem,self.y,self.noise)
        self.assertTrue(torch.isfinite(h).all());self.assertGreater(float(d['sigma'].min()),0.)
        with torch.no_grad():m.prior.out.bias[c.width:]=-1000
        _,d=m._trajectory(mem,noise=self.noise)
        self.assertGreaterEqual(float(d['sigma'].min()),c.sigma_min)


if __name__=='__main__':unittest.main(verbosity=2)
