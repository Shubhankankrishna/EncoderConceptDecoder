import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.distributions import Normal,kl_divergence
from ..gram import AstraGRAM
from ..model import AstraHRM,Config
from ..controls import make_model,copy_common_initialization,NonrecurrentCodePolicy,StateActionPolicy
from ..objectives import gaussian_kl,joint_mode_elbo,conditional_distillation,reward_score_proxy
from ..metrics import cluster_interval,paired_difference,episode_report,code_diagnostics
from ..baselines import bfs_plan,distance_heuristic
from ..representation import serialize_observation,parse_observation,fixed_observed_blocks,require_representation_contract
from ..benchmark import make_benchmark
from ..value import ValueContract,HorizonValue,search_state_key
from ..continual import GatedLoRA,AdapterSlots,BoundedVerifiedMemory,PromotionMetrics,PromotionThresholds
from ..isolation import output_path,ROOT,LEGACY,require_legacy_idle
from .test_gram_regression import small


class NumericalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(39)

    def test_nearly_equal_fp32_kl_and_gradient(self):
        sp=torch.full((100,),.13);sq=(sp*(1+torch.linspace(-1e-5,1e-5,100))).requires_grad_()
        mp=torch.zeros_like(sp);mq=torch.linspace(-1e-7,1e-7,100).requires_grad_()
        actual=gaussian_kl(mq,sq,mp,sp)
        ref=kl_divergence(Normal(mq.double(),sq.double()),Normal(mp.double(),sp.double()))
        self.assertTrue((actual>=0).all())
        torch.testing.assert_close(actual.double(),ref,atol=2e-16,rtol=3e-5)
        g=torch.autograd.grad(actual.sum(),(mq,sq),retain_graph=True)
        rg=torch.autograd.grad(ref.sum(),(mq,sq))
        for x,y in zip(g,rg):torch.testing.assert_close(x,y,atol=2e-9,rtol=1e-5)

    def test_terminal_elbo_multiple_phases(self):
        m=AstraGRAM(small());x=torch.randint(17,(2,7));y=torch.randint(17,(2,7));s=torch.randint(30,(2,8))
        noise=torch.randn(4,2,3,24);mem=m.encode_context(x,s)
        with torch.no_grad():m.posterior.out.bias[:24]=.03
        h,d=m.full_bptt(mem,y,noise,phases=2)
        expected=(m.conditional_nll(mem,h,y).sum(1)+d['kl']).mean()/7
        actual,stats=m(x,s,y,noise=noise,phases=2)
        torch.testing.assert_close(expected,actual);self.assertEqual(stats['objective_kind'],'terminal_full_ELBO')
        aux,info=m(x,s,y,noise=noise,phases=2,auxiliary_weight=.2)
        torch.testing.assert_close(aux,actual+.2*info['auxiliary_nll'])
        self.assertEqual(info['objective_kind'],'deep_supervision_augmented')
        with self.assertRaises(ValueError):m(x,s,y,corrupt=.1)
        with self.assertRaises(ValueError):m(x,s,y,phases=2,gradient_mode='gram_truncated_last_transition')

    def test_joint_mode_bound_and_router_gradients(self):
        bounds=torch.tensor([[-2.,-4.]],requires_grad=True);pi=torch.zeros_like(bounds,requires_grad=True)
        # Deterministic bounds permit exact optimal responsibility in this toy.
        q=bounds.detach().clone().requires_grad_()
        actual=joint_mode_elbo(bounds,pi,q)
        expected=torch.logsumexp(bounds-math.log(2),-1)
        torch.testing.assert_close(actual,expected)
        actual.sum().backward();self.assertIsNotNone(pi.grad)
        arbitrary=joint_mode_elbo(bounds.detach(),pi.detach(),torch.zeros_like(q))
        self.assertLess(float(arbitrary),float(expected.detach()))

    def test_distillation_prefix_measure_and_teacher_stop_gradient(self):
        t=torch.randn(2,3,5,requires_grad=True);s=torch.randn_like(t,requires_grad=True)
        loss,label=conditional_distillation(t,s,prefix_measure='data')
        self.assertEqual(label,'data_prefix_conditional_KL_surrogate');loss.mean().backward()
        self.assertIsNone(t.grad);self.assertIsNotNone(s.grad)
        with self.assertRaises(ValueError):conditional_distillation(t,s,prefix_measure='unknown')

    def test_loo_reward_gradient_by_exact_enumeration(self):
        # Enumerate both independent Bernoulli draws and their true probability.
        z=torch.tensor(.4,requires_grad=True);p=z.sigmoid();expected_grad=0.
        for a in (0,1):
            for b in (0,1):
                r=torch.tensor([[float(a)],[float(b)]])
                logp=torch.stack([torch.log(p if v else 1-p) for v in (a,b)])[:,None]
                proxy=reward_score_proxy(logp,r)
                grad=torch.autograd.grad(proxy,z,retain_graph=True)[0]
                prob=(p if a else 1-p)*(p if b else 1-p)
                expected_grad+=float((prob*grad).detach())
        self.assertAlmostEqual(expected_grad,float((p*(1-p)).detach()),places=6)


class ModelControlTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(39)

    def test_cached_generation_matches_reference(self):
        for m in (AstraGRAM(small()),AstraHRM(Config(**{k:v for k,v in asdict(small()).items() if k in Config.__dataclass_fields__}))):
            x=torch.randint(17,(2,7));s=torch.randint(30,(2,8));mem=m.encode_context(x,s);h,_=m.reason(mem)
            y=torch.zeros_like(x)
            for i,n in enumerate(m.config.levels):
                z=m.logits(m.decode_hidden(mem,h,y,i),i)
                a,b=m.offsets[i:i+2];y[:,a:b]=z.argmax(-1)
            actual=m.generate_from_state(mem,h)
            self.assertTrue(torch.equal(y,actual))

    def test_cached_logits_and_one_summary_per_previous_level(self):
        m=AstraGRAM(small());x=torch.randint(17,(2,7));s=torch.randint(30,(2,8))
        mem=m.encode_context(x,s);h,_=m.reason(mem);y=torch.randint(17,(2,7))
        for i in range(3):
            prev=m.summaries(y)[:,:i]
            a=m.decode_hidden(mem,h,y,i);b=m.decode_hidden(mem,h,level=i,previous_summaries=prev)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        with patch.object(m,'summarize_level',wraps=m.summarize_level) as fn:
            m.generate_from_state(mem,h)
            self.assertEqual(fn.call_count,2)

    def test_state_only_invariance_to_code_input(self):
        c=small();c.context_source='state';m=AstraGRAM(c)
        x=torch.randint(17,(2,7));s=torch.randint(30,(2,8))
        torch.testing.assert_close(m.encode_context(x,s),m.encode_context((x+1)%17,s),rtol=0,atol=0)

    def test_code_only_invariance_to_state_and_multiblock_order(self):
        c=small();c.context_source='codes';m=AstraGRAM(c)
        x=torch.randint(17,(2,2,7));s=torch.randint(30,(2,8))
        torch.testing.assert_close(m.encode_context(x,s),m.encode_context(x,s+1),rtol=0,atol=0)
        self.assertGreater(float((m.encode_context(x,s)-m.encode_context(x.flip(1),s)).abs().max().detach()),1e-6)

    def test_raw_and_hybrid_context(self):
        for kind,slots in [('raw',7),('hybrid',7),('pooled',3)]:
            c=small();c.context_source='codes';c.context_pooling=kind;m=AstraGRAM(c)
            mem=m.encode_context(torch.randint(17,(2,2,7)),torch.zeros(2,8,dtype=torch.long))
            self.assertEqual(mem.shape,(2,slots*2,24))

    def test_true_nonrecurrent_control_and_common_initialization(self):
        c={k:v for k,v in asdict(small()).items() if k in Config.__dataclass_fields__}
        a=make_model('code_hrm',c);b=make_model('code_gram',{**c,'gaussian_hidden':16})
        names=copy_common_initialization(a,b)
        self.assertIn('fast.ffout.weight',names)
        for n in names:torch.testing.assert_close(a.state_dict()[n],b.state_dict()[n],atol=0,rtol=0)
        nr=make_model('code_nonrecurrent',c)
        self.assertFalse(any(n.startswith(('fast.','slow.')) for n,p in nr.named_parameters()))
        x=torch.randint(17,(2,7));s=torch.randint(30,(2,8));y=torch.randint(17,(2,7))
        nr(x,s,y)[0].backward();self.assertIsNotNone(nr.code_out.weight.grad)

    def test_action_control_has_no_codec_and_legality_is_explicit(self):
        c=Config(width=24,heads=3,ff=32,state_length=8,ctx_layers=1)
        m=StateActionPolicy(c);s=torch.randint(30,(2,8))
        self.assertFalse(hasattr(m,'code'))
        legal=torch.zeros(2,7,dtype=torch.bool);legal[:,4]=True
        self.assertTrue((m.generate_actions(s,legal)==4).all())
        loss=m(s,torch.tensor([1,2]));loss.backward()
        self.assertTrue(torch.isfinite(m.head.weight.grad).all())


class MeasurementTests(unittest.TestCase):
    def test_cluster_estimand_uses_same_point_and_interval(self):
        values=[1.]*9+[0.];groups=['a']*9+['b']
        a=cluster_interval(values,groups,draws=100);b=cluster_interval(values,groups,estimand='group',draws=100)
        self.assertAlmostEqual(a['mean'],.9);self.assertAlmostEqual(b['mean'],.5)
        self.assertEqual(a['estimand'],'example_weighted')
        paired=paired_difference(values,values,groups,draws=100)
        self.assertEqual(paired['ci95'],[0.,0.])

    def test_required_length_not_evaluation_horizon(self):
        records=[dict(minimum_required_steps=2,steps_to_success=2,invalid=False,group='a'),
                 dict(minimum_required_steps=22,steps_to_success=None,invalid=False,group='b')]
        report=episode_report(records)
        self.assertEqual(report['20+']['episodes'],1)
        self.assertEqual(report['20+']['solved_by_horizon']['20']['mean'],0.)
        self.assertIsNone(report['10-19']['solved_by_horizon'])

    def test_variable_code_accuracy_is_separate(self):
        actions=np.array([[1,2,3],[1,2,4]])
        d=code_diagnostics(actions[:1],actions[1:],actions)
        self.assertEqual(d['variable_positions'],1);self.assertEqual(d['variable_position_accuracy'],0.)
        self.assertAlmostEqual(d['all_position_accuracy'],2/3)


class RepresentationTests(unittest.TestCase):
    def test_complete_state_roundtrip_and_wall_difference(self):
        a=dict(domain='grid',n=3,walls=[0]*9);b={**a,'walls':[0,1,0,0,0,0,0,0,0]}
        x=serialize_observation(a,0,8);y=serialize_observation(b,0,8)
        self.assertNotEqual(x,y);self.assertEqual(parse_observation(x),(a,0,8))
        with self.assertRaises(ValueError):parse_observation(x.replace('"goal":8','"goal":9'))

    def test_observed_block_overflow_and_information_gates(self):
        self.assertEqual(fixed_observed_blocks([1,2,3],block_size=2,blocks=2),[[1,2],[3,0]])
        with self.assertRaises(ValueError):fixed_observed_blocks(list(range(5)),block_size=2,blocks=2)
        with self.assertRaises(ValueError):require_representation_contract(context_source='codes',complete_observation=True,codec_roundtrip_passed=False)
        with self.assertRaises(ValueError):require_representation_contract(context_source='state',complete_observation=True,codec_roundtrip_passed=True,case_sensitive_domain=True,case_preserving=False)

    def test_hard_benchmark_quota_split_and_solver(self):
        eps,_=make_benchmark(per_stratum=1,seed=151804,max_maps=20000)
        self.assertEqual(len(eps),9);self.assertEqual(len({e['group'] for e in eps}),9)
        for e in eps:
            p=bfs_plan(e['env'],e['initial'],e['goal'])
            self.assertEqual(len(p['actions']),e['minimum_required_steps'])
            if e['length_bucket']=='20+':self.assertGreaterEqual(len(p['actions']),20)
            self.assertEqual(parse_observation(e['trajectory'][0]['text'])[1],e['initial'])


class FutureContractTests(unittest.TestCase):
    def test_horizon_value_policy_identity_and_zero_horizon(self):
        c=ValueContract('p1','r1','greedy_prior',20,3);m=HorizonValue(c,hidden=4)
        with torch.no_grad():
            for p in m.parameters():p.fill_(.2)
        x=torch.ones(2,3);v=m(x,torch.tensor([1,10]),contract_id=c.identity)
        self.assertNotEqual(float(v[0].detach()),float(v[1].detach()))
        self.assertEqual(float(m(x[:1],torch.tensor([0]),contract_id=c.identity).detach()),0.)
        with self.assertRaises(ValueError):m(x,torch.tensor([1,2]),contract_id='wrong')
        with self.assertRaises(ValueError):m.validate_label(dict(contract_id=c.identity,horizon_left=0,terminal_success=1))
        self.assertNotEqual(search_state_key(env_version='e',state=1,horizon_left=1,policy_version='p'),search_state_key(env_version='e',state=1,horizon_left=2,policy_version='p'))

    def test_fixed_adapter_capacity_promote_and_rollback(self):
        slots=AdapterSlots(GatedLoRA(3,2,rank=2),4096);memory=BoundedVerifiedMemory()
        initial=copy.deepcopy(slots.state_dict());bytes_before=slots.stored_bytes()
        c=slots.begin_candidate()
        with torch.no_grad():c.b.fill_(1)
        before=PromotionMetrics(.3,.8,.7,.5);after=PromotionMetrics(.5,.8,.7,.5)
        self.assertTrue(slots.decide(before,after,new_version='p2',thresholds=PromotionThresholds(),memory=memory))
        self.assertEqual(slots.stored_bytes(),bytes_before);self.assertEqual(len(slots.slots),2)
        slots.rollback(memory);self.assertEqual(slots.policy_version,'base')
        torch.testing.assert_close(slots.slots[slots.active].b,initial['slots.0.b'])
        slots.begin_candidate()
        self.assertFalse(slots.decide(before,PromotionMetrics(.6,.5,.4,.2),new_version='bad',thresholds=PromotionThresholds()))
        self.assertEqual(slots.policy_version,'base')

    def test_lora_initial_identity_and_bounded_verified_memory(self):
        layer=GatedLoRA(3,2);self.assertTrue((layer(torch.ones(2,3),torch.ones(2,1))==0).all())
        mem=BoundedVerifiedMemory(max_entries=2,max_bytes=300)
        for i in range(4):mem.write(str(i),dict(split='train',verified=True,policy_version='base',value=i))
        self.assertLessEqual(len(mem.entries),2);self.assertLessEqual(mem.bytes,300)
        with self.assertRaises(ValueError):mem.write('bad',dict(split='test',verified=True,policy_version='base'))
        mem.reset('new');self.assertEqual(mem.bytes,0)
        with self.assertRaises(ValueError):mem.write('stale',dict(split='train',verified=True,policy_version='base'))


class IsolationTests(unittest.TestCase):
    def test_output_guard_rejects_protected_paths_and_symlink(self):
        with self.assertRaises(ValueError):output_path(LEGACY/'model.py')
        self.assertEqual(output_path(ROOT/'reports/test.json'),ROOT/'reports/test.json')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=Path(temp)/'link';p.symlink_to(LEGACY,target_is_directory=True)
            with self.assertRaises(ValueError):output_path(p/'anything')

    def test_heavy_jobs_refuse_active_training(self):
        with patch('v15_astra_v3.isolation.active_legacy_processes',return_value=[123]):
            with self.assertRaises(RuntimeError):require_legacy_idle()


if __name__=='__main__':unittest.main(verbosity=2)
