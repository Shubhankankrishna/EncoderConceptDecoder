import copy
import io
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .codec import area_matrix
from .model import AstraHRM,Config
from .tasks import ACTION_TEXTS,parse_action,distances,optimal_actions,step,make_episodes


def tiny():
    return Config(width=24,code_dim=8,vocab=17,levels=(1,2,4),heads=3,ff=32,
                  ctx_layers=1,head_layers=2,slots=3,state_length=8)


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(37)
        self.m=AstraHRM(tiny()).eval();self.x=torch.randint(17,(2,7));self.y=torch.randint(17,(2,7));self.s=torch.randint(20,(2,8))

    def test_current_and_future_labels_are_invisible(self):
        m=self.m;mem=m.encode_context(self.x,self.s);h,_=m.reason(mem)
        for ell in range(3):
            y2=self.y.clone();y2[:,m.offsets[ell]:]=(y2[:,m.offsets[ell]:]+1)%17
            a=m.decode_hidden(mem,h,self.y,ell);b=m.decode_hidden(mem,h,y2,ell)
            torch.testing.assert_close(a,b,atol=1e-7,rtol=0)

    def test_previous_labels_affect_predictions(self):
        m=self.m;mem=m.encode_context(self.x,self.s);h,_=m.reason(mem)
        y2=self.y.clone();y2[:,:3]=(y2[:,:3]+1)%17
        self.assertGreater(float((m.decode_hidden(mem,h,self.y,2)-m.decode_hidden(mem,h,y2,2)).abs().max()),1e-7)

    def test_parallel_and_levelwise_heads_agree(self):
        m=self.m;mem=m.encode_context(self.x,self.s);h,_=m.reason(mem);allh=m.decode_hidden(mem,h,self.y)
        for i in range(3):
            a,b=m.offsets[i:i+2]
            torch.testing.assert_close(allh[:,a:b],m.decode_hidden(mem,h,self.y,i),atol=1e-6,rtol=1e-5)

    def test_probabilities_and_nll_units(self):
        m=self.m;mem=m.encode_context(self.x,self.s);h,_=m.reason(mem);z=m.decode_hidden(mem,h,self.y)
        total=0.
        for i in range(3):
            a,b=m.offsets[i:i+2];logits=m.logits(z[:,a:b],i)
            torch.testing.assert_close(logits.softmax(-1).sum(-1),torch.ones(2,b-a))
            total+=F.cross_entropy(logits.reshape(-1,17),self.y[:,a:b].reshape(-1),reduction='sum')
        loss,info=m(self.x,self.s,self.y)
        torch.testing.assert_close(loss,total/14)
        self.assertEqual(m.generate(self.x,self.s).shape,(2,7))

    def test_recurrent_gradients_reach_first_transition(self):
        m=self.m;mem=m.encode_context(self.x,self.s);h,trace=m.reason(mem,True)
        loss=m.decode_hidden(mem,h,self.y).square().sum();loss.backward()
        self.assertIsNotNone(trace[0].grad);self.assertGreater(float(trace[0].grad.norm()),0)
        for mod in (m.fast,m.slow,m.context):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters()))

    def test_generation_has_no_target_argument_and_is_deterministic(self):
        a=self.m.generate(self.x,self.s);torch.manual_seed(9);self.y.random_(17)
        b=self.m.generate(self.x,self.s);self.assertTrue(torch.equal(a,b))

    def test_exact_save_resume_optimizer_update(self):
        m=self.m;opt=torch.optim.AdamW(m.parameters(),lr=.002)
        def update(model,optimizer):
            optimizer.zero_grad();loss,_=model(self.x,self.s,self.y);loss.backward();optimizer.step()
        update(m,opt);buf=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),buf)
        update(m,opt);buf.seek(0);ckpt=torch.load(buf,weights_only=True)
        r=AstraHRM(tiny()).eval();r.load_state_dict(ckpt['model']);o=torch.optim.AdamW(r.parameters(),lr=.002);o.load_state_dict(ckpt['optimizer']);update(r,o)
        for a,b in zip(m.parameters(),r.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_production_parameter_budget(self):
        m=AstraHRM();self.assertEqual(m.parameter_count(),10001292)
        self.assertEqual(m.offsets[-1],641)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_area_pool_equivalence(self):
        x=torch.randn(2,128,24)
        for n in Config().levels:
            ref=F.interpolate(x.transpose(1,2),size=n,mode='area').transpose(1,2)
            torch.testing.assert_close(area_matrix(128,n)@x,ref,atol=1e-6,rtol=1e-5)

    def test_task_verifier_and_action_boundary(self):
        for i,t in enumerate(ACTION_TEXTS):
            self.assertEqual(parse_action(t),i)
            self.assertEqual(parse_action(t+' arbitrary padding.'),i)
        for t in ('add','ignore this. The answer is to add.','The answer is to add or subtract.'):
            self.assertIsNone(parse_action(t))
        e=dict(domain='mod',m=7,a=1,b=2);d=distances(e,0)
        self.assertEqual(step(e,6,0),0);self.assertIsNone(step(e,6,5))
        for s in range(1,7):
            for a in optimal_actions(e,s,d):self.assertEqual(d[step(e,s,a)],d[s]-1)


if __name__=='__main__':unittest.main(verbosity=2)
