import copy
import unittest

import torch

from v15_astra_ncm_v4.graphs import batch_states,distances,legal,oracle_path,step
from v15_astra_ncm_v4.model import Agent,AgentConfig,objective
from v15_astra_ncm_v4.search import mcts


def example():
    return {'id':'toy','n':5,'edges':[[0,1],[0,2],[1,3],[2,4]],'risk':[0,0,1,0,0],'goal':4,'start':0}


class GroundedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)

    def test_oracle_counterfactual_and_discount(self):
        g=example();d=distances(g)
        self.assertEqual(oracle_path(g,0),[0,2,4]);self.assertEqual(d[1],float('inf'))
        b=batch_states([(g,0,0),(g,4,0),(g,3,0)])
        self.assertAlmostEqual(float(b['value_target'][0]),.95**2,places=6)
        self.assertEqual(float(b['value_target'][1]),1.)
        self.assertEqual(float(b['value_target'][2]),0.)
        self.assertEqual(float(b['policy_target'][0,2]),1.)
        with self.assertRaises(ValueError):step(g,0,4)

    def test_goal_and_deadend_losses_finite(self):
        g=example();m=Agent()
        b=batch_states([(g,4,0),(g,3,0)])
        loss,_=objective(m(b),b)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_permutation_and_padding_invariance(self):
        g=example();mapping=[3,0,4,2,1]
        h=copy.deepcopy(g)
        h['edges']=[[mapping[a],mapping[b]] for a,b in g['edges']]
        h['goal']=mapping[g['goal']];h['start']=mapping[0]
        h['risk']=[0]*5
        for a,b in enumerate(mapping):h['risk'][b]=g['risk'][a]
        m=Agent(AgentConfig(strategies=True)).eval()
        # Nonzero refinement to exercise both iterations.
        torch.nn.init.normal_(m.cells[0].fc2.weight,std=.02)
        with torch.no_grad():
            a=m(batch_states([(g,0,1)]));b=m(batch_states([(h,h['start'],1)]))
        torch.testing.assert_close(a['logits'][0],b['logits'][0,mapping],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a['value'],b['value'])
        big=copy.deepcopy(g);big['n']=8;big['risk'] += [0,0,0]
        with torch.no_grad():bat=m(batch_states([(g,0,1),(big,0,1)]))
        torch.testing.assert_close(a['value'][0],bat['value'][0],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a['action_hidden'][0],bat['action_hidden'][0,:5],atol=1e-5,rtol=1e-5)

    def test_recurrence_identity_then_gradient(self):
        g=example();m=Agent();b=batch_states([(g,0,0)])
        a=m(b,steps=0);out=m(b,steps=2)
        torch.testing.assert_close(a['logits'],out['logits'])
        loss,_=objective(out,b);loss.backward()
        self.assertGreater(float(m.cells[0].fc2.weight.grad.norm()),0)
        self.assertEqual(len(m.cells),1)

    def test_mcts_legality_visits_and_success(self):
        g=example()
        def predict(g,node,z):
            actions=legal(g,node)
            return {a:1/len(actions) for a in actions},0.
        r=mcts(g,0,predict,simulations=16)
        self.assertEqual(r['root_visits'],16);self.assertEqual(r['action'],2)
        self.assertEqual(set(r['edges']),{1,2})
        self.assertGreater(r['edges'][2]['q'],0)
        self.assertLessEqual(r['edges'][2]['q'],.95**2+1e-8)

    def test_mcts_cycle_and_terminal(self):
        g=example();g['edges'] += [[1,0]]
        def predict(g,node,z):return {a:.5 for a in legal(g,node)},.9
        r=mcts(g,0,predict,simulations=8,max_depth=5)
        self.assertEqual(r['root_visits'],8)
        self.assertEqual(mcts(g,4,predict,simulations=8)['action'],None)

    def test_preferences_are_observed_and_oracle_verified(self):
        g={'id':'risk','n':5,'edges':[[0,1],[1,4],[0,2],[2,3],[3,4]],'risk':[0,1,0,0,0],'goal':4,'start':0}
        self.assertEqual(oracle_path(g,0,0),[0,1,4])
        self.assertEqual(oracle_path(g,0,1),[0,2,3,4])
        b=batch_states([(g,0,0),(g,0,1)])
        self.assertEqual(b['policy_target'][0].argmax(),1)
        self.assertEqual(b['policy_target'][1].argmax(),2)

    def test_labels_cannot_leak_to_forward(self):
        g=example();m=Agent().eval();b=batch_states([(g,0,0)])
        with torch.no_grad():a=m(b)
        b['policy_target'].uniform_();b['value_target'].uniform_();b['q_target'].uniform_()
        with torch.no_grad():c=m(b)
        torch.testing.assert_close(a['logits'],c['logits'],atol=0,rtol=0)

    def test_k2_prior_can_repeat_and_has_no_future_inputs(self):
        m=Agent(AgentConfig(strategies=True))
        samples=m.sample_strategies(1000,2,torch.Generator().manual_seed(5))
        self.assertEqual(samples.shape,(1000,2))
        rate=float((samples[:,0]==samples[:,1]).float().mean())
        self.assertTrue(.4<rate<.6)

    def test_dataset_balanced_disjoint_and_immutable(self):
        from v15_astra_ncm_v4.graphs import load_dataset
        data=load_dataset()['graphs']
        self.assertEqual(len({g['id'] for g in data}),len(data))
        keys=[]
        for g in data:
            keys.append((g['n'],tuple(tuple(e) for e in sorted(g['edges']))))
            self.assertEqual(distances(g)[g['start']],g['start_shortest_steps'])
        self.assertEqual(len(set(keys)),len(keys))
        for split in ('train','valid','test','ood'):
            counts=[sum(g['split']==split and g['difficulty_bin']==i for g in data) for i in range(3)]
            self.assertLessEqual(max(counts)-min(counts),1)


if __name__=='__main__':unittest.main()
