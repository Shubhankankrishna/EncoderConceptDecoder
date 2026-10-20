import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..isolation import ROOT,sha256
from ..benchmark import make_benchmark
from ..audit import validate_episodes
from ..dataset import Dataset,encode_dataset
from ..raw import raw_windows,build_raw,RawCodeDataset
from ..multiplan import enumerate_shortest,verify_plan,encloses_wall,PlanSampler
from ..grounding import GroundedActor,StrategyRouter,CandidateCritic,candidate_outcome
from ..gram import AstraGRAM
from ..value import ValueContract
from ..objectives import episode_mode_elbo,joint_mode_elbo
from ..diagnostics import prior_sensitivity
from ..continual import AdapterSlots,GatedLoRA,PromotionMetrics,PromotionThresholds
from ..controls import make_model
from ..evaluate import evaluate
from ..evaluate_raw import evaluate_raw
from ..codec import FrozenNCMCodec
from .. import train as runner
from .test_gram_regression import small


class IntegrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.random.default_generator.manual_seed(71)

    def fixture(self,directory):
        eps,_=make_benchmark(per_stratum=1,seed=151804)
        d=directory/'data';d.mkdir();p=d/'episodes.jsonl'
        p.write_text(''.join(json.dumps(e)+'\n' for e in eps))
        (d/'manifest.json').write_text(json.dumps(dict(episodes_sha=sha256(p))))
        return d

    def test_crash_resume_matches_continuous_training_exactly(self):
        # Four tiny CPU updates on a disposable fixture, never a real run.
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d=Path(temp);data=self.fixture(d);config=d/'tiny.json'
            config.write_text(json.dumps(dict(width=24,heads=3,ff=32,ctx_layers=1)))
            def args(name):
                return runner.parser().parse_args(['--run',str(d/name),'--data',str(data),'--arm','state_action',
                    '--steps','4','--seconds','60','--batch','2','--accum','1','--device','cpu',
                    '--config',str(config),'--save-every','0.000001'])
            save=runner.save
            def fail_after_saved_step_two(*a,**kw):
                save(*a,**kw)
                if a[3]==2:raise RuntimeError('simulated process failure after durable checkpoint')
            with patch('v15_astra_v3.isolation.active_legacy_processes',return_value=[]):
                runner.train(args('continuous'))
                with patch.object(runner,'save',side_effect=fail_after_saved_step_two):
                    with self.assertRaisesRegex(RuntimeError,'simulated'):runner.train(args('resumed'))
                runner.train(args('resumed'))
                changed=args('resumed');changed.seed+=1
                with self.assertRaisesRegex(ValueError,'registered'):runner.train(changed)
            a=torch.load(d/'continuous/latest.pt',weights_only=True);b=torch.load(d/'resumed/latest.pt',weights_only=True)
            self.assertEqual(a['step'],4);self.assertEqual(b['step'],4)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            self.assertEqual(a['numpy_rng'],b['numpy_rng']);torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)
            m=make_model('state_action',b['config']);m.load_state_dict(b['model']);m.train()
            rng=torch.get_rng_state().clone();report=evaluate(m,Dataset(data),n=12,closed_loop=False)
            self.assertTrue(m.training);torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(len(report['decisions']),12)
            self.assertAlmostEqual(report['model_optimal']['mean'],np.mean([r['optimal'] for r in report['decisions']]))

    def test_all_heavy_entrypoints_stop_before_read_or_write(self):
        with patch('v15_astra_v3.isolation.active_legacy_processes',return_value=[123]):
            with self.assertRaises(RuntimeError):runner.train(None)
            with self.assertRaises(RuntimeError):encode_dataset('/does/not/exist')
            with self.assertRaises(RuntimeError):build_raw('/does/not/exist',ROOT/'should_not_exist')
            with self.assertRaises(RuntimeError):FrozenNCMCodec()
        self.assertFalse((ROOT/'should_not_exist').exists())

    def test_auditor_rejects_wrong_observations_and_labels(self):
        eps,_=make_benchmark(per_stratum=1,seed=151804)
        bad=copy.deepcopy(eps);bad[0]['trajectory'][0]['text']='walls omitted'
        with self.assertRaises(ValueError):validate_episodes(bad)
        bad=copy.deepcopy(eps);bad[0]['trajectory'][0]['remaining']+=1
        with self.assertRaises(ValueError):validate_episodes(bad)
        with self.assertRaises(ValueError):validate_episodes(eps,[eps[0]['group']])

    def test_raw_documents_causal_dedup_and_label_firewall(self):
        def doc(i,text,family=None,cluster=None):return dict(doc_id=str(i),family=family or str(i),duplicate_cluster=cluster or str(i),license='fixture-only',source='synthetic unit fixture',text=text,case_sensitive=False)
        docs=[doc(1,'abcdefghijklmno'),doc(2,'abcdefghijklmno'),doc(3,'pqrstuvwxyzabcd')]
        r=raw_windows(docs,lambda t:[ord(c) for c in t],history=2,block_size=4)
        self.assertEqual(r['dropped_exact_duplicates'],1)
        for e in r['examples']:
            d=next(x for x in r['documents'] if x['doc_id']==e['doc_id'])
            sequence=e['context']+[e['target']]
            self.assertTrue(any(sequence==d['block_ids'][i:i+3] for i in range(len(d['block_ids'])-2)))
            self.assertFalse({'action','reward','state'}&set(e))
        with self.assertRaises(ValueError):raw_windows([{**docs[0],'action':1}],lambda s:[1])
        with self.assertRaises(ValueError):raw_windows([{**docs[0],'case_sensitive':True}],lambda s:[1])
        # Near duplicates assigned the same declared cluster share a family split.
        r=raw_windows([doc(1,'abcdefghi',cluster='same'),doc(2,'xyzabcdef',cluster='same')],lambda s:list(s.encode()),block_size=2)
        self.assertEqual(r['documents'][0]['group'],r['documents'][1]['group'])

    def test_raw_gram_backward_and_prior_target_independence(self):
        c=small();c.context_source='codes';m=AstraGRAM(c)
        x=torch.randint(17,(2,2,7));y=torch.randint(17,(2,7));noise=torch.randn(2,2,3,24)
        loss,_=runner.model_loss(m,dict(x=x,y=y));loss.backward()
        self.assertIsNotNone(m.prior.out.weight.grad)
        prior=m.generate(x,None,noise=noise)
        # Training-only posterior deletion leaves the exact fixed-noise inference intact.
        inference=AstraGRAM(c,include_posterior=False);inference.load_state_dict(m.inference_state_dict(),strict=True)
        self.assertTrue(torch.equal(prior,inference.generate(x,None,noise=noise)))
        d=prior_sensitivity(m,x,None,y,noise)
        self.assertGreater(d['prior_state_rms_difference'],0)
        self.assertEqual(len(d['conditional_logit_rms_by_level']),3)

    def test_raw_cache_loader_and_free_generation_lane(self):
        from ..tasks import group_split
        from ..model import Config
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d=Path(temp);group=next('unit:'+str(i) for i in range(1000) if group_split('unit:'+str(i))=='valid')
            np.save(d/'codes.npy',np.random.default_rng(2).integers(0,17,(3,641),dtype=np.uint16))
            (d/'documents.json').write_text(json.dumps([dict(doc_id='fixture',group=group,split='valid',block_ids=[0,1,2])]))
            (d/'examples.json').write_text(json.dumps([dict(doc_id='fixture',group=group,split='valid',context=[0,1],target=2)]))
            manifest=dict(kind='raw_concept_windows_v3',history=2,unique_blocks=3,codec={'codec_id':'unit_fixture_only'},
                files={name:sha256(d/name) for name in ('codes.npy','documents.json','examples.json')})
            manifest['episodes_sha']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            (d/'manifest.json').write_text(json.dumps(manifest));data=RawCodeDataset(d)
            self.assertEqual(set(data.batch([0])),{'x','y'});self.assertEqual(data.batch([0])['x'].shape,(1,2,641))
            class FixtureCodec:
                def assert_matches(self,identity):
                    if identity['codec_id']!='unit_fixture_only':raise ValueError('fixture mismatch')
                def decode_text(self,codes):return ['synthetic decoder fixture']*len(codes)
            c=dict(width=24,heads=3,code_dim=8,vocab=17,ff=32,slots=3,ctx_layers=1,head_layers=1,context_source='codes')
            m=make_model('code_hrm',c);m.train();rng=torch.get_rng_state().clone()
            result=evaluate_raw(m,data,FixtureCodec(),indices=[0],n=1)
            self.assertEqual(result['records'][0]['free_generated_text'],'synthetic decoder fixture')
            self.assertEqual(result['indices'],[0]);self.assertTrue(m.training)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            (d/'examples.json').write_text('[]')
            with self.assertRaises(ValueError):RawCodeDataset(d)

    def test_verified_routes_are_more_than_commuting_moves(self):
        env=dict(domain='grid',n=3,walls=[0,0,0,0,1,0,0,0,0])
        plans,capped=enumerate_shortest(env,3,5)
        self.assertFalse(capped);self.assertEqual(len(plans),2)
        for p in plans:verify_plan(env,3,5,p)
        self.assertTrue(encloses_wall(env,plans[0]['states'],plans[1]['states']))
        open_grid={**env,'walls':[0]*9};p,_=enumerate_shortest(open_grid,0,4)
        self.assertFalse(encloses_wall(open_grid,p[0]['states'],p[1]['states']))
        bad=copy.deepcopy(plans[0]);bad['states'][1]=4
        with self.assertRaises(ValueError):verify_plan(env,3,5,bad)

    def test_actor_critic_firewall_strategy_persistence_and_gradients(self):
        c=small();base=AstraGRAM(c);actor=GroundedActor(base,with_history=True,action_width=8)
        x=torch.randint(17,(2,7));state=torch.randint(30,(2,8));history=torch.tensor([[1,7],[2,3]])
        noise=torch.randn(2,2,3,24);router=StrategyRouter(24)
        strategy,_=router.begin(base.encode_context(x,state),('e1','e2'),'p1',mode=[0,1])
        original=strategy.tokens.detach().clone();controls=dict(strategy=strategy,episode_ids=('e1','e2'),policy_version='p1')
        a=actor(x,state,history,noise=noise,**controls)
        with self.assertRaises(TypeError):actor(x,state,history,target_action=torch.tensor([0,1]),**controls)
        changed=history.clone();changed[0,0]=2
        b=actor(x,state,changed,noise=noise,**controls)
        self.assertGreater(float((a-b).abs().max().detach()),0)
        torch.testing.assert_close(strategy.tokens,original,rtol=0,atol=0)
        branch=strategy.branch();branch.tokens.data.add_(1)
        torch.testing.assert_close(strategy.tokens,original,rtol=0,atol=0)
        with self.assertRaises(ValueError):strategy.validate(('wrong','e2'),'p1')
        y=torch.randint(17,(2,7));loss,stats=actor.supervised_loss(x,state,torch.tensor([0,1]),target_codes=y,history=history,noise=noise,**controls)
        loss.backward();self.assertIsNotNone(router.bank.grad)
        self.assertEqual(stats['objective'],'single_step_action_terminal_ELBO')

    def test_candidate_critic_both_outcomes_policy_and_horizon(self):
        contract=ValueContract('p1','r1','fixed_greedy',10,4);critic=CandidateCritic(contract,action_width=3,hidden=8)
        f=torch.ones(1,4,requires_grad=True);a=torch.tensor([[0,1]])
        loss=critic.loss(f,a,torch.tensor([3]),torch.tensor([[1,0]]),contract_id=contract.identity)
        loss.backward();self.assertIsNone(f.grad);self.assertIsNotNone(critic.action_encoder.table.weight.grad)
        with self.assertRaises(ValueError):critic(f,a,[0],contract_id=contract.identity)
        with self.assertRaises(ValueError):critic(f,a,[3],contract_id='stale')
        env=dict(domain='grid',n=3,walls=[0]*9)
        good=candidate_outcome(env,0,1,4,1,lambda *args:None,contract_id=contract.identity,episode_group='g',split='train')
        self.assertEqual(good['transition_calls'],1)
        self.assertEqual(good['terminal_success'],1)
        bad=candidate_outcome(env,0,1,3,1,lambda *args:None,contract_id=contract.identity,episode_group='g',split='train')
        self.assertEqual(bad['terminal_success'],0);self.assertFalse(bad['legal'])

    def test_episode_mode_pays_one_router_kl(self):
        b=torch.tensor([[[-1.,-2.],[-3.,-1.],[999.,999.]]]);mask=torch.tensor([[True,True,False]])
        router=torch.tensor([[.3,-.2]]);q=torch.tensor([[1.,-1.]])
        actual=episode_mode_elbo(b,mask,router,q)
        expected=joint_mode_elbo(torch.tensor([[-4.,-3.]]),router,q)
        torch.testing.assert_close(actual,expected)
        per_step=sum(joint_mode_elbo(b[:,t],router,q) for t in (0,1))
        self.assertGreater(float(actual),float(per_step))

    def test_adapter_checkpoint_restores_open_candidate_and_rollback(self):
        slots=AdapterSlots(GatedLoRA(3,2),4096);slots.begin_candidate()
        restored=AdapterSlots(GatedLoRA(3,2),4096);restored.load_state_dict(slots.state_dict())
        self.assertTrue(restored.candidate_open)
        self.assertTrue(all(p.requires_grad for p in restored.slots[1].parameters()))
        before=PromotionMetrics(.3,.8,.7,.5);after=PromotionMetrics(.5,.8,.7,.5)
        restored.decide(before,after,new_version='p2',thresholds=PromotionThresholds())
        slots.load_state_dict(restored.state_dict());slots.rollback();self.assertEqual(slots.policy_version,'base')


if __name__=='__main__':unittest.main(verbosity=2)
