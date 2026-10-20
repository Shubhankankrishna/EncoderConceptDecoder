"""One final held-out report after validation-only checkpoint selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .common import ROOT,atomic_json,heavy_job,seed_all,sha
from .evaluate import evaluate,rollout,predictor
from .graphs import load_dataset
from .model import Agent,AgentConfig,PlanConditioner


def main(args):
    torch.set_num_threads(2);seed_all(1919)
    data=load_dataset()['graphs'];result={'timestamp':time.time(),'models':{},'renderer':None}
    for arm in sorted(args.run.iterdir()):
        path=arm/'best.pt'
        if not path.exists() or arm.name=='renderer':continue
        if time.time()>args.deadline-60:break
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=Agent(AgentConfig(**saved['config'])).to('mps');model.load_state_dict(saved['model']);model.eval()
        reports={}
        for split in ('test','ood'):
            graphs=[g for g in data if g['split']==split]
            for z in ([0,1] if model.config.strategies else [0]):
                if time.time()>args.deadline-60:break
                reports[f'{split}_strategy{z}']=evaluate(model,graphs,'mps',z,search=arm.name.startswith(('r2_','strategy2_')))
        if model.config.strategies:
            pred=predictor(model,'mps')
            graphs=[g for g in data if g['split']=='test']
            paired=[(rollout(g,pred,0),rollout(g,pred,1)) for g in graphs]
            both=[(a,b) for a,b in paired if a['success'] and b['success']]
            reports['strategy_diversity']={'both_successful':len(both),'total':len(paired),
                'distinct_successful_paths':sum(a['path']!=b['path'] for a,b in both),
                'semantics':'Different explicit cost preferences; not latent strategy discovery'}
            generator=torch.Generator().manual_seed(742)
            draws=model.sample_strategies(len(graphs),2,generator)
            candidates=[]
            for g,zs in zip(graphs,draws.tolist()):
                attempts=[rollout(g,pred,z) for z in zs]
                successful=[a for a in attempts if a['success']]
                # Compare candidates under one stated objective: unit edge count.
                best=min(successful,key=lambda a:len(a['path'])) if successful else None
                candidates.append({'graph_id':g['id'],'z_draws':zs,'success':bool(best),
                    'selected_path':best['path'] if best else None,'model_calls':sum(a['model_calls'] for a in attempts),
                    'seconds':sum(a['seconds'] for a in attempts)})
            reports['independent_k2']={'episodes':candidates,'success_rate':sum(x['success'] for x in candidates)/len(candidates),
                'selection':'Choose verified successful candidate with fewest edges; draws may coincide; twice the rollout cost is reported'}
        result['models'][arm.name]={'selected_step':saved['step'],'checkpoint_sha':sha(path),'evaluation':reports}
        atomic_json(args.run/'final_results.partial.json',result)
        del model
    path=args.run/'renderer/best.pt'
    if path.exists() and time.time()<args.deadline-180:
        from .ncm import load_ncm
        from .cache import load_cache
        from .adapt import evaluate_renderer
        net,codec,inference=load_ncm('mps');encoder=PlanConditioner().to('mps')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        net.load_adapters(saved['adapters']);encoder.load_state_dict(saved['conditioner'])
        cache=load_cache()
        result['renderer']={split:evaluate_renderer(net,encoder,cache,inference,saved['step'],split,24) for split in ('test','ood')}
        from .pipeline import load_agent,solve_and_render
        # Select the planning checkpoint by VALIDATION performance only.
        candidates=[(float(torch.load(p,map_location='cpu',weights_only=True)['score']),p)
                    for p in args.run.glob('r2_seed*/best.pt')]
        if candidates:
            selected=max(candidates,key=lambda x:x[0])[1]
            agent,agent_step=load_agent(selected)
            joint=[]
            for g in [g for g in data if g['split']=='test'][:8]:
                if time.time()>args.deadline-60:break
                joint.append(solve_and_render(g,agent,net,encoder,inference,search=False))
            result['joint_pipeline']={'planner_checkpoint':str(selected),'planner_step':agent_step,
                'renderer_step':saved['step'],'cases':joint,'oracle_repair':False,
                'meaning':'Small held-out end-to-end probe; structured plan remains canonical'}
    result['completed']=time.time()
    expected=[s['name'] for s in json.loads((args.run/'plan.json').read_text())['stages']]
    result['all_structured_models_evaluated']=set(result['models'])==set(expected) and all(
        'test_strategy0' in r['evaluation'] and 'ood_strategy0' in r['evaluation'] for r in result['models'].values())
    result['limitations']=['Structured planning and language rendering are separate tasks.',
        'Search comparisons report actual costs; no equal-cost superiority claim is made.',
        'Two training seeds and small renderer panels are exploratory.',
        'Successor-copy accuracy and hard-mask legality are not reasoning evidence.',
        'Ordinary-text retention uses local FineWeb, not official LM1B benchmark data.']
    atomic_json(args.run/'final_results.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--deadline',type=float,required=True)
    args=ap.parse_args()
    with heavy_job():main(args)
