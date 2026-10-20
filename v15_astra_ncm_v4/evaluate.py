"""Held-out grounded metrics, separate from language and implementation health."""
import random
import time

import torch

from .graphs import GAMMA, batch_states, cost, distances, legal, oracle_path, step
from .search import mcts


@torch.no_grad()
def predictor(model,device):
    def predict(g,node,strategy):
        b = batch_states([(g,node,strategy)],device,labels=False)
        out = model(b)
        p = out['logits'].softmax(-1)[0].cpu().tolist()
        return {a:p[a] for a in legal(g,node)},float(out['value'][0])
    return predict


def rollout(g,predict,strategy=0,mode='greedy',simulations=8,seed=0):
    rng = random.Random(seed)
    path = [g['start']]
    calls = env_calls = 0
    total_cost = 0
    started = time.perf_counter()
    for _ in range(2*g['n']):
        current = path[-1]
        if current==g['goal']: break
        actions = legal(g,current)
        if not actions: break
        if mode=='random':
            action = rng.choice(actions)
        elif mode=='mcts':
            result = mcts(g,current,predict,strategy,simulations)
            action = result['action'];calls += result['model_calls'];env_calls += result['environment_steps']
        else:
            priors,_ = predict(g,current,strategy);calls += 1
            action = max(priors,key=lambda a:(priors[a],-a))
        next_node,_ = step(g,current,action);env_calls += 1
        total_cost += cost(g,action,strategy)
        path.append(next_node)
        if next_node in path[:-1]: break
    success = path[-1]==g['goal']
    optimum = distances(g,strategy)[g['start']]
    return {'graph_id':g['id'],'path':path,'success':success,'cost':total_cost,
            'optimal_cost':optimum,'stretch':total_cost/optimum if success else None,
            'return':GAMMA**total_cost if success else 0.,'model_calls':calls,
            'environment_steps':env_calls,'seconds':time.perf_counter()-started}


@torch.no_grad()
def evaluate(model,graphs,device='cpu',strategy=0,search=False):
    was_training = model.training;model.eval()
    items = [(g,c,strategy) for g in graphs for c in range(g['n'])]
    mass = ce = mse = q_mse = correct = trans = active_count = edge_count = 0.
    for start in range(0,len(items),64):
        b = batch_states(items[start:start+64],device)
        out = model(b)
        active = b['legal'].any(-1)
        p = out['logits'].softmax(-1)
        optimal = b['policy_target']>0
        mass += float((p*optimal).sum(-1)[active].sum())
        ce += float((-(b['policy_target']*out['logits'].log_softmax(-1)).sum(-1))[active].sum())
        top = out['logits'].argmax(-1)
        correct += float(optimal.gather(1,top[:,None]).squeeze(-1)[active].sum())
        mse += float(((out['value']-b['value_target'])**2).sum())
        q_mse += float(((out['q']-b['q_target'])**2)[b['legal']].sum())
        target = torch.arange(b['legal'].shape[1],device=device)[None,:]
        trans += float((out['transition'].argmax(-1)==target)[b['legal']].sum())
        active_count += int(active.sum());edge_count += int(b['legal'].sum())
    predict = predictor(model,device)
    modes = ['random','greedy'] + (['mcts8','mcts16'] if search else [])
    results = {}
    for mode in modes:
        rows = [rollout(g,predict,strategy,'mcts' if mode.startswith('mcts') else mode,
                        int(mode[4:]) if mode.startswith('mcts') else 8,seed=i+1987)
                for i,g in enumerate(graphs)]
        successes = sum(r['success'] for r in rows)
        p = successes/max(1,len(rows));n = len(rows)
        denom = 1+3.8416/n
        center = (p+3.8416/(2*n))/denom
        half = 1.96*((p*(1-p)/n+3.8416/(4*n*n))**.5)/denom
        results[mode] = {'success_rate':p,'success_wilson95':[center-half,center+half],
            'mean_return':sum(r['return'] for r in rows)/n,'seconds':sum(r['seconds'] for r in rows),
            'model_calls':sum(r['model_calls'] for r in rows),'environment_steps':sum(r['environment_steps'] for r in rows),
            'successful_mean_stretch':sum(r['stretch'] for r in rows if r['success'])/max(1,successes),
            'episodes':rows}
    model.train(was_training)
    return {'graph_count':len(graphs),'state_count':len(items),'strategy':strategy,
            'oracle_action_mass':mass/max(1,active_count),'policy_ce':ce/max(1,active_count),
            'optimal_top1':correct/max(1,active_count),'discounted_value_mse':mse/len(items),
            'q_mse':q_mse/max(1,edge_count),'successor_copy_accuracy':trans/max(1,edge_count),
            'rollouts':results,'oracle':'Exact Dijkstra reaches every selected start; not a neural result'}
