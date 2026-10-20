"""Paired single/multiple-plan panels with conservative grid route classes.

Only shortest paths are enumerated, with a cap; absence of a second retained
route is not proof of uniqueness. A distinct class must take internally
disjoint branches around at least one wall, not just commute two open moves.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .tasks import step,distances,optimal_actions,observed
from .representation import serialize_observation
from .dataset import Dataset
from .isolation import output_path,atomic_json,sha256


def enumerate_shortest(env,start,goal,max_paths=32):
    if max_paths<2:raise ValueError('At least two paths must be considered')
    dist=distances(env,goal);stack=[(start,[],[start])];plans=[]
    while stack and len(plans)<max_paths:
        s,actions,states=stack.pop()
        if s==goal:
            plans.append(dict(actions=actions,states=states,total_cost=len(actions)));continue
        for a in reversed(optimal_actions(env,s,dist)):
            nxt=step(env,s,a);stack.append((nxt,actions+[a],states+[nxt]))
    return plans,bool(stack)


def verify_plan(env,start,goal,plan):
    s=start;states=[s]
    for a in plan['actions']:
        s=step(env,s,a)
        if s is None:raise ValueError('Illegal teacher action')
        states.append(s)
    if s!=goal or states!=plan['states'] or len(states)!=len(set(states)):
        raise ValueError('Teacher plan fails replay or contains cycles')
    if len(plan['actions'])!=plan['total_cost']:raise ValueError('Wrong path cost')
    return True


def encloses_wall(env,path_a,path_b):
    """Find a divergence/rejoin whose simple grid polygon contains a wall."""
    if env['domain']!='grid':return False
    n=env['n'];locations={s:i for i,s in enumerate(path_b)}
    common=[(i,locations[s]) for i,s in enumerate(path_a) if s in locations]
    for (i,j),(ii,jj) in zip(common,common[1:]):
        if min(ii-i,jj-j)<3:continue
        polygon=path_a[i:ii+1]+path_b[j:jj][::-1]
        points=[(s%n,s//n) for s in polygon]
        for wall,blocked in enumerate(env['walls']):
            if not blocked:continue
            x,y=wall%n,wall//n;inside=False
            for (x1,y1),(x2,y2) in zip(points,points[1:]+points[:1]):
                if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1:inside=not inside
            if inside:return True
    return False


def paired_panels(episodes,max_paths=32):
    problems=[];skipped=0;capped=0
    for e in episodes:
        plans,truncated=enumerate_shortest(e['env'],e['initial'],e['goal'],max_paths)
        capped+=truncated;selected=None
        for i,a in enumerate(plans):
            for b in plans[i+1:]:
                if encloses_wall(e['env'],a['states'],b['states']):selected=[a,b];break
            if selected:break
        if selected is None:skipped+=1;continue
        solutions=[]
        for i,p in enumerate(selected):
            verify_plan(e['env'],e['initial'],e['goal'],p)
            solutions.append(dict(**p,solution_id=f'route_{i}',strategy_class_eval_only=f'obstacle_route_{i}',verified=True))
        problems.append(dict(problem_id=e['episode'],group=e['group'],split=e['split'],env=e['env'],
            initial=e['initial'],goal=e['goal'],horizon_budget=40,eval_only_optimal_distance=e['minimum_required_steps'],
            solutions=solutions,single_solution=0,enumeration_capped=truncated))
    return problems,dict(problems=len(problems),skipped_no_certified_second_route=skipped,enumeration_capped=capped,
                         splits=dict(Counter(p['split'] for p in problems)),
                         class_definition='two simple shortest paths with disjoint divergence branches enclosing a wall',
                         coverage='at most 32 lexicographic shortest paths; not exhaustive class counting')


class PlanSampler:
    """Uniform problem -> uniform retained plan -> uniform teacher step.

    Both panels use identical problem IDs and update counts. Multiple plans
    replace references, not multiply optimizer updates. Labels stay separate
    from actor inputs, which contain current observed state and past actions.
    """
    def __init__(self,problems,panel='single',split='train'):
        if panel not in ('single','multi'):raise ValueError('Unknown supervision panel')
        self.problems=[p for p in problems if p['split']==split];self.panel=panel
        if not self.problems:raise ValueError('No problems in split')

    def sample(self,rng):
        p=self.problems[int(rng.integers(len(self.problems)))];plans=p['solutions']
        plan=plans[p['single_solution'] if self.panel=='single' else int(rng.integers(len(plans)))]
        t=int(rng.integers(len(plan['actions'])));s=plan['states'][t]
        inputs=dict(state=observed(p['env'],s,p['goal'])[1],
                    text=serialize_observation(p['env'],s,p['goal']),history=plan['actions'][:t],
                    horizon_left=p['horizon_budget']-t)
        labels=dict(action=plan['actions'][t],next_state=plan['states'][t+1],
                    strategy_class=plan['strategy_class_eval_only'],problem_id=p['problem_id'])
        return inputs,labels


def build_panels(source,out):
    out=output_path(out)
    if out.exists():raise FileExistsError('Multi-plan panels are immutable')
    data=Dataset(source);problems,audit=paired_panels(data.episodes)
    out.mkdir(parents=True);p=out/'problems.json';p.write_text(json.dumps(problems))
    manifest=dict(source_dataset_sha=data.manifest['episodes_sha'],problems_sha=sha256(p),**audit,
                  training_launched=False,stage='S0 data plumbing only; P1 has not run',
                  panels=dict(single='one fixed plan per problem',multi='uniform among two certified routes'))
    atomic_json(out/'manifest.json',manifest);return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(build_panels(args.source,args.out),indent=2))
