"""Generate a new, group-disjoint hard benchmark without touching legacy data."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .tasks import group_split,distances,optimal_actions,step,observed
from .representation import serialize_observation,parse_observation,SCHEMA
from .baselines import distance_heuristic,heuristic_rollout,bfs_plan
from .metrics import length_bucket
from .isolation import ROOT,LEGACY,output_path,atomic_json,sha256


def make_benchmark(per_stratum=8,seed=151803,max_maps=20000,excluded_groups=()):
    """One episode per map, quotas per split and minimum required path length.

    Random walls plus a minority of serpentine corridors provide real 20+
    action paths. Corridor family is recorded separately: length is not an
    assertion of branching complexity. All returned transitions are exact.
    """
    if per_stratum<1:raise ValueError('Positive quota required')
    rng=np.random.default_rng(seed);counts=Counter();result=[];seen=set(excluded_groups)
    bins=('5-9','10-19','20+');splits=('train','valid','test')
    for attempt in range(max_maps):
        n=9;walls=(rng.random(n*n)<rng.uniform(.23,.38)).astype(int).tolist();recipe='random_walls'
        if attempt%7==0:
            recipe='serpentine_corridor';walls=[1]*(n*n)
            for row in range(0,n,2):
                for x in range(n):walls[row*n+x]=0
                if row+1<n:walls[(row+1)*n+(n-1 if (row//2)%2==0 else 0)]=0
            # Distinct corridor topologies, while preserving a long subpath.
            for k in rng.choice([r*n+x for r in range(1,n,2) for x in range(n) if walls[r*n+x]],size=int(rng.integers(0,3)),replace=False):walls[int(k)]=0
        group='grid:'+hashlib.sha256(json.dumps([n,walls]).encode()).hexdigest()
        if group in seen:continue
        seen.add(group);split=group_split(group)
        available=[b for b in bins if counts[(split,b)]<per_stratum]
        if not available:continue
        env=dict(domain='grid',n=n,walls=walls,group=group)
        cells=[s for s,w in enumerate(walls) if not w]
        if len(cells)<2:continue
        for goal in rng.choice(cells,min(6,len(cells)),replace=False).tolist():
            dist=distances(env,goal)
            candidates=[s for s,d in dist.items() if 5<=d<=40 and length_bucket(d) in available]
            if not candidates:continue
            # Prefer starts at which greedy geometry disagrees with shortest path.
            hard=[s for s in candidates if distance_heuristic(env,s,goal) not in optimal_actions(env,s,dist)]
            initial=int(rng.choice(hard or candidates));bucket=length_bucket(dist[initial]);s=initial;trace=[]
            while s!=goal:
                opts=optimal_actions(env,s,dist);a=int(rng.choice(opts));nxt=step(env,s,a)
                text=serialize_observation(env,s,goal);parsed,ss,gg=parse_observation(text)
                assert ss==s and gg==goal and parsed['walls']==walls
                trace.append(dict(current=s,next=nxt,action=a,valid_actions=opts,remaining=dist[s],
                                  text=text,state=observed(env,s,goal)[1]))
                s=nxt
            reference=bfs_plan(env,initial,goal)
            assert len(reference['actions'])==len(trace)
            result.append(dict(episode=f'{group}:{goal}:{initial}',group=group,split=split,env=env,
                               initial=initial,goal=goal,trajectory=trace,minimum_required_steps=len(trace),
                               length_bucket=bucket,recipe=recipe,
                               initial_heuristic_optimal=distance_heuristic(env,initial,goal) in trace[0]['valid_actions'],
                               heuristic_rollout=heuristic_rollout(env,initial,goal,40)))
            counts[(split,bucket)]+=1;break
        if all(counts[(s,b)]>=per_stratum for s in splits for b in bins):return result,attempt+1
    raise RuntimeError(f'Could not fill benchmark quotas within {max_maps} maps: {dict(counts)}')


def write_benchmark(directory,**kwargs):
    out=output_path(directory)
    if out.exists():raise FileExistsError('A benchmark version is immutable; use a new output directory')
    legacy=json.loads((LEGACY/'artifacts/data/source_manifest.json').read_text())
    episodes,attempts=make_benchmark(excluded_groups=legacy['group_splits'],**kwargs)
    out.mkdir(parents=True)
    path=out/'episodes.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    counts=Counter((e['split'],e['length_bucket']) for e in episodes)
    manifest=dict(version='astra-hard-grid-v3',observation_schema=SCHEMA,seed=kwargs.get('seed',151803),
                  source_sha=sha256(__file__),episodes_sha=sha256(path),episodes=len(episodes),maps_attempted=attempts,
                  counts={s:{b:counts[(s,b)] for b in ('5-9','10-19','20+')} for s in ('train','valid','test')},
                  split_rule='original topology hash before trace construction; all legacy groups excluded',
                  excluded_legacy_manifest_sha=sha256(LEGACY/'artifacts/data/source_manifest.json'),
                  context='complete current state including every wall; no future state or labels',
                  codec_encoded=False,code_only_accepted=False,required_observed_blocks=4,
                  training_launched=False,minimum_length_is_not_complexity=True,
                  recipes=dict(Counter(e['recipe'] for e in episodes)))
    atomic_json(out/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'artifacts/hard_grid_v3')
    ap.add_argument('--per-stratum',type=int,default=8);args=ap.parse_args()
    print(json.dumps(write_benchmark(args.out,per_stratum=args.per_stratum),indent=2))
