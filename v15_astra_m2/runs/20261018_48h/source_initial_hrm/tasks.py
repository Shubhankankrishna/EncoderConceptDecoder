"""Two exact deterministic planning environments; no learned transition oracle."""
import hashlib
import json
import re
from collections import deque

import numpy as np

VERSION = 'astra-modular-grid-v1'
ACTIONS = ('add', 'multiply', 'subtract', 'left', 'right', 'up', 'down')
ACTION_TEXTS = tuple(f'The answer is to {a}.' for a in ACTIONS)
STATE_LENGTH = 88


def parse_action(text):
    # First complete sentence is the action boundary, fixed for all examples.
    # No target token length is used at inference. Remaining decoder text is
    # explicitly outside the one-action grammar and is logged in evaluations.
    m = re.match(r'^the answer is to (add|multiply|subtract|left|right|up|down)\s*\.', text.strip().lower())
    return ACTIONS.index(m.group(1)) if m else None


def group_split(group):
    n = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'train' if n < 80 else 'valid' if n < 90 else 'test'


def step(env, state, action):
    if env['domain'] == 'mod':
        m, a, b = env['m'], env['a'], env['b']
        if action not in (0, 1, 2):
            return None
        return ((state+a) % m, (state*b) % m, (state-a) % m)[action]
    if action not in (3, 4, 5, 6):
        return None
    n = env['n']
    x, y = state % n, state // n
    dx, dy = {3:(-1,0),4:(1,0),5:(0,-1),6:(0,1)}[action]
    xx, yy = x+dx, y+dy
    if 0 <= xx < n and 0 <= yy < n and not env['walls'][yy*n+xx]:
        return yy*n+xx
    return None


def distances(env, goal):
    n = env['m'] if env['domain']=='mod' else env['n']**2
    rev = [[] for _ in range(n)]
    acts = (0,1,2) if env['domain']=='mod' else (3,4,5,6)
    for s in range(n):
        if env['domain']=='grid' and env['walls'][s]:
            continue
        for a in acts:
            t = step(env,s,a)
            if t is not None:
                rev[t].append(s)
    dist={goal:0}; queue=deque([goal])
    while queue:
        t=queue.popleft()
        for s in rev[t]:
            if s not in dist:
                dist[s]=dist[t]+1; queue.append(s)
    return dist


def optimal_actions(env, s, dist):
    acts = (0,1,2) if env['domain']=='mod' else (3,4,5,6)
    return [a for a in acts if (t:=step(env,s,a)) is not None and dist.get(t,999)==dist[s]-1]


def observed(env, state, goal):
    vec=np.zeros(STATE_LENGTH,np.int64)
    if env['domain']=='mod':
        vec[:6]=[1,env['m'],state,goal,env['a'],env['b']]
        text=(f'The number is {state}. The goal is {goal}. The modulus is {env["m"]}. '
              f'You can add {env["a"]}, multiply by {env["b"]}, or subtract {env["a"]}.')
    else:
        n=env['n'];vec[:6]=[2,n,state%n,state//n,goal%n,goal//n]
        vec[7:7+n*n]=np.array(env['walls'])+1
        text=(f'The maze has {n} rows and {n} columns. You are at row {state//n}, column {state%n}. '
              f'The goal is row {goal//n}, column {goal%n}. Choose left, right, up, or down.')
    return text, vec.tolist()


def make_episodes(seed=1518, modular_rules=1000, grid_maps=3000, goals_per_group=3, starts_per_goal=3):
    rng=np.random.default_rng(seed)
    envs=[]
    # Exact rule families split before selecting goals, starts, or trace steps.
    candidates=[(m,a,b) for m in range(11,72) for a in range(1,6) for b in (2,3,4,5)]
    rng.shuffle(candidates)
    for m,a,b in candidates[:modular_rules]:
        envs.append(dict(domain='mod',m=m,a=a,b=b,group=f'mod:{m}:{a}:{b}',ood=False))
    for i in range(grid_maps):
        n=int(rng.choice([5,6,7])); walls=(rng.random(n*n)<.22).astype(int).tolist()
        digest=hashlib.sha256(json.dumps([n,walls]).encode()).hexdigest()
        envs.append(dict(domain='grid',n=n,walls=walls,group=f'grid:{digest}',ood=False))
    # OOD data is separate from the validation sweep and final ID test.
    for m in (79,83,89,97,101,107,113):
        for a in (2,4):
            envs.append(dict(domain='mod',m=m,a=a,b=3,group=f'mod:{m}:{a}:3',ood=True))
    for i in range(40):
        n=9;walls=(rng.random(n*n)<.25).astype(int).tolist()
        digest=hashlib.sha256(json.dumps([n,walls]).encode()).hexdigest()
        envs.append(dict(domain='grid',n=n,walls=walls,group=f'grid:{digest}',ood=True))
    seen=set()
    for env in envs:
        if env['group'] in seen: continue
        seen.add(env['group'])
        split='ood' if env['ood'] else group_split(env['group'])
        possible=list(range(env['m'])) if env['domain']=='mod' else [i for i,w in enumerate(env['walls']) if not w]
        if len(possible)<2:continue
        goals=rng.choice(possible,min(goals_per_group,len(possible)),replace=False).tolist()
        for goal in goals:
            dist=distances(env,goal)
            starts=[s for s in possible if s in dist and 1<=dist[s]<=20]
            if not starts:continue
            for initial in rng.choice(starts,min(starts_per_goal,len(starts)),replace=False).tolist():
                episode=f'{env["group"]}:{goal}:{initial}'
                s=initial;trajectory=[]
                while s!=goal:
                    opts=optimal_actions(env,s,dist)
                    action=int(rng.choice(opts)); nxt=step(env,s,action)
                    assert nxt is not None and dist[nxt]==dist[s]-1
                    text,vec=observed(env,s,goal)
                    trajectory.append(dict(text=text,state=vec,action=action,valid_actions=opts,
                        current=s,next=nxt,remaining=dist[s]))
                    s=nxt
                yield dict(episode=episode,group=env['group'],split=split,env=env,
                           goal=goal,initial=initial,trajectory=trajectory)
