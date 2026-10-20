"""Complete directed graph observations; exact dynamics; disjoint graph instances."""
import heapq
import json
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch

from .common import ROOT, atomic_json, sha

GAMMA = 0.95


def legal(g, node):
    return [b for a, b in g['edges'] if a == node] if node != g['goal'] else []


def step(g, node, action):
    if action not in legal(g, node):
        raise ValueError('Illegal graph action')
    return action, action == g['goal']


def cost(g, action, strategy):
    return 1 + 2 * strategy * g['risk'][action]


def distances(g, strategy=0):
    """Reverse Dijkstra labels; never passed into model inputs."""
    reverse = defaultdict(list)
    for a, b in g['edges']:
        reverse[b].append(a)
    d = [float('inf')] * g['n']
    d[g['goal']] = 0
    queue = [(0, g['goal'])]
    while queue:
        dv, v = heapq.heappop(queue)
        if dv != d[v]:
            continue
        for u in reverse[v]:
            candidate = dv + cost(g, v, strategy)
            if candidate < d[u]:
                d[u] = candidate
                heapq.heappush(queue, (candidate, u))
    return d


def oracle_path(g, start, strategy=0):
    d = distances(g, strategy)
    if d[start] == float('inf'):
        return []
    path = [start]
    while path[-1] != g['goal']:
        node = min(legal(g, path[-1]), key=lambda a: (cost(g, a, strategy) + d[a], a))
        path.append(node)
        if len(path) > g['n'] + 1:
            raise RuntimeError('Oracle cycle')
    return path


def as_networkx(g):
    h = nx.DiGraph()
    for i in range(g['n']):
        h.add_node(i, semantic=str((g['risk'][i], i == g['goal'])))
    h.add_edges_from(g['edges'])
    return h


def build_dataset(path=ROOT/'data/graphs.json', count=2000, seed=191026):
    path = Path(path)
    if path.exists():
        return load_dataset(path)
    rng = random.Random(seed)
    graphs = []
    buckets = defaultdict(list)
    split_counts = {'train': int(count*.8), 'valid': int(count*.1), 'test': count-int(count*.9),
                    'ood': max(40, count//10)}
    for split, quota in split_counts.items():
        for index in range(quota):
            while True:
                n = rng.randint(16, 20) if split == 'ood' else rng.randint(5, 12)
                density = rng.uniform(.08, .16) if split == 'ood' else rng.uniform(.13, .32)
                edges = [[a,b] for a in range(n) for b in range(n) if a != b and rng.random() < density]
                g = {'n':n, 'edges':edges, 'risk':[rng.randrange(2) for _ in range(n)],
                     'goal':rng.randrange(n), 'split':split, 'id':f'{split}_{index:05d}'}
                d = distances(g)
                difficulty=index%3
                if split=='ood':
                    low,high=((2,3),(4,5),(6,20))[difficulty]
                else:
                    low,high=((2,2),(3,3),(4,12))[difficulty]
                candidates = [i for i,v in enumerate(d) if low <= v <= high]
                if not candidates:
                    continue
                # Reject graph isomorphisms even when goal/risk labels differ.
                topology = nx.DiGraph(); topology.add_nodes_from(range(n)); topology.add_edges_from(edges)
                key = nx.weisfeiler_lehman_graph_hash(topology, iterations=4)
                if any(nx.is_isomorphic(topology, h) for h in buckets[key]):
                    continue
                buckets[key].append(topology)
                g['start'] = rng.choice(candidates)
                g['difficulty_bin']=difficulty
                g['start_shortest_steps']=int(d[g['start']])
                graphs.append(g)
                break
    payload = {'version':2, 'seed':seed, 'graphs':graphs, 'splits':split_counts,
               'semantics':'complete graph; goal absorbing; gamma=.95; strategy0 cost1; strategy1 cost1+2*risk(destination)',
               'dedup':'global topology WL buckets with exact directed isomorphism rejection',
               'start_strata':'equal quotas 2 / 3 / 4+ hops in-distribution; 2-3 / 4-5 / 6+ hops OOD'}
    atomic_json(path, payload)
    atomic_json(path.with_suffix('.manifest.json'), {'sha256':sha(path), 'splits':split_counts,
        'graph_count':len(graphs), 'states_train':sum(g['n'] for g in graphs if g['split']=='train'),
        'counterfactual_train_edges':sum(len(g['edges']) for g in graphs if g['split']=='train')})
    return payload


def load_dataset(path=ROOT/'data/graphs.json'):
    path = Path(path)
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    if manifest['sha256'] != sha(path):
        raise ValueError('Dataset hash mismatch')
    return json.loads(path.read_text())


def batch_states(items, device='cpu', labels=True):
    """Each item is (graph, current_node, strategy); no oracle features in inputs."""
    b, n = len(items), max(g['n'] for g,_,_ in items)
    feat = torch.zeros(b,n,5)
    adj = torch.zeros(b,n,n)
    mask = torch.zeros(b,n,dtype=torch.bool)
    current = torch.tensor([c for _,c,_ in items])
    goal = torch.tensor([g['goal'] for g,_,_ in items])
    z = torch.tensor([s for _,_,s in items])
    valid = torch.zeros(b,n,dtype=torch.bool)
    policy = torch.zeros(b,n)
    q = torch.zeros(b,n)
    value = torch.zeros(b)
    for k,(g,c,s) in enumerate(items):
        nn = g['n']; mask[k,:nn] = True
        for u,v in g['edges']: adj[k,u,v] = 1
        feat[k,:nn,0] = torch.tensor(g['risk'])
        feat[k,c,1] = 1; feat[k,g['goal'],2] = 1
        feat[k,:nn,3] = adj[k,:nn,:nn].sum(-1)/max(1,nn-1)
        feat[k,:nn,4] = adj[k,:nn,:nn].sum(-2)/max(1,nn-1)
        actions = legal(g,c); valid[k,actions] = True
        if labels:
            d = distances(g,s)
            value[k] = GAMMA**d[c]
            for a in actions: q[k,a] = GAMMA**(cost(g,a,s)+d[a])
            if actions:
                best = max(q[k,a].item() for a in actions)
                optimal = [a for a in actions if abs(q[k,a].item()-best)<1e-6]
                policy[k,optimal] = 1/len(optimal)
    return {k:v.to(device) for k,v in {'features':feat, 'adj':adj, 'node_mask':mask,
        'current':current, 'goal':goal, 'strategy':z, 'legal':valid,
        'policy_target':policy, 'q_target':q, 'value_target':value}.items()}
