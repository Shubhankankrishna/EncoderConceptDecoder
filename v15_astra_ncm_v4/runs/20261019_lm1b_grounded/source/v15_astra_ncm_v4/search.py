"""Exact-environment PUCT; cycle cutoffs are local to a simulation path."""
from dataclasses import dataclass, field
import math
import time

from .graphs import GAMMA, cost, legal, step


@dataclass
class Edge:
    prior: float
    visits: int = 0
    total: float = 0.
    child: object = None


@dataclass
class Node:
    state: int
    visits: int = 0
    edges: dict = field(default_factory=dict)
    expanded: bool = False


def mcts(g,start,predict,strategy=0,simulations=16,c_puct=1.5,max_depth=None):
    if simulations<1:
        raise ValueError('Positive simulation budget required')
    began = time.perf_counter()
    root = Node(start)
    counters = {'model_calls':0,'environment_steps':0,'simulations':simulations}
    cached = {}
    max_depth = max_depth or 2*g['n']

    def expand(node):
        if node.state == g['goal']:
            return 1.
        actions = legal(g,node.state)
        if not actions:
            return 0.
        if node.state not in cached:
            priors,value = predict(g,node.state,strategy)
            if any(a not in actions for a in priors):
                raise ValueError('Predictor returned illegal action')
            norm = sum(max(0.,priors.get(a,0.)) for a in actions)
            priors = {a:max(0.,priors.get(a,0.))/norm if norm else 1/len(actions) for a in actions}
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid leaf value')
            cached[node.state] = priors,value
            counters['model_calls'] += 1
        priors,value = cached[node.state]
        node.edges = {a:Edge(priors[a]) for a in actions}
        node.expanded = True
        return value

    expand(root)
    for _ in range(simulations):
        node = root
        visited = {start}
        path = []
        leaf_value = 0.
        for depth in range(max_depth):
            if node.state == g['goal']:
                leaf_value = 1.; break
            if not legal(g,node.state):
                leaf_value = 0.; break
            if not node.expanded:
                leaf_value = expand(node); break
            def score(a):
                e = node.edges[a]
                return e.total/max(1,e.visits)+c_puct*e.prior*math.sqrt(max(1,node.visits))/(1+e.visits)
            action = max(node.edges,key=lambda a:(score(a),node.edges[a].prior,-a))
            edge = node.edges[action]
            path.append((node,edge,cost(g,action,strategy)))
            if edge.child is None:
                successor,_ = step(g,node.state,action)
                counters['environment_steps'] += 1
                edge.child = Node(successor)
            node = edge.child
            if node.state in visited:
                leaf_value = 0.; break
            visited.add(node.state)
        else:
            leaf_value = 1. if node.state==g['goal'] else 0.
        if not path:
            root.visits += 1
        for parent,edge,edge_cost in reversed(path):
            leaf_value *= GAMMA**edge_cost
            edge.visits += 1; edge.total += leaf_value; parent.visits += 1
    assert root.visits==simulations
    action = max(root.edges,key=lambda a:(root.edges[a].visits,root.edges[a].total/max(1,root.edges[a].visits),-a)) if root.edges else None
    return {'action':action,'root_visits':root.visits,
            'edges':{a:{'visits':e.visits,'q':e.total/max(1,e.visits),'prior':e.prior} for a,e in root.edges.items()},
            **counters,'seconds':time.perf_counter()-began}
