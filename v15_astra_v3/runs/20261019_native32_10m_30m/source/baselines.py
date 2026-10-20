"""Known-rule baselines. Exact planners are explicitly labeled solver controls."""
from collections import deque
from .tasks import step


def legal_actions(env,state):
    return [a for a in range(7) if step(env,state,a) is not None]


def distance_heuristic(env,state,goal):
    legal=legal_actions(env,state)
    if not legal:return None
    def score(action):
        t=step(env,state,action)
        if env['domain']=='grid':
            n=env['n'];return abs(t%n-goal%n)+abs(t//n-goal//n)
        d=abs(t-goal);return min(d,env['m']-d)
    return min(legal,key=lambda a:(score(a),a))


def bfs_plan(env,start,goal,max_expansions=100000):
    """Return action sequence, expansions, transition calls; no label lookup."""
    queue=deque([start]);parent={start:None};expanded=0;calls=0
    while queue:
        s=queue.popleft()
        if s==goal:
            actions=[]
            while parent[s] is not None:
                prev,a=parent[s];actions.append(a);s=prev
            return dict(actions=actions[::-1],expansions=expanded,transition_calls=calls,complete=True)
        if expanded>=max_expansions:
            return dict(actions=None,expansions=expanded,transition_calls=calls,complete=False)
        expanded+=1
        for a in range(7):
            t=step(env,s,a);calls+=1
            if t is not None and t not in parent:
                parent[t]=(s,a);queue.append(t)
    return dict(actions=None,expansions=expanded,transition_calls=calls,complete=not queue)


def heuristic_rollout(env,start,goal,horizon=20):
    s=start
    if s==goal:return dict(steps_to_success=0,invalid=False)
    for k in range(1,horizon+1):
        a=distance_heuristic(env,s,goal)
        if a is None:return dict(steps_to_success=None,invalid=True)
        s=step(env,s,a)
        if s==goal:return dict(steps_to_success=k,invalid=False)
    return dict(steps_to_success=None,invalid=False)
