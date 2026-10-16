"""Measures the three designed properties of GraphNav (advisor consult 2)."""
import itertools, numpy as np, json, sys
from data_v5 import *

def all_rule_seqs():
    out = []
    for nseg in (2, 3, 4):
        for cols in itertools.product(range(N_COL), repeat=nseg):
            if any(cols[i] == cols[i+1] for i in range(nseg-1)): continue
            for durs in itertools.product((2, 3, 4), repeat=nseg):
                out.append((cols, durs))
    return out
SEQS = all_rule_seqs()

def first_arrival_ok(f, start, goal, cols, durs):
    cur = start
    hops = [c for c, d in zip(cols, durs) for _ in range(d)]
    for i, c in enumerate(hops):
        cur = f[c, cur]
        if cur == goal: return i == len(hops) - 1
    return False

def count_walks(f, start, goal):
    # valid walks: any colour per hop, <=12 hops, first arrival at goal
    cnt = 0; frontier = {start: 1}
    for h in range(MAX_HOPS):
        nf = {}
        for v, n in frontier.items():
            for c in range(N_COL):
                u = f[c, v]
                if u == goal: cnt += n
                else: nf[u] = nf.get(u, 0) + n
        frontier = nf
    return cnt

rng = np.random.RandomState(123)
n_rule, n_walk, sw, runs, blind = [], [], [], [], []
for i in range(500):
    x, m, meta = gen_instance(rng)
    f, s, g = meta["f"], meta["start"], meta["goal"]
    assert m.sum() == (meta["n_hops"] + 1) * 4
    n_rule.append(sum(first_arrival_ok(f, s, g, c, d) for c, d in SEQS))
    n_walk.append(count_walks(f, s, g))
    hc = meta["hop_cols"]; sw += [hc[j] != hc[j+1] for j in range(len(hc)-1)]; runs += meta["durs"]
    # goal-blind floor: sample the generator's rule policy, success = first arrival at this goal
    succ = 0
    for _ in range(50):
        cols, durs = gen_rules(rng)
        succ += first_arrival_ok(f, s, g, cols, durs)
    blind.append(succ / 50)
n_rule = np.array(n_rule)
res = dict(rule_seqs_total=len(SEQS), rule_paths_median=float(np.median(n_rule)), rule_paths_mean=float(n_rule.mean()),
           frac_exactly_one=float((n_rule == 1).mean()), frac_ge3=float((n_rule >= 3).mean()),
           walks_median=float(np.median(n_walk)), switch_rate=float(np.mean(sw)), mean_run=float(np.mean(runs)),
           goal_blind_floor=float(np.mean(blind)))
print(json.dumps(res, indent=1))
