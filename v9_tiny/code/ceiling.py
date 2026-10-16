"""Exact demonstration next-action distribution for standard GraphNav prefixes (draft §6.6 G2.2(b)).

The standard generator (frozen data.py:120-129) draws a walk program: 2-4 colour runs (uniform), the first colour
uniform over 3, each later colour uniform over the 2 others, each run 2-4 hops (uniform): 2,322 programs per graph
and start. It keeps a program only if the final node is the goal and no earlier node (start included) equals it.
Conditional on the prompt (graph, start, goal), the demonstration is a program drawn with probability proportional
to its prior weight among the accepted ones ending at the goal.
"""
import itertools
from functools import lru_cache
import numpy as np
import v9paths
from data import parse_prompt, N_COL

DONE_ACTION = N_COL  # actions: colours 0..2, DONE = 3


@lru_cache(maxsize=1)
def programs():
    out = []
    for runs in (2, 3, 4):
        for first in range(N_COL):
            for rest in itertools.product(range(N_COL - 1), repeat=runs - 1):
                colours = [first]
                for r in rest:
                    prev = colours[-1]; colours.append(r + (r >= prev))
                for durations in itertools.product((2, 3, 4), repeat=runs):
                    seq = tuple(c for c, d in zip(colours, durations) for _ in range(d))
                    weight = (1 / 3) * (1 / 3) * (1 / 2) ** (runs - 1) * (1 / 3) ** runs
                    out.append((seq, weight))
    assert len(out) == 2322
    return out


def accepted(prompt):
    """Accepted programs for this prompt as (colour sequence, weight), with weights normalised."""
    task, edges, start, goal, _ = parse_prompt(prompt)
    if task != 'graph':
        raise ValueError('standard graph prompts only')
    keep = []
    for seq, w in programs():
        cur = start; visited = [start]; ok = True
        for c in seq:
            cur = edges[(cur, c)]; visited.append(cur)
        if cur != goal or goal in visited[:-1]:
            continue
        keep.append((seq, w))
    total = sum(w for _, w in keep)
    return [(s, w / total) for s, w in keep]


def next_action_distribution(prompt, prefix_colours, progs=None):
    """P(next action | prompt, demonstration prefix of colours); DONE when a consistent program has exactly that length."""
    progs = accepted(prompt) if progs is None else progs
    h = len(prefix_colours); p = np.zeros(N_COL + 1); pref = tuple(prefix_colours)
    for seq, w in progs:
        if seq[:h] == pref:
            p[DONE_ACTION if len(seq) == h else seq[h]] += w
    s = p.sum()
    if s <= 0:
        raise ValueError('prefix inconsistent with every accepted program')
    return p / s


def top2_mass(p):
    return float(np.sort(p)[-2:].sum())


def self_test():
    import numpy as np
    from data import gen_symbolic, COL0, DONE
    rng = np.random.RandomState(5); checked = 0
    for _ in range(30):
        x, m, d, meta = gen_symbolic(rng, 10, 'graph')
        p = meta['prompt_len']; resp = x[p:p + meta['response_len']]
        cols = [int(resp[i]) - COL0 for i in range(0, len(resp) - 4, 4)]
        progs = accepted(x[:p])
        assert any(seq == tuple(cols) for seq, _ in progs), 'demonstration must be an accepted program'
        for h in range(len(cols) + 1):
            dist = next_action_distribution(x[:p], cols[:h], progs)
            assert abs(dist.sum() - 1) < 1e-9
            nxt = DONE_ACTION if h == len(cols) else cols[h]
            assert dist[nxt] > 0
            checked += 1
    return checked


if __name__ == '__main__':
    print(len(programs()), self_test())
