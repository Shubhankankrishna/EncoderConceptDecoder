"""GraphNav generator derived from v5_tiny/code/data_v5.py.
The text pipeline and strict V6 verifier live in data.py.
"""
import os
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
TS_GLOB = os.path.expanduser("~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/*/data/")

N_NODE_TOK, N_COL = 24, 3
SPECIALS = (["<pad>", "<eos>", "<gnav>", "<sep>", "<to>", "<done>"]
            + [f"<n{i}>" for i in range(N_NODE_TOK)] + [f"<c{i}>" for i in range(N_COL)])
PAD, EOS, GNAV, SEP, TO, DONE = range(6)
NODE0 = 6
COL0 = NODE0 + N_NODE_TOK
NSPEC = len(SPECIALS)            # 33
VOCAB = 2048
T = 192
K = 4
N_NODES = 10
MAX_HOPS = 16
HEADER_CHUNKS = 1 + N_NODES * N_COL      # 31 chunks = 124 tokens
WALK_START = HEADER_CHUNKS * K           # token index of the first hop
HOP_FMT = "to"  # V6 fixes the declared [colour, <to>, node, <sep>] protocol.
                                          # "srcto": [cur, colour, node, <to>]  (header-mirroring offsets + walk cue)


def gen_graph(rng):
    labels = rng.choice(N_NODE_TOK, N_NODES, replace=False)
    f = np.zeros((N_COL, N_NODES), dtype=np.int64)
    for c in range(N_COL):
        for v in range(N_NODES):
            u = rng.randint(N_NODES - 1)
            f[c, v] = u + (u >= v)            # no fixed points
    return labels, f


def gen_rules(rng):
    nseg = rng.randint(2, 5)
    cols, durs, prev = [], [], -1
    for _ in range(nseg):
        c = rng.randint(N_COL - 1) if prev >= 0 else rng.randint(N_COL)
        if prev >= 0:
            c = c + (c >= prev)
        cols.append(c); durs.append(rng.randint(2, 5)); prev = c
    return cols, durs


def gen_instance(rng, max_tries=100):
    """Returns tokens [T], loss mask [T], meta dict. Node indices are instance-local 0..N-1."""
    for _ in range(max_tries):
        labels, f = gen_graph(rng)
        start = rng.randint(N_NODES)
        cols, durs = gen_rules(rng)
        nodes, hop_cols, seg_id, seg_pos = [start], [], [], []
        for si, (c, d) in enumerate(zip(cols, durs)):
            for j in range(d):
                nodes.append(f[c, nodes[-1]]); hop_cols.append(c); seg_id.append(si); seg_pos.append(j)
        goal = nodes[-1]
        if goal in nodes[:-1]:
            continue
        x = np.full(T, PAD, dtype=np.int64)
        x[0:4] = [GNAV, NODE0 + labels[goal], NODE0 + labels[start], SEP]
        edges = [(v, c) for v in range(N_NODES) for c in range(N_COL)]
        order = rng.permutation(len(edges))
        p = 4
        for e in order:
            v, c = edges[e]
            x[p:p + 4] = [NODE0 + labels[v], COL0 + c, NODE0 + labels[f[c, v]], SEP]; p += 4
        assert p == WALK_START
        for c, prev, u in zip(hop_cols, nodes[:-1], nodes[1:]):
            if HOP_FMT == "src":
                x[p:p + 4] = [NODE0 + labels[prev], COL0 + c, NODE0 + labels[u], SEP]
            elif HOP_FMT == "srcto":
                x[p:p + 4] = [NODE0 + labels[prev], COL0 + c, NODE0 + labels[u], TO]
            else:
                x[p:p + 4] = [COL0 + c, TO, NODE0 + labels[u], SEP]
            p += 4
        x[p:p + 4] = [DONE, SEP, SEP, SEP]; p += 4
        m = np.zeros(T, dtype=np.float32)
        m[WALK_START - 1:p - 1] = 1.0   # predict tokens WALK_START..p-1 (inputs at positions WALK_START-1..p-2)
        meta = dict(labels=labels, f=f, start=start, goal=goal, cols=cols, durs=durs, hop_cols=hop_cols,
                    seg_id=seg_id, seg_pos=seg_pos, n_hops=len(hop_cols))
        return x, m, meta
    raise RuntimeError("no instance")


def gen_hardgoal(rng, min_dist=4, max_tries=200):
    """Hard-goal OOD instance (pre-registered 11:20): same graph/header format, goal = uniformly random node at
    shortest-path distance >= min_dist from start (any colours), so reaching it is not guaranteed by rule-structured
    walks.  Only the header is meaningful (no demonstration); used for free-generation success only."""
    for _ in range(max_tries):
        labels, f = gen_graph(rng)
        start = rng.randint(N_NODES)
        dist = {start: 0}; frontier = [start]
        while frontier:
            nxt = []
            for v in frontier:
                for c in range(N_COL):
                    u = f[c, v]
                    if u not in dist:
                        dist[u] = dist[v] + 1; nxt.append(u)
            frontier = nxt
        far = [v for v, d in dist.items() if d >= min_dist]
        if not far:
            continue
        goal = far[rng.randint(len(far))]
        x = np.full(T, PAD, dtype=np.int64)
        x[0:4] = [GNAV, NODE0 + labels[goal], NODE0 + labels[start], SEP]
        edges = [(v, c) for v in range(N_NODES) for c in range(N_COL)]
        p = 4
        for e in rng.permutation(len(edges)):
            v, c = edges[e]
            x[p:p + 4] = [NODE0 + labels[v], COL0 + c, NODE0 + labels[f[c, v]], SEP]; p += 4
        return x, dict(labels=labels, f=f, start=start, goal=goal, dist=dist[goal])
    raise RuntimeError("no hard-goal instance")


def hardgoal_set(n, seed, min_dist=4):
    rng = np.random.RandomState(seed)
    xs, metas = zip(*[gen_hardgoal(rng, min_dist) for _ in range(n)])
    return np.stack(xs), list(metas)


def graph_batch(rng, B):
    xs, ms, metas = zip(*[gen_instance(rng) for _ in range(B)])
    return np.stack(xs), np.stack(ms), list(metas)


def parse_header(x):
    """Recover (label->local index, f, start, goal) from the header tokens alone (what a deployed verifier sees)."""
    goal_l, start_l = int(x[1]) - NODE0, int(x[2]) - NODE0
    lab = {}
    edges = {}
    for ch in range(1, HEADER_CHUNKS):
        v, c, u, _ = (int(t) for t in x[4 * ch:4 * ch + 4])
        edges[(v - NODE0, c - COL0)] = u - NODE0
    return edges, start_l, goal_l



def fixed_graph_set(n, seed):
    rng = np.random.RandomState(seed)
    xs, ms, metas = graph_batch(rng, n)
    return xs, ms, metas

