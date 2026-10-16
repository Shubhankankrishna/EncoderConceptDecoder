"""Grid-rule induction (Dataset_v13.md §5.2): ARC-like episodes with a bounded, parametric rule grammar.

An episode is three demonstration pairs and one query; the target is the query's output grid. Every episode is
answer-identifiable within its declared grammar: every program of depth <= 2 that is consistent with all three
demonstrations gives the same query output (checked analytically below, parameters inferred per skeleton).

Primitive types (S0): geo (one of the 8 dihedral maps), trans (offset dx, dy in [-2, 2], zero fill, no clipping),
recol (a consistent map of the non-background colours), sel (keep the largest or smallest object, canvas kept).
S2 adds crop (the largest object's tight bounding box). Compositions apply two types in the canonical order
sel -> geo -> trans -> crop -> recol, which loses nothing: the type pairs used here commute up to a re-fit parameter
(recol is always last, so its map is inferred from the other types' output). (trans, crop) is excluded: it equals crop.
Objects are single-colour 4-connected components, never touching each other.
"""
import numpy as np

BG = 0
GEO = ('id', 'r90', 'r180', 'r270', 'fh', 'fv', 'tr', 'atr')
OFFS = tuple((dx, dy) for dy in range(-2, 3) for dx in range(-2, 3) if (dx, dy) != (0, 0))
TYPE_ORDER = ('sel', 'geo', 'trans', 'crop', 'recol')


def geo(g, k):
    return {'id': g, 'r90': np.rot90(g, -1), 'r180': np.rot90(g, 2), 'r270': np.rot90(g, 1), 'fh': g[:, ::-1],
            'fv': g[::-1, :], 'tr': g.T, 'atr': np.rot90(g, 2).T}[k].copy()


def objects(g):
    """Single-colour 4-connected components: list of (colour, cells as (r, c) array)."""
    h, w = g.shape; seen = np.zeros_like(g, dtype=bool); out = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != BG and not seen[r, c]:
                col = g[r, c]; stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and g[yy, xx] == col:
                            seen[yy, xx] = True; stack.append((yy, xx))
                out.append((col, np.array(cells)))
    return out


def trans(g, off):
    dx, dy = off; h, w = g.shape
    ys, xs = np.nonzero(g != BG)
    if len(ys) and (ys.min() + dy < 0 or ys.max() + dy >= h or xs.min() + dx < 0 or xs.max() + dx >= w):
        return None                                                           # clipping: undefined
    out = np.zeros_like(g); out[ys + dy, xs + dx] = g[ys, xs]
    return out


def sel(g, which):
    obs = objects(g)
    if not obs: return None
    sizes = [len(c) for _, c in obs]
    target = max(sizes) if which == 'largest' else min(sizes)
    if sizes.count(target) != 1: return None                                  # ties: undefined
    col, cells = obs[sizes.index(target)]
    out = np.zeros_like(g); out[cells[:, 0], cells[:, 1]] = col
    return out


def crop(g):
    obs = objects(g)
    if not obs: return None
    sizes = [len(c) for _, c in obs]
    if sizes.count(max(sizes)) != 1: return None
    col, cells = obs[sizes.index(max(sizes))]
    y0, y1, x0, x1 = cells[:, 0].min(), cells[:, 0].max(), cells[:, 1].min(), cells[:, 1].max()
    box = g[y0:y1 + 1, x0:x1 + 1]
    if np.any((box != BG) & (box != col)): return None                         # another object inside the box
    out = np.zeros_like(box); m = np.zeros_like(box, dtype=bool)
    m[cells[:, 0] - y0, cells[:, 1] - x0] = True; out[m] = col
    return out


def recol(g, cmap):
    out = g.copy()
    for a, b in cmap.items(): out[g == a] = b
    return out


def apply(prog, g):
    """prog: tuple of (type, param). Returns the output grid or None if undefined."""
    for t, p in prog:
        if g is None: return None
        g = {'geo': lambda x: geo(x, p), 'trans': lambda x: trans(x, p), 'sel': lambda x: sel(x, p),
             'recol': lambda x: recol(x, p), 'crop': crop}[t](g)
    return g


# ---------------------------------------------------------------- inference of parameters (identifiability)
def infer_recol(pairs):
    """Map colour -> colour consistent with every (input, output) pair, or None."""
    cmap = {}
    for a, b in pairs:
        if a is None or a.shape != b.shape: return None
        if np.any((a == BG) != (b == BG)): return None
        for c in np.unique(a[a != BG]):
            vals = np.unique(b[a == c])
            if len(vals) != 1: return None
            if cmap.setdefault(int(c), int(vals[0])) != int(vals[0]): return None
    return cmap


def candidates(types_allowed, depth2_pairs):
    """Skeletons in canonical order: singles of each allowed type plus the allowed ordered type pairs."""
    sk = [(t,) for t in types_allowed]
    sk += [tuple(p) for p in depth2_pairs]
    return sk


def param_space(t):
    return {'geo': GEO, 'trans': OFFS, 'sel': ('largest', 'smallest'), 'crop': (None,)}[t]


def consistent_outputs(skeleton, demos, queries):
    """Yield the query outputs of every parameterisation of `skeleton` consistent with all demos."""
    head = [t for t in skeleton if t != 'recol']
    has_recol = skeleton[-1] == 'recol'
    spaces = [param_space(t) for t in head]
    def rec(i, prog):
        if i == len(head):
            mids = [apply(prog, a) for a, _ in demos]
            if any(m is None for m in mids): return
            if has_recol:
                cmap = infer_recol([(m, b) for m, (_, b) in zip(mids, demos)])
                if cmap is None: return
                full = prog + (('recol', cmap),)
                outs = []
                for q in queries:
                    mq = apply(prog, q)
                    if mq is None: yield None; return
                    if any(int(c) not in cmap for c in np.unique(mq[mq != BG])): yield None; return
                    outs.append(recol(mq, cmap))
                yield full, outs
            else:
                if all(m.shape == b.shape and np.array_equal(m, b) for m, (_, b) in zip(mids, demos)):
                    yield prog, [apply(prog, q) for q in queries]
            return
        for p in spaces[i]:
            yield from rec(i + 1, prog + ((head[i], p),))
    yield from rec(0, ())


def identifiable(true_prog, demos, query, skeletons):
    """True iff every consistent program in the grammar gives the true query output (and is defined)."""
    truth = apply(true_prog, query)
    if truth is None: return False
    for sk in skeletons:
        for res in consistent_outputs(sk, demos, [query]):
            if res is None: return False                                     # a consistent program is undefined
            _, (out,) = res
            if out is None or out.shape != truth.shape or not np.array_equal(out, truth): return False
    return True


def determined_prefix(demos, skeletons):
    """For k = 1, 2: is demo k+1's output determined by demos 1..k within the grammar?"""
    flags = []
    for k in (1, 2):
        prev, nxt = demos[:k], demos[k]
        truth = nxt[1]; ok = True; seen = False
        for sk in skeletons:
            for res in consistent_outputs(sk, prev, [nxt[0]]):
                seen = True
                if res is None: ok = False; break
                _, (out,) = res
                if out is None or out.shape != truth.shape or not np.array_equal(out, truth): ok = False; break
            if not ok: break
        flags.append(ok and seen)
    return flags


# ---------------------------------------------------------------- random grids
def blob(rng, n):
    cells = {(0, 0)}
    while len(cells) < n:
        y, x = list(cells)[rng.integers(len(cells))]
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[rng.integers(4)]
        cells.add((y + dy, x + dx))
    a = np.array(sorted(cells)); a -= a.min(0)
    return a


def random_grid(rng, h, w, n_obj, colours, max_size):
    """Place n_obj single-colour blobs of distinct sizes where possible, non-touching (one-cell margin)."""
    for _ in range(60):
        g = np.zeros((h, w), dtype=np.int64); occ = np.zeros((h + 2, w + 2), dtype=bool); ok = True
        cols = rng.choice(colours, size=n_obj, replace=n_obj > len(colours))
        sizes = rng.integers(1, max_size + 1, size=n_obj)
        for k in range(n_obj):
            placed = False
            for _ in range(30):
                b = blob(rng, int(sizes[k]))
                bh, bw = b[:, 0].max() + 1, b[:, 1].max() + 1
                if bh > h or bw > w: continue
                y0, x0 = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
                ys, xs = b[:, 0] + y0, b[:, 1] + x0
                if occ[ys + 1, xs + 1].any(): continue
                g[ys, xs] = cols[k]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        occ[ys + 1 + dy, xs + 1 + dx] = True
                placed = True; break
            if not placed: ok = False; break
        if ok and (g != BG).any(): return g
    return None


# ---------------------------------------------------------------- ladders
# ID ladder on size (rungs 0..7 trained): two one-row rungs, then square-ish 2D sides 3..8. Extension: sides 9, 10.
ID_RUNGS = [('1d', 6, 8), ('1d', 9, 12), ('2d', 3), ('2d', 4), ('2d', 5), ('2d', 6), ('2d', 7), ('2d', 8)]
EXT_RUNGS = [('2d', 9), ('2d', 10)]


def dims(rng, rung):
    if rung[0] == '1d': return 1, int(rng.integers(rung[1], rung[2] + 1))
    s = rung[1]
    return int(rng.integers(max(2, s - 1), s + 1)), int(rng.integers(max(2, s - 1), s + 1))


def sample_param(rng, t, colours_present, palette):
    if t == 'geo': return GEO[rng.integers(len(GEO))]
    if t == 'trans': return OFFS[rng.integers(len(OFFS))]
    if t == 'sel': return ('largest', 'smallest')[rng.integers(2)]
    if t == 'crop': return None
    if t == 'recol':
        src = sorted(colours_present); dst = list(rng.permutation(palette)[:len(src)])
        cmap = {int(a): int(b) for a, b in zip(src, dst)}
        if all(a == b for a, b in cmap.items()):                              # must change something
            a = src[0]; cmap[int(a)] = int([c for c in palette if c != a][rng.integers(len(palette) - 1)])
        return cmap
    raise ValueError(t)


def make_episode(rng, skeleton, rung, grammar, param_ok=None, max_tries=400):
    """Sample an identifiable episode for `skeleton` (tuple of types) at size rung `rung`.

    grammar: list of skeletons for the identifiability check. param_ok(t, p) -> bool restricts parameter values
    (the new-parameter split). Returns dict(demos, query, out, prog, determined) or None."""
    for _ in range(max_tries):
        h, w = dims(rng, rung)
        palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
        n_col = int(rng.integers(2, 6)); colours = palette[:n_col]
        cells = h * w
        n_obj = 1 if cells < 9 else int(rng.integers(1, min(4, max(2, cells // 12)) + 1))
        if 'sel' in skeleton or 'crop' in skeleton: n_obj = max(2, n_obj)
        max_size = max(1, min(6, cells // max(3, 2 * n_obj)))
        grids = [random_grid(rng, h, w, n_obj, colours, max_size) for _ in range(4)]
        if any(g is None for g in grids): continue
        present = sorted({int(c) for g in grids for c in np.unique(g[g != BG])})
        prog = []
        for t in skeleton:
            p = sample_param(rng, t, present, palette)
            if t == 'trans' and h == 1: p = ((-2, -1, 1, 2)[rng.integers(4)], 0)    # one-row grids: dy = 0
            if param_ok is not None and not param_ok(t, p): break
            prog.append((t, p))
        if len(prog) != len(skeleton): continue
        prog = tuple(prog)
        outs = [apply(prog, g) for g in grids]
        if any(o is None for o in outs): continue
        if all(np.array_equal(g, o) for g, o in zip(grids, outs)) and skeleton != ('geo',): continue
        demos = list(zip(grids[:3], outs[:3]))
        if not identifiable(prog, demos, grids[3], grammar): continue
        return dict(demos=demos, query=grids[3], out=outs[3], prog=prog, determined=determined_prefix(demos, grammar),
                    hw=(h, w))
    return None


def canon(ep):
    """Semantic instance key: grids up to a consistent relabelling of non-background colours."""
    seq = [g for d in ep['demos'] for g in d] + [ep['query']]
    order = {}
    for g in seq:
        for c in g.flatten():
            if c != BG and int(c) not in order: order[int(c)] = len(order) + 1
    parts = []
    for g in seq:
        parts.append(f"{g.shape[0]}x{g.shape[1]}:" + ''.join(str(order.get(int(c), 0)) for c in g.flatten()))
    return '|'.join(parts)
