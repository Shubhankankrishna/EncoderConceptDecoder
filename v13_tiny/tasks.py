"""Task generators for the V13 tiny test: Dataset_v13.md §5 (four reasoning families) and §8.1 (continual phases).

A generator returns an Example or None (rejected; the caller retries). An Example holds
  prompt  list of pieces; a piece is a special token (tok.SPECIALS) or plain text, BPE-encoded on its own;
  target  list of pieces ending with '<eos>' (the scored answer);
  extra   (start, end) piece ranges inside the prompt that are also scored (grid demonstration outputs already
          determined by the earlier demonstrations, §5.2's registered option); the metric stays on the target;
  key     the semantic-instance key that the hash split (§6.1) is computed on;
  meta    task, rung, phase, and the facts the verifier needs (never shown to the model).

Formats are frozen here for every architecture (§5.3): numbers in prompts are written most-significant digit first,
every numeric answer least-significant digit first, a negative number carries one '<neg>' token, and each digit is
its own token. Ladders (§7.4): every task type has one ladder of rungs; the first `train` rungs are trained, the rest
are test-only (the extension part). Endpoints are fixed a priori (no reference pilot ran; DEVIATIONS.md).
"""
import hashlib
from collections import deque

import numpy as np

import grid as G

# ------------------------------------------------------------------------------------------------ shared helpers
def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


def bucket(key, ns='v13s0'):
    """§6.1: stable 80/5/5/10 split of semantic instances."""
    b = h64(ns + '|' + key) % 100
    return 'train' if b < 80 else 'calib' if b < 85 else 'dev' if b < 90 else 'test'


def num_in(n):
    n = int(n); return (['<neg>'] if n < 0 else []) + list(str(abs(n)))


def num_out(n):
    n = int(n); return (['<neg>'] if n < 0 else []) + list(str(abs(n))[::-1])


class Example(dict):
    pass


def ex(prompt, target, key, task, rung, extra=(), **meta):
    return Example(prompt=prompt, target=target, extra=list(extra), key=key, meta=dict(task=task, rung=rung, **meta))


def choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


# ------------------------------------------------------------------------------------------------ grid family
COL = [f'<c{i}>' for i in range(10)]
ALT_COL = [f'<k{i}>' for i in range(10)]                       # S4 rendering: reserved colour tokens + legend
GRID_SINGLES_S0 = [('sel',), ('geo',), ('trans',), ('recol',)]
GRID_PAIRS_S0 = [('sel', 'geo'), ('geo', 'trans'), ('geo', 'recol')]
GRID_PAIRS_S1 = [('sel', 'trans')]                              # temporarily unseen until S1
GRID_PAIRS_DEV = [('sel', 'recol')]                             # permanent dev composition
GRID_PAIRS_TEST = [('trans', 'recol')]                          # permanent test composition
GRID_S2 = [('crop',)]                                           # new primitive
GRID_S5 = [('geo', 'crop'), ('crop', 'recol')]                  # composition after acquisition
_ALL_S0 = ['sel', 'geo', 'trans', 'recol']
GRAMMAR_S0 = G.candidates(_ALL_S0, [(a, b) for i, a in enumerate(G.TYPE_ORDER) for b in G.TYPE_ORDER[i + 1:]
                                    if a in _ALL_S0 and b in _ALL_S0])
GRAMMAR_S2 = GRAMMAR_S0 + [('crop',), ('sel', 'crop'), ('geo', 'crop'), ('crop', 'recol')]
GRID_RUNGS = G.ID_RUNGS + G.EXT_RUNGS                           # 8 trained (1-D rows, sides 3-8) + sides 9, 10


def held_colour_pair(a, b):
    """New-parameter panel (§5.2): colour substitutions a -> b held out of training by hash (~1 in 8)."""
    return h64(f'recol|{a}->{b}') % 8 == 0


def held_offset(off):
    return h64(f'trans|{off[0]},{off[1]}') % 6 == 0


HELD_OFFSETS = [o for o in G.OFFS if held_offset(o)] + [o for o in [(-2, 0), (-1, 0), (1, 0), (2, 0)] if held_offset(o)]


def param_is_new(t, p):
    if t == 'recol': return any(held_colour_pair(a, b) for a, b in p.items() if a != b)
    if t == 'trans': return held_offset(p)
    return False


def grid_pieces(g, cols):
    h, w = g.shape
    out = num_in(h) + ['<x>'] + num_in(w)
    for r in range(h):
        out += ['<nl>'] + [cols[int(c)] for c in g[r]]
    return out


def gen_grid(rng, task, rung, mode='train', phase='S0'):
    """task: 'grid1' (one primitive) or 'grid2' (two). mode: train | id | newparam | comp_dev | comp_test.
    phase picks the primitive/pair inventory (S0, S1, S2, S3 (largest only), S4 (alt colours), S5, S6)."""
    if phase in ('S2',) or (phase == 'S5' and task == 'grid1'):
        skels, grammar = GRID_S2, GRAMMAR_S2
    elif phase == 'S5':
        skels, grammar = GRID_S5, GRAMMAR_S2
    elif task == 'grid1':
        skels, grammar = GRID_SINGLES_S0, GRAMMAR_S0
    elif mode == 'comp_dev':
        skels, grammar = GRID_PAIRS_DEV, GRAMMAR_S0
    elif mode == 'comp_test':
        skels, grammar = GRID_PAIRS_TEST, GRAMMAR_S0
    elif phase == 'S1':
        skels, grammar = GRID_PAIRS_S1, GRAMMAR_S0
    elif phase == 'S6':
        skels, grammar = GRID_PAIRS_S0 + GRID_PAIRS_S1, GRAMMAR_S0
    else:
        skels, grammar = GRID_PAIRS_S0, GRAMMAR_S0
    if phase in ('S3', 'S3c'):                                  # dense largest (S3) vs its counterpart smallest
        skels = [('sel',)] if task == 'grid1' else [('sel', 'geo')]
    if mode == 'newparam':                                      # panel only: force a held-out parameter value
        oned = GRID_RUNGS[rung][0] == '1d'                      # one-row grids have no held-out offset
        skels = [s for s in skels if 'recol' in s or ('trans' in s and not oned)]
    sk = skels[int(rng.integers(len(skels)))]
    if mode == 'newparam':
        ok = lambda t, p: param_is_new(t, p) if t in ('recol', 'trans') else True
    else:
        ok = lambda t, p: not param_is_new(t, p)
    if phase in ('S3', 'S3c'):
        want_sel = 'largest' if phase == 'S3' else 'smallest'; ok0 = ok
        ok = lambda t, p: ok0(t, p) and (t != 'sel' or p == want_sel)
    ep = G.make_episode(rng, sk, GRID_RUNGS[rung], grammar, param_ok=ok, max_tries=60)
    if ep is None: return None
    if mode == 'newparam' and not any(param_is_new(t, p) for t, p in ep['prog']): return None
    cols = COL
    prompt = ['<grid>']
    if phase == 'S4':                                           # reserved colour vocabulary with an explicit legend
        perm = [int(x) for x in rng.permutation(10)]
        cols = [ALT_COL[perm[i]] for i in range(10)]
        prompt += ['<legend>']
        for c in range(10): prompt += [cols[c], COL[c]]
    extra = []
    for k, (a, b) in enumerate(ep['demos']):
        prompt += ['<in>'] + grid_pieces(a, cols) + ['<out>']
        s = len(prompt); prompt += grid_pieces(b, cols)
        if k >= 1 and ep['determined'][k - 1]: extra.append((s, len(prompt)))
    prompt += ['<in>'] + grid_pieces(ep['query'], cols) + ['<out>']
    target = grid_pieces(ep['out'], cols) + ['<eos>']
    key = 'grid|' + '+'.join(sk) + '|' + G.canon(ep)
    prog = [(t, (sorted(p.items()) if isinstance(p, dict) else p)) for t, p in ep['prog']]
    return ex(prompt, target, key, task, rung, extra, skeleton='+'.join(sk), prog=str(prog), hw=ep['hw'],
              phase=phase, mode=mode)


# ------------------------------------------------------------------------------------------------ programs family
ARITH_RUNGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]                 # executed operations; 8 trained
LIST_RUNGS = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]                 # list length; 8 trained
REG_RUNGS = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]                  # executed assignments; 8 trained
GRAPH_RUNGS = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]                # nodes; 8 trained
LIST_SINGLES = ['min', 'max', 'sort', 'rsort', 'filter']
LIST_COMPS_S0 = [('filter', 'min'), ('filter', 'sort')]
LIST_COMPS_S1 = [('filter', 'max')]
LIST_COMPS_TEST = [('filter', 'rsort')]
VARS_S0 = list('abcdef')
VARS_S4 = list('pqrstu')                                         # S4 rendering: reserved variable names
LETTERS = list('ABCDEFGHIJKLMNOP')


def gen_arith(rng, task, rung, mode='train', phase='S0'):
    n = ARITH_RUNGS[rung]
    ops = ['+', '-']
    if phase == 'S2': ops = ['*', '/']
    if phase == 'S5': ops = ['+', '-', '*', '/']
    if phase == 'S3': ops = ['+']
    if phase == 'S3c': ops = ['-']
    for _ in range(40):
        v = int(rng.integers(-20, 21)); prompt = ['<arith>'] + num_in(v); ok = True; key = [str(v)]
        for _ in range(n):
            op = choice(rng, ops)
            if op in '+-':
                b = int(rng.integers(-20, 21)); v2 = v + b if op == '+' else v - b
            elif op == '*':
                b = int(rng.integers(-9, 10)); v2 = v * b
            else:
                divs = [d for d in range(-12, 13) if d not in (0, 1, -1) and v % d == 0] or [1]
                b = choice(rng, divs) if v != 0 else int(choice(rng, [2, 3, -2, 5])); v2 = v // b
            if abs(v2) > 999: ok = False; break
            prompt += ['<' + {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op] + '>'] + num_in(b); v = v2
            key.append(op + str(b))
        if ok:
            return ex(prompt + ['<ans>'], num_out(v) + ['<eos>'], 'arith|' + ' '.join(key), task, rung, answer=v,
                      phase=phase, mode=mode)
    return None


def list_apply(op, xs, k):
    if op == 'min': return min(xs) if xs else None
    if op == 'max': return max(xs) if xs else None
    if op == 'sort': return sorted(xs)
    if op == 'rsort': return sorted(xs, reverse=True)
    if op == 'filter': return [x for x in xs if x > k]
    raise ValueError(op)


def gen_list(rng, task, rung, mode='train', phase='S0'):
    n = LIST_RUNGS[rung]
    if mode == 'comp_test': prog = choice(rng, LIST_COMPS_TEST)
    elif phase == 'S1': prog = choice(rng, LIST_COMPS_S1)
    elif phase == 'S3': prog = ('max',)
    elif phase == 'S3c': prog = ('min',)
    elif phase == 'S6': prog = choice(rng, [(s,) for s in LIST_SINGLES] + LIST_COMPS_S0 + LIST_COMPS_S1)
    else: prog = choice(rng, [(s,) for s in LIST_SINGLES] + LIST_COMPS_S0)
    xs = [int(x) for x in rng.integers(0, 100, size=n)]
    if rng.random() < 0.3 and n >= 3:                            # duplicates sampled explicitly (§5.3)
        xs[int(rng.integers(n))] = xs[int(rng.integers(n))]
    k = int(rng.integers(10, 90))
    val = xs
    for op in prog:
        val = list_apply(op, val, k)
        if val is None: return None
    if 'filter' in prog and (len(list_apply('filter', xs, k)) == 0): return None
    prompt = ['<list>'] + ['<' + op + '>' for op in prog]
    if 'filter' in prog: prompt += ['<gt>'] + num_in(k)
    prompt += ['[']
    for i, x in enumerate(xs): prompt += ([','] if i else []) + num_in(x)
    prompt += [']', '<ans>']
    if isinstance(val, list):
        target = ['[']
        for i, x in enumerate(val): target += ([','] if i else []) + num_out(x)
        target += [']']
    else:
        target = num_out(val)
    return ex(prompt, target + ['<eos>'], f'list|{"+".join(prog)}|{k}|{xs}', task, rung, prog='+'.join(prog),
              phase=phase, mode=mode)


def gen_reg(rng, task, rung, mode='train', phase='S0'):
    """Register programs: assignment, add/sub, one conditional, bounded repeat; executed assignments = the rung."""
    n = REG_RUNGS[rung]
    vars_ = VARS_S4 if phase == 'S4' else VARS_S0
    for _ in range(40):
        env = {}; prog = []; done = 0; ok = True
        n_init = min(n, int(rng.integers(2, 4)))
        for v in rng.permutation(vars_)[:n_init]:
            c = int(rng.integers(0, 10)); env[str(v)] = c; prog.append([str(v), '='] + num_in(c)); done += 1
        used_if = False
        while done < n and ok:
            names = list(env)
            x = choice(rng, vars_ if rng.random() < 0.3 else names); y = choice(rng, names)
            op = choice(rng, ['+', '-'])
            if rng.random() < 0.5: z, zv = None, int(rng.integers(1, 10))
            else: z = choice(rng, names); zv = env[z]
            sign = 1 if op == '+' else -1
            kind = rng.random()
            if kind < 0.2 and n - done >= 2 and x in env:            # bounded repeat: x = x op c, r times
                r = int(rng.integers(2, min(3, n - done) + 1)); c = int(rng.integers(1, 10))
                val = env[x] + sign * c * r
                if abs(val) > 999: ok = False; break
                env[x] = val; prog.append(['<rep>'] + num_in(r) + ['<{>', x, '=', x, op] + num_in(c) + ['<}>']); done += r
                continue
            rhs = [y, op] + ([z] if z else num_in(zv))
            val = env[y] + sign * zv
            if abs(val) > 999: ok = False; break
            if kind < 0.35 and not used_if and len(names) >= 2:     # one conditional
                a, b = choice(rng, names), choice(rng, names)
                prog.append(['<if>', a, '>', b, '<then>', x, '='] + rhs + ['<end>']); used_if = True
                if env[a] > env[b]: env[x] = val; done += 1
            else:
                env[x] = val; prog.append([x, '='] + rhs); done += 1
        if not ok or done != n: continue
        ret = choice(rng, list(env))
        prompt = ['<reg>']
        for i, s in enumerate(prog): prompt += ([';'] if i else []) + s
        prompt += ['<ret>', ret, '<ans>']
        key = 'reg|' + ' '.join(' '.join(s) for s in prog) + '|' + ret
        return ex(prompt, num_out(env[ret]) + ['<eos>'], key, task, rung, answer=env[ret], phase=phase, mode=mode)
    return None


def bfs_dist(adj, s):
    d = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in d: d[v] = d[u] + 1; q.append(v)
    return d


def gen_graph(rng, task, rung, mode='train', phase='S0'):
    """Undirected graphs; query kinds reach (yes/no, balanced), dist, path (any shortest path is accepted)."""
    n = GRAPH_RUNGS[rung]
    qk = choice(rng, ['reach', 'dist', 'path'])
    labels = [LETTERS[i] for i in rng.permutation(16)[:n]]
    want_yes = qk != 'reach' or rng.random() < 0.5
    for _ in range(40):
        comp = np.zeros(n, dtype=int)
        if not want_yes or rng.random() < 0.3:
            comp = (rng.random(n) < 0.5).astype(int); comp[0], comp[-1] = 0, 1
        edges = set()
        for c in (0, 1):
            mem = [i for i in range(n) if comp[i] == c]
            for j in range(1, len(mem)):                            # random tree per component
                a, b = mem[j], mem[int(rng.integers(j))]; edges.add((min(a, b), max(a, b)))
            for _ in range(int(rng.integers(0, max(1, len(mem) // 2) + 1))):
                if len(mem) >= 2:
                    a, b = rng.choice(mem, 2, replace=False); edges.add((int(min(a, b)), int(max(a, b))))
        adj = {i: set() for i in range(n)}
        for a, b in edges: adj[a].add(b); adj[b].add(a)
        s, t = [int(x) for x in rng.choice(n, 2, replace=False)]
        d = bfs_dist(adj, s)
        reach = t in d
        if qk == 'reach' and reach != want_yes: continue
        if qk != 'reach' and not reach: continue
        el = list(edges); order = rng.permutation(len(el))
        prompt = ['<graph>']
        for i, j in enumerate(order):
            a, b = el[int(j)]
            if rng.random() < 0.5: a, b = b, a
            prompt += ([','] if i else []) + [labels[a], labels[b]]
        prompt += ['<' + qk + '>', labels[s], labels[t], '<ans>']
        if qk == 'reach': target = ['<yes>' if reach else '<no>']
        elif qk == 'dist': target = num_out(d[t])
        else:                                                       # canonical shortest path: BFS parents, label order
            par = {s: None}; q = deque([s])
            while q:
                u = q.popleft()
                for v in sorted(adj[u], key=lambda i: labels[i]):
                    if v not in par: par[v] = u; q.append(v)
            path = [t]
            while path[-1] != s: path.append(par[path[-1]])
            target = [labels[i] for i in path[::-1]]
        canon = sorted(tuple(sorted((labels[a], labels[b]))) for a, b in edges)
        key = f'graph|{qk}|{canon}|{labels[s]}{labels[t]}'
        return ex(prompt, target + ['<eos>'], key, task, rung, qkind=qk, edges=[(labels[a], labels[b]) for a, b in edges],
                  src=labels[s], dst=labels[t], dist=d.get(t), phase=phase, mode=mode)
    return None


# ------------------------------------------------------------------------------------------------ state family
NAMES_S0 = ['Mira', 'Jo', 'Sam', 'Lily', 'Tom', 'Ben', 'Anna', 'Max', 'Sue', 'Tim', 'Kate', 'Leo', 'Nina', 'Ray',
            'Zoe', 'Dan', 'Eva', 'Finn', 'Ivy', 'Jack', 'Lucy', 'Mia', 'Ned', 'Owen']
NAMES_S4 = ['Quill', 'Vesna', 'Yorick', 'Wren', 'Xander', 'Ulla', 'Pavo', 'Odile']   # reserved for S4
PLACES = ['kitchen', 'hall', 'shed', 'garden', 'attic', 'barn', 'park', 'yard', 'cellar', 'porch', 'school', 'shop']
OBJECTS = ['key', 'ball', 'book', 'cup', 'hat', 'coin', 'lamp', 'box', 'pen', 'bell', 'kite', 'map']
STATE_RUNGS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 40]                 # events; 8 trained
STATE_TASKS_S0 = ['loc', 'owner', 'count']


def gen_state(rng, task, rung, mode='train', phase='S0'):
    """Entities, places, objects; events move / take / drop / give (give needs co-location). Queries: loc, owner,
    count; 'link' (S1): where is the person who has X. S3: owner queries densely (counterpart: loc)."""
    n_ev = STATE_RUNGS[rung]
    names = NAMES_S4 if phase == 'S4' else NAMES_S0
    for _ in range(60):
        P = int(rng.integers(2, 5)); K = int(rng.integers(2, 5)); L = int(rng.integers(3, 6))
        ppl = [names[i] for i in rng.permutation(len(names))[:P]]
        objs = [OBJECTS[i] for i in rng.permutation(len(OBJECTS))[:K]]
        locs = [PLACES[i] for i in rng.permutation(len(PLACES))[:L]]
        at = {p: choice(rng, locs) for p in ppl}
        oat = {o: choice(rng, locs) for o in objs}; holder = {o: None for o in objs}
        facts = [('at', p, at[p]) for p in ppl] + [('oat', o, oat[o]) for o in objs]
        events = []
        for _ in range(n_ev * 4):
            if len(events) == n_ev: break
            kind = choice(rng, ['move', 'move', 'take', 'drop', 'give'])
            p = choice(rng, ppl)
            if kind == 'move':
                dest = choice(rng, [l for l in locs if l != at[p]]); at[p] = dest; events.append(('move', p, dest))
            elif kind == 'take':
                cands = [o for o in objs if holder[o] is None and oat[o] == at[p]]
                if not cands: continue
                o = choice(rng, cands); holder[o] = p; events.append(('take', p, o))
            elif kind == 'drop':
                cands = [o for o in objs if holder[o] == p]
                if not cands: continue
                o = choice(rng, cands); holder[o] = None; oat[o] = at[p]; events.append(('drop', p, o))
            else:
                cands = [o for o in objs if holder[o] == p]
                others = [q for q in ppl if q != p and at[q] == at[p]]
                if not cands or not others: continue
                o = choice(rng, cands); q = choice(rng, others); holder[o] = q; events.append(('give', p, o, q))
        if len(events) != n_ev: continue
        where = lambda o: at[holder[o]] if holder[o] else oat[o]
        qkind = {'S1': 'link', 'S3': 'owner', 'S3c': 'loc'}.get(phase) or choice(rng, STATE_TASKS_S0)
        if phase == 'S6': qkind = choice(rng, STATE_TASKS_S0 + ['link'])
        if qkind == 'loc':
            x = choice(rng, ppl + objs); ans = at[x] if x in at else where(x)
        elif qkind == 'owner':
            x = choice(rng, objs); ans = holder[x] or 'nobody'
        elif qkind == 'count':
            x = choice(rng, ppl); ans = sum(holder[o] == x for o in objs)
        else:
            held = [o for o in objs if holder[o]]
            if not held: continue
            x = choice(rng, held); ans = at[holder[x]]
        english = h64('render|' + str(facts) + str(events)) % 2 == 0
        prompt = ['<state>']
        def sent(s): prompt.append(s)
        for f in facts:
            if english: sent(f'{f[1]} is in the {f[2]}.' if f[0] == 'at' else f'The {f[1]} is in the {f[2]}.')
            else: prompt.extend([f[1], '<at>', f[2], ';'])
        for e in events:
            if english:
                if e[0] == 'move': sent(f'{e[1]} moves to the {e[2]}.')
                elif e[0] == 'take': sent(f'{e[1]} takes the {e[2]}.')
                elif e[0] == 'drop': sent(f'{e[1]} drops the {e[2]}.')
                else: sent(f'{e[1]} gives the {e[2]} to {e[3]}.')
            else:
                prompt.extend([e[1], '<' + e[0] + '>', e[2]] + ([e[3]] if len(e) > 3 else []) + [';'])
        if english:
            sent({'loc': f'Where is {"the " if x in objs else ""}{x}?', 'owner': f'Who has the {x}?',
                  'count': f'How many things does {x} have?', 'link': f'Where is the one who has the {x}?'}[qkind])
        else:
            prompt.extend(['<q_' + qkind + '>', x])
        prompt.append('<ans>')
        target = (num_out(ans) if qkind == 'count' else [ans]) + ['<eos>']
        # semantic key: the world with names/objects/places canonicalised by first appearance (§6.1)
        ren = {}
        for tok in [z for f in facts for z in f[1:]] + [z for e in events for z in e[1:]] + [x]:
            ren.setdefault(tok, f'e{len(ren)}')
        key = f'state|{qkind}|' + ';'.join(','.join([f[0]] + [ren[z] for z in f[1:]]) for f in facts) + '|' + \
              ';'.join(','.join([e[0]] + [ren[z] for z in e[1:]]) for e in events) + '|' + ren[x]
        return ex(prompt, target, key, task, rung, qkind=qkind, english=english, answer=str(ans), phase=phase, mode=mode)
    return None


# ------------------------------------------------------------------------------------------------ planning family
PLAN_RUNGS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 13), (14, 16), (17, 20), (21, 25), (26, 30)]  # 8 trained
ACTS = ['<U>', '<D>', '<L>', '<R>']
ALT_ACTS = ['<a0>', '<a1>', '<a2>', '<a3>']
MOVES = {'<U>': (-1, 0), '<D>': (1, 0), '<L>': (0, -1), '<R>': (0, 1)}
CELL = {'.': '<.>', '#': '<#>', 'S': '<S>', 'A': '<A>', 'B': '<B>', 'C': '<C>', 'k': '<kk>', 'D': '<DD>'}


def plan_bfs(m, start, goals, keydoor=False):
    """Exact shortest plan over (position, goals reached, key held, door open). Returns the canonical action list."""
    h, w = m.shape
    s0 = (start, 0, False, False); par = {s0: None}; q = deque([s0])
    while q:
        st = q.popleft(); (r, c), k, hk, op = st
        if k == len(goals):
            acts = []
            while par[st] is not None: st, a = par[st]; acts.append(a)
            return acts[::-1]
        nxt = []
        for a in ACTS:
            dr, dc = MOVES[a]; rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or m[rr, cc] == '#': continue
            if m[rr, cc] == 'D' and not op: continue
            kk = k + 1 if (rr, cc) == goals[k] else k
            nxt.append((a, ((rr, cc), kk, hk, op)))
        if keydoor:
            if m[r, c] == 'k' and not hk: nxt.append(('<pick>', ((r, c), k, True, op)))
            if hk and not op and any(0 <= r + dr < h and 0 <= c + dc < w and m[r + dr, c + dc] == 'D'
                                     for dr, dc in MOVES.values()):
                nxt.append(('<open>', ((r, c), k, hk, True)))
        for a, ns in nxt:
            if ns not in par: par[ns] = (st, a); q.append(ns)
    return None


def plan_simulate(m, start, goals, acts, keydoor=False, legend=None):
    """Execute a generated plan; success iff every goal is reached in order and the plan ends on the last goal."""
    h, w = m.shape; (r, c) = start; k = 0; hk = False; op = False
    for a in acts:
        if legend: a = legend.get(a, a)
        if a in MOVES:
            dr, dc = MOVES[a]; rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or m[rr, cc] == '#' or (m[rr, cc] == 'D' and not op): return False
            r, c = rr, cc
            if k < len(goals) and (r, c) == goals[k]: k += 1
        elif a == '<pick>' and keydoor and m[r, c] == 'k': hk = True
        elif a == '<open>' and keydoor and hk and any(0 <= r + dr < h and 0 <= c + dc < w and m[r + dr, c + dc] == 'D'
                                                      for dr, dc in MOVES.values()): op = True
        else: return False
    return k == len(goals) and (r, c) == goals[-1]


def gen_plan(rng, task, rung, mode='train', phase='S0'):
    """task 'plan': one goal (nav) or two ordered goals (A then B / B then A). S1: three ordered goals; S2: key/door;
    S5: key/door with two ordered goals; S4: action tokens renamed with a legend; S3: A then B only (counterpart B, A)."""
    lo, hi = PLAN_RUNGS[rung]
    kind = {'S1': 'order3', 'S2': 'keydoor', 'S5': 'keydoor2'}.get(phase) or choice(rng, ['nav', 'order2'])
    if phase in ('S3', 'S3c'): kind = 'order2'
    if phase == 'S6': kind = choice(rng, ['nav', 'order2', 'order3'])
    keydoor = kind.startswith('keydoor')
    side_lo, side_hi = (4, 8) if rung < 8 else (7, 10)
    if keydoor: side_lo = max(side_lo, 5)
    for _ in range(80):
        h, w = int(rng.integers(side_lo, side_hi + 1)), int(rng.integers(side_lo, side_hi + 1))
        m = np.where(rng.random((h, w)) < rng.uniform(0.1, 0.3), '#', '.').astype('<U1')
        if keydoor:                                               # a wall column with one door; the goal behind it
            col = int(rng.integers(2, w - 1)); m[:, col] = '#'; dr_ = int(rng.integers(h)); m[dr_, col] = 'D'
            left = [(r, c) for r in range(h) for c in range(col) if m[r, c] == '.']
            right = [(r, c) for r in range(h) for c in range(col + 1, w) if m[r, c] == '.']
            if len(left) < 3 or len(right) < 2: continue
            pick = rng.permutation(len(left))
            start, kpos = left[pick[0]], left[pick[1]]; m[kpos] = 'k'
            gl = [right[i] for i in rng.permutation(len(right))[:(2 if kind == 'keydoor2' else 1)]]
            if kind == 'keydoor2' and len(gl) < 2: continue
        else:
            free = [(r, c) for r in range(h) for c in range(w) if m[r, c] == '.']
            ng = {'nav': 1, 'order2': 2, 'order3': 3}[kind]
            if len(free) < ng + 1: continue
            pick = rng.permutation(len(free))
            start = free[pick[0]]; gl = [free[pick[i]] for i in range(1, ng + 1)]
        names = ['A', 'B', 'C'][:len(gl)]
        for (r, c), nm in zip(gl, names): m[r, c] = nm
        m[start] = 'S'
        order = list(range(len(gl)))
        if kind == 'order2' and phase != 'S3' and (phase == 'S3c' or rng.random() < 0.5): order = [1, 0]
        goals = [gl[i] for i in order]
        acts = plan_bfs(m, start, goals, keydoor)
        if acts is None or not (lo <= len(acts) <= hi): continue
        acts_out = acts; legend = None; prompt = ['<plan>']
        if phase == 'S4':
            perm = [int(x) for x in rng.permutation(4)]
            fwd = {ACTS[i]: ALT_ACTS[perm[i]] for i in range(4)}; legend = {v: k for k, v in fwd.items()}
            prompt += ['<legend>'] + [z for a in ACTS for z in (fwd[a], a)]
            acts_out = [fwd.get(a, a) for a in acts]
        prompt += num_in(h) + ['<x>'] + num_in(w)
        for r in range(h): prompt += ['<nl>'] + [CELL[ch] for ch in m[r]]
        prompt += ['<goto>'] + [z for j, i in enumerate(order) for z in (['<then>'] if j else []) + [CELL[names[i]]]]
        prompt += ['<ans>']
        key = f'plan|{kind}|' + ''.join(m.flatten()) + f'|{h}x{w}|{order}'
        return ex(prompt, acts_out + ['<eos>'], key, task, rung, kind=kind, map=m.tolist(), start=start, goals=goals,
                  optimal=len(acts), keydoor=keydoor, legend=legend, phase=phase, mode=mode)
    return None


# ------------------------------------------------------------------------------------------------ registry
# task -> (family, generator, rungs, trained rungs, needs generation at evaluation)
TASKS = {
    'grid1': ('grid', gen_grid, len(GRID_RUNGS), 8, False),
    'grid2': ('grid', gen_grid, len(GRID_RUNGS), 8, False),
    'arith': ('prog', gen_arith, len(ARITH_RUNGS), 8, False),
    'list': ('prog', gen_list, len(LIST_RUNGS), 8, False),
    'reg': ('prog', gen_reg, len(REG_RUNGS), 8, False),
    'graph': ('prog', gen_graph, len(GRAPH_RUNGS), 8, True),
    'state': ('state', gen_state, len(STATE_RUNGS), 8, False),
    'plan': ('plan', gen_plan, len(PLAN_RUNGS), 8, True),
}
FAMILY_TASKS = {'grid': ['grid1', 'grid2'], 'prog': ['arith', 'list', 'reg', 'graph'], 'state': ['state'],
                'plan': ['plan']}
# §5.7 inductive-bias tags, fixed before any run from the task definitions (favoured by; copy versus compute)
TAGS = {'grid1': ('neither known', 'copy-heavy'), 'grid2': ('neither known', 'copy-heavy'),
        'arith': ('neither known', 'compute'), 'list': ('attention', 'copy-heavy'),
        'reg': ('recurrence', 'compute'), 'graph': ('neither known', 'mixed'),
        'state': ('attention (binding) and recurrence (long histories)', 'copy'),
        'plan': ('neither known', 'compute')}


def generate(task, rung, rng, mode='train', phase='S0', split_ns='v13s0', want=None, max_tries=200):
    """One accepted example of `task` at `rung` whose semantic key falls in split `want` (default: by mode)."""
    fam, fn, *_ = TASKS[task]
    want = want or ('train' if mode == 'train' else 'test')
    for _ in range(max_tries):
        e = fn(rng, task, rung, mode=mode, phase=phase)
        if e is None: continue
        if want != 'any' and bucket(e['key'], split_ns) != want: continue   # 'any': disjoint from training by construction
        e['meta']['family'] = fam
        return e
    return None
