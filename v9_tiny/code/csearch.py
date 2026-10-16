"""Concept search (draft §9.3-§9.4, I15): RH(H) receding horizon, and one-tree search G(R) / IMAG, over K codes +
STOP, lockstep over B problems, full recomputation through hgen.Engine primitives (no caches).

Children are imagined (c~_{j+1} = chat_j^(k), one core pass from the nearest grounded ancestor's concepts) or grounded
(decode the pending chunks under their stored plan records, re-encode). A plan record is immutable: once a chunk has
been decoded from a record, every later decode of that prefix applies the same record (node.over). Leaf values:
'rollout' = verified greedy grounded completion with selector-argmax codes; 'value' = V_psi(u0). STOP: verified at
grounded nodes, tau_psi at imagined nodes only if the terminal head is trusted. RH executes the most-visited root action
(ties -> higher prior -> lower index); one-tree search ends at a verified accepted STOP or when the budget is spent, then
decodes the most-visited path and completes it greedily. Every verifier call and node is counted.
"""
import gc
import time
from functools import wraps
import numpy as np
import torch
from torch import nn
import v9paths
from model import D
from data import DONE, SEP, verify
import hgen
from hgen import CH, Item, classify, stop_legal

C_PUCT, DIR_ALPHA, DIR_W = 1.5, .3, .25
DONE_CHUNK = (DONE, SEP, SEP, SEP)


def _release_search_memory(method):
    """Reclaim completed MPS trees after their stack frame has released local refs.

    Nodes link parents and children, so refcounting alone retains their tensors.
    Variable prefix/batch shapes also leave unused buffers in the MPS allocator.
    Run only at a completed search boundary; labels/records are already on CPU.
    """
    @wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            if self.dev.type == 'mps':
                gc.collect()
                torch.mps.empty_cache()
    return run


class Heads(nn.Module):
    """V_psi and tau_psi: 128 -> 1 each on the code-free u^(0) (§8.2), versioned per S4 iteration."""
    def __init__(self):
        super().__init__(); self.v = nn.Linear(D, 1); self.t = nn.Linear(D, 1)
        self.v_trusted = True; self.t_trusted = True

    def value(self, u):
        return torch.sigmoid(self.v(u.float())).squeeze(-1)

    def term(self, u):
        return torch.sigmoid(self.t(u.float())).squeeze(-1)


class Node:
    __slots__ = ('parent', 'action', 'depth', 'children', 'N', 'W', 'prior', 'grounded', 'tokens', 'codes', 'over', 'ctx', 'base', 'preds', 'recs',
                 'u0', 'chatk', 'ak', 'value', 'terminal', 'stop_value', 'cur', 'hops', 'stop_ok')

    def __init__(self, parent, action, depth, K):
        self.parent, self.action, self.depth = parent, action, depth
        self.children = {}; self.N = np.zeros(K + 1); self.W = np.zeros(K + 1); self.prior = None
        self.grounded = False; self.tokens = None; self.codes = None; self.over = {}; self.ctx = None; self.base = None; self.preds = []; self.recs = []
        self.u0 = self.chatk = self.ak = None; self.value = None; self.terminal = False; self.stop_value = None
        self.cur = self.hops = None; self.stop_ok = True


class Problem:
    def __init__(self, item, hop_limit, seed):
        self.item = item; self.P = item.P; self.limit = hop_limit; self.chunks = []; self.codes = []
        self.done = False; self.success = False; self.reason = None; self.cur = item.start; self.hops = 0
        self.rng = np.random.RandomState(seed); self.episode = 0
        self.roots = []          # executed root records (RH): u0, visit distribution, prior, stop label, chosen action
        self.verifier_calls = 0; self.nodes = 0; self.executed_tokens = 0; self.rollout_states = []; self.answer = None

    def tokens(self):
        return np.concatenate([self.item.prompt] + [np.array(c) for c in self.chunks]).astype(np.int64) if self.chunks else np.array(self.item.prompt, dtype=np.int64)

    def code_array(self):
        a = np.full(self.limit + len(self.item.prompt) // CH + 8, -1, dtype=np.int64)
        J0 = self.P // CH - 1
        for i, k in enumerate(self.codes):
            a[J0 + i] = k
        return a


class Search:
    """cfg: H (depth cap; one-tree uses the hop limit), children 'imagined'|'grounded', ground_every R (one-tree G(R);
    None = IMAG), leaf 'rollout'|'value', budget (expansions per search), noise (root Dirichlet), heads, one_tree."""

    def __init__(self, engine, cfg):
        self.e = engine; self.hs = engine.hs; self.K = engine.K; self.cfg = cfg; self.dev = engine.device
        self.heads = cfg.get('heads')
        self.stats = dict(expansions=0, imagined=0, grounded=0, replayed_chunks=0, rollouts=0, rollout_chunks=0, verifier_calls=0)

    # ---------------------------------------------------------------- priors and heads
    def stop_prob(self, u0):
        if self.heads is None:
            return torch.full((len(u0),), 1. / (self.K + 1), device=u0.device)
        return self.heads.term(u0)

    def leaf_value(self, u0):
        return self.heads.value(u0)

    def set_heads(self, node, u0, pi, chatk, ak, t):
        """pi: selector probabilities (numpy row), t: stop probability (float); converted once per batch by the caller."""
        node.u0 = u0; node.chatk = chatk; node.ak = ak
        node.stop_ok = node.grounded or (self.heads is not None and self.heads.t_trusted)
        node.prior = np.r_[(1 - t) * pi, t] if node.stop_ok else np.r_[pi, 0.]

    def ground_rows(self, nodes):
        """Grounded heads for nodes holding tokens (batched state_heads); stores each node's grounded concept context."""
        x, docs, L = self.e.batch([n.tokens for n in nodes])
        st = self.e.state_heads(x, docs, L)
        tau = self.stop_prob(st['u0']).float().cpu().numpy(); pi = torch.softmax(st['pi_logits'].float(), -1).cpu().numpy()   # one sync per batch
        for i, n in enumerate(nodes):
            self.set_heads(n, st['u0'][i], pi[i], st['chatk'][i], st['ak'][i], float(tau[i]))
            n.ctx = dict(J=int(st['J'][i]), c=st['c'][i])
        return st

    def imagine_rows(self, nodes):
        """Imagined nodes: concept inputs = the grounded base's concepts + the committed predictions since the base."""
        for depth in sorted({len(n.preds) for n in nodes}):
            group = [n for n in nodes if len(n.preds) == depth]
            J = torch.tensor([n.base.ctx['J'] for n in group], device=self.dev); M = int(J.max()) + depth + 1
            c_in = torch.zeros(len(group), M, D, device=self.dev); cd_in = torch.full((len(group), M), -1, dtype=torch.long, device=self.dev)
            for i, n in enumerate(group):
                j = n.base.ctx['J']; c_in[i, :j + 1] = n.base.ctx['c'][:j + 1]; cd_in[i, :j + 1] = 0
                c_in[i, j + 1:j + 1 + depth] = torch.stack(n.preds); cd_in[i, j + 1:j + 1 + depth] = 0
            H = self.hs.heads(c_in, cd_in); idx = J + depth; at = hgen.Engine.at
            u0 = at(H['u0'], idx); tau = self.stop_prob(u0).float().cpu().numpy(); pi = torch.softmax(at(H['pi_logits'], idx).float(), -1).cpu().numpy()
            chatk, ak = at(H['chatk'], idx), at(H['ak'], idx)
            for i, n in enumerate(group):
                self.set_heads(n, u0[i], pi[i], chatk[i], ak[i], float(tau[i]))
            self.stats['imagined'] += len(group)

    # ---------------------------------------------------------------- decoding under plan records
    def code_matrix(self, rows_codes, width):
        cm = torch.full((len(rows_codes), width), -1, dtype=torch.long, device=self.dev)
        for i, c in enumerate(rows_codes):
            ca = torch.as_tensor(c[:width], device=self.dev); cm[i, :len(ca)] = ca
        return cm

    @torch.no_grad()
    def decode_over(self, x, docs, L, codes, overs):
        """Greedy-decode the next chunk of every row under plan records: codes per chunk, plus immutable overrides
        {chunk j: (chat, a)} for chunks decoded from imagined records. Without overrides this is Engine.decode_chunk."""
        if not any(overs):
            return self.e.decode_chunk(x, docs, L, codes)
        hs = self.hs; h, c, cd = self.e.context(x, docs); M = c.shape[1]
        H = hs.heads(c, cd, need_codes=bool((codes[:, :M] >= 0).any()))
        chat, a = hs.plan(H, codes[:, :M]); chat = chat.clone(); a = a.clone()
        for r, ov in enumerate(overs):
            for j, (ch_, a_) in ov.items():
                if j < M:
                    chat[r, j] = ch_; a[r, j] = a_
        rows = torch.arange(len(x), device=x.device)
        for i in range(CH):
            if i:
                h = self.f_encode(x, docs)
            tok = self.e.logits_at(self.e.hidden(h, chat, a, docs), L + i - 1).argmax(-1)
            x[rows, L + i] = tok; docs[rows, L + i] = 0
        return x.gather(1, (L[:, None] + torch.arange(CH, device=x.device)[None]))

    def f_encode(self, x, docs):
        return self.hs.f.encode(x, docs)

    def finish_child(self, pb, parent_cur, parent_hops, n, ch):
        cl = classify(pb.item, parent_cur, parent_hops, ch)
        if cl['kind'] == 'done':
            n.terminal = True; n.value = float(self.verify_tokens(pb, n.tokens)); n.cur, n.hops = parent_cur, parent_hops
        elif not cl['legal'] or parent_hops + 1 >= pb.limit:
            n.terminal = True; n.value = 0.; pb.verifier_calls += 1; self.stats['verifier_calls'] += 1   # an illegal prefix is a verified failure
            n.cur, n.hops = parent_cur, parent_hops + 1
        else:
            n.cur, n.hops = cl['node'], parent_hops + 1
        return cl

    def decode_children(self, problems, parents, actions):
        """Grounded children of grounded parents: decode the next chunk under code k at the parent's last chunk."""
        x, docs, L = self.e.batch([p.tokens for p in parents])
        codes = self.code_matrix([p.codes for p in parents], x.shape[1] // CH + 1)
        for i, p in enumerate(parents):
            codes[i, len(p.tokens) // CH - 1] = actions[i]
        chunk = self.decode_over(x, docs, L, codes, [p.over for p in parents]).cpu().numpy()
        out = []
        for i, (pb, p) in enumerate(zip(problems, parents)):
            ch = tuple(int(v) for v in chunk[i]); n = Node(p, actions[i], p.depth + 1, self.K); n.grounded = True; n.over = p.over
            n.tokens = np.r_[p.tokens, chunk[i]].astype(np.int64); n.codes = p.codes.copy(); n.codes[len(p.tokens) // CH - 1] = actions[i]
            self.finish_child(pb, p.cur, p.hops, n, ch); out.append(n)
        self.stats['grounded'] += len(out)
        return out

    def replay(self, problems, nodes):
        """Ground imagined nodes (§9.4): from each node's grounded base, decode the pending chunks in order under their
        stored plan records (immutable overrides), re-encode. A pending chunk that is illegal or DONE ends the node."""
        groups = {}
        for pb, n in zip(problems, nodes):
            groups.setdefault(len(n.recs), []).append((pb, n))
        for npend, grp in groups.items():
            state = [dict(tokens=n.base.tokens.copy(), codes=n.base.codes.copy(), over=dict(n.base.over), cur=n.base.cur, hops=n.base.hops, alive=True) for _, n in grp]
            path_codes = []
            for _, n in grp:
                acts = []; m = n
                while m is not n.base:
                    acts.append(m.action); m = m.parent
                path_codes.append(acts[::-1])
            for t in range(npend):
                live = [i for i, s in enumerate(state) if s['alive']]
                if not live:
                    break
                x, docs, L = self.e.batch([state[i]['tokens'] for i in live]); codes = self.code_matrix([state[i]['codes'] for i in live], x.shape[1] // CH + 1)
                overs = []
                for r, i in enumerate(live):
                    J = len(state[i]['tokens']) // CH - 1; k = path_codes[i][t]; codes[r, J] = k; state[i]['codes'][J] = k
                    state[i]['over'][J] = grp[i][1].recs[t]; overs.append(state[i]['over'])
                chunk = self.decode_over(x, docs, L, codes, overs).cpu().numpy(); self.stats['replayed_chunks'] += len(live)
                for r, i in enumerate(live):
                    pb, n = grp[i]; ch = tuple(int(v) for v in chunk[r]); s = state[i]
                    s['tokens'] = np.r_[s['tokens'], chunk[r]].astype(np.int64)
                    tmp = Node(None, None, 0, self.K); tmp.tokens = s['tokens']
                    cl = self.finish_child(pb, s['cur'], s['hops'], tmp, ch)
                    if tmp.terminal:
                        n.terminal = True; n.value = tmp.value; s['alive'] = False
                    else:
                        s['cur'], s['hops'] = tmp.cur, tmp.hops
            for (pb, n), s in zip(grp, state):
                n.grounded = True; n.tokens = s['tokens']; n.codes = s['codes']; n.over = s['over']; n.cur, n.hops = s['cur'], s['hops']
                n.preds = []; n.recs = []; n.base = None
        self.stats['grounded'] += len(nodes)

    def verify_tokens(self, pb, tokens):
        pb.verifier_calls += 1; self.stats['verifier_calls'] += 1
        return verify(pb.item.prompt, tokens[pb.P:])['success']

    @torch.no_grad()
    def rollout(self, problems, nodes, record=True):
        """Verified greedy grounded completion with selector-argmax codes from each grounded node. Returns values and
        the completed token arrays."""
        live = [i for i, n in enumerate(nodes) if not n.terminal]
        vals = {i: n.value for i, n in enumerate(nodes) if n.terminal}; final = {i: n.tokens for i, n in enumerate(nodes) if n.terminal}
        toks = {i: nodes[i].tokens.copy() for i in live}; codes = {i: nodes[i].codes.copy() for i in live}; over = {i: nodes[i].over for i in live}
        cur = {i: (nodes[i].cur, nodes[i].hops) for i in live}
        self.stats['rollouts'] += len(live)
        while live:
            x, docs, L = self.e.batch([toks[i] for i in live])
            st = self.e.state_heads(x, docs, L, need_codes=False)
            k = st['pi_logits'].argmax(-1)
            cmat = self.code_matrix([codes[i] for i in live], x.shape[1] // CH + 1)
            for r, i in enumerate(live):
                J = len(toks[i]) // CH - 1; cmat[r, J] = k[r]; codes[i][J] = int(k[r])
                if record:
                    problems[i].rollout_states.append((st['u0'][r].cpu(), stop_legal(problems[i].item, *cur[i])))
            chunk = self.decode_over(x, docs, L, cmat, [over[i] for i in live]).cpu().numpy(); self.stats['rollout_chunks'] += len(live)
            nxt = []
            for r, i in enumerate(live):
                toks[i] = np.r_[toks[i], chunk[r]].astype(np.int64); ch = tuple(int(v) for v in chunk[r])
                cl = classify(problems[i].item, cur[i][0], cur[i][1], ch)
                if cl['kind'] == 'done' or not cl['legal'] or cur[i][1] + 1 >= problems[i].limit:
                    if cl['kind'] == 'done':
                        vals[i] = float(self.verify_tokens(problems[i], toks[i]))
                    else:
                        vals[i] = 0.; problems[i].verifier_calls += 1; self.stats['verifier_calls'] += 1
                    final[i] = toks[i]
                else:
                    cur[i] = (cl['node'], cur[i][1] + 1); nxt.append(i)
            live = nxt
        return [vals[i] for i in range(len(nodes))], [final[i] for i in range(len(nodes))]

    # ---------------------------------------------------------------- search
    def puct(self, node, noise=None):
        prior = node.prior if noise is None else (1 - DIR_W) * node.prior + DIR_W * noise
        Ns = node.N.sum(); q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.)
        score = q + C_PUCT * prior * np.sqrt(Ns) / (1 + node.N)
        score = np.where(prior > 0, score, -np.inf)
        return int(np.argmax(score))         # ties -> lower action index

    def grounding_rule(self, depth):
        cfg = self.cfg
        if cfg['children'] == 'grounded':
            return 'decode'
        R = cfg.get('ground_every')
        return 'replay' if (R is not None and depth % R == 0) else 'imagine'

    def simulate(self, act, roots, noise, budget, deadline=None):
        """PUCT simulations in lockstep until each tree has `budget` expansions (or 4x budget simulations).
        Returns per-problem accepted grounded STOP node (or None)."""
        cfg = self.cfg; K = self.K; H = cfg['H']
        expanded = np.zeros(len(act), int); sims = np.zeros(len(act), int); solved = [None] * len(act)
        while True:
            if deadline is not None and time.monotonic() > deadline:
                break
            todo = [i for i in range(len(act)) if solved[i] is None and expanded[i] < budget and sims[i] < 4 * budget]
            if not todo:
                break
            new = {'decode': [], 'replay': [], 'imagine': []}; paths, leaves = {}, {}
            for i in todo:
                node = roots[i]; path = []
                while True:
                    a = self.puct(node, noise[i] if node is roots[i] else None); path.append((node, a))
                    if a == K:
                        if node.stop_value is None:
                            node.stop_value = float(self.verify_tokens(act[i], np.r_[node.tokens, DONE_CHUNK])) if node.grounded else float(node.prior[K])
                        leaves[i] = ('value', node.stop_value)
                        if node.grounded and node.stop_value == 1. and (node is roots[i] or cfg.get('one_tree')):
                            solved[i] = node              # a verified accepted STOP at a grounded node is a decoded, verified answer
                        break
                    child = node.children.get(a)
                    if child is None:
                        child = Node(node, a, node.depth + 1, K); node.children[a] = child
                        rule = self.grounding_rule(child.depth)
                        if rule == 'decode' and not node.grounded:
                            rule = 'replay'
                        if rule in ('imagine', 'replay'):
                            child.base = node if node.grounded else node.base
                            child.preds = ([] if node.grounded else node.preds) + [node.chatk[a]]
                            child.recs = ([] if node.grounded else node.recs) + [(node.chatk[a], node.ak[a])]
                        new[rule].append((i, node, a, child)); leaves[i] = ('new', child); break
                    if child.terminal or child.depth >= H:
                        leaves[i] = ('value', child.value); node = child; break
                    node = child
                paths[i] = path; sims[i] += 1
            grounded_new = []
            if new['decode']:
                kids = self.decode_children([act[i] for i, *_ in new['decode']], [n for _, n, _, _ in new['decode']], [a for _, _, a, _ in new['decode']])
                for (i, parent, a, ph), kid in zip(new['decode'], kids):
                    parent.children[a] = kid; leaves[i] = ('new', kid); grounded_new.append((i, kid))
            if new['replay']:
                self.replay([act[i] for i, *_ in new['replay']], [c for _, _, _, c in new['replay']])
                grounded_new += [(i, c) for i, _, _, c in new['replay']]
            if grounded_new:
                live = [k for _, k in grounded_new if not k.terminal]
                if live:
                    self.ground_rows(live)
                if cfg['leaf'] == 'rollout':
                    vals, _ = self.rollout([act[i] for i, _ in grounded_new], [k for _, k in grounded_new])
                    for (_, kid), v in zip(grounded_new, vals):
                        kid.value = v
                elif live:
                    v = self.leaf_value(torch.stack([k.u0 for k in live])).cpu().numpy()
                    for kid, vv in zip(live, v):
                        kid.value = float(vv)
            if new['imagine']:
                if cfg['leaf'] != 'value':
                    raise ValueError('imagined children need leaf = value (rollout leaves require grounded children)')
                kids = [c for _, _, _, c in new['imagine']]; self.imagine_rows(kids)
                v = self.leaf_value(torch.stack([c.u0 for c in kids])).cpu().numpy()
                for c, vv in zip(kids, v):
                    c.value = float(vv)
            for i in todo:
                kind, x = leaves[i]; v = x.value if kind == 'new' else x
                if kind == 'new':
                    expanded[i] += 1; act[i].nodes += 1; self.stats['expansions'] += 1
                for node, a in paths[i]:
                    node.N[a] += 1; node.W[a] += v
        return solved

    def make_roots(self, act):
        roots = []
        for p in act:
            n = Node(None, None, 0, self.K); n.grounded = True; n.tokens = p.tokens(); n.codes = p.code_array(); n.cur, n.hops = p.cur, p.hops
            roots.append(n)
        self.ground_rows(roots)
        noise = []
        for i, p in enumerate(act):
            if self.cfg.get('noise'):
                v = p.rng.dirichlet([DIR_ALPHA] * (self.K + 1)) * (roots[i].prior > 0); noise.append(v / max(v.sum(), 1e-12))
            else:
                noise.append(None)
        return roots, noise

    @torch.no_grad()
    @_release_search_memory
    def step(self, problems, deadline=None):
        """One receding-horizon step for every active problem."""
        K = self.K; act = [p for p in problems if not p.done]
        if not act:
            return
        roots, noise = self.make_roots(act)
        solved = self.simulate(act, roots, noise, self.cfg['budget'], deadline)
        chosen = []
        for i, p in enumerate(act):
            r = roots[i]; N = r.N.copy()
            order = sorted(range(K + 1), key=lambda a: (-N[a], -r.prior[a], a))
            a = order[0] if N.sum() > 0 else int(np.argmax(r.prior))
            if solved[i] is not None:
                a = K                                  # a verified accepted STOP at the grounded root ends the problem (§9.3)
            ch = r.children.get(a) if a < K else None
            p.roots.append(dict(u0=r.u0.cpu(), imag_u0=ch.u0.cpu() if ch is not None and not ch.grounded and ch.u0 is not None else None, visits=N[:K].copy(),
                                stop_visits=float(N[K]), prior=r.prior.copy(), stop_label=stop_legal(p.item, p.cur, p.hops), action=a, cur=p.cur, hops=p.hops))
            chosen.append(a)
        for i, a in enumerate(chosen):
            if a == K:
                p = act[i]; p.chunks.append(DONE_CHUNK); p.done = True
                p.success = bool(roots[i].stop_value == 1.) if roots[i].stop_value is not None else bool(self.verify_tokens(p, p.tokens())); p.reason = 'stop'
        exec_rows = [(i, a) for i, a in enumerate(chosen) if a < K]
        if exec_rows:
            fresh = [(i, a) for i, a in exec_rows if not (roots[i].children.get(a) is not None and roots[i].children[a].grounded)]
            if fresh:
                kids = self.decode_children([act[i] for i, _ in fresh], [roots[i] for i, _ in fresh], [a for _, a in fresh])
                for (i, a), kid in zip(fresh, kids):
                    roots[i].children[a] = kid
            for i, a in exec_rows:
                p = act[i]; kid = roots[i].children[a]; ch = tuple(int(v) for v in kid.tokens[-CH:])
                p.chunks.append(ch); p.codes.append(a); p.executed_tokens += CH
                cl = classify(p.item, p.cur, p.hops, ch)
                if cl['kind'] == 'done':
                    p.done = True; p.success = bool(kid.value == 1.) if kid.terminal and kid.value is not None else bool(self.verify_tokens(p, p.tokens())); p.reason = 'decoded_done'
                elif not cl['legal']:
                    p.done = True; p.success = False; p.reason = 'illegal'
                else:
                    p.cur, p.hops = cl['node'], p.hops + 1
                    if p.hops >= p.limit:
                        p.done = True; p.success = False; p.reason = 'hop_limit'

    @torch.no_grad()
    @_release_search_memory
    def one_tree(self, problems, budget, deadline=None):
        """G(R) / IMAG (§9.1): one search per problem from the prompt; then the most-visited path is decoded (pending
        chunks replayed under their records) and completed greedily, and the answer verified."""
        act = [p for p in problems if not p.done]
        if not act:
            return
        roots, noise = self.make_roots(act)
        solved = self.simulate(act, roots, noise, budget, deadline)
        ends = []
        for i, p in enumerate(act):
            if solved[i] is not None:
                p.done = True; p.success = True; p.reason = 'verified_stop'; p.answer = np.r_[solved[i].tokens, DONE_CHUNK]; continue
            node = roots[i]; stop = False
            while node.N.sum() > 0:
                a = sorted(range(self.K + 1), key=lambda a: (-node.N[a], -node.prior[a], a))[0]
                if a == self.K:
                    stop = True; break
                if a not in node.children:
                    break
                node = node.children[a]
                if node.terminal:
                    break
            ends.append((p, node, stop))
        need = [(p, n) for p, n, _ in ends if not n.grounded]
        if need:
            self.replay([p for p, _ in need], [n for _, n in need])
        for p, n, stop in ends:
            if stop and not n.terminal:
                p.done = True; p.answer = np.r_[n.tokens, DONE_CHUNK]; p.success = bool(self.verify_tokens(p, p.answer)); p.reason = 'stop'
        rest = [(p, n) for p, n, _ in ends if not p.done]
        if rest:
            vals, toks = self.rollout([p for p, _ in rest], [n for _, n in rest], record=False)
            for (p, n), v, t in zip(rest, vals, toks):
                p.done = True; p.success = bool(v == 1.); p.answer = t; p.reason = 'path_completion'

    def solve(self, problems, wall=None):
        t0 = time.monotonic()
        while any(not p.done for p in problems):
            if wall is not None and time.monotonic() - t0 > wall:
                for p in problems:
                    if not p.done:
                        p.done = True; p.success = False; p.reason = 'wall_clock'
                break
            self.step(problems)
        return problems


def problems_from(items, hop_limit, seed0):
    return [Problem(it, hop_limit, seed0 + i) for i, it in enumerate(items)]
