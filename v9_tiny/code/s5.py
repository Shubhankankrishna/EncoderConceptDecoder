"""S5 methods and budgets (draft §9.5-§9.6, I17).

Budgets are multiples b of one greedy answer's wall-clock t_g, measured on the same batch (lockstep over a batch of
items, so every method batches across items like greedy decoding). Search methods size their expansions per search
adaptively from the measured time per expansion so that the expected wall-clock meets the budget; a hard stop at 2x
the budget marks unfinished items as failures ('wall_clock'); overruns are reported.
Methods: rh (RH(H), decoder-free), rh_grounded (mechanism comparison), g (one-tree G(R), G(1) grounded; IMAG = R None),
token_tree (3 colours + STOP, code-free prior, grounded children), best_of_n (T = 1 code-free sampling), greedy.
"""
import math, time
import numpy as np
import torch
import v9paths
from data import DONE, COL0, N_COL, verify
from runtime import sync
import hgen, csearch, baselines, kernels
from hgen import CH, classify

HARD_STOP = 2.


class TokenSearch(csearch.Search):
    """Token tree (§9.6): actions = the 3 colour tokens at a hop's first position plus STOP (the DONE chunk); prior =
    the decoder's softmax over those four tokens under code-free plans; each child forces its first token and completes
    the hop greedily; leaves = verified greedy code-free completion, or a value head on the decoder's final-layer state."""

    def __init__(self, engine, cfg):
        super().__init__(engine, cfg); self.K = N_COL

    def code_matrix(self, rows_codes, width):
        return torch.full((len(rows_codes), width), -1, dtype=torch.long, device=self.dev)

    @torch.no_grad()
    def ground_rows(self, nodes):
        x, docs, L = self.e.batch([n.tokens for n in nodes]); hs = self.hs
        h, c, cd = self.e.context(x, docs); H = hs.heads(c, cd, need_codes=False)
        hid = self.e.hidden(h, H['chat0'], H['a0'], docs); last = hid[torch.arange(len(nodes), device=self.dev), L - 1]
        logits = last @ hs.f.emb.weight.t(); cand = torch.tensor([COL0 + i for i in range(N_COL)] + [DONE], device=self.dev)
        prior = logits[:, cand].float().softmax(-1).cpu().numpy()
        for i, n in enumerate(nodes):
            n.u0 = last[i]; n.prior = prior[i]; n.stop_ok = True; n.ctx = None
        return None

    @torch.no_grad()
    def decode_free(self, x, docs, L, force=None):
        """Greedy chunk under code-free plans (chat0, A0), optionally forcing the first token (hgen.decode_chunk
        assumes at least one code when K > 1, so the token tree decodes here)."""
        hs = self.hs; h, c, cd = self.e.context(x, docs); H = hs.heads(c, cd, need_codes=False); rows = torch.arange(len(x), device=x.device)
        for i in range(CH):
            if i:
                h = hs.f.encode(x, docs)
            tok = self.e.logits_at(self.e.hidden(h, H['chat0'], H['a0'], docs), L + i - 1).argmax(-1)
            if i == 0 and force is not None:
                tok = force
            x[rows, L + i] = tok; docs[rows, L + i] = 0
        return x.gather(1, (L[:, None] + torch.arange(CH, device=x.device)[None]))

    def decode_children(self, problems, parents, actions):
        x, docs, L = self.e.batch([p.tokens for p in parents])
        force = torch.tensor([COL0 + a for a in actions], device=self.dev)
        chunk = self.decode_free(x, docs, L, force).cpu().numpy(); out = []
        for i, (pb, p) in enumerate(zip(problems, parents)):
            ch = tuple(int(v) for v in chunk[i]); n = csearch.Node(p, actions[i], p.depth + 1, self.K); n.grounded = True
            n.tokens = np.r_[p.tokens, chunk[i]].astype(np.int64); n.codes = p.codes; self.finish_child(pb, p.cur, p.hops, n, ch); out.append(n)
        self.stats['grounded'] += len(out)
        return out

    @torch.no_grad()
    def rollout(self, problems, nodes, record=False):
        live = [i for i, n in enumerate(nodes) if not n.terminal]
        vals = {i: n.value for i, n in enumerate(nodes) if n.terminal}; final = {i: n.tokens for i, n in enumerate(nodes) if n.terminal}
        toks = {i: nodes[i].tokens.copy() for i in live}; cur = {i: (nodes[i].cur, nodes[i].hops) for i in live}; self.stats['rollouts'] += len(live)
        while live:
            x, docs, L = self.e.batch([toks[i] for i in live])
            chunk = self.decode_free(x, docs, L).cpu().numpy(); self.stats['rollout_chunks'] += len(live); nxt = []
            for r, i in enumerate(live):
                toks[i] = np.r_[toks[i], chunk[r]].astype(np.int64); ch = tuple(int(v) for v in chunk[r]); cl = classify(problems[i].item, *cur[i], ch)
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

    def grounding_rule(self, depth):
        return 'decode'


@torch.no_grad()
def greedy_batch(engine, items, mode='codes'):
    """Reference greedy answers and t_g (wall per item on this batch)."""
    sync(); t = time.monotonic()
    if mode == 'codes' and engine.K > 1:
        g = engine.greedy([it.prompt for it in items], max_new=384 - items[0].P)
        ok = [v['success'] for v in g['verification']]; hops = [int((g['lengths'][i] - items[i].P) // CH) for i in range(len(items))]
    else:
        res = baselines.sample_answers(engine, items, temp=0., code_policy='free'); ok = [r[2] for r in res]; hops = [(len(r[0]) - items[i].P) // CH for i, r in enumerate(res)]
    sync(); good = [h for h, o in zip(hops, ok) if o]
    # Expected executed steps per item: hop count of the successful greedy answers (failed ones run to the length limit).
    return ok, (time.monotonic() - t) / len(items), float(np.mean(good)) if good else float(np.median(hops))


def run_search(engine, items, method, budget_s, cfg, hop_limit, seed, mean_hops, token=False):
    """RH-type methods: adaptive expansions per step to fit budget_s (whole batch). Returns per-item records."""
    S = (TokenSearch if token else csearch.Search)(engine, dict(cfg))
    ps = csearch.problems_from(items, hop_limit, seed); sync(); t0 = time.monotonic(); B = 4; per_exp = None
    while any(not p.done for p in ps):
        el = time.monotonic() - t0
        if el > HARD_STOP * budget_s:
            for p in ps:
                if not p.done:
                    p.done = True; p.success = False; p.reason = 'wall_clock'
            break
        live = [p for p in ps if not p.done]; steps_left = max(1., mean_hops + 1 - np.mean([p.hops for p in live]))
        if per_exp is not None:
            B = int(np.clip((budget_s - el) / steps_left / per_exp, 1, 512))
        S.cfg['budget'] = B; e0 = S.stats['expansions']; s0 = time.monotonic()
        S.step(ps, deadline=t0 + HARD_STOP * budget_s); sync()
        ne = S.stats['expansions'] - e0
        if ne > 0:
            per_exp = (time.monotonic() - s0) / max(ne / len(live), 1)        # seconds per (expansion per problem) in lockstep
    wall = time.monotonic() - t0
    return [dict(success=bool(p.success), reason=p.reason, verifier_calls=p.verifier_calls, nodes=p.nodes, executed_tokens=p.executed_tokens) for p in ps], dict(S.stats, wall_s=wall, overrun=wall / budget_s - 1)


def run_one_tree(engine, items, budget_s, cfg, hop_limit, seed, probe_budget=8):
    """G(R)/IMAG: a short probe tree measures time per expansion, then one tree per item with the expansions that fit."""
    S = csearch.Search(engine, dict(cfg, one_tree=True)); probe = csearch.problems_from(items[:min(8, len(items))], hop_limit, seed + 99999)
    sync(); t = time.monotonic(); S.one_tree(probe, probe_budget); sync(); dt = (time.monotonic() - t) / max(probe_budget, 1)
    budget = int(np.clip(budget_s / dt * min(8, len(items)) / len(items), 1, 4096))
    S = csearch.Search(engine, dict(cfg, one_tree=True)); ps = csearch.problems_from(items, hop_limit, seed); sync(); t0 = time.monotonic()
    S.one_tree(ps, budget, deadline=t0 + HARD_STOP * budget_s); sync(); wall = time.monotonic() - t0
    return [dict(success=bool(p.success), reason=p.reason, verifier_calls=p.verifier_calls, nodes=p.nodes) for p in ps], dict(S.stats, wall_s=wall, overrun=wall / budget_s - 1, expansions_per_item=budget)


def run_best_of_n(engine, items, budget_s, hop_limit, seed):
    gen = torch.Generator().manual_seed(seed); solved = np.zeros(len(items), bool); calls = np.zeros(len(items), int); samples = 0; sync(); t0 = time.monotonic()
    while time.monotonic() - t0 < budget_s and not solved.all():
        idx = np.flatnonzero(~solved); res = baselines.sample_answers(engine, [items[i] for i in idx], temp=1., code_policy='free', hop_limit=hop_limit, gen=gen)
        for i, r in zip(idx, res):
            calls[i] += 1; solved[i] |= r[2]
        samples += len(idx); sync()
    wall = time.monotonic() - t0
    return [dict(success=bool(s), verifier_calls=int(c)) for s, c in zip(solved, calls)], dict(samples=samples, wall_s=wall, overrun=wall / budget_s - 1)


def answer_loop(engine, items, budget_s, inner_cfg, cont_kernel, lr, bound, hop_limit, seed):
    """AB-MCTS-A (§9.3): per item, Thompson sampling between GEN (a fresh inner RH episode with fresh root noise) and
    CONT (refine a failed answer with the kernel at a uniformly chosen committed chunk). Binary feedback; shared budget."""
    rng = np.random.RandomState(seed); n = len(items); solved = np.zeros(n, bool); answers = [[] for _ in range(n)]
    stats = dict(gen=np.zeros(n, int), cont=np.zeros(n, int), calls=np.zeros(n, int)); sync(); t0 = time.monotonic(); ep = 0
    while time.monotonic() - t0 < budget_s and not solved.all():
        idx = np.flatnonzero(~solved); gen_idx, cont_idx = [], []
        for i in idx:
            g = rng.beta(1, 1 + stats['gen'][i]); c = rng.beta(1, 1 + stats['cont'][i]) if answers[i] else -1
            (cont_idx if c > g else gen_idx).append(i)
        if gen_idx:
            S = csearch.Search(engine, dict(inner_cfg, noise=True)); ps = csearch.problems_from([items[i] for i in gen_idx], hop_limit, seed + 1000 * (ep + 1))
            S.solve(ps)
            for i, p in zip(gen_idx, ps):
                stats['gen'][i] += 1; stats['calls'][i] += p.verifier_calls; solved[i] |= p.success
                if not p.success:
                    answers[i].append((p.tokens(), p.code_array()))
        for i in cont_idx:
            if time.monotonic() - t0 > budget_s:
                break
            toks, codes = answers[i][rng.randint(len(answers[i]))]
            r = kernels.refine(engine, items[i], toks, codes, cont_kernel, lr, bound, hop_limit, rng)
            stats['cont'][i] += 1; stats['calls'][i] += r['verifier_calls']; solved[i] |= r['success']
            if not r['success']:
                answers[i].append((r['tokens'], r['codes']))
        ep += 1; sync()
    wall = time.monotonic() - t0
    return [dict(success=bool(s), gen=int(g), cont=int(c), verifier_calls=int(v)) for s, g, c, v in zip(solved, stats['gen'], stats['cont'], stats['calls'])], dict(wall_s=wall, overrun=wall / budget_s - 1, rounds=ep)
