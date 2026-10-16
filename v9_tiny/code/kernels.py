"""Refine kernels (draft §8.3): GradCuit on the code embedding at one chunk (e = e_k + delta through W_z) and the
random-direction control with the same step norm. Reward R_grad = o - 1 (0 accepted -> stop, -1 rejected), loss
-R_grad * sum_t log p(x_t | plans(e_k + delta)) on the current failed answer, teacher-forced; Adam, <= 10 updates;
after each update the answer is regenerated greedily from that chunk on and verified. Bound: RMS(delta) <= .5 RMS(e_k).
"""
import math
import numpy as np
import torch
import v9paths
from data import DONE, verify
from hgen import CH, classify

MAX_UPDATES = 10


def plans_with(hs, x, docs, codes, j=None, e_prime=None):
    """Plans under the plan records `codes`, with chunk j's plan recomputed from the edited code table e_prime."""
    h, c, cd, _ = hs.context(x, docs)
    H = hs.heads(c, cd)
    M = c.shape[1]
    if codes.shape[1] < M:                       # pad plan records with code-free (-1) chunks
        codes = torch.cat([codes, codes.new_full((codes.shape[0], M - codes.shape[1]), -1)], 1)
    codes = codes[:, :M]; chat, a = hs.plan(H, codes)
    if e_prime is not None:
        H2 = hs.heads(c, cd, e=e_prime); k = codes[:, j]
        rows = torch.arange(len(x), device=x.device)
        chat = chat.clone(); a = a.clone()
        chat[:, j] = H2['chatk'][rows, j, k]; a[:, j] = H2['ak'][rows, j, k]
    return h, chat, a


def answer_logprob(hs, x, docs, codes, P, L, j=None, e_prime=None):
    """Teacher-forced sum of log p over response tokens P..L-1 (one row)."""
    h, chat, a = plans_with(hs, x, docs, codes, j, e_prime)
    logits = hs.f.decode(h, chat, a, docs)
    lp = logits[:, P - 1:L - 1].float().log_softmax(-1).gather(-1, x[:, P:L, None]).squeeze(-1)
    return lp.sum()


@torch.no_grad()
def regenerate(engine, item, tokens, codes, j, e_prime, hop_limit):
    """Keep tokens up to the end of chunk j; regenerate greedily from chunk j+1 with chunk j's plan from e_prime and
    selector-argmax codes after it. Returns (tokens, codes)."""
    hs = engine.hs; dev = engine.device; keep = (j + 1) * CH
    toks = np.array(tokens[:keep]); cd = codes.copy(); cd[j + 1:] = -1
    cur, hops, legal = item.walk([tuple(int(v) for v in toks[i:i + CH]) for i in range(item.P, keep, CH)])
    first = True
    while True:
        x, docs, L = engine.batch([toks]); J = len(toks) // CH - 1
        if len(cd) < x.shape[1] // CH:                      # the regenerated answer may run longer than the original
            cd = np.r_[cd, np.full(x.shape[1] // CH - len(cd), -1, dtype=cd.dtype)]
        if not first and engine.K > 1:
            st = engine.state_heads(x, docs, L, need_codes=False); cd[J] = int(st['pi_logits'][0].argmax())
        ca = torch.as_tensor(cd[:x.shape[1] // CH], device=dev)[None]
        h, chat, a = plans_with(hs, x, docs, ca, j, e_prime)
        for i in range(CH):
            if i:
                h = hs.f.encode(x, docs)
            hid = engine.hidden(h, chat, a, docs); tok = engine.logits_at(hid, L + i - 1).argmax(-1)
            x[0, L + i] = tok; docs[0, L + i] = 0
        ch = tuple(int(v) for v in x[0, L:L + CH].cpu()); toks = np.r_[toks, ch].astype(np.int64); first = False
        cl = classify(item, cur, hops, ch)
        if cl['kind'] == 'done' or not cl['legal'] or hops + 1 >= hop_limit:
            return toks, cd
        cur, hops = cl['node'], hops + 1


def project(delta, ek, bound):
    if bound is None:
        return delta
    lim = bound * ek.square().mean().sqrt(); r = delta.square().mean().sqrt()
    return delta * torch.clamp(lim / r.clamp_min(1e-12), max=1.)


def refine(engine, item, tokens, codes, kernel='gradcuit', lr=1e-2, bound=.5, hop_limit=20, rng=None, reward_from_o=False):
    """One CONT refinement of a failed answer. Returns dict(success, updates, verifier_calls, tokens, codes, j, trace).
    reward_from_o=True is the contract-6 mutant (R = o): it must be caught as a no-op."""
    hs = engine.hs; dev = engine.device; rng = rng or np.random.RandomState(0)
    J0 = item.P // CH - 1; committed = [j for j in range(J0, len(tokens) // CH - 1) if codes[j] >= 0]
    if not committed:
        return dict(success=False, updates=0, verifier_calls=0, tokens=tokens, codes=codes, j=None, trace=[], note='no committed chunk')
    j = int(committed[rng.randint(len(committed))]); k = int(codes[j])
    E = hs.codes.e.detach(); delta = torch.zeros_like(E[k], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr); trace = []; calls = 0; toks, cd = np.array(tokens), codes.copy()
    o = float(verify(item.prompt, toks[item.P:])['success']); calls += 1
    for u in range(MAX_UPDATES):
        if o == 1.:
            break
        R = o if reward_from_o else o - 1.
        x, docs, L = engine.batch([toks]); ca = torch.as_tensor(cd[:x.shape[1] // CH], device=dev)[None]
        e_prime = E.clone(); e_prime[k] = E[k] + delta
        if kernel == 'gradcuit':
            with torch.enable_grad():
                loss = -R * answer_logprob(hs, x, docs, ca, item.P, int(L[0]), j, e_prime)
                opt.zero_grad(); loss.backward()
            g = float(delta.grad.norm()); before = delta.detach().clone(); opt.step()
            with torch.no_grad():
                delta.copy_(project(delta, E[k], bound))
            step = float((delta.detach() - before).norm())
        else:  # random direction with Adam's nominal step norm lr * sqrt(d) and no reward gradient (I16)
            g = None; d = torch.as_tensor(rng.standard_normal(delta.shape[0]), dtype=delta.dtype, device=dev); d = d / d.norm() * lr * math.sqrt(delta.shape[0])
            with torch.no_grad():
                delta.add_(d); delta.copy_(project(delta, E[k], bound))
            step = float(d.norm())
        e_prime = E.clone(); e_prime[k] = E[k] + delta.detach()
        toks, cd = regenerate(engine, item, toks, cd, j, e_prime, hop_limit)
        o = float(verify(item.prompt, toks[item.P:])['success']); calls += 1
        trace.append(dict(update=u + 1, grad_norm=g, step_norm=step, delta_rms=float(delta.detach().square().mean().sqrt()), accepted=bool(o)))
    return dict(success=bool(o), updates=len(trace), verifier_calls=calls, tokens=toks, codes=cd, j=j, trace=trace)
