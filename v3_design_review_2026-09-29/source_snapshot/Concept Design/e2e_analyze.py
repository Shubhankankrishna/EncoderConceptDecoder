"""Post-training analysis of end-to-end checkpoints (chain task), and the XM decision metrics (branch task)."""
import sys, math, json, torch, torch.nn.functional as F
from e2e import *


def load(task, arm, K, seed):
    m = Model(arm, K); m.load_state_dict(torch.load(f"e2e_{task}_{arm}_K{K}_{seed}.pt")); m.eval(); return m


@torch.no_grad()
def chain_report(m, n=128, Rs=(1, 2, 3, 4, 6, 8, 12, 16)):
    g = torch.Generator().manual_seed(9001); ids, tgt, meta = make_batch(n, g, hops=(1, 2, 3, 4, 5)); doc = torch.zeros_like(ids)
    pos = torch.stack([torch.tensor([q[0] + 8 for q in mt[1]]) for mt in meta]); hs = torch.tensor([[q[2] for q in mt[1]] for mt in meta]); truth = tgt.gather(1, pos)
    ks = [None] if m.K == 1 else list(range(m.K)); loop = m.core.arm.startswith('loop'); out = {}
    def acc(R, off=False, k_list=ks):
        lp = torch.logsumexp(torch.stack([m(ids, doc, R, k, off)[0].log_softmax(-1) for k in k_list]), 0) - math.log(len(k_list))
        pr = lp.gather(1, pos[..., None].expand(-1, -1, VT)).argmax(-1)
        return [round((pr == truth)[hs == h].float().mean().item(), 3) for h in (1, 2, 3, 4, 5)]
    for R in (Rs if loop else [4]): out[f'R{R}'] = acc(R)
    out['plan_off'] = acc(4, off=True)
    if m.K > 1: out['per_k_R4'] = [acc(4, k_list=[k]) for k in ks]
    # codebook health on real encoder outputs
    tok, c_hat, _ = m.enc(ids, doc); nidx = m.core.assign(c_hat); used = [(torch.bincount(nidx[..., j].flatten(), minlength=m.core.N) > 0).float().mean().item() for j in range(m.core.Sg)]
    q = m.core.e[torch.arange(m.core.Sg), nidx].flatten(-2); ve = 1 - (c_hat - q).pow(2).sum(-1).mean() / (c_hat - c_hat.mean((0, 1))).pow(2).sum(-1).mean()
    out['codebook_used'] = round(sum(used) / len(used), 3); out['codebook_var_explained'] = round(ve.item(), 3)
    # NCP: CE vs entropy of the marginal code distribution (information gain in nats per segment)
    _, lg, _ = m.core(c_hat, doc[:, ::4], 4, None if m.K == 1 else 0); ce = ncp_ce(m.core, lg, c_hat, doc[:, ::4]).mean().item()
    marg = torch.stack([torch.bincount(nidx[..., j].flatten(), minlength=m.core.N).float() for j in range(m.core.Sg)]); marg = marg / marg.sum(1, keepdim=True)
    H = -(marg * marg.clamp_min(1e-12).log()).sum(1).mean().item(); out['ncp_ce'] = round(ce, 3); out['code_marginal_entropy'] = round(H, 3)
    return out


def pass_dynamics(m, n=32, R=16):
    """State norm, step size and gradient reaching each pass (answer NLL at the query positions)."""
    g = torch.Generator().manual_seed(77); ids, tgt, meta = make_batch(n, g, hops=(3,)); doc = torch.zeros_like(ids)
    tok, c_hat, _ = m.enc(ids, doc); doc_c = doc[:, ::4]; X = c_hat.detach()
    streams = [X] if m.K == 1 else [X, m.core.z[0].view(1, 1, -1).expand_as(X)]
    S = torch.zeros_like(X); states = []
    for r in range(R):
        S = m.core.layer(streams, S, doc_c); S.retain_grad(); states.append(S)
    u = m.core.out_norm(S); _, cp = m.core.head(u); out = m.dec(tok.detach(), u, cp, doc); lt = out @ m.enc.E.weight.T / math.sqrt(out.shape[-1])
    mask = tgt >= 0; F.cross_entropy(lt[mask], tgt[mask]).backward()
    qc = torch.tensor([q[0] // 4 + 1 for q in meta[0][1]])                   # concept holding each query's end
    norms = [round(s[:, qc].norm(dim=-1).mean().item(), 2) for s in states]
    steps = [round((states[i][:, qc] - states[i - 1][:, qc]).norm(dim=-1).mean().item(), 3) for i in range(1, R)]
    grads = [float(f"{s.grad[:, qc].norm(dim=-1).mean().item():.3g}") for s in states]
    return dict(state_norm=norms, step_size=steps, grad_norm_at_pass=grads)


@torch.no_grad()
def branch_report(m, n=64, R=4):
    g = torch.Generator().manual_seed(5150); ids, tgt, meta = make_batch(n, g, hops=(2,), branch=True, path=True); doc = torch.zeros_like(ids)
    ks = [None] if m.K == 1 else list(range(m.K)); mask = tgt >= 0
    lps = torch.stack([m(ids, doc, R, k)[0].log_softmax(-1).gather(-1, tgt.clamp_min(0)[..., None]).squeeze(-1) * mask for k in ks])   # K, B, T
    per_doc = lps.sum(-1)                                                                      # K, B
    out = dict(nll_mixture=round(-(torch.logsumexp(per_doc, 0) - math.log(len(ks))).sum().item() / mask.sum().item(), 4))
    if m.K > 1:
        tok, c_hat, _ = m.enc(ids, doc)
        ncp = torch.stack([ncp_ce(m.core, m.core(c_hat, doc[:, ::4], R, k)[1], c_hat, doc[:, ::4]).sum(-1) for k in ks])
        ks_ = ncp.argmin(0); out['nll_selected_k_LEAKY'] = round(-per_doc[ks_, torch.arange(n)].sum().item() / mask.sum().item(), 4)
        out['nll_single_random_k'] = round(-per_doc.mean(0).sum().item() / mask.sum().item(), 4)
    # generation: greedy path per hypothesis (K>1) or greedy + 4 samples (K=1); coherence and coverage
    coh, cov = [], []
    gen = torch.Generator().manual_seed(3)
    def generate(k, sample):
        cur = ids.clone()
        for qi in range(7):
            for step in range(2):
                p = meta[0][1][qi][0] + 8 + step
                lp = m(cur, doc, R, k)[0][:, p].log_softmax(-1)
                nxt = torch.multinomial(lp.exp(), 1, generator=gen)[:, 0] if sample else lp.argmax(-1)
                cur[:, p + 1] = nxt
        return cur
    runs = [(k, False) for k in ks] if m.K > 1 else [(None, False)] + [(None, True)] * 4
    gens = [generate(k, s) for k, s in runs]
    for b, (S, qs) in enumerate(meta):
        for (base, x, h) in qs:
            valid = {(y, z) for y in S[x] for z in S[y]}
            paths = [(gq[b, base + 9].item(), gq[b, base + 10].item()) for gq in gens]
            coh += [paths[0] in valid] if m.K == 1 else [p in valid for p in paths]
            cov.append(len(set(paths) & valid) / len(valid))
    out['greedy_path_coherent'] = round(sum(coh) / len(coh), 3); out['coverage_4_generations'] = round(sum(cov) / len(cov), 3)
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == 'chain':
        res = {}
        for spec in sys.argv[2:]:
            arm, K, seed = spec.split(':'); m = load('chain', arm, int(K), int(seed))
            r = chain_report(m)
            if arm.startswith('loop'): r['dynamics'] = pass_dynamics(m)
            res[spec] = r; print(spec, json.dumps(r), flush=True)
        json.dump(res, open('chain_analysis.json', 'w'))
    else:
        res = {}
        for spec in sys.argv[2:]:
            arm, K, seed = spec.split(':'); r = branch_report(load('branch', arm, int(K), int(seed))); res[spec] = r; print(spec, json.dumps(r), flush=True)
        json.dump(res, open('branch_analysis.json', 'w'))
