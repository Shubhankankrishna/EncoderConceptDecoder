"""Per-pass dynamics for a trained arm: read RMS, |S|, step, turn angle, gradient reaching each pass; plus accuracy at R = 4, 8, 12, 16."""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, '..')
from exp_v2 import Core, chain_batch
from exp_s0 import CoreS0
def load(arm, seed):
    sd = torch.load(f'{arm}_{seed}.pt') if arm != 'base0' else torch.load('../v2/v2_L1S_0.pt'); d = sd['X'].shape[1]
    m = (CoreS0('L1S', sd['X'], sd['codes'], sd['assign'], d, 176) if 's0' in sd else Core('L1S', sd['X'], sd['codes'], sd['assign'], d, 176))
    m.load_state_dict(sd); m.eval(); return m
def dyn(m, R=12):
    g = torch.Generator().manual_seed(777); ids, tgt, kk = chain_batch(256, g, hops=(6,))
    Xs = m.X[ids]; doc = torch.zeros_like(ids); L = m.layers[0]
    S = m.s0.expand_as(Xs) if hasattr(m, 's0') else torch.zeros_like(Xs); S0n = S[kk > 0].norm(dim=-1).mean().item()
    rows, states = [], []
    for r in range(R):
        x1 = L.r1([Xs, S]); Sn = L(Xs, S, doc); Sn.retain_grad(); states.append(Sn)
        cos = F.cosine_similarity(Sn, S, -1).clamp(-1, 1) if (r or hasattr(m, 's0')) else None
        rows.append(dict(p=r + 1, read_rms=round(x1.pow(2).mean(-1).sqrt()[kk > 0].mean().item(), 3), S_norm=round(Sn[kk > 0].norm(dim=-1).mean().item(), 2),
                         step=round((Sn - S)[kk > 0].norm(dim=-1).mean().item(), 2), turn=None if cos is None else round(torch.rad2deg(torch.acos(cos))[kk > 0].mean().item(), 1)))
        S = Sn
    lp = m.concept_logp(m.head(S)); mask = tgt >= 0; F.nll_loss(lp[mask], tgt[mask]).backward()
    for r, s in enumerate(states): rows[r]['grad'] = float(f"{s.grad[kk > 0].norm(dim=-1).mean().item():.3g}")
    return S0n, rows
@torch.no_grad()
def acc(m, Rs=(4, 8, 12, 16)):
    g = torch.Generator().manual_seed(777); ids, tgt, kk = chain_batch(512, g, hops=(6,)); out = {}
    for R in Rs:
        pred = m.concept_logp(m(ids, R)).argmax(-1); out[f'R{R}'] = [round((pred == tgt)[kk == k].float().mean().item(), 3) for k in range(1, 7)]
    return out
if __name__ == "__main__":
    res = {}
    for spec in sys.argv[1:]:
        arm, seed = spec.split(':') if ':' in spec else (spec, 0); m = load(arm, seed)
        S0n, rows = dyn(m); res[spec] = dict(S0_norm=round(S0n, 2), acc=acc(m), dyn=rows)
        print(spec, 'S0 norm', round(S0n, 2), res[spec]['acc'])
        for r in rows: print('  ', r)
    json.dump(res, open('probe_s0_' + '_'.join(s.replace(':', '') for s in sys.argv[1:]) + '.json', 'w'))
