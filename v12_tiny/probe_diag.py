"""Probe diagnostics: which latent does the base represent linearly? (pilot follow-up)

Targets per trajectory row: abstract action (4), target colour (8), target cell (G^2 location of the target
colour), last visited colour (9 incl. none), plus a label-copy sanity feature. python -B probe_diag.py RUN
"""
import json, sys
import numpy as np
import torch
import pinpad as P
from data import Split
from evaluate import load_run
from probes import fit_probe


def main(run, device='mps', n=1200, ckpt='final'):
    torch.set_grad_enabled(False)
    model, arm = load_run(run, device, ckpt)
    dev = Split('dev'); W = P.World()
    feats = None; lab = {k: [] for k in ('abs', 'col', 'cell', 'last', 'parity')}; ep_of = []
    for s in range(0, n, 64):
        idx = np.arange(s, min(s + 64, n))
        b = dev.batch(idx, device=device, pad_to=176)
        out = model(b['ids'])
        layers = list(out['resid'])
        if model.kind == 'ECD':
            from model import token_view
            T = out['logits'].shape[1]
            layers += [token_view(out['u'], T, model.k)] + [token_view(st, T, model.k) for st in out['cstates']]
        if feats is None: feats = [[] for _ in layers]
        ids = b['ids'].cpu().numpy(); aa = b['abs_act'].cpu().numpy(); tc = b['tgt_col'].cpu().numpy()
        for bi, i in enumerate(idx):
            tok = ids[bi]; colours = [W.tok_cell(tok[2 + 2 * j]) for j in range(W.O)]
            last = -1; rows = []
            for t in range(len(tok)):
                if aa[bi, t] < 0: continue
                if t > 0 and tc[bi, t - 1] >= 0 and tc[bi, t] != tc[bi, t - 1]: last = tc[bi, t - 1]
                c = colours[tc[bi, t]]
                lab['abs'].append(aa[bi, t]); lab['col'].append(tc[bi, t]); lab['cell'].append(c[0] * W.G + c[1])
                lab['last'].append(last + 1); lab['parity'].append(tc[bi, t] % 2); rows.append(t); ep_of.append(i)
            for li, e in enumerate(layers): feats[li].append(e[bi, rows].float().cpu())
    feats = [torch.cat(f) for f in feats]; ep_of = np.array(ep_of); half = int((ep_of < n // 2).sum())
    res = {}
    torch.set_grad_enabled(True)
    for k, ncls in (('abs', 4), ('col', 8), ('cell', W.G * W.G), ('last', 9), ('parity', 2)):
        y = torch.tensor(np.array(lab[k]), dtype=torch.long)
        res[k] = [round(fit_probe(X[:half], y[:half], X[half:], y[half:], ncls), 3) for X in feats]
        res[k + '_chance'] = round(float(np.bincount(y[half:].numpy()).max() / (len(y) - half)), 3)
    y = torch.tensor(np.array(lab['abs']), dtype=torch.long)
    cheat = torch.nn.functional.one_hot(y, 4).float() + 0.1 * torch.randn(len(y), 4)
    res['sanity_label_copy'] = fit_probe(cheat[:half], y[:half], cheat[half:], y[half:], 4)
    # effective rank (exp Shannon entropy of normalised singular values; NextLat Table 1) per layer
    def erank(X):
        s = torch.linalg.svdvals(X - X.mean(0)); q = s / s.sum()
        return float(torch.exp(-(q * q.clamp(min=1e-12).log()).sum()))
    res['effective_rank'] = [round(erank(X[:8000]), 1) for X in feats]
    res['layers'] = 'token resid e_0..e_L' + (' + concept u, concept states (token view)' if model.kind == 'ECD' else '')
    print(json.dumps(res)); return res


if __name__ == '__main__':
    r = main(sys.argv[1], ckpt=sys.argv[2] if len(sys.argv) > 2 else 'final')
    from pathlib import Path
    (Path(__file__).resolve().parent / 'runs' / sys.argv[1] / f"probe_diag_{sys.argv[2] if len(sys.argv) > 2 else 'final'}.json").write_text(json.dumps(r, indent=1))
