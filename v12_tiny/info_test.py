"""Information test (Decision 3.4): does the concept state u_g carry information about the next segment's codes
beyond what C's own input p_g already carries? (V10.3's forecast was a worse predictor than its own input.)

Per family (pinpad, math, text) and per codebook segment s, probes predict idx_s(tau_{g+1}) from (a) u_g, (b) p_g
(the pooled input to C); fit on train-split rows, scored on dev rows (as the NCP head). Two probe classes: linear
softmax, and an MLP with the NCP head's shape fit from scratch (the fair comparison with the head). Baselines: majority code, and copy (idx_s(tau_g) as the prediction). Also reported: the trained NCP
head's own accuracy on the same rows.

  python -B info_test.py RUN CKPT      -> runs/RUN/info_test_CKPT.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Split, BPEFamily
from evaluate import load_run
from losses import group_targets, group_valid
from probes import fit_probe
import train

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def collect(model, kind, batches, cb, head, src):
    """Features from `model`; targets (summaries + codebook) from `src` on the same rows (src = model normally)."""
    U, Pp, P2, cur, nxt, head_ok = [], [], [], [], [], []
    for b in batches:
        out = model(b['ids']); k = model.k
        raw = group_targets(out if src is model else src(b['ids']), k, kind); G = raw.shape[1]
        if G < 2: continue
        v = group_valid(b, G, k)
        ok = v[:, 1:] & v[:, :-1]                                                   # both g and g+1 inside the sequence
        idx_n = cb.assign(cb.normalise(raw[:, 1:][ok])); idx_c = cb.assign(cb.normalise(raw[:, :-1][ok]))
        u = out['u'][:, :-1][ok]
        p = out['pooled']; pprev = torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], 1)
        U.append(u.float().cpu()); Pp.append(p[:, :-1][ok].float().cpu())
        P2.append(torch.cat([pprev[:, :-1][ok], p[:, :-1][ok]], -1).float().cpu())
        nxt.append(idx_n.cpu()); cur.append(idx_c.cpu())
        head_ok.append((head(u).argmax(-1) == idx_n).float().cpu())
    return torch.cat(U), torch.cat(Pp), torch.cat(P2), torch.cat(cur), torch.cat(nxt), torch.cat(head_ok)


def family_batches(fam, device, split='dev'):
    """Probes are fit on train-split rows and scored on dev rows (the NCP head was also trained on train data)."""
    if fam == 'pinpad':
        sp = Split(split); n = 1000 if split == 'dev' else 1500
        for s in range(0, n, 128):
            yield sp.batch(np.arange(s, min(s + 128, n)), device=device, pad_to=176)
    else:
        f = BPEFamily(fam, split); bs = 32 if fam == 'text' else 128
        n = (400 if fam == 'text' else min(f.n, 2048)) if split == 'dev' else (600 if fam == 'text' else 3072)
        idx = np.arange(f.n) if split == 'dev' else np.random.default_rng(0).permutation(f.n)
        for s in range(0, n, bs):
            yield f.batch(idx[s:min(s + bs, n)], None, device, pad_to=256 if fam == 'text' else 96)


def fit_mlp(X, y, Xt, yt, classes, steps=600, lr=3e-3, wd=1e-4):
    """Probe with the NCP head's architecture (RMSNorm -> 2d GELU -> classes), fit from scratch."""
    torch.manual_seed(0)
    d = X.shape[1]
    net = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, classes))
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    for _ in range(steps):
        i = torch.randint(0, len(X), (1024,))
        l = torch.nn.functional.cross_entropy(net(X[i]), y[i]); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        return float((net(Xt).argmax(-1) == yt).float().mean())


def main(run, ckpt, device='mps', max_rows=12000, cb_from=None):
    model, arm = load_run(run, device, ckpt); model.eval()
    src, src_arm = model, arm
    if cb_from:
        r2, c2 = (cb_from.split(':') + ['final'])[:2]
        src, src_arm = load_run(r2, device, c2); src.eval()
    kind = train.ARMS[src_arm].get('ncp')
    if kind is None: raise SystemExit('codebook source has no NCP codebook')
    cb, head = src.codebook, src.aux['ncp_head']
    N, S = cb.N, cb.S
    res = dict(codebook_from=cb_from or run)
    def sub(x, n):
        if len(x[0]) <= n: return x
        sel = torch.randperm(len(x[0]), generator=torch.Generator().manual_seed(0))[:n]
        return [a[sel] for a in x]
    for fam in ('pinpad', 'math', 'text'):
        tr = sub(list(collect(model, kind, family_batches(fam, device, 'train'), cb, head, src)), max_rows)
        dv = sub(list(collect(model, kind, family_batches(fam, device, 'dev'), cb, head, src)), max_rows)
        r = dict(rows_train=len(tr[0]), rows_dev=len(dv[0]))
        names = dict(u=0, p=1, p2=2)
        for nm, j in names.items():
            r['acc_' + nm] = float(np.mean([fit_probe(tr[j], tr[4][:, s], dv[j], dv[4][:, s], N) for s in range(S)]))
            r['mlp_' + nm] = float(np.mean([fit_mlp(tr[j], tr[4][:, s], dv[j], dv[4][:, s], N) for s in range(S)]))
        cur, nxt = dv[3], dv[4]
        r['acc_copy'] = float((cur == nxt).float().mean())
        r['acc_majority'] = float(np.mean([np.bincount(tr[4][:, s].numpy(), minlength=N).argmax() == nxt[:, s].numpy() for s in range(S)]))
        r['acc_ncp_head'] = float(dv[5].mean())
        r['code_ppl_next'] = float(np.mean([np.exp(-(lambda q: (q[q > 0] * np.log(q[q > 0])).sum())(np.bincount(nxt[:, s].numpy(), minlength=N) / len(nxt))) for s in range(S)]))
        res[fam] = r; print(fam, {k: round(v, 3) for k, v in r.items()}, flush=True)
    tag = f'_cb{cb_from.split(":")[0]}' if cb_from else ''
    (ROOT / 'runs' / run / f'info_test_{ckpt}{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    cbf = a[a.index('--codebook-from') + 1] if '--codebook-from' in a else None
    main(a[0], a[1], cb_from=cbf)
