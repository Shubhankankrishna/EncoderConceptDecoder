"""Paper-faithful metacontroller on a frozen base (ETA paper §"Unsupervised metacontroller", D.2, E.1, C.4).

Timestep = one environment step. For token-rate control the decision row of step j is the position row
p_{j-1} (index header + 2j, which predicts action a_j); its controller U_j is applied to rows header+2j and
header+2j+1. For group-rate control (ECD 'concept' point) the timestep is a concept group.

    e_j    = residual at the control point (pre-control, frozen base)
    s      = f_emb(e_{1:J})                      non-causal sequence embedding (bi-GRU, mean over valid steps), n_s = 32
    h_j    = GRU(e_j, h_{j-1})                   n_h = 32 (history, causal)
    mu, logvar = f_enc(e_j, h_{j-1}, s)          hidden 64, n_z = 8
    z~_j   = mu + sigma * eps
    beta_j = sigmoid(f_switch(e_j, h_{j-1}, z_{j-1}))   causal (E.1: q(beta | e_{1:t}))
    z_j    = beta_j z~_j + (1 - beta_j) z_{j-1}
    U_j    = A(z_j) B(z_j)  rank 16, hypernetwork hidden 32;  e <- e + U_j e
    L      = action NLL of the controlled frozen base + alpha * sum_j KL(N(mu_j, Sigma_j) || N(0, I))

  python -B meta.py RUN ALPHA SEED [STEPS] [ctrl_point]   -> runs/RUN/meta/a{ALPHA}_s{SEED}.{json,pt}
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from evaluate import load_run

ROOT = Path(__file__).resolve().parent


class MetaController(nn.Module):
    def __init__(self, d, nz=8, nh=32, ns=32, enc_h=64, dec_h=32, r=16):
        super().__init__()
        self.d, self.nz, self.r = d, nz, r
        self.inp = nn.Linear(d, nh)
        self.emb_gru = nn.GRU(nh, ns, batch_first=True, bidirectional=True); self.emb_out = nn.Linear(2 * ns, ns)
        self.gru = nn.GRU(nh, nh, batch_first=True)                                 # history; independent of z
        self.enc = nn.Sequential(nn.Linear(nh + nh + ns, enc_h), nn.GELU(), nn.Linear(enc_h, 2 * nz))
        self.sw_xh = nn.Linear(nh + nh, enc_h); self.sw_z = nn.Linear(nz, enc_h, bias=False)   # f_switch layer 1, split
        self.sw_out = nn.Linear(enc_h, 1)
        self.hyp = nn.Sequential(nn.Linear(nz, dec_h), nn.GELU(), nn.Linear(dec_h, 2 * d * r))
        nn.init.normal_(self.hyp[-1].weight, 0, 0.02 / (dec_h ** 0.5)); nn.init.zeros_(self.hyp[-1].bias)

    def history(self, x):
        """h_{j-1} for every j (zeros at j = 0): the causal history state the encoder and switch read."""
        hs, _ = self.gru(x)
        return torch.cat([torch.zeros_like(hs[:, :1]), hs[:, :-1]], 1)

    def switch(self, x_j, hprev_j, z_prev, pre=None):
        a = (self.sw_xh(torch.cat([x_j, hprev_j], -1)) if pre is None else pre) + self.sw_z(z_prev)
        return torch.sigmoid(self.sw_out(F.gelu(a)))

    def sequence(self, e, mask, sample=True, beta_threshold=None, z_override=None):
        """e: (B, J, d) decision-row residuals; mask (B, J). Returns z (B,J,nz), beta, mu, logvar.
        Only the switch/integration recursion is sequential; history, embedding and encoder run in parallel."""
        B, J, _ = e.shape
        x = self.inp(e)
        hs, _ = self.emb_gru(x)
        m = mask[..., None].float()
        s = self.emb_out((hs * m).sum(1) / m.sum(1).clamp(min=1))                     # (B, ns)
        hp = self.history(x)
        mu, lv = self.enc(torch.cat([x, hp, s[:, None].expand(-1, J, -1)], -1)).chunk(2, -1)
        lv = lv.clamp(-8, 4)
        zt = mu + (torch.randn_like(mu) * (0.5 * lv).exp() if sample else 0)
        pre = self.sw_xh(torch.cat([x, hp], -1))
        z = x.new_zeros(B, self.nz); zs, betas = [], []
        for j in range(J):
            beta = torch.ones(B, 1, device=x.device, dtype=x.dtype) if j == 0 else self.switch(None, None, z, pre[:, j])
            if beta_threshold is not None: beta = (beta > beta_threshold).to(x.dtype)
            z = beta * zt[:, j] + (1 - beta) * z
            zs.append(z); betas.append(beta[:, 0])
        return torch.stack(zs, 1), torch.stack(betas, 1), mu, lv

    def controllers(self, z):
        AB = self.hyp(z)
        A, Bm = AB.split(self.d * self.r, -1)
        return A.view(*z.shape[:-1], self.d, self.r), Bm.view(*z.shape[:-1], self.r, self.d)


def decision_rows(b, max_steps=None):
    """Token-rate decision rows header + 2j for j < n_steps (valid where the row predicts an action)."""
    h = b['header']; L = b['lengths']
    nsteps = ((L - 1 - h) // 2)                                                      # actions in the episode
    J = int(nsteps.max()) if max_steps is None else max_steps
    j = torch.arange(J, device=h.device)[None]
    rows = h[:, None] + 2 * j
    mask = j < nsteps[:, None]
    return rows.clamp(max=b['ids'].shape[1] - 1), mask


class Controlled:
    """Builds the ctrl callable for one batch: reads e at decision rows, runs the metacontroller, writes U."""
    def __init__(self, model, mc, b, sample=True, beta_threshold=None):
        self.model, self.mc, self.b, self.sample, self.thr = model, mc, b, sample, beta_threshold
        self.info = {}

    def __call__(self, e):
        B, N, D = e.shape
        if N == self.b['ids'].shape[1]:                                              # token rate
            rows, mask = decision_rows(self.b)
            ed = torch.gather(e, 1, rows[..., None].expand(-1, -1, D))
            md = next(self.mc.parameters()).device                                  # recurrence may live on CPU
            z, beta, mu, lv = self.mc.sequence(ed.to(md), mask.to(md), self.sample, self.thr)
            A, Bm = self.mc.controllers(z)
            A, Bm = A.to(e.device), Bm.to(e.device)
            u = torch.einsum('bjrd,bjd->bjr', Bm, ed)
            # the same U_j acts on rows header+2j and header+2j+1
            e2 = e.clone()
            for off in (0, 1):
                ok = mask & ((rows + off) <= N - 1)                                 # never double-control the last row
                r = (rows + off).clamp(max=N - 1)
                er = torch.gather(e, 1, r[..., None].expand(-1, -1, D))
                delta = torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, er)) * ok[..., None]
                e2 = e2.scatter_add(1, r[..., None].expand(-1, -1, D), delta)
            self.info = dict(z=z, beta=beta, mu=mu, lv=lv, mask=mask, rows=rows)
            return e2
        # group rate (ECD concept stream): groups whose release row lies in the trajectory
        k = self.model.k; G = N
        rel = torch.arange(G, device=e.device) * k + k - 1
        mask = (rel[None] >= self.b['header'][:, None]) & (rel[None] < self.b['lengths'][:, None] - 1)
        md = next(self.mc.parameters()).device
        z, beta, mu, lv = self.mc.sequence(e.to(md), mask.to(md), self.sample, self.thr)
        A, Bm = self.mc.controllers(z)
        A, Bm = A.to(e.device), Bm.to(e.device)
        delta = torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, e)) * mask[..., None]
        self.info = dict(z=z, beta=beta, mu=mu, lv=lv, mask=mask, rows=rel[None].expand(B, -1))
        return e + delta


def kl_std(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1)


def step_loss(model, mc, b, alpha, sample=True):
    c = Controlled(model, mc, b, sample)
    out = model(b['ids'], ctrl=c)
    lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
    nll = (lp * b['is_act']).sum() / b['is_act'].sum()
    m = c.info['mask'].float().to(c.info['mu'].device)
    kl = ((kl_std(c.info['mu'], c.info['lv']) * m).sum() / b['is_act'].sum().to(m.device)).to(nll.device)  # per action
    return nll, kl, c.info


def switch_metrics(info, b, thr=0.5, tol=2, key='abs_act'):
    """Switch precision/recall/F1 against ground-truth changes of `key` ('abs_act' = pair, 'tgt_col' = colour).
    Units are mapped to environment-step times (token rate: unit j = step j; concept rate: a group's release row r
    -> step (r - header) // 2), and matching uses a tolerance of `tol` env steps, so both control points are scored
    on the same clock (a concept-rate switch can only land every 2 steps)."""
    rows, mask, beta = info['rows'].cpu(), info['mask'].cpu(), info['beta'].detach().cpu()
    aa = b[key].cpu(); hdr = b['header'].cpu()
    lab = torch.gather(aa, 1, (rows + 1).clamp(max=aa.shape[1] - 1))               # latent of the next token
    prev = torch.cat([torch.full_like(lab[:, :1], -9), lab[:, :-1]], 1)
    steps = ((rows - hdr[:, None]) // 2).numpy()
    tp = fp = fn = 0; n_pred = n_true = 0
    for i in range(rows.shape[0]):
        m = mask[i].numpy()
        # true switch times from the per-step label sequence (independent of the unit clock)
        tok_rows = torch.arange(int(hdr[i]), int(b['lengths'][i]) - 1, 2)
        seq = aa[i, (tok_rows + 1).clamp(max=aa.shape[1] - 1)].numpy()
        t_idx = [j for j in range(1, len(seq)) if seq[j] != seq[j - 1] and seq[j] >= 0 and seq[j - 1] >= 0]
        p_idx = [int(steps[i, j]) for j in range(1, len(m)) if m[j] and beta[i, j] > thr]
        n_pred += len(p_idx); n_true += len(t_idx); used = set()
        for p in p_idx:
            hit = [t for t in t_idx if abs(t - p) <= tol and t not in used]
            if hit: tp += 1; used.add(hit[0])
            else: fp += 1
        fn += len(t_idx) - len(used)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    B = rows.shape[0]
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                switches_per_ep=n_pred / B, true_switches_per_ep=n_true / B,
                mean_beta=float((beta * mask).sum() / mask.sum()),
                frac_intermediate=float((((beta > 0.1) & (beta < 0.9)) & mask).sum() / mask.sum()))


@torch.no_grad()
def evaluate_mc(model, mc, split, device, n=1024, alpha=0.0):
    nll_t = kl_t = 0.0; cnt = 0; agg = []
    for s in range(0, n, 128):
        b = split.batch(np.arange(s, min(s + 128, n)), lam_obs=0.0, device=device, pad_to=176)
        nll, kl, info = step_loss(model, mc, b, alpha, sample=False)
        na = int(b['is_act'].sum()); nll_t += float(nll) * na; kl_t += float(kl) * na; cnt += na
        agg.append(dict(**switch_metrics(info, b), **{'colour_' + k: v for k, v in switch_metrics(info, b, key='tgt_col').items()},
                        **{'tol1_' + k: v for k, v in switch_metrics(info, b, tol=1).items() if k in ('f1', 'precision', 'recall')}))
    sw = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
    return dict(nll=nll_t / cnt, rate=kl_t / cnt, **sw)


def main(run, alpha, seed, steps=2000, ctrl_point=None, device='mps', B=64, lr=1e-3, mc_device='cpu'):
    model, arm = load_run(run, device)
    for p in model.parameters(): p.requires_grad_(False)
    if model.kind == 'ECD': model.ctrl_point = ctrl_point or 'concept'
    elif ctrl_point and ctrl_point.startswith('L'): model.ctrl_layer = int(ctrl_point[1:])
    else: ctrl_point = 'mid'
    torch.manual_seed(seed)
    mc = MetaController(model.d).to(mc_device)
    torch.set_num_threads(2)
    opt = torch.optim.AdamW(mc.parameters(), lr, weight_decay=0.03)
    meta, dm = Split('meta'), Split('dev_meta'); s = Sampler(meta, B, seed)
    t0 = time.time(); hist = []
    for step in range(steps):
        b = meta.batch(s.next_idx(), lam_obs=0.0, device=device, pad_to=176)      # fixed shape (MPS graph cache)
        nll, kl, info = step_loss(model, mc, b, alpha)
        if device == 'mps' and step % 25 == 0: torch.mps.empty_cache()
        # (B = 64, 2,000 steps: 128k episodes, ~2 passes over the clean meta split; paper: 64k x 512)
        loss = nll + alpha * kl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == steps - 1:
            sw = switch_metrics(info, b)
            hist.append(dict(step=step, t=round(time.time() - t0, 1), nll=float(nll), kl=float(kl), **sw))
    ev = evaluate_mc(model, mc, dm, device, alpha=alpha)
    tag = f"{model.ctrl_point if model.kind == 'ECD' else ctrl_point}_a{alpha}_s{seed}" + ('' if steps == 2000 else f'_n{steps}')
    outd = ROOT / 'runs' / run / 'meta'; outd.mkdir(exist_ok=True)
    res = dict(run=run, alpha=alpha, seed=seed, steps=steps, ctrl_point=tag.split('_a')[0], eval=ev, hist=hist,
               elapsed=time.time() - t0)
    (outd / f'{tag}.json').write_text(json.dumps(res, indent=1))
    torch.save(mc.state_dict(), outd / f'{tag}.pt')
    print(json.dumps(dict(run=run, tag=tag, **ev, elapsed=round(time.time() - t0))), flush=True)
    return res


if __name__ == '__main__':
    a = sys.argv
    main(a[1], float(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 2000, a[5] if len(a) > 5 and a[5] != '-' else None)
