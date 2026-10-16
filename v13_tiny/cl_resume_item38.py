"""Continual learning on the V13 tiny test (Dataset_v13.md §8.1, §9, §10.2; v13_NOTES.md §10).

    python3 cl.py --method seq|replay|v13 --base runs/base/final.pt --out runs/cl [--steps 100]

All methods branch from the same base checkpoint, see the same ordered incoming rows (equal incoming exposure, §11.1),
and update the same LoRA sites (rank 4: every projection of E2's sublayers, the concept module and the decoder):
  seq     one plain LoRA trained through S1..S6, no replay (sequential adaptation);
  replay  the same plus 20% replay batches from a reservoir of seen rows capped at 8 MiB (§9);
  v13     the design's method: per phase a fresh gated adapter. Wake: an RBF correction bank after C's last layer is
          written with frozen weights (V10.2's learn_episode: value = -eta * dL/du, cap RMS 0.1, sigma 0.3, partition by
          the segment-1 PQ code of m_g, capacity 8,192 rows). Sleep: the bank-corrected frozen model is distilled into
          the phase adapter (KL on scored tokens, gate held open), then the bank is reset. At phase end the adapter's
          gates are fitted: per adapted matrix, a diagonal 4-component GMM of its inputs on the phase data (Phi_pos)
          against one on every earlier family and phase (Phi_neg); the adapter acts only where Phi_pos > Phi_neg.
  frozen  (no command) the base itself.
Gates read the live input of each adapted matrix (LSL's own form, not V10.3's extra adapter-off pass) and are not
smoothed over tokens (DEVIATIONS.md). After every phase the adapter state is saved; the continual matrix A[i, j] is
evaluated afterwards by cl_eval.py.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
import model as M
import ncp as N
import tok as TK
from evaluate import PHASE_TASKS

RANK, SIGMA, ETA, VCAP, BANK_CAP = 4, 0.3, 100.0, 0.1, 8192
LR, REPLAY_EVERY, RES_ROWS, WAKE_BATCHES = 2e-3, 5, 1000, 4
PHASES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
CL_SEED = 7001


# ------------------------------------------------------------------------------------------------ adapters and gates
class GMMGate(nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        for nm, g in (('p', pos), ('n', neg)):
            self.register_buffer(nm + '_logw', torch.tensor(np.log(g.weights_), dtype=torch.float32))
            self.register_buffer(nm + '_mu', torch.tensor(g.means_, dtype=torch.float32))
            self.register_buffer(nm + '_var', torch.tensor(g.covariances_, dtype=torch.float32))

    def logp(self, x, nm):
        lw, mu, var = getattr(self, nm + '_logw'), getattr(self, nm + '_mu'), getattr(self, nm + '_var')
        const = lw - 0.5 * torch.log(2 * math.pi * var).sum(-1)
        z = x.float().unsqueeze(-2) - mu
        return torch.logsumexp(const - 0.5 * (z.square() / var).sum(-1), -1)

    def forward(self, x):
        """The same density ratio as logp's, expanded into two matmuls about a shared centre (DEVIATIONS item 38):
        (x - mu)^2 / var = xc^2 / var - 2 xc mc / var + mc^2 / var with xc = x - c, mc = mu - c. logp's form
        materialises (rows, 4, d) tensors per gate; with three fitted gates it ran S3 at 12.8 GB of driver memory."""
        if getattr(self, '_fc', None) is None or self._fc[1].device != x.device:
            K = self.p_mu.shape[0]
            mu = torch.cat([self.p_mu, self.n_mu]); var = torch.cat([self.p_var, self.n_var])
            lw = torch.cat([self.p_logw, self.n_logw])
            ctr = mu.mean(0); iv = 1.0 / var; mc = mu - ctr
            const = lw - 0.5 * torch.log(2 * math.pi * var).sum(-1) - 0.5 * (mc.square() * iv).sum(-1)
            self._fc = (K, ctr, (-0.5 * iv).t().contiguous(), (mc * iv).t().contiguous(), const)
        K, ctr, qa, qb, const = self._fc
        xc = x.float() - ctr
        sc = xc.square() @ qa + xc @ qb + const
        return (torch.logsumexp(sc[..., :K], -1) > torch.logsumexp(sc[..., K:], -1)).to(x.dtype).unsqueeze(-1)


class LoRALinear(nn.Module):
    """A frozen linear layer with a list of rank-r adapters; adapter i acts with its gate (or always if ungated)."""
    def __init__(self, base):
        super().__init__()
        self.base = base; self.A = nn.ParameterList(); self.Bm = nn.ParameterList(); self.gates = nn.ModuleList()
        self.gated = []; self.enabled = []; self.record = None

    def add(self, rank, gated):
        dev = self.base.weight.device
        self.A.append(nn.Parameter(torch.randn(rank, self.base.in_features, device=dev) / math.sqrt(self.base.in_features)))
        self.Bm.append(nn.Parameter(torch.zeros(self.base.out_features, rank, device=dev)))
        self.gates.append(nn.Identity()); self.gated.append(gated); self.enabled.append(True)
        return len(self.A) - 1

    def forward(self, x):
        y = self.base(x)
        if self.record is not None:                                    # a random subset of rows per call
            x2 = x.detach().reshape(-1, x.shape[-1])
            sel = torch.randperm(x2.shape[0], device=x2.device)[:256]
            self.record.append(x2[sel].cpu())
        for i in range(len(self.A)):
            if not self.enabled[i]: continue
            d = F.linear(F.linear(x, self.A[i]), self.Bm[i])
            if self.gated[i] and not isinstance(self.gates[i], nn.Identity): d = d * self.gates[i](x)
            y = y + d
        return y


def site_linears(model):
    """(name, parent module, attribute) of every adapted projection: E2, concept module, decoder."""
    out = []
    def add(prefix, mod, names):
        for a in names:
            if hasattr(mod, a) and isinstance(getattr(mod, a), (nn.Linear, LoRALinear)): out.append((f'{prefix}.{a}', mod, a))
    for i, layer in enumerate(model.e2):
        add(f'e2.{i}.att', layer.att, ('wqkv', 'wo')); add(f'e2.{i}.ffn', layer.ffn, ('w13', 'w2'))
    for stack, nm in ((model.cm, 'cm'), (model.dec, 'dec')):
        for i, sub in enumerate(stack.subs):
            m = sub.a if isinstance(sub, M.DecLocal) else sub
            add(f'{nm}.{i}', m, ('wqkv', 'wq', 'wkv', 'wo', 'w13', 'w2'))
    return out


def wrap(model):
    mods = {}
    for name, parent, attr in site_linears(model):
        lin = getattr(parent, attr)
        if not isinstance(lin, LoRALinear):
            lin = LoRALinear(lin); setattr(parent, attr, lin)
        mods[name] = lin
    return mods


def new_adapter(mods, gated, seed):
    torch.manual_seed(seed)
    idx = {n: m.add(RANK, gated) for n, m in mods.items()}
    return [p for m in mods.values() for p in (m.A[-1], m.Bm[-1])], idx


def set_adapter(mods, i, on):
    for m in mods.values():
        if i < len(m.enabled): m.enabled[i] = on


def adapter_state(mods):
    st = {}
    for n, m in mods.items():
        st[n] = dict(A=[a.detach().cpu() for a in m.A], B=[b.detach().cpu() for b in m.Bm], gated=list(m.gated),
                     gates=[(g.state_dict() if isinstance(g, GMMGate) else None) for g in m.gates])
    return st


def load_adapter_state(model, st):
    mods = wrap(model); dev = next(model.parameters()).device
    for n, s in st.items():
        m = mods[n]
        for a, b, gd, gs in zip(s['A'], s['B'], s['gated'], s['gates']):
            i = m.add(a.shape[0], gd)
            with torch.no_grad(): m.A[i].copy_(a.to(dev)); m.Bm[i].copy_(b.to(dev))
            if gs is not None:
                g = GMMGate.__new__(GMMGate); nn.Module.__init__(g)
                for k, v in gs.items(): g.register_buffer(k, v.to(dev))
                m.gates[i] = g
    return mods


# ------------------------------------------------------------------------------------------------ correction bank
class Bank:
    """RBF correction bank read once after C's last layer: u += sum_i k_sigma(q, k_i) [pi_i = pi_g] v_i."""
    def __init__(self, d, dev, cap=BANK_CAP, sigma=SIGMA):
        self.K = torch.zeros(cap, d, device=dev); self.V = torch.zeros(cap, d, device=dev)
        self.C = torch.full((cap,), -1, dtype=torch.long, device=dev); self.n = 0; self.cap = cap; self.sigma = sigma
        self.writes = 0; self.overflow = 0

    def reset(self):
        self.K.zero_(); self.V.zero_(); self.C.fill_(-1); self.n = 0

    def read(self, q, codes):
        if self.n == 0: return torch.zeros_like(q)
        K, V, C = self.K[:self.n], self.V[:self.n], self.C[:self.n]
        d2 = (2 - 2 * q @ K.t()).clamp(min=0)
        w = torch.exp(-d2 / (2 * self.sigma ** 2)) * (codes[..., None] == C).to(q.dtype)
        return w @ V


def make_hook(bank, cb, delta=None, store=None):
    def hook(u, m, ctx):
        q = M.rms(u) / math.sqrt(u.shape[-1])
        codes = cb.assign(cb.normalise(m.detach()))[..., 0]
        out = u + bank.read(q.detach(), codes)
        if delta is not None: out = out + delta
        if store is not None: store['q'] = q.detach(); store['codes'] = codes; store['gdoc'] = ctx['gdoc']
        return out
    return hook


def ce_loss(logits, tgt, w, norm):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction='none').view(tgt.shape)
    return (ce * w).sum() / norm


def bank_write(model, bank, cb, b, norm, dev):
    ids, doc, tgt, w = b
    B, T = ids.shape; G = T // model.c.k
    delta = torch.zeros(B, G, model.c.d, device=dev, requires_grad=True); store = {}
    logits = model(ids, doc, u_hook=make_hook(bank, cb, delta, store))[0]
    loss = ce_loss(logits, tgt, w, norm)
    g, = torch.autograd.grad(loss, delta)
    valid = (store['gdoc'] > 0) & (g.norm(dim=-1) > 0)
    keys, codes, grads = store['q'][valid], store['codes'][valid], g[valid]
    room = bank.cap - bank.n
    if len(keys) > room: bank.overflow += len(keys) - room; keys, codes, grads = keys[:room], codes[:room], grads[:room]
    if len(keys):
        maxn = float(grads.norm(dim=-1).max())
        eta = min(ETA, VCAP * math.sqrt(grads.shape[1]) / (maxn + 1e-12))
        sl = slice(bank.n, bank.n + len(keys))
        bank.K[sl] = keys; bank.V[sl] = -eta * grads; bank.C[sl] = codes; bank.n += len(keys); bank.writes += len(keys)
    return float(loss)


# ------------------------------------------------------------------------------------------------ data
def phase_rows(phase, steps, B, pool, cache=None):
    """The fixed ordered incoming batches of a phase (identical for every method): family-homogeneous batches cycling
    through the phase's families, rungs uniform over the trained rungs 0-7. Cached once per (phase, steps, B)."""
    import pickle
    if cache is not None:
        cp = Path(cache) / f'rows_{phase}_{steps}x{B}_{CL_SEED}.pkl'
        if cp.exists(): return pickle.loads(cp.read_bytes())
    import tasks as T
    tl = PHASE_TASKS[phase] if phase != 'S6' else list(T.TASKS)
    fams = {}
    for t in tl: fams.setdefault(T.TASKS[t][0], []).append(t)
    order = list(fams)
    jobs = []
    for s in range(steps):
        f = order[s % len(order)]
        for r in range(B):
            jobs.append((f, s * B + r, 1.0, CL_SEED, phase, fams[f], list(range(8))))
    rows = pool.starmap(D.make_row, jobs, chunksize=4)
    out = [(order[s % len(order)], D.collate(rows[s * B:(s + 1) * B])) for s in range(steps)]
    if cache is not None:
        cp.parent.mkdir(parents=True, exist_ok=True); tmp = cp.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(out)); tmp.replace(cp)
    return out


def s0_rows(n_per_family, B, pool, base_state):
    """Rows of the base stream itself (row k of family f, at its approximate curriculum fraction k / k_end)."""
    jobs = []; rng = np.random.default_rng(CL_SEED)
    for f in ('grid', 'prog', 'state', 'plan'):
        kend = max(1, base_state['k'].get(f, 1))
        for k in rng.choice(kend, size=min(n_per_family, kend), replace=False):
            jobs.append((f, int(k), float(k) / kend, 1301, 'S0', None, None))
    rows = pool.starmap(D.make_row, jobs, chunksize=4)
    return rows


def to_dev(b, dev):
    return (torch.from_numpy(b['ids']).long().to(dev), torch.from_numpy(b['doc']).long().to(dev),
            torch.from_numpy(b['tgt']).long().to(dev), torch.from_numpy(b['w']).to(dev))


def collate_rows(rows):
    return D.collate(rows)


# ------------------------------------------------------------------------------------------------ gates
@torch.no_grad()
def collect_inputs(model, mods, batches, dev, cap_rows=3000, seed=0):
    rng = np.random.default_rng(seed)
    for m in mods.values(): m.record = []
    for b in batches:
        ids, doc, _, _ = to_dev(b, dev); model(ids, doc)
    out = {}
    for n, m in mods.items():
        x = torch.cat(m.record).cpu().double().numpy(); m.record = None
        if len(x) > cap_rows: x = x[rng.choice(len(x), cap_rows, replace=False)]
        out[n] = x
    return out


def fit_gates(model, mods, adapter_i, pos_batches, neg_batches, dev, seed):
    from sklearn.mixture import GaussianMixture
    os.environ['OMP_NUM_THREADS'] = '1'
    pos = collect_inputs(model, mods, pos_batches, dev, seed=seed)
    neg = collect_inputs(model, mods, neg_batches, dev, seed=seed + 1)
    rep = {}
    for n, m in mods.items():
        fit = lambda r: GaussianMixture(4, covariance_type='diag', reg_covar=1e-4, init_params='random_from_data',
                                        n_init=1, random_state=seed, max_iter=60).fit(r)
        gp, gn = fit(pos[n]), fit(neg[n])
        gate = GMMGate(gp, gn).to(dev); m.gates[adapter_i] = gate
        with torch.no_grad():
            op = float(gate(torch.tensor(pos[n], dtype=torch.float32, device=dev)).mean())
            on = float(gate(torch.tensor(neg[n], dtype=torch.float32, device=dev)).mean())
        rep[n] = (round(op, 3), round(on, 3))
    return rep


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['seq', 'replay', 'v13'])
    ap.add_argument('--base', default='runs/base/final.pt'); ap.add_argument('--out', default='runs/cl')
    ap.add_argument('--steps', type=int, default=100); ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--phases', default=','.join(PHASES)); ap.add_argument('--device', default='mps')
    ap.add_argument('--sleep-epochs', type=int, default=1); ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--res-rows', type=int, default=RES_ROWS)                  # smaller only for plumbing tests
    ap.add_argument('--resume', action='store_true', help='v13 only: continue after the last saved phase (item 38)')
    ap.add_argument('--ignore-hold', action='store_true')
    a = ap.parse_args()
    hold = HERE / a.out / 'HOLD'                                       # DEVIATIONS item 38: wait for the resumed run
    if hold.exists() and not a.ignore_hold:
        print(time.strftime('%H:%M:%S'), 'waiting while', hold, 'exists (DEVIATIONS item 38)', flush=True)
        t_h = time.time()
        while hold.exists() and time.time() - t_h < 4 * 3600: time.sleep(30)
        print(time.strftime('%H:%M:%S'), f'hold released after {(time.time() - t_h) / 60:.1f} min', flush=True)
    dev = torch.device(a.device)
    out = HERE / a.out / a.method; out.mkdir(parents=True, exist_ok=True)
    done = []
    if a.resume:
        assert a.method == 'v13', '--resume is for the v13 method only'
        done = [ph for ph in a.phases.split(',') if (out / f'after_{ph}.pt').exists()]
    tk = TK.Tok()
    model = M.V13(M.Cfg(V=tk.vocab)).to(dev)
    z = torch.load(HERE / a.base, map_location=dev, weights_only=False)
    model.load_state_dict(z['model']); model.train(); model.set_ckpt(True)   # train mode only switches checkpointing on
    cb = N.PQCodebook(model.c.d, model.c.pq_S, model.c.pq_N).to(dev); cb.load_state_dict(z['codebook']); cb._init_py = True
    for p in model.parameters(): p.requires_grad_(False)
    mods = wrap(model)
    import multiprocessing as mp
    pool = mp.get_context('spawn').Pool(a.workers, initializer=D._init_worker)
    base_state = torch.load(HERE / 'runs/base/latest.pt', map_location='cpu', weights_only=False)['stream'] \
        if (HERE / 'runs/base/latest.pt').exists() else dict(k={})
    if done:                                                           # the saved phases' gated adapters and Phi_neg rows
        st = torch.load(out / f'after_{done[-1]}.pt', map_location='cpu', weights_only=False)
        mods = load_adapter_state(model, st['adapters'])
        main._prev = []
        for ph in done:
            main._prev += [b for _, b in phase_rows(ph, a.steps, a.B, pool, cache=HERE / a.out / 'rows')[:3]]
        print('resumed after', done[-1], '-', len(next(iter(mods.values())).A), 'adapters per site;',
              len(main._prev), 'earlier-phase batches for Phi_neg', flush=True)
    log = open(out / 'log.jsonl', 'a')
    text = D.TextStream(tk); neg_pool = []
    if a.method == 'v13':                                              # Phi_neg's base-stream and text rows
        s0_gate_rows = s0_rows(8, a.B, pool, base_state); text_rows = [text.row() for _ in range(8)]
        neg_pool = [collate_rows(s0_gate_rows[i:i + a.B]) for i in range(0, len(s0_gate_rows), a.B)] + [collate_rows(text_rows)]
    reservoir, seen = [], 0
    if a.method == 'replay':
        reservoir = s0_rows(a.res_rows // 5, a.B, pool, base_state) + [text.row() for _ in range(a.res_rows // 5)]
        seen = int(sum(base_state['k'].values())) if base_state.get('k') else len(reservoir)
        rng_res = np.random.default_rng(CL_SEED + 5)
    params = None
    if a.method in ('seq', 'replay'):
        params, _ = new_adapter(mods, gated=False, seed=CL_SEED)
        opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    bank = Bank(model.c.d, dev) if a.method == 'v13' else None
    t_all = time.time()
    for phase in a.phases.split(','):
        pi = PHASES.index(phase)                                       # as before for the default --phases
        if phase in done: continue
        t0 = time.time()
        batches = phase_rows(phase, a.steps, a.B, pool, cache=HERE / a.out / 'rows')
        rec = dict(phase=phase, method=a.method, steps=a.steps, rows=a.steps * a.B,
                   tokens=int(sum(b['used'] for _, b in batches)))
        if a.method in ('seq', 'replay'):
            for p in params: p.requires_grad_(True)
            losses, srcs = [], []; t1 = time.time()
            for s, (fam, b) in enumerate(batches):
                lr = LR * min(1.0, (s + 1) / 10)
                for g in opt.param_groups: g['lr'] = lr
                todo = [b]
                if a.method == 'replay' and s % REPLAY_EVERY == REPLAY_EVERY - 1:
                    rf = reservoir[int(rng_res.integers(len(reservoir)))]['family']
                    same = [r for r in reservoir if r['family'] == rf]
                    todo.append(collate_rows([same[int(i)] for i in rng_res.integers(0, len(same), a.B)]))
                for bi, bb in enumerate(todo):
                    ids, doc, tgt, w = to_dev(bb, dev)
                    loss = ce_loss(model(ids, doc)[0], tgt, w, bb['norm'])
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    losses.append(float(loss.detach())); srcs.append('in' if bi == 0 else 'replay')
                if a.method == 'replay':                                   # reservoir sampling over seen rows
                    for r in range(a.B):
                        seen += 1; j = int(rng_res.integers(seen))
                        if j < len(reservoir):
                            reservoir[j] = dict(ids=b['ids'][r], doc=b['doc'][r], tgt=b['tgt'][r], w=b['w'][r],
                                                norm=float(b['w'][r].sum()), used=int((b['doc'][r] > 0).sum()),
                                                scored=0, ndoc=0, stats={}, family=fam)
            for p in params: p.requires_grad_(False)
            inc = [l for l, src in zip(losses, srcs) if src == 'in']; rec['s_train'] = round(time.time() - t1, 1)
            rec.update(online_first=float(np.mean(inc[:3])), online_last=float(np.mean(inc[-3:])),
                       loss_first=float(np.mean(inc[:10])), loss_last=float(np.mean(inc[-10:])))
        else:                                                              # v13: wake / sleep per chunk, gates at the end
            prm, idx = new_adapter(mods, gated=True, seed=CL_SEED + pi)
            ai = len(next(iter(mods.values())).A) - 1
            optv = torch.optim.AdamW(prm, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
            wake_l, sleep_l, step = [], [], 0
            online = []; tw = ts = 0.0
            for c0 in range(0, len(batches), WAKE_BATCHES):
                chunk = batches[c0:c0 + WAKE_BATCHES]
                set_adapter(mods, ai, True); bank.reset()                  # wake on everything consolidated so far
                t1 = time.time()
                for k_, (_, b) in enumerate(chunk):                        # wake: weights frozen, bank writes
                    wl = bank_write(model, bank, cb, to_dev(b, dev), b['norm'], dev); wake_l.append(wl)
                    if k_ == 0: online.append(wl)                          # loss on new rows before any update
                tw += time.time() - t1; t1 = time.time()
                for p in prm: p.requires_grad_(True)
                for ep in range(a.sleep_epochs):                           # sleep: distil teacher(bank) into adapter
                    for _, b in chunk:
                        ids, doc, tgt, w = to_dev(b, dev)
                        with torch.no_grad():                              # teacher = student + bank
                            tl = torch.log_softmax(model(ids, doc, u_hook=make_hook(bank, cb))[0].float(), -1)
                        sl = torch.log_softmax(model(ids, doc)[0].float(), -1)
                        kl = (torch.exp(tl) * (tl - sl)).sum(-1)
                        loss = (kl * w).sum() / b['norm']
                        for g in optv.param_groups: g['lr'] = LR * min(1.0, (step + 1) / 10)
                        optv.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(prm, 1.0); optv.step(); sleep_l.append(float(loss)); step += 1
                for p in prm: p.requires_grad_(False)
                ts += time.time() - t1
            bank.reset(); set_adapter(mods, ai, True); t1 = time.time()
            pos_b = [b for _, b in batches[:12]]
            neg_b = neg_pool + [x for x in getattr(main, '_prev', [])]
            rep = fit_gates(model, mods, ai, pos_b, neg_b, dev, seed=CL_SEED + pi); tg = time.time() - t1
            main._prev = getattr(main, '_prev', []) + [b for _, b in batches[:3]]
            op = np.mean([v[0] for v in rep.values()]); on = np.mean([v[1] for v in rep.values()])
            rec.update(online_first=float(np.mean(online[:3])), online_last=float(np.mean(online[-3:])),
                       wake_loss_first=float(np.mean(wake_l[:4])), wake_loss_last=float(np.mean(wake_l[-4:])),
                       sleep_kl_first=float(np.mean(sleep_l[:4])), sleep_kl_last=float(np.mean(sleep_l[-4:])),
                       bank_writes=bank.writes, bank_overflow=bank.overflow, gate_open_pos=float(op), gate_open_neg=float(on),
                       s_wake=round(tw, 1), s_sleep=round(ts, 1), s_gates=round(tg, 1))
            bank.writes = 0; bank.overflow = 0
        rec['seconds'] = round(time.time() - t0, 1)
        if dev.type == 'mps': rec['mps_driver_gb'] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
        torch.save(dict(adapters=adapter_state(mods), phase=phase, method=a.method, rec=rec), out / f'after_{phase}.pt')
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps(rec), flush=True)
    pool.terminate()
    print('done', a.method, f'{(time.time() - t_all) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
