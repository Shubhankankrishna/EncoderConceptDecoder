"""S6 Local Support Learning, scaled configuration (draft §10.1, §10.4, §10.6).

LSL-scaled: rank-16 LoRA on q, k, v, o, w12, w3 of every gated block; one gate for q/k/v (block input n1(x)),
one each for o, w12 and w3 on that matrix's own input; JL projection 64; Phi_pos 16 / Phi_neg 32 diagonal GMMs;
temporal smoothing s_t = rho*g_t + (1-rho)*s_{t-1}, applied iff s_t > .5, reset at document boundaries.
Always-open: the same adapters, applied everywhere. The newest bank is held open while it trains (paper Alg. 4).
Execution is full recomputation (no caches), so smoothing state is exact per sequence.
"""
import copy, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from model import D, HEADS, FF, Block, System, Foundation, positions, rotate, freeze, count, chunk_info
from runtime import load_system, atomic_save

GROUPS = ('qkv', 'o', 'w12', 'w3')
IN_DIM = dict(qkv=D, o=D, w12=D, w3=FF)
MATS = dict(qkv=(('q', D), ('k', D), ('v', D)), o=(('o', D),), w12=(('w12', 2 * FF),), w3=(('w3', D),))
JL, POS_K, NEG_K, RANK, SCALE = 64, 16, 32, 16, 2.


def component_logp(h, mu, var, logw):
    """log w_k + log N(h; mu_k, diag var_k) via the quadratic expansion (no (n, K, d) intermediate)."""
    prec = 1 / var
    quad = h.square() @ prec.t() - 2 * h @ (mu * prec).t() + (mu.square() * prec + var.log() + math.log(2 * math.pi)).sum(-1)
    return logw - .5 * quad


def log_density(h, mu, var, logw):
    return torch.logsumexp(component_logp(h, mu, var, logw), -1)


def smooth(hit, docs, rho):
    """s_t = rho*hit_t + (1-rho)*s_{t-1}, reset at document boundaries (and at -1 positions). hit: (b, t) float."""
    if rho >= 1:
        return hit
    b, t = hit.shape; idx = torch.arange(t, device=hit.device)
    lag = (idx[:, None] - idx[None, :]).clamp_min(0).float()
    first = torch.ones_like(docs, dtype=torch.bool); first[:, 1:] = docs[:, 1:] != docs[:, :-1]
    seg = torch.cumsum(first.long(), 1)
    same = (seg[:, :, None] == seg[:, None, :]) & (idx[:, None] >= idx[None, :])[None]
    w = rho * (1 - rho) ** lag
    return torch.einsum('btu,bu->bt', w[None] * same, hit)


class Bank(nn.Module):
    """One phase's rank-16 adapters for one block, plus its Phi_pos per gate group."""
    def __init__(self):
        super().__init__()
        self.A = nn.ParameterDict(); self.B = nn.ParameterDict()
        for g in GROUPS:
            for name, out in MATS[g]:
                self.A[name] = nn.Parameter(torch.randn(RANK, IN_DIM[g]) * .02); self.B[name] = nn.Parameter(torch.zeros(out, RANK))
            self.register_buffer(f'pos_mu_{g}', torch.zeros(POS_K, JL)); self.register_buffer(f'pos_var_{g}', torch.ones(POS_K, JL)); self.register_buffer(f'pos_logw_{g}', torch.full((POS_K,), -math.log(POS_K)))
        self.force = True   # True open, False closed, None fitted gate

    def delta(self, name, x):
        return SCALE * F.linear(F.linear(x, self.A[name]), self.B[name])


class LSLBlock(Block):
    """Block with per-matrix phase banks and gates. Created by converting an existing Block in place."""
    def setup(self, block_index, rho):
        self.lsl_banks = nn.ModuleList(); self.rho = rho; self.telemetry = None
        for g in GROUPS:
            gen = torch.Generator().manual_seed(99000 + 10 * block_index + GROUPS.index(g))
            self.register_buffer(f'proj_{g}', (torch.randn(JL, IN_DIM[g], generator=gen) / JL ** .5).to(self.qkv.weight.device))
            self.register_buffer(f'neg_mu_{g}', torch.zeros(NEG_K, JL, device=self.qkv.weight.device)); self.register_buffer(f'neg_var_{g}', torch.ones(NEG_K, JL, device=self.qkv.weight.device))
            self.register_buffer(f'neg_logw_{g}', torch.full((NEG_K,), -math.log(NEG_K), device=self.qkv.weight.device))
        self.capture_groups = None

    def gate(self, g, x, docs, bank, p):
        valid = (docs >= 0).float()
        if bank.force is True:
            out = valid
        elif bank.force is False:
            out = torch.zeros_like(valid)
        else:
            h = F.linear(x, getattr(self, f'proj_{g}'))
            hit = (log_density(h, getattr(bank, f'pos_mu_{g}'), getattr(bank, f'pos_var_{g}'), getattr(bank, f'pos_logw_{g}')) >
                   log_density(h, getattr(self, f'neg_mu_{g}'), getattr(self, f'neg_var_{g}'), getattr(self, f'neg_logw_{g}'))).float() * valid
            out = (smooth(hit, docs, self.rho) > .5).float() * valid
        if self.telemetry is not None:
            self.telemetry(self, g, p, out, valid)
        return out

    def adapted(self, g, x, docs, base, names):
        if self.capture_groups is not None:
            self.capture_groups(self, g, x, docs)
        outs = [base] if len(names) == 1 else list(base.split(D, -1))
        for p, bank in enumerate(self.lsl_banks):
            gt = self.gate(g, x, docs, bank, p)[..., None]
            for i, name in enumerate(names):
                outs[i] = outs[i] + gt * bank.delta(name, x)
        return outs[0] if len(names) == 1 else torch.cat(outs, -1)

    def forward(self, x, docs, pos=None, cache=None, return_cache=False):
        if cache is not None or return_cache:
            raise ValueError('LSL blocks run by full recomputation (no caches)')
        b, t, _ = x.shape
        pos = positions(docs) if pos is None else pos
        a = self.n1(x)
        qkv = self.adapted('qkv', a, docs, self.qkv(a), ('q', 'k', 'v'))
        q, k, v = qkv.view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, pos), rotate(k, pos)
        causal = torch.arange(t, device=x.device)[None, :] <= torch.arange(t, device=x.device)[:, None]
        mask = (docs[:, :, None] == docs[:, None, :]) & causal
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None]).transpose(1, 2).reshape(b, t, D)
        x = x + self.adapted('o', att, docs, self.o(att), ('o',))
        n2 = self.n2(x)
        g, u = self.adapted('w12', n2, docs, self.w12(n2), ('w12',)).chunk(2, -1)
        mid = F.silu(g) * u
        return x + self.adapted('w3', mid, docs, self.w3(mid), ('w3',))


def gated_blocks(s):
    f = s.foundation
    return list(f.enc) + list(f.con) + list(f.dec)


def convert(s, rho=.3):
    """S6-base: every E3, C4 and D3 block becomes an LSL block (10 gated blocks)."""
    for i, blk in enumerate(gated_blocks(s)):
        blk.__class__ = LSLBlock; blk.setup(i, rho)
    return s


def add_phase(s, force=True):
    for blk in gated_blocks(s):
        bank = Bank().to(blk.qkv.weight.device); bank.force = force; blk.lsl_banks.append(bank)
    return [blk.lsl_banks[-1] for blk in gated_blocks(s)]


def phases(s):
    return len(gated_blocks(s)[0].lsl_banks) if isinstance(gated_blocks(s)[0], LSLBlock) else 0


def set_force(s, phase, force):
    for blk in gated_blocks(s):
        blk.lsl_banks[phase].force = force


# ------------------------------------------------------------------ GMMs
def fit_gmm(X, k, seed, restarts=3, iters=50, floor_frac=1e-4):
    """Diagonal EM (V8 fit_gmm's protocol: 3 random-example starts, <= 50 iterations), in torch on the device."""
    X = X.float(); n, d = X.shape; ref = X.var(0); floor = (floor_frac * ref).clamp_min(1e-6); best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        mu = X[torch.randperm(n, generator=g)[:k].to(X.device)].clone(); var = ref.clamp_min(floor).expand(k, d).clone(); logw = torch.full((k,), -math.log(k), device=X.device)
        prev = -float('inf'); it = 0
        for it in range(1, iters + 1):
            lp = component_logp(X, mu, var, logw)
            norm = torch.logsumexp(lp, 1); score = float(norm.mean()); resp = (lp - norm[:, None]).exp(); nk = resp.sum(0).clamp_min(1e-8)
            mu = resp.t() @ X / nk[:, None]; var = (resp.t() @ X.square() / nk[:, None] - mu.square()).clamp_min(floor); logw = (nk / nk.sum()).log()
            if abs(score - prev) < 1e-4:
                break
            prev = score
        final = float(torch.logsumexp(component_logp(X, mu, var, logw), 1).mean())
        if best is None or final > best[3]:
            best = (mu, var, logw, final, it)
    if not all(torch.isfinite(t).all() for t in best[:3]):
        raise FloatingPointError('nonfinite GMM')
    return dict(mu=best[0], var=best[1], logw=best[2], score=best[3], iterations=best[4])


@torch.no_grad()
def capture(s, batches, cap=65536, seed=0):
    """Reservoir of JL-projected inputs per (block, group) over valid positions of the given batches."""
    store = {}; rng = torch.Generator().manual_seed(seed)
    def hook(blk, g, x, docs):
        key = (id(blk), g); h = F.linear(x[docs >= 0], getattr(blk, f'proj_{g}')).float()
        cur = store.get(key)
        cur = h if cur is None else torch.cat([cur, h])
        if len(cur) > cap:
            cur = cur[torch.randperm(len(cur), generator=rng)[:cap].to(cur.device)]
        store[key] = cur
    for blk in gated_blocks(s):
        blk.capture_groups = hook
    try:
        for b in batches:
            s(b['x'], b['docs'])
    finally:
        for blk in gated_blocks(s):
            blk.capture_groups = None
    return store


def fit_negative(s, batches, seed=99500):
    store = capture(s, batches); info = []
    for i, blk in enumerate(gated_blocks(s)):
        for g in GROUPS:
            r = fit_gmm(store[(id(blk), g)], NEG_K, seed + 10 * i + GROUPS.index(g))
            getattr(blk, f'neg_mu_{g}').copy_(r['mu']); getattr(blk, f'neg_var_{g}').copy_(r['var']); getattr(blk, f'neg_logw_{g}').copy_(r['logw'])
            info.append(dict(block=i, group=g, n=len(store[(id(blk), g)]), score=r['score'], iterations=r['iterations']))
    return info


def fit_positive(s, phase, batches, seed=99700, target=None):
    """Phi_pos for bank `phase` from activations on phase data (paper Alg. 1: after training, on M + adapter).
    target: write into these banks instead (G6.0 proxy with the adapter at zero)."""
    store = capture(s, batches); info = []
    for i, blk in enumerate(gated_blocks(s)):
        bank = blk.lsl_banks[phase] if target is None else target[i]
        for g in GROUPS:
            r = fit_gmm(store[(id(blk), g)], POS_K, seed + 10 * i + GROUPS.index(g))
            getattr(bank, f'pos_mu_{g}').copy_(r['mu']); getattr(bank, f'pos_var_{g}').copy_(r['var']); getattr(bank, f'pos_logw_{g}').copy_(r['logw'])
            info.append(dict(block=i, group=g, n=len(store[(id(blk), g)]), score=r['score'], iterations=r['iterations']))
    return info


@torch.no_grad()
def hit_rates(s, banks, batches, rho=.3, by_position=None):
    """Smoothed gate rates of the given (per-block) banks on batches: per (block, group) means, joint all-open share."""
    sums = {}; joint_open = []; n_tok = 0
    def tel(blk, g, p, out, valid):
        pass
    per_pos = []
    for b in batches:
        x, docs = b['x'], b['docs']
        rows = {}
        def hook(blk, g, xin, d):
            bi = gated_blocks(s).index(blk); bank = banks[bi]
            h = F.linear(xin, getattr(blk, f'proj_{g}'))
            hit = (log_density(h, getattr(bank, f'pos_mu_{g}'), getattr(bank, f'pos_var_{g}'), getattr(bank, f'pos_logw_{g}')) >
                   log_density(h, getattr(blk, f'neg_mu_{g}'), getattr(blk, f'neg_var_{g}'), getattr(blk, f'neg_logw_{g}'))).float() * (d >= 0)
            gate = (smooth(hit, d, rho) > .5).float() * (d >= 0)
            rows[(bi, g)] = (gate, d >= 0)
        for blk in gated_blocks(s):
            blk.capture_groups = hook
        try:
            s(x, docs)
        finally:
            for blk in gated_blocks(s):
                blk.capture_groups = None
        for key, (gate, valid) in rows.items():
            a, c = sums.get(key, (0., 0.)); sums[key] = (a + float(gate.sum()), c + float(valid.sum()))
        tok = [k for k in rows if k[0] < 3 or k[0] >= 7]  # token-position blocks (encoder, decoder) share positions
        all_open = torch.stack([rows[k][0] for k in tok]).prod(0); valid = rows[tok[0]][1]
        joint_open.append(float(all_open.sum()) / max(float(valid.sum()), 1))
    rates = {f'{b}/{g}': v[0] / max(v[1], 1) for (b, g), v in sums.items()}
    vals = np.array(list(rates.values()))
    return dict(per_matrix=rates, mean=float(vals.mean()), p95=float(np.quantile(vals, .95)), max=float(vals.max()), joint_all_open_token_blocks=float(np.mean(joint_open)))


# ------------------------------------------------------------------ checkpoints
def save(path, s, meta):
    atomic_save(path, dict(lsl=meta, model=s.state_dict()))


def load(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False); meta = ck['lsl']
    s, _ = load_system(meta['anchor'], 'cpu')
    if meta['arm'] != 'vq':
        convert(s, meta['rho'])
        for force in meta['forces']:
            add_phase(s, force)
    s.load_state_dict(ck['model'], strict=True); freeze(s)
    if meta['arm'] != 'vq':
        for p, force in enumerate(meta['forces']):
            set_force(s, p, force)
    return s.to(device), meta
