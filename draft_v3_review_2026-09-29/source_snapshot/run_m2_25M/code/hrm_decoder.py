"""HRM THINKER - Decoder v1 (design candidate).  Reference implementation + exact unit tests.

What the decoder receives (and nothing else):
  tok      (B, T, d)   encoder token stream = zcRMSNorm(L), reach 63
  u        (B, M, dc)  concept-core output state per chunk (post-loop only)
  c_pred   (B, M, dc)  NCP prediction of the NEXT chunk, made at each chunk
  doc      (B, T)      document ids
  E        (V, d)      the encoder's embedding matrix (tied output head)

Plan (target-aligned concept shift): position t predicts x_{t+1}; it reads chunk j*(t) = floor((t+1)/4) - 1, the newest
chunk whose last token 4j*+3 <= t.  P_t = W_u u_{j*} + W_c c_pred_{j*} + o_{r(t)}; r(t) = (t+1) mod 4 is the index of
x_{t+1} inside its chunk.  A learned null plan replaces W_u u + W_c c when j* < 0 or chunk j* is in another document.

Layout "confined" (proposed):  D0 = tok + P  (ungated add, as NCP injects its concept).  Each layer:
      x = GRRead(D);  q = W_Q x;  k, v = W_K tok, W_V tok  (keys/values from tok ONLY, window w);  D += s * attn;  FFN on D.
  Nothing ever reads D at another position, so a position sees only its own plan, and reach = 63 + (w - 1) at ANY depth.
  It differs from the NCP-style stack in one line: where keys and values come from.
Layout "stack" (control, NCP-style): h0 = tok + P, then n_layers one-stream gated-residual SWA layers (encoder S-layer
  anatomy, q/k/v convs).  Reach = 63 + n_layers * (w - 1 + 3); later positions read ~reach/4 past plans.
plan_in = "read" (rejected arm, kept to reproduce its runs): P as a separate read-only stream with a floored gate
  instead of the add; it learned the 3-token carry task in 1 of 2 seeds, the add in 2 of 2.
Output: logits = zcRMSNorm(D) @ E^T.   Optional NextLat auxiliary (training only).
Run `python hrm_decoder.py` for the unit tests (CPU, float64, exact equality).
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import (EncoderConfig, rms, ZCRMSNorm, GRRead, GRWrite, GatedAttention, SwiGLU,
                         apply_partial_rope, rope_cache, build_masks)


@dataclass
class DecoderConfig:
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1408
    n_layers: int = 4
    window: int = 8               # keys t-window+1 .. t (over tok)
    rope_dims: int = 16
    rope_base: float = 10000.0
    chunk: int = 4
    d_concept: int = 0            # 0 -> d_model
    gr_rank: int = 32
    gr_read_bias: float = 2.0
    plan_floor: float = 0.5       # "read" arm only: g_P in [floor, 1)
    layout: str = "confined"      # "confined" | "stack"
    plan_in: str = "add"          # confined only: "add" = D0 = tok + P (default) | "read" = rejected arm
    enc_reach: int = 63
    conv_kernel: int = 4          # used by the "stack" control only
    eps: float = 1e-6

    @property
    def dc(self):
        return self.d_concept or self.d_model

    def reach(self):
        """Exact reach of the decoder output WITHOUT the core (tokens back)."""
        if self.layout == "confined":
            return self.enc_reach + (self.window - 1)
        return self.enc_reach + self.n_layers * ((self.window - 1) + (self.conv_kernel - 1))

    def attn_cfg(self):           # EncoderConfig view, to reuse the frozen encoder's attention block in the control arm
        return EncoderConfig(d_model=self.d_model, n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
                             head_dim=self.head_dim, d_ff=self.d_ff, window=self.window, rope_dims=self.rope_dims,
                             rope_base=self.rope_base, conv_kernel=self.conv_kernel, gr_rank=self.gr_rank,
                             gr_read_bias=self.gr_read_bias)


# ----------------------------------------------------------------------------- plan
def plan_index(doc, k=4):
    """j*(t) = floor((t+1)/k) - 1 and validity (j* >= 0 and chunk j* in the same document as t)."""
    B, T = doc.shape
    t = torch.arange(T, device=doc.device)
    j = (t + 1) // k - 1
    jc = j.clamp_min(0)
    ok = (j >= 0)[None, :] & (doc[:, jc * k] == doc)
    return jc[None, :].expand(B, T), ok, (t + 1) % k


class PlanBuilder(nn.Module):
    """P_t = W_u u_{j*} + W_c c_pred_{j*} + o_{r(t)}   (null plan instead of W_u u + W_c c when invalid)."""
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        d, dc = cfg.d_model, cfg.dc
        self.k = cfg.chunk
        self.Wu = nn.Linear(dc, d, bias=False)
        self.Wc = nn.Linear(dc, d, bias=False)
        self.null = nn.Parameter(torch.randn(d))                  # unit RMS, learned
        self.offset = nn.Parameter(torch.zeros(cfg.chunk, d))     # which token of the chunk is being predicted

    def forward(self, u, c_pred, doc, core_off=False, use_u=True, use_c=True):
        B, T = doc.shape
        d = self.null.shape[0]
        idx, ok, r = plan_index(doc, self.k)
        if core_off:
            p = self.null.expand(B, T, d)
        else:
            per_chunk = 0
            if use_u:
                per_chunk = per_chunk + self.Wu(u)
            if use_c:
                per_chunk = per_chunk + self.Wc(c_pred)
            if not torch.is_tensor(per_chunk):
                per_chunk = torch.zeros(B, u.shape[1], d, dtype=u.dtype, device=u.device)
            p = torch.gather(per_chunk, 1, idx[..., None].expand(B, T, d))
            p = torch.where(ok[..., None], p, self.null.expand(B, T, d))
        return p + self.offset[r]


# ----------------------------------------------------------------------------- confined layout
class PlanRead(nn.Module):
    """Gated-residual read of {D, P}:
         x = 1/2 [ g_D * (1 + gamma_D) * rms(D)  +  g_P * rms(P) ]
         g_D = sigmoid(U_D V_D rms(D) + b),   g_P = f + (1 - f) * sigmoid(U_P V_P rms(P) + b),  f = plan_floor.
       P has no learnable gain (a gain could learn to be 0 per channel and close the route)."""
    def __init__(self, d, rank, bias, floor):
        super().__init__()
        self.VD, self.VP = nn.Linear(d, rank, bias=False), nn.Linear(d, rank, bias=False)
        self.UD, self.UP = nn.Linear(rank, d), nn.Linear(rank, d)
        for U in (self.UD, self.UP):
            nn.init.zeros_(U.weight)
            nn.init.constant_(U.bias, bias)
        self.gamma = nn.Parameter(torch.zeros(d))
        self.floor = floor

    def gates(self, D, P):
        nD, nP = rms(D), rms(P)
        gD = torch.sigmoid(self.UD(self.VD(nD)))
        gP = self.floor + (1 - self.floor) * torch.sigmoid(self.UP(self.VP(nP)))
        return nD, nP, gD, gP

    def forward(self, D, P):
        nD, nP, gD, gP = self.gates(D, P)
        return 0.5 * (gD * nD * (1 + self.gamma) + gP * nP)


class ConfinedLayer(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        c, d = cfg, cfg.d_model
        H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
        self.cfg = c
        if c.plan_in == "read":
            self.read_attn = PlanRead(d, c.gr_rank, c.gr_read_bias, c.plan_floor)
            self.read_ffn = PlanRead(d, c.gr_rank, c.gr_read_bias, c.plan_floor)
        else:
            self.read_attn, self.read_ffn = GRRead(d, 1, c.gr_rank, c.gr_read_bias), GRRead(d, 1, c.gr_rank, c.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.q = nn.Linear(d, H * dh, bias=False)                 # queries: from the plan-aware private read
        self.k = nn.Linear(d, Hk * dh, bias=False)                # keys / values: from tok only
        self.v = nn.Linear(d, Hk * dh, bias=False)
        self.o = nn.Linear(H * dh, d, bias=False)
        self.gate = nn.Linear(d, H * dh, bias=False)
        self.q_norm, self.k_norm = ZCRMSNorm(dh), ZCRMSNorm(dh)
        self.ffn = SwiGLU(d, c.d_ff)

    def forward(self, D, P, tok, mask, rope):
        c = self.cfg
        B, T, _ = D.shape
        H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
        read = (lambda R: R(D, P)) if c.plan_in == "read" else (lambda R: R([D]))
        x = read(self.read_attn)
        q = self.q_norm(self.q(x).view(B, T, H, dh).transpose(1, 2))
        k = self.k_norm(self.k(tok).view(B, T, Hk, dh).transpose(1, 2))
        v = self.v(tok).view(B, T, Hk, dh).transpose(1, 2)
        cos, sin = rope
        q, k = apply_partial_rope(q, cos, sin, c.rope_dims), apply_partial_rope(k, cos, sin, c.rope_dims)
        k, v = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, H * dh)
        D = D + self.write_attn(x) * self.o(y * torch.sigmoid(self.gate(x)))
        x = read(self.read_ffn)
        return D + self.write_ffn(x) * self.ffn(x)


# ----------------------------------------------------------------------------- stack layout (control)
class StackLayer(nn.Module):
    """One-stream gated residual, encoder S-layer anatomy (SWA + q/k/v convs + partial RoPE + gated output)."""
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        ac, d = cfg.attn_cfg(), cfg.d_model
        self.read_attn, self.read_ffn = GRRead(d, 1, cfg.gr_rank, cfg.gr_read_bias), GRRead(d, 1, cfg.gr_rank, cfg.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn, self.ffn = GatedAttention(ac, "S"), SwiGLU(d, cfg.d_ff)

    def forward(self, h, doc, mask, rope):
        x = self.read_attn([h])
        h = h + self.write_attn(x) * self.attn(x, doc, mask, rope)
        x = self.read_ffn([h])
        return h + self.write_ffn(x) * self.ffn(x)


# ----------------------------------------------------------------------------- decoder
class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig = DecoderConfig()):
        super().__init__()
        assert cfg.layout in ("confined", "stack")
        self.cfg = cfg
        self.plan = PlanBuilder(cfg)
        L = ConfinedLayer if cfg.layout == "confined" else StackLayer
        self.layers = nn.ModuleList([L(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = ZCRMSNorm(cfg.d_model)

    def forward(self, tok, u, c_pred, doc, E, core_off=False, use_u=True, use_c=True, P_override=None):
        c = self.cfg
        B, T, _ = tok.shape
        P = self.plan(u, c_pred, doc, core_off, use_u, use_c) if P_override is None else P_override
        swa, _ = build_masks(doc, c.window)
        rope = rope_cache(T, c.rope_dims, c.rope_base, tok.device, tok.dtype)
        if c.layout == "confined":
            D = tok + P if c.plan_in == "add" else tok          # D is private to its position in both variants
            for layer in self.layers:
                D = layer(D, P, tok, swa, rope)
        else:
            D = tok + P
            for layer in self.layers:
                D = layer(D, doc, swa, rope)
        h = self.out_norm(D)
        return h @ E.T, dict(h=h, P=P)


class NextLat(nn.Module):
    """Training-only latent dynamics (Teoh et al. 2025), horizon d = 1:
         h_hat_{t+1} = h_t + f_psi(LN[h_t ; rms(E x_{t+1})])
         L_h  = SmoothL1(sg[h_{t+1}], h_hat_{t+1})
         L_KL = KL( p(. | sg h_{t+1}) || p(. | h_hat_{t+1}) ), head frozen (E detached).
       Terms crossing a document boundary are masked."""
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d
        self.ln = nn.LayerNorm(2 * d)
        self.f = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def loss(self, h, ids, doc, E):
        Ed = E.detach()
        h_t, tgt = h[:, :-1], h[:, 1:].detach()
        e = rms(Ed[ids[:, 1:]])
        h_hat = h_t + self.f(self.ln(torch.cat([h_t, e], -1)))
        valid = (doc[:, 1:] == doc[:, :-1]).to(h.dtype)
        l_h = F.smooth_l1_loss(h_hat, tgt, reduction="none").mean(-1)
        lp_post = F.log_softmax(tgt @ Ed.T, -1)
        lp_prior = F.log_softmax(h_hat @ Ed.T, -1)
        kl = (lp_post.exp() * (lp_post - lp_prior)).sum(-1)
        n = valid.sum().clamp_min(1)
        return (l_h * valid).sum() / n, (kl * valid).sum() / n


def param_groups(model, weight_decay=0.1):
    """No decay: embeddings, Engram table, log_tau, conv kernels, biases, codebook, null plan, chunk offsets."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if (n.endswith(".bias") or "embed." in n or "engram.table" in n or "log_tau" in n or n.endswith(".w")
                or n.endswith("vocab.e") or n.endswith("plan.null") or n.endswith("plan.offset")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


# ----------------------------------------------------------------------------- unit tests
def _randomize(m, scale=0.2):
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * scale)


def _small(layout, n_layers, **kw):
    return DecoderConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, n_layers=n_layers,
                         rope_dims=8, gr_rank=8, layout=layout, **kw)


def run_unit_tests(verbose=True):
    torch.manual_seed(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    T, M, V = 128, 32, 50
    E = torch.randn(V, 64)
    doc = torch.zeros(1, T, dtype=torch.long)
    tok = torch.randn(1, T, 64)
    u, cp = torch.randn(1, M, 64), torch.randn(1, M, 64)
    first_change = lambda a, b: torch.nonzero((a - b).abs().amax(-1)[0] != 0).flatten()

    # D1: decoder reach over tok, for several depths
    for layout, pin in (("confined", "read"), ("confined", "add"), ("stack", "read")):
        for nl in (1, 2, 6):
            cfg = _small(layout, nl, enc_reach=0, plan_in=pin)
            dec = Decoder(cfg); _randomize(dec); dec.eval()
            p = 40
            tok2 = tok.clone(); tok2[0, p] += torch.randn(64)
            with torch.no_grad():
                a, _ = dec(tok, u, cp, doc, E); b, _ = dec(tok2, u, cp, doc, E)
            ch = first_change(a, b)
            assert ch.min().item() == p and ch.max().item() - p == cfg.reach(), (layout, nl, ch.max().item() - p, cfg.reach())
    if verbose:
        print("PASS D1 reach over tok: confined (both plan inputs) = w-1 = 7 at depth 1, 2 and 6; stack = 10 per layer (10, 20, 60)")

    # D2: plan confinement. Perturb u of chunk j -> which positions change?
    j = 10
    u2 = u.clone(); u2[0, j] += torch.randn(64)
    own = set(range(4 * j + 3, 4 * j + 7))                          # positions whose j*(t) = j
    for layout, pin in (("confined", "read"), ("confined", "add"), ("stack", "read")):
        cfg = _small(layout, 3, enc_reach=0, plan_in=pin)
        dec = Decoder(cfg); _randomize(dec); dec.eval()
        with torch.no_grad():
            a, _ = dec(tok, u, cp, doc, E); b, _ = dec(tok, u2, cp, doc, E)
        ch = set(first_change(a, b).tolist())
        if layout == "confined":
            assert ch == own, sorted(ch)
        else:
            assert own <= ch and max(ch) == 4 * j + 6 + cfg.reach(), (sorted(ch)[:3], max(ch))
            n_stack = len(ch)
    if verbose:
        print(f"PASS D2 plan confinement: chunk {j}'s plan changes ONLY positions {sorted(own)} (confined, both plan inputs); "
              f"the stack spreads it to {n_stack} positions (= later positions read it as a past plan)")

    # D3: null plan and the target-aligned index
    doc2 = torch.zeros(1, T, dtype=torch.long); doc2[0, 64:] = 1
    idx, ok, r = plan_index(doc2)
    t = torch.arange(T)
    assert torch.all(4 * idx[0][ok[0]] + 3 <= t[ok[0]]), "plan chunk must end at or before t"
    assert torch.all(idx[0][ok[0]] + 1 == (t[ok[0]] + 1) // 4), "u_{j*} predicts the chunk containing x_{t+1}"
    assert ok[0].tolist() == [not (i % 64 < 3) for i in range(T)], "null exactly at the first 3 tokens of each document"
    cfg = _small("confined", 2, enc_reach=0); dec = Decoder(cfg); _randomize(dec); dec.eval()
    u3 = u.clone(); u3[0, :16] += 5.0                                  # edit every chunk of document 0
    with torch.no_grad():
        a, _ = dec(tok, u, cp, doc2, E); b, _ = dec(tok, u3, cp, doc2, E)
    assert torch.equal(a[0, 64:], b[0, 64:]), "document 1 must not read document 0's plans"
    if verbose:
        print("PASS D3 target-aligned index (chunk j* ends <= t, predicts the chunk of x_{t+1}); null plan at the first 3 tokens "
              "of every document; no plan leaks across documents")

    # D4: the plan gate cannot close
    cfg = _small("confined", 2, enc_reach=0, plan_in="read"); dec = Decoder(cfg); _randomize(dec)
    with torch.no_grad():
        for L in dec.layers:
            for R in (L.read_attn, L.read_ffn):
                R.UP.bias.fill_(-1e4)
        _, _, _, gP = dec.layers[0].read_attn.gates(tok, torch.randn(1, T, 64))
        a, _ = dec(tok, u, cp, doc, E); b, _ = dec(tok, u2, cp, doc, E)
    assert gP.min().item() >= 0.5 and not torch.equal(a, b)
    if verbose:
        print("PASS D4 (read arm) plan gate floor: with the gate driven to -1e4 it stays >= 0.5 and the plan still reaches the logits")
    torch.set_default_dtype(old)

    # D5: init values and gradient paths at the M2 size (float32)
    torch.manual_seed(0)
    cfg = DecoderConfig()
    dec = Decoder(cfg); nl = NextLat(cfg.d_model)
    V, T = 8192, 128
    E = nn.Parameter(torch.randn(V, 512) * 0.02)
    ids = torch.randint(0, V, (2, T)); doc = torch.zeros(2, T, dtype=torch.long)
    tok = rms(torch.randn(2, T, 512)).requires_grad_(); u = rms(torch.randn(2, T // 4, 512)).requires_grad_()
    cp = torch.randn(2, T // 4, 512).requires_grad_()
    logits, aux = dec(tok, u, cp, doc, E)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
    R0 = dec.layers[0].read_attn
    gD = torch.sigmoid(R0.U[0](R0.V[0](rms(tok + aux["P"]))))
    sw = dec.layers[0].write_attn(tok)
    assert abs(gD.mean().item() - 0.8808) < 1e-3 and torch.allclose(sw, torch.ones_like(sw))
    g = torch.autograd.grad(ntp, [u, cp, tok, dec.plan.Wu.weight, dec.plan.offset, E], retain_graph=True)
    assert all(x.abs().sum() > 0 for x in g)
    lh, lkl = nl.loss(aux["h"], ids, doc, E)
    gE, gpsi = torch.autograd.grad(lh + lkl, [E, nl.f[-1].weight], retain_graph=True, allow_unused=True)
    assert gE is None and gpsi.abs().sum() > 0
    n_dec = sum(p.numel() for p in dec.parameters())
    n_layer = sum(p.numel() for p in dec.layers[0].parameters())
    n_nl = sum(p.numel() for p in nl.parameters())
    if verbose:
        print(f"PASS D5 init: read gate {gD.mean():.3f}, write scale 1.000; "
              f"NTP at init {ntp.item():.2f} nats (ln V = {math.log(V):.2f})")
        print("     gradients: NTP -> u, c_pred, tok, W_u, offsets, E | NextLat -> psi, never E (frozen head)")
        print(f"     params (M2 config): decoder {n_dec/1e6:.2f}M ({n_layer/1e6:.2f}M per layer x {cfg.n_layers}, "
              f"plan {sum(p.numel() for p in dec.plan.parameters())/1e6:.2f}M) | NextLat psi {n_nl/1e6:.2f}M (training only)")
        print(f"     reach without the core: {cfg.reach()} tokens at any depth (w = {cfg.window})")
    return True


if __name__ == "__main__":
    run_unit_tests()
