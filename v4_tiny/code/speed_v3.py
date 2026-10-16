"""v3 throughput paths (each is opt-in and must pass its parity test before use; see DECISIONS.md V3-xx).

1. Block-local sliding-window attention for the window-8 layers (encoder S layers, confined decoder layers).
   v1 builds a (B, 1, T, T) mask and runs SDPA over all T keys although each query may see only the last `window`
   positions. Here queries are split into blocks of `block` positions; block b attends to keys
   [b*block - (window-1), (b+1)*block) with the SAME mask rule (causal, t - j < window, same document). Masked entries are
   exactly zero after the softmax in both forms, so the result equals the full form up to float reduction order.
   Scores: T * (block + window - 1) instead of T * T  (1024 * 71 vs 1024 * 1024 at block 64).
2. Micro-step-wide doc-shift mask memo (speed.py's memo, extended from one encoder forward to the whole micro-step:
   the core's and the encoder's ShortConvs share it). Keys are (data_ptr, device, shape, shift), so the tensors must stay
   alive for the memo's lifetime (they do: doc / cdoc / cdoc2 are held by the caller until the backward).
"""
import contextlib
import torch
import torch.nn.functional as F

import hrm_encoder as _E
import hrm_decoder as _D
import speed as _speed
from hrm_encoder import apply_partial_rope

_CFG = dict(block=0)


def _memo():
    return getattr(_speed._tls, "memo", None)                           # dict only inside step_memo / an encoder forward


def _key(t, *extra):
    return (t.data_ptr(), str(t.device), tuple(t.shape), tuple(t.stride())) + extra


_orig_shift = _speed._orig_mask


def memo_shift_mask(doc, i):
    """speed.py's memo with strides in the key (a view such as doc[:, ::4] shares doc's data_ptr)."""
    memo = _memo()
    if memo is None:
        return _orig_shift(doc, i)
    k = _key(doc, "shift", i)
    m = memo.get(k)
    if m is None:
        m = memo[k] = _orig_shift(doc, i)
    return m


_E.same_doc_shift_mask = memo_shift_mask


def block_mask(doc, window, block):
    """(B * nb, 1, block, block + window - 1) bool: query i (global) may attend key j (global) iff 0 <= i - j < window,
    j >= 0 and doc[i] == doc[j]. Cached only inside the current memo scope (never across steps)."""
    memo = _memo()
    key = _key(doc, "block", window, block)
    if memo is not None and key in memo:
        return memo[key]
    B, T = doc.shape
    nb, W = T // block, block + window - 1
    dpad = F.pad(doc, (window - 1, 0), value=-1)                        # (B, T + window - 1); -1 never matches
    kd = dpad.unfold(1, W, block)                                        # (B, nb, W)
    qd = doc.view(B, nb, block)
    qi = torch.arange(block, device=doc.device)[:, None] + (window - 1)  # query position inside the padded key window
    kj = torch.arange(W, device=doc.device)[None, :]
    rel = (qi - kj)                                                      # i - j
    ok_pos = (rel >= 0) & (rel < window)                                 # (block, W)
    m = ok_pos[None, None] & (qd[..., :, None] == kd[..., None, :])      # (B, nb, block, W)
    m = m.reshape(B * nb, 1, block, W)
    if memo is not None:
        memo[key] = m
    return m


def local_sdpa(q, k, v, doc, window, block):
    """q: (B, H, T, dh); k, v: (B, H, T, dh) (already GQA-repeated). Returns (B, H, T, dh)."""
    B, H, T, dh = q.shape
    nb, W = T // block, block + window - 1
    qb = q.view(B, H, nb, block, dh).transpose(1, 2).reshape(B * nb, H, block, dh)
    def kv_blocks(x):
        xp = F.pad(x, (0, 0, window - 1, 0))                             # (B, H, T + w - 1, dh)
        xb = xp.unfold(2, W, block)                                      # (B, H, nb, dh, W)
        return xb.permute(0, 2, 1, 4, 3).reshape(B * nb, H, W, dh)
    y = F.scaled_dot_product_attention(qb, kv_blocks(k), kv_blocks(v), attn_mask=block_mask(doc, window, block))
    return y.view(B, nb, H, block, dh).transpose(1, 2).reshape(B, H, T, dh)


# ---------------------------------------------------------------- patched forwards (arithmetic identical except the SDPA)
_orig_ga_forward = _E.GatedAttention.forward
_orig_cl_forward = _D.ConfinedLayer.forward


def _ga_forward(self, x, doc, mask, rope):
    blk = _CFG["block"]
    T = x.shape[1]
    if self.kind != "S" or not blk or T % blk:
        return _orig_ga_forward(self, x, doc, mask, rope)
    B, T, _ = x.shape
    c = self.cfg
    H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
    q = self.conv_q(self.q(x), doc).view(B, T, H, dh).transpose(1, 2)
    k = self.conv_k(self.k(x), doc).view(B, T, Hk, dh).transpose(1, 2)
    v = self.conv_v(self.v(x), doc).view(B, T, Hk, dh).transpose(1, 2)
    q, k = self.q_norm(q), self.k_norm(k)
    cos, sin = rope
    q = apply_partial_rope(q, cos, sin, c.rope_dims)
    k = apply_partial_rope(k, cos, sin, c.rope_dims)
    rep = H // Hk
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    y = local_sdpa(q, k, v, doc, c.window, blk)
    y = y.transpose(1, 2).reshape(B, T, H * dh)
    return self.o(y * torch.sigmoid(self.gate(x)))


def _cl_forward(self, D, P, tok, mask, rope):
    blk = _CFG["block"]
    c = self.cfg
    B, T, _ = D.shape
    doc = _CTX.get("doc")
    if not blk or T % blk or doc is None or doc.shape != (B, T):
        return _orig_cl_forward(self, D, P, tok, mask, rope)
    H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
    read = (lambda R: R(D, P)) if c.plan_in == "read" else (lambda R: R([D]))
    x = read(self.read_attn)
    q = self.q_norm(self.q(x).view(B, T, H, dh).transpose(1, 2))
    k = self.k_norm(self.k(tok).view(B, T, Hk, dh).transpose(1, 2))
    v = self.v(tok).view(B, T, Hk, dh).transpose(1, 2)
    cos, sin = rope
    q, k = apply_partial_rope(q, cos, sin, c.rope_dims), apply_partial_rope(k, cos, sin, c.rope_dims)
    k, v = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
    y = local_sdpa(q, k, v, doc, c.window, blk).transpose(1, 2).reshape(B, T, H * dh)
    D = D + self.write_attn(x) * self.o(y * torch.sigmoid(self.gate(x)))
    x = read(self.read_ffn)
    return D + self.write_ffn(x) * self.ffn(x)


_CTX = {}
_orig_dec_forward = _D.Decoder.forward


def _dec_forward(self, tok, u, c_pred, doc, E, **kw):
    _CTX["doc"] = doc                                                    # the confined layers need doc for the block mask
    try:
        return _orig_dec_forward(self, tok, u, c_pred, doc, E, **kw)
    finally:
        _CTX.pop("doc", None)


def enable_block_attention(block=64):
    """Route window-8 attention through local_sdpa (block must divide T). block=0 restores v1's full-mask SDPA."""
    _CFG["block"] = block
    if block:
        _E.GatedAttention.forward = _ga_forward
        _D.ConfinedLayer.forward = _cl_forward
        _D.Decoder.forward = _dec_forward
    else:
        _E.GatedAttention.forward = _orig_ga_forward
        _D.ConfinedLayer.forward = _orig_cl_forward
        _D.Decoder.forward = _orig_dec_forward


@contextlib.contextmanager
def step_memo():
    """Doc-shift mask memo for a whole micro-step (encoder + core ShortConvs). Nested encoder forwards reuse it."""
    prev = getattr(_speed._tls, "memo", None)
    own = prev is None
    if own:
        _speed._tls.memo = {}
    try:
        yield
    finally:
        if own:
            _speed._tls.memo = None


# ---------------------------------------------------------------- 3. torch.compile of whole layers (inductor on MPS)
# Found in V3-006: inductor on MPS (torch 2.10) miscompiles the BACKWARD of the key zcRMSNorm gain inside the compiled
# attention (core.layers.*.attn.k_norm.gamma gradient 8.5x too large; logits and every other gradient match to 1e-6).
# Not caused by repeat_interleave or the transposed layout (both tested). Workaround: the k_norm call runs eagerly
# (torch.compiler.disable, one graph break per attention); then all gradients match within 1e-5 (compile_check.py).
def compile_layer(layer, active):
    """Compile layer.forward (dynamic=False); the compiled path is used only when active(layer) is true (training)."""
    import torch as _t
    for mod in layer.modules():
        kn = getattr(mod, "k_norm", None)
        if kn is not None and not getattr(kn, "_eager_pinned", False):
            kn.forward = _t.compiler.disable(kn.forward)
            kn._eager_pinned = True
    eager = layer.forward
    comp = _t.compile(eager, dynamic=False)
    layer.forward = lambda *x, _e=eager, _c=comp, _l=layer, **k: (_c if active(_l) else _e)(*x, **k)
