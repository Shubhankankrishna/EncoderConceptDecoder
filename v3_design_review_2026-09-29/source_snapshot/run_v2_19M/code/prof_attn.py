import time, torch, torch.nn.functional as F
from hrm_encoder import build_masks, same_doc_shift_mask
dev = torch.device("mps"); torch.manual_seed(0)
B, H, T, dh, W = 2, 4, 1024, 64, 8
q = torch.randn(B, H, T, dh, device=dev, requires_grad=True)
k = torch.randn(B, H, T, dh, device=dev, requires_grad=True)
v = torch.randn(B, H, T, dh, device=dev, requires_grad=True)
doc = torch.zeros(B, T, dtype=torch.long, device=dev); doc[:, 600:] = 1
swa, full = build_masks(doc, W)
fmask = torch.zeros(swa.shape, device=dev).masked_fill(~swa, float("-inf"))


def t(name, fn, n=10):
    for _ in range(2):
        fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.mps.synchronize(); print(f"{name:34s} {1000 * (time.time() - t0) / n:7.2f} ms")


def local_attn(q, k, v, doc, W):
    """Exact sliding-window attention over keys t-W+1..t (same doc), without the T x T matrix."""
    Bq, Hq, Tq, d = q.shape
    kp = F.pad(k, (0, 0, W - 1, 0)); vp = F.pad(v, (0, 0, W - 1, 0))
    ku = kp.unfold(2, W, 1)                     # (B, H, T, d, W)
    vu = vp.unfold(2, W, 1)
    s = torch.einsum("bhtd,bhtdw->bhtw", q, ku) / d ** 0.5
    ok = torch.stack([same_doc_shift_mask(doc, W - 1 - j).squeeze(-1) for j in range(W)], -1)  # (B, T, W), j = W-1 is t itself
    s = s.masked_fill(~ok[:, None], float("-inf"))
    return torch.einsum("bhtw,bhtdw->bhtd", s.softmax(-1), vu)


ref = F.scaled_dot_product_attention(q, k, v, attn_mask=swa)
loc = local_attn(q, k, v, doc, W)
print("local == sdpa(swa):", (ref - loc).abs().max().item())
for name, fn in [("sdpa bool swa", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=swa)),
                 ("sdpa float swa", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=fmask)),
                 ("sdpa bool full", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=full)),
                 ("sdpa is_causal", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
                 ("local window", lambda: local_attn(q, k, v, doc, W))]:
    t(name + " fwd", fn)
    t(name + " fwd+bwd", lambda fn=fn: fn().sum().backward())
