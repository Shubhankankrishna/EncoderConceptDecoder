import time, torch
from model import build_model
from hrm_encoder import build_masks, rope_cache
dev = torch.device("mps"); torch.manual_seed(0)
m, e, _ = build_model(); m = m.to(dev).train()
B, T = 2, 1024
doc = torch.zeros(B, T, dtype=torch.long, device=dev)
swa, full = build_masks(doc, 8); rope = rope_cache(T, 16, 1e4, dev, torch.float32)
x = torch.randn(B, T, 256, device=dev, requires_grad=True)
S0, F3 = m.enc.layers[0], m.enc.layers[3]


def t(name, fn, n=10):
    for _ in range(2):
        fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.mps.synchronize(); print(f"{name:30s} {1000 * (time.time() - t0) / n:7.2f} ms")


def fb(L, msk):
    y = L.attn(x, doc, msk, rope); y.sum().backward()


t("S attn fwd", lambda: S0.attn(x, doc, swa, rope))
t("F attn fwd", lambda: F3.attn(x, doc, full, rope))
t("S attn fwd+bwd", lambda: fb(S0, swa))
t("F attn fwd+bwd", lambda: fb(F3, full))
t("S conv_q only", lambda: S0.attn.conv_q(S0.attn.q(x), doc))
t("S ffn fwd", lambda: S0.ffn(x))
t("S layer fwd", lambda: S0(x, x, doc, swa, rope))
