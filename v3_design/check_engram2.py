"""Exact CPU check (float64, small config): v1 encoder + a second Engram read that shares the table.

Read 2 sits after layer index 4 (first S of the second SSSF block), gathers the SAME table rows as read 1 (the n-gram key is
depth-invariant) and has its own W_k, W_v (zero-init), gated read and write. Variant A: read 2 has NO short conv. Variant B:
read 2 keeps a conv (kernel 4). Measures the L-stream reach by editing one token and finding the last changed position,
and checks W_v2 = 0 => output identical to the one-read encoder at init.
"""
import copy, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_m2_25M", "code"))
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, Engram, GRRead, GRWrite, ShortConv, rms, build_masks, rope_cache,
                         engram_indices)


class EngramRead2(nn.Module):
    def __init__(self, cfg, table, conv):
        super().__init__()
        d, de = cfg.d_model, cfg.n_tables * cfg.engram_dim
        self.table = table                                   # shared nn.Embedding (not a copy)
        self.Wk = nn.Linear(de, d, bias=False)
        self.Wv = nn.Linear(de, d, bias=False); nn.init.zeros_(self.Wv.weight)
        self.read = GRRead(d, 1, cfg.gr_rank, cfg.gr_read_bias)
        self.conv = ShortConv(d, cfg.conv_kernel, zero_init=False) if conv else None
        self.write = GRWrite(d, 2)
        self.d = d

    def forward(self, L, G, e, doc):
        h = self.read([L]); k, v = self.Wk(e), self.Wv(e)
        a = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(self.d))
        y = a * v
        y = F.silu(self.conv(y, doc) if self.conv is not None else y)
        s = self.write(h)
        return L + s[..., 0:1] * y, G + s[..., 1:2] * y


class Encoder2(Encoder):
    def __init__(self, cfg, read2_after=4, conv2=False):
        super().__init__(cfg)
        self.read2_after = read2_after
        self.engram2 = EngramRead2(cfg, self.engram.table, conv2)

    def forward(self, ids, doc, eng_idx):
        c = self.cfg; B, T = ids.shape
        swa, full = build_masks(doc, c.window)
        cos, sin = rope_cache(T, c.rope_dims, c.rope_base, ids.device, self.embed.weight.dtype)
        e0 = rms(self.embed(ids)); L, G = e0, e0
        e = self.engram.table(eng_idx).reshape(B, T, -1)       # gathered ONCE, used by both reads
        for li, layer in enumerate(self.layers):
            L, G = layer(L, G, doc, swa if layer.kind == "S" else full, (cos, sin))
            if li == c.engram_after:
                L, G, _ = self.engram(L, G, eng_idx, doc)
            if li == self.read2_after:
                L, G = self.engram2(L, G, e, doc)
        mix = torch.softmax(self.crc_router(rms(G)), dim=-1)
        m = mix[..., 0:1] * e0 + mix[..., 1:2] * rms(L) + mix[..., 2:3] * rms(G)
        cvec = self.pool(m.reshape(B, T // c.chunk, c.chunk * c.d_model))
        return self.tok_norm(L), math.sqrt(c.dc) * cvec / (cvec.norm(dim=-1, keepdim=True) + c.eps), dict(L=L, G=G)


def reach(enc, cfg, T=256, p=100):
    ids = torch.randint(0, cfg.vocab_size, (1, T)); doc = torch.zeros(1, T, dtype=torch.long)
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % cfg.vocab_size
    with torch.no_grad():
        _, _, a = enc(ids, doc, engram_indices(ids, doc, cfg)); _, _, b = enc(ids2, doc, engram_indices(ids2, doc, cfg))
    dL = (a["L"] - b["L"]).abs().amax(-1)[0]
    return int(torch.nonzero(dL != 0).max()) - p


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    cfg = EncoderConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, rope_dims=8,
                        engram_rows=1021, engram_dim=8)
    base = Encoder(cfg)
    with torch.no_grad():
        for nme, prm in base.named_parameters():
            if "engram.table" not in nme:
                prm.add_(torch.randn_like(prm) * 0.2)
    for conv2 in (False, True):
        enc = Encoder2(cfg, conv2=conv2); enc.load_state_dict(base.state_dict(), strict=False)
        ids = torch.randint(0, 64, (1, 128)); doc = torch.zeros(1, 128, dtype=torch.long)
        with torch.no_grad():
            t1, c1, _ = base(ids, doc, engram_indices(ids, doc, cfg)); t2, c2, _ = enc(ids, doc, engram_indices(ids, doc, cfg))
        same = torch.equal(t1, t2) and torch.equal(c1, c2)
        # randomise read 2 (incl. W_v) to measure its reach contribution
        with torch.no_grad():
            for nme, prm in enc.engram2.named_parameters():
                prm.add_(torch.randn_like(prm) * 0.2)
        shared = enc.engram2.table is enc.engram.table
        print(f"read2 conv={conv2}: identical to 1-read at init (W_v2=0): {same} | table shared: {shared} | "
              f"L reach = {reach(enc, cfg)} (1-read: {reach(base, cfg)}, cfg.reach()={cfg.reach()}) | "
              f"extra params (d 256 real size): see v3_budget R7")
