"""LLM-stack baseline for the v4-vs-LLM tiny experiment: a plain decoder-only transformer, pre-norm residual stream,
3 : 1 sliding-window : full attention (pattern SSSF repeated), recurrence placed and sized to mirror v4.

Mirror of v4 (tiny, recommended defaults: pooler attention only, decoder Engram reads off):
  v4   encoder 8 layers SSSFSSSF, F layers applied twice (enc_loop 2)  -> core 4 layers x M = 2 -> decoder 4 layers
  LLM  prelude 8 layers SSSFSSSF, F layers applied twice (enc_loop 2)  -> middle 4 layers SSSF x M = 2 -> coda 4 SSSF
  16 unique layers each, 22 sequential layer applications each. The LLM has no hierarchy: one residual stream over all
  T tokens, no pooler, no concept module, no codebook, no plan, no confinement, no Engram, no shared FFN.
Same primitives as v4 (hrm_encoder.GatedAttention: GQA 4 heads x 16 / 2 KV heads, q/k zcRMSNorm, output gate, short
conv on q/k/v; 'S' = sliding window + partial RoPE, 'F' = full causal + NoPE + per-head temperature, the Llama-4-style
local-RoPE / global-NoPE split; SwiGLU FFN; zcRMSNorm), same tied embedding (input rms-normalised, as v4), same NTP
target mask. Sliding window 60 tokens (T / 8 at T 480), served by speed_v3's block-local path (block 60).
FFN width 304 sets the compute parameters to within 0.5% of v4-rec's 1,212,032.
"""
import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

import compat  # noqa: F401
import speed as _speed
from hrm_encoder import EncoderConfig, GatedAttention, SwiGLU, ZCRMSNorm, rms, build_masks, rope_cache

PAD = 0
TINY = dict(d=64, V=8192, heads=4, kv=2, head_dim=16, rope=8, ff=304, window=60, pattern="SSSF" * 4,
            prelude=8, middle=4, M=2, enc_loop=2)
TEST = dict(TINY, V=64)


class LLMLayer(nn.Module):
    def __init__(self, cfg, kind, ff):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = ZCRMSNorm(cfg.d_model), ZCRMSNorm(cfg.d_model)
        self.attn = GatedAttention(cfg, kind)
        self.ffn = SwiGLU(cfg.d_model, ff)

    def forward(self, x, doc, mask, rope):
        x = x + self.attn(self.n1(x), doc, mask, rope)
        return x + self.ffn(self.n2(x))


class LLMStack(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = dict(c)
        self.k = 1                                                     # no chunking (rows are still padded per v4's k)
        self.ecfg = EncoderConfig(vocab_size=c["V"], d_model=c["d"], n_heads=c["heads"], n_kv_heads=c["kv"],
                                  head_dim=c["head_dim"], d_ff=c["ff"], window=c["window"], rope_dims=c["rope"],
                                  chunk=6, d_concept=c["d"])
        self.embed = nn.Embedding(c["V"], c["d"])
        self.layers = nn.ModuleList([LLMLayer(self.ecfg, kind, c["ff"]) for kind in c["pattern"]])
        self.out_norm = ZCRMSNorm(c["d"])
        P, Mid = c["prelude"], c["middle"]
        assert P + Mid < len(self.layers)
        self.schedule = self.make_schedule()
        self._init(len(self.schedule))

    def make_schedule(self, M=None, enc_loop=None):
        """Sequence of layer indices applied: prelude (F layers enc_loop times), middle block M times, coda."""
        c = self.c
        M = c["M"] if M is None else M
        loops = c["enc_loop"] if enc_loop is None else enc_loop
        P, Mid, N = c["prelude"], c["middle"], len(self.layers)
        s = []
        for i in range(P):
            s += [i] * (loops if self.layers[i].kind == "F" else 1)
        s += list(range(P, P + Mid)) * M
        s += list(range(P + Mid, N))
        return s

    def _init(self, depth):
        nn.init.normal_(self.embed.weight, std=0.02)
        for name, mod in self.named_modules():                         # nn.Linear only (short convs keep their init)
            if isinstance(mod, nn.Linear):
                std = 0.02 / math.sqrt(2 * depth) if (name.endswith("attn.o") or name.endswith("ffn.w2")) else 0.02
                nn.init.normal_(mod.weight, std=std)

    def forward(self, ids, doc, M=None, enc_loop=None, collect=False):
        own = getattr(_speed._tls, "memo", None) is None
        if own:
            _speed._tls.memo = {}
        try:
            c = self.ecfg
            B, T = ids.shape
            swa, full = build_masks(doc, c.window)
            rope = rope_cache(T, c.rope_dims, c.rope_base, ids.device, self.embed.weight.dtype)
            x = rms(self.embed(ids))
            states = [x] if collect else None
            sched = self.schedule if (M is None and enc_loop is None) else self.make_schedule(M, enc_loop)
            for li in sched:
                layer = self.layers[li]
                x = layer(x, doc, swa if layer.kind == "S" else full, rope)
                if collect:
                    states.append(x)
            h = self.out_norm(x)
            return h @ self.embed.weight.t(), dict(h=h, states=states, schedule=sched)
        finally:
            if own:
                _speed._tls.memo = None


def build_model(size="tiny", **over):
    c = dict({"tiny": TINY, "test": TEST}[size]); c.update(over)
    return LLMStack(c)


def target_mask(ids, doc):
    return (doc[:, 1:] == doc[:, :-1]) & (ids[:, 1:] != PAD)


def forward(m, ids, doc, eidx=None, M=None, enc_loop=None, collect=False, **_ignored):
    """Same call shape as model_v4.forward (eidx unused: no Engram)."""
    import speed_v3
    speed_v3._CTX["doc"] = doc
    return m(ids, doc, M=M, enc_loop=enc_loop, collect=collect)


def loss_terms(m, ids, doc, eidx=None, count_hits=True):
    logits, aux = forward(m, ids, doc)
    tm = target_mask(ids, doc)
    tgt = ids[:, 1:].masked_fill(~tm, -100)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                          reduction="sum") / tm.sum().clamp_min(1)
    return ntp, dict(ntp=float(ntp.detach())), aux


def param_table(m):
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    emb = m.embed.weight.numel()
    comp = n(m) - emb
    S = next(l for l in m.layers if l.kind == "S"); Fl = next(l for l in m.layers if l.kind == "F")
    return dict(compute=comp, embedding=emb, layer_S=n(S), layer_F=n(Fl), unique_layers=len(m.layers),
                depth=len(m.schedule), schedule=m.schedule, engram_table=0, codebooks=0)


NO_DECAY_KEYS = ("embed.", "log_tau")


def param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        if nme.endswith(".bias") or nme.endswith(".w") or any(s in nme for s in NO_DECAY_KEYS):   # v4's rule
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


if __name__ == "__main__":
    import model_v4 as M4
    v4 = M4.param_table(M4.build_model("tiny", pool="attn", dec_reads=()))["compute"]
    for ff in (288, 296, 304, 312):
        t = param_table(build_model("tiny", ff=ff))
        print(f"ff {ff}: compute {t['compute']:,} vs v4-rec {v4:,} ({100 * (t['compute'] / v4 - 1):+.2f}%) "
              f"S {t['layer_S']:,} F {t['layer_F']:,} unique {t['unique_layers']} depth {t['depth']}")
    print("schedule", param_table(build_model("tiny"))["schedule"])
