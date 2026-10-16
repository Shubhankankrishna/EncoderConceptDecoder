"""HRM THINKER v2 (19M): encoder (v1 design, d 256, Engram 80M) -> concept module D6 (16 experts) -> confined decoder,
plus the training-only NextLat head. Spec: run_m2_25M/model_v2/MODEL_V2_19M.md.

forward(m, ids, doc, eidx, R, core_off, plan_swap, use_u, use_c) keeps v1's signature, so the v1 evaluation code ports.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import compat  # noqa: F401  (EncoderLayer._apply rename, D-001 of v1)
import speed   # noqa: F401  (memoised doc-shift masks inside one encoder forward; identical arithmetic, V-006)
from hrm_encoder import EncoderConfig, Encoder, rms
from concept_core_d6 import ConceptCoreD6
from hrm_decoder import DecoderConfig, Decoder, NextLat

S_SEG, N_CODES = 48, 128
FULL = dict(d=256, V=8192, n_heads=4, n_kv=2, head_dim=64, d_ff=704, d_concept=384, engram_rows=208333, engram_dim=48,
            gr_rank=32, E=16, d_expert=512, top_k=2, nl_hidden=512)
SMALL = dict(d=64, V=8192, n_heads=4, n_kv=2, head_dim=16, d_ff=176, d_concept=96, engram_rows=1021, engram_dim=8,
             gr_rank=8, E=4, d_expert=64, top_k=2, nl_hidden=128)


class HRMThinkerV2(nn.Module):
    def __init__(self, c, S=S_SEG, N=N_CODES):
        super().__init__()
        self.c = dict(c)
        self.ecfg = EncoderConfig(vocab_size=c["V"], d_model=c["d"], n_heads=c["n_heads"], n_kv_heads=c["n_kv"],
                                  head_dim=c["head_dim"], d_ff=c["d_ff"], d_concept=c["d_concept"],
                                  engram_rows=c["engram_rows"], engram_dim=c["engram_dim"], gr_rank=c["gr_rank"],
                                  rope_dims=min(16, c["head_dim"]))
        self.dcfg = DecoderConfig(d_model=c["d"], n_heads=c["n_heads"], n_kv_heads=c["n_kv"], head_dim=c["head_dim"],
                                  d_ff=c["d_ff"], n_layers=4, window=8, layout="confined", plan_in="add",
                                  d_concept=c["d_concept"], gr_rank=c["gr_rank"], rope_dims=min(16, c["head_dim"]))
        self.enc = Encoder(self.ecfg)
        self.core = ConceptCoreD6(dc=c["d_concept"], d=c["d"], H=c["n_heads"], Hk=c["n_kv"], dh=c["head_dim"],
                                  E=c["E"], d_expert=c["d_expert"], top_k=c["top_k"], gr_rank=c["gr_rank"], S=S, N=N)
        self.dec = Decoder(self.dcfg)
        self.dec.plan.Wu = nn.Linear(c["d"], c["d"], bias=False)          # u arrives at d (not in PQ space)
        self.nl = NextLat(c["d"], c["nl_hidden"])


def build_model(small=False, **over):
    c = dict(SMALL if small else FULL); c.update(over)
    m = HRMThinkerV2(c)
    return m, m.ecfg, m.dcfg


def forward(m, ids, doc, eidx, R, core_off=False, plan_swap=False, use_u=True, use_c=True, return_passes=False):
    """Evaluation / generation path (R passes, no windows). Returns logits (B,T,V) and aux."""
    tok, c_hat, eaux = m.enc(ids, doc, eidx)
    cdoc = doc[:, ::4]
    out = m.core(c_hat, cdoc, R, return_passes=return_passes)
    u, lc, c_pred = out[:3]
    if plan_swap:
        u_dec, c_dec = u.roll(1, dims=0), c_pred.roll(1, dims=0)
    else:
        u_dec, c_dec = u, c_pred
    logits, daux = m.dec(tok, u_dec, c_dec, doc, m.enc.embed.weight, core_off=core_off, use_u=use_u, use_c=use_c)
    aux = dict(c_hat=c_hat, cdoc=cdoc, logits_c=lc, u=u, c_pred=c_pred, h=daux["h"], P=daux["P"], enc=eaux, tok=tok)
    if return_passes:
        aux["passes"] = out[3]
    return logits, aux


def nextlat_losses(m, h, logits, ids, doc):
    """NextLat (Teoh et al.), horizon 1, on the decoder's final hidden state h = out_norm(D):
       h_hat_{t+1} = h_t + f(LN[h_t ; rms(sg E x_{t+1})]);  L_h = SmoothL1(sg h_{t+1}, h_hat);
       L_KL = KL(softmax(sg logits_{t+1}) || softmax(h_hat E^T)) with E frozen. The posterior logits are the main
       forward's logits (identical to sg(h_{t+1}) E^T, tied head). Pairs crossing a document boundary are masked."""
    E = m.enc.embed.weight.detach()
    h_t, tgt = h[:, :-1], h[:, 1:].detach()
    e = rms(E[ids[:, 1:]])
    h_hat = h_t + m.nl.f(m.nl.ln(torch.cat([h_t, e], -1)))
    valid = (doc[:, 1:] == doc[:, :-1]).to(h.dtype)
    n = valid.sum().clamp_min(1)
    l_h = (F.smooth_l1_loss(h_hat.to(E.dtype), tgt.to(E.dtype), reduction="none").mean(-1) * valid).sum() / n
    lp_post = F.log_softmax(logits[:, 1:].detach().to(E.dtype), -1)
    lp_prior = F.log_softmax((h_hat @ E.T).to(E.dtype), -1)
    kl = (lp_post.exp() * (lp_post - lp_prior)).sum(-1)
    return l_h, (kl * valid).sum() / n


def param_table(m):
    """v1 convention (PROMPT 2.2): compute = encoder non-embedding + core without codebook + decoder."""
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    enc_all = n(m.enc); table = m.enc.engram.table.weight.numel(); emb = m.enc.embed.weight.numel()
    core = n(m.core); code = m.core.vocab.e.numel(); dec = n(m.dec)
    compute = (enc_all - table - emb) + (core - code) + dec
    return {"encoder_non_embedding": enc_all - table - emb, "engram_table": table, "token_embedding": emb,
            "concept_module": core, "codebook": code, "decoder": dec, "compute_params": compute,
            "stored_total": compute + table + emb + code, "nextlat_psi_training_only": n(m.nl)}


def param_groups(model, weight_decay=0.1):
    """No decay: embeddings, Engram table, temperatures, conv kernels, biases, codebook, null plan, chunk offsets,
    norm gains (zcRMSNorm gammas are decayed in v1's encoder grouping; kept as v1 here) and LayerNorm params."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if (n.endswith(".bias") or "embed." in n or "engram.table" in n or "log_tau" in n or n.endswith(".w")
                or n.endswith("vocab.e") or n.endswith("plan.null") or n.endswith("plan.offset")
                or n.startswith("nl.ln")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


if __name__ == "__main__":
    torch.manual_seed(0)
    m, e, d = build_model(**{"engram_rows": 11})
    t = param_table(m)
    t["engram_table"] = 8 * 208333 * 48
    expect = 18833513
    print({k: f"{v/1e6:.4f}M" for k, v in t.items()})
    print("compute params", t["compute_params"], "expected", expect, "OK" if t["compute_params"] == expect else "MISMATCH")
