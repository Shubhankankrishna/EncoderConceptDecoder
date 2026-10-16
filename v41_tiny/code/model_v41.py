"""v4.1: token-memory retrieval at decoder layer 2, with private plans.

The v4 backbone and initial weights are preserved. Only R's four zero-initialised
log temperatures are added. The parameter-identical stack control reads its
private decoder state for K/V in all four layers; it has no short convolutions.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_v4 as M4
from hrm_encoder import apply_partial_rope, build_masks, rope_cache
from hrm_decoder import ConfinedLayer


class MemoryLayer(ConfinedLayer):
    def __init__(self, cfg, retrieval=False, stack=False):
        super().__init__(cfg)
        self.retrieval, self.stack = retrieval, stack
        self.kind = "R" if retrieval else "S"
        if retrieval:
            self.log_tau = nn.Parameter(torch.zeros(cfg.n_heads))

    def forward(self, D, P, tok, mask, rope):
        c = self.cfg
        B, T, _ = D.shape
        H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
        x = self.read_attn([D])
        memory = x if self.stack else tok
        q = self.q_norm(self.q(x).view(B, T, H, dh).transpose(1, 2))
        k = self.k_norm(self.k(memory).view(B, T, Hk, dh).transpose(1, 2))
        v = self.v(memory).view(B, T, Hk, dh).transpose(1, 2)
        if self.retrieval:
            q = q * self.log_tau.exp()[None, :, None, None]
        else:
            cos, sin = rope
            q = apply_partial_rope(q, cos, sin, c.rope_dims)
            k = apply_partial_rope(k, cos, sin, c.rope_dims)
        k = k.repeat_interleave(H // Hk, 1)
        v = v.repeat_interleave(H // Hk, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, H * dh)
        D = D + self.write_attn(x) * self.o(y * torch.sigmoid(self.gate(x)))
        x = self.read_ffn([D])
        return D + self.write_ffn(x) * self.ffn(x)


class DecoderV41(nn.Module):
    def __init__(self, base, stack=False):
        super().__init__()
        self.cfg, self.plan, self.out_norm = base.cfg, base.plan, base.out_norm
        self.eng, self.reads_after = nn.ModuleList(), ()
        self.stack, self.retrieval_index = stack, 2
        layers = []
        for i, original in enumerate(base.layers):
            if i == 2 or stack:
                # Preserve the original matrices and the RNG state: the only
                # new parameter is log_tau, exactly zero at initialisation.
                with torch.random.fork_rng(devices=[]):
                    layer = MemoryLayer(self.cfg, retrieval=(i == 2), stack=stack)
                missing, extra = layer.load_state_dict(original.state_dict(), strict=False)
                assert missing == (["log_tau"] if i == 2 else []) and not extra
                layers.append(layer)
            else:
                layers.append(original)
        self.layers = nn.ModuleList(layers)

    def masks(self, doc, R_local=False):
        local, full = build_masks(doc, self.cfg.window)
        return local, local if R_local else full

    def forward(self, tok, u, c_pred, doc, E, e=None, core_off=False,
                use_u=True, use_c=True, reads_off=(), R_local=False,
                P_override=None):
        assert e is None, "v4.1 decoder must not receive Engram rows"
        c = self.cfg
        P = self.plan(u, c_pred, doc, core_off, use_u, use_c) if P_override is None else P_override
        local, retrieval = self.masks(doc, R_local)
        rope = rope_cache(tok.shape[1], c.rope_dims, c.rope_base, tok.device, tok.dtype)
        D = tok + P
        sv3 = sys.modules.get("speed_v3")
        old_doc = None if sv3 is None else sv3._CTX.get("doc")
        if sv3 is not None:
            sv3._CTX["doc"] = doc
        try:
            for i, layer in enumerate(self.layers):
                D = layer(D, P, tok, retrieval if i == self.retrieval_index else local, rope)
        finally:
            if sv3 is not None:
                if old_doc is None:
                    sv3._CTX.pop("doc", None)
                else:
                    sv3._CTX["doc"] = old_doc
        h = self.out_norm(D)
        return h @ E.t(), dict(P=P, h=h, dec_gates=[])


def build_model(size="tiny", stack=False, **over):
    cfg = dict(k=6, pool="attn", dec_reads=())
    cfg.update(over)
    assert cfg["pool"] == "attn" and not cfg["dec_reads"]
    m = M4.build_model(size, **cfg)
    m.dec = DecoderV41(m.dec, stack=stack)
    m.c.update(retrieval=True, stack=stack)
    return m


def forward(m, ids, doc, eidx, core_off=False, plan_swap=False, use_u=True,
            use_c=True, M=None, enc_loop=None, engram_off=False,
            enc_shared_off=False, core_shared_off=False, pool_ffn_off=False,
            dec_reads_off=(), enc_engram_off=False, R_local=False, constant_plan=None):
    tok, C, e, eaux = m.enc(ids, doc, eidx, enc_loop=enc_loop,
                          engram_off=engram_off or enc_engram_off,
                          shared_off=enc_shared_off, pool_ffn_off=pool_ffn_off)
    cdoc = doc[:, ::m.k]
    S, states = m.core.run(C, cdoc, M=M, shared_off=core_shared_off)
    u = m.core.out_norm(S)
    q, lc, c_pred = m.core.head(u)
    if plan_swap:
        u, c_pred = u.roll(1, dims=0), c_pred.roll(1, dims=0)
    if constant_plan is not None:
        u = constant_plan[0].to(u).expand_as(u)
        c_pred = constant_plan[1].to(c_pred).expand_as(c_pred)
    if m.c.get("train_no_plan", False):
        use_u = use_c = False
    logits, daux = m.dec(tok, u, c_pred, doc, m.enc.embed.weight,
                        core_off=core_off, use_u=use_u, use_c=use_c, R_local=R_local)
    return logits, dict(c_hat=C, cdoc=cdoc, q=q, logits_c=lc, u=u,
                        c_pred=c_pred, states=states, enc=eaux, tok=tok, **daux)


def loss_terms(m, ids, doc, eidx, count_hits=True):
    # Reuse the exactly same objective while dispatching to the v4.1 graph.
    logits, aux = forward(m, ids, doc, eidx)
    nonempty, complete, pair_ok = M4.chunk_masks(ids, doc, m.k)
    vqc, vqf, n0, n = m.core.vq_terms(aux["c_hat"], complete, count_hits)
    cec, cef = m.core.ncp_terms(aux["q"], aux["logits_c"], n0, n, pair_ok)
    tm = M4.target_mask(ids, doc)
    tgt = ids[:, 1:].masked_fill(~tm, -100)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                          tgt.reshape(-1), ignore_index=-100, reduction="sum")
    terms = dict(ntp=ntp/tm.sum().clamp_min(1),
                 ce_coarse=cec/pair_ok.sum().clamp_min(1),
                 ce_fine=cef/pair_ok.sum().clamp_min(1),
                 vq_coarse=vqc/complete.sum().clamp_min(1),
                 vq_fine=vqf/complete.sum().clamp_min(1))
    loss = terms["ntp"] + .05*terms["ce_coarse"] + .1*terms["ce_fine"] + terms["vq_coarse"] + terms["vq_fine"]
    aux.update(complete=complete, nonempty=nonempty,
               pair_ok=pair_ok, n0=n0, n=n)
    return loss, {k: float(v.detach()) for k, v in terms.items()}, aux


param_table = M4.param_table
param_groups = M4.param_groups
