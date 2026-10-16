"""Post-hoc analysis of a finished arm (consult #14): (1) pooler weights split by slot, so position and content effects
separate; (2) paired standard errors for the switch-off deltas (per validation row).
python decomp_v4tiny.py <arm> [<arm> ...]   -> runs/<arm>/decomp.json"""
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import model_v4 as M4
import data_v4 as D

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
torch.set_num_threads(2)


def load(arm):
    ck = torch.load(os.path.join(RUNS, arm, "ckpt.pt"), map_location="cpu", weights_only=False)
    m = M4.HRMThinkerV4(ck["cfg"]); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["args"]


@torch.no_grad()
def per_row_ntp(m, t, d, canon, **kw):
    out = []
    for a in range(0, len(t), 8):
        ids, doc, ei = D.to_batch(t[a:a + 8], d[a:a + 8], canon, m.ecfg)
        lg, _ = M4.forward(m, ids, doc, ei, **kw)
        tm = M4.target_mask(ids, doc)
        tgt = ids[:, 1:].masked_fill(~tm, -100)
        ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), tgt, ignore_index=-100, reduction="none")
        out += list(zip(ce.sum(1).tolist(), tm.sum(1).tolist()))
    return np.array(out)


@torch.no_grad()
def alphas(m, t, d, canon):
    A, I = [], []
    for a in range(0, len(t), 8):
        ids, doc, ei = D.to_batch(t[a:a + 8], d[a:a + 8], canon, m.ecfg)
        _, aux = M4.forward(m, ids, doc, ei)
        A.append(aux["enc"]["pool_attn"]); I.append(ids)
    return torch.cat(A), torch.cat(I)


def slot_split(A, ids, k, mask_a, mask_b):
    """per slot: mean alpha (averaged over heads) for tokens in class a vs class b, on FULL chunks only."""
    B, M, _, H = A.shape
    real = (ids != 0).view(B, M, k)
    full = real.all(-1, keepdim=True).expand_as(real)
    a_ = A.mean(-1)
    ma, mb = mask_a.view(B, M, k) & full, mask_b.view(B, M, k) & full
    rows = []
    for r in range(k):
        va, vb = a_[..., r][ma[..., r]], a_[..., r][mb[..., r]]
        rows.append(dict(slot=r, a=float(va.mean()) if len(va) else None, b=float(vb.mean()) if len(vb) else None,
                         n_a=int(len(va)), n_b=int(len(vb)),
                         ratio=(float(va.mean() / vb.mean()) if len(va) and len(vb) else None)))
    # position share of variance: alpha (head-mean) on full chunks, between-slot vs total
    x = a_[full.view(B, M, k)].view(-1)
    slot_idx = torch.arange(k).repeat(int(len(x) / k))
    means = torch.stack([x[slot_idx == r].mean() for r in range(k)])
    between = means[slot_idx].var(unbiased=False) / x.var(unbiased=False)
    return rows, float(between)


def main():
    tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json"))
    canon = D.Canon(); tr, va = D.TextStore("train"), D.TextStore("val")
    top = torch.zeros(8192, dtype=torch.bool); top[torch.as_tensor(np.argsort(-D.token_freq(tr))[:100])] = True
    probes_va = D.TinyProbes(tok, D.filler_sentences(tr, tok, seed=99), "val")
    for arm in sys.argv[1:]:
        m, a = load(arm)
        k, T = a["k"], a["T"]
        res = dict(arm=arm)
        vt, vd = D.val_text_rows(va, T, k, n_rows=64)
        # paired switch-off deltas
        base = per_row_ntp(m, vt, vd, canon)
        res["val_ntp"] = float(base[:, 0].sum() / base[:, 1].sum())
        res["base_rows"] = base.tolist()                                     # per-row (sum nats, count): paired cross-arm SEs
        inter = dict(core_off=dict(core_off=True), plan_swap=dict(plan_swap=True), c_off=dict(use_c=False), M1=dict(M=1),
                     loop_off=dict(enc_loop=1), dec_reads_off=dict(dec_reads_off=(0, 1)), engram_off=dict(engram_off=True),
                     enc_shared_off=dict(enc_shared_off=True), core_shared_off=dict(core_shared_off=True))
        e = m.enc
        res["write_shares"] = dict(enc_shared_gain=e.shared_gain.detach().tolist(), enc_shared_w2_norm=float(e.shared.w2.weight.norm()),
                                   core_shared_w2_norm=float(m.core.shared.w2.weight.norm()),
                                   dec_read_Wv_norm=[float(r.Wv.weight.norm()) for r in m.dec.eng])
        if getattr(e.pooler, "ffn", None) is not None:
            res["write_shares"].update(pool_ffn_gain=float(e.pooler.g), pool_ffn_w2_norm=float(e.pooler.ffn.w2.weight.norm()),
                                       pool_ffn_w1_norm=float(e.pooler.ffn.w1.weight.norm()))
        if a["pool"] in ("attnffn", "gconv"):
            inter["pool_ffn_off"] = dict(pool_ffn_off=True)
        inter["enc_engram_off"] = dict(enc_engram_off=True)                 # encoder reads off, decoder reads kept
        if os.environ.get("DECOMP_TRIM"):                                    # trio: only what tier 3 needs
            inter = {kk: v for kk, v in inter.items() if kk in ("core_off", "plan_swap", "pool_ffn_off", "dec_reads_off",
                                                                "enc_engram_off")}
        # synthetic (probe-item) rows: do the decoder reads help where the training mix put half its rows?
        items_s = D.probe_items(probes_va, 64, seed=3000)
        st, sd, _ = D.item_rows(items_s, T, k)
        b0, b1 = per_row_ntp(m, st, sd, canon), per_row_ntp(m, st, sd, canon, dec_reads_off=(0, 1))
        dd = b1[:, 0] / b1[:, 1] - b0[:, 0] / b0[:, 1]
        res["probe_rows_dec_reads_off"] = dict(intact=float(b0[:, 0].sum() / b0[:, 1].sum()),
                                               delta=float(b1[:, 0].sum() / b1[:, 1].sum() - b0[:, 0].sum() / b0[:, 1].sum()),
                                               paired_se=float(dd.std(ddof=1) / math.sqrt(len(dd))))
        res["switch_offs"] = {}
        for nme, kw in inter.items():
            x = per_row_ntp(m, vt, vd, canon, **kw)
            dlt = x[:, 0] / x[:, 1] - base[:, 0] / base[:, 1]                  # per-row delta in nats/token
            res["switch_offs"][nme] = dict(delta=float(x[:, 0].sum() / x[:, 1].sum() - res["val_ntp"]),
                                           row_mean=float(dlt.mean()), paired_se=float(dlt.std(ddof=1) / math.sqrt(len(dlt))))
        if a["pool"] != "linear":
            A, ids = alphas(m, vt, vd, canon)
            istop = top[ids]
            res["text_by_slot"], res["text_position_var_share"] = slot_split(A, ids, k, istop, ~istop)
            items = D.probe_items(probes_va, 96, seed=2000)
            pt, pd, where = D.item_rows(items, T, k)
            A2, ids2 = alphas(m, pt, pd, canon)
            tmpl_ids = set(tok.encode(" The code of").ids) | set(tok.encode(" is").ids) | {tok.encode(".").ids[0]}
            TC, Q, dot = tok.encode(" The code of").ids, tok.encode(" What is the code of").ids, tok.encode(".").ids[0]
            content = torch.zeros_like(ids2, dtype=torch.bool); template = torch.zeros_like(ids2, dtype=torch.bool)
            for idx, (r, off, L) in enumerate(where):
                seq = items[idx]["ids"]
                qpos = next(i for i in range(len(seq)) if seq[i:i + len(Q)] == Q)
                i = 0
                while i < qpos:
                    if seq[i:i + len(TC)] == TC:
                        j = i
                        while j < qpos and seq[j] != dot:
                            j += 1
                        for p_ in range(i, j + 1):
                            (template if seq[p_] in tmpl_ids else content)[r, off + p_] = True
                        i = j + 1
                    else:
                        i += 1
            res["facts_by_slot"], res["facts_position_var_share"] = slot_split(A2, ids2, k, content, template)
        json.dump(res, open(os.path.join(RUNS, arm, os.environ.get("DECOMP_OUT", "decomp.json")), "w"), indent=1)
        print(json.dumps(res, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
