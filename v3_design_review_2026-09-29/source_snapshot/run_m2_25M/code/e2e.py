"""End-to-end test of Decoder v1: frozen encoder v1.0 -> concept core (v2 stand-in) -> decoder.

  python e2e.py --test                      pipeline unit tests (float64, exact)
  python e2e.py --arm confined --steps N    train one arm on the long-range recall task (harness.py task)
Arms: confined (proposed, D0 = tok + P) | stack (NCP-style control) | confined_read (rejected arm);
--nextlat adds the NextLat auxiliary.  --task single | multi (--n_facts 1 = the 3-token carry task).
Task: 4 facts [F k v S] at the start, ~204 tokens of Markov filler, 4 queries [Q k P P][P P P v]; every answer is
>= 200 tokens after its fact, far beyond the token-path reach (70 confined / 83 stack), so it can only come through
the concept core.  Chance on answers = 1/32.
"""
import argparse, json, math, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import EncoderConfig, Encoder, engram_indices
from concept_core_v2 import ConceptCoreV2
from hrm_decoder import DecoderConfig, Decoder, NextLat, param_groups, plan_index
from harness import MarkovFiller, make_batch, FILL0, V, T


def make_batch_multi(B, filler, g, T_, n_facts=4, L=3):
    """Multi-token copy: each value is a random L-token string that must be written out from ONE plan.
       facts [k v1 v2 v3] x4 | Markov filler | queries [Q k P P][P v1 v2 v3] x4.
       The query chunk's plan is the only source of v1, v2, v3 (offsets 1, 2, 3 of the answer chunk); the decoder must
       unroll one chunk-level plan into 3 tokens.  32^3 strings: nothing can be memorised."""
    from harness import KEYS, VALS, Q_, P_, BOS
    ids = torch.zeros(B, T_, dtype=torch.long); vpos = torch.zeros(B, n_facts * L, dtype=torch.long)
    for b in range(B):
        keys = torch.tensor(KEYS)[torch.randperm(len(KEYS), generator=g)[:n_facts]]
        vals = torch.tensor(VALS)[torch.randint(0, len(VALS), (n_facts, L), generator=g)]
        seq = [BOS, P_, P_, P_]
        for k, v in zip(keys.tolist(), vals.tolist()):
            seq += [k] + v
        seq += filler.sample(T_ - len(seq) - 8 * n_facts, g)
        for qi, j in enumerate(torch.randperm(n_facts, generator=g).tolist()):
            vpos[b, qi * L:(qi + 1) * L] = torch.arange(len(seq) + 5, len(seq) + 5 + L)
            seq += [Q_, keys[j].item(), P_, P_, P_] + vals[j].tolist()
        assert len(seq) == T_, len(seq)
        ids[b] = torch.tensor(seq)
    return ids, vpos


class HRMThinker(nn.Module):
    def __init__(self, ecfg, dcfg, S=16, N=64, nextlat=False):
        super().__init__()
        self.enc, self.core, self.dec = Encoder(ecfg), ConceptCoreV2(ecfg, S, N), Decoder(dcfg)
        self.nl = NextLat(dcfg.d_model) if nextlat else None

    def forward(self, ids, doc, eidx, R, **kw):
        tok, c_hat, _ = self.enc(ids, doc, eidx)
        cdoc = doc[:, ::4]
        u, lc, c_pred = self.core(c_hat, cdoc, R)
        logits, daux = self.dec(tok, u, c_pred, doc, self.enc.embed.weight, **kw)
        return logits, dict(c_hat=c_hat, cdoc=cdoc, logits_c=lc, h=daux["h"], P=daux["P"], u=u)


def configs(d=128, layout="confined", n_dec=2, window=8, V_=V):
    ecfg = EncoderConfig(vocab_size=V_, d_model=d, n_heads=4, n_kv_heads=2, head_dim=d // 4, d_ff=int(2.75 * d),
                         rope_dims=8, engram_rows=4099, engram_dim=16, gr_rank=16)
    plan_in = "read" if layout.endswith("_read") else "add"          # confined | confined_read (rejected arm) | stack
    dcfg = DecoderConfig(d_model=d, n_heads=4, n_kv_heads=2, head_dim=d // 4, d_ff=int(2.75 * d), n_layers=n_dec,
                         window=window, rope_dims=8, gr_rank=16, layout=layout.split("_")[0], plan_in=plan_in)
    return ecfg, dcfg


# ----------------------------------------------------------------------------- pipeline unit tests
def pipeline_tests():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    Vs, Tn, p = 64, 256, 101
    diff = lambda a, b: torch.nonzero((a - b).abs().amax(-1)[0] != 0).flatten()
    for layout in ("confined", "confined_read", "stack"):
        ecfg, dcfg = configs(d=64, layout=layout, n_dec=3, V_=Vs)
        ecfg.engram_rows, ecfg.engram_dim, ecfg.gr_rank, dcfg.gr_rank = 1021, 8, 8, 8
        m = HRMThinker(ecfg, dcfg, S=8, N=16)
        with torch.no_grad():
            for n_, q in m.named_parameters():
                q.add_(torch.randn_like(q) * 0.2)
        m.eval()
        ids = torch.randint(0, Vs, (1, Tn)); doc = torch.zeros(1, Tn, dtype=torch.long)
        ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % Vs
        run = lambda i, d, **kw: m(i, d, engram_indices(i, d, ecfg), 3, **kw)
        with torch.no_grad():
            (a, xa), (b, xb) = run(ids, doc), run(ids2, doc)
            (a0, _), (b0, _) = run(ids, doc, core_off=True), run(ids2, doc, core_off=True)
        ch, ch0 = diff(a, b), diff(a0, b0)
        assert ch.min().item() == p, "causality: nothing before the edit may change"
        assert ch0.max().item() - p == dcfg.reach() and torch.equal(a0[0, p + dcfg.reach() + 1:], b0[0, p + dcfg.reach() + 1:])
        assert ch.max().item() == Tn - 1, "with the core on, the edit must reach the end of the row"
        first_plan = diff(xa["P"], xb["P"]).min().item()
        assert first_plan == 4 * (p // 4) + 3, first_plan
        print(f"PASS [{layout}] causality; core OFF: reach exactly {dcfg.reach()} (encoder 63 + decoder "
              f"{dcfg.reach() - 63}), bit-identical beyond; core ON: reaches the row end; the plan first changes at "
              f"t={first_plan} (end of token {p}'s chunk)")
        doc2 = torch.zeros(1, Tn, dtype=torch.long); doc2[0, 128:] = 1
        ids3 = ids.clone(); ids3[0, 120] = (ids[0, 120] + 5) % Vs
        with torch.no_grad():
            (c1, _), (c2, _) = run(ids, doc2), run(ids3, doc2)
        assert torch.equal(c1[0, 128:], c2[0, 128:])
    print("PASS no leakage across documents through encoder -> core -> decoder (all layouts)")

    # gradient routing at float32
    torch.set_default_dtype(torch.float32); torch.manual_seed(0)
    ecfg, dcfg = configs(); m = HRMThinker(ecfg, dcfg, nextlat=True)
    ids = torch.randint(0, V, (2, T)); doc = torch.zeros(2, T, dtype=torch.long); ei = engram_indices(ids, doc, ecfg)
    with torch.no_grad():
        m.core.vocab.init_from(m.enc(ids, doc, ei)[1])
    logits, x = m(ids, doc, ei, 2)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
    ncp, vq = m.core.losses(x["c_hat"], x["logits_c"], x["cdoc"])
    gw = lambda L, ps: [torch.autograd.grad(L, q, retain_graph=True, allow_unused=True)[0] for q in ps]
    ps = [m.core.vocab.e, m.core.layer.attn.q.weight, m.enc.layers[3].attn.q.weight, m.enc.layers[7].attn.q.weight]
    g = gw(ntp, ps); assert g[0] is None and all(q.abs().sum() > 0 for q in g[1:])
    g = gw(ncp, ps); assert g[0] is None and g[1].abs().sum() > 0 and g[2].abs().sum() > 0
    g = gw(vq, ps); assert g[0].abs().sum() > 0 and all(q is None for q in g[1:])
    lo, _ = m(ids, doc, ei, 2, core_off=True)
    g = gw(F.cross_entropy(lo[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)), ps[1:])
    assert all(q is None or q.abs().sum() == 0 for q in g), "with the core off, NTP must not reach F4/F8"
    print("PASS gradients: NTP -> core + encoder FULL layers F4, F8 (only via the plan), never the codebook | "
          "NCP -> core + encoder | VQ -> codebook only")


# ----------------------------------------------------------------------------- training
EVAL_GROUP = 0          # tokens per answer string (multi task: 3)


def evaluate(m, batches, ecfg, R=4, **kw):
    m.eval(); tot = n = fl = fn = vc = vn = ex = nq = 0.0; off = 0
    with torch.no_grad():
        for ids, vpos, ei in batches:
            doc = torch.zeros_like(ids)
            logits, _ = m(ids, doc, ei, R, **kw)
            nll = -F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
            tot += nll.sum().item(); n += nll.numel()
            f = ids[:, 1:] >= FILL0; fl += nll[f].sum().item(); fn += f.sum().item()
            b = torch.arange(len(ids))[:, None].expand_as(vpos)
            hit = logits[b, vpos - 1].argmax(-1) == ids[b, vpos]
            vc += hit.sum().item(); vn += vpos.numel()
            if EVAL_GROUP:
                h4 = hit.view(len(ids), -1, EVAL_GROUP)
                off = off + h4.float().sum((0, 1)) if torch.is_tensor(off) else h4.float().sum((0, 1))
                ex += h4.all(-1).sum().item(); nq += h4.shape[0] * h4.shape[1]
    m.train()
    out = dict(loss=round(tot / n, 4), filler=round(fl / fn, 4), value_acc=round(vc / vn, 4))
    if nq:
        out.update(exact=round(ex / nq, 4), by_offset=[round(x, 3) for x in (off / nq).tolist()])
    return out


def train(a):
    torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    ecfg, dcfg = configs(layout=a.arm, n_dec=a.n_dec, window=a.window)
    m = HRMThinker(ecfg, dcfg, nextlat=a.nextlat)
    n_dec = sum(q.numel() for q in m.dec.parameters())
    opt = torch.optim.AdamW(param_groups(m, 0.1), lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    filler = MarkovFiller(seed=123)
    gtr, gev = torch.Generator().manual_seed(1000 + a.seed), torch.Generator().manual_seed(999)
    zeros = lambda ids: torch.zeros_like(ids)
    import harness
    mk = make_batch if a.task == 'single' else (lambda B, f, g: make_batch_multi(B, f, g, harness.T, a.n_facts))
    global EVAL_GROUP; EVAL_GROUP = 0 if a.task == 'single' else 3
    evalb = []
    for _ in range(4):
        ids, vpos = mk(32, filler, gev); evalb.append((ids, vpos, engram_indices(ids, zeros(ids), ecfg)))
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        ids, vpos = mk(a.bs, filler, gtr); doc = zeros(ids); ei = engram_indices(ids, doc, ecfg)
        R = int(torch.randint(1, 5, (1,), generator=gtr))
        if step == 1:
            with torch.no_grad():
                m.core.vocab.init_from(m.enc(ids, doc, ei)[1])      # k-means++ codebook init from real concepts
        logits, x = m(ids, doc, ei, R)
        ntp = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        b = torch.arange(len(ids))[:, None].expand_as(vpos)
        val = F.cross_entropy(logits[b, vpos - 1].reshape(-1, V), ids[b, vpos].reshape(-1))
        ncp, vq = m.core.losses(x["c_hat"], x["logits_c"], x["cdoc"])
        loss = ntp + a.value_weight * val + a.alpha * ncp + vq
        lh = lkl = torch.zeros(())
        if m.nl is not None:
            lh, lkl = m.nl.loss(x["h"], ids, doc, m.enc.embed.weight)
            loss = loss + a.lam_h * lh + a.lam_kl * lkl
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        m.core.vocab.restart_dead(x["c_hat"])
        if not torch.isfinite(loss):
            print("NON-FINITE", step); break
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(m, evalb, ecfg); off = evaluate(m, evalb, ecfg, core_off=True)
            rec = dict(step=step, ntp=round(ntp.item(), 4), ncp=round(ncp.item(), 3), nl_h=round(lh.item(), 4),
                       nl_kl=round(lkl.item(), 4), gn=round(float(gn), 2), **ev, value_acc_core_off=off["value_acc"],
                       loss_core_off=off["loss"], sec=round(time.time() - t0))
            log.append(rec); print(json.dumps(rec), flush=True)
    final = dict(intact=evaluate(m, evalb, ecfg), core_off=evaluate(m, evalb, ecfg, core_off=True),
                 u_off=evaluate(m, evalb, ecfg, use_u=False), c_off=evaluate(m, evalb, ecfg, use_c=False),
                 R1=evaluate(m, evalb, ecfg, R=1), R8=evaluate(m, evalb, ecfg, R=8))
    print("FINAL", json.dumps(final), flush=True)
    json.dump(dict(args=vars(a), decoder_params=n_dec, reach=dcfg.reach(), log=log, final=final), open(a.out, "w"), indent=1)
    torch.save(m.state_dict(), a.out.replace(".json", ".pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--arm", default="confined")
    ap.add_argument("--nextlat", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_dec", type=int, default=2)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam_h", type=float, default=1.0)
    ap.add_argument("--lam_kl", type=float, default=1.0)
    ap.add_argument("--value_weight", type=float, default=2.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--task", default="single")
    ap.add_argument("--n_facts", type=int, default=4)
    ap.add_argument("--out", default="run.json")
    a = ap.parse_args()
    import harness; harness.T = a.T                      # row length used by make_batch
    pipeline_tests() if a.test else train(a)
