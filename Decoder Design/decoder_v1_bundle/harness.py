"""Verification harness for Encoder v1.0: encoder + stand-in concept core + stand-in decoder.

Task (synthetic, V=512, T=256, one document per row):
  [BOS P P P] [F k v S]x4 facts | ~204 tokens of local Markov filler | 4 queries: [Q k P P][P P P v]
The value v is >200 tokens after its fact, far beyond the token-path reach (encoder 63 + decoder 20 = 83),
so it can only be predicted through the concept core (two-stream arm) or through G (bypass arm).

Arms:
  two_stream : decoder reads L (the frozen design)
  bypass     : decoder reads G instead (NCP-style token path; the F6 failure mode)
Eval ablation: plan zeroed ("core off").
"""
import math, time, json, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, engram_indices, param_groups, rms, ZCRMSNorm,
                         GatedAttention, SwiGLU, ShortConv, build_masks, rope_cache)

PAD, F_, S_, Q_, P_, BOS = 0, 1, 2, 3, 4, 5
KEYS = list(range(10, 42))        # 32 keys
VALS = list(range(42, 74))        # 32 values
FILL0, V = 100, 512
N_FACTS, T = 4, 256


class MarkovFiller:
    def __init__(self, seed=0, branching=4):
        g = torch.Generator().manual_seed(seed)
        n = V - FILL0
        self.succ = torch.randint(0, n, (n, branching), generator=g) + FILL0
        self.prob = torch.softmax(torch.randn(n, branching, generator=g) * 1.5, -1)

    def sample(self, length, g):
        out = [int(torch.randint(FILL0, V, (1,), generator=g))]
        for _ in range(length - 1):
            row = out[-1] - FILL0
            j = int(torch.multinomial(self.prob[row], 1, generator=g))
            out.append(int(self.succ[row, j]))
        return out


def make_batch(B, filler, g):
    ids = torch.zeros(B, T, dtype=torch.long)
    vpos = torch.zeros(B, N_FACTS, dtype=torch.long)
    for b in range(B):
        keys = torch.tensor(KEYS)[torch.randperm(len(KEYS), generator=g)[:N_FACTS]]
        vals = torch.tensor(VALS)[torch.randint(0, len(VALS), (N_FACTS,), generator=g)]
        seq = [BOS, P_, P_, P_]
        for k, v in zip(keys.tolist(), vals.tolist()):
            seq += [F_, k, v, S_]
        n_fill = T - len(seq) - 8 * N_FACTS
        seq += filler.sample(n_fill, g)
        order = torch.randperm(N_FACTS, generator=g).tolist()
        for qi, j in enumerate(order):
            vpos[b, qi] = len(seq) + 7
            seq += [Q_, keys[j].item(), P_, P_, P_, P_, P_, vals[j].item()]
        ids[b] = torch.tensor(seq)
    return ids, vpos


class DecLayer(nn.Module):
    """Single-stream PreNorm SWA block reusing the encoder's gated attention (stand-in decoder)."""
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = ZCRMSNorm(cfg.d_model), ZCRMSNorm(cfg.d_model)
        self.attn, self.ffn = GatedAttention(cfg, "S"), SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, h, doc, mask, rope):
        h = h + self.attn(self.n1(h), doc, mask, rope)
        return h + self.ffn(self.n2(h))


class CoreLayer(nn.Module):
    """Stand-in concept core block: full causal attention over concepts (NoPE, gated), SwiGLU."""
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = ZCRMSNorm(cfg.d_model), ZCRMSNorm(cfg.d_model)
        self.attn, self.ffn = GatedAttention(cfg, "F"), SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, z, cdoc, mask):
        z = z + self.attn(self.n1(z), cdoc, mask, None)
        return z + self.ffn(self.n2(z))


class Harness(nn.Module):
    def __init__(self, cfg, arm="two_stream", n_core=2, n_dec=2):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.enc = Encoder(cfg)
        self.core = nn.ModuleList([CoreLayer(cfg) for _ in range(n_core)])
        self.core_norm = ZCRMSNorm(cfg.d_model)
        self.plan = nn.Linear(cfg.dc, cfg.d_model, bias=False)
        self.dec = nn.ModuleList([DecLayer(cfg) for _ in range(n_dec)])
        self.out_norm = ZCRMSNorm(cfg.d_model)

    def forward(self, ids, doc, eng_idx, core_off=False):
        c = self.cfg
        B, T_ = ids.shape
        tok, c_hat, aux = self.enc(ids, doc, eng_idx)
        # concept core (consumes c_hat WITHOUT stop-gradient)
        cdoc = doc[:, ::c.chunk]
        _, cmask = build_masks(cdoc, 1)
        z = c_hat
        for layer in self.core:
            z = layer(z, cdoc, cmask)
        z = self.core_norm(z)
        # causal shift: tokens of chunk n read core output of chunk n-1 (zero for n=0 or doc change)
        zp = F.pad(z, (0, 0, 1, 0))[:, :-1]
        same = (F.pad(cdoc, (1, 0), value=-1)[:, :-1] == cdoc).unsqueeze(-1)
        zp = zp * same
        if core_off:
            zp = torch.zeros_like(zp)
        plan = self.plan(zp).repeat_interleave(c.chunk, dim=1)           # (B, T, d)
        base = tok if self.arm == "two_stream" else self.enc.tok_norm(aux["G"])
        h = base + plan
        swa, _ = build_masks(doc, c.window)
        rope = rope_cache(T_, c.rope_dims, c.rope_base, ids.device, h.dtype)
        for layer in self.dec:
            h = layer(h, doc, swa, rope)
        logits = self.out_norm(h) @ self.enc.embed.weight.T                # tied head
        return logits, aux


def evaluate(model, batches, core_off=False):
    model.eval()
    tot, n, vc, vn, fl, fn = 0.0, 0, 0, 0, 0.0, 0
    with torch.no_grad():
        for ids, vpos, eidx in batches:
            doc = torch.zeros_like(ids)
            logits, _ = model(ids, doc, eidx, core_off=core_off)
            lp = F.log_softmax(logits[:, :-1], -1)
            tgt = ids[:, 1:]
            nll = -lp.gather(-1, tgt[..., None])[..., 0]
            tot += nll.sum().item(); n += nll.numel()
            fmask = (tgt >= FILL0)
            fl += nll[fmask].sum().item(); fn += fmask.sum().item()
            pred = logits.argmax(-1)
            for b in range(ids.shape[0]):
                for p in vpos[b].tolist():
                    vc += int(pred[b, p - 1] == ids[b, p]); vn += 1
    model.train()
    return dict(loss=tot / n, filler_loss=fl / fn, value_acc=vc / vn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="two_stream")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--value_weight", type=float, default=2.0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    cfg = EncoderConfig(vocab_size=V, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32, d_ff=352,
                        rope_dims=8, engram_rows=4099, engram_dim=16, gr_rank=16)
    model = Harness(cfg, a.arm)
    n_par = sum(p.numel() for n, p in model.named_parameters() if "engram.table" not in n)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    filler = MarkovFiller(seed=123)
    gtrain = torch.Generator().manual_seed(1000 + a.seed)
    geval = torch.Generator().manual_seed(999)
    evalb = []
    for _ in range(8):
        ids, vpos = make_batch(32, filler, geval)
        evalb.append((ids, vpos, engram_indices(ids, torch.zeros_like(ids), cfg)))
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        ids, vpos = make_batch(a.bs, filler, gtrain)
        doc = torch.zeros_like(ids)
        eidx = engram_indices(ids, doc, cfg)
        logits, aux = model(ids, doc, eidx)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        # extra term on the long-range answer positions (prediction made at vpos-1)
        bidx = torch.arange(ids.shape[0])[:, None].expand_as(vpos)
        vlog = logits[bidx, vpos - 1]                        # (B, N_FACTS, V)
        loss = loss + a.value_weight * F.cross_entropy(vlog.reshape(-1, V), ids[bidx, vpos].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not torch.isfinite(loss):
            print("NON-FINITE LOSS", step); break
        if step % 250 == 0 or step == a.steps:
            ev = evaluate(model, evalb)
            ev_off = evaluate(model, evalb, core_off=True)
            rec = dict(step=step, train_loss=loss.item(), grad_norm=float(gn), **ev,
                       value_acc_core_off=ev_off["value_acc"], loss_core_off=ev_off["loss"],
                       engram_Wv=float(model.enc.engram.Wv.weight.detach().norm()),
                       mix=[round(x, 3) for x in aux["mix"].mean((0, 1)).tolist()],
                       tau=[round(x, 3) for x in model.enc.layers[3].attn.log_tau.exp().tolist()],
                       sec=round(time.time() - t0))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    json.dump(dict(arm=a.arm, params_non_table=n_par, log=log), open(a.out, "w"), indent=1)
    torch.save(model.state_dict(), a.out.replace(".json", ".pt"))


if __name__ == "__main__":
    main()
