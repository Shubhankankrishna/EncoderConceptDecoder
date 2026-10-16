"""CPU <-> MPS parity (draft r2 §8.5 step 6): fp32 MPS forward and gradients against the CPU reference, at the FULL
config (B 1, T 1024, tiny Engram table so CPU memory stays small), with and without the speed paths.
Tolerance: max |d logits| < 1e-3 (the v1 mps_smoke_test bar), gradient relative error < 1e-2. Exit 1 on failure.

  python mps_check.py
"""
import copy, sys, time
import torch, torch.nn.functional as F
import model as M
import speed_v3 as SV
from model import build_model, rms
from hrm_encoder import engram_indices


def run(m, ids, doc, ei, dev):
    m.zero_grad(set_to_none=True)
    ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)
    with SV.step_memo():
        tok, C, _ = m.enc(ids, doc, ei)
        S3, _, Ss = m.core.run(C, doc[:, ::4], "3", short="2b")
        u = m.core.out_norm(S3); lc, cp = m.core.head(u)
        logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1)) + \
            ((rms(Ss) - rms(S3).detach()) ** 2).mean()
    loss.backward()
    g = {n: p.grad.detach().float().cpu() for n, p in m.named_parameters() if p.grad is not None}
    return logits.detach().float().cpu(), g


def main():
    ok_all = True
    torch.manual_seed(0)
    m, ecfg, _ = build_model("full", engram_rows=1021)
    with torch.no_grad():                                               # open every zero-init piece
        for n, p in m.named_parameters():
            if "engram.table" not in n:
                p.add_(torch.randn_like(p) * 0.02)
    B, T = 1, 1024
    ids = torch.randint(2, 8192, (B, T)); doc = torch.zeros(B, T, dtype=torch.long); doc[0, 500:] = 1
    ei = engram_indices(ids, doc, ecfg)
    for blk in (0, 64):
        SV.enable_block_attention(blk)
        t0 = time.time(); lc_, gc = run(m, ids, doc, ei, "cpu"); tc = time.time() - t0
        mm = copy.deepcopy(m).to("mps")
        run(mm, ids, doc, ei, "mps")                                    # warm-up
        t0 = time.time(); lm_, gm = run(mm, ids, doc, ei, "mps"); torch.mps.synchronize(); tm = time.time() - t0
        dl = (lc_ - lm_).abs().max().item()
        rel = max(((gc[k] - gm[k]).norm() / gc[k].norm().clamp_min(1e-12)).item() for k in gc if gc[k].norm() > 0)
        worst = max((k for k in gc if gc[k].norm() > 0), key=lambda k: ((gc[k] - gm[k]).norm() / gc[k].norm()).item())
        ok = dl < 1e-3 and rel < 1e-2
        ok_all &= ok
        print(f"block_attn={blk:3d}: max|d logits| {dl:.2e}, max grad rel err {rel:.2e} ({worst}), cpu {tc:.1f}s mps {tm:.2f}s "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        del mm
        torch.mps.empty_cache()
    SV.enable_block_attention(0)
    print("MPS PARITY", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
