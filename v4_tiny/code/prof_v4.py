"""Profile a few training steps of the tiny model on CPU (memo on), B = 8 vs 16."""
import sys, time, torch
import model_v4 as M4, data_v4 as D
from tokenizers import Tokenizer
import os
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
import speed_v3; speed_v3.enable_block_attention(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json"))
tr = D.TextStore("train"); canon = D.Canon()
pf = D.TinyProbes(tok, D.filler_sentences(tr, tok, n_docs=300), "train")
for B in (8,):
    m = M4.build_model("tiny", k=6); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    st = D.MixedStream(tr, pf, 480, 6, B, 0.5, seed=0)
    bs = [D.to_batch(*st.next(), canon, m.ecfg) for _ in range(4)]
    with torch.no_grad():
        _, c, _, _ = m.enc(*bs[0]); m.core.init_codebooks(c.reshape(-1, 64)[:2000])
    def step(b):
        loss, _, _ = M4.loss_terms(m, *b); opt.zero_grad(); loss.backward(); opt.step()
    step(bs[0])
    t0 = time.time(); real = 0
    for b in bs[1:]:
        step(b); real += int((b[0] != 0).sum())
    dt = time.time() - t0
    print(f"B={B}: {dt/3:.2f} s/step, {real/dt:.0f} real tok/s", flush=True)
    if B == 8:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as p:
            step(bs[1])
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=14))
