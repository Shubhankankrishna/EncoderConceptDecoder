"""<= 20 min experiment: does torch.compile (inductor, CPU) speed up a tiny-v4 training step? fp32 parity vs eager."""
import os, sys, time, torch
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
torch.set_num_threads(1)
import speed_v3; speed_v3.enable_block_attention(60)
import model_v4 as M4, data_v4 as D
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json"))
tr = D.TextStore("train"); canon = D.Canon()
pf = D.TinyProbes(tok, D.filler_sentences(tr, tok, n_docs=300), "train")
st = D.MixedStream(tr, pf, 480, 6, 4, 0.5, seed=0)
bs = [D.to_batch(*st.next(), canon, M4.build_model("tiny").ecfg) for _ in range(8)]
def run(compiled):
    torch.manual_seed(0)
    m = M4.build_model("tiny", k=6)
    with torch.no_grad():
        _, c, _, _ = m.enc(*bs[0]); m.core.init_codebooks(c.reshape(-1, 64)[:1000], torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    fn = torch.compile(M4.loss_terms, dynamic=False) if compiled else M4.loss_terms
    losses = []; t0 = None; real = 0
    for i, b in enumerate(bs):
        if i == 3:
            t0 = time.time()
        loss, terms, _ = fn(m, *b, count_hits=False)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if i >= 3:
            real += int((b[0] != 0).sum())
    return losses, real / (time.time() - t0)
t = time.time(); le, se = run(False); print(f"eager {se:.0f} real tok/s", flush=True)
t = time.time(); lc, sc = run(True); print(f"compiled {sc:.0f} real tok/s (incl. nothing of compile; compile+3 warm steps took {time.time()-t:.0f}s)")
print("loss eager   ", [round(x, 5) for x in le[:4]]); print("loss compiled", [round(x, 5) for x in lc[:4]])
print("speedup", round(sc / se, 2), "max |dloss| first 3 steps", max(abs(a - b) for a, b in zip(le[:3], lc[:3])))
