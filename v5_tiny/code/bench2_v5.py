import time, torch
from model_v5 import *
def run(kind, compile_, B=64, T=192, steps=15):
    dev="mps"
    m = (NCPModel(2048) if kind == "ncp" else TransformerLM(2048)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=False)
    x = torch.randint(0, 2048, (B, T), device=dev)
    cm = torch.ones(B, T // 4 - 1, device=dev)
    def lossf(x):
        out = m(x)
        lg = out["logits"] if isinstance(out, dict) else out
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, 2048), x[:, 1:].reshape(-1))
        if isinstance(out, dict):
            loss = loss + ncp_mse(out["chat"], out["C"], cm)[0] + m.vq(out["C"])[1]
        return loss
    f = torch.compile(lossf) if compile_ else lossf
    def step():
        loss = f(x); opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(4): step()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(steps): step()
    torch.mps.synchronize(); dt = (time.time() - t0) / steps
    print(f"{kind} compile={compile_} {dt*1000:.0f} ms/step {B*T/dt:,.0f} tok/s", flush=True)
for k in ("ncp","v0"):
    run(k, False)
    try: run(k, True)
    except Exception as e: print("compile failed:", type(e).__name__, str(e)[:300])
