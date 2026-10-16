import time, torch, sys
from model_v5 import *
def bench(dev, dtype, kind, B=64, T=192, steps=15, threads=8):
    torch.set_num_threads(threads)
    m = (NCPModel(2048) if kind == "ncp" else TransformerLM(2048)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 2048, (B, T), device=dev)
    def step():
        with torch.autocast(device_type=dev, dtype=dtype, enabled=dtype != torch.float32):
            out = m(x)
            lg = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(lg[:, :-1].float().reshape(-1, 2048), x[:, 1:].reshape(-1))
            if isinstance(out, dict):
                loss = loss + ncp_mse(out["chat"], out["C"], torch.ones(B, T // 4 - 1, device=dev))[0] + m.vq(out["C"])[1]
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(3): step()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.time()
    for _ in range(steps): step()
    if dev == "mps": torch.mps.synchronize()
    dt = (time.time() - t0) / steps
    print(f"{kind:4s} {dev} {str(dtype):15s} B={B} params={count_params(m):,} {dt*1000:.0f} ms/step {B*T/dt:,.0f} tok/s", flush=True)
for kind in ("ncp", "v0"):
    bench("mps", torch.float32, kind)
    bench("mps", torch.bfloat16, kind)
    bench("mps", torch.float32, kind, B=128)
    bench("cpu", torch.float32, kind, steps=5)
