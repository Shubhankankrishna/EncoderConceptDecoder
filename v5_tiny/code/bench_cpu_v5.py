import time, torch, sys
from model_v5 import *
torch.set_num_threads(int(sys.argv[1]))
m = TransformerLM(2048); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 2048, (64, 192))
f = torch.compile(lambda x: token_nll(m(x), x, torch.ones_like(x, dtype=torch.float)))
def step():
    l = f(x).sum() / x.numel(); opt.zero_grad(); l.backward(); opt.step()
for _ in range(3): step()
t0 = time.time()
for _ in range(5): step()
dt = (time.time() - t0) / 5
print(f"cpu threads={sys.argv[1]} {64*192/dt:,.0f} tok/s")
