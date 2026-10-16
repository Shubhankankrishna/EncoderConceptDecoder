import time, math, torch, torch.nn.functional as F
from concept_core_d6 import ExpertPool
dev = torch.device("mps"); torch.manual_seed(0)
N, d, E, k = 512, 256, 16, 2
x = torch.randn(N, d, device=dev); logits = torch.randn(N, E, device=dev); pool = ExpertPool(d, 512, E).to(dev)
T = {}
def s(name, t0):
    torch.mps.synchronize(); T[name] = T.get(name, 0) + time.time() - t0; return time.time()
for it in range(23):
    if it == 3: T.clear()
    torch.mps.synchronize(); t = time.time()
    probs = logits.float().softmax(-1)
    v1, i1 = probs.max(-1); v2, i2 = probs.scatter(-1, i1.unsqueeze(-1), -1.0).max(-1)
    gate_v, idx = torch.stack([v1, v2], -1), torch.stack([i1, i2], -1)
    gate_v = gate_v / gate_v.sum(-1, keepdim=True); t = s("route", t)
    counts = F.one_hot(idx, E).sum(1).float(); f = counts.mean(0) / k; P = probs.mean(0)
    lb = E * (f * P).sum(); z = torch.logsumexp(logits.float(), -1).pow(2).mean(); t = s("aux", t)
    C = int(math.ceil(1.25 * N * k / E))
    flat_e = idx.reshape(-1); flat_n = torch.arange(N, device=dev).repeat_interleave(k); flat_g = gate_v.reshape(-1)
    oh = F.one_hot(flat_e, E); pos = (oh.cumsum(0) * oh).sum(-1) - 1; keep = pos < C
    slot = torch.where(keep, flat_e * C + pos, torch.full_like(pos, E * C)); t = s("positions", t)
    src = torch.full((E * C + 1,), N, dtype=torch.long, device=dev).scatter(0, slot, flat_n)
    x_pad = torch.cat([x, x.new_zeros(1, d)], 0); xe = x_pad[src[:E * C]].view(E, C, d); t = s("gather", t)
    ye = pool.expert_fn(xe).reshape(E * C, d); t = s("experts", t)
    ye = torch.cat([ye, ye.new_zeros(1, d)], 0)
    contrib = ye[slot] * (flat_g * keep.float()).unsqueeze(-1)
    y = x.new_zeros(N, d).index_add(0, flat_n, contrib); t = s("combine", t)
    dr = 1 - keep.float().mean(); t = s("drop", t)
for k_, v in T.items(): print(f"{k_:10s} {1000*v/20:6.3f} ms")
