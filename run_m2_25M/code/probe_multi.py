"""Where does the multi-token value get lost? Linear probes on a trained checkpoint."""
import sys, torch, torch.nn.functional as F, harness
from e2e import HRMThinker, configs, make_batch_multi, MarkovFiller
from hrm_encoder import engram_indices
torch.set_num_threads(2); torch.manual_seed(0)
path, T_ = sys.argv[1], int(sys.argv[2]); harness.T = T_
layout = "stack" if "stack" in path else "confined"
ecfg, dcfg = configs(layout=layout); m = HRMThinker(ecfg, dcfg); m.load_state_dict(torch.load(path), strict=False); m.eval()
g = torch.Generator().manual_seed(5); filler = MarkovFiller(seed=123)
feats = {k: [] for k in ("fact_c", "query_c", "query_u")}; labs = {k: [] for k in ("fact_key", "fact_v1", "q_key", "q_v1", "q_v2", "q_v3")}
with torch.no_grad():
    for _ in range(40):
        ids, vpos = make_batch_multi(32, filler, g, T_); doc = torch.zeros_like(ids)
        tok, c_hat, _ = m.enc(ids, doc, engram_indices(ids, doc, ecfg)); u, _, _ = m.core(c_hat, doc[:, ::4], 4)
        for b in range(32):
            for f in range(4):                                   # fact chunks 1..4
                feats["fact_c"].append(c_hat[b, 1 + f]); labs["fact_key"].append(ids[b, 4 * (1 + f)]); labs["fact_v1"].append(ids[b, 4 * (1 + f) + 1])
            for q in range(4):
                p0 = vpos[b, 3 * q].item(); qc = (p0 - 5) // 4        # query chunk index
                feats["query_c"].append(c_hat[b, qc]); feats["query_u"].append(u[b, qc])
                labs["q_key"].append(ids[b, 4 * qc + 1])
                for i, k in enumerate(("q_v1", "q_v2", "q_v3")): labs[k].append(ids[b, p0 + i])
def probe(X, y, epochs=300):
    X = torch.stack(X); y = torch.stack(y); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n = int(0.8 * len(X)); cls = y.unique(); ymap = {c.item(): i for i, c in enumerate(cls)}; yy = torch.tensor([ymap[v.item()] for v in y])
    W = torch.zeros(X.shape[1], len(cls), requires_grad=True); b = torch.zeros(len(cls), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(epochs):
        loss = F.cross_entropy(X[:n] @ W + b, yy[:n]) + 1e-3 * W.pow(2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return ((X[n:] @ W + b).argmax(-1) == yy[n:]).float().mean().item()
print(path)
print(f"fact concept -> key {probe(feats['fact_c'], labs['fact_key']):.3f} | -> v1 {probe(feats['fact_c'], labs['fact_v1']):.3f}")
print(f"query concept -> queried key {probe(feats['query_c'], labs['q_key']):.3f}")
print(f"core output u at query chunk -> v1 {probe(feats['query_u'], labs['q_v1']):.3f}, v2 {probe(feats['query_u'], labs['q_v2']):.3f}, v3 {probe(feats['query_u'], labs['q_v3']):.3f}  (chance 0.031)")
