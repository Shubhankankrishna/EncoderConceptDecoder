"""Node-lookup diagnostic (advisor consult 8): graph-only 4-layer TransformerLM; node NLL every 250 steps."""
import os, sys, json, time, numpy as np, torch
import data_v5 as dv
from model_v5 import TransformerLM, token_nll
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
torch.manual_seed(0)
out = sys.argv[1]
m = TransformerLM(dv.VOCAB, n_layers=4)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
rng = np.random.RandomState(0)
xv, mv, _ = dv.fixed_graph_set(256, seed=21)
xv, mv = torch.from_numpy(xv), torch.from_numpy(mv)
f = open(out, "w")
t0 = time.time()
for step in range(2001):
    lr = 2e-3 * min(1.0, (step + 1) / 100)
    for g in opt.param_groups: g["lr"] = lr
    if step % 250 == 0:
        with torch.no_grad():
            nl = token_nll(m(xv), xv, mv); tgt = xv[:, 1:]
            isn = ((tgt >= dv.NODE0) & (tgt < dv.NODE0 + dv.N_NODE_TOK)).float() * mv[:, :-1]
            isc = ((tgt >= dv.COL0) & (tgt < dv.COL0 + dv.N_COL)).float() * mv[:, :-1]
            r = dict(step=step, fmt=dv.HOP_FMT, node=float((nl * isn).sum() / isn.sum()), colour=float((nl * isc).sum() / isc.sum()),
                     graph=float(nl.sum() / mv[:, :-1].sum()), wall=time.time() - t0)
            # per chunk offset of the predicted token (offset = (target index) % 4), walk region only
            off = (torch.arange(1, xv.shape[1]) % 4).expand_as(tgt)
            for o in range(4):
                w = (off == o).float() * mv[:, :-1]
                r[f"off{o}"] = float((nl * w).sum() / w.sum())
        f.write(json.dumps(r) + "\n"); f.flush(); print(r, flush=True)
    x, mm, _ = dv.graph_batch(rng, 32)
    x, mm = torch.from_numpy(x), torch.from_numpy(mm)
    loss = token_nll(m(x), x, mm).sum() / mm[:, :-1].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
