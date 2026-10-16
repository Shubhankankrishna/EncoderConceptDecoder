"""Teacher-calibrated swap reference (pre-registered 15:00): NCP teacher at T=1, two seeds, 512 standard instances."""
import json, os, numpy as np, torch
import data_v5 as dv
from common_v5 import RUNS, eval_sets, generate_walks, save_json, maybe_compile, DEV
from eval_a import load
from eval_b import hop_colours

model, _ = load("A_ncp_s0")
f = maybe_compile(lambda x: model(x)["logits"])
xg = eval_sets("val")[1][:512]


@torch.no_grad()
def gen(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(xg.copy()); x[:, dv.WALK_START:] = dv.PAD; x = x.to(DEV)
    outs = []
    for i in range(0, len(x), 128):
        xb = x[i:i + 128].clone()
        for pos in range(dv.WALK_START - 1, dv.T - 1):
            lg = f(xb)[:, pos].float()
            xb[:, pos + 1] = torch.multinomial(lg.softmax(-1).cpu(), 1, generator=g).squeeze(1).to(DEV)
        outs.append(xb.cpu().numpy())
    return np.concatenate(outs)


A, B = gen(1), gen(2)
fa = np.array([a[dv.WALK_START] for a in A]); fb = np.array([b[dv.WALK_START] for b in B])
d1 = (fa != fb).astype(float)
dseq = np.array([float(hop_colours(a) != hop_colours(b)) for a, b in zip(A, B)])
res = dict(first_colour_differs=float(d1.mean()), se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
           colour_sequence_differs=float(dseq.mean()))
print(res)
save_json(os.path.join(RUNS, "A_ncp_s0", "teacher_swapref.json"), res)
