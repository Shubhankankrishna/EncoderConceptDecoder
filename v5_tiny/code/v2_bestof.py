"""Equal-budget token-search baseline (pre-registered 14:50): the no-Z V2 core with T=1 token sampling,
best-of-(U+1)=11 by the header-only verifier, on the hard-goal and standard sets."""
import json, os, numpy as np, torch
import data_v5 as dv
from common_v5 import RUNS, eval_sets, save_json
from eval_b import load_system, make_logits_fn, generate, starts_every

sysm, _ = load_system("B_v2_s0")
f = make_logits_fn(sysm)
res = {}
for name, xs in (("hardgoal", dv.hardgoal_set(512, seed=31)[0]), ("standard", eval_sets("val")[1][:512])):
    acc = np.zeros(len(xs), bool); first = None
    for k in range(11):
        X = generate(f, xs, starts_every, seed=5000 + k, z_mode="mean", eps_mode="off", temperature=1.0)
        ok = np.array([dv.verify(r[:dv.WALK_START], r[dv.WALK_START:])[0] for r in X])
        first = ok if first is None else first
        acc |= ok
    res[name] = dict(best_of_11=float(acc.mean()), se=float(acc.std(ddof=1) / np.sqrt(len(acc))),
                     single_T1=float(first.mean()), flags=acc.astype(float).tolist())
    print(name, res[name]["best_of_11"], res[name]["single_T1"], flush=True)
save_json(os.path.join(RUNS, "B_v2_s0", "bestof11_T1.json"), res)
