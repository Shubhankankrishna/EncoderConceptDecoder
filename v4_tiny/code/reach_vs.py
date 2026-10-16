"""Reach diagnostic (VS_LLM_REPORT §6, pre-specified 00:55): per-bucket action readings for a v4 / LLM pair.

python reach_vs.py vs_pin_v4_s0_6M vs_pin_llm_s0_6M   -> runs/<v4 name>/reach.json and a printed table
"""
import json
import os
import sys
import numpy as np
import torch

import analyze_vs as A
import run_vs as R


def main(v4n, llmn):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import speed_v3
    speed_v3.enable_block_attention(60)
    sets = R.pin_eval_sets()
    U = {}
    for n in (v4n, llmn):
        m, _ = A.load(n)
        U[n] = {s: A.pin_units(m, sets[s]) for s in ("heldin", "heldout", "long")}
    out, lines = {}, ["| set | bucket | actions | reading | v4 | LLM | v4 − LLM (paired SE) |", "|---|---|---|---|---|---|---|"]
    for s in ("heldin", "heldout", "long"):
        for z in ("near", "partial", "far"):
            nact = int(np.nansum(U[v4n][s][f"n_{z}"]))
            for r in ("act_opt", "act_opt_b2", "ntp_action"):
                k = f"{r}_{z}"
                a, b = U[v4n][s][k], U[llmn][s][k]
                d, se, npair = A.paired(a, b)
                out[f"{s}/{k}"] = dict(v4=float(np.nanmean(a)), llm=float(np.nanmean(b)), diff=d, se=se, n_episodes=npair,
                                        n_actions=nact)
                lines.append(f"| {s} | {z} | {nact} | {r} | {np.nanmean(a):.3f} | {np.nanmean(b):.3f} | {d:+.3f} ({se:.3f}) |")
    json.dump(out, open(os.path.join(A.RUNS, v4n, "reach.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
