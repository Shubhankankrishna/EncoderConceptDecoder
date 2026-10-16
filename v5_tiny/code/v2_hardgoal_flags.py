"""Per-instance hard-goal flags for the no-Z V2 core (for paired comparisons; pre-registered 15:10)."""
import os, json, numpy as np
import data_v5 as dv
from common_v5 import RUNS, save_json
from eval_b import load_system, make_logits_fn, hardgoal_success
sysm, _ = load_system("B_v2_s0")
r = hardgoal_success(sysm, make_logits_fn(sysm))
save_json(os.path.join(RUNS, "B_v2_s0", "hardgoal_flags.json"), r)
print({k: v["success"] for k, v in r.items() if isinstance(v, dict)})
