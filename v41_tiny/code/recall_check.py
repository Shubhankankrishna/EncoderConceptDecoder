"""Recall with items built from the TRAINING filler pool (fresh rng seed, new names/values) vs the validation pool."""
import os, sys, json, torch
import data_v4 as D, decomp_v4tiny as DC, run_v4tiny as RV
from tokenizers import Tokenizer
torch.set_num_threads(1)
tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json")); canon = D.Canon(); tr = D.TextStore("train")
ptr = D.TinyProbes(tok, D.filler_sentences(tr, tok), "train")
pva = D.TinyProbes(tok, D.filler_sentences(tr, tok, seed=99), "val")
out = {}
for arm in sys.argv[1:]:
    m, a = DC.load(arm); k = a["k"]
    r = {}
    for nm, pf in (("train_fillers", ptr), ("val_fillers", pva)):
        its = D.probe_items(pf, 192, seed=4242)
        r[nm] = dict(intact=RV.item_acc(m, its, canon, 480, k), plan_swap=RV.item_acc(m, its, canon, 480, k, plan_swap=True),
                     core_off=RV.item_acc(m, its, canon, 480, k, core_off=True))
    out[arm] = r
    print(arm, json.dumps(r), flush=True)
    json.dump(r, open(os.path.join(DC.RUNS, arm, "recall_check.json"), "w"), indent=1)
