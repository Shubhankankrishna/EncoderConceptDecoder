"""Decoder Engram reads off: paired NTP delta on probe rows built from the TRAINING filler pool vs the validation pool."""
import os, sys, json, math, torch
import data_v4 as D, decomp_v4tiny as DC
from tokenizers import Tokenizer
torch.set_num_threads(1)
tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json")); canon = D.Canon(); tr = D.TextStore("train")
pools = dict(train_fillers=D.TinyProbes(tok, D.filler_sentences(tr, tok), "train"),
             val_fillers=D.TinyProbes(tok, D.filler_sentences(tr, tok, seed=99), "val"))
for arm in sys.argv[1:]:
    m, a = DC.load(arm); k = a["k"]; r = {}
    for nm, pf in pools.items():
        t, d, _ = D.item_rows(D.probe_items(pf, 64, seed=5151), 480, k)
        b0, b1 = DC.per_row_ntp(m, t, d, canon), DC.per_row_ntp(m, t, d, canon, dec_reads_off=(0, 1))
        dd = b1[:, 0] / b1[:, 1] - b0[:, 0] / b0[:, 1]
        r[nm] = dict(intact=float(b0[:, 0].sum() / b0[:, 1].sum()), delta=float(b1[:, 0].sum() / b1[:, 1].sum() - b0[:, 0].sum() / b0[:, 1].sum()),
                     paired_se=float(dd.std(ddof=1) / math.sqrt(len(dd))))
    json.dump(r, open(os.path.join(DC.RUNS, arm, "reads_check.json"), "w"), indent=1)
    print(arm, json.dumps(r), flush=True)
