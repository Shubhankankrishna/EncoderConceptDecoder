"""Data checks before training (PROMPT 4.6): every document (and padding run) spans a multiple of 4 positions and starts
at a multiple of 4; every probe's evidence-to-answer distance is > 90 tokens (>= the 128 floor); 3 decoded samples per
family (test ID and OOD) written to runs/data_samples.md."""
import json, os, random, sys
import numpy as np
import torch
from tokenizers import Tokenizer

D = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs")
tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
res = {}
for split in ("train", "val", "test", "maintenance"):
    doc = np.load(os.path.join(D, split, "doc.npy"), mmap_mode="r")
    tk = np.load(os.path.join(D, split, "tok.npy"), mmap_mode="r")
    bad_len = bad_start = n_docs = 0
    for s in range(0, len(doc), 8192):
        d = np.asarray(doc[s:s + 8192])
        change = np.diff(d, axis=1) != 0
        for r in range(len(d)):
            b = np.concatenate([[0], np.nonzero(change[r])[0] + 1, [d.shape[1]]])
            ln = np.diff(b)
            bad_len += int((ln % 4 != 0).sum()); bad_start += int((b[:-1] % 4 != 0).sum()); n_docs += len(ln)
    res[split] = dict(rows=int(len(doc)), doc_spans=n_docs, spans_not_multiple_of_4=bad_len, starts_not_multiple_of_4=bad_start,
                      max_token_id=int(np.asarray(tk[:2000]).max()))
    assert bad_len == 0 and bad_start == 0, (split, bad_len, bad_start)
    print(split, res[split], flush=True)

items = {s: torch.load(os.path.join(D, "eval", f"{s}.pt"), weights_only=False) for s in ("test", "ood")}
for s, its in items.items():
    for it in its:
        if it["family"].startswith("lr_"):
            assert it["dist"] > 90 and it["dist"] >= 128, (s, it["family"], it["dist"])
            # the evidence must lie beyond the token path's reach at every scored position
            for (a, e, _) in it["spans"]:
                assert a - 1 - 70 > a - it["dist"], "evidence within reach"
res["probe_distance_assert"] = "PASS: every probe item (test, ood) has evidence >= 128 tokens before its first answer token"
print(res["probe_distance_assert"])

rng = random.Random(0)
lines = ["# Data samples (3 random items per family, decoded; answer spans in [[...]])", ""]
for s, its in items.items():
    fams = sorted({x["family"] for x in its})
    for f in fams:
        pool = [x for x in its if x["family"] == f]
        lines.append(f"## {s} / {f}")
        for x in rng.sample(pool, 3):
            ids = x["ids"]
            parts, prev = [], 0
            for a, e, _ in sorted(x["spans"]):
                parts.append(tok.decode(ids[prev:a])); parts.append("[[" + tok.decode(ids[a:e]) + "]]"); prev = e
            parts.append(tok.decode(ids[prev:]))
            txt = "".join(parts).replace("\n", "⏎")
            if len(txt) > 1400:
                txt = txt[:500] + " … " + txt[-800:]
            meta = {k: v for k, v in x["meta"].items() if k != "dists"}
            lines.append(f"- ({len(ids)} tokens, dist {x.get('dist')}, {meta}) `{txt}`")
        lines.append("")
# three packed training rows
tr = np.load(os.path.join(D, "train", "tok.npy"), mmap_mode="r")
lines.append("## three packed training rows (first 600 characters each; <eos> marks document ends)")
for r in rng.sample(range(len(tr)), 3):
    lines.append(f"- row {r}: `" + tok.decode(list(np.asarray(tr[r])), skip_special_tokens=False).replace("<pad>", "").replace("\n", "⏎")[:600] + "`")
open(os.path.join(OUT, "data_samples.md"), "w").write("\n".join(lines))
json.dump(res, open(os.path.join(OUT, "data_checks.json"), "w"), indent=1)
print("wrote runs/data_samples.md, runs/data_checks.json")
