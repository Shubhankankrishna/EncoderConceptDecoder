"""Build tokenizer, packed train/val/test rows and evaluation sets (deterministic).

python dataset/generators/build.py --stage all
Outputs under dataset/: tokenizer.json, train/rows_{tok,doc}.u16/u8, validation/, test/, eval/*.pt,
DATASET_MANIFEST.json. Everything is seeded; see mixture.json for proportions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import tasks as T  # noqa: E402

SEQ = 512
K = 4
PAD, EOS = 0, 1
SEEDS = {"train": 1000, "val": 2000, "test": 3000, "ood": 4000, "tokenizer": 5000, "pack": 6000,
         "maintenance": 7000}


def load_mixture():
    return json.load(open(os.path.join(DS, "mixture.json")))


# ---------------------------------------------------------------- language sources
def wiki_detok(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([.,;:!?)'])", r"\1", s)
    s = re.sub(r"([(]) ", r"\1", s)
    s = s.replace(" n't", "n't").replace(" 's", "'s")
    return s


def wiki_articles(path):
    lines = pd.read_parquet(path).text.tolist()
    arts, cur = [], []
    for ln in lines:
        if re.match(r"^ = [^=].* = \n$", ln):
            if cur:
                arts.append("".join(cur).strip())
            cur = [ln.strip(" =\n") + "\n"]
        elif ln.strip():
            if re.match(r"^ = = .* = = \n$", ln):
                cur.append("\n" + ln.strip(" =\n") + "\n")
            else:
                cur.append(wiki_detok(ln.strip()) + "\n")
    if cur:
        arts.append("".join(cur).strip())
    return [a for a in arts if len(a) > 200]


def tinystories(path):
    return [t.strip() for t in pd.read_parquet(path).text.tolist() if len(t.strip()) > 50]


def sentences(stories, n, rng):
    out = []
    for s in rng.sample(stories, min(len(stories), n)):
        for sent in re.split(r"(?<=[.!?])\s+", s):
            if 20 < len(sent) < 120:
                out.append(sent)
    return out


# ---------------------------------------------------------------- tokenizer
def train_tokenizer(texts, vocab_size, out):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, tr)
    tok.save(out)
    return tok


def encode_example(tok, ex):
    ids, spans = [], []
    for text, scored, label in ex.segments:
        e = tok.encode(text).ids
        if scored:
            spans.append((len(ids), len(ids) + len(e), label))
        ids.extend(e)
    ids.append(EOS)
    return ids, spans


# ---------------------------------------------------------------- generation
def make_generator(family):
    for table in (T.SYMBOLIC, T.RETRIEVAL, T.CODE, T.TOOL):
        if family in table:
            return table[family]
    if family == "variable_binding_newnames":
        return T.gen_variable_binding_newnames
    raise KeyError(family)


def gen_examples(family, split, n, seed, filler):
    rng = random.Random(f"{family}|{split}|{seed}")
    g = make_generator(family)
    out = []
    for _ in range(n):
        out.append(g(rng, split, filler) if family in T.NEEDS_FILLER else g(rng, split))
    return out


# ---------------------------------------------------------------- packing
def pack(docs, rng, splittable_flags):
    """docs: list of np.uint16 arrays (EOS included). Greedy chunk-aligned packing into SEQ rows.
    Splittable (language) docs may be cut into k-aligned fragments (>=64 tokens); others never cross a row."""
    order = list(range(len(docs)))
    rng.shuffle(order)
    total = sum(len(d) + (-len(d)) % K for d in docs)
    cap = total // SEQ + len(docs) // 4 + 16
    toks = np.full((cap, SEQ), PAD, dtype=np.uint16)
    dids = np.zeros((cap, SEQ), dtype=np.uint8)
    r, pos, did = 0, 0, 0

    def flush():
        nonlocal r, pos, did
        if pos > 0:
            dids[r, pos:] = did  # trailing filler: its own document id, all PAD, no targets
            r += 1
        pos, did = 0, 0

    for i in order:
        ids = docs[i]
        pad = (-len(ids)) % K
        if pad:
            ids = np.concatenate([ids, np.full(pad, PAD, dtype=np.uint16)])
        while len(ids):
            rem = SEQ - pos
            if len(ids) <= rem:
                toks[r, pos:pos + len(ids)] = ids
                dids[r, pos:pos + len(ids)] = did
                pos += len(ids)
                did += 1
                ids = ids[:0]
            elif splittable_flags[i] and rem >= 64:
                toks[r, pos:] = ids[:rem]
                dids[r, pos:] = did
                did += 1
                pos = SEQ
                ids = ids[rem:]
            elif len(ids) > SEQ and not splittable_flags[i]:
                ids = ids[:0]
            else:
                flush()
                continue
            if pos >= SEQ:
                flush()
    flush()
    return toks[:r], dids[:r]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tokens", type=float, default=90e6)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n_eval", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time()
    mix = load_mixture()
    raw = os.path.join(DS, "raw")
    rng = random.Random(SEEDS["tokenizer"])
    print("loading language", flush=True)
    ts_train = tinystories(os.path.join(raw, "tinystories_train0.parquet"))
    ts_train, ts_train_maint = ts_train[:-5000], ts_train[-5000:]
    ts_valid = tinystories(os.path.join(raw, "tinystories_valid.parquet"))
    ts_val, ts_test = ts_valid[: len(ts_valid) // 2], ts_valid[len(ts_valid) // 2:]
    wk_train = wiki_articles(os.path.join(raw, "wikitext103_train0.parquet"))
    wk_train, wk_train_maint = wk_train[:-300], wk_train[-300:]
    wk_val = wiki_articles(os.path.join(raw, "wikitext103_valid.parquet"))
    wk_test = wiki_articles(os.path.join(raw, "wikitext103_test.parquet"))
    print(f"tinystories {len(ts_train)}/{len(ts_val)}/{len(ts_test)} wiki {len(wk_train)}/{len(wk_val)}/{len(wk_test)}",
          flush=True)
    filler = {"train": sentences(ts_train, 20000, random.Random(1)),
              "val": sentences(ts_val, 4000, random.Random(2)),
              "test": sentences(ts_test, 4000, random.Random(3))}
    filler["ood"] = filler["test"]
    filler["maintenance"] = filler["train"]

    # ---- tokenizer: trained only on training data (language + generated train examples)
    tok_path = os.path.join(DS, "tokenizer.json")
    fams = [f for grp in mix["families"].values() for f in grp.get("families", [])]
    tok_texts = rng.sample(ts_train, 40000) + rng.sample(wk_train, 3000)
    for f in fams:
        tok_texts += [e.text for e in gen_examples(f, "train", 1500, SEEDS["tokenizer"], filler["train"])]
    tok = train_tokenizer(tok_texts, args.vocab, tok_path)
    assert tok.token_to_id("<pad>") == PAD and tok.token_to_id("<eos>") == EOS
    print(f"tokenizer trained ({time.time() - t0:.0f}s)", flush=True)

    manifest = {"generator_version": T.GENERATOR_VERSION, "seq_len": SEQ, "k": K, "seeds": SEEDS,
                "tokenizer_sha256": hashlib.sha256(open(tok_path, "rb").read()).hexdigest(),
                "mixture": mix, "splits": {}}

    def lang_docs(texts, max_tokens, rng_):
        out, n = [], 0
        idx = list(range(len(texts)))
        rng_.shuffle(idx)
        for batch_start in range(0, len(idx), 2000):
            b = [texts[i] for i in idx[batch_start:batch_start + 2000]]
            for e in tok.encode_batch(b):
                ids = np.array(e.ids + [EOS], dtype=np.uint16)
                out.append(ids)
                n += len(ids)
                if n >= max_tokens:
                    return out, n
        return out, n

    dropped = {}

    def synth_docs(family, split, target_tokens, seed):
        out, n, rounds = [], 0, 0
        while n < target_tokens:
            exs = gen_examples(family, split, 5000, seed + rounds, filler[split])
            enc = tok.encode_batch([s[0] for ex in exs for s in ex.segments])
            j = 0
            for ex in exs:
                ids = []
                for _ in ex.segments:
                    ids.extend(enc[j].ids)
                    j += 1
                ids.append(EOS)
                if len(ids) + (-len(ids)) % K <= SEQ:
                    out.append(np.array(ids, dtype=np.uint16))
                    n += len(ids)
                else:
                    dropped[(family, split)] = dropped.get((family, split), 0) + 1
            rounds += 1
        return out, n

    # ---- packed LM rows for train / val / test
    for split, budget in [("train", args.train_tokens), ("val", 2.0e6), ("test", 2.0e6), ("maintenance", 0.3e6)]:
        docs, flags, stats = [], [], {}
        seed = SEEDS[split]
        gsplit = "train" if split == "maintenance" else split  # maintenance: training distribution, own seeds
        for grp_name, grp in mix["families"].items():
            share = grp["share"] * budget
            if grp_name == "language":
                srcs = {"tinystories": {"train": ts_train, "val": ts_val, "test": ts_test,
                                        "maintenance": ts_train_maint}[split],
                        "wikitext": {"train": wk_train, "val": wk_val, "test": wk_test, "maintenance": wk_train_maint}[split]}
                for name, w in grp["sources"].items():
                    d, n = lang_docs(srcs[name], share * w, random.Random(f"{name}|{split}|{seed}"))
                    docs += d
                    flags += [True] * len(d)
                    stats[name] = {"docs": len(d), "tokens": n}
            else:
                per = share / len(grp["families"])
                for f in grp["families"]:
                    d, n = synth_docs(f, gsplit, per, seed)
                    docs += d
                    flags += [False] * len(d)
                    stats[f] = {"docs": len(d), "tokens": n}
            print(f"{split}: {grp_name} done ({time.time() - t0:.0f}s)", flush=True)
        # duplicate / leakage bookkeeping: sequence hashes of generated docs
        hashes = [hashlib.sha1(d.tobytes()).hexdigest() for d, fl in zip(docs, flags) if not fl]
        toks, dids = pack(docs, random.Random(SEEDS["pack"] + seed), flags)
        outdir = os.path.join(DS, {"train": "train", "val": "validation", "test": "test", "maintenance": "maintenance"}[split])
        os.makedirs(outdir, exist_ok=True)
        toks.tofile(os.path.join(outdir, "rows_tok.u16"))
        dids.tofile(os.path.join(outdir, "rows_doc.u8"))
        np.save(os.path.join(outdir, "synthetic_doc_hashes.npy"), np.array(hashes))
        nonpad = int((toks != PAD).sum())
        if split != "train":  # leakage: generated documents shared with the training split
            tr_h = set(np.load(os.path.join(DS, "train", "synthetic_doc_hashes.npy")).tolist())
            manifest.setdefault("leakage", {})[split] = {"synthetic_docs": len(hashes),
                                                         "identical_to_train": sum(h in tr_h for h in hashes)}
        manifest["splits"][split] = {"rows": int(toks.shape[0]), "nonpad_tokens": nonpad,
                                     "pad_fraction": 1 - nonpad / toks.size, "families": stats,
                                     "rows_sha256": hashlib.sha256(toks.tobytes()).hexdigest(),
                                     "n_synthetic_docs": len(hashes), "n_unique_synthetic": len(set(hashes))}
        print(f"{split}: {toks.shape[0]} rows, nonpad {nonpad:,} ({time.time() - t0:.0f}s)", flush=True)

    # ---- evaluation sets (one example per row, teacher-forced scoring of answer spans)
    train_hashes = set(np.load(os.path.join(DS, "train", "synthetic_doc_hashes.npy")).tolist())
    evald = os.path.join(DS, "eval")
    os.makedirs(evald, exist_ok=True)
    eval_manifest = {}
    for split in ["val", "test", "ood"]:
        items = []
        fam_list = [f for f in fams] + (["variable_binding_newnames"] if split == "ood" else [])
        for f in fam_list:
            g_split = "val" if f == "variable_binding_newnames" and split == "ood" else split
            exs = gen_examples(f, "ood" if split == "ood" and f != "variable_binding_newnames" else g_split,
                               args.n_eval, SEEDS[split] + 17, filler[split])
            n_leak = 0
            for ex in exs:
                ids, spans = encode_example(tok, ex)
                if hashlib.sha1(np.array(ids, dtype=np.uint16).tobytes()).hexdigest() in train_hashes:
                    n_leak += 1
                    continue
                # source distance for retrieval families: last occurrence of the answer ids in the prompt
                dist = None
                if spans and f in ("kv_retrieval", "copy", "needle", "json_lookup", "yaml_lookup"):
                    a0, a1, _ = spans[0]
                    ans = ids[a0:a1]
                    for s in range(a0 - len(ans), -1, -1):
                        if ids[s:s + len(ans)] == ans:
                            dist = a0 - s
                            break
                items.append({"ids": ids, "spans": spans, "family": f, "meta": ex.meta, "src_distance": dist,
                              "length": len(ids)})
            eval_manifest.setdefault(split, {})[f] = {"n": sum(1 for it in items if it["family"] == f),
                                                      "dropped_train_duplicates": n_leak}
        torch.save(items, os.path.join(evald, f"{split}.pt"))
        print(f"eval {split}: {len(items)} items ({time.time() - t0:.0f}s)", flush=True)
    manifest["eval"] = eval_manifest
    manifest["dropped_over_seq_len"] = {f"{f}|{s}": n for (f, s), n in dropped.items()}
    manifest["build_seconds"] = time.time() - t0
    json.dump(manifest, open(os.path.join(DS, "DATASET_MANIFEST.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
