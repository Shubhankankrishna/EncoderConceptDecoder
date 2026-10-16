"""Build tokenizer, canonical-id table, packed T = 1024 shards, evaluation sets and MANIFEST.json (PROMPT section 4-5).

    python build_data.py --out ../data [--train_tokens 2.6e8]

Reuses the D-MIX design and generators (tasks.py, copied unchanged from the earlier programme) and the D-MIX raw
parquet files (the same HF datasets the PROMPT names: roneneldan/TinyStories, Salesforce/wikitext 103-raw-v1).
Mixture (by training tokens): language 45%, symbolic 22%, code 10%, retrieval 13%, long-range probes 10%.
Deterministic: every generator draws from random.Random(f"{family}|{split}|{seed}").
"""
import argparse, hashlib, json, os, random, re, sys, time, unicodedata
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import tasks as TK                                                    # noqa: E402  (D-MIX generators, unchanged)
import probes as PR                                                   # noqa: E402

RAW = os.path.expanduser("~/Desktop/HRM THINKER/experiments/HRM_CC_DUALIRC_vs_NANBEIGE42_M2_10_20M_20260923/dataset/raw")
SEQ, K, PAD, EOS = 1024, 4, 0, 1
SEEDS = {"train": 1000, "val": 2000, "test": 3000, "ood": 4000, "tokenizer": 5000, "pack": 6000, "maintenance": 7000}
MIX = {"language": 0.45, "symbolic": 0.22, "code": 0.10, "retrieval": 0.13, "longrange": 0.10}
LANG_SPLIT = {"tinystories": 0.45, "wikitext": 0.55}
GROUPS = {"symbolic": list(TK.SYMBOLIC), "code": list(TK.CODE), "retrieval": list(TK.RETRIEVAL), "longrange": PR.PROBES}


# ------------------------------------------------------------------ language (functions as in D-MIX build.py)
def wiki_detok(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([.,;:!?)'])", r"\1", s)
    s = re.sub(r"([(]) ", r"\1", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


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


def sentences(texts, n_docs, rng, lo=20, hi=150):
    out = []
    for s in rng.sample(texts, min(len(texts), n_docs)):
        for sent in re.split(r"(?<=[.!?])\s+", s.replace("\n", " ")):
            if lo < len(sent) < hi and "=" not in sent:
                out.append(sent.strip())
    return out


# ------------------------------------------------------------------ tokenizer and canonical ids
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


def build_canonical_map(tok, V):
    """Encoder spec 5.8: decode, strip one leading space marker, NFKC, lowercase; equal strings share an id.
    Special tokens are decoded with their names (<pad>, <eos>) so they keep distinct canonical ids (DECISIONS D-013)."""
    key_to_id, canon = {}, [0] * V
    for i in range(V):
        s = tok.decode([i], skip_special_tokens=False)
        s = s[1:] if s[:1] in (" ", "Ġ") else s
        s = unicodedata.normalize("NFKC", s).lower()
        canon[i] = key_to_id.setdefault(s, len(key_to_id))
    return np.array(canon, dtype=np.int64)


def token_bytes(tok, V):
    return np.array([len(tok.decode([i]).encode("utf-8")) if i > EOS else 0 for i in range(V)], dtype=np.int64)


# ------------------------------------------------------------------ packing
def pack(docs, rng, splittable):
    """Greedy k-aligned packing into SEQ rows (D-MIX algorithm). Each doc (EOS included) is padded with PAD to a
    multiple of 4; padding keeps the doc id of the document it pads. Splittable (language) docs may be cut into
    4-aligned fragments >= 64 tokens (each fragment its own doc id); generated docs never cross a row."""
    order = list(range(len(docs)))
    rng.shuffle(order)
    total = sum(len(d) + (-len(d)) % K for d in docs)
    cap = total // SEQ + len(docs) // 8 + 64
    toks = np.full((cap, SEQ), PAD, dtype=np.uint16)
    dids = np.zeros((cap, SEQ), dtype=np.int32)
    r, pos, did = 0, 0, 0
    src = []                                   # per row: list of (doc index, start in row)

    def flush():
        nonlocal r, pos, did
        if pos > 0:
            dids[r, pos:] = did                # trailing filler: its own doc id, all PAD, no targets
            r += 1
            src.append([])
        pos, did = 0, 0

    src.append([])
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
                src[r].append((i, pos))
                pos += len(ids); did += 1
                ids = ids[:0]
            elif splittable[i] and rem >= 64:
                toks[r, pos:] = ids[:rem]
                dids[r, pos:] = did
                src[r].append((i, pos))
                did += 1; pos = SEQ
                ids = ids[rem:]
            elif len(ids) > SEQ and not splittable[i]:
                ids = ids[:0]
            else:
                flush()
                continue
            if pos >= SEQ:
                flush()
    flush()
    return toks[:r], dids[:r], src[:r]


def same_doc_context(dids):
    """For each row: number of same-document tokens before position SEQ-16 (context of the scored last 16 positions)."""
    last = dids[:, SEQ - 16]
    ctx = np.zeros(len(dids), dtype=np.int64)
    for r in range(len(dids)):
        start = np.argmax(dids[r] == last[r])
        ctx[r] = SEQ - 16 - start
    return ctx


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--train_tokens", type=float, default=2.6e8)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    # ---- language sources and splits (as D-MIX)
    ts_train = tinystories(os.path.join(RAW, "tinystories_train0.parquet"))
    ts_train, ts_maint = ts_train[:-5000], ts_train[-5000:]
    ts_valid = tinystories(os.path.join(RAW, "tinystories_valid.parquet"))
    ts_val, ts_test = ts_valid[: len(ts_valid) // 2], ts_valid[len(ts_valid) // 2:]
    wk_train = wiki_articles(os.path.join(RAW, "wikitext103_train0.parquet"))
    wk_train, wk_maint = wk_train[:-300], wk_train[-300:]
    wk_val = wiki_articles(os.path.join(RAW, "wikitext103_valid.parquet"))
    wk_test = wiki_articles(os.path.join(RAW, "wikitext103_test.parquet"))
    log(f"tinystories train/maint/val/test {len(ts_train)}/{len(ts_maint)}/{len(ts_val)}/{len(ts_test)}; "
        f"wikitext {len(wk_train)}/{len(wk_maint)}/{len(wk_val)}/{len(wk_test)}")
    lang = {"train": {"tinystories": ts_train, "wikitext": wk_train},
            "maintenance": {"tinystories": ts_maint, "wikitext": wk_maint},
            "val": {"tinystories": ts_val, "wikitext": wk_val}, "test": {"tinystories": ts_test, "wikitext": wk_test}}
    # filler banks per split: natural sentences from that split's text only (train filler for train/maintenance)
    fb = {"train": sentences(ts_train, 20000, random.Random(1)) + sentences(wk_train, 2000, random.Random(11)),
          "val": sentences(ts_val, 5000, random.Random(2)) + sentences(wk_val, 60, random.Random(12)),
          "test": sentences(ts_test, 5000, random.Random(3)) + sentences(wk_test, 61, random.Random(13))}
    fb["ood"], fb["maintenance"] = fb["test"], fb["train"]
    dmix_filler = {k: [s for s in v if len(s) < 120] for k, v in fb.items()}     # D-MIX copy/needle filler
    log("filler sentences:", {k: len(v) for k, v in fb.items()})

    # ---- tokenizer (training text only)
    rng = random.Random(SEEDS["tokenizer"])
    tok_texts = rng.sample(ts_train, 40000) + rng.sample(wk_train, 3000)
    for grp in ("symbolic", "code", "retrieval"):
        for f in GROUPS[grp]:
            g = random.Random(f"{f}|train|{SEEDS['tokenizer']}")
            gen = TK.SYMBOLIC.get(f) or TK.CODE.get(f) or TK.RETRIEVAL.get(f)
            tok_texts += [(gen(g, "train", dmix_filler["train"]) if f in TK.NEEDS_FILLER else gen(g, "train")).text
                          for _ in range(1500)]
    for f in PR.PROBES:
        g = random.Random(f"{f}|train|{SEEDS['tokenizer']}")
        tok_texts += [PR.probe_text_for_tokenizer(g, f, fb["train"], PR.VALUE_CANDIDATES) for _ in range(1500)]
    tok_path = os.path.join(out, "tokenizer.json")
    if a.eval_only:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
    else:
        tok = train_tokenizer(tok_texts, a.vocab, tok_path)
    assert tok.token_to_id("<pad>") == PAD and tok.token_to_id("<eos>") == EOS
    V = tok.get_vocab_size()
    values = [w for w in PR.VALUE_CANDIDATES if len(tok.encode(" " + w).ids) == 1]
    canon = build_canonical_map(tok, V)
    np.save(os.path.join(out, "canon.npy"), canon)
    np.save(os.path.join(out, "token_bytes.npy"), token_bytes(tok, V))
    log(f"tokenizer V={V}; single-token values {len(values)}/{len(PR.VALUE_CANDIDATES)}; canonical ids {canon.max() + 1}")

    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M"), seq_len=SEQ, k=K, pad=PAD, eos=EOS, seeds=SEEDS,
                    mixture=MIX, language_sources=LANG_SPLIT, raw_dir=RAW,
                    raw_sha256=open(os.path.join(RAW, "SHA256SUMS.txt")).read().split("\n"),
                    tokenizer_sha256=hashlib.sha256(open(tok_path, "rb").read()).hexdigest(), vocab=V,
                    canonical_ids=int(canon.max() + 1), probe_floor_tokens=PR.FLOOR, probe_values=values,
                    probe_chance=dict(lr_recall1=1 / len(values), lr_carry3_token=1 / len(values),
                                      lr_multi4_token=1 / len(values)),
                    fineweb_edu_used=False, tool_traj_used=False, splits={})
    json.dump(manifest, open(os.path.join(out, "MANIFEST.partial.json"), "w"), indent=1)

    facs = {s: PR.ProbeFactory(tok, fb[s], values, s) for s in ("train", "val", "test", "ood", "maintenance")}

    def gen_dmix(f, split, n, seed):
        g = random.Random(f"{f}|{split}|{seed}")
        gen = TK.SYMBOLIC.get(f) or TK.CODE.get(f) or TK.RETRIEVAL.get(f) or (
            TK.gen_variable_binding_newnames if f == "variable_binding_newnames" else None)
        return [gen(g, split, dmix_filler[split if split in dmix_filler else "test"]) if f in TK.NEEDS_FILLER
                else gen(g, split) for _ in range(n)]

    def encode_dmix(exs):
        enc = tok.encode_batch([s[0] for ex in exs for s in ex.segments])
        j, outl = 0, []
        for ex in exs:
            ids, spans = [], []
            for text, scored, label in ex.segments:
                e = enc[j].ids; j += 1
                if scored:
                    spans.append((len(ids), len(ids) + len(e), label))
                ids.extend(e)
            ids.append(EOS)
            outl.append(dict(ids=ids, spans=spans, family=ex.family, meta=ex.meta))
        return outl

    def gen_probe(f, split, n, seed):
        g = random.Random(f"{f}|{split}|{seed}")
        fac = facs[split]
        items = [getattr(fac, f)(g) for _ in range(n)]
        for it in items:
            assert it["dist"] >= PR.FLOOR, (f, it["dist"])
            assert all(e - s == 3 for s, e, _ in it["spans"]) or f not in ("lr_carry3", "lr_multi4")
            if f == "lr_recall1":
                assert all(e - s == 1 for s, e, _ in it["spans"])
        return items

    dropped = {}

    def synth_docs(f, split, target, seed, is_probe):
        docs, n, rnd, hashes = [], 0, 0, []
        while n < target:
            items = gen_probe(f, split, 2000, seed + rnd) if is_probe else encode_dmix(gen_dmix(f, split, 5000, seed + rnd))
            for it in items:
                ids = it["ids"]
                if len(ids) + (-len(ids)) % K <= SEQ:
                    arr = np.array(ids, dtype=np.uint16)
                    docs.append(arr); n += len(arr)
                else:
                    dropped[f"{f}|{split}"] = dropped.get(f"{f}|{split}", 0) + 1
            rnd += 1
        return docs, n

    def lang_docs(texts, budget, rng_):
        idx = list(range(len(texts)))
        rng_.shuffle(idx)
        docs, n = [], 0
        for b0 in range(0, len(idx), 2000):
            for e in tok.encode_batch([texts[i] for i in idx[b0:b0 + 2000]]):
                arr = np.array(e.ids + [EOS], dtype=np.uint16)
                docs.append(arr); n += len(arr)
                if n >= budget:
                    return docs, n, True
        return docs, n, False

    # ---- packed training-style shards
    train_hashes = None
    if a.eval_only:
        manifest = json.load(open(os.path.join(out, "MANIFEST.json")))
        train_hashes = set(np.load(os.path.join(out, "train", "synthetic_doc_sha1.npy")).tolist())
        budgets = {}
    else:
        budgets = {"train": a.train_tokens, "val": min(4e6, a.train_tokens), "test": min(4e6, a.train_tokens), "maintenance": min(2e6, a.train_tokens)}
    for split, budget in budgets.items():
        seed = SEEDS[split]
        docs, flags, stats, hashes = [], [], {}, []
        for grp, share in MIX.items():
            if grp == "language":
                for name, w in LANG_SPLIT.items():
                    d, n, full = lang_docs(lang[split][name], share * w * budget, random.Random(f"{name}|{split}|{seed}"))
                    docs += d; flags += [True] * len(d)
                    stats[name] = dict(docs=len(d), tokens=int(n), budget_met=full)
            else:
                per = share * budget / len(GROUPS[grp])
                for f in GROUPS[grp]:
                    d, n = synth_docs(f, split if split != "maintenance" else "maintenance", per, seed,
                                      grp == "longrange") if grp == "longrange" else \
                        synth_docs(f, "train" if split == "maintenance" else split, per, seed, False)
                    hashes += [hashlib.sha1(x.tobytes()).hexdigest() for x in d]
                    docs += d; flags += [False] * len(d)
                    stats[f] = dict(docs=len(d), tokens=int(n))
            log(f"{split}: {grp} generated")
        toks, dids, _ = pack(docs, random.Random(SEEDS["pack"] + seed), flags)
        sd = os.path.join(out, split)
        os.makedirs(sd, exist_ok=True)
        np.save(os.path.join(sd, "tok.npy"), toks)
        np.save(os.path.join(sd, "doc.npy"), dids)
        nonpad = int((toks != PAD).sum())
        entry = dict(rows=int(toks.shape[0]), row_tokens=int(toks.size), nonpad_tokens=nonpad,
                     pad_fraction=round(1 - nonpad / toks.size, 4), families=stats,
                     n_synthetic_docs=len(hashes), n_unique_synthetic=len(set(hashes)),
                     tok_sha256=hashlib.sha256(toks.tobytes()).hexdigest())
        if split == "train":
            train_hashes = set(hashes)
            np.save(os.path.join(sd, "synthetic_doc_sha1.npy"), np.array(sorted(train_hashes)))
        else:
            entry["synthetic_docs_identical_to_train"] = int(sum(h in train_hashes for h in hashes))
        manifest["splits"][split] = entry
        log(f"{split}: {toks.shape[0]} rows, nonpad {nonpad:,}")
        json.dump(manifest, open(os.path.join(out, "MANIFEST.partial.json"), "w"), indent=1)
        del docs, flags, toks, dids

    # ---- language evaluation rows (language only, packed like training; PROMPT 5.1: 256 rows each)
    tb = np.load(os.path.join(out, "token_bytes.npy"))
    if not a.eval_only:
        manifest["lang_eval"] = {}
    for split, srcs in ([] if a.eval_only else (("val", {"tinystories": ts_val, "wikitext": wk_val}), ("test", {"tinystories": ts_test, "wikitext": wk_test}))):
        for name, texts in srcs.items():
            docs = [np.array(e.ids + [EOS], dtype=np.uint16) for e in tok.encode_batch(texts)]
            toks, dids, _ = pack(docs, random.Random(SEEDS["pack"] + SEEDS[split] + len(name)), [True] * len(docs))
            toks, dids = toks[:256], dids[:256]
            ctx = same_doc_context(dids)
            d = os.path.join(out, "lang_eval")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{split}_{name}_tok.npy"), toks)
            np.save(os.path.join(d, f"{split}_{name}_doc.npy"), dids)
            np.save(os.path.join(d, f"{split}_{name}_ctx16.npy"), ctx)
            manifest["lang_eval"][f"{split}_{name}"] = dict(rows=int(len(toks)), nonpad_tokens=int((toks != PAD).sum()),
                                                            bytes=int(tb[toks.astype(np.int64)].sum()),
                                                            rows_ctx16_ge_512=int((ctx >= 512).sum()),
                                                            ctx16_median=int(np.median(ctx)))
            log(f"lang_eval {split}_{name}: {len(toks)} rows, ctx16>=512 in {(ctx >= 512).sum()}")

    # ---- item evaluation sets: 200 per synthetic family (val/test ID, OOD), 512 per probe family (val/test/OOD)
    manifest["eval"] = {}
    ev = os.path.join(out, "eval")
    os.makedirs(ev, exist_ok=True)
    for split in ("val", "test", "ood"):
        items, stats = [], {}
        fams = [f for g in ("symbolic", "code", "retrieval") for f in GROUPS[g]]
        if split == "ood":
            fams.append("variable_binding_newnames")
        for f in fams + PR.PROBES:
            is_probe = f in PR.PROBES
            n = 512 if is_probe else 200
            seed = SEEDS[split] + 17
            kept, leak, toolong, seen, rnd = [], 0, 0, set(), 0
            while len(kept) < n and rnd < 200:                  # sample fresh candidates until n non-leaked items
                if is_probe:
                    cand = gen_probe(f, split, n, seed + 1000 * rnd)
                else:
                    gsplit = "val" if f == "variable_binding_newnames" else split
                    cand = encode_dmix(gen_dmix(f, gsplit, n, seed + 1000 * rnd))
                rnd += 1
                for it in cand:
                    if len(it["ids"]) + (-len(it["ids"])) % K > SEQ:
                        toolong += 1; continue
                    h = hashlib.sha1(np.array(it["ids"], dtype=np.uint16).tobytes()).hexdigest()
                    if h in train_hashes:
                        leak += 1; continue
                    if h in seen:
                        continue                                # no duplicate items inside one eval set
                    seen.add(h)
                    kept.append(it)
                    if len(kept) == n:
                        break
            items += kept
            dd = [it["dist"] for it in kept if it.get("dist") is not None]
            stats[f] = dict(n=len(kept), dropped_train_duplicates=leak, dropped_too_long=toolong, candidate_rounds=rnd,
                            **(dict(dist_min=int(min(dd)), dist_median=int(np.median(dd)), dist_max=int(max(dd))) if dd else {}),
                            answer_tokens=sorted({e - s for it in kept for s, e, _ in it["spans"]}))
        torch.save(items, os.path.join(ev, f"{split}.pt"))
        manifest["eval"][split] = stats
        log(f"eval {split}: {len(items)} items")

    if not a.eval_only:
        manifest["dropped_over_seq_len"] = dropped
        manifest["build_seconds"] = round(time.time() - t0)
    else:
        manifest["eval_rebuilt"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    if os.path.exists(os.path.join(out, "MANIFEST.partial.json")):
        os.remove(os.path.join(out, "MANIFEST.partial.json"))
    log("DONE")


if __name__ == "__main__":
    main()
