"""Data for the v4 tiny experiment (CPU).

Text: documents re-extracted from v3's packed shards (run_m2_25M/data/{train,val}, read-only), stored once as a flat
token array + offsets (v4_tiny/data/). Rows are re-packed per k: every document padded to a multiple of k, rows of T
tokens, the SAME document order for every k (so arms see the same documents; padding differs, as it would at full size).
Probes: v3's ProbeFactory families recall1 / carry3 / multi4 with gaps of 80-200 tokens (beyond the decoder's reach of
70, so only the concept route can answer), fillers = real sentences from the training documents. Items are packed with
k-token padding, one doc id per item, every item starting on a k boundary.
"""
import os
import random
import numpy as np
import torch
from tokenizers import Tokenizer

from hrm_encoder import engram_indices
from probes import ProbeFactory, VALUE_CANDIDATES

PAD, EOS = 0, 1
SRC = "/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "v4_tiny", "data"))
FAMILIES = ("lr_recall1", "lr_carry3", "lr_multi4")


# ----------------------------------------------------------------------------- one-time extraction
def extract_docs(split, n_rows, seed=0):
    """Real documents (pads stripped) from a random subset of v3's packed rows."""
    tok = np.load(os.path.join(SRC, split, "tok.npy"), mmap_mode="r")
    doc = np.load(os.path.join(SRC, split, "doc.npy"), mmap_mode="r")
    rows = np.sort(np.random.RandomState(seed).choice(tok.shape[0], min(n_rows, tok.shape[0]), replace=False))
    flat, offs = [], [0]
    for r in rows:
        t = np.asarray(tok[r]).astype(np.int64); d = np.asarray(doc[r])
        cuts = np.flatnonzero(np.diff(d)) + 1
        for seg_t in np.split(t, cuts):
            seg = seg_t[seg_t != PAD]
            if len(seg) >= 2:
                flat.append(seg.astype(np.uint16)); offs.append(offs[-1] + len(seg))
    return np.concatenate(flat), np.array(offs, dtype=np.int64)


def build_store(n_train_rows=40000, n_val_rows=600):
    os.makedirs(OUT, exist_ok=True)
    for split, n in (("train", n_train_rows), ("val", n_val_rows)):
        f = os.path.join(OUT, f"text_{split}.npz")
        if not os.path.exists(f):
            flat, offs = extract_docs(split, n)
            np.savez(f, flat=flat, offs=offs)
            print(f"{split}: {len(offs) - 1} documents, {len(flat):,} tokens -> {f}")


class TextStore:
    def __init__(self, split):
        z = np.load(os.path.join(OUT, f"text_{split}.npz"))
        self.flat, self.offs = z["flat"], z["offs"]
        self.n = len(self.offs) - 1

    def doc(self, i):
        return self.flat[self.offs[i]:self.offs[i + 1]].astype(np.int64)


def token_freq(store, V=8192):
    return np.bincount(store.flat.astype(np.int64), minlength=V)


# ----------------------------------------------------------------------------- packing
def pack_sequences(seqs, T, k, first_doc_id=0):
    """Pack token sequences (each its own document) into rows of T: each padded to a multiple of k, a new row when it
    does not fit; the row tail is PAD with a fresh doc id. Sequences longer than T are cut into T-sized pieces.
    Returns tok (R, T), doc (R, T), where = [(row, offset, length)] per sequence piece."""
    rows_t, rows_d, where = [], [], []
    cur_t = np.zeros(T, np.int64); cur_d = np.zeros(T, np.int64); pos = 0; did = first_doc_id
    def flush():
        nonlocal cur_t, cur_d, pos, did
        if pos:
            cur_d[pos:] = did; did += 1
            rows_t.append(cur_t); rows_d.append(cur_d)
        cur_t = np.zeros(T, np.int64); cur_d = np.zeros(T, np.int64); pos = 0
    for s in seqs:
        for a in range(0, len(s), T):
            piece = np.asarray(s[a:a + T], np.int64)
            L = len(piece); Lp = -(-L // k) * k
            if pos + Lp > T:
                flush()
            assert pos % k == 0
            cur_t[pos:pos + L] = piece; cur_d[pos:pos + Lp] = did
            where.append((len(rows_t), pos, L))
            pos += Lp; did += 1
    flush()
    return np.stack(rows_t), np.stack(rows_d), where


# ----------------------------------------------------------------------------- probes
class TinyProbes(ProbeFactory):
    """v3's probe families with fillers taken from real sentences and gaps of 80-200 tokens ('far': 200-300)."""
    def __init__(self, tok, fill_seqs, split, gap=(80, 200)):
        self.tok, self.split, self.ood = tok, split, False
        syl = ["ka", "lo", "mi", "ra", "te", "su", "no", "vi", "pe", "da", "ro", "li", "ma", "tu", "se", "ne"]
        self.names = sorted({a + b + c for a in syl for b in syl for c in syl if len({a, b, c}) == 3})
        self.values = VALUE_CANDIDATES
        self.fill = fill_seqs
        self._cache = {}
        self.gap_range = gap

    def gap(self, rng):
        return rng.randint(*self.gap_range)

    def item(self, rng, fam):
        return getattr(self, fam)(rng)


def filler_sentences(store, tok, n_docs=3000, seed=0):
    """Sentences (5-40 tokens) cut at '.' from real training documents."""
    dot = tok.encode(".").ids[0]
    rng = np.random.RandomState(seed)
    out = []
    for i in rng.choice(store.n, min(n_docs, store.n), replace=False):
        d = store.doc(i)
        cuts = np.flatnonzero(d == dot) + 1
        for s in np.split(d, cuts):
            if 5 <= len(s) <= 40 and s[-1] == dot and EOS not in s:
                out.append([int(x) for x in s])
    return out


# ----------------------------------------------------------------------------- streams
class Canon:
    def __init__(self):
        self.table = torch.from_numpy(np.load(os.path.join(SRC, "canon.npy")))

    def __call__(self, ids):
        return self.table[ids]


def to_batch(tok_rows, doc_rows, canon, ecfg):
    ids = torch.from_numpy(np.asarray(tok_rows, np.int64)); doc = torch.from_numpy(np.asarray(doc_rows, np.int64))
    return ids, doc, engram_indices(canon(ids), doc, ecfg)


class MixedStream:
    """Training rows: with probability p_syn a row of packed probe items, else a row of packed text documents.
    The row-type sequence, the document order and the item sequence depend only on the seed, not on k."""
    def __init__(self, store, probes, T, k, B, p_syn=0.5, seed=0, families=FAMILIES):
        self.store, self.probes, self.T, self.k, self.B, self.p = store, probes, T, k, B, p_syn
        self.rng_type = random.Random(seed)
        self.rng_item = random.Random(seed + 1)
        self.order = np.random.RandomState(seed).permutation(store.n)
        self.di = 0
        self.fams = families
        self.buf_t, self.buf_d = [], []
        self.pending = None

    def _text_row(self):
        while True:
            if self.buf_t:
                return self.buf_t.pop(0), self.buf_d.pop(0)
            seqs = [self.store.doc(self.order[(self.di + j) % self.store.n]) for j in range(64)]
            self.di += 64
            t, d, _ = pack_sequences(seqs, self.T, self.k)
            self.buf_t, self.buf_d = list(t[:-1]), list(d[:-1])        # drop the last (partial) row

    def _syn_row(self):
        items, n = [], 0
        while True:
            it = self.pending if self.pending is not None else \
                self.probes.item(self.rng_item, self.rng_item.choice(self.fams))
            self.pending = None
            L = -(-len(it["ids"]) // self.k) * self.k
            if L > self.T:
                continue
            if n + L > self.T:
                self.pending = it                                       # carried to the next probe row
                break
            items.append(it["ids"]); n += L
        t, d, _ = pack_sequences(items, self.T, self.k)
        assert len(t) == 1
        return t[0], d[0]

    def next(self):
        tt, dd = [], []
        for _ in range(self.B):
            t, d = self._syn_row() if self.rng_type.random() < self.p else self._text_row()
            tt.append(t); dd.append(d)
        return np.stack(tt), np.stack(dd)


def val_text_rows(store, T, k, n_rows=64, seed=123):
    order = np.random.RandomState(seed).permutation(store.n)
    seqs, i, rows = [], 0, None
    while True:
        seqs += [store.doc(order[j % store.n]) for j in range(i, i + 200)]
        i += 200
        t, d, _ = pack_sequences(seqs, T, k)
        if len(t) > n_rows:
            return t[:n_rows], d[:n_rows]


def probe_items(probes, n_per_family, seed, families=FAMILIES):
    rng = random.Random(seed)
    return [probes.item(rng, f) for f in families for _ in range(n_per_family)]


def item_rows(items, T, k):
    t, d, where = pack_sequences([it["ids"] for it in items], T, k)
    return t, d, where


if __name__ == "__main__":
    build_store()
    tr, va = TextStore("train"), TextStore("val")
    tok = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
    fill = filler_sentences(tr, tok)
    print("filler sentences", len(fill))
    pf = TinyProbes(tok, fill, "val")
    its = probe_items(pf, 4, 0)
    for it in its[:3]:
        print(it["family"], len(it["ids"]), it["dist"], tok.decode(it["ids"][-40:]))
    for k in (4, 6, 8):
        t, d, w = item_rows(its, 480, k)
        assert all(off % k == 0 for _, off, _ in w)
        vt, vd = val_text_rows(va, 480, k, 8)
        print(f"k={k}: probe rows {t.shape}, val rows {vt.shape}, real frac {(vt != 0).mean():.3f}")
