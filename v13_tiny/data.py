"""Fixed-shape rows for training and evaluation (Dataset_v13.md §5.1, §6.3, §7.3).

One shape everywhere: rows of T = 1,024 tokens. A row holds whole documents of one family; every document starts at a
multiple of 4, so the concept-group clock restarts per document and no group straddles two documents (applied
identically to every architecture). Arrays per row:
  ids  tokens; doc  document number in the row (0 = tail padding); tgt  the next token; w  the weight on predicting
  tgt[t] at row t (procedural: 1/n_scored per example on the answer tokens and on determined demonstration outputs,
  zero on prompts; text: 1 on every next-token prediction); norm  the row's loss normaliser (examples, or tokens).
Batch loss = sum(w * CE) / sum(norm): per-example normalised CE averaged within the batch (§5.1); text is token CE.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

T_ROW = 1024
HERE = Path(__file__).resolve().parent
FAMILIES = ('grid', 'prog', 'state', 'plan', 'text')
SHARES = {'grid': .225, 'prog': .225, 'state': .225, 'plan': .225, 'text': .10}
TIER_RUNGS = ([0, 1], [2, 3, 4, 5], [6, 7])                      # prerequisite / working / train-side challenge


def curriculum(frac):
    """§7.3, by elapsed fraction of the base budget (wall clock)."""
    if frac < 0.1: return (0.7, 0.3, 0.0)
    if frac < 0.3: return (0.3, 0.6, 0.1)
    return (0.15, 0.65, 0.2)


def seed_of(*parts):
    return int.from_bytes(hashlib.blake2b('|'.join(map(str, parts)).encode(), digest_size=8).digest(), 'little')


class RowBuilder:
    def __init__(self, tok, T=T_ROW):
        self.tok, self.T = tok, T
        self.reset()

    def reset(self):
        T, pad = self.T, self.tok.pad
        self.ids = np.full(T, pad, np.int32); self.doc = np.zeros(T, np.int32)
        self.tgt = np.full(T, pad, np.int32); self.w = np.zeros(T, np.float32)
        self.pos = 0; self.ndoc = 0; self.norm = 0.0; self.spans = []; self.n_scored = 0

    def encode(self, e):
        ids, starts = self.tok.pieces(e['prompt'] + e['target'])
        npr = len(e['prompt'])
        t0 = starts[npr]
        scored = list(range(t0, len(ids)))
        for a, b in e.get('extra', ()):
            scored += list(range(starts[a], starts[b]))
        return ids, t0, sorted(set(scored))

    def fits(self, n):
        return self.pos + ((n + 3) // 4) * 4 <= self.T

    def add(self, ids, scored, text=False, keep=None):
        """Append one document; returns False if it does not fit."""
        L = len(ids); Lp = ((L + 3) // 4) * 4
        if self.pos + Lp > self.T: return False
        p = self.pos; self.ndoc += 1
        self.ids[p:p + L] = ids; self.doc[p:p + Lp] = self.ndoc
        self.tgt[p:p + L - 1] = ids[1:]
        if text:
            self.w[p:p + L - 1] = 1.0; self.norm += L - 1; self.n_scored += L - 1
        else:
            wt = 1.0 / len(scored)
            for j in scored: self.w[p + j - 1] = wt
            self.norm += 1.0; self.n_scored += len(scored)
        if keep is not None: self.spans.append((p, keep))
        self.pos += Lp
        return True

    def out(self):
        return dict(ids=self.ids.copy(), doc=self.doc.copy(), tgt=self.tgt.copy(), w=self.w.copy(),
                    norm=self.norm, used=int((self.doc > 0).sum()), scored=self.n_scored, ndoc=self.ndoc)


# ------------------------------------------------------------------------------------------------ worker side
_W = {}


def _init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    import tok, tasks
    _W['tok'] = tok.Tok(); _W['tasks'] = tasks


def make_row(family, k, frac, seed, phase='S0', task_list=None, rungs=None, T=T_ROW):
    """Row k of `family`'s training stream: examples drawn from the curriculum at elapsed fraction `frac`, packed until
    one does not fit (then up to 3 more candidates are tried, so tails are filled), rest padded."""
    if not _W: _init_worker()
    tk, Tm = _W['tok'], _W['tasks']
    rng = np.random.default_rng(seed_of(seed, family, phase, k))
    rb = RowBuilder(tk, T)
    tasks_ = task_list or Tm.FAMILY_TASKS[family]
    misses = 0; stats = {}
    while misses < 4:
        task = tasks_[int(rng.integers(len(tasks_)))]
        if rungs is not None: rung = int(rungs[int(rng.integers(len(rungs)))])
        else:
            tier = int(rng.choice(3, p=curriculum(frac))); rung = int(rng.choice(TIER_RUNGS[tier]))
        e = Tm.generate(task, rung, rng, mode='train', phase=phase)
        if e is None: misses += 1; continue
        ids, t0, scored = rb.encode(e)
        if not rb.add(ids, scored): misses += 1; continue
        s = stats.setdefault(f'{task}/r{rung}', [0, 0]); s[0] += 1; s[1] += len(ids)
    o = rb.out(); o['stats'] = stats; o['family'] = family
    return o


# ------------------------------------------------------------------------------------------------ text (main process)
class TextStream:
    """TinyStories training sample (deduplicated, whole stories), pre-tokenised once and packed in order."""
    def __init__(self, tok, n_stories=320000, cache=HERE / 'data' / 'ts_train.npz'):
        self.tok = tok
        if cache.exists():
            z = np.load(cache); self.flat, self.off = z['flat'], z['off']
        else:
            import pyarrow.parquet as pq
            shard = HERE / 'data' / 'raw_tinystories' / 'data' / 'train-00000-of-00004-2d5a1467fff1081b.parquet'
            texts = pq.read_table(shard).column('text').to_pylist()
            seen, keep = set(), []
            for s in texts:
                s = s.strip(); hs = hashlib.blake2b(s.encode(), digest_size=8).digest()
                if s and hs not in seen: seen.add(hs); keep.append(s)
                if len(keep) >= n_stories: break
            enc = tok.tk.encode_batch(keep, add_special_tokens=False)
            lens = [min(len(e.ids), T_ROW - 6) for e in enc]
            self.off = np.zeros(len(enc) + 1, np.int64); self.off[1:] = np.cumsum(lens)
            self.flat = np.concatenate([np.asarray(e.ids[:l], np.int16) for e, l in zip(enc, lens)])
            np.savez(cache, flat=self.flat, off=self.off)
        self.n = len(self.off) - 1
        self.order = np.random.default_rng(7).permutation(self.n)
        self.ptr = 0

    def story(self, i):
        return [self.tok.sid['<text>']] + self.flat[self.off[i]:self.off[i + 1]].astype(np.int64).tolist() + [self.tok.eos]

    def row(self, T=T_ROW):
        rb = RowBuilder(self.tok, T)
        while True:
            ids = self.story(int(self.order[self.ptr % self.n]))
            if not rb.add(ids, None, text=True): break
            self.ptr += 1
        o = rb.out(); o['stats'] = {'text': [o['ndoc'], o['used']]}; o['family'] = 'text'
        return o


def val_stories(tok, n=512):
    import pyarrow.parquet as pq
    shard = HERE / 'data' / 'raw_tinystories' / 'data' / 'validation-00000-of-00001-869c898b519ad725.parquet'
    texts = pq.read_table(shard).column('text').to_pylist()
    rng = np.random.default_rng(11); idx = rng.permutation(len(texts))[:n]
    out = []
    for i in idx:
        ids = tok.text(texts[int(i)].strip())
        if 8 < len(ids) < T_ROW - 6: out.append([tok.sid['<text>']] + ids + [tok.eos])
    return out


# ------------------------------------------------------------------------------------------------ the stream
class Stream:
    """Family-homogeneous batches under a token-quota scheduler (§5.1): the next batch's family is the one furthest
    below its share of processed non-padding tokens. Rows come from a worker pool, requested ahead in order."""
    def __init__(self, tok, B, seed, n_workers=5, ahead=2, state=None, phase='S0', families=None, shares=None,
                 task_lists=None, rungs=None, text=None):
        import multiprocessing as mp
        self.tok, self.B, self.seed, self.phase = tok, B, seed, phase
        self.families = families or FAMILIES
        self.shares = shares or SHARES
        self.task_lists = task_lists or {}
        self.rungs = rungs
        self.tokens = {f: 0 for f in self.families}; self.k = {f: 0 for f in self.families}
        self.text = text if 'text' in self.families else None
        if state:
            self.tokens.update(state['tokens']); self.k.update(state['k'])
            if self.text is not None: self.text.ptr = state.get('text_ptr', 0)
        self.pool = mp.get_context('spawn').Pool(n_workers, initializer=_init_worker)
        self.ahead = ahead; self.pending = {f: [] for f in self.families if f != 'text'}
        self.frac = 0.0

    def state(self):
        return dict(tokens=dict(self.tokens), k=dict(self.k), text_ptr=self.text.ptr if self.text else 0)

    def _fill(self, f):
        while len(self.pending[f]) < self.ahead * self.B:
            k = self.k[f] + len(self.pending[f])
            self.pending[f].append(self.pool.apply_async(
                make_row, (f, k, self.frac, self.seed, self.phase, self.task_lists.get(f), self.rungs)))

    def next_family(self):
        tot = sum(self.tokens.values()) + 1
        return max(self.families, key=lambda f: self.shares[f] * tot - self.tokens[f])

    def batch(self, frac):
        self.frac = frac
        for f in self.pending: self._fill(f)
        f = self.next_family()
        if f == 'text':
            rows = [self.text.row() for _ in range(self.B)]
        else:
            rows = [self.pending[f].pop(0).get() for _ in range(self.B)]
            self.k[f] += self.B
            self._fill(f)
        self.tokens[f] += sum(r['used'] for r in rows)
        return f, collate(rows)

    def close(self):
        self.pool.terminate()


def collate(rows):
    st = {}
    for r in rows:
        for key, (a, b) in r['stats'].items():
            s = st.setdefault(key, [0, 0]); s[0] += a; s[1] += b
    return dict(ids=np.stack([r['ids'] for r in rows]), doc=np.stack([r['doc'] for r in rows]),
                tgt=np.stack([r['tgt'] for r in rows]), w=np.stack([r['w'] for r in rows]),
                norm=float(sum(r['norm'] for r in rows)), used=sum(r['used'] for r in rows),
                scored=sum(r['scored'] for r in rows), stats=st)
