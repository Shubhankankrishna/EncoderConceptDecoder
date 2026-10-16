"""Data loading: packed T = 1024 shards (uint16 tokens, int32 doc ids), Engram indices on CPU, NTP target mask.

NTP mask (PROMPT 4.5): the target of position t is ids[t+1]; it counts only if doc[t+1] == doc[t] (same document)
and ids[t+1] != PAD (padding is never a target; EOS is a real target).
"""
import os, threading, queue
import numpy as np
import torch
from hrm_encoder import engram_indices

PAD, EOS = 0, 1


class Canon:
    def __init__(self, data_dir):
        self.table = torch.from_numpy(np.load(os.path.join(data_dir, "canon.npy")))

    def __call__(self, ids):
        return self.table[ids]


def make_batch(tok_rows, doc_rows, canon, ecfg):
    ids = torch.from_numpy(tok_rows.astype(np.int64))
    doc = torch.from_numpy(doc_rows.astype(np.int64))
    eidx = engram_indices(canon(ids), doc, ecfg)
    return ids, doc, eidx


def target_mask(ids, doc):
    """(B, T-1) bool: which next-token targets count."""
    return (doc[:, 1:] == doc[:, :-1]) & (ids[:, 1:] != PAD)


class ShardStream:
    """Iterates the training shard in a fixed random row order (seeded); the cursor (rows consumed) is resumable.
    A background thread prepares the next batch (ids, doc, Engram indices) on CPU while the GPU works."""

    def __init__(self, data_dir, split, ecfg, batch_rows, seed=0, cursor=0, prefetch=4):
        d = os.path.join(data_dir, split)
        self.tok = np.load(os.path.join(d, "tok.npy"), mmap_mode="r")
        self.doc = np.load(os.path.join(d, "doc.npy"), mmap_mode="r")
        self.n = self.tok.shape[0]
        self.order = np.random.RandomState(seed).permutation(self.n)
        self.canon = Canon(data_dir)
        self.ecfg, self.B = ecfg, batch_rows
        self.cursor = cursor                       # rows handed to the trainer so far
        self._next = cursor                        # rows scheduled by the prefetch thread
        self.q = queue.Queue(maxsize=prefetch)
        self._stop = False
        self.th = threading.Thread(target=self._work, daemon=True)
        self.th.start()

    def _rows(self, start, B):
        idx = self.order[np.arange(start, start + B) % self.n]
        idx_sorted = np.sort(idx)                  # memmap-friendly read, then restore the order
        inv = np.argsort(np.argsort(idx))
        return np.asarray(self.tok[idx_sorted])[inv], np.asarray(self.doc[idx_sorted])[inv]

    def _work(self):
        while not self._stop:
            B = self.B
            t, d = self._rows(self._next, B)
            batch = make_batch(t, d, self.canon, self.ecfg)
            self._next += B
            self.q.put((B, batch))

    def next(self):
        B, batch = self.q.get()
        self.cursor += B
        return batch

    def epochs(self):
        return self.cursor / self.n

    def close(self):
        self._stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def load_rows(data_dir, split):
    d = os.path.join(data_dir, split)
    return np.load(os.path.join(d, "tok.npy"), mmap_mode="r"), np.load(os.path.join(d, "doc.npy"), mmap_mode="r")


def pack_items(items, T=1024):
    """Pack evaluation items (each ends with EOS) into T-token rows, each item padded to a multiple of 4 with its own
    doc id. Returns tok (R, T), doc (R, T), and per item (row, offset)."""
    rows_t, rows_d, where = [], [], []
    cur_t, cur_d, pos, did = np.zeros(T, np.int64), np.zeros(T, np.int64), 0, 0
    for it in items:
        ids = list(it["ids"])
        ids += [PAD] * ((-len(ids)) % 4)
        if pos + len(ids) > T:
            cur_d[pos:] = did
            rows_t.append(cur_t); rows_d.append(cur_d)
            cur_t, cur_d, pos, did = np.zeros(T, np.int64), np.zeros(T, np.int64), 0, 0
        cur_t[pos:pos + len(ids)] = ids
        cur_d[pos:pos + len(ids)] = did
        where.append((len(rows_t), pos))
        pos += len(ids); did += 1
    if pos:
        cur_d[pos:] = did
        rows_t.append(cur_t); rows_d.append(cur_d)
    return np.stack(rows_t), np.stack(rows_d), where
