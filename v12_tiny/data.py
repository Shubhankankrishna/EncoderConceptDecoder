"""Episode batches from the packed pinpad splits (right-padded, per-target weights, ground truth)."""
from pathlib import Path
import numpy as np
import torch
import pinpad as P

DATA = Path(__file__).resolve().parent / 'data'


class Split:
    def __init__(self, name, world=None):
        if name == 'test':
            raise PermissionError('the test split is sealed')
        z = np.load(DATA / f'{name}.npz')
        self.tokens, self.tgt_col, self.abs_act = z['tokens'], z['tgt_col'], z['abs_act']
        self.off, self.header, self.task_ids = z['offsets'], z['header'], z['task_ids']
        self.n = len(self.off) - 1
        self.world = world or P.World()

    def episode(self, i):
        a, b = self.off[i], self.off[i + 1]
        return self.tokens[a:b], self.tgt_col[a:b], self.abs_act[a:b], int(self.header[i])

    def batch(self, idx, lam_obs=0.1, device='cpu', pad_to=None):
        eps = [self.episode(i) for i in idx]
        T = max(len(e[0]) for e in eps) if pad_to is None else max(pad_to, max(len(e[0]) for e in eps))
        B = len(eps)
        ids = np.zeros((B, T), np.int64); w = np.zeros((B, T), np.float32)
        tc = np.full((B, T), -1, np.int64); aa = np.full((B, T), -1, np.int64)
        isact = np.zeros((B, T), bool); lens = np.zeros(B, np.int64); hdr = np.zeros(B, np.int64)
        for b, (tok, col, ab, h) in enumerate(eps):
            n = len(tok); ids[b, :n] = tok; tc[b, :n] = col; aa[b, :n] = ab; lens[b] = n; hdr[b] = h
            # target at position i is tok[i+1]; score targets after p_0 (index h)
            for i in range(h, n - 1):
                nxt = tok[i + 1]
                if P.ACT0 <= nxt < P.ACT0 + 4: w[b, i] = 1.0; isact[b, i] = True
                elif nxt == P.END: w[b, i] = 1.0
                else: w[b, i] = lam_obs
        t = lambda x: torch.from_numpy(x).to(device)
        tgt = np.zeros_like(ids); tgt[:, :-1] = ids[:, 1:]
        return dict(ids=t(ids), targets=t(tgt), weights=t(w), tgt_col=t(tc), abs_act=t(aa), is_act=t(isact),
                    lengths=t(lens), header=t(hdr))


class Sampler:
    """Fresh-without-replacement episode order per epoch, deterministic in the seed. Length bucketing: a pool of
    `pool` batches is drawn, sorted by length, cut into batches, and the batches are shuffled (cuts padding)."""
    def __init__(self, split, batch, seed, pool=32):
        self.split, self.B, self.pool = split, batch, pool
        self.lens = np.diff(split.off)
        self.rng = np.random.default_rng(seed); self.perm = self.rng.permutation(split.n); self.pos = 0
        self.queue = []

    def _refill(self):
        n = self.B * self.pool
        if self.pos + n > len(self.perm):
            self.perm = self.rng.permutation(self.split.n); self.pos = 0
        idx = self.perm[self.pos:self.pos + n]; self.pos += n
        idx = idx[np.argsort(self.lens[idx], kind='stable')]
        batches = [idx[i:i + self.B] for i in range(0, n, self.B)]
        order = self.rng.permutation(len(batches))
        self.queue = [batches[i] for i in order]

    def next_idx(self):
        if not self.queue: self._refill()
        return self.queue.pop()


# ----------------------------------------------------------------------------------------------------------
# Mixed corpus (advisor #4, user instruction "don't overfit to one kind of data distribution").
# Joint vocabulary: pinpad ids 0..65 unchanged; the V10.3-tiny BPE (1,024, read-only from v10_3_tiny/data) is
# offset by BPE_OFF = 66. Families: pinpad (scored after the header), text (random 256-token crops, all tokens
# scored), math (answer tokens only: response-only scoring). newmath is held out (sleep test). Sudoku excluded:
# its fixed 171-token layout is the known period-alignment confound for fixed groups.
# ----------------------------------------------------------------------------------------------------------
BPE_DIR = Path(__file__).resolve().parent.parent / 'v10_3_tiny' / 'data'
BPE_OFF = 66
VOCAB = BPE_OFF + 1024
MIX = (('pinpad', 0.6), ('text', 0.25), ('math', 0.15))
BATCH = dict(pinpad=128, text=32, math=128, newmath=128)
CTX = 256


class BPEFamily:
    def __init__(self, family, split):
        base = BPE_DIR / f'{family}_{split}'
        self.arr = np.load(str(base) + '.npy', mmap_mode='r')
        self.off = np.load(str(base) + '_offsets.npy'); self.ans = np.load(str(base) + '_answer.npy')
        self.lens = np.diff(self.off); self.n = len(self.lens); self.family = family

    def window(self, i, rng):
        a, L = self.off[i], self.lens[i]
        if self.family == 'text' and L > CTX:
            s = int(rng.integers(0, L - CTX + 1)) if rng is not None else 0
            return np.asarray(self.arr[a + s:a + s + CTX], np.int64), 0
        return np.asarray(self.arr[a:a + min(L, CTX)], np.int64), int(self.ans[i])

    def batch(self, idx, rng=None, device='cpu', pad_to=None):
        rows = [self.window(i, rng) for i in idx]
        T = max(len(r[0]) for r in rows) if pad_to is None else pad_to
        B = len(rows)
        ids = np.zeros((B, T), np.int64); w = np.zeros((B, T), np.float32); lens = np.zeros(B, np.int64)
        for b, (tok, ans) in enumerate(rows):
            n = len(tok); ids[b, :n] = tok + BPE_OFF; lens[b] = n
            if self.family == 'text': w[b, :n - 1] = 1.0
            else: w[b, max(ans - 1, 0):n - 1] = 1.0                    # targets ans..n-1 (answer tokens + EOS)
        t = lambda x: torch.from_numpy(x).to(device)
        tgt = np.zeros_like(ids); tgt[:, :-1] = ids[:, 1:]
        z = np.full((B, T), -1, np.int64)
        return dict(ids=t(ids), targets=t(tgt), weights=t(w), tgt_col=t(z), abs_act=t(z.copy()),
                    is_act=t(np.zeros((B, T), bool)), lengths=t(lens), header=t(np.zeros(B, np.int64)), family=self.family)


class Mixed:
    """Family-per-batch stream, deterministic in the seed (identical across arms with the same seed)."""
    def __init__(self, seed, lam_obs=0.1):
        self.rng = np.random.default_rng(seed); self.lam = lam_obs
        self.pin = Split('train'); self.pin_s = Sampler(self.pin, BATCH['pinpad'], seed)
        self.fam = dict(text=BPEFamily('text', 'train'), math=BPEFamily('math', 'train'))
        tl = self.fam['text'].lens.astype(np.float64)
        self.text_p = np.minimum(tl, 4 * CTX) / np.minimum(tl, 4 * CTX).sum()   # long docs: more windows, capped
        self.math_s = Sampler(_N(self.fam['math'].n), BATCH['math'], seed + 1)
        self.names = [m[0] for m in MIX]; self.p = np.array([m[1] for m in MIX])

    def next(self, device):
        fam = self.names[int(self.rng.choice(len(self.names), p=self.p))]
        # lengths are rounded up to multiples of 32: a handful of shapes keeps the MPS allocator cache bounded
        bucket = lambda L: int(np.ceil(L / 32) * 32)
        if fam == 'pinpad':
            idx = self.pin_s.next_idx()
            b = self.pin.batch(idx, lam_obs=self.lam, device=device, pad_to=bucket(int(self.pin_s.lens[idx].max())))
            b['family'] = 'pinpad'
        elif fam == 'text':
            idx = self.rng.choice(self.fam['text'].n, BATCH['text'], p=self.text_p)
            b = self.fam['text'].batch(idx, self.rng, device, pad_to=CTX)
        else:
            idx = self.math_s.next_idx(); b = self.fam['math'].batch(idx, None, device, pad_to=96)
        return b, idx


class _N:
    """Minimal stand-in so Sampler can permute a BPE family (only .n and .off are used)."""
    def __init__(self, n):
        self.n = n; self.off = np.arange(n + 1)
