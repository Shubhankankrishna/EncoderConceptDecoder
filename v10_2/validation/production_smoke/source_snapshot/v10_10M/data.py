import json
from pathlib import Path
import numpy as np
import torch
from .prepare_data import DATA


class Corpus:
    def __init__(self, split='train', seed=1011, data_dir=DATA):
        self.rng = np.random.default_rng(seed)
        self.data_dir, self.split = Path(data_dir), split
        self.arrays = {}
        self.manifest = json.loads((self.data_dir/'manifest.json').read_text())
        for family in ('text', 'math', 'sudoku_easy', 'sudoku_hard'):
            base = self.data_dir/f'{family}_{split}'
            self.arrays[family] = (np.load(str(base)+'.npy', mmap_mode='r'),
                                   np.load(str(base)+'_offsets.npy'), np.load(str(base)+'_answer.npy'))

    def batch(self, batch_size, context=384, family=None, progress=0, device='cpu'):
        if family is None:
            # Batch mixture; log actual supervised tokens per family as well.
            # Sudoku easy fraction anneals .9 -> .6 within its 15% allocation.
            easy = .9-.3*min(1., progress)
            family = self.rng.choice(list(self.arrays), p=[.65, .20, .15*easy, .15*(1-easy)])
        arr, offsets, starts = self.arrays[family]
        if len(starts) == 0:
            raise ValueError('Empty required data family '+family)
        seqs, answer_positions, chosen = [], [], []
        for _ in range(batch_size):
            i = int(self.rng.integers(len(starts)))
            seq = np.array(arr[offsets[i]:offsets[i+1]], dtype=np.int64)
            answer = int(starts[i])
            if len(seq) > context+1:
                if family != 'text':
                    raise ValueError('Never truncate a symbolic answer')
                begin = int(self.rng.integers(0, len(seq)-context))
                seq = seq[begin:begin+context+1]
                answer = 1
            seqs.append(seq); answer_positions.append(answer); chosen.append(i)
        t = min(context, ((max(len(s) for s in seqs)-2)//8+1)*8)
        x = np.zeros((batch_size, t), np.int64)
        y, mask = np.zeros_like(x), np.zeros_like(x, dtype=bool)
        lengths = []
        for b, (s, a) in enumerate(zip(seqs, answer_positions)):
            n = len(s)-1
            x[b, :n], y[b, :n] = s[:-1], s[1:]
            mask[b, max(0, a-1):n] = True
            lengths.append(n)
        return dict(ids=torch.tensor(x, device=device), targets=torch.tensor(y, device=device),
                    target_mask=torch.tensor(mask, device=device), lengths=torch.tensor(lengths, device=device),
                    family=family, chosen=chosen)


def verify_sudoku(puzzle, response):
    # Strict 81 digit answer; no acceptance for copied clues, prefixes or padding.
    answer = response.strip()
    if len(answer) != 81 or any(ch not in '123456789' for ch in answer):
        return dict(valid=False, reason='format')
    if any(q != '0' and q != a for q, a in zip(puzzle, answer)):
        return dict(valid=False, reason='clues')
    grid = np.asarray([int(c) for c in answer]).reshape(9, 9)
    expected = set(range(1, 10))
    for i in range(9):
        if set(grid[i]) != expected or set(grid[:, i]) != expected:
            return dict(valid=False, reason='row_or_column')
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if set(grid[r:r+3, c:c+3].flat) != expected:
                return dict(valid=False, reason='box')
    return dict(valid=True, reason='accepted')
