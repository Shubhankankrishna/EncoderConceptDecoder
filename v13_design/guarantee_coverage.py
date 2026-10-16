"""Where the guarantee bites, for v13_NOTES.md §12.2: the share of scored targets whose row has part of its own
document beyond the decoder's direct reach R.

    python3 v13_design/guarantee_coverage.py

A row t (0-based within its document) sees tokens t-R..t directly, so some of its document lies beyond reach when
t > R. Shares are weighted by the training scores of §12.2 and taken over the training split.
  - pinpad: V12's episodes (v12_tiny/data/train.npz); targets after the header, actions and END 1, positions 0.1.
  - text:   V10.3-tiny documents, drawn uniformly; a document longer than 1,024 tokens gives one 1,024-token crop,
            shorter ones are packed whole (each is its own document). Every target scored.
  - math:   V10.3-tiny problems, packed whole; answer tokens and EOS scored.
  - recall: the fraction of answers whose fact lies beyond R, with distances {48, 96, 192, 384, 768} drawn uniformly.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REACHES = (71, 101, 360)                     # D-confined and TF-confined; D-own-SWA; TF-SWA-360
CTX = 1024
ACT0, END = 4, 3                             # v12_tiny/pinpad.py


def pinpad(R):
    z = np.load(ROOT / 'v12_tiny/data/train.npz')
    tok, off, hdr = z['tokens'], z['offsets'], z['header']
    beyond = total = 0.0
    for i in range(len(hdr)):
        a, b, h = off[i], off[i + 1], int(hdr[i])
        nxt = tok[a + h + 1:b].astype(np.int64)                 # targets of rows h .. n-2
        w = np.where(((nxt >= ACT0) & (nxt < ACT0 + 4)) | (nxt == END), 1.0, 0.1)
        rows = np.arange(h, b - a - 1)
        total += w.sum(); beyond += w[rows > R].sum()
    return beyond / total


def bpe_family(name, R):
    base = ROOT / f'v10_3_tiny/data/{name}_train'
    lens = np.diff(np.load(str(base) + '_offsets.npy'))
    if name == 'text':
        n = np.minimum(lens, CTX) - 1                           # rows 0 .. n-1 scored
        return np.clip(n - R, 0, None).sum() / n.sum()
    ans = np.load(str(base) + '_answer.npy')                    # first answer token; rows ans-1 .. L-2 scored
    first = np.maximum(ans - 1, 0)
    last = lens - 2
    n = last - first + 1
    beyond = np.clip(last - np.maximum(first, R + 1) + 1, 0, None)
    return beyond.sum() / n.sum(), int(lens.max())


def recall(R):
    d = np.array([48, 96, 192, 384, 768])
    return float((d > R).mean())


def main():
    print(f'{"family":10s}' + ''.join(f'{"R = " + str(R):>12s}' for R in REACHES))
    print(f'{"pinpad":10s}' + ''.join(f'{pinpad(R):12.1%}' for R in REACHES))
    print(f'{"text":10s}' + ''.join(f'{bpe_family("text", R):12.1%}' for R in REACHES))
    m = [bpe_family('math', R) for R in REACHES]
    print(f'{"math":10s}' + ''.join(f'{x[0]:12.1%}' for x in m) + f'   (longest problem {m[0][1]} tokens)')
    print(f'{"recall":10s}' + ''.join(f'{recall(R):12.1%}' for R in REACHES))


if __name__ == '__main__':
    main()
