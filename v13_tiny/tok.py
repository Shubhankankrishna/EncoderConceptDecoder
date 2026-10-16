"""Shared tokenizer (Dataset_v13.md §6.3): at most 2,048 entries, reserved structural tokens, single-token digits,
colours and actions, byte-level BPE fallback for text. Fitted only on the TinyStories training sample and S0 renderings.

    python3 tok.py fit        fits and writes data/tokenizer.json (run once, before any scored step)

Pieces (tasks.py) are encoded one by one: a special piece maps to its id, any other piece is BPE-encoded on its own
(cached), so the boundary between prompt and target is exact.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB = 2048
SPECIALS = (['<pad>', '<bos>', '<eos>', '<ans>', '<neg>', '<x>', '<nl>', '<legend>', '<text>',
             '<grid>', '<arith>', '<list>', '<reg>', '<graph>', '<state>', '<plan>',
             '<in>', '<out>'] + [f'<c{i}>' for i in range(10)] + [f'<k{i}>' for i in range(10)] +
            ['<add>', '<sub>', '<mul>', '<div>', '<min>', '<max>', '<sort>', '<rsort>', '<filter>', '<gt>',
             '<rep>', '<{>', '<}>', '<if>', '<then>', '<end>', '<ret>',
             '<reach>', '<dist>', '<path>', '<yes>', '<no>',
             '<at>', '<move>', '<take>', '<drop>', '<give>', '<put>', '<q_loc>', '<q_owner>', '<q_count>', '<q_link>',
             '<.>', '<#>', '<S>', '<A>', '<B>', '<C>', '<kk>', '<DD>', '<U>', '<D>', '<L>', '<R>', '<pick>', '<open>',
             '<goto>'] + [f'<a{i}>' for i in range(4)] + [f'<rsv{i}>' for i in range(24)])
PATH = HERE / 'data' / 'tokenizer.json'


def fit(n_stories=120000):
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    import tasks as T
    shard = HERE / 'data' / 'raw_tinystories' / 'data' / 'train-00000-of-00004-2d5a1467fff1081b.parquet'
    stories = pq.read_table(shard).column('text').to_pylist()[:n_stories]
    rng = np.random.default_rng(13)
    texts = []                                                    # text pieces of S0 renderings (no S1-S6 content)
    for task in T.TASKS:
        for i in range(3000 if task in ('state',) else 300):
            e = T.generate(task, i % 8, rng, mode='train', phase='S0')
            if e: texts.append(' '.join(p for p in e['prompt'] + e['target'] if p not in SPECIALS))
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS, min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(stories + texts, trainer=tr)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tk.save(str(PATH))
    print('vocab', tk.get_vocab_size(), 'saved', PATH)


class Tok:
    def __init__(self, path=PATH):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.vocab = self.tk.get_vocab_size()
        assert self.vocab <= VOCAB, self.vocab
        self.sid = {s: self.tk.token_to_id(s) for s in SPECIALS}
        assert all(v is not None for v in self.sid.values())
        assert [self.sid[s] for s in SPECIALS] == list(range(len(SPECIALS)))
        self.cache = {}
        self.pad, self.eos = self.sid['<pad>'], self.sid['<eos>']

    def piece(self, p):
        if p in self.sid: return [self.sid[p]]
        ids = self.cache.get(p)
        if ids is None:
            ids = self.tk.encode(p, add_special_tokens=False).ids
            if len(self.cache) < 200000: self.cache[p] = ids
        return ids

    def pieces(self, ps):
        out, starts = [], []
        for p in ps:
            starts.append(len(out)); out.extend(self.piece(p))
        starts.append(len(out))
        return out, starts

    def text(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=False)

    def hash(self):
        import hashlib
        return hashlib.sha256(PATH.read_bytes()).hexdigest()[:16]


if __name__ == '__main__':
    if sys.argv[1:] == ['fit']:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
        fit()
