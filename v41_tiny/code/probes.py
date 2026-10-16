"""Long-range probe families (PROMPT 4.4). Every item puts its evidence at least FLOOR tokens before the answer
(FLOOR = 128 > 90 required; chosen above the S2 stack decoder's reach of 103, DECISIONS D-011).

Documents are assembled from pre-encoded token segments, so a name or value has exactly the same tokens in the fact
and in the answer (each segment is encoded once, with its leading space, and reused). Distances are measured in
tokens on the assembled document and asserted.

Item = dict(ids, spans=[(start, end, label)], family, meta, dist) with ids ending in EOS; spans index the answer tokens
(the prediction for token i is logits[i-1]).
"""
import random

FLOOR = 128
EOS = 1

# identifier syllables: ID set = D-MIX SYL_A (already in training text); OOD set disjoint from SYL_A and D-MIX's SYL_B
SYL_ID = ["ka", "lo", "mi", "ra", "te", "su", "no", "vi", "pe", "da", "ro", "li", "ma", "tu", "se", "ne"]
SYL_OOD = ["bo", "ga", "hi", "ju", "ke", "ly", "mu", "ny", "po", "ri", "sa", "ti", "wo", "ze", "di", "fe"]

VALUE_CANDIDATES = (
    "red blue green black white brown pink gold silver gray orange purple yellow "
    "cat dog cow pig hen fox owl bee ant bat bear bird duck fish frog goat horse lion mouse sheep snake tiger wolf "
    "apple pear plum lemon grape melon peach cherry bread cake milk tea rice egg corn bean nut pie jam soup "
    "cup box hat bag key pen book lamp door bed chair table ball kite boat car bus ship train drum bell ring coin "
    "star moon sun tree leaf rock sand snow rain wind fire lake hill road river stone flower grass cloud "
    "king queen boy girl baby man woman friend farmer doctor teacher "
    "big small tall short happy sad fast slow warm cold soft loud dark bright old new"
).split()


class ProbeFactory:
    """Holds the tokenizer-dependent pieces (encoded filler sentences, names, values) for one split."""

    def __init__(self, tok, filler_sentences, values, split):
        self.tok, self.split = tok, split
        self.ood = split == "ood"
        syl = SYL_OOD if self.ood else SYL_ID
        self.names = sorted({a + b + c for a in syl for b in syl for c in syl if len({a, b, c}) == 3})
        self.values = values
        enc = lambda strs: [e.ids for e in tok.encode_batch(strs)]
        self.fill = enc([" " + s.strip() for s in filler_sentences])
        self._cache = {}

    def e(self, s):
        """Encode a segment once and reuse it (identical tokens wherever it appears)."""
        if s not in self._cache:
            self._cache[s] = self.tok.encode(s).ids
        return self._cache[s]

    def filler_sents(self, rng, n_tokens):
        out, n = [], 0
        while n < n_tokens:
            s = self.fill[rng.randrange(len(self.fill))]
            out.append(s)
            n += len(s)
        return out

    def filler(self, rng, n_tokens):
        return [t for s in self.filler_sents(rng, n_tokens) for t in s]

    def gap(self, rng):
        lo, hi = (500, 820) if self.ood else (FLOOR, 420)
        return rng.randint(lo, hi)

    # ------------------------------------------------------------------ families
    def lr_recall1(self, rng):
        return self._facts(rng, n_facts=1, L=1, fam="lr_recall1")

    def lr_carry3(self, rng):
        return self._facts(rng, n_facts=1, L=3, fam="lr_carry3")

    def lr_multi4(self, rng):
        return self._facts(rng, n_facts=4, L=3, fam="lr_multi4")

    def _facts(self, rng, n_facts, L, fam):
        names = rng.sample(self.names, n_facts)
        vals = [[rng.choice(self.values) for _ in range(L)] for _ in range(n_facts)]
        ids = self.filler(rng, rng.randint(0, 40))
        ends = []
        for nm, v in zip(names, vals):
            ids += self.e(" The code of") + self.e(" " + nm) + self.e(" is")
            for w in v:
                ids += self.e(" " + w)
            ends.append(len(ids) - 1)                       # evidence = last value token of the fact
            ids += self.e(".")
        target_gap = self.gap(rng)
        ids += self.filler(rng, target_gap)
        spans, dists = [], []
        for qi in rng.sample(range(n_facts), n_facts):      # random query order
            ids += self.e(" What is the code of") + self.e(" " + names[qi]) + self.e("?")
            ids += self.e(" The code of") + self.e(" " + names[qi]) + self.e(" is")
            s = len(ids)
            for w in vals[qi]:
                ids += self.e(" " + w)
            spans.append((s, len(ids), f"q{qi}"))
            dists.append(s - ends[qi])
            ids += self.e(".")
        ids.append(EOS)
        return dict(ids=ids, spans=spans, family=fam, dist=min(dists),
                    meta=dict(n_facts=n_facts, L=L, dists=dists, gap=target_gap))

    def lr_hop(self, rng):
        h = rng.randint(5, 6) if self.ood else rng.randint(1, 4) if self.split == "train" else rng.randint(2, 4)
        L = h + rng.randint(1, 2)                           # chain longer than the query: the answer is never the end
        nodes = rng.sample(self.names, L + 1)
        start = rng.randint(0, L - h - 1) if L - h - 1 > 0 else 0
        facts = [(nodes[i], nodes[i + 1]) for i in range(L)]
        rng.shuffle(facts)
        ids = self.filler(rng, rng.randint(0, 40))
        last_chain_fact = 0
        for a, b in facts:
            ids += self.e(" " + a) + self.e(" points to") + self.e(" " + b)
            last_chain_fact = len(ids) - 1                  # evidence = the latest chain fact (its last token)
            ids += self.e(".")
        target_gap = self.gap(rng)
        ids += self.filler(rng, target_gap)
        ids += self.e(" Start at") + self.e(" " + nodes[start]) + self.e(" and follow") + self.e(" " + str(h)) + \
            self.e(" links. You reach")
        s = len(ids)
        ids += self.e(" " + nodes[start + h])
        span = (s, len(ids), f"h{h}")
        ids += self.e(".")
        ids.append(EOS)
        return dict(ids=ids, spans=[span], family="lr_hop", dist=s - last_chain_fact,
                    meta=dict(depth=h, chain=L, gap=target_gap))

    def lr_bind(self, rng):
        k = rng.randint(4, 6) if self.ood else rng.randint(1, 3)
        var = rng.choice(self.names)
        others = rng.sample([n for n in self.names if n != var], 4)
        ids = self.filler(rng, rng.randint(0, 40))
        early = [(var, rng.randint(0, 20)) for _ in range(k)] + [(rng.choice(others), rng.randint(0, 20)) for _ in range(2)]
        rng.shuffle(early)
        while early[-1][0] != var and any(n == var for n, _ in early):   # make sure x's latest binding ends the block
            j = max(i for i, (n, _) in enumerate(early) if n == var)
            early.append(early.pop(j))
        last_x, xval = 0, None
        for n, v in early:
            ids += self.e(" let") + self.e(" " + n) + self.e(" =") + self.e(" " + str(v))
            if n == var:
                last_x, xval = len(ids) - 1, v              # evidence = last token of x's latest binding
            ids += self.e(" .")
        target_gap = self.gap(rng)
        sents = self.filler_sents(rng, target_gap)
        # other variables are rebound between filler sentences (they may be close to the query; x never is)
        n_rebind = rng.randint(2, 4)
        for _ in range(n_rebind):
            sents.insert(rng.randint(0, len(sents)), self.e(" let") + self.e(" " + rng.choice(others)) + self.e(" =") +
                         self.e(" " + str(rng.randint(0, 20))) + self.e(" ."))
        ids += [t for s_ in sents for t in s_]
        c = rng.randint(1, 9)
        ids += self.e(" " + var) + self.e(" +") + self.e(" " + str(c)) + self.e(" = ")
        s = len(ids)
        ids += self.e(str(xval + c))                        # digits only are scored (the space is in the prompt)
        span = (s, len(ids), "sum")
        ids += self.e(" .")
        ids.append(EOS)
        return dict(ids=ids, spans=[span], family="lr_bind", dist=s - last_x,
                    meta=dict(bindings_of_x=k, gap=target_gap))


PROBES = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"]


def probe_text_for_tokenizer(rng, family, filler_sentences, values):
    """Plain-text version of a probe (for tokenizer training only, before the tokenizer exists)."""
    names = [a + b + c for a in SYL_ID for b in SYL_ID for c in SYL_ID if len({a, b, c}) == 3]
    fill = " ".join(rng.choice(filler_sentences) for _ in range(rng.randint(3, 8)))
    if family in ("lr_recall1", "lr_carry3", "lr_multi4"):
        L = 1 if family == "lr_recall1" else 3
        n = 4 if family == "lr_multi4" else 1
        fs = [(rng.choice(names), " ".join(rng.choice(values) for _ in range(L))) for _ in range(n)]
        return "".join(f" The code of {a} is {v}." for a, v in fs) + " " + fill + \
            "".join(f" What is the code of {a}? The code of {a} is {v}." for a, v in fs)
    if family == "lr_hop":
        ns = rng.sample(names, 5)
        return "".join(f" {a} points to {b}." for a, b in zip(ns, ns[1:])) + " " + fill + \
            f" Start at {ns[0]} and follow 3 links. You reach {ns[3]}."
    x = rng.choice(names)
    return f" let {x} = {rng.randint(0, 20)} . " + fill + f" {x} + 3 = {rng.randint(3, 23)} ."
