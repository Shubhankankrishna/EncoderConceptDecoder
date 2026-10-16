"""Deterministic task generators for the HRM-CC vs Nanbeige-loop experiment (generator version v1).

Every generator: gen(rng, split) -> Example(segments=[(text, scored, label), ...], family, meta).
Documents are the concatenation of segment texts; scored segments are the evaluated answer spans
(teacher-forced greedy exact match == all scored tokens argmax-correct). Training uses dense NTP on
every token of every document (both models identically).

Splits: 'train', 'val', 'test' share the in-distribution (ID) structural ranges and differ only by seed.
'ood' uses held-out structure: longer chains / deeper hops, larger graphs, unseen identifier alphabet,
unseen operator combinations, unseen permutation sizes, longer distances.
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field

GENERATOR_VERSION = "v1"

# ---------------------------------------------------------------- identifier alphabets
SYL_A = ["ka", "lo", "mi", "ra", "te", "su", "no", "vi", "pe", "da", "ro", "li", "ma", "tu", "se", "ne"]
SYL_B = ["zu", "xe", "qo", "wy", "jo", "fa", "gi", "bu", "hy", "ce", "ox", "ek"]  # OOD identifier alphabet
NAMES_A = [a + b for a in SYL_A for b in SYL_A if a != b]          # 240 names
NAMES_B = [a + b for a in SYL_B for b in SYL_B if a != b]          # 132 names (never in training)
FIRST_A = ["Anna", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jade", "Kai", "Lena", "Milo",
           "Nora", "Omar", "Pia", "Quin", "Rosa", "Sam", "Tara", "Uma", "Vic", "Wren", "Yara"]
FIRST_B = ["Zeno", "Xia", "Yusuf", "Wanda", "Otto", "Ines", "Hugo", "Greta", "Felix", "Edda", "Dora", "Cyril"]
STORES = ["Nova", "Orbit", "Maple", "Harbor", "Summit", "Delta", "Pine", "Crest"]
ITEMS = ["apple", "pear", "bread", "milk", "rice", "tea", "soap", "pen", "lamp", "cup", "salt", "egg"]
CITIES = ["Oslo", "Lima", "Pune", "Kyiv", "Doha", "Rome", "Nice", "Bern"]


@dataclass
class Example:
    segments: list  # list of (text, scored: bool, label)
    family: str
    meta: dict = field(default_factory=dict)

    @property
    def text(self):
        return "".join(s[0] for s in self.segments)


def names_for(split):
    return NAMES_B if split == "ood_names" else NAMES_A


# ---------------------------------------------------------------- B. symbolic / algorithmic reasoning
def gen_pointer_chase(rng, split):
    """Follow a chain of pointers h hops. ID: 8-16 pointers, hops 1-4. OOD: 20-28 pointers, hops 5-8."""
    ood = split == "ood"
    n = rng.randint(20, 28) if ood else rng.randint(8, 16)
    hops = rng.randint(5, 8) if ood else rng.randint(1, 4)
    nodes = rng.sample(NAMES_A, n)
    succ = nodes[1:] + nodes[:1]
    rng.shuffle(succ)
    ptr = dict(zip(nodes, succ))
    facts = [f"{a} -> {ptr[a]}" for a in nodes]
    rng.shuffle(facts)
    start = rng.choice(nodes)
    path, cur = [], start
    for _ in range(hops):
        cur = ptr[cur]
        path.append(cur)
    prompt = "pointers: " + " ; ".join(facts) + f" . start {start} , follow {hops} . path :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(path), True, "answer"), (" .", False, "end")],
                   "pointer_chase", {"depth": hops, "size": n})


def gen_graph_walk(rng, split):
    """Labeled directed graph; follow a sequence of edge labels. ID nodes<=8, walk<=4; OOD nodes 10-16, walk 5-7."""
    ood = split == "ood"
    n = rng.randint(10, 16) if ood else rng.randint(4, 8)
    L = rng.randint(5, 7) if ood else rng.randint(1, 4)
    labels = ["red", "blue", "green"]
    nodes = [f"n{i}" for i in rng.sample(range(40), n)]
    edges = {(u, c): rng.choice(nodes) for u in nodes for c in labels}
    lines = [f"{u} {c} {v}" for (u, c), v in edges.items()]
    rng.shuffle(lines)
    cur = rng.choice(nodes)
    walk = [rng.choice(labels) for _ in range(L)]
    start, visited = cur, []
    for c in walk:
        cur = edges[(cur, c)]
        visited.append(cur)
    prompt = "graph: " + " ; ".join(lines) + f" . from {start} go " + " ".join(walk) + " . visit :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(visited), True, "answer"), (" .", False, "end")],
                   "graph_walk", {"depth": L, "size": n})


def gen_state_machine(rng, split):
    """Finite-state program over input symbols. ID states<=5, input len 2-6; OOD states 7-9, input 8-12."""
    ood = split == "ood"
    ns = rng.randint(7, 9) if ood else rng.randint(3, 5)
    Lin = rng.randint(8, 12) if ood else rng.randint(2, 6)
    states = [f"s{i}" for i in range(ns)]
    syms = ["a", "b"]
    delta = {(s, c): rng.choice(states) for s in states for c in syms}
    rules = [f"{s} {c} {t}" for (s, c), t in delta.items()]
    rng.shuffle(rules)
    inp = [rng.choice(syms) for _ in range(Lin)]
    cur, trace = "s0", []
    for c in inp:
        cur = delta[(cur, c)]
        trace.append(cur)
    prompt = "fsm: " + " ; ".join(rules) + " . start s0 input " + " ".join(inp) + " . states :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(trace), True, "answer"), (" .", False, "end")],
                   "state_machine", {"depth": Lin, "size": ns})


def gen_permutation(rng, split):
    """Apply a permutation r times to a list. ID size 3-5, r 1-3; OOD size 6-7 (unseen family), r 1-3."""
    ood = split == "ood"
    n = rng.randint(6, 7) if ood else rng.randint(3, 5)
    r = rng.randint(1, 3)
    perm = list(range(n))
    rng.shuffle(perm)
    items = rng.sample(ITEMS, n)
    seq = items[:]
    for _ in range(r):
        seq = [seq[p] for p in perm]
    prompt = f"perm {' '.join(map(str, perm))} applied {r} times to {' '.join(items)} gives :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(seq), True, "answer"), (" .", False, "end")],
                   "permutation", {"depth": r, "size": n})


def gen_variable_binding(rng, split):
    """Chains of assignments with copies and +/- constants. ID chain 1-4; OOD chain 5-8 or unseen names."""
    ood = split == "ood"
    depth = rng.randint(5, 8) if ood else rng.randint(1, 4)
    names = NAMES_A
    vs = rng.sample(names, depth + 3)
    val = rng.randint(0, 20)
    lines = [f"{vs[0]} = {val}"]
    cur = val
    for i in range(1, depth + 1):
        if rng.random() < 0.5:
            lines.append(f"{vs[i]} = {vs[i - 1]}")
        else:
            c = rng.randint(1, 9)
            op = rng.choice(["+", "-"])
            cur = cur + c if op == "+" else cur - c
            lines.append(f"{vs[i]} = {vs[i - 1]} {op} {c}")
    # distractors
    for dv in vs[depth + 1:]:
        lines.insert(rng.randint(0, len(lines)), f"{dv} = {rng.randint(0, 20)}")
    prompt = "program: " + " ; ".join(lines) + f" . print {vs[depth]} :"
    return Example([(prompt, False, "prompt"), (f" {cur}", True, "answer"), (" .", False, "end")],
                   "variable_binding", {"depth": depth, "size": len(lines)})


def gen_variable_binding_newnames(rng, split):
    """OOD-only: same as variable_binding (ID depth range) but with the unseen identifier alphabet B."""
    depth = rng.randint(1, 4)
    vs = rng.sample(NAMES_B, depth + 3)
    val = rng.randint(0, 20)
    lines = [f"{vs[0]} = {val}"]
    cur = val
    for i in range(1, depth + 1):
        if rng.random() < 0.5:
            lines.append(f"{vs[i]} = {vs[i - 1]}")
        else:
            c = rng.randint(1, 9)
            op = rng.choice(["+", "-"])
            cur = cur + c if op == "+" else cur - c
            lines.append(f"{vs[i]} = {vs[i - 1]} {op} {c}")
    for dv in vs[depth + 1:]:
        lines.insert(rng.randint(0, len(lines)), f"{dv} = {rng.randint(0, 20)}")
    prompt = "program: " + " ; ".join(lines) + f" . print {vs[depth]} :"
    return Example([(prompt, False, "prompt"), (f" {cur}", True, "answer"), (" .", False, "end")],
                   "variable_binding_newnames", {"depth": depth, "size": len(lines)})


def gen_kv_updates(rng, split):
    """set/get over a key-value store with overwrites. ID ops 4-12; OOD ops 20-30."""
    ood = split == "ood"
    nops = rng.randint(20, 30) if ood else rng.randint(4, 12)
    keys = rng.sample(NAMES_A, rng.randint(2, 5))
    store, ops = {}, []
    for _ in range(nops):
        k = rng.choice(keys)
        v = rng.randint(0, 99)
        store[k] = v
        ops.append(f"set {k} {v}")
    q = rng.choice(list(store))
    prompt = "store: " + " ; ".join(ops) + f" . get {q} :"
    return Example([(prompt, False, "prompt"), (f" {store[q]}", True, "answer"), (" .", False, "end")],
                   "kv_updates", {"depth": sum(1 for o in ops if o.split()[1] == q), "size": nops})


def gen_relational(rng, split):
    """Family-tree composition: 'X is the parent of Y'. Query the h-th descendant. ID h 1-3; OOD h 4-6."""
    ood = split == "ood"
    h = rng.randint(4, 6) if ood else rng.randint(1, 3)
    people = rng.sample(FIRST_A, min(len(FIRST_A), h + 1 + rng.randint(2, 5)))
    chain = people[: h + 1]
    facts = [f"{chain[i]} is the parent of {chain[i + 1]}" for i in range(h)]
    others = people[h + 1:]
    for o in others:
        facts.append(f"{o} is the parent of {rng.choice(FIRST_A)}")
    rng.shuffle(facts)
    prompt = "facts: " + " . ".join(facts) + f" . who is {h} generations below {chain[0]} ?"
    return Example([(prompt, False, "prompt"), (" " + chain[h], True, "answer"), (" .", False, "end")],
                   "relational", {"depth": h, "size": len(facts)})


OPS_ID = ["add", "sub", "mul"]


def gen_arith_transform(rng, split):
    """start value then a sequence of ops mod 100. ID: 1-4 ops, 'mul' never directly after 'sub'.
    OOD: 3-6 ops, always containing the held-out pair 'sub' then 'mul' (unseen operator combination).
    Eval buckets OOD by meta.depth: 3-4 ops isolates the unseen combination; 5-6 also exceeds ID depth."""
    ood = split == "ood"
    n = rng.randint(3, 6) if ood else rng.randint(1, 4)
    def has_pair(o):
        return any(a == "sub" and b == "mul" for a, b in zip(o, o[1:]))

    ops = [rng.choice(OPS_ID) for _ in range(n)]
    if ood:
        if not has_pair(ops):
            i = rng.randrange(n - 1)
            ops[i], ops[i + 1] = "sub", "mul"
    else:
        while has_pair(ops):
            ops = [rng.choice(OPS_ID) for _ in range(n)]
    v = rng.randint(0, 99)
    steps, cur = [], v
    for o in ops:
        c = rng.randint(2, 9)
        cur = (cur + c) % 100 if o == "add" else (cur - c) % 100 if o == "sub" else (cur * c) % 100
        steps.append((o, c, cur))
    prompt = f"calc: start {v} ; " + " ; ".join(f"{o} {c}" for o, c, _ in steps) + " . values :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(str(c) for _, _, c in steps), True, "answer"),
                    (" .", False, "end")], "arith_transform", {"depth": n, "size": n})


def gen_stack_ops(rng, split):
    """push/pop sequence; answer = popped values in order then final stack. ID ops 3-10; OOD ops 14-20."""
    ood = split == "ood"
    n = rng.randint(14, 20) if ood else rng.randint(3, 10)
    st, ops, popped = [], [], []
    for _ in range(n):
        if st and rng.random() < 0.4:
            popped.append(st.pop())
            ops.append("pop")
        else:
            v = rng.choice(ITEMS)
            st.append(v)
            ops.append(f"push {v}")
    ans = " ".join(popped) + " | " + " ".join(st)
    prompt = "stack: " + " ; ".join(ops) + " . popped | final :"
    return Example([(prompt, False, "prompt"), (" " + ans, True, "answer"), (" .", False, "end")],
                   "stack_ops", {"depth": n, "size": n})


def gen_nested_scopes(rng, split):
    """Nested blocks with shadowing; each print shows the innermost binding. ID nesting 1-3; OOD 4-6."""
    ood = split == "ood"
    depth = rng.randint(4, 6) if ood else rng.randint(1, 3)
    var = rng.choice(NAMES_A)
    toks, outs, stack = [], [], []
    v0 = rng.randint(0, 9)
    toks.append(f"{var} = {v0}")
    stack.append(v0)
    for _ in range(depth):
        v = rng.randint(0, 9)
        toks.append("{")
        if rng.random() < 0.8:
            toks.append(f"{var} = {v}")
            stack.append(v)
        else:
            stack.append(stack[-1])
        toks.append(f"print {var}")
        outs.append(stack[-1])
    for _ in range(depth):
        toks.append("}")
        stack.pop()
        toks.append(f"print {var}")
        outs.append(stack[-1])
    prompt = "scopes: " + " ".join(toks) + " . output :"
    return Example([(prompt, False, "prompt"), (" " + " ".join(map(str, outs)), True, "answer"), (" .", False, "end")],
                   "nested_scopes", {"depth": depth, "size": len(toks)})


def gen_brackets(rng, split):
    """Complete an unbalanced bracket prefix. ID open depth 1-6; OOD depth 8-12."""
    ood = split == "ood"
    depth = rng.randint(8, 12) if ood else rng.randint(1, 6)
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    seq, st = [], []
    while len(st) < depth:
        if st and rng.random() < 0.3:
            seq.append(pairs[st.pop()])
        else:
            o = rng.choice(list(pairs))
            seq.append(o)
            st.append(o)
    close = [pairs[o] for o in reversed(st)]
    prompt = "close: " + " ".join(seq) + " ->"
    return Example([(prompt, False, "prompt"), (" " + " ".join(close), True, "answer"), (" .", False, "end")],
                   "brackets", {"depth": depth, "size": len(seq)})


def gen_seq_transform(rng, split):
    """reverse / sort / dedupe a list of numbers. ID length 3-8; OOD length 12-16."""
    ood = split == "ood"
    n = rng.randint(12, 16) if ood else rng.randint(3, 8)
    xs = [rng.randint(0, 9) for _ in range(n)]
    op = rng.choice(["reverse", "sort", "dedupe"])
    if op == "reverse":
        ys = xs[::-1]
    elif op == "sort":
        ys = sorted(xs)
    else:
        seen, ys = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                ys.append(x)
    prompt = f"{op}: " + " ".join(map(str, xs)) + " ->"
    return Example([(prompt, False, "prompt"), (" " + " ".join(map(str, ys)), True, "answer"), (" .", False, "end")],
                   "seq_transform", {"depth": n, "size": n, "op": op})


# ---------------------------------------------------------------- D. exact detail / copy / retrieval
def rand_ident(rng, kind):
    if kind == "uuid":
        h = "0123456789abcdef"
        return "".join(rng.choice(h) for _ in range(8)) + "-" + "".join(rng.choice(h) for _ in range(4))
    if kind == "code":
        return rng.choice(["get", "set", "make", "load", "run"]) + "_" + rng.choice(NAMES_A) + str(rng.randint(0, 99))
    if kind == "number":
        return str(rng.randint(1000, 999999))
    if kind == "name":
        return rng.choice(FIRST_A) + " " + rng.choice(NAMES_A).capitalize()
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def gen_kv_retrieval(rng, split, filler_bank=None, max_pairs=None):
    """Many key: value pairs, then query one key. Distance (in pairs) between the queried pair and the
    query is recorded. ID pairs 4-24; OOD (longer context) pairs 40-60."""
    ood = split == "ood"
    n = rng.randint(40, 60) if ood else rng.randint(4, max_pairs or 24)
    kind = rng.choice(["uuid", "code", "number", "word"])
    keys = rng.sample(NAMES_A, n)
    vals = [rand_ident(rng, kind) for _ in range(n)]
    qi = rng.randrange(n)
    body = " ; ".join(f"{k} : {v}" for k, v in zip(keys, vals))
    prompt = "records: " + body + f" . value of {keys[qi]} ?"
    return Example([(prompt, False, "prompt"), (" " + vals[qi], True, "answer"), (" .", False, "end")],
                   "kv_retrieval", {"pairs_after": n - 1 - qi, "size": n, "kind": kind})


def gen_copy(rng, split, filler_bank):
    """Copy a random string after a filler gap. The gap (filler sentences) controls distance."""
    ood = split == "ood"
    kind = rng.choice(["uuid", "code", "number", "name", "word"])
    s = " ".join(rand_ident(rng, kind) for _ in range(rng.randint(1, 3)))
    nfill = rng.randint(10, 18) if ood else rng.choice([0, 1, 2, 3, 5, 8])
    filler = " ".join(rng.choice(filler_bank) for _ in range(nfill))
    prompt = f"remember: {s} . {filler} repeat the remembered text :"
    return Example([(prompt, False, "prompt"), (" " + s, True, "answer"), (" .", False, "end")],
                   "copy", {"gap_sentences": nfill, "kind": kind})


def gen_needle(rng, split, filler_bank):
    """A code inserted into filler prose; asked at the end. Needle depth (sentences after needle) varies."""
    ood = split == "ood"
    code = rand_ident(rng, rng.choice(["uuid", "word", "number"]))
    nbefore = rng.randint(0, 6)
    nafter = rng.randint(14, 24) if ood else rng.randint(0, 12)
    sents = [rng.choice(filler_bank) for _ in range(nbefore)] + [f"The secret code is {code} ."] + \
            [rng.choice(filler_bank) for _ in range(nafter)]
    prompt = " ".join(sents) + " What is the secret code ?"
    return Example([(prompt, False, "prompt"), (" " + code, True, "answer"), (" .", False, "end")],
                   "needle", {"sentences_after": nafter})


def gen_json_lookup(rng, split):
    """Nested JSON object; query a key path. ID depth 1-2; OOD depth 3-4."""
    ood = split == "ood"
    depth = rng.randint(3, 4) if ood else rng.randint(1, 2)

    def build(d):
        keys = rng.sample(NAMES_A, rng.randint(2, 3))
        return {k: (build(d - 1) if d > 1 and i == 0 else rng.randint(0, 999)) for i, k in enumerate(keys)}

    obj = build(depth)
    path, cur = [], obj
    for _ in range(depth - 1):
        k = next(iter(cur))
        path.append(k)
        cur = cur[k]
    k = rng.choice([kk for kk in cur if not isinstance(cur[kk], dict)])
    path.append(k)
    cur = cur[k]
    prompt = "json: " + json.dumps(obj) + " . get " + ".".join(path) + " :"
    return Example([(prompt, False, "prompt"), (f" {cur}", True, "answer"), (" .", False, "end")],
                   "json_lookup", {"depth": len(path)})


# ---------------------------------------------------------------- E. code / structured text
def gen_py_exec(rng, split):
    """Tiny Python function + call; answer = printed value. ID 1-3 statements; OOD 4-6 statements."""
    ood = split == "ood"
    n = rng.randint(4, 6) if ood else rng.randint(1, 3)
    fn = rng.choice(["f", "g", "h", "calc", "step"])
    arg = rng.choice(["x", "n", "v"])
    lines = [f"def {fn}({arg}):"]
    env = {arg: rng.randint(0, 9)}
    x0 = env[arg]
    prev = arg
    for i in range(n):
        var = rng.choice(["a", "b", "c", "t", "y"]) + str(i)
        c = rng.randint(1, 5)
        op = rng.choice(["+", "-", "*"])
        env[var] = eval(f"{env[prev]} {op} {c}")
        lines.append(f"    {var} = {prev} {op} {c}")
        prev = var
    lines.append(f"    return {prev}")
    prompt = "\n".join(lines) + f"\nprint({fn}({x0}))\n# output:"
    return Example([(prompt, False, "prompt"), (f" {env[prev]}", True, "answer"), ("\n", False, "end")],
                   "py_exec", {"depth": n})


def gen_json_build(rng, split):
    """Emit a JSON object from a key=value description (syntactic exactness)."""
    n = rng.randint(2, 5)
    keys = rng.sample(NAMES_A, n)
    vals = [rng.choice([rng.randint(0, 99), rng.choice(ITEMS), rng.choice([True, False])]) for _ in range(n)]
    desc = " , ".join(f"{k}={json.dumps(v)}" for k, v in zip(keys, vals))
    out = json.dumps(dict(zip(keys, vals)))
    prompt = f"to json: {desc} ->"
    return Example([(prompt, False, "prompt"), (" " + out, True, "answer"), ("\n", False, "end")],
                   "json_build", {"size": n})


def gen_yaml_like(rng, split):
    """YAML-like config; query a nested field (identifier copy through structure)."""
    sec = rng.sample(NAMES_A, 3)
    lines, table = [], {}
    for s in sec:
        lines.append(f"{s}:")
        for k in rng.sample(["host", "port", "user", "mode", "path", "level"], 3):
            v = rand_ident(rng, rng.choice(["code", "number", "word"]))
            lines.append(f"  {k}: {v}")
            table[(s, k)] = v
    (s, k), v = rng.choice(list(table.items()))
    prompt = "\n".join(lines) + f"\n# {s}.{k} ="
    return Example([(prompt, False, "prompt"), (" " + v, True, "answer"), ("\n", False, "end")],
                   "yaml_lookup", {})


def gen_sort_trace(rng, split):
    """Algorithm trace: bubble-sort passes over a small list (each pass is an answer step)."""
    n = rng.randint(3, 5)
    xs = [rng.randint(0, 9) for _ in range(n)]
    passes, a = [], xs[:]
    for _ in range(n - 1):
        for j in range(n - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
        passes.append(" ".join(map(str, a)))
    prompt = "bubble sort trace of " + " ".join(map(str, xs)) + " :"
    return Example([(prompt, False, "prompt"), (" " + " | ".join(passes), True, "answer"), ("\n", False, "end")],
                   "sort_trace", {"depth": n - 1})


def gen_expr_eval(rng, split):
    """Evaluate a nested arithmetic expression (mod 100). ID nesting 1-2; OOD 3-4."""
    ood = split == "ood"
    depth = rng.randint(3, 4) if ood else rng.randint(1, 2)

    def build(d):
        if d == 0:
            return str(rng.randint(0, 9))
        return f"( {build(d - 1)} {rng.choice(['+', '-', '*'])} {build(rng.randint(0, d - 1))} )"

    e = build(depth)
    return Example([("expr: " + e + " =", False, "prompt"), (f" {eval(e) % 100}", True, "answer"), ("\n", False, "end")],
                   "expr_eval", {"depth": depth})


# ---------------------------------------------------------------- F. tool trajectories
def gen_tool_traj(rng, split):
    """Deterministic simulated tools. GOAL -> CALL -> OBS -> ... -> ANSWER. Every CALL and the ANSWER are
    scored segments (next-action accuracy, final-state accuracy). With p=0.2 an injected error observation
    appears after the first call; the correct next action is to retry the same call (recovery)."""
    ood = split == "ood"
    prices = {(s, i): rng.randint(1, 20) for s in STORES for i in ITEMS}
    rate = {c: rng.randint(2, 5) for c in ["EUR", "INR", "JPY"]}
    temps = {c: rng.randint(-5, 35) for c in CITIES}
    kind = "cost_convert" if ood else rng.choice(["cost", "compare", "convert", "weather"])  # OOD = unseen tool composition
    segs = []
    inject = rng.random() < 0.2

    def call(txt, obs, lbl):
        segs.append((f"\nCALL {txt}", True, lbl))
        if inject and lbl == "action1":
            segs.append(("\nOBS error: timeout", False, "obs_err"))
            segs.append((f"\nCALL {txt}", True, "recovery"))
        segs.append((f"\nOBS {obs}", False, "obs"))

    if kind == "cost":
        s, it, q = rng.choice(STORES), rng.choice(ITEMS), rng.randint(2, 6)
        segs.append((f"GOAL: cost of {q} {it} at {s}", False, "goal"))
        p = prices[(s, it)]
        call(f"price(store={s}, item={it})", p, "action1")
        call(f"calc({q}*{p})", q * p, "action2")
        ans = q * p
    elif kind == "compare":
        it = rng.choice(ITEMS)
        s1, s2 = rng.sample(STORES, 2)
        segs.append((f"GOAL: cheaper store for {it}: {s1} or {s2}", False, "goal"))
        p1, p2 = prices[(s1, it)], prices[(s2, it)]
        call(f"price(store={s1}, item={it})", p1, "action1")
        call(f"price(store={s2}, item={it})", p2, "action2")
        ans = s1 if p1 <= p2 else s2
    elif kind == "convert":
        a, c = rng.randint(2, 30), rng.choice(list(rate))
        segs.append((f"GOAL: convert {a} USD to {c}", False, "goal"))
        call(f"rate(USD, {c})", rate[c], "action1")
        call(f"calc({a}*{rate[c]})", a * rate[c], "action2")
        ans = a * rate[c]
    elif kind == "weather":
        c1, c2 = rng.sample(CITIES, 2)
        segs.append((f"GOAL: warmer city: {c1} or {c2}", False, "goal"))
        call(f"weather(city={c1})", temps[c1], "action1")
        call(f"weather(city={c2})", temps[c2], "action2")
        ans = c1 if temps[c1] >= temps[c2] else c2
    else:  # OOD composition: cost then currency conversion (unseen tool combination)
        s, it, q, c = rng.choice(STORES), rng.choice(ITEMS), rng.randint(2, 6), rng.choice(list(rate))
        segs.append((f"GOAL: cost of {q} {it} at {s} in {c}", False, "goal"))
        p = prices[(s, it)]
        call(f"price(store={s}, item={it})", p, "action1")
        call(f"rate(USD, {c})", rate[c], "action2")
        call(f"calc({q}*{p}*{rate[c]})", q * p * rate[c], "action3")
        ans = q * p * rate[c]
    segs.append(("\nANSWER", False, "ans_prefix"))
    segs.append((f" {ans}", True, "final"))
    segs.append(("\n", False, "end"))
    return Example(segs, "tool_traj", {"kind": kind, "injected_error": inject})


SYMBOLIC = {"pointer_chase": gen_pointer_chase, "graph_walk": gen_graph_walk, "state_machine": gen_state_machine,
            "permutation": gen_permutation, "variable_binding": gen_variable_binding, "kv_updates": gen_kv_updates,
            "relational": gen_relational, "arith_transform": gen_arith_transform, "stack_ops": gen_stack_ops,
            "nested_scopes": gen_nested_scopes, "brackets": gen_brackets, "seq_transform": gen_seq_transform}
RETRIEVAL = {"kv_retrieval": gen_kv_retrieval, "copy": gen_copy, "needle": gen_needle, "json_lookup": gen_json_lookup}
CODE = {"py_exec": gen_py_exec, "json_build": gen_json_build, "yaml_lookup": gen_yaml_like,
        "sort_trace": gen_sort_trace, "expr_eval": gen_expr_eval}
TOOL = {"tool_traj": gen_tool_traj}
NEEDS_FILLER = {"copy", "needle"}
