"""Compare the memory repair against the archived search, with exact CPU records."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

V9 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V9 / 'code'))
import csearch, hgen, hncp, train_s4

folder = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('before_memory_repair', folder / 'code/csearch.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
torch.set_num_threads(4)
hs, _ = hncp.load(V9 / 'runs/S2_K8_dense_s911/checkpoint.pt', 'cpu')
eng = hgen.Engine(hs)
items = train_s4.fresh_problems(512, 178100)[:6]
torch.manual_seed(921)
heads = csearch.Heads()
for p in heads.parameters():
    p.requires_grad_(False)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


checks = []
for name, cfg in (
    ('grounded_rollout', dict(H=1, children='grounded', leaf='rollout', budget=32, noise=True)),
    ('imagined_value', dict(H=2, children='imagined', leaf='value', budget=8, noise=True, heads=heads)),
    ('one_tree_replay', dict(H=20, children='imagined', ground_every=2, leaf='value', budget=8, noise=False, heads=heads, one_tree=True)),
):
    rows = []
    for module in (old, csearch):
        ps = module.problems_from(items, 20, 179100)
        search = module.Search(eng, copy.copy(cfg))
        if cfg.get('one_tree'):
            search.one_tree(ps, cfg['budget'])
        else:
            search.solve(ps)
        keep = ('chunks', 'codes', 'done', 'success', 'reason', 'cur', 'hops', 'roots',
                'verifier_calls', 'nodes', 'executed_tokens', 'rollout_states', 'answer')
        rows.append(dict(stats=search.stats, problems=[dict({k: getattr(p, k) for k in keep}, rng_state=p.rng.get_state()) for p in ps]))
    assert equal(*rows), name
    if not cfg.get('one_tree'):
        ps = csearch.problems_from(items, 20, 179100)
        ps, stats = train_s4.solve_in_workers(eng, ps, copy.copy(cfg), 6, max_steps=2)
        worker = dict(stats=stats, problems=[dict({k: getattr(p, k) for k in keep}, rng_state=p.rng.get_state()) for p in ps])
        assert equal(rows[0], worker), name + ' process handoff'
    checks.append(name)
    print(json.dumps(dict(check=name, exact=True)), flush=True)
result = dict(status='passed', exact_records=True, rng_state_preserved=True, worker_handoffs_exact=True, cases=checks)
(folder / 'parity.json').write_text(json.dumps(result, indent=2))
