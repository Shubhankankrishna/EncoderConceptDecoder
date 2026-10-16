"""One S5 cell: (model, method, suite, budget multiple) -> per-item records (draft §9.6, I17).

Concept methods use the S4 model (trained selector, versioned heads, register policy). Token methods use the chosen
token decoder (strongest S4 token arm, or the promoted decoder) in code-free mode. Budgets are b x t_g, with t_g the
promoted model's batched greedy wall-clock per item on the same batch (measured once per suite and seed, cached)."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import DATA, SMOKE, read_json, write_json
from runtime import DEVICE, atomic_save
import hncp, hgen, csearch, s5, baselines, train_s4
from hgen import Item, CH

BATCH = 64


def suite_items(split, name, n):
    a = np.load(DATA / f'{split}_{name}.npz'); n = min(n, len(a['x']))
    return [Item(a['x'][i], a['m'][i], int(a['prompt_len'][i])) for i in range(n)]


def concept_model(s4_dir):
    ck = torch.load(Path(s4_dir) / 'checkpoint.pt', map_location='cpu', weights_only=False)
    hs, _ = hncp.load(str(Path(s4_dir) / 'checkpoint.pt'), DEVICE); heads = csearch.Heads(); heads.load_state_dict(ck['heads'])
    pol = ck.get('policy') or dict(value_leaves=False, tau_trusted=False); heads.v_trusted = bool(pol['value_leaves']); heads.t_trusted = bool(pol['tau_trusted'])
    for p in heads.parameters():
        p.requires_grad_(False)
    return hs, heads.to(DEVICE), pol


def token_model(source, adapter=None):
    hs, _ = hncp.load(source, DEVICE)
    if adapter:
        baselines.load_lora(hs, torch.load(adapter, map_location='cpu', weights_only=False)['state'])
    for p in hs.parameters():
        p.requires_grad_(False)
    return hs


@torch.no_grad()
def fit_token_value(hs, s4_dir, out):
    """Token tree value head: 128 -> 1 on the decoder's final-layer state at the last grounded position, fitted with
    L_V on S4's executed trajectories and outcomes (every executed prefix labelled with its trajectory's outcome)."""
    eng = hgen.Engine(hs); X, Y = [], []
    for f in sorted(Path(s4_dir).glob('iter*/traj.pt')):
        for t in torch.load(f, map_location='cpu', weights_only=False)[:2048]:
            for h in range(len(t['chunks'])):
                X.append(np.concatenate([t['prompt']] + [np.array(c) for c in t['chunks'][:h]]).astype(np.int64)); Y.append(float(t['success']))
    feats = []
    ts = s5.TokenSearch(eng, dict(H=1, children='grounded', leaf='value', budget=1))
    for i in range(0, len(X), 128):
        nodes = []
        for r in X[i:i + 128]:
            n = csearch.Node(None, None, 0, 3); n.tokens = r; nodes.append(n)
        ts.ground_rows(nodes); feats += [n.u0.cpu() for n in nodes]
    heads = csearch.Heads()
    with torch.enable_grad():
        train_s4.fit_linear(heads.v, torch.stack(feats).float(), torch.tensor(Y), lambda z, y: (torch.sigmoid(z.squeeze(-1)) - y).square().mean())
    atomic_save(out, dict(heads=heads.state_dict(), n=len(Y)))
    return heads


def tg_for(hs, split, suite, n, cache):
    key = f'{split}_{suite}_{n}'; c = read_json(cache) if Path(cache).exists() else {}
    if key not in c:
        eng = hgen.Engine(hs); items = suite_items(split, suite, n); tgs, hops, oks = [], [], []
        for i in range(0, len(items), BATCH):
            ok, tg, mh = s5.greedy_batch(eng, items[i:i + BATCH], 'codes'); tgs.append(tg); hops.append(mh); oks += ok
        c[key] = dict(t_g=float(np.mean(tgs)), mean_hops=float(np.mean(hops)), greedy_success=[bool(v) for v in oks]); write_json(cache, c)
    return c[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='promoted forward checkpoint (S3 or S2)'); ap.add_argument('--s4', help='S4 expert-iteration folder')
    ap.add_argument('--token-adapter'); ap.add_argument('--method', required=True)
    ap.add_argument('--split', default='dev'); ap.add_argument('--suite', required=True); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--mult', type=float, required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--H', type=int, default=1); ap.add_argument('--R', default='1'); ap.add_argument('--tt-leaf', default='rollout'); ap.add_argument('--tt-value')
    ap.add_argument('--node-match', type=float, help='fixed expansions per step (mechanism comparison, node-matched rerun)')
    ap.add_argument('--kernel', default='gradcuit'); ap.add_argument('--lr', type=float, default=1e-2); ap.add_argument('--tg-cache', required=True)
    a = ap.parse_args(); torch.set_num_threads(4)
    hop_limit = 40 if a.suite == 'hard14' else 20; items = suite_items(a.split, a.suite, a.n)
    ref, _ = hncp.load(a.model, DEVICE); tg = tg_for(ref, a.split, a.suite, a.n, a.tg_cache); del ref
    rows, stats = [], []; t0 = time.monotonic()
    if a.method in ('token_tree', 'best_of_n'):
        hs = token_model(a.model, a.token_adapter); eng = hgen.Engine(hs); heads = None
        if a.method == 'token_tree' and a.tt_leaf == 'value':
            heads = csearch.Heads(); heads.load_state_dict(torch.load(a.tt_value, map_location='cpu', weights_only=False)['heads']); heads = heads.to(DEVICE)
            for p in heads.parameters():
                p.requires_grad_(False)
    else:
        hs, heads, pol = concept_model(a.s4); eng = hgen.Engine(hs)
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]; budget = a.mult * tg['t_g'] * len(batch); seed = 91000 + 1000 * a.seed + i
        if a.method in ('rh', 'rh_grounded'):
            value_ok = heads.v_trusted
            cfg = dict(H=a.H if a.method == 'rh' else 1, children='imagined' if (a.method == 'rh' and value_ok) else 'grounded', leaf='value' if value_ok else 'rollout',
                       noise=False, heads=heads)
            if a.node_match:
                S = csearch.Search(eng, dict(cfg, budget=max(1, int(round(a.node_match))))); ps = csearch.problems_from(batch, hop_limit, seed); tt = time.monotonic(); S.solve(ps)
                r = [dict(success=bool(p.success), reason=p.reason, verifier_calls=p.verifier_calls, nodes=p.nodes) for p in ps]; st = dict(S.stats, wall_s=time.monotonic() - tt)
            else:
                r, st = s5.run_search(eng, batch, a.method, budget, cfg, hop_limit, seed, tg['mean_hops'])
        elif a.method == 'g':
            R = None if a.R == 'inf' else int(a.R); value_ok = heads.v_trusted
            cfg = dict(H=hop_limit, children='grounded' if R == 1 else 'imagined', ground_every=R, leaf='value' if value_ok else 'rollout', noise=False, heads=heads)
            if not value_ok and R != 1:
                cfg.update(children='grounded', ground_every=1)          # rollout leaves need grounded children (I17)
            r, st = s5.run_one_tree(eng, batch, budget, cfg, hop_limit, seed)
        elif a.method == 'token_tree':
            r, st = s5.run_search(eng, batch, 'tt', budget, dict(H=1, children='grounded', leaf=a.tt_leaf, noise=False, heads=heads), hop_limit, seed, tg['mean_hops'], token=True)
        elif a.method == 'best_of_n':
            r, st = s5.run_best_of_n(eng, batch, budget, hop_limit, seed)
        elif a.method == 'answer_loop':
            value_ok = heads.v_trusted
            inner = dict(H=a.H, children='imagined' if value_ok else 'grounded', leaf='value' if value_ok else 'rollout', budget=8, heads=heads)
            r, st = s5.answer_loop(eng, batch, budget, inner, a.kernel, a.lr, .5, hop_limit, seed)
        elif a.method == 'greedy':
            ok, _, _ = s5.greedy_batch(eng, batch, 'codes'); r = [dict(success=bool(v)) for v in ok]; st = {}
        else:
            raise ValueError(a.method)
        rows += r; stats.append(dict(st, budget_s=budget, batch=len(batch)))
        if SMOKE and i >= 0:
            pass
    out = dict(method=a.method, split=a.split, suite=a.suite, n=len(items), mult=a.mult, t_g=tg['t_g'], seed=a.seed, config=vars(a),
               success=[bool(r['success']) for r in rows], rate=float(np.mean([r['success'] for r in rows])), records=rows, batches=stats,
               verifier_calls=int(sum(r.get('verifier_calls', 0) for r in rows)), nodes=int(sum(r.get('nodes', 0) for r in rows)),
               wall_s=time.monotonic() - t0, overrun_mean=float(np.mean([s.get('overrun', 0) for s in stats])))
    write_json(a.out, out); print(json.dumps({k: out[k] for k in ('method', 'suite', 'mult', 'rate', 'wall_s')}))


if __name__ == '__main__':
    main()
