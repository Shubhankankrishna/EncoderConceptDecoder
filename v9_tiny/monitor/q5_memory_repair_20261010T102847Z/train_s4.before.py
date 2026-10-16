"""S4 expert iteration over codes (draft §8.2): receding-horizon search -> root visit distributions and verified
outcomes -> selector pi, value V_psi, terminal tau_psi. Everything else frozen. Heads versioned per iteration and
validated against the register's G4 rows before use (fallbacks per I15/I16).
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, SMOKE, digest, read_json, write_json
from runtime import DEVICE, atomic_save
from data import gen_symbolic
import hncp, hgen, csearch
from hgen import Item, CH, stop_legal

SUITE_SPEC = dict(standard=(10, False), hard10=(10, True))


def fresh_problems(n, seed):
    """Fresh graph problems from the frozen generator, half standard, half hard10; evaluation hashes excluded."""
    excluded = set((DATA / 'evaluation_symbolic_hashes.txt').read_text().split()); rng = np.random.RandomState(seed); items = []; seen = set()
    while len(items) < n:
        name = 'standard' if len(items) % 2 == 0 else 'hard10'; nodes, hard = SUITE_SPEC[name]
        x, m, d, meta = gen_symbolic(rng, nodes, 'graph', hard, 4, 'shuffle', 192, 'random')
        if meta['canonical'] in excluded or meta['canonical'] in seen:
            continue
        seen.add(meta['canonical']); items.append(Item(x, m, meta['prompt_len']))
    return items


# ---------------------------------------------------------------- metrics (register G4)
def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool); npos, nneg = int(y.sum()), int((~y).sum())
    if npos < 30 or nneg < 30:
        return None, npos, nneg
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks for ties
        idx = s == v; ranks[idx] = ranks[idx].mean()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)), npos, nneg


def calib(scores, labels):
    p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6); y = np.asarray(labels, float)
    brier = float(np.mean((p - y) ** 2)); base = float(np.mean((y.mean() - y) ** 2)) if len(y) else 0.
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None
    bins = np.minimum((p * 10).astype(int), 9); rel = [dict(bin=b, n=int((bins == b).sum()), mean_p=float(p[bins == b].mean()), rate=float(y[bins == b].mean())) for b in range(10) if (bins == b).any()]
    return dict(brier=brier, base_rate_brier=base, logloss=ll, reliability=rel, n=len(y))


def v_row(scores, labels, pass_auc, flag_auc=None):
    a, npos, nneg = auroc(scores, labels); c = calib(scores, labels)
    if a is None:
        zone = 'inconclusive'; note = f'undefined ({npos}, {nneg})'
    elif a >= pass_auc and c['brier'] < c['base_rate_brier']:
        zone = 'pass'; note = None
    elif flag_auc is not None and a >= flag_auc and c['brier'] < c['base_rate_brier']:
        zone = 'partial'; note = 'flagged'
    else:
        zone = 'fail'; note = None
    return dict(auroc=a, n_pos=npos, n_neg=nneg, zone=zone, note=note, **c)


# ---------------------------------------------------------------- labels from searched trajectories
def records(problems):
    roots, rollouts = [], []
    for p in problems:
        o = float(p.success)
        for r in p.roots:
            roots.append(dict(r, o=o, problem=id(p)))
        rollouts += [dict(u0=u, stop_label=l) for u, l in p.rollout_states]
    return roots, rollouts


@torch.no_grad()
def complete_states(engine, problems):
    """u0 at the complete decoded answer of each problem (G4-V complete)."""
    rows = [p.tokens() for p in problems]; out = []
    for i in range(0, len(rows), 64):
        x, d, L = engine.batch(rows[i:i + 64]); st = engine.state_heads(x, d, L, need_codes=False); out += [u.cpu() for u in st['u0']]
    return out


@torch.no_grad()
def imagined_pairs(engine, problems, depth):
    """Depth-d imagined states along each executed code sequence from each executed root t, paired with the executed
    state t+d (G2.3 pairing): returns (u0_imagined, o, stop_label_of_executed_counterpart)."""
    hs = engine.hs; out = []
    for p in problems:
        P = p.P; chunks = p.chunks
        for t in range(len(p.codes)):
            if t + depth > len(p.codes) or t + depth >= len(p.roots):
                break
            toks = np.concatenate([p.item.prompt] + [np.array(c) for c in chunks[:t]]).astype(np.int64)
            x, d, L = engine.batch([toks]); st = engine.state_heads(x, d, L)
            J = st['J']; c = st['c'][:, :int(J[0]) + 1]; cd = st['cd'][:, :int(J[0]) + 1]; preds = [st['chatk'][0, p.codes[t]]]
            for s in range(1, depth):
                H, _ = engine.imagine(c, cd, J, torch.stack(preds)[None]); idx = J + s
                preds.append(hgen.Engine.at(H['chatk'], idx)[0, p.codes[t + s]])
            H, _ = engine.imagine(c, cd, J, torch.stack(preds)[None])
            out.append((hgen.Engine.at(H['u0'], J + depth)[0].cpu(), float(p.success), p.roots[t + depth]['stop_label']))
    return out


# ---------------------------------------------------------------- fitting
def fit_linear(module, X, target, loss_fn, epochs=300, lr=1e-2, wd=1e-4):
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); loss = loss_fn(module(X), target); loss.backward(); opt.step()
    return float(loss)


def fit_iteration(hs, heads, roots, rollouts):
    """L_pi on root visit distributions (temperature 1), L_V on (root u0, o), L_tau on grounded states."""
    dev = hs.codes.selector.weight.device          # the selector is fitted where it currently lives (main() moves it to CPU)
    with_codes = [r for r in roots if r['visits'].sum() > 0]
    X = torch.stack([r['u0'] for r in with_codes]).to(dev).float(); Y = torch.tensor(np.stack([r['visits'] / r['visits'].sum() for r in with_codes]), dtype=torch.float32, device=dev)
    sel = hs.codes.selector
    for q in sel.parameters():
        q.requires_grad_(True)
    lp = fit_linear(sel, X, Y, lambda z, y: -(y * z.float().log_softmax(-1)).sum(-1).mean())
    for q in sel.parameters():
        q.requires_grad_(False)
    Xv = torch.stack([r['u0'] for r in roots]).float(); yv = torch.tensor([r['o'] for r in roots], dtype=torch.float32)
    lv = fit_linear(heads.v, Xv, yv, lambda z, y: (torch.sigmoid(z.squeeze(-1)) - y).square().mean())
    gs = [(r['u0'], r['stop_label']) for r in roots] + [(r['u0'], r['stop_label']) for r in rollouts]
    Xt = torch.stack([u for u, _ in gs]).float(); yt = torch.tensor([float(l) for _, l in gs])
    lt = fit_linear(heads.t, Xt, yt, lambda z, y: F.binary_cross_entropy_with_logits(z.squeeze(-1), y))
    return dict(selector_ce=lp, value_mse=lv, terminal_bce=lt, n_roots=len(roots), n_selector=len(with_codes), n_terminal=len(gs), stop_positive=int(yt.sum()))


@torch.no_grad()
def validate(engine, heads, problems, H):
    """G4 rows on held-out problems (heads on CPU; states recomputed on the engine device where needed)."""
    roots, rollouts = records(problems)
    comp = complete_states(engine, problems); o = [float(p.success) for p in problems]
    rows = dict(complete=v_row(heads.value(torch.stack(comp)).numpy(), o, .80, .70))
    rows['intermediate'] = v_row(heads.value(torch.stack([r['u0'] for r in roots])).numpy(), [r['o'] for r in roots], .70)
    tau_ok = True
    for d in range(1, H + 1):
        pairs = imagined_pairs(engine, problems, d)
        if pairs:
            u = torch.stack([a for a, _, _ in pairs]); rows[f'imagined_d{d}'] = v_row(heads.value(u).numpy(), [b for _, b, _ in pairs], .70)
            tp = heads.term(u).numpy(); illegal = np.array([not c for _, _, c in pairs])
            rows[f'tau_imagined_fp_d{d}'] = float((tp[illegal] >= .5).mean()) if illegal.any() else None
            tau_ok &= rows[f'tau_imagined_fp_d{d}'] is not None and rows[f'tau_imagined_fp_d{d}'] <= .02
        else:
            rows[f'imagined_d{d}'] = dict(zone='inconclusive', note='no executed pairs'); tau_ok = False
    gs = [(r['u0'], r['stop_label']) for r in roots] + [(r['u0'], r['stop_label']) for r in rollouts]
    legal = [u for u, l in gs if l]
    rows['tau_recall'] = float((heads.term(torch.stack(legal)).numpy() >= .5).mean()) if legal else None
    rows['tau'] = dict(zone='pass' if tau_ok and rows['tau_recall'] is not None and rows['tau_recall'] >= .90 else 'fail', n_legal=len(legal))
    return rows


def leaf_policy(rows, H):
    """Register consequences: V used (value leaves, decoder-free children) only if complete passes (or flagged),
    intermediate passes and every imagined depth <= H passes; otherwise the iteration falls back to verified grounded
    completion at iteration-0 scale (I16). tau trusted at imagined nodes only if G4-tau passes."""
    ok = rows['complete']['zone'] in ('pass', 'partial') and rows['intermediate']['zone'] == 'pass' and all(rows.get(f'imagined_d{d}', {}).get('zone') == 'pass' for d in range(1, H + 1))
    return dict(value_leaves=bool(ok), tau_trusted=rows['tau']['zone'] == 'pass', flagged=rows['complete']['zone'] == 'partial')


def solve_batched(engine, items, cfg, hop_limit, seed, frontier=64):
    problems = csearch.problems_from(items, hop_limit, seed); s = csearch.Search(engine, cfg); t0 = time.monotonic()
    for i in range(0, len(problems), frontier):
        s.solve(problems[i:i + frontier])
    return problems, dict(s.stats, wall_s=time.monotonic() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--H', type=int, default=1); ap.add_argument('--budget', type=int, default=32); ap.add_argument('--iterations', type=int, default=4)
    ap.add_argument('--n0', type=int, default=512); ap.add_argument('--n', type=int, default=4096); ap.add_argument('--frontier', type=int, default=64)
    ap.add_argument('--rh-allowed', type=int, default=1)
    ap.add_argument('--resume', action='store_true', help='resume is automatic from state.json (completed iterations are reloaded)')
    a = ap.parse_args(); torch.set_num_threads(4); torch.manual_seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hs, ck = hncp.load(a.source, DEVICE); engine = hgen.Engine(hs); heads = csearch.Heads()
    state = read_json(out / 'state.json') if (out / 'state.json').exists() else dict(done=[], policy=None)
    for i in state['done']:                                   # resume: reload the last completed iteration's selector and heads
        z = torch.load(out / f'iter{i}' / 'heads.pt', map_location='cpu', weights_only=False)
        hs.codes.selector.load_state_dict(z['selector']); heads.load_state_dict(z['heads'])
    write_json(out / 'config.json', dict(vars(a), source_sha256=digest(a.source), device=DEVICE, trainable=dict(selector=sum(p.numel() for p in hs.codes.selector.parameters()), heads=sum(p.numel() for p in heads.parameters()))))
    for i in range(a.iterations + 1):
        if i in state['done']:
            continue
        pol = state['policy']; fallback = i > 0 and not pol['value_leaves']
        grounded = i == 0 or fallback or not a.rh_allowed
        cfg = dict(H=a.H, children='grounded' if grounded else 'imagined', leaf='rollout' if (i == 0 or fallback) else 'value', budget=a.budget, noise=True)
        if i > 0:
            hd = copy.deepcopy(heads).to(DEVICE); hd.t_trusted = pol['tau_trusted']; cfg['heads'] = hd
        n = a.n0 if (i == 0 or fallback) else a.n
        items = fresh_problems(n, 86000 + 100 * a.seed + i)
        problems, stats = solve_batched(engine, items, cfg, 20, 87000 + 100 * a.seed + 10 * i, a.frontier)
        train = [p for j, p in enumerate(problems) if j % 5]; hold = [p for j, p in enumerate(problems) if j % 5 == 0]
        roots, rollouts = records(train)
        t1 = time.monotonic(); new = copy.deepcopy(heads).cpu(); hs.codes.selector.cpu(); fit = fit_iteration(hs, new, roots, rollouts); hs.codes.selector.to(DEVICE)
        rows = validate(engine, new, hold, a.H); policy = leaf_policy(rows, a.H); learn_s = time.monotonic() - t1
        heads = new; d = out / f'iter{i}'; d.mkdir(exist_ok=True)
        atomic_save(d / 'heads.pt', dict(selector=hs.codes.selector.state_dict(), heads=heads.state_dict(), iteration=i))
        # Executed trajectories (prompt, executed chunks, outcome): the token tree's value head is fitted on these (§9.6).
        atomic_save(d / 'traj.pt', [dict(prompt=p.item.prompt, chunks=p.chunks, success=bool(p.success)) for p in problems])
        summ = dict(iteration=i, mode=cfg['children'] + '/' + cfg['leaf'], fallback=fallback, problems=n, success=float(np.mean([p.success for p in problems])),
                    success_by_suite={nm: float(np.mean([p.success for j, p in enumerate(problems) if (j % 2 == 0) == (nm == 'standard')])) for nm in ('standard', 'hard10')},
                    verifier_calls=int(sum(p.verifier_calls for p in problems)), nodes=int(sum(p.nodes for p in problems)), executed_tokens=int(sum(p.executed_tokens for p in problems)),
                    unlabelled_fraction=0., reasons={r: sum(p.reason == r for p in problems) for r in {p.reason for p in problems}}, search=stats, fit=fit, G4=rows, next_policy=policy, learning_s=learn_s)
        write_json(d / 'summary.json', summ)
        state['done'].append(i); state['policy'] = policy; write_json(out / 'state.json', state)
        print(json.dumps({k: summ[k] for k in ('iteration', 'mode', 'success', 'verifier_calls', 'nodes')}), flush=True)
    hncp.save(out / 'checkpoint.pt', hs, dict(config=dict(stage='S4', source=a.source), heads=heads.state_dict(), policy=state['policy']))
    write_json(out / 'train_summary.json', dict(status='completed', completed_steps=a.iterations + 1, iterations=state['done'], policy=state['policy']))


if __name__ == '__main__':
    main()
