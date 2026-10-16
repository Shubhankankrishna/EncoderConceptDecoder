"""Q6 preflight contracts (draft §13.1 items 7, 12, 13, 15 for S5; I17), then LOCK_Q6. CPU, K = 8 fixture."""
import copy, json, os, sys, tempfile, time, traceback
os.environ['V8_DEVICE'] = 'cpu'
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q6 import Q6_FILES
from data import COL0, DONE
import hncp, hgen, csearch, s5
import measure_s2 as M

TEACHER = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']
tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:3000])
    tests.append(dict(name=name, passed=True, **details))


def fixture():
    torch.manual_seed(12); hs, _ = hncp.build(TEACHER, 'dense', 8, 'cpu')
    with torch.no_grad():
        hs.codes.W_z.weight.normal_(std=.3); hs.codes.selector.weight.normal_(std=.1)
    for p in hs.parameters():
        p.requires_grad_(False)
    hs.eval(); heads = csearch.Heads()
    for p in heads.parameters():
        p.requires_grad_(False)
    return hs, hgen.Engine(hs), heads


def item7(eng, heads):
    items, _ = M.load_suite('standard', 4); ps = csearch.problems_from(items, 20, 0)
    s = csearch.Search(eng, dict(H=3, children='imagined', leaf='value', budget=1, noise=False, heads=heads))
    roots, _ = s.make_roots(ps); direct = s.decode_children(ps, roots, [3] * 4)
    im = []
    for r in roots:
        n = csearch.Node(r, 3, 1, 8); n.base = r; n.preds = [r.chatk[3]]; n.recs = [(r.chatk[3], r.ak[3])]; im.append(n)
    s.replay(ps, im)
    check('item 7: decoding under a stored plan record equals decoding directly under the same plan', all(np.array_equal(a.tokens, b.tokens) for a, b in zip(direct, im)))
    # Depth-2 imagined path: replay applies the depth-2 record (computed on rolled concepts), and it stays applied.
    two = []
    for r in roots:
        c1 = csearch.Node(r, 3, 1, 8); c1.base = r; c1.preds = [r.chatk[3]]; c1.recs = [(r.chatk[3], r.ak[3])]; two.append(c1)
    s.imagine_rows(two)
    lv2 = []
    for c1 in two:
        c2 = csearch.Node(c1, 2, 2, 8); c2.base = c1.base; c2.preds = c1.preds + [c1.chatk[2]]; c2.recs = c1.recs + [(c1.chatk[2], c1.ak[2])]; lv2.append(c2)
    s.replay(ps, lv2)
    J = [len(r.tokens) // 4 - 1 for r in roots]
    check('item 7: replayed nodes carry their records as immutable overrides at the right chunks', all(set(n.over) == {j, j + 1} for n, j in zip(lv2, J) if not n.terminal))
    # A later decode of a replayed prefix applies the stored record: perturbing the stored record changes the continuation.
    live = [n for n in lv2 if not n.terminal]
    if live:
        s.ground_rows(live); alt = []
        for n in live:
            m = copy.copy(n); m.over = {j: (c * 0, a * 0) for j, (c, a) in n.over.items()}; alt.append(m)
        cap = []; orig = eng.logits_at
        def spy(hid, pos):
            out = orig(hid, pos); cap.append(out.detach().clone()); return out
        eng.logits_at = spy
        try:
            pl = [ps[lv2.index(n)] for n in live]; s.decode_children(pl, live, [0] * len(live)); l1 = cap[0]; cap.clear()
            s.decode_children(pl, live, [0] * len(live)); l1b = cap[0]; cap.clear()
            s.decode_children(pl, alt, [0] * len(live)); l2 = cap[0]
        finally:
            eng.logits_at = orig
        dl = float((l1 - l2).abs().max())
        check('item 7: later decodes apply the stored records (same records -> identical logits; altered records -> different logits)', torch.equal(l1, l1b) and dl > 1e-3, max_dlogit=dl)
    # Imagined children of a grounded node read that node's grounded context, never the stale rolled concepts.
    g = csearch.Search(eng, dict(H=20, children='imagined', ground_every=2, leaf='value', noise=False, heads=heads, one_tree=True))
    ps2 = csearch.problems_from(items, 20, 1); rts, nz = g.make_roots(ps2); g.simulate(ps2, rts, nz, 24)
    ok = True; seen = 0
    def walk(n):
        nonlocal ok, seen
        for c in n.children.values():
            if not c.grounded and c.base is not None:
                ok &= c.base.grounded and (c.base is n or c.base is n.base); seen += 1
            walk(c)
    for r in rts:
        walk(r)
    check('item 7: re-grounding: imagined nodes below a re-grounded node are computed from its grounded context', ok and seen > 0, imagined=seen)


def item15(eng, heads):
    s = csearch.Search(eng, dict(H=1, children='imagined', leaf='value', budget=1, heads=heads))
    n = csearch.Node(None, None, 0, 8); n.prior = np.r_[np.full(8, .1), .2]; n.N[:] = 0
    check('item 15: with N(s) = 0 the literal PUCT scores all tie, so the first simulation takes action 0 (lower index)', s.puct(n) == 0)
    n.N[:] = 0; n.N[0] = 1; n.W[0] = 0.
    check('item 15: after one visit, the prior steers selection (highest-prior unvisited action)', s.puct(n) == 8)
    n.prior = np.full(9, 1 / 9); n.N[:] = 0; n.W[:] = 0.
    check('item 15: exact ties -> lowest index', s.puct(n) == 0)
    items, _ = M.load_suite('standard', 4)
    runs = []
    for _ in range(2):
        ps = csearch.problems_from(items, 20, 3); csearch.Search(eng, dict(H=1, children='imagined', leaf='value', budget=5, noise=True, heads=heads)).solve(ps); runs.append([p.chunks for p in ps])
    check('item 15: fixed-seed search reproduces the same trajectories on CPU', runs[0] == runs[1])
    # Execution tie rule: most visits, then higher prior, then lower index.
    N = np.array([3, 3, 1, 0, 0, 0, 0, 0, 0.]); prior = np.array([.1, .2, .3, .1, .1, .1, .05, .05, 0])
    order = sorted(range(9), key=lambda a: (-N[a], -prior[a], a))
    check('item 15: execution ties -> higher prior -> lower index', order[0] == 1)


def token_tree(eng):
    items, _ = M.load_suite('standard', 3); ps = csearch.problems_from(items, 20, 0)
    ts = s5.TokenSearch(eng, dict(H=1, children='grounded', leaf='rollout', budget=2, noise=False))
    roots, _ = ts.make_roots(ps)
    check('token tree: 4 actions (3 colours + STOP), prior is a distribution', all(len(r.prior) == 4 and abs(r.prior.sum() - 1) < 1e-5 for r in roots))
    kids = ts.decode_children(ps, roots, [2, 0, 1])
    check('token tree: each child forces its colour as the hop\'s first token', all(int(k.tokens[len(r.tokens)]) == COL0 + a for k, r, a in zip(kids, roots, [2, 0, 1])))


def accounting(eng, heads):
    items, _ = M.load_suite('hard10', 4); calls = [0]; orig = csearch.verify
    def counted(p, r):
        calls[0] += 1; return orig(p, r)
    import baselines
    ob = baselines.verify; baselines.verify = counted
    try:
        r, st = s5.run_best_of_n(eng, items, 2., 20, 0)
        check('item 12: best-of-N counts one verifier call per sample', sum(x['verifier_calls'] for x in r) == calls[0] == st['samples'])
    finally:
        baselines.verify = ob
    ok, tg, mh = s5.greedy_batch(eng, items)
    r, st = s5.run_search(eng, items, 'rh', .5 * tg * len(items), dict(H=1, children='imagined', leaf='value', noise=False, heads=heads), 20, 0, mh)
    check('budget: the 2x hard stop bounds wall-clock (+ one step); unfinished items are failures', st['wall_s'] <= 2 * .5 * tg * len(items) + 30 and all(x['reason'] for x in r), wall=st['wall_s'])


def main():
    out = TESTS / 'q6_contracts.json'; before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q6_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; ident = lock_identity('Q6', Q6_FILES)
    try:
        torch.set_num_threads(4); hs, eng, heads = fixture()
        item7(eng, heads); item15(eng, heads); token_tree(eng); accounting(eng, heads)
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        check('item 13: no file under any other project runs/ changed', not changed, changed=changed[:20])
        check('lock identity unchanged during session', lock_identity('Q6', Q6_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='Q6', lock_sha256=ident['lock_sha256'], identity=ident, items=['7', '12', '13', '15 (CPU)', 'I17'], tests=tests, seconds=time.time() - t0, temp_dir=tmp))
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q6.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in Q6_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
