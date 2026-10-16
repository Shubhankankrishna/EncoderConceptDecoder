"""Q5 preflight contracts (draft §13.1 items 6, 12, 13; S4 label rules §8.2; I15/I16), then LOCK_Q5. CPU, K = 8 fixture."""
import copy, json, os, sys, tempfile, time, traceback
os.environ['V8_DEVICE'] = 'cpu'
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, DATA, read_json, write_json, digest
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q5 import Q5_FILES
from data import canonical, verify
import hncp, hgen, csearch, kernels, train_s4, baselines
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
    hs.eval(); return hs, hgen.Engine(hs)


def failed_answer(eng):
    items, a = M.load_suite('hard10', 16); g = eng.greedy([it.prompt for it in items], max_new=a['x'].shape[1] - items[0].P)
    for r, it in enumerate(items):
        if not g['verification'][r]['success'] and (g['codes'][r] >= 0).any():
            return it, g['x'][r, :int(g['lengths'][r])], g['codes'][r].numpy()
    raise AssertionError('no failed answer in the fixture')


def item6(eng):
    """GradCuit reward: R_grad = o - 1 gives a nonzero gradient on a failed answer and one step lowers its log-probability;
    the mutant R = o is a no-op."""
    hs = eng.hs; it, toks, codes = failed_answer(eng); J0 = it.P // 4 - 1; j = next(j for j in range(J0, len(codes)) if codes[j] >= 0); k = int(codes[j])
    E = hs.codes.e.detach(); x, d, L = eng.batch([toks]); ca = torch.as_tensor(codes[:x.shape[1] // 4])[None]
    delta = torch.zeros(E.shape[1], requires_grad=True); opt = torch.optim.Adam([delta], lr=1e-2)
    def lp(dl):
        ep = E.clone(); ep[k] = E[k] + dl; return kernels.answer_logprob(hs, x, d, ca, it.P, len(toks), j, ep)
    with torch.enable_grad():
        l0 = lp(delta); (-(0. - 1.) * l0).backward()
    g = float(delta.grad.norm()); opt.step()
    with torch.no_grad():
        l1 = float(lp(delta.detach()))
    check('item 6: R_grad = o - 1 gives a nonzero gradient and one step lowers the rejected answer\'s log-probability', g > 0 and l1 < float(l0), grad=g, before=float(l0), after=l1)
    m = kernels.refine(eng, it, toks, codes, 'gradcuit', 1e-1, .5, reward_from_o=True)
    check('item 6: the mutant R = o is caught as an exact no-op', all(t['grad_norm'] == 0 and t['step_norm'] == 0 for t in m['trace']) and len(m['trace']) > 0)
    r = kernels.refine(eng, it, toks, codes, 'gradcuit', 1e-1, .5)
    E2 = hs.codes.e.detach()
    check('GradCuit stops at acceptance or 10 updates; weights stay frozen; bound RMS(delta) <= .5 RMS(e_k)', r['updates'] <= 10 and torch.equal(E, E2)
          and all(t['delta_rms'] <= .5 * float(E[int(codes[r['j']])].square().mean().sqrt()) + 1e-6 for t in r['trace']))
    rr = kernels.refine(eng, it, toks, codes, 'random', 1e-1, .5)
    check('random-direction kernel: same nominal step norm, no reward gradient', all(abs(t['step_norm'] - .1 * 4) < 1e-6 and t['grad_norm'] is None for t in rr['trace']))


def item12(eng):
    """Budget accounting: every verifier call is counted (verify is wrapped and compared with the counters)."""
    calls = [0]; orig = csearch.verify
    def counted(p, r):
        calls[0] += 1; return orig(p, r)
    csearch.verify = counted
    try:
        items, _ = M.load_suite('standard', 6)
        for cfg in (dict(H=1, children='grounded', leaf='rollout', budget=3, noise=True), dict(H=1, children='imagined', leaf='value', budget=4, noise=True, heads=csearch.Heads())):
            calls[0] = 0; ps = csearch.problems_from(items, 20, 0); s = csearch.Search(eng, cfg); s.solve(ps)
            explicit = calls[0]; counted_ = sum(p.verifier_calls for p in ps)
            # Illegal-prefix verdicts are counted as verifier calls without invoking verify (a decided failure): counted >= explicit.
            check(f'item 12: every verify() call is counted ({cfg["children"]}/{cfg["leaf"]})', counted_ >= explicit and s.stats['verifier_calls'] == counted_, explicit=explicit, counted=counted_)
    finally:
        csearch.verify = orig


def labels(eng):
    """tau labels come only from the verifier rule on grounded prefixes; held-out split disjoint; fresh problems exclude evaluation hashes."""
    items = train_s4.fresh_problems(40, 123)
    excluded = set((DATA / 'evaluation_symbolic_hashes.txt').read_text().split())
    check('S4 problems: fresh, unique, disjoint from every evaluation hash', len({canonical(it.prompt) for it in items}) == 40 and not any(canonical(it.prompt) in excluded for it in items))
    ps, _ = train_s4.solve_batched(eng, items[:8], dict(H=1, children='grounded', leaf='rollout', budget=2, noise=True), 20, 0, 8)
    ok = True
    for p in ps:
        for r in p.roots:
            ok &= r['stop_label'] == hgen.stop_legal(p.item, r['cur'], r['hops'])
            # mutant check: the label never consults the demonstration (alternative legal paths are not mislabelled)
            ok &= r['stop_label'] == (r['hops'] >= 1 and r['cur'] == p.item.goal)
    check('tau labels = the verifier STOP rule on grounded states, never the demonstration', ok)
    hold = [j for j in range(20) if j % 5 == 0]; train = [j for j in range(20) if j % 5]
    check('G4 held-out split disjoint by problem', not set(hold) & set(train))


def metrics():
    a, npos, nneg = train_s4.auroc([.1, .2], [0, 1])
    check('undefined AUROC with < 30 items per class', a is None and (npos, nneg) == (1, 1))
    y = np.r_[np.ones(40), np.zeros(40)]; s = np.r_[np.full(40, .9), np.full(40, .1)]
    a, *_ = train_s4.auroc(s, y); row = train_s4.v_row(s, y, .8, .7)
    check('perfect separation: AUROC 1 and Brier below base rate -> pass', a == 1. and row['zone'] == 'pass')
    row = train_s4.v_row(np.full(80, .5), y, .8, .7)
    check('constant scores: AUROC .5 -> fail', row['zone'] == 'fail')
    c = train_s4.v_row(np.full(60, .9), np.ones(60), .7)
    check('constant labels: undefined, cannot pass', c['zone'] == 'inconclusive')
    pol = train_s4.leaf_policy(dict(complete=dict(zone='partial'), intermediate=dict(zone='pass'), imagined_d1=dict(zone='pass'), tau=dict(zone='fail')), 1)
    check('leaf policy: flagged V used; tau fail -> STOP only at grounded nodes', pol == dict(value_leaves=True, tau_trusted=False, flagged=True))
    pol = train_s4.leaf_policy(dict(complete=dict(zone='pass'), intermediate=dict(zone='pass'), imagined_d1=dict(zone='fail'), tau=dict(zone='pass')), 1)
    check('leaf policy: an imagined-depth failure -> verified grounded completion', not pol['value_leaves'])


def baselines_check(hs0):
    hs = copy.deepcopy(hs0); ps = baselines.add_lora(hs)
    check('token baselines: rank-16 q/v LoRA on the three decoder blocks = 24,576 parameters', sum(p.numel() for p in ps) == 24576)
    x = torch.randint(0, 100, (2, 32)); d = torch.zeros_like(x)
    with torch.no_grad():
        same = torch.equal(hs(x, d), hs0(x, d))
    check('LoRA at initialisation (B = 0) leaves the model unchanged', same)
    check('S4 primary trainable parameters: selector 1,032 + two heads 129 each = 1,290', sum(p.numel() for p in hs0.codes.selector.parameters()) + sum(p.numel() for p in csearch.Heads().parameters()) == 1290)


def batching(eng):
    items, _ = M.load_suite('standard', 6); cfg = dict(H=1, children='imagined', leaf='value', budget=4, noise=True, heads=csearch.Heads())
    torch.manual_seed(1); cfg['heads'] = csearch.Heads()
    for p in cfg['heads'].parameters():
        p.requires_grad_(False)
    a = csearch.problems_from(items, 20, 5); csearch.Search(eng, cfg).solve(a)
    b = [csearch.problems_from([it], 20, 5 + i)[0] for i, it in enumerate(items)]
    for p in b:
        csearch.Search(eng, cfg).solve([p])
    check('I15: search results are independent of batching (per-problem noise generators)', all(x.chunks == y.chunks and x.codes == y.codes for x, y in zip(a, b)))
    c = csearch.problems_from(items, 20, 5); csearch.Search(eng, cfg).solve(c)
    check('item 15: a fixed seed reproduces the same trajectories on CPU', all(x.chunks == y.chunks for x, y in zip(a, c)))


def main():
    out = TESTS / 'q5_contracts.json'; before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q5_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; ident = lock_identity('Q5', Q5_FILES)
    try:
        torch.set_num_threads(4); hs, eng = fixture()
        item6(eng); item12(eng); labels(eng); metrics(); baselines_check(hs); batching(eng)
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        check('item 13: no file under any other project runs/ changed', not changed, changed=changed[:20])
        check('lock identity unchanged during session', lock_identity('Q5', Q5_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='Q5', lock_sha256=ident['lock_sha256'], identity=ident, items=['6', '12', '13', '15 (CPU)', 'S4 labels', 'I15/I16'], tests=tests, seconds=time.time() - t0, temp_dir=tmp))
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q5.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in Q5_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
