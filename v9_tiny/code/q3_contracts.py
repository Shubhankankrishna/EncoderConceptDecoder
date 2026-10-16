"""Q3 preflight contracts (draft §13.1 items 1-5, 8, 9 and 13 for S2), then LOCK_Q3.

Items 1, 3, 4 run deterministically on CPU in a subprocess (s2_contracts.py). Item 5 here covers the S2 inference
paths (greedy selector decoding, decoder-free transitions); its value/terminal-head mutants belong to Q5.
"""
import re
import copy, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import V9, CODE, SPEC, TESTS, DATA, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q3 import Q3_FILES

PY = sys.executable
TEACHER = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']
tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:3000])
    tests.append(dict(name=name, passed=True, **details))


def cpu_contracts():
    r = subprocess.run([PY, '-c', 'import json,s2_contracts;print("RESULT"+json.dumps(s2_contracts.run()))'], cwd=CODE, env=dict(os.environ, V8_DEVICE='cpu'),
                       capture_output=True, text=True, timeout=3600)
    if r.returncode:
        raise AssertionError('s2_contracts failed: ' + r.stdout[-2000:] + r.stderr[-3000:])
    recs = json.loads(r.stdout.split('RESULT', 1)[1])
    for rec in recs:
        tests.append(dict(rec, name='[items 1,3,4] ' + rec['name']))
    check('items 1, 3, 4 executed', len(recs) >= 40, records=len(recs))


def fixture(k, core='dense', device='cpu'):
    import hncp
    torch.manual_seed(11)
    hs, teacher = hncp.build(TEACHER, core, k, device)
    if k > 1:
        with torch.no_grad():
            hs.codes.W_z.weight.normal_(std=.5); hs.codes.selector.weight.normal_(std=.2)
    hs.eval()
    return hs, teacher


def engine_parity():
    """The H-NCP engine at K = 1 reproduces frozen generate() token for token (standard, lookup, hard14)."""
    from runtime import load_system
    from model import freeze
    from generation import generate
    import hncp, hgen
    s, _ = load_system(str(Path(TEACHER).parents[1] / 'B_refit_dense' / 'checkpoint.pt'), 'cpu'); freeze(s); s.eval(); s.fast_generation = True
    hs = hncp.HNCP(s.foundation, 'dense', 1); hs.core.load_state_dict(s.core.state_dict()); freeze(hs); eng = hgen.Engine(hs)
    for name, n in (('standard', 24), ('lookup', 16), ('hard14', 4)):
        a = np.load(DATA / f'dev_{name}.npz'); p = int(a['prompt_len'][0]); x = a['x'][:n]; mx = min(384 - p, x.shape[1] - p)
        r = generate(s, x[:, :p], max_new=mx); g = eng.greedy([row[:p] for row in x], max_new=mx)
        same = all(np.array_equal(r['x'][i, :r['lengths'][i]], g['x'][i, :g['lengths'][i]]) for i in range(n))
        check(f'engine K=1 equals frozen generate on {name}', same and [v['success'] for v in r['verification']] == [v['success'] for v in g['verification']], n=n)


def stop_semantics():
    """Item 8: a forced DONE chunk verifies iff the current node is the goal after >= 1 hop (data.py:85); classify() agrees."""
    import hgen
    from data import verify, DONE, SEP, COL0, NODE0, TO, N_COL, N_LABELS
    n = 0; rng = np.random.RandomState(8)
    for name in ('standard', 'hard10', 'pointer'):
        a = np.load(DATA / f'dev_{name}.npz')
        for i in range(200):
            it = hgen.Item(a['x'][i], a['m'][i], int(a['prompt_len'][i]))
            nh = len(it.hops) - 1
            for h in range(nh + 1):
                cur, hops, legal = it.walk(it.hops[:h]); assert legal
                resp = np.concatenate([np.array(c) for c in it.hops[:h]] + [np.array([DONE, SEP, SEP, SEP])])
                v = verify(it.prompt, resp)['success']
                rule = hgen.stop_legal(it, cur, hops)
                if it.task == 'graph':
                    check_ok = v == rule == hgen.classify(it, cur, hops, [DONE, SEP, SEP, SEP])['legal']
                    if not check_ok:
                        check(f'STOP rule {name}/{i}/{h}', False, verify=v, rule=rule)
                    n += 1
                # A random single hop appended to the legal prefix: classify agrees with the verifier on legality.
                if it.task == 'graph' and h < nh:
                    col = rng.randint(N_COL); dst = rng.randint(N_LABELS)
                    chunk = np.array([COL0 + col, TO, NODE0 + dst, SEP])
                    vv = verify(it.prompt, np.concatenate([resp[:-4], chunk]))['reason']
                    cl = hgen.classify(it, cur, hops, chunk)
                    if cl['legal'] != (vv == 'missing_terminator'):
                        check(f'hop legality {name}/{i}/{h}', False, verifier=vv, classify=cl)
    check('item 8: forced STOP verifies exactly when cur == goal after >= 1 hop (all prefixes of 400 graph items)', n > 2000, prefixes=n)


def closed_loop_reference():
    """§6.5 rollout: subset-row implementation equals a slow per-row sequential reference on the full batch."""
    import train_s2
    from data import Mixer
    from model import chunk_info
    from runtime import ncp_weights
    for core, k in (('dense', 8), ('recurrent', 8), ('dense', 1)):
        hs, _ = fixture(k, core)
        b = Mixer(81996).batch('cpu')
        with torch.no_grad():
            h, c, cd, valid = hs.context(b['x'], b['docs'])
            spans = train_s2.choose_spans(cd, ncp_weights(b), torch.Generator().manual_seed(3), 1.0)
            fast, rolled = train_s2.closed_loop_inputs(hs, c, cd, spans)
            ref = c.clone()
            for r, j0, n in spans:
                for t in range(1, n + 1):
                    H = hs.heads(ref, cd); src = j0 + t - 1
                    if k == 1:
                        pred = H['chat0'][r, src]
                    else:
                        d = (H['chatk'][r, src] - c[r, src + 1][None]).square().mean(-1); pred = H['chatk'][r, src, int(d.argmin())]
                    ref[r, src + 1] = pred
        err = float((fast - ref).abs().max()); changed = int(((fast - c).abs().sum(-1) > 0).sum())
        check(f'closed loop {core} K={k}: subset rollout equals sequential reference', err < 1e-5 and changed == rolled and rolled > 0, error=err, rolled=rolled, spans=len(spans))
        # Spans stay inside one document and start at a scored source chunk.
        cm = ncp_weights(b) > 0
        check(f'closed loop {core} K={k}: spans start at scored chunks and stay in-document',
              all(bool(cm[r, j0]) and all(int(cd[r, j0 + t]) == int(cd[r, j0]) for t in range(1, n + 1)) for r, j0, n in spans))


def restart_rule():
    import train_s2
    from runtime import optimizer
    hs, _ = fixture(8)
    hs.codes.e.requires_grad_(True)
    opt = optimizer(hs, .001, 'REFIT'); loss = hs.codes.e.square().sum(); loss.backward(); opt.step()
    acc = dict(wins=torch.tensor([500, 300, 2, 0, 100, 50, 40, 8]))
    before = hs.codes.e.detach().clone(); logs = []
    dead = train_s2.restart_dead(hs, opt, acc, torch.Generator().manual_seed(0), lambda **o: logs.append(o))
    K = 8; share = np.array([500, 300, 2, 0, 100, 50, 40, 8]) / 1000
    expect = [i for i in range(K) if share[i] < 1 / (10 * K)]
    moved = [i for i in range(K) if not torch.equal(before[i], hs.codes.e[i].detach())]
    near = all(float((hs.codes.e[i] - before[0]).norm()) < 1.5 for i in dead)
    st = opt.state[hs.codes.e]
    check('restart rule: dead = win share < 1/(10K); split from the top code; Adam state zeroed', dead == expect == moved and near and
          all(float(st[kk][i].abs().sum()) == 0 for kk in ('exp_avg', 'exp_avg_sq') for i in dead) and int(acc['wins'].sum()) == 0, dead=dead, expect=expect)


def hindsight_mutants():
    """Item 5 (S2 paths): inference is invariant to hidden demonstrations; mutants that read them are caught."""
    import hgen, measure_s2 as M
    from data import COL0, TO, NODE0, SEP, N_COL, N_LABELS
    hs, _ = fixture(8); eng = hgen.Engine(hs)
    items, _ = M.load_suite('standard', 24)
    def perturbed(items):
        out = []
        for it in items:
            q = copy.copy(it); rng = np.random.RandomState(it.start + 7)
            q.hops = [tuple(int(v) for v in [COL0 + rng.randint(N_COL), TO, NODE0 + rng.randint(N_LABELS), SEP]) for _ in it.hops[:-1]] + [it.hops[-1]]
            q.response = np.concatenate([np.array(h) for h in q.hops]); out.append(q)
        return out
    def greedy_probe(policy_factory, its):
        return eng.greedy([it.prompt for it in its], max_new=68, policy=policy_factory(its))['x']
    real = lambda its: 'argmax'
    def kstar_policy(its):
        # Mutant: the deployment code is the hindsight winner against the demonstration's next chunk.
        def pol(st):
            J = st['J']; out = []
            with torch.no_grad():
                x, docs, L = eng.batch([np.concatenate([its[i].prompt] + [np.array(h) for h in its[i].hops[:1]]) for i in range(len(J))])
                nxt = eng.pooled_next(x, docs, torch.full_like(L, its[0].P))
            d = (st['chatk'] - nxt[:, None]).square().mean(-1)
            return d.argmin(-1)
        return pol
    ok_real = np.array_equal(greedy_probe(real, items), greedy_probe(real, perturbed(items)))
    caught = not np.array_equal(greedy_probe(kstar_policy, items), greedy_probe(kstar_policy, perturbed(items)))
    check('item 5: selector decoding invariant to the hidden demonstration', ok_real)
    check('item 5: mutant feeding k* into the inference plan is caught', caught)
    # Decoder-free transitions must not read the true next chunk.
    gold = M.gold_states(eng, items)
    for s in gold:
        s['P'] = items[0].P
    M.fill_codes(eng, gold)
    def imag_decisions(its):
        recs = M.g23(eng, copy.deepcopy(gold), its, 'argmax', n_max=2)
        return [[st['agree'] for st in r['steps']] for r in recs], [[st['distance'] for st in r['steps']] for r in recs]
    base = imag_decisions(items)
    check('item 5: imagined transitions invariant to the hidden demonstration', base == imag_decisions(perturbed(items)))
    orig = hgen.Engine.imagine
    current = {}
    def mutant(self, c, cd, J, preds):
        its = current['items']; rows = []
        for r in range(len(J)):
            st = gold[r % len(gold)]; it = its[st['item']]; h = st['hops']
            rows.append(np.concatenate([it.prompt] + [np.array(q) for q in it.hops[:min(h + preds.shape[1], len(it.hops))]]))
        x, docs, L = self.batch(rows); hh = self.f.encode(x, docs); cc = self.f.pool(hh)
        true = torch.stack([cc[r, int(J[r]) + 1:int(J[r]) + 1 + preds.shape[1]] for r in range(len(J))])
        return orig(self, c, cd, J, true[:, :preds.shape[1]] if true.shape[1] == preds.shape[1] else preds)
    try:
        hgen.Engine.imagine = mutant
        current['items'] = items; a = imag_decisions(items); current['items'] = perturbed(items); b = imag_decisions(perturbed(items))
    finally:
        hgen.Engine.imagine = orig
    check('item 5: mutant decoder-free transition reading the true next chunk is caught', a != b)


def compile_parity():
    """Compiled heads/decode equal eager: outputs, and gradients of a smooth probe loss through both functions (V8 tolerance).
    The full S2 objective contains discrete argmin choices that may flip on float-level ties, so its loss is compared
    at a looser 1e-3 relative and its gradients are not compared."""
    import hncp, train_s2
    from data import Mixer
    from runtime import DEVICE
    from model import chunk_info
    b = Mixer(81995).batch('cpu'); bd = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
    for core, k, p in (('dense', 1, 0.), ('dense', 1, .25), ('dense', 8, .25), ('recurrent', 1, .25), ('recurrent', 8, .25)):
        torch.manual_seed(5); hs, teacher = hncp.build(TEACHER, core, k, DEVICE)
        with torch.no_grad():
            h, c, cd, valid = hs.context(bd['x'], bd['docs'])
        g = torch.Generator().manual_seed(4)
        res = {}
        for mode in ('eager', 'compiled'):
            m = copy.deepcopy(hs)
            hf = torch.compile(m.heads) if mode == 'compiled' else m.heads; df = torch.compile(m.f.decode) if mode == 'compiled' else m.f.decode
            H = hf(c, cd)
            chat, a = (H['chat0'], H['a0']) if k == 1 else (H['chatk'][:, :, 1], H['ak'][:, :, 1])
            logits = df(h, chat, a, bd['docs'])
            torch.manual_seed(6); w1 = torch.randn_like(logits[:, :, :8]); probe = (logits[:, :, :8] * w1).mean() + sum(v.float().square().mean() for kk, v in H.items() if torch.is_tensor(v) and v.requires_grad)
            probe.backward()
            cfg = dict(steps=12000, k=k, m=2, p_cl=p, variant='detached', alpha=.1, eps=.05, lam_sel=.1)
            if mode == 'compiled':
                cfg['heads_fn'] = hf; cfg['decode_fn'] = df
            loss, _ = train_s2.objective(copy.deepcopy(m) if mode == 'eager' else m, teacher, bd, cfg, 100, torch.Generator().manual_seed(9), None, cpu=b)
            res[mode] = (float(probe), {n: q.grad.detach().cpu() for n, q in m.named_parameters() if q.grad is not None}, float(loss), {kk: v.detach().cpu() for kk, v in H.items() if torch.is_tensor(v)})
        errs = [dict(p=n, max_abs=float((gg - res['compiled'][1][n]).abs().max()), rel=float((gg - res['compiled'][1][n]).norm() / gg.norm().clamp_min(1e-8))) for n, gg in res['eager'][1].items()]
        out_err = max(float((v - res['compiled'][3][kk]).abs().max() / v.abs().max().clamp_min(1e-8)) for kk, v in res['eager'][3].items())
        w = max(errs, key=lambda e: e['rel'])
        check(f'compile parity S2 {core} K={k} p_cl={p}', out_err < 1e-4 and all(e['max_abs'] < 2e-4 and e['rel'] < .01 for e in errs) and abs(res['eager'][2] - res['compiled'][2]) / res['eager'][2] < 1e-3,
              output_rel=out_err, worst_grad=w, objective_loss=[res['eager'][2], res['compiled'][2]])


def main():
    out = TESTS / 'q3_contracts.json'; before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q3_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; ident = lock_identity('Q3', Q3_FILES)
    try:
        torch.set_num_threads(4)
        cpu_contracts()
        os.environ['V8_DEVICE'] = os.environ.get('V8_DEVICE', '')
        engine_parity(); stop_semantics(); closed_loop_reference(); restart_rule(); hindsight_mutants(); compile_parity()
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        check('item 13: no file under any other project runs/ changed; V9 runs/ only by the live Q1 queue', not changed, changed=changed[:20])
        v9changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' in k]
        check('item 13: V9 runs/ changes come only from the running production queue (any block: status, driver, launch, queue log, stage folders)', all(re.search(r'/runs/(S\d|V8|status|driver|launch|queue|G1|Q\d|Q1_|S1_|s\d+_)', k) for k in v9changed), sample=v9changed[:10])
        check('lock identity unchanged during session', lock_identity('Q3', Q3_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='Q3', lock_sha256=ident['lock_sha256'], identity=ident, items=['1', '3', '4', '5 (S2 paths)', '8', '9 (driver-enforced teacher SHA; drift measured)', '13'],
                             tests=tests, seconds=time.time() - t0, temp_dir=tmp, torch=torch.__version__))
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q3.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in Q3_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
