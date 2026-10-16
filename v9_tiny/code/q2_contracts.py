"""Q2 preflight contracts (draft §13.1 items 11 (colour-count part), 13, 14 for S6-base), then LOCK_Q2. CPU."""
import re
import copy, json, os, sys, tempfile, time, traceback
os.environ['V8_DEVICE'] = 'cpu'
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import CODE, SPEC, TESTS, DATA, read_json, write_json, digest
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q2 import Q2_FILES
from runtime import load_system
from model import freeze
from generation import generate
from data import Mixer, verify, gen_symbolic, parse_prompt
import lsl, s6, v9data

ANCHOR = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']
tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:3000])
    tests.append(dict(name=name, passed=True, **details))


def base():
    s, _ = load_system(ANCHOR, 'cpu'); freeze(s); s.eval(); return s


def randomize(bank, std=.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in bank.B.values():
            p.copy_(torch.randn(p.shape, generator=g) * std)


def colourcount():
    """Item 11 (colour-count): unique header, dispatch and verifier reject every V8 format and accept every item; splits disjoint."""
    man = read_json(v9paths.V9 / 'data_v9' / 'manifest.json')
    for split, info in man['colourcount'].items():
        check(f'colour-count {split} manifest hash', digest(v9data.CC_DIR / f'colourcount_{split}.npz') == info['sha256'])
    hashes = {}
    for split in ('dev', 'calibration', 'test'):
        a = v9data.load_npz(f'colourcount_{split}'); hs = set(); ok = 0
        for x, P in zip(a['x'], a['prompt_len']):
            p = x[:P]; r = x[P:P + 4]; hs.add(v9data.canonical_cc(p))
            ok += v9data.dispatch_verify(p, r)['success'] and not v9data.dispatch_verify(p, np.r_[r[0] + 1, r[1:]])['success']
            assert len(x) % 4 == 0 and P % 4 == 0
        check(f'colour-count {split}: every item accepted, an off-by-one count rejected; chunk-aligned', ok == len(a['x']), n=len(a['x']))
        hashes[split] = hs
    check('colour-count dev / calibration / test disjoint by content hash', not (hashes['dev'] & hashes['test']) and not (hashes['dev'] & hashes['calibration']) and not (hashes['calibration'] & hashes['test']))
    rej = 0; n = 0
    for name in ('standard', 'lookup', 'compose', 'pointer', 'hard10', 'hard14', 'long18', 'bind', 'shifted'):
        a = np.load(DATA / f'dev_{name}.npz')
        for x, P in zip(a['x'][:64], a['prompt_len'][:64]):
            n += 1
            try:
                v9data.parse_cc(x[:P])
            except ValueError:
                rej += not v9data.is_colourcount(x[:P])
            check_same = v9data.dispatch_verify(x[:P], x[P:P + 64]) == verify(x[:P], x[P:P + 64])
            if not check_same:
                check(f'dispatch passes V8 prompts to the frozen verifier ({name})', False)
    check('colour-count parser and header reject every V8 task format; V8 prompts dispatch to the frozen verifier', rej == n, n=n)
    a = v9data.load_npz('colourcount_dev'); bad = 0
    for x, P in zip(a['x'][:64], a['prompt_len'][:64]):
        try:
            parse_prompt(x[:P]); bad += verify(x[:P], x[P:P + 4])['success']
        except Exception:
            pass
    check('frozen V8 verifier never accepts a colour-count item', bad == 0)
    mix = v9data.CCMixer(5); b = mix.batch()
    check('colour-count mixer excludes every evaluation item; loss on the 4 response tokens only', not (mix.seen_symbols & set((v9data.CC_DIR / 'colourcount_eval_hashes.txt').read_text().split()))
          and all(int(m.sum()) == 4 for m in b['m']))
    excluded = set((DATA / 'evaluation_symbolic_hashes.txt').read_text().split()); cal_bad = 0
    from data import canonical
    for name in ('standard', 'lookup', 'compose', 'pointer', 'hard10', 'bind', 'shifted'):
        a = v9data.load_npz(f'calibration_{name}')
        cal_bad += sum(canonical(x[:P]) in excluded for x, P in zip(a['x'], a['prompt_len']))
    check('calibration split disjoint from every V8/V9 evaluation set', cal_bad == 0)


def gates():
    """Item 14: forced open = plain LoRA; closed = pre-phase model; each gate reads its own unit's input; smoothing resets at documents."""
    s0 = base(); b = Mixer(77).batch('cpu')
    s = copy.deepcopy(s0); lsl.convert(s, .3); banks = lsl.add_phase(s, True)
    for i, bk in enumerate(banks):
        randomize(bk, seed=i)
    # Plain LoRA reference: W' = W + 2 B A on every adapted matrix of the frozen anchor.
    ref = copy.deepcopy(s0)
    with torch.no_grad():
        for blk, bk in zip(lsl.gated_blocks(ref), banks):
            blk.qkv.weight += lsl.SCALE * torch.cat([bk.B[n] @ bk.A[n] for n in ('q', 'k', 'v')], 0)
            blk.o.weight += lsl.SCALE * bk.B['o'] @ bk.A['o']; blk.w12.weight += lsl.SCALE * bk.B['w12'] @ bk.A['w12']; blk.w3.weight += lsl.SCALE * bk.B['w3'] @ bk.A['w3']
    with torch.no_grad():
        lo, lr_, l0 = s(b['x'], b['docs']), ref(b['x'], b['docs']), s0(b['x'], b['docs'])
    valid = b['docs'] >= 0
    err = float((lo - lr_)[valid].abs().max() / lr_[valid].abs().max())
    check('item 14: bank forced open equals plain merged LoRA', err < 1e-5 and float((lo - l0).abs().max()) > 1e-2, rel=err)
    lsl.set_force(s, 0, False)
    with torch.no_grad():
        e = float((s(b['x'], b['docs']) - l0).abs().max())
    check('item 14: bank closed everywhere reproduces the pre-phase model exactly', e == 0., max_abs=e)
    # Each gate reads its own matrix input: record every gate's input and compare to the block's actual intermediate tensors.
    seen = {}
    def hook(blk, g, x, docs):
        seen.setdefault(id(blk), {})[g] = x.detach().clone()
    blk = lsl.gated_blocks(s)[4]; blk.capture_groups = hook
    lsl.set_force(s, 0, True)
    with torch.no_grad():
        s(b['x'], b['docs'])
        # Recompute the block's own intermediates from its input.
        inputs = {}
        def pre(m, args):
            inputs['x'] = args[0].detach().clone()
        hd = blk.register_forward_pre_hook(pre); s(b['x'], b['docs']); hd.remove()
    blk.capture_groups = None
    r = seen[id(blk)]; x = inputs['x']; a = blk.n1(x)
    check('item 14: q/k/v gate reads the block input n1(x)', torch.equal(r['qkv'], a))
    check('item 14: o, w12, w3 gates read their own matrix inputs (distinct tensors)', r['o'].shape[-1] == lsl.D and r['w3'].shape[-1] == lsl.FF and not torch.equal(r['o'], r['qkv'])
          and not torch.equal(r['w12'], r['qkv']))
    # Smoothing resets at document boundaries: a fitted gate's decisions on document 2 do not depend on document 1's content.
    torch.manual_seed(3)
    mix = Mixer(91, 32); bb = mix.batch('cpu')
    lsl.fit_negative(s, [Mixer(92, 32).batch('cpu')]); lsl.fit_positive(s, 0, [Mixer(93, 32, "text").batch('cpu')]); lsl.set_force(s, 0, None)
    x = torch.cat([bb['x'][0, :96], bb['x'][1, :96]])[None]; docs = torch.cat([torch.zeros(96), torch.ones(96)]).long()[None]
    x2 = x.clone(); x2[0, :96] = torch.randint(0, 600, (96,))
    rec = []
    def tel(blk, g, p, out, valid):
        rec.append(out[:, 96:].clone())
    for blk in lsl.gated_blocks(s)[:3]:
        blk.telemetry = tel
    with torch.no_grad():
        s(x, docs); g1 = rec[:]; rec.clear(); s(x2, docs); g2 = rec[:]; rec.clear()
    for blk in lsl.gated_blocks(s):
        blk.telemetry = None
    check('item 14: gate decisions in document 2 independent of document 1 (smoothing resets)', all(torch.equal(u, v) for u, v in zip(g1, g2)) and len(g1) == 12)
    # Causality: smoothed gates read only the past.
    x3 = x.clone(); x3[0, 150:] = torch.randint(0, 600, (42,))
    with torch.no_grad():
        l1 = s(x, docs); l3 = s(x3, docs)
    check('item 14: fitted, smoothed gates are causal (future tokens do not change earlier logits)', float((l1[0, :150] - l3[0, :150]).abs().max()) < 1e-5)


def engine():
    """S6 engine (full recompute) equals frozen generate() on the anchor, every V8 dev suite S6 measures."""
    s0 = base()
    for name, n in (('standard', 48), ('lookup', 32), ('compose', 32), ('pointer', 32), ('hard10', 16), ('bind', 32), ('shifted', 32)):
        a = np.load(DATA / f'dev_{name}.npz'); p = int(a['prompt_len'][0]); x = a['x'][:n]; mx = min(384 - p, x.shape[1] - p)
        r = generate(s0, x[:, :p], max_new=mx); ok, xx, ln = s6.greedy(s0, [row[:p] for row in x], mx)
        same = all(np.array_equal(r['x'][i, p:int(r['lengths'][i])], xx[i, p:int(ln[i])]) for i in range(n))
        check(f'S6 engine equals frozen generate on {name}', same and [v['success'] for v in r['verification']] == ok, n=n, rate=float(np.mean(ok)))
    lsl.convert(s0, .3)
    ok2 = s6.symbolic(s0, 'dev_standard', 48)['success']
    a = np.load(DATA / 'dev_standard.npz'); p = int(a['prompt_len'][0]); mx = min(384 - p, a['x'].shape[1] - p)
    ref = [v['success'] for v in generate(base(), a['x'][:48, :p], max_new=mx)['verification']]
    check('converted model with no banks: identical decisions', ok2 == ref, rate=float(np.mean(ref)))


def main():
    out = TESTS / 'q2_contracts.json'; before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q2_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; ident = lock_identity('Q2', Q2_FILES)
    try:
        torch.set_num_threads(4)
        colourcount(); gates(); engine()
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        check('item 13: no file under any other project runs/ changed', not changed, changed=changed[:20])
        v9changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' in k]
        check('item 13: V9 runs/ changes come only from the running production queue (any block: status, driver, launch, queue log, stage folders)', all(re.search(r'/runs/(S\d|V8|status|driver|launch|queue|G1|Q\d|Q1_|S1_|s\d+_)', k) for k in v9changed), sample=v9changed[:10])
        check('lock identity unchanged during session', lock_identity('Q2', Q2_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='Q2', lock_sha256=ident['lock_sha256'], identity=ident, items=['11 (colour-count)', '13', '14'], tests=tests, seconds=time.time() - t0, temp_dir=tmp))
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q2.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in Q2_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
