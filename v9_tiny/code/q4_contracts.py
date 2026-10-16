"""Q4 preflight contracts (draft §13.1 item 2 and the S3 ownership / distillation / data-order rules of §7.2, item 13),
then LOCK_Q4. CPU, on a K = 8 S2 fixture built here from the seed-811 teacher."""
import re
import copy, json, os, subprocess, sys, tempfile, time, traceback
os.environ['V8_DEVICE'] = 'cpu'
import numpy as np
import torch
import v9paths
from v9paths import CODE, SPEC, TESTS, read_json, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_q4 import Q4_FILES
from runtime import load_system
from model import freeze
from data import Mixer
import hncp, s3_eval, train_s3, register_s3

TEACHER = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']
PY = sys.executable
tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:3000])
    tests.append(dict(name=name, passed=True, **details))


def fixture(tmp):
    torch.manual_seed(12); hs, _ = hncp.build(TEACHER, 'dense', 8, 'cpu')
    with torch.no_grad():
        for p in hs.core.parameters():
            p.add_(torch.randn_like(p) * .01)
        hs.codes.W_z.weight.normal_(std=.3); hs.codes.selector.weight.normal_(std=.1)
    path = f'{tmp}/s2.pt'; hncp.save(path, hs, dict(config=dict(teacher=TEACHER, teacher_sha256=v9paths.digest(TEACHER), k=8, m=2, core='dense', seed=911)))
    return path


def transplant(src):
    hs, _ = hncp.load(src, 'cpu'); hs.eval(); t, _ = load_system(TEACHER, 'cpu'); freeze(t); t.eval(); b = Mixer(4).batch('cpu')
    with torch.no_grad():
        e0 = float((s3_eval.transplant(hs, 0.)(b['x'], b['docs']) - t(b['x'], b['docs'])).abs().max())
        e1 = float((s3_eval.transplant(hs, 1.)(b['x'], b['docs']) - hs(b['x'], b['docs'])).abs().max())
        mid = float((s3_eval.transplant(hs, .6)(b['x'], b['docs']) - t(b['x'], b['docs'])).abs().max())
    check('item 2: lambda = 0 reproduces the teacher logits exactly', e0 == 0., max_abs=e0)
    check('lambda = 1 equals the student code-free path; lambda = .6 differs from the teacher', e1 == 0. and mid > 1e-4, lam1=e1, lam06=mid)


def ownership(src, tmp):
    z = torch.load(src, map_location='cpu', weights_only=False)['model']; res = {}
    for arm in ('dec', 'enc', 'joint'):
        out = f'{tmp}/{arm}'
        r = subprocess.run([PY, str(CODE / 'train_s3.py'), '--source', src, '--teacher', TEACHER, '--out', out, '--arm', arm, '--steps', '3', '--seed', '921'], cwd=tmp,
                           env=dict(os.environ), capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise AssertionError(arm + ' train failed ' + r.stderr[-2000:])
        a = torch.load(f'{out}/checkpoint.pt', map_location='cpu', weights_only=False)
        res[arm] = a
        changed = {k for k in z if not torch.equal(a['model'][k], z[k])}
        allowed = {'dec': ('f.dec.', 'f.nf.', 'f.gamma'), 'enc': ('f.dec.', 'f.nf.', 'f.gamma', 'f.enc.', 'f.enc_norm.', 'f.emb.'), 'joint': ('',)}[arm]
        bad = sorted(k for k in changed if not k.startswith(allowed))
        must = {'dec': ('f.dec.',), 'enc': ('f.dec.', 'f.enc.', 'f.emb.'), 'joint': ('f.dec.', 'f.enc.', 'core.', 'codes.', 'f.Q')}[arm]
        missing = [p for p in must if not any(k.startswith(p) for k in changed)]
        check(f'S3 ownership {arm}: only owned parameters change; owned ones do', not bad and not missing, bad=bad[:10], missing=missing)
    check('S3a-dec keeps the tied embedding, codebook, PQ heads, core, codes frozen', not any(k.startswith(('f.emb', 'f.Q', 'f.heads', 'core.', 'codes.', 'f.enc'))
                                                                                               and not torch.equal(res['dec']['model'][k], z[k]) for k in z))
    check('same data order across arms at one seed', res['dec']['mixer']['seen_symbols'] == res['joint']['mixer']['seen_symbols'] == res['enc']['mixer']['seen_symbols'])
    cfg = res['joint']['config']
    check('S3b records constant delta = .1 and the S2 teacher SHA', cfg['delta'] == .1 and cfg['teacher_sha256'] == v9paths.digest(TEACHER))


def distill_targets(src):
    """S3b targets come from the ORIGINAL teacher on its own encoder: moving the student encoder must not move them."""
    hs, _ = hncp.load(src, 'cpu'); t, _ = load_system(TEACHER, 'cpu'); freeze(t); b = Mixer(6).batch('cpu')
    seen = []
    orig = t.foundation.concept
    def spy(c, cd):
        seen.append(c.detach().clone()); return orig(c, cd)
    t.foundation.concept = spy
    try:
        train_s3.teacher_distill(hs, t, b)
        with torch.no_grad():
            for p in hs.f.enc.parameters():
                p.add_(torch.randn_like(p) * .05)
        train_s3.teacher_distill(hs, t, b)
    finally:
        t.foundation.concept = orig
    with torch.no_grad():
        _, cT, _ = t.context(b['x'], b['docs']); _, cS, _, _ = hs.context(b['x'], b['docs'])
    check('S3b distillation targets are computed from the teacher\'s own encoder (invariant to the student encoder)',
          torch.equal(seen[0], seen[1]) and torch.equal(seen[0], cT) and not torch.equal(cS, cT))
    # Mutant: targets from the student's (adapted) encoder are caught by the same probe.
    def mutant(hs, t, b):
        h, c, cd, valid = hs.context(b['x'], b['docs']); return t.foundation.concept(c, cd)
    check('mutant reading the adapted student encoder for teacher targets is detectable', not torch.equal(mutant(hs, t, b), t.foundation.concept(cT, chunk(b))))


def chunk(b):
    from model import chunk_info
    return chunk_info(b['docs'])[0]


def g3_register():
    SP = Path_scratch()
    k8 = read_json(SP / 'm_k8.json'); k1 = read_json(SP / 'm_k1.json')
    same = register_s3.g3(k8, k8, k1)
    check('G3 against itself: G2.1(b) ratio 1, paired differences 0', abs(same['G2.1b']['ratio'] - 1) < 1e-9 and same['standard_vs_S2']['measure']['point'] == 0
          and all(v['ok'] for v in same['transitions_no_worse'].values()))
    worse = copy.deepcopy(k8); worse['greedy']['standard']['success'] = [False] * len(worse['greedy']['standard']['success'])
    check('G3 fails when dev standard success collapses', register_s3.g3(worse, k8, k1)['G3'] == 'fail')
    k1g = register_s3.g3(k1, k1, k1)
    check('G3 on V9-K1 (no codes) reports authority as not applicable', isinstance(k1g['authority'], str))
    d = register_s3.decide(dict(dec=[dict(G3='pass'), dict(G3='pass')], joint=[dict(G3='pass'), dict(G3='fail')]))
    check('seed rule: a split arm is seed-dependent; S3a-dec preferred', d['status'] == dict(dec='pass', joint='seed-dependent') and d['replaces_S2'] == 'dec')


def Path_scratch():
    from pathlib import Path
    return Path(os.environ.get('V9_FIXTURES') or TESTS / 'fixtures')   # quick measurements of the S2 K8 / K1 CPU fixtures


def main():
    out = TESTS / 'q4_contracts.json'; before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q4_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; ident = lock_identity('Q4', Q4_FILES)
    try:
        torch.set_num_threads(4)
        src = fixture(tmp); transplant(src); ownership(src, tmp); distill_targets(src); g3_register()
        after = runs_snapshot(); changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k)
        check('item 13: no file under any other project runs/ changed', not changed, changed=changed[:20])
        v9changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' in k]
        check('item 13: V9 runs/ changes come only from the running production queue (any block: status, driver, launch, queue log, stage folders)', all(re.search(r'/runs/(S\d|V8|status|driver|launch|queue|G1|Q\d|Q1_|S1_|s\d+_)', k) for k in v9changed), sample=v9changed[:10])
        check('lock identity unchanged during session', lock_identity('Q4', Q4_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='Q4', lock_sha256=ident['lock_sha256'], identity=ident, items=['2', 'S3 ownership/distillation/data order', '13'], tests=tests, seconds=time.time() - t0, temp_dir=tmp))
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q4.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in Q4_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
