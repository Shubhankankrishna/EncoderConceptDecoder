"""QF preflight contracts: the final-analysis statistics (one-sided margin tests, TOST, Holm within a stage, missing runs),
then LOCK_QF. CPU only, synthetic data."""
import json, os, sys, tempfile, time, traceback
import numpy as np
import v9paths
from v9paths import SPEC, TESTS, write_json
from q1_contracts import runs_snapshot
from queue_base import lock_identity
from drive_qf import QF_FILES
import final_analysis as F
from stats import holm

tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:2000])
    tests.append(dict(name=name, passed=True, **details))


def main():
    out = TESTS / 'qf_contracts.json'; before = runs_snapshot(); t0 = time.time(); status = 'failed'; ident = lock_identity('QF', QF_FILES)
    try:
        rng = np.random.RandomState(0); n = 512
        b = (rng.rand(2, n) < .5).astype(float); a = b.copy(); flip = rng.rand(2, n) < .10; a[flip & (b == 0)] = 1     # a better by ~5 points
        r = F.compare(list(a), list(b), 'x', 'diff'); check('difference test detects a ~5-point gain', r['p'] < .01 and r['pooled']['delta'] > .03, p=r['p'])
        r = F.compare(list(b), list(b), 'x', 'diff'); check('identical arms: p = 1', r['p'] == 1.)
        r = F.compare(list(a), list(b), 'x', 'noninf', .02); check('non-inferiority (margin 2 points) passes for a better arm', r['p'] < .01)
        r = F.compare(list(b), list(a), 'x', 'noninf', .02); check('non-inferiority fails for an arm ~5 points worse', r['p'] > .5)
        r = F.compare(list(b), list(b), 'x', 'bound', .02); check('bound claim (< +2 points) holds for identical arms', r['p'] < .05)
        r = F.compare(list(a), list(b), 'x', 'bound', .02); check('bound claim fails when the gain is ~5 points', r['p'] > .5)
        r = F.compare(list(b), list(b), 'x', 'equiv', .05); check('TOST equivalence within +-5 points for identical arms', r['p'] < .01)
        r = F.compare([None], [None], 'x', 'diff'); check('missing runs are reported as missing', r['status'] == 'missing')
        adj = holm(dict(L1=.01, H14=.04, H16=.03)); check('Holm within a stage family', abs(adj['L1'] - .03) < 1e-12 and abs(adj['H16'] - .06) < 1e-12 and abs(adj['H14'] - .06) < 1e-12)
        after = runs_snapshot(); check('item 13: no runs/ file changed', not [k for k in set(before) | set(after) if before.get(k) != after.get(k) and '/v9_tiny/runs/' not in k])
        check('lock identity unchanged during session', lock_identity('QF', QF_FILES) == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        write_json(out, dict(status=status, block='QF', lock_sha256=ident['lock_sha256'], identity=ident, tests=tests, seconds=time.time() - t0))
        if status == 'passed':
            write_json(SPEC / 'LOCK_QF.json', dict(**ident, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[p.name for p in QF_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
