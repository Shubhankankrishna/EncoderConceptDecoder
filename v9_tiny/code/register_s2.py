"""S2 gate register (draft §12.3): zones, the asymmetric inconclusive rule, the seed-split rule, consequences.

Inputs are measure_s2.py outputs for V9-K8 and V9-K1 of one core at one seed. Pooling over suites is declared in
spec/PRE_REGISTRATION.md I11 (written before any S2 result).
"""
import json
import numpy as np
import v9paths
from v9paths import read_json, write_json
import measure_s2 as M

R = 10000


def boot_mean(v, seed=0):
    v = np.asarray(v, float); rng = np.random.RandomState(seed)
    if len(v) == 0:
        return dict(point=None, interval=[None, None], n=0)
    idx = rng.randint(len(v), size=(R, len(v))); s = v[idx].mean(1)
    return dict(point=float(v.mean()), interval=[float(np.quantile(s, .025)), float(np.quantile(s, .975))], n=len(v))


def boot_ratio(num, den, seed=0):
    num = np.asarray(num, float); den = np.asarray(den, float); rng = np.random.RandomState(seed)
    idx = rng.randint(len(num), size=(R, len(num))); s = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return dict(point=float(num.sum() / max(den.sum(), 1e-12)), interval=[float(np.quantile(s, .025)), float(np.quantile(s, .975))], n=len(num))


def boot_text_diff(a, b, seed=0):
    """Paired document bootstrap of NLL difference over both text families pooled."""
    na = np.concatenate([a[s]['numerators'] for s in ('edu', 'ts')]); nb = np.concatenate([b[s]['numerators'] for s in ('edu', 'ts')])
    den = np.concatenate([a[s]['denominators'] for s in ('edu', 'ts')])
    return boot_ratio(na - nb, den, seed)


def zone(iv, pass_lo=None, pass_hi=None, fail_lo=None, fail_hi=None):
    """pass: whole interval in pass zone; fail: whole interval in fail zone; else partial/inconclusive.
    pass_lo: pass zone is >= pass_lo; pass_hi: pass zone is <= pass_hi; fail_lo: fail zone is > fail_lo; fail_hi: fail zone < fail_hi."""
    lo, hi = iv
    if lo is None:
        return 'missing'
    ok = (pass_lo is None or lo >= pass_lo) and (pass_hi is None or hi <= pass_hi)
    if ok:
        return 'pass'
    if (fail_lo is not None and lo > fail_lo) or (fail_hi is not None and hi < fail_hi):
        return 'fail'
    return 'partial'


def worst(*z):
    order = ['missing', 'fail', 'partial', 'pass']
    return min(z, key=order.index)


def pooled_states(meas, dist, suites=('standard', 'hard10')):
    return [(s, r) for s in suites for r in meas['states'][s][dist]['g24']]


def rows_for_seed(k8, k1):
    out = {}
    # G2.1(a) oracle graph NLL vs V9-K1 (paired per item, dev standard).
    a = np.array(k8['nll']['graph']['oracle']['per_item']); b = np.array(k1['nll']['graph']['codefree']['per_item'])
    iv = boot_mean(a - b)
    out['G2.1a'] = dict(measure=iv, zone=zone(iv['interval'], pass_hi=-.05, fail_lo=0.))
    # G2.1(b) swap at the last prompt chunk: per-item sensitivity; legal fraction among changed hops.
    first = pooled_states(k8, 'first')
    sens = boot_mean([r['sensitivity'] for _, r in first])
    legal = boot_ratio([sum(r['changed_legal']) for _, r in first], [len(r['changed_legal']) for _, r in first])
    distinct = float(np.mean([r['distinct_legal'] for _, r in first]))
    zs = zone(sens['interval'], pass_lo=.30, fail_hi=.15); zl = zone(legal['interval'], pass_lo=.90, fail_hi=.70)
    out['G2.1b'] = dict(sensitivity=sens, legal_among_changed=legal, distinct_legal_first_hops=distinct, zone=worst(zs, zl) if 'fail' not in (zs, zl) else 'fail')
    out['G2.1'] = dict(zone='pass' if out['G2.1a']['zone'] == out['G2.1b']['zone'] == 'pass' else 'fail' if 'fail' in (out['G2.1a']['zone'], out['G2.1b']['zone']) else 'partial')
    # G2.2(a) deployable likelihoods vs V9-K1, graph and text.
    za = []
    detail = {}
    for mode, p_hi, f_lo in (('sample', .05, .10), ('argmax', .02, .05)):
        g = boot_mean(np.array(k8['nll']['graph'][mode]['per_item']) - b)
        t = boot_text_diff(k8['nll']['text'][mode], k1['nll']['text']['codefree'])
        detail[mode] = dict(graph=g, text=t)
        za += [zone(g['interval'], pass_hi=p_hi, fail_lo=f_lo), zone(t['interval'], pass_hi=p_hi, fail_lo=f_lo)]
    out['G2.2a'] = dict(**detail, zone=worst(*za) if 'fail' not in za else 'fail',
                        consequence_if_partial='single-sample in +.05..+.10: decoder plans use selector-argmax only; argmax +.02..+.05: proceed flagged')
    # G2.2(b) coverage of the demonstrated next action by the top-2 codes, against the exact ceiling (D_gold).
    cov = [c for s in ('standard', 'hard10') for c in k8['states'][s]['gold']['coverage']]
    ratio = boot_ratio([c['covered'] for c in cov], [c['ceiling'] for c in cov])
    out['G2.2b'] = dict(ratio=ratio, coverage=float(np.mean([c['covered'] for c in cov])), ceiling=float(np.mean([c['ceiling'] for c in cov])),
                        zone=zone(ratio['interval'], pass_lo=.90, fail_hi=.75))
    # G2.2(c) calibration and live codes.
    cal = k8['nll'].get('calibration')
    K = k8['k']
    ze = zone(cal['ece_interval'], pass_hi=.05, fail_lo=.10)
    live = cal['live_codes']; zl = 'pass' if live >= K / 2 else 'fail' if live < K / 4 else 'partial'
    out['G2.2c'] = dict(ece=cal['ece'], ece_interval=cal['ece_interval'], live_codes=live, kstar_logloss=cal['kstar_logloss'], zone=worst(ze, zl) if 'fail' not in (ze, zl) else 'fail')
    # G2.2(d) no harm: argmax greedy success vs V9-K1 on standard (n = 2048), non-inferiority margin -2 points.
    s8 = np.array(k8['greedy']['standard']['success'], float); s1 = np.array(k1['greedy']['standard']['success'], float)
    nh = boot_mean(s8 - s1)
    out['G2.2d'] = dict(measure=nh, hard10=boot_mean(np.array(k8['greedy']['hard10']['success'], float) - np.array(k1['greedy']['hard10']['success'], float)),
                        zone='pass' if nh['interval'][0] >= -.02 else 'fail')
    sub = [out[r]['zone'] for r in ('G2.2a', 'G2.2b', 'G2.2c')]
    out['G2.2'] = dict(zone='pass' if all(z == 'pass' for z in sub) and out['G2.2d']['zone'] == 'pass' else 'fail' if 'fail' in sub else 'partial',
                       not_promoted=out['G2.2d']['zone'] != 'pass')
    # G2.4 on D_self (standard + hard10 pooled).
    st = pooled_states(k8, 'self')
    br = boot_mean([r['distinct_legal'] >= 2 for _, r in st if r['progress_options'] >= 2])
    cv = boot_mean([r['useful'] for _, r in st if r['progress_options'] >= 1])
    pe_num = [sum(r.get('pred_exec_agree', [])) for _, r in st]; pe_den = [len(r.get('pred_exec_agree', [])) for _, r in st]
    pe = boot_ratio(pe_num, pe_den)
    out['G2.4_branching'] = dict(measure=br, duplicate_rate=float(np.mean([r['duplicate_rate'] for _, r in st])), zone=zone(br['interval'], pass_lo=.80, fail_hi=.50))
    out['G2.4_coverage'] = dict(measure=cv, zone=zone(cv['interval'], pass_lo=.95, fail_hi=.85))
    stop_rows = [r for _, r in st if r['stop_legal']]
    out['G2.4_stop'] = dict(stop_coverage=float(np.mean([r['stop_covered'] for r in stop_rows])) if stop_rows else None, stop_states=len(stop_rows),
                            false_stop_per_code=float(np.mean([np.mean(r['false_stop_codes']) for _, r in st if not r['stop_legal']])),
                            false_stop_argmax=float(np.mean([r['argmax_false_stop'] for _, r in st if not r['stop_legal']])), zone='reported')
    out['G2.4_pred_exec'] = dict(measure=pe, zone=zone(pe['interval'], pass_lo=.90, fail_hi=.75))
    out['G2.4_port'] = dict(change_rate=float(np.mean([np.mean(r['port_changes']) for _, r in st])), zone='reported')
    out['G2.4_semantics'] = semantics([r for _, r in st], K)
    later = [r for _, r in st]
    out['G2.4_later_hops'] = dict(sensitivity=boot_mean([r['sensitivity'] for r in later]), zone='reported')
    # G2.3-H on D_self (standard + hard10 pooled), both rules.
    g23 = {}
    for rule in ('argmax', 'uniform'):
        recs = k8['states']['standard']['self']['g23'][rule] + k8['states']['hard10']['self']['g23'][rule]
        sm = M.g23_summary(recs)
        g23[rule] = {n: dict(agreement=boot_mean(np.array(v['agreement_vector'], float)), legality=boot_mean(np.array(v['legality_vector'], float)), distance=v['distance']) for n, v in sm.items()}
    out['G2.3_detail'] = g23
    hz = {}
    for H in (1, 2, 4):
        zs = []
        for n in range(1, H + 1):
            for rule in ('argmax', 'uniform'):
                a = g23[rule][n]['agreement']['interval']; l = g23[rule][n]['legality']['interval']
                if n < H:
                    zs.append('pass' if a[0] >= .90 and l[0] >= .95 else 'fail')
                else:
                    za_, zl_ = zone(a, pass_lo=.90, fail_hi=.80), zone(l, pass_lo=.95, fail_hi=.85)
                    zs.append('fail' if 'fail' in (za_, zl_) else worst(za_, zl_))
        hz[H] = 'fail' if 'fail' in zs else worst(*zs)
    out['G2.3_H'] = hz
    return out


def semantics(rows, K):
    """NMI between code index and decoded colour (pooled), and per-code colour consistency across contexts."""
    pairs = [(k, a) for r in rows for k, a in enumerate(r['actions']) if a >= 0]
    if not pairs or K < 2:
        return None
    ks = np.array([p[0] for p in pairs]); acts = np.array([p[1] for p in pairs])
    def H(v):
        _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log(p)).sum())
    joint = ks * 10 + acts; mi = H(ks) + H(acts) - H(joint); nmi = mi / max(np.sqrt(H(ks) * H(acts)), 1e-12)
    keep = {int(k): float(np.max(np.bincount(acts[ks == k], minlength=5)) / max((ks == k).sum(), 1)) for k in range(K) if (ks == k).any()}
    return dict(nmi_pooled=float(nmi), modal_colour_share_per_code=keep, v4_reference_nmi=.02)


def combine(per_seed):
    """Rule 4: pass needs both seeds to pass, fail needs both to fail; otherwise inconclusive (seed-dependent)."""
    keys = [k for k, v in per_seed[0].items() if isinstance(v, dict) and 'zone' in v] + ['G2.3_H1', 'G2.3_H2', 'G2.3_H4']
    out = {}
    for k in keys:
        zs = [ps['G2.3_H'][int(k[-1])] if k.startswith('G2.3_H') else ps[k]['zone'] for ps in per_seed]
        if len(zs) == 1:
            out[k] = dict(zone=zs[0], seeds=zs, single_seed=True)
        elif all(z == 'pass' for z in zs):
            out[k] = dict(zone='pass', seeds=zs)
        elif all(z == 'fail' for z in zs):
            out[k] = dict(zone='fail', seeds=zs)
        elif all(z == 'reported' for z in zs):
            out[k] = dict(zone='reported', seeds=zs)
        else:
            out[k] = dict(zone='inconclusive' if ('pass' in zs and 'fail' in zs) else 'partial', seeds=zs, seed_dependent='pass' in zs and 'fail' in zs)
    return out


def consequences(c):
    """Register consequences (§12.3, §6.8) from combined rows."""
    z = lambda k: c[k]['zone']
    res = {}
    res['concept_branches_run'] = z('G2.1') != 'fail'
    res['promoted_k8'] = z('G2.2d') == 'pass'
    res['prior_not_deployable'] = z('G2.2') == 'fail' and z('G2.1') != 'fail'
    res['decoder_free_search'] = not ('fail' in (z('G2.4_branching'), z('G2.4_coverage')))
    res['decoder_free_flag'] = res['decoder_free_search'] and ('pass' != z('G2.4_branching') or 'pass' != z('G2.4_coverage'))
    hstar = 0
    for H in (1, 2, 4):
        if z(f'G2.3_H{H}') == 'pass':
            hstar = H
        else:
            break
    res['H_star'] = hstar
    res['RH1_allowed'] = z('G2.3_H1') != 'fail' and res['decoder_free_search']
    res['RH1_exploratory'] = res['RH1_allowed'] and not (z('G2.3_H1') == 'pass' and z('G2.4_branching') == 'pass' and z('G2.4_coverage') == 'pass')
    res['IMAG'] = 'not run' if z('G2.4_pred_exec') == 'fail' else 'flagged' if z('G2.4_pred_exec') != 'pass' else 'run'
    res['L3_confirmatory'] = res['RH1_allowed']
    return res


def decide(pairs, out_path):
    """pairs: list of (k8_measure_path, k1_measure_path) per seed."""
    per = [rows_for_seed(read_json(a), read_json(b)) for a, b in pairs]
    comb = combine(per); cons = consequences(comb)
    res = dict(per_seed=per, combined=comb, consequences=cons, inputs=[list(map(str, p)) for p in pairs])
    write_json(out_path, res); return res
