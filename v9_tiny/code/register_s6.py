"""S6 register rows G6.0-G6.5, LSL credit and the P3 overlap prediction (draft §10.7, §12.3; definitions in I11)."""
import numpy as np
from register_s2 import boot_mean, boot_ratio

FOUND = ('standard', 'lookup', 'compose', 'pointer', 'hard10')


def sym(e, name):
    return np.array(e['symbolic'][name]['success'], float)


def text_diff(a, b, key):
    return boot_ratio(np.array(a['text'][key]['numerators']) - np.array(b['text'][key]['numerators']), np.array(a['text'][key]['denominators']))


def phase_rows(mode, earlier, pre, post, others, forced_open=None):
    """pre/post: measure_model outputs for one arm; others: post-phase outputs of 'open' and 'vq' arms (same phase, same seed)."""
    r = {}
    if mode == 'text':
        inst = text_diff(post, pre, 'continual'); r['install'] = inst
        r['G6.1'] = 'pass' if inst['interval'][1] < 0 else 'fail'
    else:
        inst = boot_mean(sym(post, mode) - sym(pre, mode)); r['install'] = inst
        r['G6.1'] = 'pass' if inst['interval'][0] >= .10 else 'fail'
    ret = {}
    for name in FOUND + tuple(m for m in earlier if m != 'text'):
        ret[name] = boot_mean(sym(post, name) - sym(pre, name))
    for key in ('base_edu', 'base_ts') + (('continual',) if 'text' in earlier else ()):
        ret['text_' + key] = text_diff(post, pre, key)
    r['retention'] = ret
    r['G6.2'] = 'pass' if all((v['interval'][1] <= .005) if k.startswith('text_') else (v['interval'][0] >= -.02) for k, v in ret.items()) else 'fail'
    pooled = [n for n in FOUND + tuple(m for m in earlier if m != 'text')]
    if 'open' in others:
        d = np.concatenate([sym(post, n) - sym(others['open'], n) for n in pooled]); r['vs_always_open'] = boot_mean(d)
        r['G6.3'] = 'pass' if r['vs_always_open']['interval'][0] > 0 else 'fail'
    if 'vq' in others:
        if mode == 'text':
            v = text_diff(post, others['vq'], 'continual'); r['vs_vq_install'] = v; r['G6.4'] = 'pass' if v['interval'][1] <= .01 else 'fail'
        else:
            v = boot_mean(sym(post, mode) - sym(others['vq'], mode)); r['vs_vq_install'] = v; r['G6.4'] = 'pass' if v['interval'][0] >= -.05 else 'fail'
    if forced_open is not None:
        if mode == 'text':
            v = boot_ratio(np.array(post['text']['continual']['numerators']) - np.array(forced_open['numerators']), np.array(forced_open['denominators']))
            r['same_weight'] = v; r['G6.5'] = 'pass' if v['interval'][1] <= .01 else 'fail'
        else:
            v = boot_mean(sym(post, mode) - np.array(forced_open['success'], float)); r['same_weight'] = v; r['G6.5'] = 'pass' if v['interval'][0] >= -.05 else 'fail'
    r['lsl_credit'] = all(r.get(k) == 'pass' for k in ('G6.1', 'G6.2', 'G6.3', 'G6.4'))
    return r


def h12(lsl_post, open_post):
    v = boot_mean(sym(lsl_post, 'standard') - sym(open_post, 'standard'))
    within = v['interval'][0] >= -.05 and v['interval'][1] <= .05
    wrong = v['interval'][0] > .05
    return dict(interval=v, confirmed=within, prediction_wrong=wrong, not_confirmed=not within and not wrong)
