"""S6 one-phase timing pilot (draft §10.7 budget): Phi_neg fit, training per update, Phi_pos fit and each old-task
evaluation, timed separately on the run device, in a temporary directory. Writes tests/s6_pilot.json."""
import json, sys, tempfile, time, argparse
import numpy as np
import torch
import v9paths
from v9paths import SPEC, TESTS, SMOKE, read_json, write_json
from runtime import DEVICE, sync, load_system
from model import freeze
from data import Mixer
import lsl, s6

PILOT_STEPS = 5 if SMOKE else 200


def timed(f):
    sync(); t = time.time(); r = f(); sync(); return r, time.time() - t


def main():
    torch.set_num_threads(4); anchor = read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']; tmp = tempfile.mkdtemp(prefix='v9_s6_pilot_'); out = {}
    s, _ = load_system(anchor, DEVICE); freeze(s); lsl.convert(s, .3)
    mix = Mixer(99100, 64, 'foundation'); bs = [mix.batch(DEVICE) for _ in range(s6.NB)]
    _, out['phi_neg_fit_s'] = timed(lambda: lsl.fit_negative(s, bs))
    for arm in ('lsl', 'vq'):
        _, t = timed(lambda: s6.train_phase(argparse.Namespace(source=anchor, out=f'{tmp}/{arm}', arm=arm, mode='text', phase=1, steps=PILOT_STEPS, seed=931, resume=False)))
        j = read_json(f'{tmp}/{arm}/train_summary.json'); out[f'{arm}_phase_total_s_{PILOT_STEPS}'] = t; out[f'{arm}_train_s_per_update'] = j['wall_s'] / PILOT_STEPS
    m = s6.load_any(f'{tmp}/lsl/checkpoint.pt'); m.eval()
    _, out['phi_pos_fit_s'] = timed(lambda: lsl.fit_positive(m, 0, s6.phase_batches('text', 1, s6.NB)))
    for name, n in s6.FOUNDATION_SUITES:  # s6.CAP limits n in smoke mode
        _, out[f'eval_{name}_{n}_s'] = timed(lambda: s6.symbolic(m, f'dev_{name}', n))
    _, out['eval_text_256_s'] = timed(lambda: s6.text(m, 'base', 'edu'))
    _, out['eval_colourcount_512_s'] = timed(lambda: s6.symbolic(m, 'colourcount_dev', 512))
    _, out['g60_s'] = timed(lambda: s6.g60(f'{tmp}/lsl/checkpoint.pt', 'colourcount', ['text'], 931))
    measure = sum(v for k, v in out.items() if k.startswith('eval_')) + 3 * out['eval_text_256_s'] + 2 * out['eval_colourcount_512_s']
    per_lsl = out['lsl_train_s_per_update'] * 2000 + out['phi_pos_fit_s'] + measure * 2 + out['g60_s']
    per_other = out['vq_train_s_per_update'] * 2000 + measure
    out['estimate'] = dict(per_lsl_phase_s=per_lsl, per_other_phase_s=per_other, q2_total_h=(2 * 3 * (per_lsl + 2 * per_other) + 2 * out['phi_neg_fit_s']) / 3600,
                           note='measured on %s while Q1 trains on the same GPU' % DEVICE)
    out_path = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else TESTS / 's6_pilot.json'
    write_json(out_path, dict(device=DEVICE, pilot_steps=PILOT_STEPS, temp_dir=tmp, **out)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
