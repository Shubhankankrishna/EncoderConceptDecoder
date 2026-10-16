"""S4/S5 support: dev greedy evaluation per arm, the nodes/s and iteration-time pilot (I15 sizing rule), and the
GradCuit lr / norm-bound pilot on failed development answers (§8.3)."""
import argparse, json, time
import numpy as np
import torch
import v9paths
from v9paths import DATA, SMOKE, read_json, write_json
from runtime import DEVICE, sync
import hncp, hgen, csearch, train_s4, baselines, kernels
import measure_s2 as M


def load_arm(source, adapter=None):
    hs, _ = hncp.load(source, DEVICE)
    if adapter:
        z = torch.load(adapter, map_location='cpu', weights_only=False)
        if z['method'] in ('rft', 'grpo'):
            baselines.load_lora(hs, z['state'])
        else:
            hs.codes.selector.load_state_dict(z['state'])
    elif source.endswith('checkpoint.pt') and 'heads' in torch.load(source, map_location='cpu', weights_only=False):
        pass                                                   # S4 checkpoint: selector already inside the model state
    for p in hs.parameters():
        p.requires_grad_(False)
    return hs, hgen.Engine(hs)


def greedy_eval(source, mode, adapter=None, n=None):
    """mode 'codes' = selector-argmax plans (EI, REINFORCE, promoted reference); 'free' = code-free (token arms)."""
    hs, eng = load_arm(source, adapter); out = {}
    for name in ('hard10', 'standard'):
        items, a = M.load_suite(name, n or (64 if SMOKE else 512)); succ = []
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            if mode == 'codes':
                g = eng.greedy([it.prompt for it in chunk], max_new=a['x'].shape[1] - chunk[0].P); succ += [v['success'] for v in g['verification']]
            else:
                succ += [ok for _, _, ok in baselines.sample_answers(eng, chunk, temp=0., code_policy='free')]
        out[name] = dict(success=[bool(v) for v in succ], rate=float(np.mean(succ)), n=len(succ))
    return out


def pilot(source, H, out):
    """Measured wall-clock for mini iterations (iteration-0 mode and decoder-free mode), nodes/s at frontier
    {1, 16, 64}, and greedy t_g; then the I15 sizing rule."""
    hs, eng = load_arm(source); res = dict(device=DEVICE, nodes_per_s={})
    heads = csearch.Heads()
    with torch.no_grad():                      # conservative timing: tau ~ 0, so episodes run to their natural end or the hop limit
        heads.t.weight.zero_(); heads.t.bias.fill_(-10.)
    for q in heads.parameters():
        q.requires_grad_(False)
    heads = heads.to(DEVICE)
    for F in ((1, 4) if SMOKE else (1, 16, 64)):
        items = train_s4.fresh_problems(F, 99900 + F)
        for kind, cfg in (('imagined', dict(H=H, children='imagined', leaf='value', budget=8, noise=True, heads=heads)),
                          ('grounded', dict(H=1, children='grounded', leaf='value', budget=8, noise=True, heads=heads))):
            ps = csearch.problems_from(items, 20, 1); s = csearch.Search(eng, cfg); sync(); t = time.monotonic(); s.step(ps); sync(); dt = time.monotonic() - t
            res['nodes_per_s'][f'{kind}_F{F}'] = s.stats['expansions'] / dt
    nb = 8 if SMOKE else 32; items = train_s4.fresh_problems(nb, 99950)
    _, st0 = train_s4.solve_batched(eng, items, dict(H=H, children='grounded', leaf='rollout', budget=4 if SMOKE else 32, noise=True), 20, 1, nb)
    nd = 8 if SMOKE else 64; items = train_s4.fresh_problems(nd, 99960)
    _, st1 = train_s4.solve_batched(eng, items, dict(H=H, children='imagined', leaf='value', budget=4 if SMOKE else 32, noise=True, heads=heads), 20, 1, nd)
    scale = 8. if SMOKE else 1.     # smoke measures at budget 4; production at 32
    it0_h = st0['wall_s'] / nb * 512 / 3600 * scale; it_h = st1['wall_s'] / nd * 4096 / 3600 * scale
    items, a = M.load_suite('hard10', 64 if not SMOKE else 8); sync(); t = time.monotonic()
    eng.greedy([it.prompt for it in items], max_new=a['x'].shape[1] - items[0].P); sync(); tg = (time.monotonic() - t) / len(items)
    # I15 rule: B_train 32 if iteration (4,096) <= 3 h and iteration 0 (512) <= 4 h; else B 16; then halve problems.
    B, n = 32, 4096
    if it_h > 3 or it0_h > 4:
        B = 16
        if it_h / 2 > 3:
            n = 2048
    res.update(iteration0_projected_h_at_B32=it0_h, iteration_projected_h_at_B32=it_h, t_g_batched_s=tg, sizing=dict(B_train=B, n=n, n0=512),
               mini=dict(iteration0=st0, iteration=st1))
    write_json(out, res); return res


def gradcuit_pilot(source, out, n=256):
    """lr in {1e-3, 1e-2, 1e-1} per norm-bound setting (bounded .5, unbounded) on failed dev answers (selector-argmax)."""
    hs, eng = load_arm(source); items, a = M.load_suite('hard10', 512); failed = []
    for i in range(0, len(items), 64):
        chunk = items[i:i + 64]; g = eng.greedy([it.prompt for it in chunk], max_new=a['x'].shape[1] - chunk[0].P)
        for r, it in enumerate(chunk):
            if not g['verification'][r]['success']:
                failed.append((it, g['x'][r, :int(g['lengths'][r])], g['codes'][r].numpy()))
    failed = failed[:(8 if SMOKE else n)]; res = dict(n_failed=len(failed), settings={})
    for bound in (.5, None):
        for lr in (1e-3, 1e-2, 1e-1):
            rng = np.random.RandomState(7); ok = [kernels.refine(eng, it, t, c, 'gradcuit', lr, bound, rng=rng)['success'] for it, t, c in failed]
            res['settings'][f'bound={bound}/lr={lr}'] = float(np.mean(ok)) if ok else None
    res['chosen'] = {str(b): max((1e-3, 1e-2, 1e-1), key=lambda lr: (res['settings'][f'bound={b}/lr={lr}'] or 0, -lr)) for b in (.5, None)}
    write_json(out, res); return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd')
    g = sub.add_parser('greedy'); g.add_argument('--source', required=True); g.add_argument('--mode', required=True); g.add_argument('--adapter'); g.add_argument('--out', required=True)
    p = sub.add_parser('pilot'); p.add_argument('--source', required=True); p.add_argument('--H', type=int, default=1); p.add_argument('--out', required=True)
    k = sub.add_parser('gradcuit'); k.add_argument('--source', required=True); k.add_argument('--out', required=True)
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.cmd == 'greedy':
        write_json(a.out, greedy_eval(a.source, a.mode, a.adapter))
    elif a.cmd == 'pilot':
        pilot(a.source, a.H, a.out)
    else:
        gradcuit_pilot(a.source, a.out)
