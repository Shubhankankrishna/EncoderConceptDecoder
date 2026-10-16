"""Correctness tests (CPU). python -B tests.py -> tests_result.json"""
import json, random
from pathlib import Path
import numpy as np
import torch
import pinpad as P
import train
from data import Split, Sampler
from model import ECD, Transformer, token_view
from losses import PQCodebook, ncp_ce, nextlat_loss, NCPHead, NextLat
from meta import MetaController, Controlled, decision_rows, switch_metrics, step_loss
from gtctrl import LowRankBank, make_ctrl, group_labels

ROOT = Path(__file__).resolve().parent
R = []


def check(name, cond, detail=''):
    R.append(dict(test=name, passed=bool(cond), detail=str(detail))); print(('PASS ' if cond else 'FAIL ') + name, detail, flush=True)


def main():
    torch.manual_seed(0)
    W = P.World(); V = W.vocab
    dm = Split('dev_meta')
    # 1. sealed test split
    try: Split('test'); check('test split sealed', False)
    except PermissionError: check('test split sealed', True)
    # 2. expert episodes replay to success through the environment
    rng = random.Random(5); ok = 0
    for i in range(200):
        task = P.parse(rng.choice(P.PRETRAIN_TASKS)); ep = P.expert_episode(W, task, rng, eps=0.05)
        if ep is None: continue
        toks = ep['tokens']; h = ep['header']
        H = (len(toks[:h + 1]) - 2) // 2
        colours = [W.tok_cell(toks[2 + 2 * j]) for j in range(W.O)]
        walls = [W.tok_cell(toks[2 + 2 * W.O + 2 * j]) for j in range(W.W)]
        st, _ = P.env_reset(W, rng, (colours, walls, W.tok_cell(toks[h])))
        for t in range(h + 1, len(toks) - 1, 2):
            pos = P.env_step(W, st, int(toks[t]) - P.ACT0, task)
            assert pos == toks[t + 1]
        ok += st['success']
    check('expert episodes replay to success', ok >= 190, ok)
    # 3. causality (future perturbation leaves earlier logits unchanged) for every arm
    b = dm.batch(np.arange(8), lam_obs=0.0)
    for arm in train.ARMS:
        m, _ = train.build(arm, V); m.eval()
        ids = b['ids']; worst = 0.0
        for cut in (26, 27, 29, 30, 31, 33, 40):
            ids2 = ids.clone(); ids2[:, cut] = (ids2[:, cut] + 1) % V
            d = (m(ids)['logits'][:, :cut] - m(ids2)['logits'][:, :cut]).abs().max().item(); worst = max(worst, d)
        check(f'causality {arm}', worst == 0.0, worst)
    # 4. ECD release mask: group g visible only at rows t >= k g + k - 1; concept-off identical before the first release
    m = ECD(V, 128); m.eval()
    rm = m.release_mask(12, 3, 'cpu')
    check('release mask rows', rm[2, 0].item() is False and rm[3, 0].item() is True and rm[7, 1].item() is True and rm[6, 1].item() is False)
    o, o0 = m(b['ids']), m(b['ids'], concept_off=True)
    check('no concept read before first release', (o['logits'][:, :3] - o0['logits'][:, :3]).abs().max().item() == 0.0)
    tv = token_view(o['u'], b['ids'].shape[1], 4)
    check('token_view latest released group', torch.allclose(tv[:, 3], o['u'][:, 0]) and torch.allclose(tv[:, 6], o['u'][:, 0]) and torch.allclose(tv[:, 7], o['u'][:, 1]) and tv[:, 2].abs().max() == 0)
    # 5. zero-initialised controllers are the identity
    for arm in ('TF7', 'ECDe'):
        m, _ = train.build(arm, V); m.eval()
        bank = LowRankBank(m.d); lab = torch.full_like(b['abs_act'], -1); lab[:, :-1] = b['abs_act'][:, 1:]
        d = (m(b['ids'], ctrl=make_ctrl(m, bank, lab))['logits'] - m(b['ids'])['logits']).abs().max().item()
        check(f'GT bank zero-init identity {arm}', d == 0.0, d)
        mc = MetaController(m.d)
        torch.nn.init.zeros_(mc.hyp[-1].weight)
        nll, kl, info = step_loss(m, mc, b, 0.1)
        base = torch.nn.functional.cross_entropy(m(b['ids'])['logits'].flatten(0, 1), b['targets'].flatten(), reduction='none').view_as(b['weights'])
        base = (base * b['is_act']).sum() / b['is_act'].sum()
        check(f'metacontroller zero-hyp identity {arm}', abs(float(nll) - float(base)) < 1e-5, float(nll) - float(base))
    # 6. token-rate control touches only decision rows and the following action rows
    m = Transformer(V, 128, 7); m.eval()
    mc = MetaController(128)
    c = Controlled(m, mc, b, sample=False)
    e = torch.randn(8, b['ids'].shape[1], 128); e2 = c(e)
    rows, mask = decision_rows(b)
    touched = (e2 - e).abs().sum(-1) > 0
    allowed = torch.zeros_like(touched)
    for i in range(8):
        for j in range(rows.shape[1]):
            if mask[i, j]:
                allowed[i, rows[i, j]] = True
                if rows[i, j] + 1 < e.shape[1]: allowed[i, rows[i, j] + 1] = True
    check('token-rate control rows', bool((touched & ~allowed).sum() == 0) and bool(touched.sum() > 0), int((touched & ~allowed).sum()))
    # 7. decision rows predict actions
    isact_at_rows = torch.gather(b['is_act'], 1, rows)[mask]
    check('decision rows predict actions', bool(isact_at_rows.all()))
    # 8. switch metrics on a synthetic perfect switcher
    info = dict(rows=rows, mask=mask)
    lab = torch.gather(b['abs_act'], 1, (rows + 1).clamp(max=b['abs_act'].shape[1] - 1))
    prev = torch.cat([torch.full_like(lab[:, :1], -9), lab[:, :-1]], 1)
    info['beta'] = ((lab != prev) & (lab >= 0) & (prev >= 0)).float()
    sm = switch_metrics(info, b)
    check('switch F1 = 1 for the ground-truth switcher', abs(sm['f1'] - 1) < 1e-9 and abs(sm['true_switches_per_ep'] - 2.0) < 1e-9, sm)
    # 9. PQ codebook: assignment is nearest, EMA update moves codes, dead-code reset keeps codes live
    cb = PQCodebook(128, 4, 32); x = torch.randn(512, 128) * 3 + 1
    cb.update(x); cb.update(x)
    tau = cb.normalise(x); idx = cb.assign(tau)
    z = tau.view(-1, 4, 32); d2 = (z[:, :, None] - cb.code[None]).pow(2).sum(-1)
    check('PQ nearest assignment', bool((d2.gather(-1, idx[..., None])[..., 0] <= d2.min(-1).values + 1e-5).all()))
    check('PQ normalisation EMA', abs(float(tau.mean())) < 0.3 and abs(float(tau.std()) - 1) < 0.3, (float(tau.mean()), float(tau.std())))
    # 10. NCP-CE and NextLat gradients reach the intended modules only
    m, aux = train.build('ECDeNL', V)
    out = m(b['ids'])
    l, info = ncp_ce(m, out, b, m.codebook, aux['ncp_head'], 'enc')
    g = torch.autograd.grad(l, [m.emb.weight, aux['ncp_head'].fc2.weight, m.dec[0].fc1.weight], allow_unused=True)
    check('NCP grad reaches encoder side and head, not decoder', g[0] is not None and g[1] is not None and g[2] is None)
    l_h, l_kl = nextlat_loss(m, out, b, aux['nl'])
    g = torch.autograd.grad(l_kl, [m.nf.g, aux['nl'].f[0].weight], allow_unused=True, retain_graph=True)
    check('NextLat KL: head frozen view, predictor trained', g[0] is None and g[1] is not None)
    # 11. Sampler determinism (stream identity across arms)
    tr = Split('dev'); s1, s2 = Sampler(tr, 16, 3), Sampler(tr, 16, 3)
    check('sampler deterministic', all((s1.next_idx() == s2.next_idx()).all() for _ in range(50)))
    # 12. group labels read the release row
    sub = torch.arange(40)[None].repeat(2, 1)
    check('group labels at release rows', group_labels(sub, 4, 5)[0].tolist() == [3, 7, 11, 15, 19])
    (ROOT / 'tests_result.json').write_text(json.dumps(dict(passed=sum(r['passed'] for r in R), total=len(R), results=R), indent=1))
    print(f"{sum(r['passed'] for r in R)}/{len(R)} pass")


if __name__ == '__main__':
    main()
