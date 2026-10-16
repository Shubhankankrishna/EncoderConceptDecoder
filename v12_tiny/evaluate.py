"""Phase-1 evaluation of a trained base (dev splits only).

  python -B evaluate.py RUN       -> runs/RUN/eval.json
- dev NLL: action NLL (nats/action), position NLL, END NLL.
- probes (probes.py): every token-rate depth; for ECD also the concept stream (token view of u and of the state
  at the concept control point).
- competence: closed-loop rollouts judged 'any' (completes some pretraining task) on fresh layouts.
- post task raw success (no controller): the base's own chance of solving the held-out composition.
- NCP: dev code accuracy and CE, and for target='dec' the shortcut monitor (accuracy with the decoder's concept
  read zeroed when computing the *target*).
- concept usage: Δ action NLL with the concept route zeroed (ECD).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pinpad as P
from data import Split
from model import token_view
from probes import probe_all
from rollout import rollout

ROOT = Path(__file__).resolve().parent


def load_run(run, device='mps', ckpt='final'):
    import train
    man = json.loads((ROOT / 'runs' / run / 'manifest.json').read_text())
    model, aux = train.build(man['arm'], man.get('vocab', P.World().vocab))
    sd = torch.load(ROOT / 'runs' / run / f'{ckpt}.pt', map_location='cpu')
    if 'model' not in sd: sd = dict(model=sd, aux={})                                 # pilot checkpoints
    model.load_state_dict(sd['model']); aux.load_state_dict(sd['aux'])
    model.aux = aux
    return model.to(device).eval(), man['arm']


@torch.no_grad()
def dev_nll(model, split, device, n=2000, concept_off=False):
    tot = dict(act=0., pos=0., end=0.); cnt = dict(act=0, pos=0, end=0)
    for s in range(0, n, 128):
        b = split.batch(np.arange(s, min(s + 128, n)), device=device, pad_to=176)
        out = model(b['ids'], concept_off=concept_off)
        lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
        scored = b['weights'] > 0
        is_end = (b['targets'] == P.END) & scored
        is_pos = scored & ~b['is_act'] & ~is_end
        for k, m in (('act', b['is_act']), ('pos', is_pos), ('end', is_end)):
            tot[k] += float((lp * m).sum()); cnt[k] += int(m.sum())
    return {k: tot[k] / max(cnt[k], 1) for k in tot}


@torch.no_grad()
def ncp_dev(model, split, device, n=1000):
    from losses import group_valid
    if not hasattr(model, 'codebook'): return None
    head = model.aux['ncp_head']; kind = None
    import train
    accs, ces, accs_off = [], [], []
    arm_cfg = [a for a in train.ARMS.values() if a.get('ncp')]
    for s in range(0, n, 128):
        b = split.batch(np.arange(s, min(s + 128, n)), device=device, pad_to=176)
        out = model(b['ids'])
        k = model.k; G = out['G']
        kind = model._ncp_kind
        if kind == 'enc': raw = out['pooled_raw']
        else: raw = out['final'][:, :G * k].reshape(b['ids'].shape[0], G, k, -1).mean(2)
        ok = group_valid(b, G, k)[:, 1:]
        idx = model.codebook.assign(model.codebook.normalise(raw[:, 1:][ok]))
        lg = head(out['u'][:, :-1][ok])
        accs.append(float((lg.argmax(-1) == idx).float().mean()))
        ces.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), idx.reshape(-1), reduction='none').view(-1, idx.shape[1]).sum(-1).mean()))
        if kind == 'dec':
            off = model(b['ids'], concept_off=True)
            raw0 = off['final'][:, :G * k].reshape(b['ids'].shape[0], G, k, -1).mean(2)
            idx0 = model.codebook.assign(model.codebook.normalise(raw0[:, 1:][ok]))
            accs_off.append(float((lg.argmax(-1) == idx0).float().mean()))
    r = dict(acc=float(np.mean(accs)), ce=float(np.mean(ces)), code_ppl=float(model.codebook.usage_perplexity()))
    if accs_off: r['acc_vs_concept_off_target'] = float(np.mean(accs_off))
    return r


@torch.no_grad()
def bpe_dev_nll(model, family, device, n=None, concept_off=False):
    """Token NLL on the fixed dev panel of a BPE family: text = first 256 tokens of each of 400 dev docs (all
    tokens scored); math/newmath = every dev problem (answer tokens + EOS)."""
    from data import BPEFamily
    f = BPEFamily(family, 'dev'); n = n or (400 if family == 'text' else f.n)
    tot = cnt = 0.0
    for s in range(0, n, 32 if family == 'text' else 128):
        idx = np.arange(s, min(s + (32 if family == 'text' else 128), n))
        b = f.batch(idx, None, device, pad_to=256 if family == 'text' else 96)
        out = model(b['ids'], concept_off=concept_off)
        lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
        tot += float((lp * b['weights']).sum()); cnt += float(b['weights'].sum())
    return tot / cnt


def main(run, device='mps', ckpt='final'):
    model, arm = load_run(run, device, ckpt)
    import train
    model._ncp_kind = train.ARMS[arm].get('ncp')
    dev = Split('dev'); world = P.World()
    res = dict(run=run, arm=arm, ckpt=ckpt, nll=dev_nll(model, dev, device))
    if getattr(model, 'emb').num_embeddings > 66:
        res['bpe_nll'] = {f: bpe_dev_nll(model, f, device) for f in ('text', 'math', 'newmath')}
        if model.kind == 'ECD':
            res['bpe_nll_concept_off'] = {f: bpe_dev_nll(model, f, device, concept_off=True) for f in ('text', 'math')}
    extra = None
    if model.kind == 'ECD':
        res['nll_concept_off'] = dev_nll(model, dev, device, concept_off=True)
        def extra(out):
            T = out['logits'].shape[1]
            return [token_view(out['u'], T, model.k)] + [token_view(s, T, model.k) for s in out['cstates']]
    res['probes'] = probe_all(model, dev, device, extra=extra)
    r = rollout(model, world, 256, 3, judge='any', device=device)
    res['competence_any'] = float(r['success'].mean()); res['competence_steps'] = float(r['steps'].mean())
    r = rollout(model, world, 256, 4, task=P.parse(P.POST_TASK), device=device)
    res['post_success_raw'] = float(r['success'].mean())
    res['ncp'] = ncp_dev(model, dev, device)
    (ROOT / 'runs' / run / f'eval_{ckpt}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(run=run, nll=res['nll'], bpe=res.get('bpe_nll'), comp=res['competence_any'], post=res['post_success_raw'],
                          probe_abs=[round(x, 3) for x in res['probes']['abs_act']], ncp=res['ncp'])), flush=True)
    return res


if __name__ == '__main__':
    main(sys.argv[1], ckpt=sys.argv[2] if len(sys.argv) > 2 else 'final')
