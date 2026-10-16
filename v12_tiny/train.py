"""Phase 1: pretrain a base on the pinpad train split.

  python -B train.py ARM SEED STEPS          -> runs/ARM_sSEED/{train.jsonl, final.pt, manifest.json}

Every arm with the same seed sees the identical episode stream (same Sampler seed and call order; a per-step
hash of the batch indices is logged). Step-based schedule: AdamW (0.9, 0.95), lr 1e-3, wd 0.03 on matrices,
warmup 200, cosine to 10%; clip 1.0 on the trunk only (auxiliary heads clipped separately, V11 lesson).
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import pinpad as P
from data import Split, Sampler, Mixed, VOCAB
from model import Transformer, ECD, n_params
from losses import ntp, PQCodebook, NCPHead, NextLat, ncp_ce, nextlat_loss

ROOT = Path(__file__).resolve().parent
D = 128
ARMS = {
    'TF6':     dict(model='TF', layers=6),
    'TF7':     dict(model='TF', layers=7),                       # size-matched to ECD
    'TFNL':    dict(model='TF', layers=7, nextlat=True),
    'ECD':     dict(model='ECD'),
    'ECDe':    dict(model='ECD', ncp='enc'),
    'ECDd':    dict(model='ECD', ncp='dec'),
    'ECDeNL':  dict(model='ECD', ncp='enc', nextlat=True),
    'ECDL2e':  dict(model='ECD', ncp='enc', loop=2),
}
LAM = dict(ncp=0.1, nl_h=1.0, nl_kl=1.0, obs=0.1)


def build(arm, vocab):
    a = ARMS[arm]
    if a['model'] == 'TF':
        m = Transformer(vocab, D, a['layers'])
    else:
        m = ECD(vocab, D, k=4, loop=a.get('loop', 1))
    aux = torch.nn.ModuleDict()
    if a.get('ncp'):
        aux['ncp_head'] = NCPHead(D); m.codebook = PQCodebook(D)
    if a.get('nextlat'):
        aux['nl'] = NextLat(D, m.emb)
    return m, aux


def param_groups(mods, lr):
    dec, nodec = [], []
    for mod in mods:
        for n, p in mod.named_parameters():
            if not p.requires_grad: continue
            (dec if p.dim() >= 2 and 'emb' not in n.split('.')[0] else nodec).append(p)
    return [dict(params=dec, weight_decay=0.03), dict(params=nodec, weight_decay=0.0)]


def lr_at(step, total, base=1e-3, warm=200):
    if step < warm: return base * (step + 1) / warm
    q = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * q)))


def main(arm, seed, steps, B=128, device='mps'):
    run = ROOT / 'runs' / f'{arm}_s{seed}'; run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model, aux = build(arm, VOCAB)
    model.to(device); aux.to(device)
    a = ARMS[arm]
    opt = torch.optim.AdamW(param_groups([model], 1e-3), 1e-3, betas=(0.9, 0.95))
    opt_aux = torch.optim.AdamW(param_groups([aux], 1e-3), 1e-3, betas=(0.9, 0.95)) if len(list(aux.parameters())) else None
    from data import MIX, BATCH
    man = dict(arm=arm, seed=seed, steps=steps, vocab=VOCAB, mix=MIX, batch=BATCH, params_model=n_params(model),
               params_aux=n_params(aux), lam=LAM, cfg=a, sudoku='excluded (fixed-layout period confound)')
    (run / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(man, flush=True)
    stream = Mixed(seed, LAM['obs'])
    log = open(run / 'train.jsonl', 'w'); t0 = time.time()
    for step in range(steps):
        b, idx = stream.next(device)
        for g in opt.param_groups: g['lr'] = lr_at(step, steps)
        if opt_aux:
            for g in opt_aux.param_groups: g['lr'] = lr_at(step, steps)
        out = model(b['ids'])
        l_ntp, l_act, _ = ntp(out, b)
        loss = l_ntp; rec = {}
        if a.get('ncp'):
            l_ncp, info = ncp_ce(model, out, b, model.codebook, aux['ncp_head'], kind=a['ncp'])
            loss = loss + LAM['ncp'] * l_ncp; rec.update(info)
        if a.get('nextlat'):
            l_h, l_kl = nextlat_loss(model, out, b, aux['nl'])
            loss = loss + LAM['nl_h'] * l_h + LAM['nl_kl'] * l_kl; rec.update(nl_h=l_h.detach(), nl_kl=l_kl.detach())
        opt.zero_grad(set_to_none=True)
        if opt_aux: opt_aux.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if opt_aux: torch.nn.utils.clip_grad_norm_(aux.parameters(), 1.0); opt_aux.step()
        opt.step()
        if device == 'mps' and step % 200 == 0: torch.mps.empty_cache()
        if (step + 1) in (4000, 8000, 12000) and step + 1 < steps:
            torch.save(dict(model=model.state_dict(), aux=aux.state_dict()), run / f'ckpt_{step + 1}.pt')
        if step % 25 == 0 or step == steps - 1:
            rec = {k: (float(v) if torch.is_tensor(v) else v) for k, v in rec.items()}
            rec.update(step=step, family=b['family'], t=round(time.time() - t0, 1), ntp=float(l_ntp), act=float(l_act), gn=float(gn),
                       stream=hashlib.md5(np.asarray(idx).tobytes()).hexdigest()[:8])
            log.write(json.dumps(rec) + '\n'); log.flush()
    torch.save(dict(model=model.state_dict(), aux=aux.state_dict()), run / 'final.pt')
    print('done', arm, seed, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
