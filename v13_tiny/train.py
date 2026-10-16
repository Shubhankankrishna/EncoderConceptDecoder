"""Base training (stage S0) of V13-tiny-v2 on the Dataset_v13 suite, one seed, on MPS.

    python3 train.py --run runs/base --hours 11 [--smoke 200] [--no-compile]

Schedules run on *elapsed training time* (the fanless M2 throttles under sustained load, so a step-count schedule set
from a cool smoke test would leave the cosine unfinished): LR warms up over 200 steps, then a cosine to 10% of 1e-3
by time fraction; the NCP weight ramps 0 -> 0.1 over the first 10% of time; the curriculum (§7.3) by time fraction.
Checkpoints are atomic, every 30 minutes and at the end; a snapshot of the weights and codebook is kept at every 10%
of the budget (the new-parameter curve of Dataset_v13 §5.2 is evaluated on those afterwards). Resumes from latest.pt.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import model as M
import ncp as N
import tok as TK

LR, WD, WARM, CLIP, LAMBDA = 1e-3, 0.03, 200, 1.0, 0.1


def param_groups(model):
    dec, nodec, table = [], [], []
    for n, p in model.named_parameters():
        if n.startswith('engram.table'): table.append(p)
        elif p.dim() >= 2: dec.append(p)
        else: nodec.append(p)
    return [dict(params=dec, weight_decay=WD, lr_mult=1.0), dict(params=nodec, weight_decay=0.0, lr_mult=1.0),
            dict(params=table, weight_decay=0.0, lr_mult=5.0)]


def lr_at(step, frac):
    w = min(1.0, (step + 1) / WARM)
    return LR * w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


def to_dev(b, dev):
    return (torch.from_numpy(b['ids']).long().to(dev), torch.from_numpy(b['doc']).long().to(dev),
            torch.from_numpy(b['tgt']).long().to(dev), torch.from_numpy(b['w']).to(dev))


def make_step(model, compile_):
    V = model.c.V

    def fwd(ids, doc, eidx, tgt, w):
        logits, m, ncp_logits = model(ids, doc, eidx)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction='none').view(tgt.shape)
        return (ce * w).sum(), m, ncp_logits
    return torch.compile(fwd, dynamic=False) if compile_ else fwd


def save_atomic(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/base'); ap.add_argument('--hours', type=float, default=11.0)
    ap.add_argument('--B', type=int, default=8); ap.add_argument('--seed', type=int, default=1301)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--no-compile', action='store_true'); ap.add_argument('--log-every', type=int, default=50)
    a = ap.parse_args()
    run = HERE / a.run; run.mkdir(parents=True, exist_ok=True)
    dev = torch.device('mps')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tk = TK.Tok()
    model = M.V13(M.Cfg(V=tk.vocab)).to(dev)
    model.set_ckpt(True)                       # activation checkpointing: 5.4 GB instead of 10.9 GB at B = 8, -7% speed
    cb = N.PQCodebook(model.c.d, model.c.pq_S, model.c.pq_N).to(dev)
    groups = param_groups(model)
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), eps=1e-8)
    budget = a.hours * 3600.0
    state = dict(step=0, elapsed=0.0, snaps=[], tokens=0, scored=0)
    stream_state = None
    ck = run / 'latest.pt'
    if ck.exists() and not a.smoke:
        z = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(z['model']); opt.load_state_dict(z['opt']); cb.load_state_dict(z['codebook'])
        cb._init_py = True
        state = z['state']; stream_state = z['stream']
        torch.set_rng_state(z['rng_cpu'])
        print(f'resumed at step {state["step"]}, elapsed {state["elapsed"] / 3600:.2f} h', flush=True)
    text = D.TextStream(tk)
    stream = D.Stream(tk, a.B, a.seed, n_workers=a.workers, ahead=3, state=stream_state, text=text)
    step_fn = make_step(model, not a.no_compile)
    logf = open(run / ('smoke.jsonl' if a.smoke else 'log.jsonl'), 'a')
    tot, grp = M.count_params(model)
    if state['step'] == 0:
        meta = dict(params=tot, groups=grp, B=a.B, T=D.T_ROW, hours=a.hours, seed=a.seed, lr=LR, wd=WD, warm=WARM,
                    clip=CLIP, ncp_lambda=LAMBDA, tokenizer=tk.hash(), vocab=tk.vocab, compile=not a.no_compile,
                    torch=torch.__version__, started=time.strftime('%Y-%m-%d %H:%M:%S'))
        (run / 'config.json').write_text(json.dumps(meta, indent=1))
    print(f'params {tot:,}; budget {a.hours} h; B {a.B} x {D.T_ROW}', flush=True)
    ema = {}; t_last_ck = time.time(); fam_tok = {}; t_log = time.time(); tok_log = 0; steps_log = 0
    snap_every = budget / 10
    model.train()
    while True:
        frac = state['elapsed'] / budget
        if (not a.smoke and frac >= 1.0) or (a.smoke and state['step'] >= a.smoke): break
        t0 = time.time()
        fam, b = stream.batch(frac)
        ids, doc, tgt, w = to_dev(b, dev)
        eidx = model.engram.keys(ids, doc)
        lr = lr_at(state['step'], frac); lam = LAMBDA * min(1.0, frac / 0.1)
        for g in opt.param_groups: g['lr'] = lr * g['lr_mult']
        ntp_sum, m, ncp_logits = step_fn(ids, doc, eidx, tgt, w)
        ntp = ntp_sum / b['norm']
        gdoc = doc[:, ::model.c.k]
        valid, idx = N.ncp_targets(m, gdoc, cb)
        ncp_l, ncp_acc = N.ncp_loss(ncp_logits, valid, idx)
        loss = ntp + lam * ncp_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.smoke and state['step'] < 2:
            bad = [n for n, p_ in model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            print('non-finite grads:', len(bad), bad[:20], 'ntp', ntp.item(), 'ncp', ncp_l.item(), 'lam', lam, flush=True)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not torch.isfinite(gn):                      # guard: skip the update, count it (logged)
            state['skipped'] = state.get('skipped', 0) + 1
            bad = [n for n, p_ in model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            print(f'step {state["step"]}: non-finite gradient in {len(bad)} tensors {bad[:6]}; update skipped', flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        state['step'] += 1; state['tokens'] += b['used']; state['scored'] += b['scored']
        fam_tok[fam] = fam_tok.get(fam, 0) + b['used']
        tok_log += b['used']; steps_log += 1
        if state['step'] % a.log_every == 0 or state['step'] <= 3 or a.smoke:
            torch.mps.synchronize()
            v = dict(ntp=ntp.item(), ncp=ncp_l.item(), ncp_acc=ncp_acc.item(), gn=gn.item())
            for k, x in v.items():
                ema[(fam, k)] = x if (fam, k) not in ema else 0.9 * ema[(fam, k)] + 0.1 * x
            dt = time.time() - t_log
            rec = dict(step=state['step'], h=round(state['elapsed'] / 3600, 4), frac=round(frac, 4), fam=fam, lr=lr,
                       lam=lam, tok_s=round(tok_log / max(dt, 1e-9)), s_step=round(dt / max(steps_log, 1), 3), **v,
                       ppl=cb.usage_perplexity().item(), mem_gb=round(torch.mps.driver_allocated_memory() / 2 ** 30, 2),
                       tau=[round(float(x), 3) for x in torch.stack([mm.qk.log_tau.exp().mean() for mm in model.modules()
                                                                   if isinstance(mm, M.FullAttn)])],
                       beta=[round(float(x), 3) for x in model.beta.softmax(0)], tokens=state['tokens'],
                       skipped=state.get('skipped', 0))
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            if state['step'] % (a.log_every * 4) == 0 or state['step'] <= 3 or a.smoke:
                fams = ' '.join(f'{f}:{ema.get((f, "ntp"), float("nan")):.3f}' for f in D.FAMILIES)
                print(f'step {state["step"]} {state["elapsed"] / 3600:.2f}h {fam:5s} ntp {v["ntp"]:.3f} ncp {v["ncp"]:.2f} '
                      f'acc {v["ncp_acc"]:.2f} gn {v["gn"]:.2f} lr {lr:.2e} lam {lam:.3f} {rec["tok_s"]} tok/s '
                      f'{rec["s_step"]}s/step mem {rec["mem_gb"]}GB ppl {rec["ppl"]:.1f} | {fams}', flush=True)
            t_log = time.time(); tok_log = 0; steps_log = 0
        state['elapsed'] += time.time() - t0
        k_snap = int(state['elapsed'] // snap_every)
        if not a.smoke and k_snap > len(state['snaps']) and k_snap <= 10:
            p = run / f'snap_{k_snap * 10:03d}.pt'
            save_atomic(dict(model=model.state_dict(), codebook=cb.state_dict(), step=state['step'],
                             elapsed=state['elapsed'], tokens=state['tokens']), p)
            state['snaps'].append(p.name)
        if not a.smoke and (time.time() - t_last_ck > 1800):
            save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), codebook=cb.state_dict(), state=state,
                             stream=stream.state(), rng_cpu=torch.get_rng_state()), ck)
            t_last_ck = time.time()
            (run / 'family_tokens.json').write_text(json.dumps(stream.tokens))
    if not a.smoke:
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), codebook=cb.state_dict(), state=state,
                         stream=stream.state(), rng_cpu=torch.get_rng_state()), ck)
        save_atomic(dict(model=model.state_dict(), codebook=cb.state_dict(), step=state['step'],
                         elapsed=state['elapsed'], tokens=state['tokens']), run / 'final.pt')
        (run / 'family_tokens.json').write_text(json.dumps(stream.tokens))
        (run / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    stream.close()
    print('finished', state['step'], 'steps', f'{state["tokens"]:,}', 'tokens', flush=True)


if __name__ == '__main__':
    main()
