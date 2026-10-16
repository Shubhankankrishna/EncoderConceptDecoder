"""Base training (stage S0) of V14-tiny on the Dataset_v13 suite, one seed, on MPS (v14_Architecture.md §11).

    python3 train.py --run runs/base --hours 11 [--smoke 200] [--no-compile] [--ckpt on|off]

V13-tiny's loop (train.py there) with V14's objective (§8.1):
    L = L_NTP + lambda(t) L_NCP + nu_S(t) (L_SIG,a + L_SIG,b + L_SIG,b,uncentred)
- the NCP targets are the EMA twin's target view tau = [a; b] (no gradient), quantised by the PQ codebook after its
  update on this batch's valid targets (as in V13: update, then assign); horizons 1-4 with w = (0.4, 0.3, 0.2, 0.1);
- the residue maps P_f are updated from the twin's tensors every batch and re-solved every 100 steps;
- the twin moves by EMA with mu(t) by elapsed time (cosine from 0.99 to 1 at 75% of training time); at 75% the twin,
  the normaliser, the codebook and the maps freeze together;
- lambda ramps 0 -> 0.1 over the first 10% of time and is halved (floor 1/8) after two consecutive gradient-share
  readings above 1 (every 500 steps, on a fixed held batch, in a separate eager step: ||grad_C lambda L_NCP|| /
  ||grad_C L_NTP|| over the concept module); nu_S = 0.05 ramps with lambda (§8.2).
Schedules, checkpoints and snapshots are V13-tiny's (elapsed time; every 30 min; every 10% of the budget).
"""
import argparse
import gc
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
import tok as TK
import v14 as V

LR, WD, WARM, CLIP, LAMBDA, NU_S = 1e-3, 0.03, 200, 1.0, 0.1, 0.05
FAM_IDX = {f: i for i, f in enumerate(V.FAMILIES)}


def param_groups(model):
    dec, nodec, table = [], [], []
    for n, p in model.named_parameters():
        if n.startswith('engram.table'): table.append(p)
        elif p.dim() >= 2 and n != 'e_h': dec.append(p)
        else: nodec.append(p)                                     # e_k takes no weight decay (§8.6)
    return [dict(params=dec, weight_decay=WD, lr_mult=1.0), dict(params=nodec, weight_decay=0.0, lr_mult=1.0),
            dict(params=table, weight_decay=0.0, lr_mult=5.0)]


def lr_at(step, frac):
    w = min(1.0, (step + 1) / WARM)
    return LR * w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


def to_dev(b, dev):
    return (torch.from_numpy(b['ids']).long().to(dev), torch.from_numpy(b['doc']).long().to(dev),
            torch.from_numpy(b['tgt']).long().to(dev), torch.from_numpy(b['w']).to(dev))


def make_fns(model, twin, consts):
    V_ = model.c.V; k = model.c.k

    def twin_fn(ids, doc, eidx, P_rows):
        return V.twin_targets(twin, ids, doc, eidx, P_rows)

    def step_fn(ids, doc, eidx, tgt, w, codes, valid_t, P_rows, om):
        ctx = model.context(doc)
        x, L_enc, G_enc = model.encode(ids, eidx, ctx)
        m, u = model.concepts(x, L_enc, G_enc, ctx)
        logits = model.decode(L_enc, u, ctx)
        ce = F.cross_entropy(logits.reshape(-1, V_), tgt.reshape(-1), reduction='none').view(tgt.shape)
        gdoc = doc[:, ::k]
        hl = model.horizon_logits(u)
        ncp, hst = V.horizon_loss(hl, codes, gdoc, valid_t)
        a, b, valid = V.online_view(model, x, L_enc, G_enc, gdoc, P_rows)
        sa, sb, sbu = V.sigreg_terms(a, b, gdoc, valid, om, consts)
        with torch.no_grad():
            same = V.same_argmax_fraction(hl.detach(), gdoc)
        return (ce * w).sum(), ncp, sa, sb, sbu, hst, same
    return twin_fn, step_fn


def save_atomic(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


def held_batch(tk, B, seed=4242):
    """A fixed batch for the gradient-share cap: B/4 rows from each reasoning family, at curriculum fraction 0.5."""
    D._init_worker()
    rows, fams = [], []
    for i, f in enumerate(('grid', 'prog', 'state', 'plan')):
        for j in range(B // 4):
            rows.append(D.make_row(f, 10_000_000 + j, 0.5, seed)); fams.append(FAM_IDX[f])
    return D.collate(rows), torch.tensor(fams)


def grad_share(model, twin, cb, maps, hb, hfam, dev, lam, consts, step):
    """||grad_C (lambda L_NCP)|| / ||grad_C L_NTP|| on the held batch (two backward passes, eager), plus the G-side
    balance read of §8.1 (norms of each loss's gradient over E2, E3 and E2's exit)."""
    twin_fn, step_fn = make_fns(model, twin, consts)
    ids, doc, tgt, w = to_dev(hb, dev)
    eidx = model.engram.keys(ids, doc)
    P_rows = maps.rows(hfam.to(dev))
    with torch.no_grad():
        tau, valid_t, _, _ = twin_fn(ids, doc, eidx, P_rows)
        codes = cb.assign(cb.normalise(tau))
    om = V.directions(step, model.c.proj, device=dev)
    ntp_sum, ncp, sa, sb, sbu, _, _ = step_fn(ids, doc, eidx, tgt, w, codes, valid_t, P_rows, om)
    ntp = ntp_sum / hb['norm']; sig = sa + sb + sbu
    C = [p for p in model.cm.parameters() if p.requires_grad]
    Gs = [p for n, p in model.named_parameters() if n.startswith(('e2.', 'e3.', 'e2_exit.'))]
    out = {}
    for name, loss, keep in (('ncp', lam * ncp, True), ('ntp', ntp, True), ('sig', sig, False)):
        gr = torch.autograd.grad(loss, C + Gs, retain_graph=keep, allow_unused=True)
        nC = math.sqrt(sum(float(g.pow(2).sum()) for g in gr[:len(C)] if g is not None))
        nG = math.sqrt(sum(float(g.pow(2).sum()) for g in gr[len(C):] if g is not None))
        out[name] = (nC, nG)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/base'); ap.add_argument('--hours', type=float, default=11.0)
    ap.add_argument('--B', type=int, default=8); ap.add_argument('--seed', type=int, default=1301)
    ap.add_argument('--workers', type=int, default=5); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--no-compile', action='store_true'); ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt', default='on', choices=['on', 'off']); ap.add_argument('--cap-every', type=int, default=500)
    a = ap.parse_args()
    run = HERE / a.run; run.mkdir(parents=True, exist_ok=True)
    dev = torch.device('mps')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tk = TK.Tok()
    model = M.V14(M.Cfg(V=tk.vocab)).to(dev)
    model.set_ckpt(a.ckpt == 'on')
    twin = V.make_twin(model)
    cb = V.Codebook(model.c.d, model.c.pq_S, model.c.pq_N).to(dev)
    maps = V.ResidueMaps(len(V.FAMILIES), 2 * model.c.d, model.c.d).to(dev)
    groups = param_groups(model)
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), eps=1e-8)
    budget = a.hours * 3600.0
    state = dict(step=0, elapsed=0.0, snaps=[], tokens=0, scored=0, lam_scale=1.0, cap_hist=[], frozen=False,
                 p_ok=[False] * (len(V.FAMILIES) + 1))
    stream_state = None
    ck = run / 'latest.pt'
    if ck.exists() and not a.smoke:
        z = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(z['model']); opt.load_state_dict(z['opt']); cb.load_state_dict(z['codebook'])
        maps.load_state_dict(z['maps']); tw_sd = {k: v for k, v in z['twin'].items() if not k.startswith('engram.table')}
        twin.load_state_dict(tw_sd, strict=False)
        state = z['state']; stream_state = z['stream']
        torch.set_rng_state(z['rng_cpu'].cpu())                      # map_location moved it to MPS
        print(f'resumed at step {state["step"]}, elapsed {state["elapsed"] / 3600:.2f} h', flush=True)
    tw, ol = V.twin_pairs(twin, model)
    consts = V.sigreg_consts(dev)
    text = D.TextStream(tk)
    stream = D.Stream(tk, a.B, a.seed, n_workers=a.workers, ahead=3, state=stream_state, text=text)
    twin_fn, step_fn = make_fns(model, twin, consts)
    if not a.no_compile:
        twin_fn = torch.compile(twin_fn, dynamic=False); step_fn = torch.compile(step_fn, dynamic=False)
    hb, hfam = held_batch(tk, a.B)
    logf = open(run / ('smoke.jsonl' if a.smoke else 'log.jsonl'), 'a')
    tot, grp = M.count_params(model)
    if state['step'] == 0:
        meta = dict(params=tot, groups=grp, twin_params=sum(p.numel() for p in tw), B=a.B, T=D.T_ROW, hours=a.hours,
                    seed=a.seed, lr=LR, wd=WD, warm=WARM, clip=CLIP, ncp_lambda=LAMBDA, nu_s=NU_S, hw=V.HW,
                    t_freeze=V.T_FREEZE, mu0=V.MU0, ckpt=a.ckpt, cfg={k: getattr(model.c, k) for k in dir(model.c)
                    if not k.startswith('_') and not callable(getattr(model.c, k))}, tokenizer=tk.hash(), vocab=tk.vocab,
                    compile=not a.no_compile, torch=torch.__version__, workers=a.workers,
                    started=time.strftime('%Y-%m-%d %H:%M:%S'))
        (run / ('smoke_config.json' if a.smoke else 'config.json')).write_text(json.dumps(meta, indent=1))
    print(f'params {tot:,} (twin {sum(p.numel() for p in tw):,}); budget {a.hours} h; B {a.B} x {D.T_ROW}; '
          f'ckpt {a.ckpt}', flush=True)
    t_last_ck = time.time(); t_log = time.time(); tok_log = 0; steps_log = 0; wait_log = 0.0
    snap_every = budget / 10
    model.train()
    while True:
        frac = state['elapsed'] / budget
        if (not a.smoke and frac >= 1.0) or (a.smoke and state['step'] >= a.smoke): break
        t0 = time.time()
        if frac >= V.T_FREEZE and not state['frozen']:
            cb.freeze(); maps.freeze(); state['frozen'] = True
            print(f'step {state["step"]}: twin, normaliser, codebook and residue maps frozen (t = {frac:.3f})', flush=True)
        tw0 = time.time(); fam, b = stream.batch(frac); wait_log += time.time() - tw0
        ids, doc, tgt, w = to_dev(b, dev)
        eidx = model.engram.keys(ids, doc)
        fi = FAM_IDX[fam]
        P_rows = maps.P[fi].expand(a.B, -1, -1)
        lr = lr_at(state['step'], frac); ramp = min(1.0, frac / 0.1)
        lam = LAMBDA * ramp * state['lam_scale']; nu = NU_S * ramp
        for g in opt.param_groups: g['lr'] = lr * g['lr_mult']
        with torch.no_grad():
            tau, valid_t, dl, dG = twin_fn(ids, doc, eidx, P_rows)
            if not state['frozen']:
                cb.update(tau.reshape(-1, tau.shape[-1]), valid_t.reshape(-1).to(tau.dtype))
                maps.update(fi, dl, dG, valid_t)
            codes = cb.assign(cb.normalise(tau))
        om = V.directions(state['step'], model.c.proj, device=dev)
        ntp_sum, ncp, sa, sb, sbu, hst, same = step_fn(ids, doc, eidx, tgt, w, codes, valid_t, P_rows, om)
        ntp = ntp_sum / b['norm']
        loss = ntp + lam * ncp + nu * (sa + sb + sbu)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if a.smoke and state['step'] < 2:
            bad = [n for n, p_ in model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            print('non-finite grads:', len(bad), bad[:20], 'ntp', ntp.item(), 'ncp', ncp.item(), flush=True)
        if not torch.isfinite(gn):                      # guard: skip the update, count it (logged)
            state['skipped'] = state.get('skipped', 0) + 1
            bad = [n for n, p_ in model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            print(f'step {state["step"]}: non-finite gradient in {len(bad)} tensors {bad[:6]}; update skipped', flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        V.ema_update(tw, ol, V.momentum(frac))
        state['step'] += 1; state['tokens'] += b['used']; state['scored'] += b['scored']
        tok_log += b['used']; steps_log += 1
        if state['step'] % 100 == 0 and not state['frozen']:
            ok = maps.solve(state['step'])
            if ok is not None: state['p_ok'] = ok
        cap = None
        if (state['step'] % a.cap_every == 0 and frac >= 0.1 and not a.smoke) or (a.smoke and state['step'] == 3):
            cap = grad_share(model, twin, cb, maps, hb, hfam, dev, lam, consts, state['step'])
            gc.collect(); torch.mps.empty_cache()                  # the eager read grew MPS memory per call (item 18)
            r = cap['ncp'][0] / max(cap['ntp'][0], 1e-12)
            state['cap_hist'].append(r)
            if len(state['cap_hist']) >= 2 and state['cap_hist'][-1] > 1 and state['cap_hist'][-2] > 1 \
                    and state['lam_scale'] > 1 / 8:
                state['lam_scale'] /= 2; state['cap_hist'] = []
                print(f'step {state["step"]}: gradient share above 1 twice; lambda scale -> {state["lam_scale"]}', flush=True)
        if state['step'] % 100 == 0 and torch.mps.driver_allocated_memory() > 8 * 2 ** 30:
            before = torch.mps.driver_allocated_memory()
            torch.mps.synchronize()
            gc.collect()
            torch.mps.empty_cache()
            print(f'step {state["step"]}: MPS cache maintenance '
                  f'{before / 2 ** 30:.2f} -> {torch.mps.driver_allocated_memory() / 2 ** 30:.2f} GB '
                  f'(live tensors {torch.mps.current_allocated_memory() / 2 ** 30:.2f} GB)', flush=True)
        if state['step'] % a.log_every == 0 or state['step'] <= 3 or a.smoke or cap is not None:
            torch.mps.synchronize()
            dt = time.time() - t_log
            H = hst.cpu().tolist()
            rec = dict(step=state['step'], h=round(state['elapsed'] / 3600, 4), frac=round(frac, 4), fam=fam, lr=lr,
                       lam=lam, nu=nu, mu=V.momentum(frac), tok_s=round(tok_log / max(dt, 1e-9)),
                       s_step=round(dt / max(steps_log, 1), 3), wait_s=round(wait_log / max(steps_log, 1), 3),
                       ntp=ntp.item(), ncp=ncp.item(), sig_a=sa.item(), sig_b=sb.item(), sig_bu=sbu.item(),
                       gn=gn.item(), hor=[[round(x, 4) for x in r_] for r_ in H], same4=round(same.item(), 4),
                       ppl_batch=[round(x, 2) for x in V.batch_perplexity(codes, valid_t, cb.N).cpu().tolist()],
                       ppl_ema=[round(x, 2) for x in cb.usage_perplexity().cpu().tolist()], resets=cb.resets.item(),
                       p_ok=state['p_ok'], lam_scale=state['lam_scale'],
                       mem_gb=round(torch.mps.driver_allocated_memory() / 2 ** 30, 2),
                       mem_live_gb=round(torch.mps.current_allocated_memory() / 2 ** 30, 2),
                       beta=[round(float(x), 3) for x in model.beta.softmax(0)], tokens=state['tokens'],
                       skipped=state.get('skipped', 0))
            if cap is not None: rec['cap'] = cap
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            if state['step'] % (a.log_every * 4) == 0 or state['step'] <= 3 or a.smoke:
                h1 = H[0]
                print(f'step {state["step"]} {state["elapsed"] / 3600:.2f}h {fam:5s} ntp {rec["ntp"]:.3f} ncp '
                      f'{rec["ncp"]:.2f} acc1 a/b {h1[1]:.2f}/{h1[2]:.2f} cc {h1[3]:.2f}/{h1[4]:.2f} sig '
                      f'{rec["sig_a"]:.4f}/{rec["sig_b"]:.4f}/{rec["sig_bu"]:.4f} lam {lam:.3f} {rec["tok_s"]} tok/s '
                      f'{rec["s_step"]}s/step wait {rec["wait_s"]}s mem {rec["mem_gb"]}GB', flush=True)
            t_log = time.time(); tok_log = 0; steps_log = 0; wait_log = 0.0
        state['elapsed'] += time.time() - t0
        k_snap = int(state['elapsed'] // snap_every)
        if not a.smoke and k_snap > len(state['snaps']) and k_snap <= 10:
            p = run / f'snap_{k_snap * 10:03d}.pt'
            save_atomic(dict(model=model.state_dict(), twin=twin.state_dict(), codebook=cb.state_dict(),
                             maps=maps.state_dict(), step=state['step'], elapsed=state['elapsed'],
                             tokens=state['tokens'], lam_scale=state['lam_scale']), p)
            state['snaps'].append(p.name)
        if not a.smoke and (time.time() - t_last_ck > 1800):
            save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), twin=twin.state_dict(),
                             codebook=cb.state_dict(), maps=maps.state_dict(), state=state, stream=stream.state(),
                             rng_cpu=torch.get_rng_state()), ck)
            t_last_ck = time.time()
            (run / 'family_tokens.json').write_text(json.dumps(stream.tokens))
    if not a.smoke:
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), twin=twin.state_dict(), codebook=cb.state_dict(),
                         maps=maps.state_dict(), state=state, stream=stream.state(), rng_cpu=torch.get_rng_state()), ck)
        save_atomic(dict(model=model.state_dict(), twin=twin.state_dict(), codebook=cb.state_dict(),
                         maps=maps.state_dict(), step=state['step'], elapsed=state['elapsed'], tokens=state['tokens'],
                         lam_scale=state['lam_scale']), run / 'final.pt')
        (run / 'family_tokens.json').write_text(json.dumps(stream.tokens))
        (run / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    stream.close()
    print('finished', state['step'], 'steps', f'{state["tokens"]:,}', 'tokens', flush=True)


if __name__ == '__main__':
    main()
