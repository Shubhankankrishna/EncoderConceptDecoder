"""S2 training (draft §6.3-§6.5): H-NCP against a frozen S1 teacher, V8 REFIT ownership and optimizer.

L_S2 = L_token(plan) + alpha * L_WTA + delta(u) * L_dist + lambda_sel * L_sel. At K = 1 with the closed loop off
this is V8's REFIT objective term for term (contract 1).
"""
import argparse, json, math, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import digest, files_hash, CODE, DATA, write_json
from model import chunk_info, freeze, count, S, CODES, D
from runtime import DEVICE, sync, seed_all, atomic_json, atomic_save, Log, lr_at, optimizer, nll, balanced, ncp_weights, implementation_hash as frozen_hash
from data import Mixer
import hncp

S2_FILES = [CODE / n for n in ('hncp.py', 'train_s2.py', 'v9paths.py')]


def distill_weight(index, steps, no_distill=False):
    """V8 REFIT schedule (train.py:128): 1 for the first 10%, linear to 0 by 30%."""
    if no_distill:
        return 0.
    frac = index / steps
    return 1. if frac < .1 else max(0., 1 - (frac - .1) / .2)


def snap(f, v):
    """Replace each PQ segment with its nearest codebook entry (closed-loop 'snap' arm)."""
    seg = v.view(*v.shape[:-1], S, D // S)
    dist = (seg.unsqueeze(-2) - f.Q).square().sum(-1)
    idx = dist.argmin(-1)
    return f.Q[torch.arange(S, device=v.device), idx].reshape(v.shape)


def kstar_prediction(hs, H, c, src, rows):
    """Committed prediction at source positions: chat^(k*) with k* = argmin distance to the true next concept."""
    if hs.codes is None:
        return H['chat0'][rows, src]
    pred = H['chatk'][rows, src]                         # (r, K, D)
    d = (pred - c[rows, src + 1][:, None]).square().mean(-1)
    return pred[torch.arange(len(rows), device=pred.device), d.argmin(-1)]


def choose_spans(cd, cm, gen, p_cl):
    """Per sequence with probability p_cl: a scored source chunk j0 uniformly, N ~ U{1..4}, truncated at the document end."""
    B, M = cd.shape; cdc = cd.cpu(); cmc = cm.cpu() > 0; spans = []
    for r in range(B):
        if float(torch.rand((), generator=gen)) >= p_cl:
            continue
        elig = cmc[r].nonzero().flatten()
        if len(elig) == 0:
            continue
        j0 = int(elig[int(torch.randint(len(elig), (), generator=gen))]); n = int(torch.randint(1, 5, (), generator=gen))
        n_ok = 0
        for t in range(1, n + 1):
            j = j0 + t
            if j > M - 1 or cdc[r, j] != cdc[r, j0] or cdc[r, j] < 0:
                break
            n_ok = t
        if n_ok:
            spans.append((r, j0, n_ok))
    return spans


def closed_loop_inputs(hs, c, cd, spans, variant='detached'):
    """§6.5: replace C_j for j0 < j <= j0+N with the committed prediction from position j-1, sequentially.

    The core is causal and document-isolated per row, so each rollout step runs the core only on the rows that
    have a span, truncated to the longest prefix needed; results equal a full-batch pass (contract-tested)."""
    if not spans:
        return c, 0
    c_in = c.clone(); steps = max(n for _, _, n in spans)
    span_rows = sorted({r for r, _, _ in spans}); pos = {r: i for i, r in enumerate(span_rows)}
    sub_rows = torch.tensor(span_rows, device=c.device)
    for t in range(1, steps + 1):
        act = [(r, j0) for r, j0, n in spans if n >= t]
        rows = torch.tensor([r for r, _ in act], device=c.device); src = torch.tensor([j0 + t - 1 for _, j0 in act], device=c.device)
        width = int(src.max()) + 1
        local = torch.tensor([pos[r] for r, _ in act], device=c.device)
        if variant == 'undetached':
            H = hs.heads(c_in[sub_rows, :width], cd[sub_rows, :width]); pred = kstar_prediction(hs, H, c[sub_rows], src, local)
            c_in = c_in.index_put((rows, src + 1), pred)
        else:
            with torch.no_grad():
                H = hs.heads(c_in[sub_rows, :width], cd[sub_rows, :width]); pred = kstar_prediction(hs, H, c[sub_rows], src, local)
                if variant == 'snap':
                    pred = snap(hs.f, pred)
            c_in = c_in.index_put((rows, src + 1), pred.detach())
    return c_in, sum(n for _, _, n in spans)


def objective(hs, teacher, b, cfg, index, gen, acc=None, diagnostics=False, cpu=None):
    """cpu: the same batch on the CPU, used only to draw closed-loop spans and sampling noise without a device sync."""
    x, docs, m, dom = b['x'], b['docs'], b['m'], b['domain']
    zero = x.new_zeros((), dtype=torch.float32)
    delta = distill_weight(index, cfg['steps'], cfg.get('no_distill', False))
    # The encoder is frozen in S2, so its pass runs without autograd; S3 arms that train it set context_grad.
    with torch.set_grad_enabled(bool(cfg.get('context_grad'))):
        h, c, cd, valid = hs.context(x, docs)
    with torch.no_grad():
        if delta:
            uT = teacher.foundation.concept(c, cd); old = (uT, teacher.foundation.port(uT), teacher.foundation.pq(uT))
    cm = ncp_weights(b)
    if cfg['p_cl'] > 0:
        src = cpu if cpu is not None else b
        spans = choose_spans(chunk_info(src['docs'])[0], ncp_weights(src), gen, cfg['p_cl'])
    else:
        spans = []
    c_in, rolled = closed_loop_inputs(hs, c, cd, spans, cfg.get('variant', 'detached'))
    H = (cfg.get('heads_fn') or hs.heads)(c_in, cd)
    target = c[:, 1:].detach()
    logs = {}
    if hs.codes is None:
        d1 = (H['chat0'][:, :-1] - target).square().mean(-1)
        ncp = balanced(d1 * cm, cm, dom)[0]
        chat, a = H['chat0'], H['a0']
        sel = None
    else:
        K = hs.k
        d = (H['chatk'][:, :-1] - target[:, :, None]).square().mean(-1)            # (B, M-1, K)
        kstar = d.argmin(-1)
        dmin = d.gather(-1, kstar[..., None]).squeeze(-1)
        wta = dmin + cfg['eps'] * (d.sum(-1) - dmin) / (K - 1)
        ncp = balanced(wta * cm, cm, dom)[0]
        with torch.no_grad():
            tau = (.1 * d.mean(-1, keepdim=True)).clamp_min(1e-4)
            r = (-d / tau).softmax(-1)
        logpi = H['pi_logits'][:, :-1].float().log_softmax(-1)
        kl = (torch.xlogy(r, r) - r * logpi).sum(-1)
        sel = balanced(kl * cm, cm, dom)[0]
        # Best-of-m within a prior-drawn subset (§6.3); unscored chunks keep the code-free plan.
        with torch.no_grad():
            probs = logpi.exp().clamp_min(1e-6); probs = probs / probs.sum(-1, keepdim=True)
            mm = min(cfg['m'], K)
            # Gumbel-top-m = m draws from probs without replacement; noise drawn on the CPU and copied without a sync.
            u = torch.rand(probs.shape, generator=gen).clamp(1e-12, 1 - 1e-7).to(d.device, non_blocking=True)
            subset = (probs.log() - (-u.log()).log()).topk(mm, -1).indices
            sub_d = d.gather(-1, subset)
            kdec = subset.gather(-1, sub_d.argmin(-1, keepdim=True)).squeeze(-1)
            codes = torch.full(cd.shape, -1, dtype=torch.long, device=x.device)
            codes[:, :-1] = torch.where(cm > 0, kdec, torch.full_like(kdec, -1))
        chat, a = hs.plan(H, codes)
        if acc is not None:
            sc = cm > 0
            acc['wins_dev'] = acc.get('wins_dev', 0) + F.one_hot(kstar, K).mul(sc[..., None]).sum((0, 1))
        if diagnostics:
            sc = cm > 0
            with torch.no_grad():
                ent = -(logpi.exp() * logpi).sum(-1)
                ck = H['chatk'][:, :-1][sc]
                pair = (ck[:, :, None] - ck[:, None]).square().mean(-1).sum((1, 2)) / (K * (K - 1))
                logs.update(selector_entropy=float(ent[sc].mean()), hypothesis_pair_distance=float(pair.mean()),
                            kdec_is_kstar=float((kdec == kstar)[sc].float().mean()), d_min=float(dmin[sc].mean()), d_mean=float(d.mean(-1)[sc].mean()))
    logits = (cfg.get('decode_fn') or hs.f.decode)(h, chat, a, docs)
    tok = nll(logits, x, m)
    ntp, domains, _ = balanced(tok, m[:, :-1], dom)
    dist = zero
    if delta:
        u0, a0, chat0 = H['u0'], H['a0'], H['chat0']
        dist = balanced(((u0 - old[0]).square().mean(-1) + (a0 - old[1]).square().mean(-1) + (chat0 - old[2]).square().mean(-1)) / 3 * valid, valid, dom)[0]
    loss = ntp + cfg['alpha'] * ncp + zero + delta * dist
    if sel is not None:
        loss = loss + cfg['lam_sel'] * sel
    logs.update(text=domains[0], symbolic=domains[1], ncp=ncp, distillation=dist, selector=sel if sel is not None else zero, delta=delta, rolled_positions=rolled, spans=len(spans))
    return loss, logs


def flush_wins(acc):
    if 'wins_dev' in acc and torch.is_tensor(acc['wins_dev']):
        acc['wins'] += acc.pop('wins_dev').long().cpu()


def restart_dead(hs, opt, acc, gen, log):
    """§6.4: dead if win share over the last 500 updates < 1/(10K); split the most-winning code; zero optimizer state."""
    flush_wins(acc); wins = acc['wins'].float(); total = wins.sum()
    if hs.codes is None or total == 0:
        acc['wins'].zero_(); return []
    share = wins / total; K = hs.k
    dead = (share < 1 / (10 * K)).nonzero().flatten().tolist(); top = int(share.argmax())
    with torch.no_grad():
        for kd in dead:
            hs.codes.e[kd] = hs.codes.e[top] + .1 * torch.randn(hs.codes.e.shape[1], generator=gen).to(hs.codes.e.device)
        st = opt.state.get(hs.codes.e, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in st and dead:
                st[key][dead] = 0
    log(event='code_restart', shares=share.tolist(), dead=dead, top=top)
    acc['wins'].zero_()
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True); ap.add_argument('--out', required=True); ap.add_argument('--core', default='dense', choices=['dense', 'recurrent'])
    ap.add_argument('--k', type=int, default=1); ap.add_argument('--m', type=int, default=2); ap.add_argument('--p-cl', type=float, default=.25)
    ap.add_argument('--variant', default='detached', choices=['detached', 'undetached', 'snap']); ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=911); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--context', type=int, default=192)
    ap.add_argument('--lr', type=float, default=.001); ap.add_argument('--alpha', type=float, default=.1); ap.add_argument('--eps', type=float, default=.05)
    ap.add_argument('--lam-sel', type=float, default=.1); ap.add_argument('--seconds', type=float, default=43200); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-distill', action='store_true'); ap.add_argument('--compiled', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4); seed_all(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hs, teacher = hncp.build(args.teacher, args.core, args.k, DEVICE)
    mix = Mixer(81000 + args.seed, args.batch, 'foundation', args.context); gen = torch.Generator().manual_seed(83000 + args.seed)
    opt = optimizer(hs, args.lr, 'REFIT')
    cfg = dict(steps=args.steps, k=args.k, m=args.m, p_cl=args.p_cl, variant=args.variant, alpha=args.alpha, eps=args.eps, lam_sel=args.lam_sel, no_distill=args.no_distill)
    if args.compiled:  # fixed-shape main pass only; closed-loop rollout passes stay eager (contract: eager/compiled parity)
        cfg['heads_fn'] = torch.compile(hs.heads); cfg['decode_fn'] = torch.compile(hs.f.decode)
    config = dict(vars(args), stage='S2', device=DEVICE, precision='float32', teacher_sha256=digest(args.teacher), v9_s2_files_sha256=files_hash(S2_FILES),
                  frozen_implementation_sha256=frozen_hash(), data_sha256=digest(DATA / 'manifest.json'), metadata=hs.metadata(),
                  trainable_parameters=sum(p.numel() for p in hs.parameters() if p.requires_grad), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    acc = dict(wins=torch.zeros(max(args.k, 1), dtype=torch.long)); history = []; start = 0
    if args.resume:
        ck = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for key in vars(args):
            if key != 'resume' and ck['config'].get(key) != config.get(key):
                raise RuntimeError('resume setting mismatch ' + key)
        if ck['config']['v9_s2_files_sha256'] != config['v9_s2_files_sha256']:
            raise RuntimeError('resume source mismatch')
        hs.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); mix.restore(ck['mixer']); gen.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        start = ck['step']; acc = ck['acc']; history = ck['restarts']
    write_json(out / 'config.json', config); log = Log(out / 'log.jsonl')
    def save(step, status):
        sync(); flush_wins(acc); atomic_save(out / 'checkpoint.pt', dict(metadata=hs.metadata(), model=hs.state_dict(), optimizer=opt.state_dict(), step=step, mixer=mix.state(),
                                                        generator=gen.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                                                        config=config, acc=acc, restarts=history, status=status))
    save(start, 'initialized')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    begin = time.monotonic(); last_save = begin; step = start; last = None
    trainable = [p for p in hs.parameters() if p.requires_grad]
    try:
        for index in range(start, args.steps):
            if stop[0] or (index > start and time.monotonic() - begin >= args.seconds):
                break
            cpu_b = mix.batch('cpu'); b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in cpu_b.items()}
            for group in opt.param_groups:
                group['lr'] = lr_at(index, args.steps, args.lr, 200) * group.get('lr_mult', 1)
            opt.zero_grad(set_to_none=True)
            diag = index + 1 == 1 or (index + 1) % 100 == 0 or index + 1 == args.steps
            loss, logs = objective(hs, teacher, b, cfg, index, gen, acc, diagnostics=diag, cpu=cpu_b)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite S2 objective; update prevented')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True); opt.step(); step = index + 1
            if hs.codes is not None and step % 500 == 0 and step < .8 * args.steps:
                history.append(dict(step=step, dead=restart_dead(hs, opt, acc, gen, lambda **o: log(step=step, **o))))
            if step == 1 or step % 100 == 0 or step == args.steps:
                sync(); dt = time.monotonic() - begin
                last = dict(step=step, planned_steps=args.steps, loss=float(loss.detach()), gradient_norm=float(norm), elapsed_s=dt, lr=opt.param_groups[0]['lr'],
                            **{k: (float(v.detach()) if isinstance(v, torch.Tensor) else v) for k, v in logs.items()})
                if hs.codes is not None:
                    flush_wins(acc); w = acc['wins'].float(); last['win_share'] = (w / w.sum().clamp_min(1)).tolist()
                log(**last); atomic_json(out / 'live.json', last); print(json.dumps(last), flush=True)
            if step % 2000 == 0 or time.monotonic() - last_save >= 600:
                save(step, 'training'); last_save = time.monotonic()
        status = 'interrupted_by_signal' if stop[0] else 'completed' if step == args.steps else 'partial_time_budget'
        save(step, status)
        write_json(out / 'train_summary.json', dict(**config, status=status, completed_steps=step, wall_s=time.monotonic() - begin, last=last, restarts=history,
                                                    lifetime_symbolic_exposure=mix.seen_tasks, scored_targets_by_task=mix.targets, frozen_teacher_preserved=True))
    except Exception as exc:
        write_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), step=step)); raise


if __name__ == '__main__':
    main()
