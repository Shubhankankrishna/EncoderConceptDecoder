"""S3 continued pretraining (draft §7.2) from the promoted S2 model.

Arms: dec (S3a-dec: decoder blocks, nf, gamma), enc (S3a-enc, optional: + encoder, enc_norm, tied embedding),
joint (S3b: everything; WTA and selector stay on; distillation at a constant delta = .1 to the ORIGINAL teacher's
code-free outputs computed from the teacher's own encoder on the same tokens).
The S2 objective is reused unchanged (train_s2.objective, distillation off); S3b adds its own distillation term.
"""
import argparse, json, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
import v9paths
from v9paths import digest, files_hash, CODE, DATA, write_json
from model import chunk_info
from runtime import DEVICE, sync, seed_all, atomic_json, atomic_save, Log, lr_at, optimizer, balanced, load_system, implementation_hash as frozen_hash
from model import freeze
from data import Mixer
import hncp, train_s2

S3_FILES = [CODE / n for n in ('train_s3.py', 'train_s2.py', 'hncp.py', 'v9paths.py')]
DELTA_S3B = .1
LR = 3e-4      # V8's lowest fine-tuning rate (CL stage); S3 continues an already trained system (I14)


def ownership(hs, arm):
    freeze(hs); f = hs.f
    mods = {'dec': [f.dec, f.nf], 'enc': [f.dec, f.nf, f.enc, f.enc_norm, f.emb], 'joint': [hs]}[arm]
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(True)
    if arm in ('dec', 'enc'):
        f.gamma.requires_grad_(True)
    return [n for n, p in hs.named_parameters() if p.requires_grad]


def teacher_distill(hs, teacher, b):
    """S3b: student code-free (u0, A0, chat0) on its own clean concepts vs the original teacher on ITS encoder."""
    x, docs, dom = b['x'], b['docs'], b['domain']
    with torch.no_grad():
        hT, cT, cdT = teacher.context(x, docs); uT = teacher.foundation.concept(cT, cdT)
        old = (uT, teacher.foundation.port(uT), teacher.foundation.pq(uT))
    h, c, cd, valid = hs.context(x, docs)
    H = hs.heads(c, cd, need_codes=False)
    per = ((H['u0'] - old[0]).square().mean(-1) + (H['a0'] - old[1]).square().mean(-1) + (H['chat0'] - old[2]).square().mean(-1)) / 3
    return balanced(per * valid, valid, dom)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arm', required=True, choices=['dec', 'enc', 'joint']); ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--p-cl', type=float, default=.25); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--seconds', type=float, default=43200)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--compiled', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4); seed_all(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hs, src = hncp.load(args.source, DEVICE)
    s2cfg = src.get('config', {})
    if s2cfg.get('teacher_sha256') and s2cfg['teacher_sha256'] != digest(args.teacher):
        raise RuntimeError('teacher differs from the S2 model\'s recorded teacher (contract 9)')
    teacher, _ = load_system(args.teacher, DEVICE); freeze(teacher)
    owned = ownership(hs, args.arm)
    # Same data order across arms at one seed (§7.2).
    mix = Mixer(84000 + args.seed, args.batch, 'foundation'); gen = torch.Generator().manual_seed(85000 + args.seed)
    opt = optimizer(hs, LR, 'REFIT')
    cfg = dict(steps=args.steps, k=hs.k, m=s2cfg.get('m', 2), p_cl=args.p_cl, variant='detached', alpha=s2cfg.get('alpha', .1), eps=s2cfg.get('eps', .05),
               lam_sel=s2cfg.get('lam_sel', .1), no_distill=True, context_grad=args.arm in ('enc', 'joint'))
    if args.compiled:
        cfg['decode_fn'] = torch.compile(hs.f.decode)
        if args.arm == 'joint':
            cfg['heads_fn'] = torch.compile(hs.heads)
    config = dict(vars(args), stage='S3', device=DEVICE, lr=LR, delta=DELTA_S3B if args.arm == 'joint' else 0., source_sha256=digest(args.source), teacher_sha256=digest(args.teacher),
                  v9_s3_files_sha256=files_hash(S3_FILES), frozen_implementation_sha256=frozen_hash(), data_sha256=digest(DATA / 'manifest.json'), metadata=hs.metadata(),
                  trainable_parameters=sum(p.numel() for p in hs.parameters() if p.requires_grad), trainable_modules=sorted({n.rsplit('.', 1)[0] for n in owned})[:60],
                  s2_config={k: s2cfg.get(k) for k in ('k', 'm', 'core', 'seed', 'teacher')}, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    start = 0; committed = dict(text=0, symbolic=0)
    if args.resume:
        ck = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if any(ck['config'].get(k) != config.get(k) for k in ('arm', 'steps', 'seed', 'p_cl', 'source_sha256', 'v9_s3_files_sha256')):
            raise RuntimeError('resume setting mismatch')
        hs.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); mix.restore(ck['mixer']); gen.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng']); start = ck['step']; committed = ck['committed']
    write_json(out / 'config.json', config); log = Log(out / 'log.jsonl')

    def save(step, status):
        sync(); atomic_save(out / 'checkpoint.pt', dict(metadata=hs.metadata(), model=hs.state_dict(), optimizer=opt.state_dict(), step=step, mixer=mix.state(), generator=gen.get_state(),
                                                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), config=config, committed=committed, status=status))
    save(start, 'initialized')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    begin = time.monotonic(); last_save = begin; step = start; last = None; trainable = [p for p in hs.parameters() if p.requires_grad]
    try:
        for index in range(start, args.steps):
            if stop[0] or (index > start and time.monotonic() - begin >= args.seconds):
                break
            cpu_b = mix.batch('cpu'); b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in cpu_b.items()}
            for group in opt.param_groups:
                group['lr'] = lr_at(index, args.steps, LR, 200) * group.get('lr_mult', 1)
            opt.zero_grad(set_to_none=True)
            loss, logs = train_s2.objective(hs, teacher, b, cfg, index, gen, None, diagnostics=False, cpu=cpu_b)
            if args.arm == 'joint':
                dist = teacher_distill(hs, teacher, b); loss = loss + DELTA_S3B * dist; logs.update(distillation=dist, delta=DELTA_S3B)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite S3 objective; update prevented')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True); opt.step(); step = index + 1
            # Committed-plan exposure: scored chunks per domain (every scored chunk carries a committed plan at K > 1).
            cm = train_s2.ncp_weights(cpu_b) > 0
            committed['symbolic'] += int(cm[cpu_b['domain'] == 1].sum()) if hs.k > 1 else 0; committed['text'] += int(cm[cpu_b['domain'] == 0].sum()) if hs.k > 1 else 0
            if step == 1 or step % 100 == 0 or step == args.steps:
                sync(); last = dict(step=step, planned_steps=args.steps, loss=float(loss.detach()), gradient_norm=float(norm), elapsed_s=time.monotonic() - begin,
                                    **{k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in logs.items()})
                log(**last); atomic_json(out / 'live.json', last); print(json.dumps(last), flush=True)
            if step % 2000 == 0 or time.monotonic() - last_save >= 600:
                save(step, 'training'); last_save = time.monotonic()
        status = 'interrupted_by_signal' if stop[0] else 'completed' if step == args.steps else 'partial_time_budget'
        save(step, status)
        write_json(out / 'train_summary.json', dict(**config, status=status, completed_steps=step, wall_s=time.monotonic() - begin, last=last, committed_plan_chunks=committed))
    except Exception as exc:
        write_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), step=step)); raise


if __name__ == '__main__':
    main()
