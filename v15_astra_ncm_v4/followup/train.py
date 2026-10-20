"""Matched continuation versus exposure to actual model-generated prefixes.

Only train-split plans supply gradient targets. Generation never receives target
codes. The codec and original 97M generator remain frozen. Validation selects
checkpoints by free route fidelity, not teacher-conditioned likelihood alone.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time

import torch
import torch.nn.functional as F

from v15_astra_ncm_v4.adapt import condition_for, parse_path, tensor
from v15_astra_ncm_v4.cache import load_cache
from v15_astra_ncm_v4.common import (ROOT, WORKSPACE, atomic_json, append_json, heavy_job,
    resource_issue, restore_rng, rng_state, save_torch, seed_all, sha, source_hashes, runtime_identity)
from v15_astra_ncm_v4.model import PlanConditioner
from v15_astra_ncm_v4.ncm import load_ncm, decoded_text

PARENT = ROOT / 'runs/20261019_lm1b_grounded'
DEFAULT_RUN = ROOT / 'runs/20261019_renderer_prefix_followup'
STOP = False


def stop(*_):
    global STOP
    STOP = True


def route_similarity(pred, gold):
    """One minus normalized edit distance, retaining insertions/deletions."""
    previous = list(range(len(gold) + 1))
    for i, value in enumerate(pred, 1):
        current = [i]
        for j, expected in enumerate(gold, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (value != expected)))
        previous = current
    return 1 - previous[-1] / max(len(pred), len(gold), 1)


@torch.no_grad()
def generated_prefix_input(net, condition, target_level):
    """Build the next input from predictions only; target codes are not accepted."""
    q = net.codec[0].model.vq
    raw = net.base.build_l0_query_input(len(condition)).detach().cpu()
    zhat = torch.zeros(len(condition), 128, 384)
    for level in range(target_level):
        predicted = net.level_logits(raw, level, condition).argmax(-1).cpu()
        zhat, raw = q.get_next_autoregressive_input(level, zhat, q.embeddings[level](predicted))
    return raw, zhat


@torch.no_grad()
def residual_target(codec, tokens, prefix_reconstruction, level):
    """Re-quantize the residual against the ACTUAL earlier reconstruction.

    The original cached residual ID is wrong after changing earlier codes. Gold
    tokens supply this supervised target only; they never enter generated_prefix_input.
    """
    q = codec.model.vq
    latent = codec.model._encode_pre_quant(tokens.cpu().long()).float()
    rest = latent - prefix_reconstruction
    reduced = q._down_for_quant(rest, level, codec.levels[level]) if level < 11 else rest
    return q._distances(reduced.reshape(-1, q.embed_dim), q.embeddings[level].weight).argmin(1).view(len(tokens), codec.levels[level])


def sources():
    return source_hashes() | {str(p.relative_to(WORKSPACE)): sha(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))}


def build():
    net, codec, inference = load_ncm('mps')
    encoder = PlanConditioner().to('mps')
    params = [p for p in net.parameters() if p.requires_grad] + list(encoder.parameters())
    optimizer = torch.optim.AdamW([
        {'params': [p for p in net.parameters() if p.requires_grad], 'lr': 1e-4},
        {'params': encoder.parameters(), 'lr': 3e-4}], weight_decay=.01)
    saved = torch.load(PARENT / 'renderer/latest.pt', map_location='cpu', weights_only=True)
    net.load_adapters(saved['adapters']); encoder.load_state_dict(saved['conditioner'])
    optimizer.load_state_dict(saved['optimizer'])
    return net, encoder, optimizer, params, inference


def update(net, encoder, optimizer, params, cache, arm):
    records = cache['records']; device = net.device
    training = [i for i, r in enumerate(records) if r['kind'] == 'plan' and r['split'] == 'train']
    retention = [i for i, r in enumerate(records) if r['kind'] == 'retention' and r['split'] == 'train']
    net.train(); encoder.train(); optimizer.zero_grad(set_to_none=True)
    total = 0.; actual_prefix_count = 0
    for _ in range(4):
        i = random.choice(training); level = random.randrange(12)
        lo, hi = net.offsets[level:level + 2]
        condition = condition_for(encoder, records, [i], device)
        raw = tensor(cache['inputs'][i:i + 1, lo:hi])
        # One common random draw keeps row/level sampling paired across arms.
        draw = random.random()
        target = tensor(cache['codes'][i:i + 1, lo:hi], device).long()
        if arm == 'on_policy' and level > 0 and draw < .5:
            raw, reconstruction = generated_prefix_input(net, condition.detach(), level)
            target = residual_target(net.codec[0], tensor(cache['tokens'][i:i + 1]), reconstruction, level).to(device)
            actual_prefix_count += 1
        elif arm == 'control' and level > 0 and draw < .2:
            altered = tensor(cache['codes'][i:i + 1]).long()
            noise = torch.rand(altered[:, :lo].shape) < .05
            altered[:, :lo] = torch.where(noise, torch.randint(8192, altered[:, :lo].shape), altered[:, :lo])
            raw = net.teacher_inputs(altered)[:, lo:hi]
        weight = .5 + .5 * 12 * (hi - lo) / 641
        loss = weight * F.cross_entropy(net.level_logits(raw, level, condition).flatten(0, 1), target.flatten()) / 4
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite renderer loss')
        loss.backward(); total += float(loss.detach())
    i = random.choice(retention); level = random.randrange(12); lo, hi = net.offsets[level:level + 2]
    raw = tensor(cache['inputs'][i:i + 1, lo:hi])
    with torch.no_grad(), net.frozen_reference():
        teacher = net.level_logits(raw, level, None).softmax(-1)
    logp = net.level_logits(raw, level, None).log_softmax(-1)
    retain = F.kl_div(logp, teacher, reduction='none').sum(-1).mean()
    if not torch.isfinite(retain): raise RuntimeError('Nonfinite retention loss')
    (.1 * retain).backward()
    grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    return {'weighted_code_loss': total, 'retention_kl': float(retain.detach()),
            'gradient_norm': float(grad), 'generated_prefix_microbatches': actual_prefix_count}


@torch.no_grad()
def evaluate(net, encoder, cache, inference, step, split='valid', count=48):
    net.eval(); encoder.eval(); records = cache['records']
    # Earlier finalizer used the first24 test records. A remaining test panel is
    # reserved for one comparison after this follow-up, never training/selection.
    selected = [i for i, r in enumerate(records) if r['kind'] == 'plan' and r['split'] == split]
    selected = selected[24:24 + count]
    cases = []; losses = []; retention = []
    for j, i in enumerate(selected):
        condition = condition_for(encoder, records, [i], net.device)
        codes = net.generate_codes(condition, temperature=0)
        text = decoded_text(net.codec[0], codes, inference)[0]; path = parse_path(text)
        wrong_condition = condition_for(encoder, records, [selected[(j + 1) % len(selected)]], net.device)
        wrong_text = decoded_text(net.codec[0], net.generate_codes(wrong_condition, temperature=0), inference)[0]
        gold = records[i]['path']
        cases.append({'graph_id': records[i]['graph_id'], 'path': gold, 'text': text,
            'parsed_path': path, 'exact': path == gold, 'route_similarity': route_similarity(path, gold),
            'shuffled_exact': parse_path(wrong_text) == gold,
            'shuffled_route_similarity': route_similarity(parse_path(wrong_text), gold)})
        if j < 24:
            nll = 0.
            for level in range(12):
                lo, hi = net.offsets[level:level + 2]
                target = tensor(cache['codes'][i:i + 1, lo:hi], net.device).long()
                logits = net.level_logits(tensor(cache['inputs'][i:i + 1, lo:hi]), level, condition)
                nll += float(F.cross_entropy(logits.flatten(0, 1), target.flatten())) * (hi - lo) / 641
            losses.append(nll)
    keep = [i for i, r in enumerate(records) if r['kind'] == 'retention' and r['split'] == 'valid'][:8]
    for j, i in enumerate(keep):
        level = (j * 5) % 12; lo, hi = net.offsets[level:level + 2]
        raw = tensor(cache['inputs'][i:i + 1, lo:hi])
        with net.frozen_reference(): teacher = net.level_logits(raw, level, None).softmax(-1)
        logp = net.level_logits(raw, level, None).log_softmax(-1)
        retention.append(float(F.kl_div(logp, teacher, reduction='none').sum(-1).mean()))
    return {'step': step, 'parent_step': 12000, 'timestamp': time.time(), 'split': split,
        'count': len(cases), 'greedy_path_exact': sum(c['exact'] for c in cases) / len(cases),
        'route_similarity': sum(c['route_similarity'] for c in cases) / len(cases),
        'shuffled_route_similarity': sum(c['shuffled_route_similarity'] for c in cases) / len(cases),
        'code_nll': sum(losses) / len(losses), 'retention_kl_probe': sum(retention) / len(retention),
        'samples': cases, 'test_policy': 'Validation selects checkpoints; remaining test panel is evaluated only after both arms.'}


def selection_score(result):
    if result['retention_kl_probe'] >= .15: return (-1., -1., -1.)
    return (result['greedy_path_exact'], result['route_similarity'], -result['code_nll'])


def main(args):
    global STOP
    STOP = False
    torch.set_num_threads(2)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    args.run.mkdir(parents=True, exist_ok=True)
    identity = {'version': 1, 'arm': args.arm, 'sources': sources(), 'runtime': runtime_identity(),
        'parent_checkpoint_sha': sha(PARENT / 'renderer/latest.pt'),
        'cache_manifest_sha': sha(ROOT / 'data/ncm_cache/manifest.json'),
        'seed': args.seed, 'steps': args.steps, 'deadline': args.deadline,
        'optimizer': 'Continue original AdamW moments/LR; no warmup reset',
        'control': '20% sparse uniform-code corruption, otherwise gold prefixes',
        'on_policy': '50% greedy generated prefixes with residual targets re-quantized against actual prefix, otherwise gold prefixes; stop-gradient roll-in',
        'selection': 'Free exact route, then normalized route edit similarity, then code NLL; retention KL < .15',
        'evaluation_every': 500, 'validation_rows': '24:72', 'untouched_test_rows': '24:96'}
    if (args.run / 'plan.json').exists():
        if json.loads((args.run / 'plan.json').read_text()) != identity:
            raise ValueError('Follow-up identity changed')
    else: atomic_json(args.run / 'plan.json', identity)
    cache = load_cache(); net, encoder, optimizer, params, inference = build()
    seed_all(args.seed); step = 0; best = (-1., -1., -1.); elapsed = 0.
    if (args.run / 'latest.pt').exists():
        saved = torch.load(args.run / 'latest.pt', map_location='cpu', weights_only=True)
        if saved['identity'] != identity: raise ValueError('Checkpoint identity changed')
        net.load_adapters(saved['adapters']); encoder.load_state_dict(saved['conditioner'])
        optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        step, best, elapsed = saved['step'], tuple(saved['best']), saved['elapsed']
    atomic_json(args.run / 'process.json', {'pid': os.getpid(), 'timestamp': time.time()})
    began = time.monotonic(); last_save = began

    def save():
        save_torch(args.run / 'latest.pt', {'adapters': net.adapter_state(), 'conditioner': encoder.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'identity': identity,
            'step': step, 'best': best, 'elapsed': elapsed + time.monotonic() - began})

    def validate():
        nonlocal best
        state = rng_state()
        try: result = evaluate(net, encoder, cache, inference, step)
        finally: restore_rng(state)
        atomic_json(args.run / f'eval_{step:07d}.json', result)
        atomic_json(args.run / 'latest_eval.json', result)
        if selection_score(result) > best:
            best = selection_score(result)
            save_torch(args.run / 'best.pt', {'adapters': net.adapter_state(), 'conditioner': encoder.state_dict(),
                'step': step, 'score': best, 'identity': identity})

    def status(reason, metrics=None):
        atomic_json(args.run / 'status.json', {'status': reason, 'step': step, 'target_steps': args.steps,
            'parent_step': 12000, 'timestamp': time.time(), 'pid': os.getpid(),
            'elapsed_seconds': elapsed + time.monotonic() - began, 'metrics': metrics})

    status('initial_evaluation' if step == 0 else 'resuming')
    if step == 0: validate(); save()
    reason = 'complete'
    limit = min(args.steps, args.until_step if args.until_step is not None else args.steps)
    while step < limit:
        if STOP or (args.run.parent / 'STOP').exists(): reason = 'stopped'; break
        if time.time() >= args.deadline: reason = 'deadline'; break
        if step % 25 == 0:
            issue = resource_issue()
            if issue: reason = issue; break
        metrics = update(net, encoder, optimizer, params, cache, args.arm); step += 1
        if step % 10 == 0:
            append_json(args.run / 'metrics.jsonl', {'step': step, 'timestamp': time.time(), **metrics})
            status('training', metrics)
        if step % 500 == 0: validate(); save(); last_save = time.monotonic()
        elif time.monotonic() - last_save >= 120: save(); last_save = time.monotonic()
    if reason == 'complete' and step < args.steps: reason = 'chunk_complete'
    save(); status(reason)
    if reason in ('low_memory', 'low_disk', 'low_battery'): raise RuntimeError(reason)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--arm', choices=('control', 'on_policy'), required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--until-step', type=int)
    ap.add_argument('--seed', type=int, default=119)
    ap.add_argument('--deadline', type=float, required=True)
    args = ap.parse_args()
    try:
        with heavy_job(): main(args)
    except Exception as error:
        atomic_json(args.run / 'status.json', {'status': 'failed', 'error': repr(error), 'timestamp': time.time()})
        raise
