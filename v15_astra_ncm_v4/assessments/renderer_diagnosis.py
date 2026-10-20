"""Validation-only interventions on the completed renderer; no weight updates."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from v15_astra_ncm_v4.adapt import condition_for, parse_path, tensor
from v15_astra_ncm_v4.cache import load_cache
from v15_astra_ncm_v4.common import ROOT, atomic_json, heavy_job, seed_all, sha
from v15_astra_ncm_v4.model import PlanConditioner
from v15_astra_ncm_v4.ncm import load_ncm, decoded_text

RUN = ROOT / 'runs/20261019_lm1b_grounded'
OUT = ROOT / 'reports/renderer_diagnosis.json'


@torch.no_grad()
def clamped_codes(net, cond, gold, levels):
    q = net.codec[0].model.vq
    zhat = torch.zeros(len(gold), 128, 384)
    raw = net.base.build_l0_query_input(len(gold)).detach().cpu()
    result = []
    for level in range(12):
        lo, hi = net.offsets[level:level + 2]
        ids = gold[:, lo:hi] if level < levels else net.level_logits(raw, level, cond).argmax(-1).cpu()
        result.append(ids)
        if level < 11:
            zhat, raw = q.get_next_autoregressive_input(level, zhat, q.embeddings[level](ids))
    return torch.cat(result, 1)


def bounded_plain(codec, codes):
    rows = codec.decode_logits(codes).argmax(-1).tolist()
    sep = codec.bpe.encode('')[-1]
    cleaned = []
    for row in rows:
        if row and row[0] == codec.eos:
            row = row[1:]
        stop = next((i for i, n in enumerate(row) if n in (codec.eos, codec.pad, sep)), len(row))
        cleaned.append(row[:stop])
    return [codec.bpe.decode(row, skip_special_tokens=True) for row in cleaned]


@torch.no_grad()
def main():
    torch.set_num_threads(2); seed_all(948)
    cache = load_cache(); net, codec, inference = load_ncm('mps')
    encoder = PlanConditioner().to('mps')
    ckpt = torch.load(RUN / 'renderer/best.pt', map_location='cpu', weights_only=True)
    net.load_adapters(ckpt['adapters']); encoder.load_state_dict(ckpt['conditioner'])
    net.eval(); encoder.eval()
    indices = [i for i, r in enumerate(cache['records']) if r['kind'] == 'plan' and r['split'] == 'valid'][:24]
    stats = {str(k): [] for k in (0, 1, 3, 6, 9, 12)}
    teacher_accuracy = [[] for _ in range(12)]
    cases = []; began = time.monotonic()
    for j, i in enumerate(indices):
        record = cache['records'][i]
        cond = condition_for(encoder, cache['records'], [i], 'mps')
        gold = tensor(cache['codes'][i:i + 1]).long()
        teacher_codes = []
        for level in range(12):
            lo, hi = net.offsets[level:level + 2]
            pred = net.level_logits(tensor(cache['inputs'][i:i + 1, lo:hi]), level, cond).argmax(-1).cpu()
            teacher_accuracy[level].append(float((pred == gold[:, lo:hi]).float().mean()))
            teacher_codes.append(pred)
        outputs = {'teacher_argmax': torch.cat(teacher_codes, 1)}
        outputs.update({str(k): clamped_codes(net, cond, gold, k) for k in (0, 1, 3, 6, 9, 12)})
        case = {'index': i, 'reference': record['text'], 'path': record['path'], 'outputs': {}}
        for mode, codes in outputs.items():
            plain = decoded_text(codec, codes, inference)[0]
            bounded = bounded_plain(codec, codes)[0]
            match = parse_path(plain) == record['path']
            case['outputs'][mode] = {'text': plain, 'boundary_text': bounded, 'exact_path': match,
                'bounded_exact_path': parse_path(bounded) == record['path'],
                'code_accuracy': float((codes == gold).float().mean())}
            if mode in stats: stats[mode].append(match)
        cases.append(case)
        print(f'diagnosed {j+1}/{len(indices)}', flush=True)
    report = {'checkpoint_sha': sha(RUN / 'renderer/best.pt'), 'step': ckpt['step'],
        'split': 'valid', 'count': len(indices), 'teacher_argmax_code_accuracy_by_level':
        [sum(v) / len(v) for v in teacher_accuracy],
        'path_exact_by_gold_prefix_levels': {k: sum(v) / len(v) for k, v in stats.items()},
        'teacher_argmax_path_exact': sum(c['outputs']['teacher_argmax']['exact_path'] for c in cases) / len(cases),
        'cases': cases, 'seconds': time.monotonic() - began,
        'limitation': 'Gold-prefix interventions are diagnosis only, not deployable inference or a new held-out score.'}
    atomic_json(OUT, report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}), flush=True)


if __name__ == '__main__':
    with heavy_job(): main()
