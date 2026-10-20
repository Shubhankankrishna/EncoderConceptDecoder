"""Actual MPS update/resume, frozen-weight and residual-target consistency tests."""
import copy
import hashlib
import json
import time

import torch

from v15_astra_ncm_v4.adapt import condition_for, tensor
from v15_astra_ncm_v4.cache import load_cache
from v15_astra_ncm_v4.common import ROOT, atomic_json, heavy_job, restore_rng, rng_state, seed_all, source_hashes
from .train import (PARENT, build, generated_prefix_input, residual_target,
                    route_similarity, selection_score, update)


def frozen_digest(net):
    h = hashlib.sha256()
    for name, p in net.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2); seed_all(119)
    cache = load_cache(); net, encoder, optimizer, params, inference = build()
    parent = json.loads((PARENT / 'plan.json').read_text())
    assert source_hashes() == parent['sources'], 'Original registered sources changed'
    assert route_similarity([1, 2, 3], [1, 2, 3]) == 1
    assert route_similarity([], [1, 2, 3]) == 0
    assert route_similarity([1, 3], [1, 2, 3]) == 2 / 3 or abs(route_similarity([1, 3], [1, 2, 3]) - 2 / 3) < 1e-12
    assert selection_score({'greedy_path_exact': 1., 'route_similarity': 1., 'code_nll': 0., 'retention_kl_probe': .16})[0] < 0
    tokens = tensor(cache['tokens'][0:2]); gold = tensor(cache['codes'][0:2]).long()
    q = net.codec[0].model.vq; zhat = torch.zeros(2, 128, 384)
    parity = []
    for level in range(12):
        lo, hi = net.offsets[level:level + 2]
        target = residual_target(net.codec[0], tokens, zhat, level)
        parity.append(float((target == gold[:, lo:hi]).float().mean()))
        assert torch.equal(target, gold[:, lo:hi]), (level, parity[-1])
        with torch.no_grad():
            zhat, _ = q.get_next_autoregressive_input(level, zhat, q.embeddings[level](gold[:, lo:hi]))
    condition = condition_for(encoder, cache['records'], [0], 'mps').detach()
    with torch.no_grad(): full_free = net.generate_codes(condition, temperature=0)
    raw, zhat = generated_prefix_input(net, condition, 6)
    torch.testing.assert_close(raw, net.teacher_inputs(full_free)[:, net.offsets[6]:net.offsets[7]], atol=0, rtol=0)
    corrected = residual_target(net.codec[0], tokens[:1], zhat, 6)
    changed_fraction = float((corrected != gold[:1, net.offsets[6]:net.offsets[7]]).float().mean())
    assert changed_fraction > 0, 'Probe must exercise different targets after a predicted prefix'
    frozen_before = frozen_digest(net)
    times = []; losses = []
    for _ in range(8):
        tick = time.monotonic()
        metrics = update(net, encoder, optimizer, params, cache, 'on_policy')
        times.append(time.monotonic() - tick); losses.append(metrics['weighted_code_loss'])
    saved = {'adapters': net.adapter_state(), 'encoder': copy.deepcopy(encoder.state_dict()),
             'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': rng_state()}
    expected_metrics = update(net, encoder, optimizer, params, cache, 'on_policy')
    expected = net.adapter_state(); expected_encoder = copy.deepcopy(encoder.state_dict())
    net.load_adapters(saved['adapters']); encoder.load_state_dict(saved['encoder'])
    optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
    repeated_metrics = update(net, encoder, optimizer, params, cache, 'on_policy')
    delta = max(float((v - expected[k]).abs().max()) for k, v in net.adapter_state().items())
    encoder_delta = max(float((v - expected_encoder[k]).abs().max()) for k, v in encoder.state_dict().items())
    assert delta == 0 and encoder_delta == 0
    assert expected_metrics == repeated_metrics
    assert frozen_before == frozen_digest(net)
    assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in net.codec[0].model.parameters())
    report = {'passed': True, 'gold_prefix_residual_target_parity_all_levels': parity,
        'model_prefix_input_exact_generation_parity': True,
        'fraction_changed_residual_targets_after_model_prefix_level6': changed_fraction,
        'resume_adapter_max_abs': delta, 'resume_conditioner_max_abs': encoder_delta,
        'resume_metrics_exact': True, 'frozen_generator_unchanged': True,
        'original_sources_unchanged': True, 'training_samples_only': True,
        'on_policy_update_seconds_mean': sum(times) / len(times), 'sparse_losses': losses,
        'mps_driver_gb': torch.mps.driver_allocated_memory() / 1024 ** 3}
    atomic_json(ROOT / 'reports/followup_preflight.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    with heavy_job(): main()
