"""Small, CPU-only arithmetic checks for the RePR/InfiLoop design documents.

This is not model code or a training experiment. Uses only the Python standard
library and writes a sibling JSON report when run directly.
"""

import itertools
import json
import math
import random
from pathlib import Path


def logaddexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def check_streaming():
    rng = random.Random(20261020)
    worst = 0.0
    for beta in (0.05, 0.55, 0.95, 0.999):
        for magnitude in (1.0, 20.0, 1000.0):
            history = []
            h = None
            log_mass = None
            for t in range(80):
                z = [rng.uniform(-3, 3) for _ in range(7)]
                score = rng.uniform(-magnitude, magnitude)
                history.append((score, z))
                if h is None:
                    h, log_mass = z[:], score
                else:
                    log_mass = logaddexp(math.log(beta) + log_mass, score)
                    gate = math.exp(score - log_mass)
                    h = [(1 - gate) * x + gate * y for x, y in zip(h, z)]
                logits = [s + (t - i) * math.log(beta)
                          for i, (s, _) in enumerate(history)]
                maximum = max(logits)
                weights = [math.exp(s - maximum) for s in logits]
                denom = sum(weights)
                direct = [sum(w * item[1][j] for w, item in zip(weights, history)) / denom
                          for j in range(7)]
                error = max(abs(x - y) for x, y in zip(h, direct))
                worst = max(worst, error)
                assert error < 1e-10
                assert all(math.isfinite(x) for x in h)
    beta = 0.55
    log_mass = 0.0
    for _ in range(1000):
        log_mass = logaddexp(math.log(beta) + log_mass, 0.0)
    gate = math.exp(-log_mass)
    assert abs(gate - (1 - beta)) < 1e-12
    return {"sequences": 12, "steps_per_sequence": 80,
            "max_absolute_error_against_explicit_softmax": worst,
            "equal_score_limiting_gate_beta_0_55": gate,
            "note": "FP64 algebra only; not a PyTorch/MPS or gradient check."}


def check_information():
    # Positive, normalized joint for A, Y1, Y2; enumerate exact marginals.
    raw = {x: 1 + 3*x[0] + 2*x[1] + x[2] + 5*(x[0] == x[2])
           for x in itertools.product((0, 1), repeat=3)}
    total = sum(raw.values())
    joint = {x: w / total for x, w in raw.items()}

    def prob(conditions):
        return sum(p for state, p in joint.items()
                   if all(state[k] == v for k, v in conditions.items()))

    worst = 0.0
    ratios = []
    for a, y1, y2 in joint:
        first = math.log(prob({0: a, 1: y1}) / prob({0: a})) - math.log(prob({1: y1}))
        second = (math.log(prob({0: a, 1: y1, 2: y2}) / prob({0: a, 1: y1}))
                  - math.log(prob({1: y1, 2: y2}) / prob({1: y1})))
        full = math.log(prob({0: a, 1: y1, 2: y2}) /
                        (prob({0: a}) * prob({1: y1, 2: y2})))
        worst = max(worst, abs(first + second - full))
        ratios.append(second)
    assert worst < 1e-12
    assert min(ratios) < 0
    # PMI of joint [[.4,.1],[.1,.4]] is symmetric but indefinite.
    diag, off = math.log(1.6), math.log(0.4)
    eigenvalues = [diag + off, diag - off]
    assert eigenvalues[0] < 0 < eigenvalues[1]
    return {"max_telescope_error": worst,
            "minimum_pointwise_conditional_log_ratio": min(ratios),
            "symmetric_pmi_eigenvalues": eigenvalues}


def parameter_ledger():
    def self_block(d, m):
        return 4 * d*d + 3 * d*m

    def cross_block(d, m):
        return 8 * d*d + 3 * d*m

    ledger = {
        "shared_text_embedding_and_tied_output_4096x384": 4096 * 384,
        "context_encoder_four_self_blocks": 4 * self_block(384, 1024),
        "context_to_workspace_initial_attention": 4 * 384**2,
        "shared_reasoner_two_self_cross_blocks": 2 * cross_block(384, 1024),
        "sixteen_workspace_queries": 16 * 384,
        "code_prior_two_self_cross_blocks": 2 * cross_block(256, 768),
        "workspace_to_prior_bridge": 384 * 256,
        "code_prior_input_embeddings_two_groups": 2 * 256 * 256,
        "code_prior_output_heads_two_groups_untied": 2 * 256 * 256,
        "text_decoder_five_self_cross_blocks": 5 * cross_block(384, 1024),
        "codec_encoder_two_self_blocks": 2 * self_block(192, 512),
        "codec_text_embedding": 4096 * 192,
        "stride_two_ordered_pair_projection": (2 * 192) * 192,
        "product_codebooks_two_groups_of_96": 2 * 256 * 96,
        "quantized_latent_to_text_decoder_bridge": 192 * 384,
        "loop_scorer_and_decay_parameter": 384 + 1,
    }
    return {"components": ledger,
            "major_weight_sum": sum(ledger.values()),
            "excluded": "Norm gains, positional/type embeddings, scalar temperatures, optional biases; "
                        "also excludes offline teacher and training-only representation heads.",
            "training_only_asymmetric_linear_heads": 2 * 192**2,
            "shared_reasoner_weight_applications_at_R6": 6 * ledger["shared_reasoner_two_self_cross_blocks"]}


def capacity():
    return {
        "new_codec_stride_two": {
            "text_tokens": 32, "slots": 16, "groups": 2, "K": 256,
            "scalar_codes": 32, "payload_bits": 16*2*8,
            "payload_with_optional_six_bit_length": 16*2*8+6,
            "raw_fixed_width_bits_4096_vocab": 32*12,
            "nominal_payload_ratio": (32*12)/(16*2*8)},
        "new_codec_stride_four": {
            "slots": 8, "scalar_codes": 16, "payload_bits": 8*2*8,
            "nominal_payload_ratio": (32*12)/(8*2*8)},
        "ncm_native": {
            "positions": sum([1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]),
            "nominal_payload_bits": 641 * 13, "text_tokens": 128},
        "exact_uniform_eight_node_ids_24_choices_min_bits": 8 * math.log2(24),
        "fp32_loop_state_bytes_B1_slots16_d384": 16 * (384+1) * 4,
        "fp32_adam_30m_params_grads_two_moments_bytes": 30_000_000 * 16,
    }


def compute_estimates():
    d, m, n, context, layers, repeats = 384, 1024, 16, 256, 2, 6
    recurrent = layers * repeats * (
        12*n*d*d + 4*n*n*d + 6*n*d*m + 4*n*context*d)
    fixed_kv = layers * 4*context*d*d
    context_encoder = 4 * (
        8*context*d*d + 4*context*context*d + 6*context*d*m)
    return {
        "convention": "2 FLOPs per multiply-accumulate; dense projections and attention only.",
        "recurrent_stage_with_fixed_context_kv_flops": recurrent + fixed_kv,
        "four_block_bidirectional_context_encoder_flops": context_encoder,
        "reasoner_context_kv_cache_fp32_bytes": layers * 2*context*d * 4,
        "excludes": "Initial attention, code prior, text decoder, norms, softmax, activations and overhead.",
    }


if __name__ == "__main__":
    report = {
        "status": "passed",
        "scope": "Design arithmetic only. No model trained, loaded, or changed.",
        "streaming": check_streaming(),
        "information": check_information(),
        "capacity": capacity(),
        "parameters": parameter_ledger(),
        "compute_estimates": compute_estimates(),
    }
    target = Path(__file__).with_name("design_math_results.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
