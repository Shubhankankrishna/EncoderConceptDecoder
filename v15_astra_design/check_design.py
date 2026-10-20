"""Analytical checks for a proposal; this is not an implementation or model test."""
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parameter_ledger(dc=128, d=256, vocab=2048, block=16,
                     sizes=(256, 256, 256, 1024), ne=4, nd=4, nc=4,
                     nl=2, nh=2, modes=4, groups=4, keydim=4,
                     experts=1, active_experts=1, rank=4):
    # Bias-free projections, parameterless RMSNorm, FFN width = 2 * model width.
    levels = len(sizes)
    codec_block = 5 * dc**2 + 3 * dc * (2 * dc)
    ctx_block = 5 * d**2 + 3 * d * (2 * d)
    # Shared within each codec stack: q,k maps, gate/importance/forget maps, gain.
    depth_maps = 2 * dc * groups * keydim + 3 * dc * groups + dc
    codec = {
        'tied_token_embedding_and_output': vocab * dc,
        'encoder_and_decoder_blocks': (ne + nd) * codec_block,
        'ema_codebook_scalars': sum(sizes) * dc,
        'residual_reconstruction_maps': levels * dc**2,
        'shared_position_and_prefix_level_embeddings': block * dc + levels * dc,
        'two_depth_memory_map_sets': 2 * depth_maps,
    }
    core = {
        'context_input_projection': dc * d,
        'context_blocks': nc * ctx_block,
        'coarse_reconstruction_projection': dc * d,
        'query_position_level_mode_embeddings': (block + levels + modes) * d,
        'hrm_self_and_cross_attention': (nl + nh) * 10 * d**2,
        'hrm_expert_ffns': (nl + nh) * experts * 3 * d * (2 * d),
        'expert_routers': (nl + nh) * d * experts if experts > 1 else 0,
        'code_heads': d * sum(sizes),
    }
    # Two slots, each on self-attention Q,V and FFN-down of every HRM block.
    # Dense tiny profiles only. At scale this policy must name selected experts.
    per_slot = (nl + nh) * (2 * rank * (2 * d) + rank * (3 * d)) + d + 1
    stored = sum(codec.values()) + sum(core.values())
    active = stored - core['hrm_expert_ffns'] + (nl + nh) * active_experts * 6 * d**2
    return {'codec': codec, 'concept': core, 'codec_total': sum(codec.values()),
            'stage_1a_gradient_parameters': sum(codec.values()) - codec['ema_codebook_scalars'],
            'stage_1b_trainable': sum(core.values()), 'base_stored': stored,
            'selected_expert_parameter_accounting': active,
            'two_dense_adapter_slots': 2 * per_slot if experts == 1 else None,
            'with_two_slots': stored + 2 * per_slot if experts == 1 else None}


def analytical_checks():
    checks = {}
    # Enumeration verifies the complete hierarchy mixture, not separately mixed levels.
    p = (.6, .4)
    roots = ((.8, .2), (.3, .7))
    fine = (((.9, .1), (.2, .8)), ((.4, .6), (.7, .3)))
    joint = lambda r, f: sum(p[k] * roots[k][r] * fine[k][r][f] for k in range(2))
    mass = sum(joint(r, f) for r, f in itertools.product(range(2), repeat=2))
    assert abs(mass - 1) < 1e-12
    post = [p[k] * roots[k][0] for k in range(2)]
    post = [v / sum(post) for v in post]
    assert abs(joint(0, 1) - sum(p[k] * roots[k][0] for k in range(2)) *
               sum(post[k] * fine[k][0][1] for k in range(2))) < 1e-12
    checks['normalized_hierarchy_mixture_and_posterior_conditioning'] = True
    # Exact responsibility-weighted gradient of -logsumexp(-loss_k), scalar finite difference.
    loss = lambda t: -math.log(.5 * math.exp(-t*t) + .5 * math.exp(-(t-2)**2))
    t, eps = .3, 1e-6
    a, b = math.exp(-t*t), math.exp(-(t-2)**2)
    analytic = (a * 2*t + b * 2*(t-2)) / (a+b)
    assert abs((loss(t+eps)-loss(t-eps))/(2*eps)-analytic) < 1e-8
    checks['mixture_responsibility_gradient'] = True
    # Scalar RLS with forgetting agrees with direct weighted ridge regression.
    m, inv, num, den = 0., 1., 0., 1.
    for key, val, weight, forget in [(1., 2., .7, .98), (-.4, 1., 1.3, .95), (.2,-1., .5,1.)]:
        k, v = math.sqrt(weight)*key, math.sqrt(weight)*val
        gain = inv*k/(forget+k*k*inv)
        m += (v-m*k)*gain
        inv = (inv-gain*k*inv)/forget
        num, den = forget*num + weight*key*val, forget*den + weight*key*key
        assert abs(m-num/den) < 1e-12 and abs(inv-1/den) < 1e-12
    checks['rls_matches_decaying_ridge_scalar_case'] = True
    # Canonical degree-two feature map implements 1 + x.y + (x.y)^2/2.
    def phi(x):
        return [1.] + list(x) + [a*a/math.sqrt(2) for a in x] + [x[i]*x[j] for i in range(len(x)) for j in range(i+1,len(x))]
    x,y = [.2,-.3,.1],[-.1,.4,.5]
    xy=sum(a*b for a,b in zip(x,y))
    assert abs(sum(a*b for a,b in zip(phi(x),phi(y))) - (1+xy+xy*xy/2)) < 1e-12
    checks['quadratic_feature_kernel'] = True
    # Policy-gradient identity through a discrete code and deterministic decoder reward.
    sigmoid=lambda t: 1/(1+math.exp(-t))
    reward=lambda t: (1-sigmoid(t))*.2+sigmoid(t)*.9
    t=.4;s=sigmoid(t)
    score_grad=(1-s)*.2*(-s)+s*.9*(1-s)
    assert abs(score_grad-(reward(t+eps)-reward(t-eps))/(2*eps)) < 1e-9
    checks['discrete_code_score_function_gradient'] = True
    tiny=parameter_ledger()
    small=parameter_ledger(nl=1,nh=1)
    assert 7_000_000 <= small['with_two_slots'] <= tiny['with_two_slots'] <= 10_000_000
    checks['both_prototypes_within_total_parameter_budget'] = True
    assert sum(a*math.log2(n) for a,n in zip((1,2,4,16),(256,256,256,1024))) == 216
    checks['codec_rate_216_bits_per_16_token_block'] = True
    return checks


def main():
    out = {'scope':'Analytical checks only. No neural model, training, gradients through an implemented HRM, latency or quality tested.',
           'tiny': parameter_ledger(), 'small': parameter_ledger(nl=1,nh=1),
           'illustrative_10t': parameter_ledger(dc=4096,d=8192,vocab=128000,block=64,
               sizes=(8192,)*6,ne=16,nd=16,nc=32,nl=48,nh=48,modes=8,
               groups=128,keydim=8,experts=256,active_experts=2),
           'fast_memory': {'feature_dimension':153,'matrix_scalars':256*153,
               'matrix_plus_momentum_fp32_bytes':2*256*153*4,
               'fixed_key_projection_scalars':(2*128+4)*16},
           'checks':analytical_checks()}
    (Path(__file__).parent/'design_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
