"""S2 contracts (draft §13.1 items 1, 3, 4): K = 1 equivalence with V8 REFIT, the B_H code path, causality.

Run on CPU (V8_DEVICE=cpu) so the comparison is deterministic; returns a list of check records.
"""
import copy, json, math
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import read_json, SPEC
from model import D, HEADS, S, CODES, DenseCore, Core, chunk_info, positions, rotate, count
from runtime import DEVICE, seed_all, optimizer, lr_at, balanced, nll, ncp_weights
from data import Mixer
import train as v8train
import hncp, train_s2


def teacher_path():
    return read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']


def rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def k1_equivalence(check, core_kind, index, steps=12000):
    """Contract 1: loss, every trainable gradient and one AdamW update equal frozen train.py REFIT to 1e-6 relative."""
    device = DEVICE; tp = teacher_path()
    seed_all(5)
    s = v8train.configure('REFIT', tp, core=core_kind, device=device)
    init_core = copy.deepcopy(s.core.state_dict())
    fn = v8train.forward_function(s, 'REFIT', False)
    b = Mixer(81999).batch(device)
    frac = index / steps; delta = (1. if frac < .1 else max(0., 1 - (frac - .1) / .2))
    loss8, *_ = v8train.objective(s, 'REFIT', b, fn, .1, delta)
    loss8.backward()
    opt8 = optimizer(s, .001, 'REFIT')
    for g in opt8.param_groups:
        g['lr'] = lr_at(index, steps, .001, 200) * g.get('lr_mult', 1)
    g8 = {n: p.grad.detach().clone() for n, p in s.named_parameters() if p.requires_grad}
    torch.nn.utils.clip_grad_norm_([p for p in s.parameters() if p.requires_grad], 1., error_if_nonfinite=True); opt8.step()
    after8 = {n: p.detach().clone() for n, p in s.named_parameters() if p.requires_grad}

    hs, teacher = hncp.build(tp, core_kind, 1, device)
    hs.core.load_state_dict(init_core)
    check(f'K1/{core_kind}: no code parameters instantiated', hs.codes is None and not any('codes' in n for n, _ in hs.named_parameters()))
    n8 = sum(p.numel() for p in s.parameters() if p.requires_grad); n9 = sum(p.numel() for p in hs.parameters() if p.requires_grad)
    check(f'K1/{core_kind}: trainable set equals V8 REFIT', n8 == n9 == (869504 if core_kind == 'dense' else 476288), v8=n8, v9=n9)
    gen = torch.Generator().manual_seed(1)
    cfg = dict(steps=steps, k=1, m=2, p_cl=0., variant='detached', alpha=.1, eps=.05, lam_sel=.1)
    loss9, _ = train_s2.objective(hs, teacher, b, cfg, index, gen)
    loss9.backward()
    opt9 = optimizer(hs, .001, 'REFIT')
    for g in opt9.param_groups:
        g['lr'] = lr_at(index, steps, .001, 200) * g.get('lr_mult', 1)
    g9 = {n: p.grad.detach().clone() for n, p in hs.named_parameters() if p.requires_grad}
    torch.nn.utils.clip_grad_norm_([p for p in hs.parameters() if p.requires_grad], 1., error_if_nonfinite=True); opt9.step()
    after9 = {n: p.detach().clone() for n, p in hs.named_parameters() if p.requires_grad}
    check(f'K1/{core_kind}/u={index}: same trainable names', sorted(g8) == sorted(g9))
    lrel = abs(float(loss8) - float(loss9)) / abs(float(loss8))
    grel = max(rel(g8[n], g9[n]) for n in g8); urel = max(rel(after8[n], after9[n]) for n in after8)
    check(f'K1/{core_kind}/u={index} (delta={delta:.2f}): loss, gradients, AdamW step within 1e-6 relative', lrel <= 1e-6 and grel <= 1e-6 and urel <= 1e-6,
          loss_rel=lrel, grad_rel=grel, update_rel=urel, loss=float(loss8))
    # The teacher must not have moved.
    check(f'K1/{core_kind}: frozen foundation untouched', all(torch.equal(a, b) for a, b in zip(teacher.foundation.parameters(), hs.f.parameters())))


def reference_block_codes(block, x, docs, delta):
    """Slow per-(position, code) reference: query/residual from x_j + delta_k, keys/values from the unperturbed stream."""
    b, t, _ = x.shape; hd = D // HEADS; pos = positions(docs)
    a = block.n1(x); qkv = block.qkv(a); _, k, v = qkv.view(b, t, 3, HEADS, hd).permute(2, 0, 3, 1, 4); k = rotate(k, pos)
    out = torch.zeros(b, t, len(delta), D, dtype=x.dtype, device=x.device)
    for kk in range(len(delta)):
        for j in range(t):
            xj = x[:, j] + delta[kk]
            q = block.qkv(block.n1(xj))[:, :D].view(b, HEADS, 1, hd)
            q = rotate(q, pos[:, j:j + 1])
            allowed = (docs[:, :j + 1] == docs[:, j:j + 1])
            att = F.scaled_dot_product_attention(q, k[:, :, :j + 1], v[:, :, :j + 1], attn_mask=allowed[:, None, None, :])
            y = xj + block.o(att.reshape(b, D))
            g, u = block.w12(block.n2(y)).chunk(2, -1)
            out[:, j, kk] = y + block.w3(F.silu(g) * u)
    return out


def code_path(check):
    """Contract 3: B_H code path; zero code = plain block; batched = one at a time = slow reference; recurrent visits 1-5 untouched."""
    torch.manual_seed(3); tp = teacher_path()
    b = Mixer(81998).batch(DEVICE)
    for core_kind in ('dense', 'recurrent'):
        hs, _ = hncp.build(tp, core_kind, 8, DEVICE)
        with torch.no_grad():
            hs.codes.W_z.weight.normal_(std=.2)  # strong codes so a mismatch cannot hide under tolerance
            h, c, cd, valid = hs.context(b['x'][:4], b['docs'][:4])
            x, blk, finish = hncp.core_stream(hs.core, c, cd)
            plain = blk(x, cd)
            zero = hncp.block_codes(blk, x, cd, torch.zeros(3, D, device=x.device))
            e0 = max(float((zero[:, :, i] - plain).abs().max() / plain.abs().max()) for i in range(3))
            check(f'{core_kind}: zero code equals the plain block', e0 < 1e-6, relative_error=e0)
            deltas = hs.codes.deltas()
            batched = hncp.block_codes(blk, x, cd, deltas)
            single = torch.stack([hncp.block_codes(blk, x, cd, deltas[i:i + 1])[:, :, 0] for i in range(8)], 2)
            eb = float((batched - single).abs().max() / single.abs().max())
            check(f'{core_kind}: all K codes in one call equal one at a time', eb < 1e-6, relative_error=eb)
            ref = reference_block_codes(blk, x[:2, :20], cd[:2, :20], deltas[:3])
            got = hncp.block_codes(blk, x[:2, :20], cd[:2, :20], deltas[:3])
            e_ref = float((ref - got).abs().max() / ref.abs().max())
            check(f'{core_kind}: keys/values unperturbed (slow reference)', e_ref < 1e-6, relative_error=e_ref, scale=float(ref.abs().max()))
            check(f'{core_kind}: codes change the output', float((batched[:, :, 0] - batched[:, :, 1]).abs().max()) > 1e-3)
            # Code-free path equals the V8 core forward exactly.
            u_ref, a_ref = hs.core(c, cd)
            H = hs.heads(c, cd)
            check(f'{core_kind}: code-free readout equals V8 core forward', torch.equal(H['u0'], u_ref) and torch.equal(H['a0'], a_ref))
            # Perturbing the code never changes the stream before B_H (all visits 1-5 for the recurrent core).
            hs2 = copy.deepcopy(hs); hs2.codes.e.add_(5.)
            x2, _, _ = hncp.core_stream(hs2.core, c, cd)
            check(f'{core_kind}: stream before B_H independent of codes', torch.equal(x, x2))
            H2 = hs2.heads(c, cd)
            check(f'{core_kind}: code-free outputs and selector independent of code embeddings', torch.equal(H['u0'], H2['u0']) and torch.equal(H['pi_logits'], H2['pi_logits']))
        n = count(hs.codes)
        check(f'{core_kind}: V9 adds 3,208 parameters at K = 8', n == 3208, n=n)
        with torch.no_grad():
            fresh = hncp.Codes(8).to(x.device); ratio = float(fresh.deltas().square().mean().sqrt() / x.square().mean().sqrt())
        check(f'{core_kind}: initial code write RMS ratio to the B_H stream recorded (§6.2)', 0 < ratio < 1, write_rms=float(fresh.deltas().square().mean().sqrt()), ratio=ratio)


def causality(check):
    """Contract 4: the code at chunk j changes logits only at positions >= 4j+3; later tokens cannot reach earlier plans."""
    torch.manual_seed(4); tp = teacher_path(); b = Mixer(81997).batch(DEVICE)
    for core_kind in ('dense', 'recurrent'):
        hs, _ = hncp.build(tp, core_kind, 4, DEVICE)
        with torch.no_grad():
            hs.codes.W_z.weight.normal_(std=.2)
        x, docs = b['x'][:2], b['docs'][:2]
        h, c, cd, valid = hs.context(x, docs)
        M = c.shape[1]
        e = hs.codes.e[None, None].expand(2, M, 4, hncp.CODE_DIM).clone().requires_grad_(True)
        H = hs.heads(c, cd, e=e)
        codes = torch.randint(0, 4, (2, M))
        chat, a = hs.plan(H, codes)
        logits = hs.f.decode(h, chat, a, docs)
        worst = 0.; live = 0
        for j in (0, 5, 17, M - 2):
            early = torch.autograd.grad(logits[:, :4 * j + 3].sum(), e, retain_graph=True, allow_unused=True)[0]
            late = torch.autograd.grad(logits[:, 4 * j + 3:4 * j + 7].sum(), e, retain_graph=True)[0]
            worst = max(worst, 0. if early is None else float(early[:, j].abs().max()))
            live += int(float(late[:, j].abs().max()) > 0)
        check(f'{core_kind}: code at chunk j cannot reach positions <= 4j+2', worst == 0., worst=worst)
        check(f'{core_kind}: code at chunk j reaches positions 4j+3..4j+6', live == 4)
        with torch.no_grad():
            j = 10; x2 = x.clone(); x2[:, 4 * (j + 1):] = torch.randint(33, 2048, x2[:, 4 * (j + 1):].shape, device=x.device)
            _, c2, _, _ = hs.context(x2, docs); H2 = hs.heads(c2, cd)
            diff = max(float((H['chatk'][:, :j + 1] - H2['chatk'][:, :j + 1]).abs().max()), float((H['pi_logits'][:, :j + 1] - H2['pi_logits'][:, :j + 1]).abs().max()))
            check(f'{core_kind}: tokens of chunk j+1 and later cannot change plans or selector at chunks <= j', diff < 1e-5, error=diff)


def run():
    torch.set_num_threads(4); records = []
    def check(name, cond, **details):
        if not cond:
            raise AssertionError(name + ': ' + json.dumps(details, default=str))
        records.append(dict(name=name, passed=True, **details))
    for core_kind in ('dense', 'recurrent'):
        for index in (0, 6000):
            k1_equivalence(check, core_kind, index)
    code_path(check); causality(check)
    return records


if __name__ == '__main__':
    for r in run():
        print(json.dumps(r))
