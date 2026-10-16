"""Behavioral contracts for causality, conditioning, recurrence and memory learning."""
import json
import math
import time
from dataclasses import replace
from pathlib import Path
import torch
from .config import ModelConfig
from .model import V10, losses, Decoder
from .memory import RBFMemory, learn_episode
from .data import verify_sudoku


def run(device='cpu'):
    torch.set_num_threads(4)
    torch.manual_seed(1011)
    c = replace(ModelConfig(), vocab_size=96, engram_rows=(101, 103, 107, 109))
    m = V10(c).place(device).eval()
    x = torch.randint(4, c.vocab_size, (2, 33), device=device)
    lengths = torch.tensor([33, 25], device=device)
    checks = {}
    tolerance = 3e-5 if device == 'mps' else 2e-6
    with torch.no_grad():
        base = m(x, lengths)
        changed = x.clone(); changed[:, 17:] = torch.randint(4, c.vocab_size, changed[:, 17:].shape, device=device)
        other = m(changed, lengths)
        err = (base['logits'][:, :17]-other['logits'][:, :17]).abs().max().item()
        assert err < tolerance, err
        checks['future_tokens_cannot_change_past_logits'] = err
        err = (base['logits'][:, :17]-m(x[:, :17])['logits']).abs().max().item()
        assert err < tolerance, err
        checks['full_vs_prefix_recomputation'] = err
        # A short row must not consume its padded suffix or another batch row.
        changed = x.clone(); changed[1, 25:] = 91; changed[0] = 90
        z = m(changed, lengths)
        err = (base['logits'][1, :25]-z['logits'][1, :25]).abs().max().item()
        assert err < tolerance
        checks['padding_and_batch_isolation'] = err
        for t in (1, 7, 8, 9, 15, 16):
            z = m(x[:, :t])
            assert z['pooled'].shape[1] == t//8 and torch.isfinite(z['logits']).all()
            assert torch.allclose(z['logits'],base['logits'][:,:t],atol=tolerance,rtol=1e-5)
        checks['partial_group_lengths'] = [1, 7, 8, 9, 15, 16]
        # Exact timing, including first forecast and slot handoff.
        f = torch.tensor([10., 20., 30., 40.]).reshape(1, 2, 2, 1)
        a = Decoder.align(f, 25).flatten().tolist()
        assert a == [0.]*7+[10.]*4+[20.]*4+[30.]*4+[40.]*4+[0.]*2
        checks['shift_rows_7_11_15_19'] = True
        # Compare batched alternatives with individually changed current code.
        p, codes = base['pooled'], base['codes']
        candidates = m.candidates(p, lengths, codes, 1)
        for k in range(c.codes):
            hist = codes[:, :2].clone(); hist[:, 1] = k
            u = m.core(p[:, :2], lengths.clamp_max(16), hist)
            pred = m.head(u[:, -1])[0]
            assert torch.allclose(candidates[:, k], pred, atol=tolerance, rtol=1e-5)
        checks['candidate_history_matches_committed_prefix'] = True
        # Mutating only a later code cannot affect earlier concept groups.
        altered = codes.clone(); altered[:, -1] = (altered[:, -1]+1)%c.codes
        u1, u2 = m.core(p, lengths, codes), m.core(p, lengths, altered)
        assert torch.allclose(u1[:, :-1], u2[:, :-1], atol=tolerance)
        checks['group_causal_concept_attention'] = True
    # Hook actual layer calls, rather than trusting config counters.
    visits = {'e1':0, 'e2':0, 'e3':0, 'concept':0, 'decoder':0}
    handles = []
    for name, layers in [('e1',m.e1), ('e2',m.e2), ('e3',m.e3), ('concept',m.core.layers), ('decoder',m.decoder.layers)]:
        for layer in layers:
            def hook(mod, args, out, name=name): visits[name] += 1
            handles.append(layer.register_forward_hook(hook))
    m(x, lengths)
    for h in handles: h.remove()
    assert visits == dict(e1=4,e2=8,e3=4,concept=8,decoder=4)
    checks['actual_layer_applications'] = visits
    stream_inputs=[];concept_inputs=[]
    h1=m.e2[0].register_forward_pre_hook(lambda mod,args:stream_inputs.append(args[0].detach().clone()))
    h2=m.core.layers[0].register_forward_pre_hook(lambda mod,args:concept_inputs.append(args[0].detach().clone()))
    m(x,lengths);h1.remove();h2.remove()
    assert len(stream_inputs)==len(concept_inputs)==2
    assert not torch.equal(stream_inputs[0],stream_inputs[1])
    assert not torch.equal(concept_inputs[0],concept_inputs[1])
    checks['retained_states_change_between_two_visits']=True
    # Real loss gradients must reach code/head/encoder/decoder and sparse table.
    m.train(); m.zero_grad(set_to_none=True)
    out = m(x, lengths)
    mask = torch.arange(33, device=device)[None, :] < lengths[:, None]-1
    loss = losses(m, out, x.roll(-1, 1), mask, boundary=1)
    loss['total'].backward()
    for name, p in [('exploration_codes',m.core.codes.weight), ('PQ',m.head.codebook),
                    ('fusion',m.decoder.fusion), ('encoder',m.e1[0].attn.qkv.weight)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0, name
    assert m.engram.table.weight.grad.is_sparse
    checks['end_to_end_and_sparse_engram_gradients'] = True
    # Empty bank identity and the first write through actual frozen model.
    m.eval().requires_grad_(False)
    bank = RBFMemory('contract-test', capacity=64)
    with torch.no_grad():
        assert torch.equal(m(x, lengths)['logits'], m(x, lengths, bank=bank.snapshot())['logits'])
    snap = bank.snapshot()
    report = learn_episode(m, bank, x, lengths, x.roll(-1,1), mask, 'feedback-1', eta=.01)
    assert report['writes'] > 0 and len(snap.keys)==0 and bank.version==1
    checks['first_write_through_real_PQ_and_decoder'] = report
    with torch.no_grad():
        before=losses(m,m(x,lengths),x.roll(-1,1),mask,vq_weight=0)['total']
        after=losses(m,m(x,lengths,bank=bank.snapshot()),x.roll(-1,1),mask,vq_weight=0)['total']
        assert after<=before+1e-6
        checks['small_real_model_memory_step']={'before':float(before),'after':float(after)}
    try:
        bank.commit(snap, torch.zeros(1,256), torch.tensor([0]), torch.ones(1,256), 'stale')
        raise AssertionError('stale commit accepted')
    except ValueError: pass
    try:
        bank.commit(bank.snapshot(), torch.zeros(1,256), torch.tensor([0]), torch.ones(1,256), 'feedback-1')
        raise AssertionError('duplicate event accepted')
    except ValueError: pass
    checks['snapshot_and_feedback_guards'] = True
    # An explicit RBF sum with code separation, and differentiable queries.
    tiny = RBFMemory('toy', width=2, capacity=2, sigma=.7)
    tiny.commit(tiny.snapshot(), torch.tensor([[0.,0.]]), torch.tensor([1]), torch.tensor([[-1.,2.]]), 'toy', eta=.1, max_value_rms=10)
    q = torch.tensor([[.1,.2],[.1,.2]], requires_grad=True)
    read = tiny.snapshot().read(q, torch.tensor([1,0]))
    expected = math.exp(-.05/(2*.7**2))*torch.tensor([.1,-.2])
    assert torch.allclose(read[0],expected) and torch.equal(read[1],torch.zeros(2))
    read.sum().backward(); assert q.grad[0].norm()>0
    checks['rbf_exact_sum_code_partition_query_gradient'] = True
    # A complete board is mandatory; copied clues and repeated placeholders fail.
    board='534678912672195348198342567859761423426853791713924856961537284287419635345286179'
    assert verify_sudoku('0'*81,board)['valid']
    assert not verify_sudoku('0'*81,'1'*81)['valid']
    assert not verify_sudoku('0'*81,board[:8])['valid']
    assert not verify_sudoku('9'+board[1:],board)['valid']
    checks['sudoku_strict_verifier'] = True
    return checks


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); a=parser.parse_args()
    start=time.monotonic(); results=run(a.device)
    result=dict(device=a.device,seconds=time.monotonic()-start,checks=results,status='passed')
    path=Path(__file__).resolve().parent/'research'/f'contracts_{a.device}.json'
    path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
