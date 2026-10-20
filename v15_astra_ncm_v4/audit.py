"""Serial official-checkpoint, codec, conditioning and MPS acceptance audit."""
import json
import platform
import time

import torch

from .common import ROOT,WORKSPACE,atomic_json,heavy_job,seed_all,sha
from .graphs import load_dataset,oracle_path
from .ncm import load_ncm,decoded_text


def ordinary_texts(count=256):
    manifest = json.loads((WORKSPACE/'v15_astra_v3/artifacts/raw_native32_20261019/documents.json').read_text())
    allowed = {d['doc_id'] for d in manifest if d['split']=='train'}
    result = []
    with (WORKSPACE/'v15_astra_v3/artifacts/fineweb_p0_20261018/documents.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['doc_id'] in allowed:
                result.append(row['text'])
            if len(result)==count:break
    return result


def main():
    torch.set_num_threads(2);seed_all(191026)
    started = time.time()
    report = {'started':started,'torch':torch.__version__,'platform':platform.platform(),'status':'running'}
    path = ROOT/'reports/upstream_audit.json'
    atomic_json(path,report)
    net,codec,inference = load_ncm()
    # Count the originals by LoRA base paths explicitly; requires_grad is authoritative.
    report['frozen_generator_parameters'] = sum(p.numel() for p in net.base.parameters() if not p.requires_grad)
    report['codec_parameters'] = sum(p.numel() for p in codec.model.parameters())
    report['lora_parameters'] = sum(p.numel() for n,p in net.base.named_parameters() if p.requires_grad)
    report['adapter_names'] = net.adapter_names
    report['levels'] = list(codec.levels)
    assert report['lora_parameters']==221184
    assert codec.offsets[-1]==641
    ordinary = ordinary_texts(100)
    graphs = [g for g in load_dataset()['graphs'] if g['split']=='train'][:100]
    symbolic = [f"Edges: {g['edges']}. Current: {g['start']}; goal: {g['goal']}. Verified path: {oracle_path(g,g['start'])}." for g in graphs]
    examples = []
    first_codes = None
    for category,texts in [('ordinary_train',ordinary),('symbolic_train',symbolic)]:
        matches = total = exact = bytes_exact = 0
        for start in range(0,len(texts),8):
            batch = texts[start:start+8]
            ids = []
            normalized = []
            for text in batch:
                row = codec.bpe.encode(text)[:128]
                ids.append(row+[codec.pad]*(128-len(row)))
                normalized.append(codec.bpe.decode(row,skip_special_tokens=True))
            ids = torch.tensor(ids)
            codes = codec.encode_ids(ids)
            if first_codes is None:first_codes=codes[:1].clone()
            pred = codec.decode_logits(codes).argmax(-1)
            active = ids!=codec.pad
            matches += int(((pred==ids)&active).sum());total += int(active.sum())
            exact += int(((pred==ids)|~active).all(-1).sum())
            decoded = codec.text_from_ids(pred)
            bytes_exact += sum(a==b for a,b in zip(batch,decoded))
            if start==0:
                examples += [{'category':category,'normalized':a,'decoded':b} for a,b in zip(normalized,decoded)]
        report[category] = {'nonpad_token_accuracy':matches/total,'exact_nonpad_block':exact/len(texts),
                            'original_byte_exact':bytes_exact/len(texts),'examples':len(texts),
                            'contract':'first128 BERT IDs; full strings may exceed window; case intentionally lost'}
        atomic_json(path,report)
    report['codec_examples'] = examples
    raw = net.teacher_inputs(first_codes)
    with torch.no_grad():
        original = net.base(torch.cat([net.raw_prefix,raw],1))
        cond = torch.randn(1,128)
        levels = [net.level_logits(raw[:,codec.offsets[j]:codec.offsets[j+1]],j,cond) for j in range(12)]
        difference = float((original-torch.cat(levels,1)).abs().max())
        assert difference<2e-4,difference
        report['zero_adapter_full_vs_level_max_abs'] = difference
        # Nonzero FiLM parity proves identical conditioning layout, including level0.
        net.film[-1].bias.fill_(.01)
        targets = net.base._apply_post_upsample_convs_multi(raw)
        x = net.base.transformer.wte(targets)+net.base.transformer.wle(net.base.lvl_1L)
        chunks=[]
        for j in range(12):
            gamma,beta=net.film(torch.cat([cond,net.level_embedding(torch.tensor([j]))],-1)).chunk(2,-1)
            chunks.append(x[:,codec.offsets[j]:codec.offsets[j+1]]*(1+gamma[:,None])+beta[:,None])
        full = net.base._run_stack(net.base.transformer.wte(net.raw_prefix),torch.cat(chunks,1))
        sparse = torch.cat([net.level_logits(raw[:,codec.offsets[j]:codec.offsets[j+1]],j,cond) for j in range(12)],1)
        difference = float((full-sparse).abs().max());assert difference<2e-4,difference
        report['nonzero_film_full_vs_level_max_abs'] = difference
        net.film[-1].bias.zero_()
        upstream_codes = net.base.generate(None,torch.tensor([[codec.eos]]),temperature=0,top_k=0,top_p=1.)
        ours = net.generate_codes(temperature=0)
        assert torch.equal(upstream_codes,ours)
        report['greedy_generation_code_parity'] = True
        report['query_greedy_text'] = decoded_text(codec,ours,inference)
        l0 = torch.multinomial(inference['level0_unigram'].float().cpu(),2,replacement=True).reshape(2,1)
        generated = net.generate_codes(temperature=1.,level0=l0,batch_size=2)
        report['published_sampler_text'] = decoded_text(codec,generated,inference,True)
        report['boundary_normalized_penalty_text'] = decoded_text(codec,generated,inference,True,True)
        report['plain_decoder_same_codes_text'] = decoded_text(codec,generated,inference,False)
    # Exact frozen weights hashed before and after one MPS adapter update.
    import hashlib
    def frozen_hash():
        h=hashlib.sha256()
        for name,p in net.named_parameters():
            if not p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    frozen_before=frozen_hash()
    original_adapters=net.adapter_state()
    level=6; inp=raw[:,codec.offsets[level]:codec.offsets[level+1]]
    with torch.no_grad():cpu=net.level_logits(inp,level,cond)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required for registered local plan')
    net.to('mps');cond=cond.to('mps')
    with torch.no_grad():gpu=net.level_logits(inp,level,cond).cpu()
    report['cpu_mps_max_abs']=float((cpu-gpu).abs().max())
    assert report['cpu_mps_max_abs']<.005
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4)
    target=first_codes[:,codec.offsets[level]:codec.offsets[level+1]].to('mps')
    tick=time.perf_counter()
    logits=net.level_logits(inp,level,cond)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
    loss.backward()
    grads=[p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    opt.step();torch.mps.synchronize()
    report['mps_sparse_training_seconds']=time.perf_counter()-tick
    report['mps_sparse_loss']=float(loss.detach())
    report['mps_allocated_bytes']=torch.mps.current_allocated_memory()
    report['mps_driver_bytes']=torch.mps.driver_allocated_memory()
    assert frozen_before==frozen_hash()
    assert any(not torch.equal(v,net.adapter_state()[n]) for n,v in original_adapters.items())
    report['frozen_generator_unchanged']=True
    report['passed']=True;report['status']='complete';report['seconds']=time.time()-started
    atomic_json(path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('codec_examples','adapter_names')}),flush=True)


if __name__=='__main__':
    with heavy_job():main()
