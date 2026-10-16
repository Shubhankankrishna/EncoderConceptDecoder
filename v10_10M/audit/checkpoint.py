"""One-off CPU checkpoint probes; these are NOT historical training gradients."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from ..config import ModelConfig
from ..model import V10, losses
from ..data import Corpus
from ..evaluate import generate
from ..prepare_data import ROOT, DATA
from ..train import source_hash
from ..telemetry.store import atomic_json


def grad_summary(model):
    groups={};layers={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.coalesce().values() if p.grad.is_sparse else p.grad
        norm=float(g.float().norm())
        group='engram_table' if name=='engram.table.weight' else name.split('.')[0]
        groups[group]=groups.get(group,0)+norm**2
        bits=name.split('.')
        layer='.'.join(bits[:2]) if bits[0] in ('e1','e2','e3') else '.'.join(bits[:3]) if bits[0] in ('core','decoder') and bits[1]=='layers' else group
        layers[layer]=layers.get(layer,0)+norm**2
    return dict(components={k:math.sqrt(v) for k,v in groups.items()},layers={k:math.sqrt(v) for k,v in layers.items()},
                all_finite=all(math.isfinite(v) for v in groups.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'runs/pretrain_24h/latest.pt'))
    p.add_argument('--output',default=str(ROOT/'audit/checkpoint_probe.json'));a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(1191);start=time.time()
    # Load the atomically replaced checkpoint once through a stable file descriptor.
    with open(a.checkpoint,'rb') as f:
        stat=os.fstat(f.fileno());ck=torch.load(f,map_location='cpu',weights_only=False)
    m=V10(ModelConfig(**ck['model_config']));m.load_state_dict(ck['model']);m.eval()
    step=ck['step'];source=ck['source_sha256'];tokens=ck['tokens'];del ck
    result=dict(run_id=Path(a.checkpoint).resolve().parent.name,kind='offline_checkpoint_probe',created_unix=start,checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_mtime=stat.st_mtime,checkpoint_bytes=stat.st_size,step=step,tokens=tokens,
        source_sha256=source,source_matches=source==source_hash(),device='cpu',threads=2,parameters=m.counts(),
        limitations='Small deterministic development probes. Gradient norms are fresh checkpoint probes, not actual past optimizer gradients. No optimizer steps or memory writes.',
        finite_parameters=all(bool(torch.isfinite(p).all()) for p in m.parameters()),families={})
    corpus=Corpus('dev',seed=1211)
    for family in corpus.arrays:
        batch=corpus.batch(2,context=192,family=family,device='cpu')
        # Three objective probes distinguish token learning from auxiliary-only learning.
        entry=dict(chosen=batch['chosen'],input_tokens=int(batch['lengths'].sum()),supervised_tokens=int(batch['target_mask'].sum()),gradient_probes={})
        for objective in ('token','concept','total'):
            m.zero_grad(set_to_none=True)
            out=m(batch['ids'],batch['lengths']);parts=losses(m,out,batch['targets'],batch['target_mask'],boundary=0 if objective=='total' else None)
            parts[objective].backward();entry['gradient_probes'][objective]=grad_summary(m)
            entry['gradient_probes'][objective]['loss']=float(parts[objective].detach())
            del out,parts
        m.zero_grad(set_to_none=True)
        with torch.no_grad():
            out=m(batch['ids'],batch['lengths']);parts=losses(m,out,batch['targets'],batch['target_mask'])
            off=m(batch['ids'],batch['lengths'],codes=out['codes'],ablate=True)
            altcodes=(out['codes']+1)%m.c.codes
            alt=m(batch['ids'],batch['lengths'],codes=altcodes)
            mask=batch['target_mask']
            entry.update(token_nll=float(parts['token']),concept_loss=float(parts['concept']),vq_loss=float(parts['vq']),
                concept_ablation_delta_nll=float(losses(m,off,batch['targets'],mask)['token']-parts['token']),
                concept_ablation_logit_rms=float((out['logits'][mask]-off['logits'][mask]).square().mean().sqrt()),
                changed_codes_logit_rms=float((out['logits'][mask]-alt['logits'][mask]).square().mean().sqrt()),
                changed_codes_forecast_rms=float((out['forecast']-alt['forecast']).square().mean().sqrt()),
                code_counts=torch.bincount(out['codes'][out['valid_groups']],minlength=8).tolist())
            # Ablate just the retrieved Engram branch with a temporary CPU-model hook.
            hook=m.engram.register_forward_hook(lambda module,args,output:args[0])
            try:noengram=m(batch['ids'],batch['lengths'])
            finally:hook.remove()
            entry['engram_ablation_delta_nll']=float(losses(m,noengram,batch['targets'],mask)['token']-parts['token'])
            # Actual learned model: future tokens, prefix group boundary and padding.
            ids=batch['ids'][:1,:33];base=m(ids)['logits'];changed=ids.clone();changed[:,17:]=(changed[:,17:]+13)%m.c.vocab_size
            future_error=float((base[:,:17]-m(changed)['logits'][:,:17]).abs().max())
            prefix_error=float((base[:,:17]-m(ids[:,:17])['logits']).abs().max())
            entry['causality']=dict(future_logit_max_error=future_error,prefix_logit_max_error=prefix_error,passed=max(future_error,prefix_error)<1e-4)
        result['families'][family]=entry
        print(json.dumps(dict(family=family,step=step,causality=entry['causality'],token_nll=entry['token_nll'])),flush=True)
    # Inspect exactly the same math examples under teacher forcing and autoregression.
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    arr,offsets,starts=corpus.arrays['math'];meta=[json.loads(x) for x in (DATA/'math_dev_meta.jsonl').read_text().splitlines()]
    examples=[]
    with torch.no_grad():
        for i in range(8):
            tok=arr[offsets[i]:offsets[i+1]].tolist();answer=int(starts[i]);x=torch.tensor([tok[:-1]])
            out=m(x);logits=out['logits'][0,answer-1:];gold=torch.tensor(tok[answer:])
            nll=float(torch.nn.functional.cross_entropy(logits,gold))
            prediction=generate(m,tok[:answer],max_new_tokens=48,device='cpu')
            eos=bool(prediction and prediction[-1]==2);text=tokenizer.decode(prediction[:-1] if eos else prediction,skip_special_tokens=False)
            examples.append(dict(identity=meta[i]['identity'],prompt=tokenizer.decode(tok[:answer],skip_special_tokens=False),
                expected=meta[i]['answer'],response=text,eos=eos,correct=eos and text.strip()==meta[i]['answer'].strip(),
                teacher_forced_nll=nll,teacher_forced_token_accuracy=float((logits.argmax(-1)==gold).float().mean()),
                first_answer_token_correct=bool(logits[0].argmax()==gold[0])))
    result['paired_math_examples']=examples
    result['learned_scales']=dict(pooler_alpha=float(m.pooler.alpha.detach()),core_anchor=float(m.core.anchor_scale.detach()),
        decoder_fusion_rms=float(m.decoder.fusion.detach().square().mean().sqrt()),decoder_crc_rms=float(m.decoder.scales.detach().square().mean().sqrt()))
    result['all_probe_checks_passed']=result['source_matches'] and result['finite_parameters'] and all(
        v['causality']['passed'] and all(g['all_finite'] for g in v['gradient_probes'].values()) for v in result['families'].values())
    result['seconds']=time.time()-start;atomic_json(a.output,result)
    print(json.dumps(dict(output=a.output,step=step,seconds=result['seconds'],passed=result['all_probe_checks_passed'])),flush=True)


if __name__=='__main__':main()
