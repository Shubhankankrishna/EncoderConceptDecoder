"""Bounded CPU/Metal parity and full-size, context-384 integration checks."""
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.train import canonical_ids,build_optimizers
from .model import V101,Architecture,losses
from .validate_training import load_examples,pack,clip_and_log
from .provenance import provenance


def main():
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.15);torch.manual_seed(313)
    c=replace(ModelConfig(),vocab_size=96,engram_rows=(101,103,107,109))
    cpu=V101(c).train();metal=V101(c).train();metal.load_state_dict(cpu.state_dict());metal.place('mps')
    ids=torch.randint(4,96,(2,33));lengths=torch.tensor([33,25]);targets=torch.randint(4,96,(2,33));mask=torch.arange(33)[None,:]<lengths[:,None]
    outputs=[];objectives=[]
    for model,device in [(cpu,'cpu'),(metal,'mps')]:
        x,l,y,m=[z.to(device) for z in (ids,lengths,targets,mask)]
        out=model(x,l);p=losses(model,out,x,l,y,m,candidate_update=True,generator=torch.Generator().manual_seed(47))
        p['total'].backward();outputs.append(out['logits'].detach().cpu());objectives.append(float(p['total'].detach().cpu()))
    logit_error=float((outputs[0]-outputs[1]).abs().max());gradient_errors={}
    for name in ['embedding.weight','encoder_concept_crc.projections.0.weight','core.layers.0.attn.qkv.weight','head.codebook','decoder.concept_projection.weight','chunk_readout.projection.weight']:
        a=dict(cpu.named_parameters())[name].grad;b=dict(metal.named_parameters())[name].grad.cpu()
        gradient_errors[name]=float((a-b).abs().max())
    assert logit_error<5e-4 and abs(objectives[0]-objectives[1])<5e-4 and max(gradient_errors.values())<1e-3
    result=dict(**provenance(),parity=dict(logit_max_absolute_error=logit_error,objective_absolute_error=abs(objectives[0]-objectives[1]),gradient_max_absolute_errors=gradient_errors),
        limitations='Brief shared-device integration; not an exclusive or thermally steady Metal performance benchmark')
    del cpu,metal,out,p,outputs;gc.collect();torch.mps.empty_cache()
    samples,identities,tokenizer=load_examples('train',context=384,count=2)
    batch=pack(samples['text'][:1]);x,l,y,m=[batch[k].to('mps') for k in ['ids','lengths','targets','target_mask']]
    full={}
    for arm in ['A','D','S']:
        torch.manual_seed(311);model=V101(ModelConfig(),Architecture.arm(arm))
        model.engram.canonical.copy_(canonical_ids(tokenizer,8192));model.place('mps').train()
        opt,table,dense=build_optimizers(model);events=[]
        for candidate in [False,False,True]:
            torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);table.zero_grad(set_to_none=True)
            out=model(x,l);p=losses(model,out,x,l,y,m,candidate_update=candidate,generator=torch.Generator().manual_seed(413))
            p['total'].backward();grads=clip_and_log(model,dense);opt.step();table.step();torch.mps.synchronize()
            events.append(dict(candidate=candidate,seconds=time.perf_counter()-start,loss=float(p['total'].detach().cpu()),
                **grads,mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30))
        full[arm]=dict(parameters=model.counts(),context=int(l[0]),batch_size=1,updates=events)
        del model,opt,table,dense,out,p;gc.collect();torch.mps.empty_cache()
    result['full_size_metal']=full
    result['text_sample']=identities['text'][0]
    dest=Path(__file__).resolve().parent/'validation/metal_parity.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
