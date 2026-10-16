"""Compare real forward/backward losses and every gradient against the reference."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.train import build_optimizers
from v10_1.model import V101
from v10_1.accumulate import backward_effective_batch
from v10_1.train import gradient_metrics as original_metrics
from .fast import backward_full_batch,gradient_metrics


def main(device):
    torch.set_num_threads(2)
    if device=='mps':torch.mps.set_per_process_memory_fraction(.2)
    c=replace(ModelConfig(),vocab_size=96,engram_rows=(101,103,107,109));torch.manual_seed(42)
    m=V101(c).place(device);results=[]
    for candidate in (False,True):
        for lens in ([33,25,16],[8,7,1],[17,9,8]):
            t=max(lens);x=torch.randint(4,96,(3,t));y=torch.randint(4,96,(3,t));lengths=torch.tensor(lens)
            mask=torch.arange(t)[None,:]<lengths[:,None];mask[:, :6]=False
            raw=dict(ids=x,lengths=lengths,targets=y,target_mask=mask)
            reference={k:v.to(device) for k,v in raw.items()};saved=[]
            for engine in ('reference','fast'):
                m.zero_grad(set_to_none=True);diagnostics={}
                if engine=='reference':loss=backward_effective_batch(m,reference,3,candidate,99,diagnostics)
                else:loss=backward_full_batch(m,raw,device,candidate,99,diagnostics)
                grad={name:(p.grad.to_dense() if p.grad.is_sparse else p.grad).detach().cpu().clone() for name,p in m.named_parameters() if p.grad is not None}
                _,_,dense=build_optimizers(m)
                norms=(original_metrics if engine=='reference' else gradient_metrics)(m,dense,True)
                saved.append((loss,grad,diagnostics,norms))
            a,b=saved;assert a[1].keys()==b[1].keys()
            le=max(abs(a[0][k]-b[0][k]) for k in a[0]);ge=max(float((a[1][k]-b[1][k]).abs().max()) for k in a[1])
            tol=5e-5 if device=='mps' else 5e-6
            assert le<tol and ge<tol,(lens,candidate,le,ge)
            for key in ('candidate_documents','chunk_supervised_targets','code_counts','objective_denominators'):assert a[2][key]==b[2][key],key
            ne=max(abs(a[3]['component_grad_norms'][k]-b[3]['component_grad_norms'][k]) for k in a[3]['component_grad_norms'])
            assert ne<tol
            results.append(dict(lengths=lens,candidate=candidate,max_loss_error=le,max_gradient_error=ge,max_component_norm_error=ne))
    dest=Path(__file__).resolve().parents[1]/'validation/throughput_review'/f'parity_{device}.json'
    dest.write_text(json.dumps(dict(status='passed',device=device,cases=results),indent=2)+'\n');print(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');main(p.parse_args().device)
