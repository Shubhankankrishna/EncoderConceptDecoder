"""Read trained weights and compare causal, dynamic, and static execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from model import DZ,freeze
from data import gen_symbolic
from runtime import load_system,atomic_json
from generation import Incremental,FastIncremental

ap=argparse.ArgumentParser();ap.add_argument('--device',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
torch.set_num_threads(4);s,_=load_system(Path(__file__).resolve().parents[1]/'runs/A_alpha0/checkpoint.pt',a.device);freeze(s);records=[]
for nodes,p,context in ((10,124,192),(14,172,384),(18,220,384)):
    row=gen_symbolic(np.random.RandomState(98000+nodes),nodes,hard=True,context=context);x=torch.tensor(row[0][None],device=a.device);z=torch.zeros(1,96,DZ,device=a.device)
    for kind in ('dynamic','static_eager','static_compiled') if a.device=='mps' else ('dynamic','static_eager'):
        if hasattr(s,'_fast_steps'):delattr(s,'_fast_steps')
        original=torch.compile
        try:
            if kind=='static_eager':torch.compile=lambda f:f
            cache=(Incremental if kind=='dynamic' else FastIncremental)(s,z)
        finally:torch.compile=original
        if kind=='static_compiled':
            s._fast_steps=tuple(v if hasattr(v,'_torchdynamo_orig_callable') else torch.compile(v) for v in s._fast_steps)
        with torch.no_grad():
            inc=cache.prefill(x[:,:p])
            for pos in range(p,p+9):
                if pos>p:inc=cache.step(x[:,pos-1])
                full=s(x[:,:pos],torch.zeros_like(x[:,:pos]),z[:,:pos//4])[:,-1];diff=(inc-full).abs()
                records.append(dict(nodes=nodes,context=context,kind=kind,pos=pos,maximum_logit_error=float(diff.max()),rms_error=float(diff.square().mean().sqrt()),strict_allclose=bool(torch.allclose(inc,full,atol=4e-4,rtol=1e-5)),argmax_agreement=bool((inc.argmax(-1)==full.argmax(-1)).all()),reference_margin=float(full.topk(2).values.diff(dim=-1).abs().min()),probability_maximum_error=float((inc.softmax(-1)-full.softmax(-1)).abs().max())))
summary={kind:dict(maximum_logit_error=max(v['maximum_logit_error'] for v in records if v['kind']==kind),strict_pass=all(v['strict_allclose'] for v in records if v['kind']==kind),all_argmax_agree=all(v['argmax_agreement'] for v in records if v['kind']==kind)) for kind in sorted({v['kind'] for v in records})}
atomic_json(a.out,dict(device=a.device,summary=summary,records=records));print(json.dumps(summary),flush=True)
