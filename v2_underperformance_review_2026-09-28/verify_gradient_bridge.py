"""Additional targeted check: K=12, three windows, real capacity drops, accumulation scaling."""
from pathlib import Path
import sys,json,copy,contextlib
import torch
sys.dont_write_bytecode=True
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64);torch.manual_seed(0)
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A/'source_snapshot/code'))
import test_v2 as test
import train
from model import build_model
m,_,_=build_model(small=True);test.randomize(m);m.train()
ids,doc,ei=test.batch(m,two_docs=True)
ref=copy.deepcopy(m);a=test.cfg_args()
stats=train.micro_step(m,ids,doc,ei,12,a,16,contextlib.nullcontext)
test.reference_step(ref,ids,doc,ei,12,a)
worst=0.;name='';max_abs=0.
for (n,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
    g=p.grad if p.grad is not None else torch.zeros_like(p)
    h=q.grad/16 if q.grad is not None else torch.zeros_like(q)
    ab=(g-h).abs().max().item();rel=ab/(h.abs().max().item()+1e-30)
    if rel>worst:worst=rel;name=n
    max_abs=max(max_abs,ab)
out=dict(K=12,windows=train.windows(12),accum=16,capacity_factor=1.25,drop=float(stats['drop']),
         maximum_relative_gradient_difference=worst,worst_parameter=name,maximum_absolute_difference=max_abs,
         carry_norm_gradient_norm=m.core.carry_norm.gamma.grad.norm().item(),passed=worst<1e-8)
(A/'gradient_bridge_check.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
assert out['passed']
