"""Verify the chosen foreach optimizer matches the scalar optimizer on Metal."""
from dataclasses import replace
from pathlib import Path
import json,torch
from .model import V102
from .accumulate import backward_effective_batch,gradient_metrics
from v10_10M.config import ModelConfig
from v10_10M.train import build_optimizers

def main():
    torch.set_num_threads(2);torch.manual_seed(51)
    c=replace(ModelConfig(),vocab_size=96,engram_rows=(101,103,107,109));initial=V102(c).state_dict()
    raw=dict(ids=torch.randint(4,96,(2,33)),lengths=torch.tensor([33,25]),targets=torch.randint(4,96,(2,33)),target_mask=torch.arange(33)[None]<torch.tensor([33,25])[:,None])
    states=[];losses=[]
    for foreach in (False,True):
        m=V102(c);m.load_state_dict(initial);m.place('mps');opt,sparse,dense=build_optimizers(m)
        for group in opt.param_groups:group['foreach']=foreach
        ls=[]
        for step in range(3):
            opt.zero_grad(set_to_none=True);sparse.zero_grad(set_to_none=True)
            value=backward_effective_batch(m,raw,8,step%2==0,seed=step)
            gradient_metrics(m,dense);opt.step();sparse.step();ls.append(value['total'])
        states.append({k:v.detach().cpu().clone() for k,v in m.state_dict().items()});losses.append(ls)
    error=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]);le=max(abs(a-b) for a,b in zip(*losses))
    assert error<5e-5 and le<5e-5,(error,le)
    result=dict(status='passed',updates=3,max_parameter_error=error,max_loss_error=le)
    Path('v10_2/validation/optimizer_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
