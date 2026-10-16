"""Independent CPU versus Metal loss/all-gradient comparison, including sparse table."""
from dataclasses import replace
import json,torch
from pathlib import Path
from v10_10M.config import ModelConfig
from .model import V102
from .accumulate import backward_effective_batch

def main():
    torch.set_num_threads(2);torch.manual_seed(830)
    c=replace(ModelConfig(),vocab_size=96,engram_rows=(101,103,107,109))
    cpu=V102(c);state={k:v.detach().clone() for k,v in cpu.state_dict().items()}
    raw=dict(ids=torch.randint(4,96,(2,33)),lengths=torch.tensor([33,25]),targets=torch.randint(4,96,(2,33)),target_mask=torch.arange(33)[None]<torch.tensor([33,25])[:,None])
    expected=backward_effective_batch(cpu,raw,8,True,seed=12)
    grads={k:(p.grad.to_dense() if p.grad.is_sparse else p.grad).clone() for k,p in cpu.named_parameters() if p.grad is not None}
    gpu=V102(c);gpu.load_state_dict(state);gpu.place('mps')
    actual=backward_effective_batch(gpu,raw,8,True,seed=12)
    errors={k:float(((p.grad.to_dense() if p.grad.is_sparse else p.grad).cpu()-grads[k]).abs().max()) for k,p in gpu.named_parameters() if p.grad is not None}
    loss_error=max(abs(actual[k]-expected[k]) for k in expected)
    assert loss_error<1e-4 and max(errors.values())<2e-4,(loss_error,max(errors.values()))
    result=dict(status='passed',max_loss_error=loss_error,max_gradient_error=max(errors.values()),parameters_compared=len(errors),cpu=expected,mps=actual)
    Path('v10_2/validation/cpu_mps_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
