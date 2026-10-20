"""Actual MPS RNG/optimizer resume check and production-shape GRAM measurement."""
import io
import time

import psutil
import torch

from .codec import ROOT
from .data import atomic_json
from .gram import AstraGRAM
from .tests_gram import small


def main():
    torch.set_num_threads(2);torch.manual_seed(71);torch.mps.manual_seed(71)
    torch.mps.set_per_process_memory_fraction(.6)
    m=AstraGRAM(small()).to('mps');o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False)
    x=torch.randint(17,(2,7),device='mps');s=torch.randint(20,(2,8),device='mps');y=torch.randint(17,(2,7),device='mps')
    def update(model,opt):
        opt.zero_grad(set_to_none=True);loss,_=model(x,s,y);loss.backward();opt.step()
    update(m,o);buf=io.BytesIO()
    torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),mps_rng=torch.mps.get_rng_state()),buf)
    update(m,o);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    buf.seek(0);ck=torch.load(buf,map_location='cpu',weights_only=True)
    restored=AstraGRAM(small()).to('mps');restored.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(restored.parameters(),lr=.001,foreach=False);opt.load_state_dict(ck['optimizer'])
    torch.mps.set_rng_state(ck['mps_rng']);update(restored,opt)
    error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in restored.state_dict().items())
    assert error<1e-6,error
    del m,o,restored,opt,ck;torch.mps.empty_cache()
    m=AstraGRAM().to('mps');o=torch.optim.AdamW(m.parameters(),lr=.0003,foreach=False)
    x=torch.randint(8192,(8,641),device='mps');s=torch.randint(114,(8,88),device='mps');y=torch.randint(8192,(8,641),device='mps')
    times=[];stats=None
    for i in range(13):
        tic=time.perf_counter();o.zero_grad(set_to_none=True)
        for _ in range(2):
            loss,stats=m(x,s,y);(loss/2).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();torch.mps.synchronize()
        if i>=3:times.append(time.perf_counter()-tic)
    report=dict(mps_stochastic_resume_max_abs_error=error,parameters=m.parameter_count(),
                batch=8,accumulation=2,optimizer_seconds=times,
                blocks_per_second=16/(sum(times)/len(times)),
                driver_bytes=torch.mps.driver_allocated_memory(),rss=psutil.Process().memory_info().rss,
                available_ram=psutil.virtual_memory().available,
                last_kl_sum=float(stats['kl_sum']),last_loss=float(loss.detach()),
                scope='13 production-shape steps with random IDs; 3 warmups; not a sustained thermal benchmark')
    atomic_json(ROOT/'reports/gram_mps_verification.json',report);print(report)


if __name__=='__main__':main()
