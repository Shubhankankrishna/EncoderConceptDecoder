"""Calibration selects execution only after matching eager values AND parameter gradients."""
import copy,json,time
import torch
from model import *
from data import Mixer
from runtime import *
from train import objective,forward_function

def main():
    torch.set_num_threads(4);seed_all(870);s=System().to(DEVICE);b=Mixer(87001).batch(DEVICE)
    with torch.no_grad():h,c,cd=s.context(b['x'],b['docs']);s.foundation.restart(c,chunk_info(b['docs'])[1],generator=torch.Generator().manual_seed(87002))
    eager=copy.deepcopy(s);compiled=copy.deepcopy(s);fn=forward_function(compiled,'A',True);out={};errors=[]
    for name,model,forward in [('eager',eager,forward_function(eager,'A')),('compiled',compiled,fn)]:
        sync();start=time.monotonic();loss,logs,*_=objective(model,'A',b,forward,.1,0);loss.backward();sync();out[name]=dict(cold_s=time.monotonic()-start,loss=float(loss.detach()))
        if name=='compiled':
            for (n,p),(nn,q) in zip(eager.named_parameters(),compiled.named_parameters()):
                if p.grad is not None:errors.append(dict(parameter=n,max_abs=float((p.grad-q.grad).abs().max()),relative=float((p.grad-q.grad).norm()/p.grad.norm().clamp_min(1e-8))))
        start=time.monotonic()
        for _ in range(12):model.zero_grad(set_to_none=True);loss,*_=objective(model,'A',b,forward,.1,0);loss.backward()
        sync();out[name]['step_s']=(time.monotonic()-start)/12;out[name]['slots_per_s']=64*192/out[name]['step_s']
    passed=abs(out['eager']['loss']-out['compiled']['loss'])<2e-5 and all(v['max_abs']<1e-4 and v['relative']<.01 for v in errors)
    out.update(gradient_parity=passed,gradient_errors=errors,chosen='compiled' if passed and out['compiled']['step_s']<out['eager']['step_s'] else 'eager',device=DEVICE,precision='float32',implementation_sha256=implementation_hash())
    atomic_json(ROOT/'tests/throughput.json',out);print(json.dumps(out))
if __name__=='__main__':main()
