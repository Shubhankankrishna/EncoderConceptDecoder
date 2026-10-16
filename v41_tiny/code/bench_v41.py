"""Synchronised CPU/MPS benchmarks and full forward/backward parity."""
import copy, json, time, torch
import run_vs as R
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import speed_v3

torch.set_num_threads(2)
speed_v3.enable_block_attention(60)

def sync(device):
    if device == "mps":
        torch.mps.synchronize()

def parity():
    torch.manual_seed(51)
    base = R.build("v41", PP.V)
    rows = PP.PinpadStream(120, 6, 2, seed=17).next()
    b = D.to_batch(*rows, PP.Ident(), base.ecfg)
    out, _ = R.fwd(base, *b)
    loss, _, _ = R.loss_fn(base, *b)
    loss.backward()
    other = copy.deepcopy(base).to("mps")
    other.zero_grad(set_to_none=True)
    ob = tuple(x.to("mps") for x in b)
    oo, _ = R.fwd(other, *ob)
    ol, _, _ = R.loss_fn(other, *ob)
    ol.backward(); sync("mps")
    errors = []
    for (n, p), (n2, q) in zip(base.named_parameters(), other.named_parameters()):
        assert n == n2
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None, n
            continue
        a, z = p.grad, q.grad.cpu()
        err = (a-z).abs().max().item()
        rel = err / max(a.abs().max().item(), 1e-8)
        assert torch.allclose(a,z,atol=2e-4,rtol=2e-3), (n,err,rel)
        errors.append((err,rel,n))
    oe = (out-oo.cpu()).abs().max().item()
    assert torch.allclose(out,oo.cpu(),atol=2e-4,rtol=2e-3), oe
    return dict(logits_max_abs=oe, loss_abs=abs(loss.item()-ol.item()),
                n_gradients=len(errors), worst_gradients=sorted(errors,reverse=True)[:10])

def benchmark(arch, device, steps=8):
    torch.manual_seed(0)
    m = R.build(arch, PP.V).to(device)
    stream = PP.PinpadStream(480,6,4,seed=0)
    batches=[tuple(x.to(device) for x in D.to_batch(*stream.next(),PP.Ident(),m.ecfg)) for _ in range(steps+2)]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95))
    def step(b):
        loss,_,_=R.loss_fn(m,*b)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    for b in batches[:2]: step(b)
    sync(device); t=time.perf_counter()
    for b in batches[2:]: step(b)
    sync(device); dt=time.perf_counter()-t
    real=sum(int((b[0]!=0).sum()) for b in batches[2:])
    return dict(arch=arch,device=device,steps=steps,seconds=dt,seconds_per_step=dt/steps,real_tokens_per_second=real/dt)

if __name__=="__main__":
    result={"parity":parity(),"benchmarks":[]}
    print(json.dumps(result["parity"]),flush=True)
    for device in ("cpu","mps"):
        for arch in ("v41","llm"):
            r=benchmark(arch,device);result["benchmarks"].append(r);print(json.dumps(r),flush=True)
    with open("../runs/device_benchmark.json","w") as f:json.dump(result,f,indent=2)
