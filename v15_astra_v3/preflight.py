"""Real-codec-cache and production-size MPS launch checks; no live legacy job."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from .controls import make_model,parameter_ledger
from .raw import RawCodeDataset
from .gram import AstraGRAM
from .isolation import ROOT,exclusive_job,atomic_json
from .train import save


@exclusive_job
def run(data_path,out):
    torch.set_num_threads(1);torch.manual_seed(151803)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.mps.set_per_process_memory_fraction(.5)
    data=RawCodeDataset(data_path)
    if data.manifest['decode_audit']['nonpad_token_accuracy']<.8:raise RuntimeError('Raw codec fidelity below preregistered 80% token gate')
    config=json.loads((ROOT/'configs/gram_shared.json').read_text());config['context_source']='codes'
    model=make_model('code_gram',config);indices=data.indices['train'][:4];cpu=data.batch(indices)
    noise=torch.randn(2,4,16,192)
    with torch.no_grad():reference=float(model(cpu['x'],None,cpu['y'],noise=noise)[0])
    model.to('mps');batch=data.batch(indices,'mps')
    with torch.no_grad():actual=float(model(batch['x'],None,batch['y'],noise=noise.to('mps'))[0])
    if abs(reference-actual)>1e-4:raise RuntimeError(f'CPU/MPS objective mismatch: {reference} vs {actual}')
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,foreach=False)
    durations=[];losses=[]
    for i in range(24):
        tic=time.perf_counter();opt.zero_grad(set_to_none=True)
        loss,stats=model(batch['x'],None,batch['y']);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.mps.synchronize()
        if not torch.isfinite(loss) or float(stats['kl_sum'])<0:raise RuntimeError('Nonfinite loss or negative stable KL')
        durations.append(time.perf_counter()-tic);losses.append(float(loss))
    if losses[-1]>=losses[0]-.01:raise RuntimeError('Fixed-batch fitting check did not improve')
    rng=np.random.default_rng(151803);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'resume_fixture.pt';identity=dict(preflight=True,codec_id=data.manifest['codec']['codec_id'])
    save(path,model,opt,24,rng,data,'code_gram',sum(durations),identity)
    def update(m,o):
        o.zero_grad(set_to_none=True);loss,_=m(batch['x'],None,batch['y']);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();torch.mps.synchronize()
        return float(loss)
    uninterrupted_loss=update(model,opt);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model,opt;gc.collect();torch.mps.empty_cache()
    ck=torch.load(path,map_location='mps',weights_only=True);resumed=make_model('code_gram',ck['config']).to('mps')
    resumed.load_state_dict(ck['model']);optimizer=torch.optim.AdamW(resumed.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,foreach=False)
    optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'].cpu());torch.mps.set_rng_state(ck['mps_rng'].cpu())
    resumed_loss=update(resumed,optimizer)
    maximum=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in resumed.state_dict().items())
    if maximum>1e-6 or abs(uninterrupted_loss-resumed_loss)>1e-5:raise RuntimeError('Stochastic MPS resume mismatch')
    fixed=torch.randn(2,4,16,192,device='mps')
    generated=resumed.generate(batch['x'],None,noise=fixed)
    infer=make_model('code_gram',ck['config'],inference=True).to('mps');infer.load_state_dict(resumed.inference_state_dict())
    if not torch.equal(generated,infer.generate(batch['x'],None,noise=fixed)):raise RuntimeError('Removing posterior changed prior inference')
    report=dict(passed=True,device='mps',dtype='float32',microbatch=4,accumulation_for_main=2,
        cpu_loss=reference,mps_loss=actual,cpu_mps_absolute_error=abs(reference-actual),
        initial_fixed_batch_loss=losses[0],final_fixed_batch_loss=losses[-1],updates=24,
        median_microbatch_update_seconds=statistics.median(durations[4:]),
        estimated_accumulated_step_seconds=2*statistics.median(durations[4:]),
        mps_stochastic_resume_max_abs_error=maximum,posterior_removed_inference_equal=True,
        driver_bytes_including_test_models=torch.mps.driver_allocated_memory(),
        dataset_sha=data.manifest['episodes_sha'],ledger=parameter_ledger(resumed),
        note='Throughput is a short benchmark; sustained fanless performance may differ. This fixture does not initialize the main model.')
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.data,args.out)
