"""Actual production-shape MPS checks, including optimizer and RNG recovery."""
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from ..controls import make_model, parameter_ledger
from ..isolation import ROOT, atomic_json, exclusive_job
from ..train import save
from .codec import LEVELS, load_codec
from .data import DATA, REPORT, ShortDataset
from .evaluate import evaluate


def write_configs():
    directory=REPORT/'configs'
    directory.mkdir(parents=True,exist_ok=True)
    for name,path in (('10m',ROOT/'configs/gram_shared.json'),
        ('30m',ROOT/'reports/scaling_20261019/candidate_configs/30m.json')):
        config=json.loads(path.read_text())
        config.update(levels=list(LEVELS),context_source='codes',context_pooling='hybrid')
        atomic_json(directory/f'{name}.json',config)
    return directory


@exclusive_job
def preflight(data_path=DATA):
    torch.set_num_threads(1)
    torch.mps.set_per_process_memory_fraction(.5)
    data=ShortDataset(data_path)
    directory=write_configs()
    results={}
    for name in ('30m','10m'):
        torch.manual_seed(151803)
        config=json.loads((directory/f'{name}.json').read_text())
        model=make_model('code_gram',config)
        ids=data.indices['train'][:4]
        cpu=data.batch(ids)
        noise=torch.randn(2,4,16,config['width'])
        with torch.no_grad():reference=float(model(cpu['x'],None,cpu['y'],noise=noise)[0])
        model.to('mps');batch=data.batch(ids,'mps')
        with torch.no_grad():actual=float(model(batch['x'],None,batch['y'],noise=noise.to('mps'))[0])
        if abs(reference-actual)>1e-4:raise RuntimeError(f'{name} CPU/MPS objective mismatch')
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,foreach=False)
        durations=[];losses=[]
        def update(m,o,accum=1):
            tic=time.perf_counter();o.zero_grad(set_to_none=True);value=0.
            for _ in range(accum):
                loss,stats=m(batch['x'],None,batch['y'])
                (loss/accum).backward();value+=float(loss.detach())/accum
                if float(stats['kl_sum'])<0 or not torch.isfinite(loss):raise RuntimeError('Invalid ELBO')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            o.step();torch.mps.synchronize()
            return value,time.perf_counter()-tic
        for _ in range(16):
            loss,duration=update(model,opt,accum=2)
            losses.append(loss);durations.append(duration)
        if losses[-1]>=losses[0]-.01:raise RuntimeError(f'{name} fixed-batch fitting failed')
        out=REPORT/f'preflight_{name}';out.mkdir(parents=True,exist_ok=True)
        fixture=out/'resume_fixture.pt'
        save(fixture,model,opt,16,np.random.default_rng(151803),data,'code_gram',sum(durations),dict(preflight=True))
        expected_loss,_=update(model,opt)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        del model,opt;gc.collect();torch.mps.empty_cache()
        ck=torch.load(fixture,map_location='mps',weights_only=True)
        resumed=make_model('code_gram',config).to('mps')
        resumed.load_state_dict(ck['model'])
        optimizer=torch.optim.AdamW(resumed.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,foreach=False)
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng'].cpu());torch.mps.set_rng_state(ck['mps_rng'].cpu())
        resumed_loss,_=update(resumed,optimizer)
        difference=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in resumed.state_dict().items())
        if difference>1e-6 or abs(expected_loss-resumed_loss)>1e-5:
            raise RuntimeError(f'{name} stochastic checkpoint recovery mismatch: {difference}')
        fixed=torch.randn(2,4,16,config['width'],device='mps')
        expected_codes=resumed.generate(batch['x'],None,noise=fixed)
        inference=make_model('code_gram',config,inference=True).to('mps')
        inference.load_state_dict(resumed.inference_state_dict(),strict=True)
        if not torch.equal(expected_codes,inference.generate(batch['x'],None,noise=fixed)):
            raise RuntimeError('Posterior removal changed generation')
        del inference,expected,ck,optimizer;gc.collect();torch.mps.empty_cache()
        codec=load_codec(data.manifest['codec'])
        # Includes rolling feedback encode/decode, shuffled-context control,
        # categorical sampling and text measurement under the real codec.
        smoke=evaluate(resumed,data,codec,n=2,chunks=2)
        atomic_json(out/'evaluation_smoke.json',smoke)
        results[name]=dict(passed=True,ledger=parameter_ledger(resumed),cpu_mps_loss_error=abs(reference-actual),
            initial_fixed_batch_loss=losses[0],final_fixed_batch_loss=losses[-1],
            median_accumulated_update_seconds=statistics.median(durations[4:]),
            exact_resume_max_abs_error=difference,posterior_removed_inference_equal=True,
            driver_bytes=torch.mps.driver_allocated_memory(),available_ram_bytes=__import__('psutil').virtual_memory().available,
            evaluation_smoke_passed=True,context_memory_vectors=8*(len(LEVELS)+32))
        print(json.dumps(dict(arm=name,**results[name])),flush=True)
        del resumed,codec,batch,cpu;gc.collect();torch.mps.empty_cache()
    # Exercise the actual new trainer across process-style reconstruction,
    # not only the shared checkpoint helper. The outer preflight owns the
    # exclusive job lock throughout these two bounded calls.
    from .train import parser,train
    arguments=parser().parse_args(['--run',str(REPORT/'cli_resume'/str(time.time_ns())/'gram'),
        '--config',str(directory/'10m.json'),'--data',str(data_path),
        '--steps','10','--schedule-steps','100','--total-seconds','120',
        '--slice-seconds','1','--deadline',str(time.time()+240),'--eval-n','1'])
    train.__wrapped__(arguments)
    path=arguments.run/'latest.pt'
    first=torch.load(path,map_location='cpu',weights_only=True)
    arguments.slice_seconds=30
    train.__wrapped__(arguments)
    second=torch.load(path,map_location='cpu',weights_only=True)
    if first['manifest_identity']!=second['manifest_identity'] or second['step']<=first['step']:
        raise RuntimeError('Trainer time-slice recovery did not advance under the same identity')
    replay=np.random.default_rng()
    replay.bit_generator.state=first['numpy_rng']
    for _ in range((second['step']-first['step'])*arguments.accum):
        replay.choice(data.indices['train'],arguments.batch,replace=False)
    if replay.bit_generator.state!=second['numpy_rng']:
        raise RuntimeError('Trainer resume changed the data-sampling sequence')
    cli_resume=dict(passed=True,first_step=first['step'],resumed_step=second['step'],numpy_sequence_verified=True)
    del first,second;gc.collect();torch.mps.empty_cache()
    report=dict(passed=True,arms=results,dataset_sha=data.manifest['episodes_sha'],
        codec_id=data.manifest['codec']['codec_id'],microbatch=4,accumulation=2,
        target_ids=32,context_ids=256,context_pooling='hybrid; finest positions plus level summaries',
        actual_trainer_slice_resume=cli_resume,
        note='Fresh models in the main experiment; preflight weights never initialize training')
    atomic_json(REPORT/'preflight.json',report)
    return report


if __name__=='__main__':preflight()
