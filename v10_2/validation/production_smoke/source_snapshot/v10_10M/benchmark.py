"""Measure actual end-to-end updates on each corpus family and batch size."""
import argparse
import json
import time
from pathlib import Path
import torch
from .model import V10, losses
from .data import Corpus
from .train import build_optimizers, clip, memory_stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='mps');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1011)
    model=V10().place(a.device).train();opt,table,dense=build_optimizers(model)
    corpus=Corpus();records=[]
    for bs in (2,4,8):
        for family in corpus.arrays:
            batch=corpus.batch(bs,family=family,device=a.device)
            for visit in range(4):
                opt.zero_grad(set_to_none=True);table.zero_grad(set_to_none=True)
                if a.device=='mps':torch.mps.synchronize()
                start=time.perf_counter()
                out=model(batch['ids'],batch['lengths'])
                n=int(out['valid_groups'].sum(-1).min())-1
                boundary=max(0,n//2) if visit%2==0 and n>0 else None
                l=losses(model,out,batch['targets'],batch['target_mask'],boundary)
                l['total'].backward();norm=clip(model,dense);opt.step();table.step()
                if a.device=='mps':torch.mps.synchronize()
                seconds=time.perf_counter()-start
                if visit:
                    records.append(dict(batch_size=bs,family=family,auxiliary=boundary is not None,
                                        seconds=seconds,tokens=int(batch['lengths'].sum()),
                                        tokens_per_second=int(batch['lengths'].sum())/seconds,
                                        loss=float(l['total'].detach()),**memory_stats(a.device)))
            print(bs,family,round(seconds,3),flush=True)
    path=Path(__file__).resolve().parent/'research'/f'benchmark_{a.device}.json'
    path.write_text(json.dumps(dict(device=a.device,records=records),indent=2));print(path)


if __name__=='__main__':main()
