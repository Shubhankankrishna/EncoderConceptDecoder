"""Alternating synchronized full-checkpoint update timings; no live-run writes."""
import gc,json,time,statistics
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.train import build_optimizers
from v10_1.model import V101
from v10_1.accumulate import backward_effective_batch
from v10_1.train import gradient_metrics as reference_metrics
from .fast import backward_full_batch,gradient_metrics

DEST=Path('v10_1/validation/throughput_review')

def main():
 torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.45)
 corpus=Corpus('train',seed=1927,data_dir='v10_1/data');cases=[]
 for family in ('text','math','sudoku_easy'):
  raw=corpus.batch(8,384,family=family)
  for candidate in (False,True):cases.append((family,candidate,raw))
 results=[]
 for block,engine in enumerate(('reference','fast','foreach','foreach','fast','reference')):
  ck=torch.load('v10_1/runs/pretrain_16h/latest.pt',map_location='cpu',weights_only=False)
  m=V101(ModelConfig(**ck['model_config']));m.load_state_dict(ck['model']);m.place('mps').train()
  if engine!='reference':m.engram.canonical=m.engram.canonical.cpu()
  opt,table,dense=build_optimizers(m);opt.load_state_dict(ck['optimizer']);table.load_state_dict(ck['engram_optimizer']);del ck
  if engine=='foreach':
   for group in opt.param_groups:group['foreach']=True
  device_cases=[{k:raw[k].to('mps') for k in ('ids','lengths','targets','target_mask')} for _,_,raw in cases]
  def update(i,detailed):
   family,candidate,raw=cases[i];opt.zero_grad(set_to_none=True);table.zero_grad(set_to_none=True);diag={}
   if engine=='reference':parts=backward_effective_batch(m,device_cases[i],8,candidate,718+i,diag)
   else:parts=backward_full_batch(m,raw,'mps',candidate,718+i,diag)
   grads=(reference_metrics if engine=='reference' else gradient_metrics)(m,dense,detailed);opt.step();table.step()
   return parts,grads
  for i in range(len(cases)):update(i,True)
  for repeat in range(2):
   for i,(family,candidate,raw) in enumerate(cases):
    torch.mps.synchronize();start=time.perf_counter();parts,grads=update(i,repeat==1 and family=='text');torch.mps.synchronize();dt=time.perf_counter()-start
    event=dict(block=block,engine=engine,family=family,candidate=candidate,detailed=repeat==1 and family=='text',seconds=dt,input_tokens=int(raw['lengths'].sum()),driver_gib=torch.mps.driver_allocated_memory()/2**30)
    results.append(event);print(json.dumps(event),flush=True)
  if block in (0,1):
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:update(0,True);torch.mps.synchronize()
   (DEST/f'profile_{engine}.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=30))
  del m,opt,table,dense,device_cases;gc.collect();torch.mps.empty_cache()
  (DEST/'benchmark.json').write_text(json.dumps(dict(scope='Full saved checkpoint, synchronized identical batches, ABCCBA ordering; brief nonthermal benchmark',events=results),indent=2)+'\n')
 summary={}
 for engine in ('reference','fast','foreach'):
  es=[e for e in results if e['engine']==engine];summary[engine]={}
  for family in ('text','math','sudoku_easy'):
   for candidate in (False,True):
    a=[e['seconds'] for e in es if e['family']==family and e['candidate']==candidate];summary[engine][f'{family}:{candidate}']=statistics.median(a)
 (DEST/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
