"""Historical checkpoint replay, not an independent replication or V8 training seed."""
import json
import numpy as np
import torch
from runtime import *
from model import freeze
from generation import generate
from evaluate import wilson

def main():
    torch.set_num_threads(4);out={}
    sources={'alpha_zero':'A_no_forecast_prefix_s0/checkpoint.pt','historical_ncp':'A_ncp_s0/checkpoint_step_9000.pt'}
    for arm,suffix in sources.items():
        path=ROOT.parent/'v6_tiny/runs'/suffix;s,_=load_system(path);freeze(s);s.fast_generation=True;results={}
        for name,n in [('standard',64),('hard',128)]:
            arr=np.load(ROOT.parent/f'v6_tiny/data/dev_{name}_s0.npz')['x'][:n];success=[];records=[]
            for i in range(0,n,32):
                r=generate(s,arr[i:i+32,:124],max_new=68);success.extend(bool(v['success']) for v in r['verification']);records.extend(dict(index=i+j,**v,response=r['x'][j,124:int(r['lengths'][j])].tolist()) for j,v in enumerate(r['verification']))
            results[name]=dict(n=n,successes=sum(success),rate=sum(success)/n,interval=wilson(sum(success),n),records=records)
        out[arm]=dict(source=str(path),results=results,checkpoint_sha256=__import__('data').digest(path),status='historical_replay')
    atomic_json(ROOT/'tests/historical_bridge.json',dict(arms=out,scope='Previously trained V6 checkpoints on inspected V6 development data; no fresh-training or confirmatory claim'))
    print(json.dumps({arm:{name:v['rate'] for name,v in row['results'].items()} for arm,row in out.items()}))
if __name__=='__main__':main()
