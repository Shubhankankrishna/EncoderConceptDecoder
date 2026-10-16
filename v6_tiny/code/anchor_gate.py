"""Development competence screen before adapter training; confirmation remains separate."""
import argparse
import numpy as np
import torch
from model import freeze
from runtime import load_system,atomic_json
from evaluate import likelihood
from generation import generate
from data import DATA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--foundation',required=True);ap.add_argument('--anchor',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(4)
    values={};graphs=np.load(DATA/'dev_standard_s0.npz')['x'][:128]
    for name,path in [('foundation',a.foundation),('anchor',a.anchor)]:
        s,_=load_system(path);freeze(s);nll=likelihood(s,'dev',128)
        flags=[]
        for start in range(0,len(graphs),16):flags.extend(v['success'] for v in generate(s,graphs[start:start+16])['verification'])
        values[name]=dict(nll={d:nll[d]['nll'] for d in ('text','graph')},standard_success=float(np.mean(flags)))
    gaps={d:values['anchor']['nll'][d]-values['foundation']['nll'][d] for d in ('text','graph')}
    success_loss=values['foundation']['standard_success']-values['anchor']['standard_success']
    passed=all(v<=.02 for v in gaps.values()) and success_loss<=.02
    atomic_json(a.out,dict(passed_mean_screen=passed,nll_gaps=gaps,standard_success_loss=success_loss,measurements=values,
                         caveat='One seed and development means; not confirmation of confidence-bound gates'))


if __name__=='__main__':main()
