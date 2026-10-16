"""Decompose long-episode action NLL into action choice and vocabulary mass."""
import json
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A,data_v4 as D,pinpad as PP,run_vs as R
ROOT=Path(__file__).resolve().parent.parent;RUNS=ROOT/'runs'

@torch.no_grad()
def run(name):
    m,args=A.load(name);eps=R.pin_eval_sets()['long']
    t,d,where=D.pack_sequences([e['tokens'] for e in eps],480,6)
    out={zone+'/'+key:np.full(len(eps),np.nan) for zone in ('before_visit6','after_visit6')
         for key in ('ntp_all_vocabulary','ntp_conditioned_on_action','nonaction_mass_penalty','END_probability','action_probability_mass','optimal_accuracy')}
    by={}
    for i,(row,off,L) in enumerate(where):by.setdefault(row,[]).append((i,off,L))
    err=0.
    for start in range(0,len(t),8):
        ids,doc,ei=D.to_batch(t[start:start+8],d[start:start+8],PP.Ident(),m.ecfg)
        logits=R.fwd(m,ids,doc,ei)[0].double()
        lp=logits.log_softmax(-1);la=logits[...,PP.ACT0:PP.ACT0+4].log_softmax(-1)
        mass=lp[...,PP.ACT0:PP.ACT0+4].logsumexp(-1)
        for bi,row in enumerate(range(start,min(start+8,len(t)))):
            for epi,off,L in by.get(row,[]):
                ep=eps[epi];stage=-1;prev=None;vals={z:{k:[] for k in ('all','cond','penalty','END','mass','acc')} for z in ('before_visit6','after_visit6')}
                for pos in range(27,L):
                    if ep['kind'][pos]!=1:continue
                    target=ep['tgt'][pos]
                    if target!=prev:stage+=1;prev=target
                    zone='before_visit6' if stage<6 else 'after_visit6'
                    j=off+pos-1;action=ep['tokens'][pos]-PP.ACT0
                    all_nll=-float(lp[bi,j,PP.ACT0+action]);cond=-float(la[bi,j,action]);penalty=-float(mass[bi,j])
                    err=max(err,abs(all_nll-cond-penalty))
                    v=vals[zone];v['all'].append(all_nll);v['cond'].append(cond);v['penalty'].append(penalty)
                    v['END'].append(float(lp[bi,j,PP.END].exp()));v['mass'].append(float(mass[bi,j].exp()))
                    v['acc'].append(int(logits[bi,j,PP.ACT0:PP.ACT0+4].argmax()) in ep['opt'][pos])
                for zone,v in vals.items():
                    for key,short in zip(('ntp_all_vocabulary','ntp_conditioned_on_action','nonaction_mass_penalty','END_probability','action_probability_mass','optimal_accuracy'),('all','cond','penalty','END','mass','acc')):
                        out[zone+'/'+key][epi]=np.mean(v[short])
    assert err<1e-10,err
    np.savez(RUNS/name/'long_audit_units.npz',**out)
    result={'seed':args['seed'],'n_episodes':len(eps),'NLL_decomposition_max_error':err,'metrics':{k:float(np.nanmean(v)) for k,v in out.items()}}
    (RUNS/name/'long_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    names=[f'{prefix}_s{s}_6M' for prefix in ('v41_pin','ref_vs_pin_llm','ref_vs_pin_v4') for s in range(3)]
    names+=['exploratory_mps_plan_pin_s0_6M','exploratory_noplan_pin_s0_6M']
    lines=['# Long-action likelihood decomposition','','Per-episode means on the same 64 long episodes. Before/after refers to six completed visits, the training horizon.','',
           '| Model | Seed | Phase | Optimal accuracy | All-vocabulary NLL | Action-conditioned NLL | Non-action penalty | END probability | Action probability mass |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        if not (RUNS/n/'final.json').exists():continue
        p=RUNS/n/'long_audit.json'
        j=json.loads(p.read_text()) if p.exists() else run(n)
        for zone in ('before_visit6','after_visit6'):
            v=j['metrics'];keys=('optimal_accuracy','ntp_all_vocabulary','ntp_conditioned_on_action','nonaction_mass_penalty','END_probability','action_probability_mass')
            lines.append(f'| {n} | {j["seed"]} | {zone} | '+' | '.join(f'{v[zone+"/"+key]:.4f}' for key in keys)+' |')
    (ROOT/'LONG_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'LONG_AUDIT.md')
