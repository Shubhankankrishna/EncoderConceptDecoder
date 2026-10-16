"""Apply the registered rules without substituting secondary controls."""
import json
from pathlib import Path
import numpy as np
from compare_v41 import paired
from report_tables_v41 import agg
ROOT=Path(__file__).resolve().parent.parent;RUNS=ROOT/'runs'

def metric(c,k,m):return c[k]['metrics'].get(m)
def main():
    c=json.loads((RUNS/'v41_compare.json').read_text())
    r=metric(c,'pin_v41_minus_noR','heldout/act_opt');l=metric(c,'pin_v41_minus_LLM','heldout/act_opt')
    t=metric(c,'text_v41_minus_noR','row_ntp');stack=metric(c,'pin_stack_minus_v41','heldout/act_opt')
    loop=metric(c,'pin_loopoff_minus_v41','heldout/act_opt_b2')
    full=bool(r and l and r['n_seeds']==3 and l['n_seeds']==3 and t and t['n_seeds']==2 and stack and stack['n_seeds']==3 and loop and loop['n_seeds']==3)
    result={'registered_matrix_complete':full,'R_P1':r,'R_minus_LLM_P1':l,'text_guard':t,'stack_P1':stack,'trained_loopoff_b2':loop}
    if r and l:
        if l['mean']>=-2*l['paired_item_se']:outcome='O1 fix confirmed'
        elif r['mean']>=2*r['paired_item_se']:outcome='O2 partial fix'
        elif r['mean']<=-2*r['paired_item_se']:outcome='O4 harm'
        else:outcome='O3 no detectable effect'
        result['R_outcome']=outcome
    result['text_guard_passes']=bool(t and t['mean']<=.02)
    result['stack_improves_P1_by_2_item_SE']=bool(stack and stack['mean']>=2*stack['paired_item_se'])
    result['keep_F_loop_by_registered_rule']=bool(loop and loop['mean']<=-2*loop['paired_item_se'])
    result['loop_text_control']='not run; the registered rule therefore uses pinpad only'
    route={}
    for label,prefix in [('v41','v41_pin'),('stack','v41_stack_pin')]:
        per=[]
        for seed in range(3):
            p=RUNS/f'{prefix}_s{seed}_6M/per_unit.npz'
            if not p.exists():continue
            a=np.load(p);key='intervention/core_off/act_opt_b2'
            if key not in a.files:continue
            z=paired(a[key],a['heldout/act_opt_b2']);per.append(dict(seed=seed,**z))
        if per:
            v=agg(per);route[label]=dict(core_off_minus_intact_b2=v,formal_route_alive=v['mean']<=-2*v['paired_item_se'])
    result['registered_route_rule']=route
    curves={}
    for prefix in ['ref_vs_pin_llm','v41_pin','v41_stack_pin','v41_loopoff_pin']:
        for seed in range(3):
            p=RUNS/f'{prefix}_s{seed}_6M/log.jsonl'
            if not p.exists():continue
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            vals=[(v['real_tokens']/1e6,v['heldin']['act_opt_b2']) for v in rows if v['kind']=='eval'][-3:]
            curves[f'{prefix}_s{seed}']=dict(last_three=vals,strictly_rising=len(vals)==3 and vals[0][1]<vals[1][1]<vals[2][1])
    result['last_three_heldin_b2']=curves
    result['R_vs_LLM_budget_limited_seeds']=[seed for seed in range(3)
        if curves.get(f'ref_vs_pin_llm_s{seed}',{}).get('strictly_rising',False)]
    result['R_vs_LLM_budget_limited']=bool(result['R_vs_LLM_budget_limited_seeds'])
    (RUNS/'registered_decisions.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['last_three_heldin_b2','registered_route_rule']},indent=2))
if __name__=='__main__':main()
