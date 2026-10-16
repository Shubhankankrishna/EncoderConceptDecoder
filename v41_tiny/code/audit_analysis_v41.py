"""Check saved inference units, final checkpoints and diagnostic consistency.

This audits artifacts only; it never reruns training or changes its metrics.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiment_queue import JOBS

ROOT=Path(__file__).resolve().parent.parent;RUNS=ROOT/'runs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):
    assert np.isclose(a,b,rtol=0,atol=1e-10),(a,b)

def audit(name):
    p=RUNS/name
    final=json.loads((p/'final.json').read_text())
    analysis=json.loads((p/'analysis.json').read_text())
    checkpoint=torch.load(p/'ckpt.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==final['step']
    assert checkpoint['real_tokens']==final['real_tokens']
    assert checkpoint['args']==final['args']
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    checks=0;units=np.load(p/'per_unit.npz')
    if analysis['data']=='text':
        assert units['row_ntp'].shape==(64,)
        assert len(units['item_hit'])==576
        same(units['row_ntp'].mean(),analysis['val_ntp']);checks+=1
        for family in np.unique(units['item_family']):
            mask=units['item_family']==family
            assert mask.sum()==192
            same(units['item_hit'][mask].mean(),analysis['probes'][family]);checks+=1
    else:
        for section,n in (('heldin',192),('heldout',200),('long',64),('roll_temp1',256),('roll_greedy',256)):
            for key,value in analysis[section].items():
                vals=units[section+'/'+key]
                assert vals.shape==(n,),(name,section,key,vals.shape)
                same(np.nanmean(vals),value);checks+=1
        for mode,values in analysis.get('interventions',{}).items():
            for key,value in values.items():
                vals=units['intervention/'+mode+'/'+key]
                assert vals.shape==(200,)
                same(np.nanmean(vals),value);checks+=1
        cf=p/'counterfactual.json'
        if cf.exists():
            fx=json.loads((RUNS/'counterfactual_fixtures.json').read_text())
            z=np.load(p/'counterfactual_units.npz')
            assert np.array_equal(z['plan_lag'],[len(f['original'])%6 for f in fx])
            for mode,values in json.loads(cf.read_text()).items():
                a,b=z[mode+'/original_action'],z[mode+'/changed_action']
                good=z[mode+'/original_correct'];changed=z[mode+'/changed_correct']
                assert len(a)==len(b)==len(good)==len(changed)==128
                assert np.array_equal(good,[x in f['original_opt'] for x,f in zip(a,fx)])
                assert np.array_equal(changed,[x in f['changed_opt'] for x,f in zip(b,fx)])
                same(good.mean(),values['original_accuracy'])
                same(changed.mean(),values['changed_accuracy'])
                same((a!=b).mean(),values['prediction_changes']);checks+=3
    diag=p/'plan_diagnostics.json'
    if diag.exists():
        z=np.load(p/'plan_diagnostic_units.npz');j=json.loads(diag.read_text())
        for label,short in (('intact','intact'),('constant_training_mean','constant'),
                            ('zero_content_keep_nulls_offsets','zero_content'),('core_off_null_everywhere','null')):
            for key,value in j[label].items():
                same(np.nanmean(z[short+'/'+key]),value);checks+=1
        # The unchanged intact evaluation must refer to the same rows/episodes.
        keys=('row_ntp',) if analysis['data']=='text' else ('act_opt','act_opt_b2','ntp_action')
        for key in keys:
            raw=units[key if key=='row_ntp' else 'heldout/'+key]
            assert np.allclose(z['intact/'+key],raw,rtol=0,atol=1e-6,equal_nan=True)
            checks+=1
    return dict(finite_state_tensors=len(checkpoint['model']),checked_aggregates=checks,
                checkpoint_sha256=sha(p/'ckpt.pt'),units_sha256=sha(p/'per_unit.npz'))

if __name__=='__main__':
    torch.set_num_threads(1)
    names=[j[0] for j in JOBS]+['exploratory_noplan_pin_s0_6M','exploratory_mps_plan_pin_s0_6M']
    pending=[];results={}
    for name in names:
        if not all((RUNS/name/f).exists() for f in ('final.json','analysis.json','per_unit.npz')):
            pending.append(name);continue
        results[name]=audit(name)
    out=dict(pending=pending,runs=results,all_complete=not pending)
    (RUNS/'analysis_artifact_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(completed=len(results),pending=pending,
                         finite_state_tensors=sum(v['finite_state_tensors'] for v in results.values()),
                         checked_aggregates=sum(v['checked_aggregates'] for v in results.values())),indent=2))
