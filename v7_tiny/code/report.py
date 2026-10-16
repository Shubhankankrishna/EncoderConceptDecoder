"""Evidence-only reporting; incomplete endpoints and quality gates remain explicit."""
import json,time
from pathlib import Path
import numpy as np
from runtime import atomic_json
from evaluate import paired,weighted_nll_pair

def read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None
def f(v,d=3):return 'pending' if v is None else f'{v:.{d}f}'
def build_report(root,out,final_status=None):
    root=Path(root);out=Path(out);plan=read(root/'plan.json') or {};lock=read(root/'training_lock.json') or {};ilock=read(root/'inference_lock.json') or {};status=read(root/'status.json') or {};lin=read(root/'lineages.json') or {}
    if final_status is not None:status['status']=final_status
    rows=[];primary=dict(status=status.get('status'),model_parameters=5365938,flat_parameters=5366208,exploration_parameters=41472,foundation_comparisons={},search={},limits=status.get('skipped',[]),smoke=plan.get('smoke',False))
    text=['# V7 tiny experiment report','',f'Generated UTC: {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}. Observed queue status: {status.get("status","pending")}.',
        '', 'This is a bounded tiny-model study. Numerical/caching smoke checks establish implementation contracts; they do not establish model quality or scaling.',
        '',f'V7 foundation: **5,365,938** active parameters. Ordinary Transformer: **5,366,208** (270 more, 0.0050%). Exploration adds **41,472** parameters; deployment has **5,407,410**. A resident frozen teacher is excluded from active deployment counts.',
        '',f'Common foundation budget: **{plan.get("common_foundation_target_tokens","pending"):,}** scored targets per arm and seed.' if plan.get('common_foundation_target_tokens') else '',
        '', '| Run | Completed scored targets | Train status | Fresh text NLL | Standard strict | Hard strict |', '|---|---:|---|---:|---:|---:|']
    for path in sorted(root.glob('*/train_summary.json')):
        name=path.parent.name;tr=read(path);test=read(path.parent/'test.json');dev=read(path.parent/'dev.json');ev=test or dev;scope='test' if test else 'dev'
        rows.append(dict(name=name,train=tr,evaluation=ev,evaluation_split=scope))
        if ev:
            st=ev['symbolic'].get('standard',{}).get('success_rate');hard=ev['symbolic'].get('hard',{}).get('success_rate');tn=ev['text']['nll']
        else:st=hard=tn=None
        text.append(f'| {name} ({scope}) | {tr["completed_targets"]:,} | {tr["status"]} | {f(tn)} | {f(st*100 if st is not None else None,1)}% | {f(hard*100 if hard is not None else None,1)}% |')
    text+=['','Matched foundation differences use whole held-out document resampling for token-weighted NLL, and paired complete problems for task success. Positive task differences favor the first arm; negative NLL differences favor it.']
    for seed in plan.get('foundation_seeds',[]):
        for a,b in [('ncp','flat'),('ncp','alpha0'),('alpha0','flat')]:
            ea=read(root/f'A_{a}_s{seed}/test.json');eb=read(root/f'A_{b}_s{seed}/test.json');ta=read(root/f'A_{a}_s{seed}/train_summary.json');tb=read(root/f'A_{b}_s{seed}/train_summary.json')
            if not all([ea,eb,ta,tb]):continue
            key=f'{a}-{b}_s{seed}';matched=ta['status']=='completed' and tb['status']=='completed' and ta['target_budget']==tb['target_budget']
            comp=dict(matched_complete_endpoints=matched,text=weighted_nll_pair(ea['text'],eb['text']),standard=paired(ea['symbolic']['standard']['success'],eb['symbolic']['standard']['success']),hard=paired(ea['symbolic']['hard']['success'],eb['symbolic']['hard']['success']))
            comp['text_preserved']=matched and comp['text']['ci95'][1]<=.02;comp['standard_preserved']=matched and comp['standard']['ci95'][0]>=-.02
            comp['quality_gain_supported']=comp['text_preserved'] and comp['standard_preserved'] and (comp['text']['ci95'][1]<0 or comp['standard']['ci95'][0]>0)
            primary['foundation_comparisons'][key]=comp
            text.append(f'\n`{key}`: text NLL difference {f(comp["text"]["difference"],4)}, 95% CI {comp["text"]["ci95"]}; standard difference {f(100*comp["standard"]["difference"],2)} percentage points, CI {[round(100*v,2) for v in comp["standard"]["ci95"]]}. Complete matched endpoints: {matched}.')
    text+=['','## Refit, exploration and deployment','']
    for seed,l in lin.get('lineages',{}).items():
        acceptance=read(Path(l['refit']).parent/'acceptance.json');au=read(Path(l['adapters']).parent/'audit.json');text.append(f'Seed {seed}: refit accepted = {l["refit_accepted"]}; zero-control anchor = `{Path(l["anchor"]).parent.name}`. Acceptance evidence: {json.dumps(acceptance)}.')
        if au:text.append(f'Anchor errors {au["anchor_errors"]}; control gradient norm {au["control_gradient_norm"]}; trained cache error {au["cache_max_abs_error"]}; deployment reload error {au["reload_max_abs_error"]}.')
    text+=['','## GradCuit and AB-MCTS','',f'Development lock: {json.dumps(ilock)}','']
    for path in sorted(root.glob('test*/summary.json')):
        se=read(path);primary['search'][path.parent.name]=se;text.append(f'### {path.parent.name}\n\nStatus: {se["status"]}; warm-up {f(se["warmup_seconds"])} seconds; timed arms charge prefill, token generation, verification, gradients, and bookkeeping. Completed attempts after deadlines receive no success credit.\n')
        text+=['| Method and budget | Problems | Success | Charged elapsed seconds |','|---|---:|---:|---:|']
        for name,arm in se['arms'].items():text.append(f'| {name} | {arm["n"]} | {f(100*arm["success_rate"] if arm["success_rate"] is not None else None,2)}% | {f(arm["total_seconds"],1)} |')
        for name,c in se['comparisons'].items():
            text.append(f'\n{name}: {f(100*c["difference"],2)} percentage points; 95% CI {[round(100*v,2) for v in c["ci95"]]}. '+('Positive paired difference supported on this seed.' if c['ci95'][0]>0 else 'Inconclusive at this sample and budget.' if c['ci95'][0]<=0<=c['ci95'][1] else 'Underperforms retry on this seed.'))
        # AB allocation must also beat the same kernels under simple allocation.
        keys=se['arms']
        for key,arm in keys.items():
            if not key.startswith('ab_hybrid@'):continue
            for control in ('hybrid_random','hybrid_chain'):
                other=keys.get(control+'@'+key.split('@')[1])
                if other:
                    aa=dict(zip(arm['items'],arm['success']));bb=dict(zip(other['items'],other['success']));ix=sorted(aa.keys()&bb.keys())
                    if ix:
                        c=paired([aa[i] for i in ix],[bb[i] for i in ix]);se['comparisons'][f'ab_hybrid-{control}@'+key.split('@')[1]]=c
                        text.append(f'\nAB-hybrid minus {control}: {f(100*c["difference"],2)} percentage points; CI {[round(100*v,2) for v in c["ci95"]]}.')
    completed=[r for r in rows if r['name'].startswith('A_') and r['train']['status']=='completed']
    primary['search_vs_transformer']={}
    for seed in plan.get('foundation_seeds',[]):
        v=read(root/f'test_search_s{seed}/summary.json');b=read(root/f'test_flat_retry_s{seed}/summary.json')
        if v and b:
            for key,arm in v['arms'].items():
                base=b['arms'].get('retry@'+key.split('@')[1])
                if not base:continue
                aa=dict(zip(arm['items'],arm['success']));bb=dict(zip(base['items'],base['success']));ix=sorted(aa.keys()&bb.keys())
                if ix:
                    comp=paired([aa[i] for i in ix],[bb[i] for i in ix]);primary['search_vs_transformer'][f'{key}_s{seed}']=comp
                    text.append(f'\n{key}, seed {seed}, versus ordinary Transformer retry at the same wall budget: {f(100*comp["difference"],2)} percentage points; CI {[round(100*x,2) for x in comp["ci95"]]}. Exploration adds 41,472 parameters to V7; this additional capacity/cost is disclosed.')
    seed_sets={family:{r['train']['seed'] for r in completed if r['name'].startswith('A_'+family+'_')} for family in ('ncp','alpha0','flat')}
    primary['complete_foundation_seeds']={k:sorted(v) for k,v in seed_sets.items()};primary['confirmation_scope']='Three complete independently trained foundations required; search RNG repetitions alone do not count.'
    primary['training_elapsed_seconds']=sum(r['train']['wall_s'] for r in rows);primary['selected_training']=lock;primary['lineages']=lin
    text+=['','## Limits and interpretation','', 'NCP uses feature-mean forecast/VQ losses and bounded routing adaptations. GradCuit uses bounded concept controls and response-mean rejected log-probability descent, rather than the paper’s raw hidden prefix/sequence-sum objective. AB-MCTS-A searches realized complete answers; latent-only dynamics and prefix-tree MCTS are outside this run.',
        '', 'Three separately trained seeds are listed individually; problem-level bootstrap intervals are conditional on each trained model. Larger widths, long-context behavior, BF16, distributed training and LLM-scale gains are not established. Semantic deduplication and graph-isomorphism independence are not claimed.',
        '', 'Token budgets are matched; parameter counts are almost exact. Additional compute for the V7 recipe, tuning and inference is recorded rather than treated as free. A partial endpoint cannot support a matched-completion comparison. Inference port interventions are distribution-shift diagnostics, not retrained ablations.']
    for item in primary['limits']:text.append(f'\n- {item["name"]}: {item["reason"]}')
    if plan.get('smoke'):text.append('\n**This report is a smoke pipeline test only. No quality conclusion is permitted.**')
    failure=read(root/'failure.json')
    if failure:primary['failure']=failure;text.append('\nFailure: '+json.dumps(failure))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(text)+'\n');atomic_json(out.with_name('PRIMARY_RESULT.json'),primary);return primary

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();build_report(a.root,a.out)
