"""Audit completed artifacts, budgets, provenance and finite training logs."""
import hashlib,json,math
from pathlib import Path
from experiment_queue import JOBS
from secondary_queue import jobs as diagnostic_jobs

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parent;RUNS=ROOT/'runs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=json.loads((ROOT/'source_manifest.json').read_text())
    unchanged={name:sha(Path(source['source'])/name)==digest for name,digest in source['sha256'].items()}
    refs=json.loads((ROOT/'baseline_manifest.json').read_text())
    ref_ok={path:sha(REPO/path)==r['sha256'] and sha(Path(r['source']))==r['sha256'] for path,r in refs.items()}
    names=[j[0] for j in JOBS]+['exploratory_noplan_pin_s0_6M','exploratory_mps_plan_pin_s0_6M']
    ready=[];pending=[];bad=[];total=0;steps=0
    for name in names:
        p=RUNS/name
        if not (p/'final.json').exists():pending.append(name);continue
        j=json.loads((p/'final.json').read_text());a=j['args']
        assert a['real_tokens']<=j['real_tokens']<a['real_tokens']+a['B']*a['T'],name
        assert j['step']>0 and (p/'ckpt.pt').stat().st_size>1e6,name
        total+=j['real_tokens'];steps+=j['step']
        rows=[json.loads(line) for line in (p/'log.jsonl').read_text().splitlines()]
        trains=[r for r in rows if r['kind']=='train']
        for r in trains:
            for k in ('ntp','ce_coarse','ce_fine','vq_coarse','vq_fine','gn'):
                if k in r and not math.isfinite(r[k]):bad.append((name,r['step'],k))
        analysis=(p/'analysis.json').exists() and (p/'per_unit.npz').exists()
        ready.append(dict(name=name,real_tokens=j['real_tokens'],steps=j['step'],wall=j['wall'],
                          device=a.get('device','cpu'),threads=a['threads'],analysis_complete=analysis))
    # Cumulative token counts at every logged training step are an additional
    # whole-run data-order check, not a replacement for byte-wise runner tests.
    count_checks={}
    for name in names:
        p=RUNS/name/'final.json'
        if not p.exists():continue
        a=json.loads(p.read_text())['args'];seed=a['seed'];data=a['data']
        ref=f'ref_vs_pin_llm_s{seed}_6M' if data=='pinpad' else f'ref_vs_text_llm_s{seed}'
        def counts(n):
            return {r['step']:r['real_tokens'] for line in (RUNS/n/'log.jsonl').read_text().splitlines()
                    if (r:=json.loads(line))['kind']=='train'}
        x,y=counts(name),counts(ref);common=x.keys()&y.keys()
        count_checks[name]=dict(shared_logged_steps=len(common),all_counts_equal=all(x[k]==y[k] for k in common))
    required=[(name,target) for name,script,target in diagnostic_jobs]
    required += [(f'ref_vs_pin_v4_s{seed}_6M',f) for seed in range(3)
                 for f in ('plan_diagnostics.json','counterfactual_units.npz')]
    required += [(f'{prefix}_s{seed}_6M','long_audit.json')
                 for prefix in ('v41_pin','ref_vs_pin_v4','ref_vs_pin_llm') for seed in range(3)]
    required += [(name,'long_audit.json') for name in names[-2:]]
    required += [('v41_pin_s0_6M','control.json'),('v41_pin_s0_6M','gradient_audit.json')]
    missing=[f'{name}/{target}' for name,target in required if not (RUNS/name/target).exists()]
    audit_file=RUNS/'analysis_artifact_audit.json'
    audit=json.loads(audit_file.read_text()) if audit_file.exists() else {}
    audit_complete=bool(audit.get('all_complete') and set(names)<=set(audit.get('runs',{})))
    result=dict(original_code_files=len(unchanged),original_code_unchanged=all(unchanged.values()),
                reference_artifacts=len(ref_ok),reference_copies_and_originals_unchanged=all(ref_ok.values()),
                completed_full_budget_runs=ready,pending_full_budget_runs=pending,
                all_completed_analyses=all(r['analysis_complete'] for r in ready),
                new_full_budget_real_tokens=total,new_full_budget_optimizer_steps=steps,
                nonfinite_training_records=bad,logged_training_token_count_checks=count_checks,
                expected_secondary_artifacts=len(required),pending_secondary_artifacts=missing,
                all_secondary_diagnostics_complete=not missing,
                analysis_artifact_audit_complete=audit_complete)
    assert result['original_code_unchanged'] and result['reference_copies_and_originals_unchanged']
    assert not bad and all(v['all_counts_equal'] for v in count_checks.values())
    (RUNS/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'final_source_manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py'))},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['completed_full_budget_runs','logged_training_token_count_checks']},indent=2))
if __name__=='__main__':main()
