"""Build a readable progress/final report strictly from saved evidence."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from runtime import ROOT,atomic_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    plan=read(ROOT/'PLAN.json') or {};state=read(ROOT/'runs'/'status.json') or {};bench=read(ROOT/'runs'/'calibration'/'benchmark.json') or {}
    lines=['# V6 tiny experiment report','',f"Updated: {datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds')}",'',
           f"Status: **{state.get('status','preparing')}**. Current stage: {state.get('stage','prelaunch checks')}.",'',
           'This is a 2.064M-parameter development experiment on one foundation/core seed. Independent foundation-seed confirmation and larger-scale stability remain untested. Text test documents are disjoint from training/dev after exact token-content deduplication, but come from the previously inspected V5 validation corpus.','',
           '## Correctness and throughput','']
    checks=read(ROOT/'tests'/'optimized_checks.json') or read(ROOT/'tests'/'checks.json') or {}
    results=checks.get('results',[]);lines.append(f"Executable contract groups: {sum(r['status']=='passed' for r in results)}/{len(results)} passed.")
    for entry in results:
        if entry['status']!='passed':lines.append(f"- FAILED: {entry['name']}: {entry.get('error')}")
    lines+=['','| Stage | Warm median step | Input slots/sec | Eager/compiled gradient error |','|---|---:|---:|---:|']
    for name,r in bench.get('training',{}).items():lines.append(f"| {name} | {r['median_step_s']:.3f}s | {r['input_slots_per_s']:,.0f} | {r['gradient_relative_error']:.2g} |")
    lines+=['',f"Selected generation execution: {bench.get('selected_generation','pending')}. Warm timings exclude compilation; long-run training logs include it. Input slots include padding and unscored headers. Actual targets and unique-data exposure are recorded in each stage summary.",'',
            '## Training','', '| Run | Status | Completed/planned steps | Active parameters | Wall hours | Text NLL / graph NLL at last logged update |','|---|---|---:|---:|---:|---|']
    for path in sorted((ROOT/'runs').glob('*/train_summary.json')):
        r=read(path);last=r.get('last') or {};text=last.get('text');graph=last.get('graph')
        scores=f"{text:.4f} / {graph:.4f}" if text is not None and graph is not None else 'selection objective; see held-out NLL'
        lines.append(f"| {path.parent.name} | {r['status']} | {r['completed_steps']}/{r['planned_steps']} | {r['active_parameters']:,} | {r['wall_s']/3600:.2f} | {scores} |")
    lines+=['','## Evaluation','', '| Evaluation | Set | Method | Success | Items | Seconds/problem |','|---|---|---|---:|---:|---:|']
    evaluations=[]
    for path in sorted((ROOT/'runs').glob('eval_*/summary.json')):
        r=read(path);evaluations.append((path.parent.name,r))
        for setname in ('standard','hard'):
            for method,v in r.get(setname,{}).items():
                if isinstance(v,dict) and 'success' in v:lines.append(f"| {path.parent.name} | {setname} | {method} | {100*v['success']:.2f}% | {v['n']} | {v['seconds_per_problem']:.3f} |")
    lines+=['','Equal attempt counts are reported separately from the locked elapsed-time comparison. Random matched updates use online gradient magnitudes and pay for that backward computation. Oracle fitting uses known valid trajectories and is excluded from ordinary repair scores.','','## Draft gates and limits','',
            '- G0: prelaunch contracts plus post-training anchor/weight audits; inspect saved evidence, including any failure.',
            '- G1/G2: report foundation/transplant NLL and prior/suffix degradation. Small mean differences alone do not establish the requested confidence margins.',
            '- G3/G4/G5: verified prior diversity, reachability, paired repair-versus-random/retry gains, and measured costs. Item bootstrap intervals describe this one trained foundation; they do not replace independent training-seed variation.',
            '- G6: shared versus local controls and L=4 likelihood are diagnostics; learned temporal abstraction is not established.',
            '- BF16/FP16, distributed workers, long-context/width scaling, elastic depth, a full ordinary-transformer control and independent foundation seeds are deferred in this time-bounded experiment. FP32 and one Apple GPU are the executed configuration.',
            '- The no-forecast control is compared with the matching 9,000-step foundation prefix under the same original 18,000-step schedule. It is not compared as an equally trained endpoint to the 18,000-step foundation.',
            '- Time-limited/partial stages are labeled explicitly. Numerical failure never counts as completion. No V5 checkpoint or result is overwritten.','',
            '## Evidence files','',
            '- `PLAN.json`, `PRE_REGISTRATION.md`, `SPEC_COMPLIANCE.md`: fixed recipe, comparisons and coverage.',
            '- `runs/status.json`, `runs/queue.jsonl`, `runs/driver.out`: queue status and failures.',
            '- `runs/*/config.json`, `log.jsonl`, `checkpoint.pt`, `train_summary.json`: training/recovery provenance.',
            '- `runs/eval_*/summary.json`, `items.jsonl`, `split_manifest.json`: NLL, every search attempt, verifier results, costs and instance identity.',
            '- `tests/optimized_checks.json`, `runs/audit_*.json`: executable and trained-weight contracts.','']
    gate=read(ROOT/'runs'/'ANCHOR_GATE.json')
    if gate:
        lines+=['## Observed anchor screen','',f"Passed development mean screen: **{gate['passed_mean_screen']}**. Text NLL gap: {gate['nll_gaps']['text']:+.5f}; graph NLL gap: {gate['nll_gaps']['graph']:+.5f}; standard success loss: {100*gate['standard_success_loss']:+.2f} points.",'',gate['caveat'],'']
    primary=read(ROOT/'runs'/'PRIMARY_RESULT.json')
    if primary:
        lines+=['## Locked elapsed-time comparison','',f"Budget: {primary['locked_batch_time_budget_s']:.3f} seconds per batch of 16. Retry reference: {primary['selected_retry_checkpoint']}.",'']
        for setname,result in primary['paired'].items():lines.append(f"- {setname}: repair minus retries {result['gain_points']:+.2f} points; paired item 95% interval [{result['ci95_points'][0]:+.2f}, {result['ci95_points'][1]:+.2f}]; n={result['n']}.")
        lines+=['',primary['promotion'],'']
    text='\n'.join(lines)
    temp=ROOT/'REPORT.md.tmp';temp.write_text(text);temp.replace(ROOT/'REPORT.md')
    atomic_json(ROOT/'runs'/'report_index.json',dict(status=state.get('status','preparing'),evaluations=[name for name,_ in evaluations],updated=datetime.now(ZoneInfo('Asia/Calcutta')).isoformat()))


if __name__=='__main__':main()
