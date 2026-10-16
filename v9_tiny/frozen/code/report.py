"""A report may state negative/partial evidence, never convert execution to scientific success."""
import json
from pathlib import Path
from runtime import ROOT

def write_report(summary):
    lines=['# V8 full-exposure 12-hour pilot','',summary['message'],'',f"Status: **{summary['status']}**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.",'',f"Policy: {summary.get('policy','see protocol')}. Scores and confidence thresholds are evidence readouts; running a stage does not mean that it succeeded."]
    if summary.get('recovery'):lines+=['','The user authorized continuation after the compiled decoding parity failure. Eager static decoding passed the same tolerance on trained weights; training compilation and the completed alpha-zero checkpoint were preserved. The original clock/deadline were retained. The failed source, preflights, lock and report are archived under `spec/recovery_2026-10-06`. This is an execution revision, not an additional training seed.']
    if (summary.get('recovery') or {}).get('serialization_recovery'):lines+=['','A later development-result export failed on a NumPy boolean. Competence and coverage flags now return native Python booleans with unchanged numerical criteria. Both completed 24,000-update foundations are reused with checksum and training-provenance checks; the completed alpha-zero development result retains its earlier evaluation provenance. The failed NCP evaluation is rerun. The source, failure and unfinalized output are archived under `spec/recovery_json_2026-10-06`; the original deadline is unchanged.']
    lines+=['','## Foundation evidence','','| Arm | Parameters | Standard dev | Lookup dev | Competence |','|---|---:|---:|---:|---|']
    for name,v in summary['foundations'].items():lines.append(f"| {name} | {v['active_parameters']:,} | {v['standard']:.1%} | {v['lookup']:.1%} | {v['competence_pass']} |")
    lines+=['','## Training exposure','']
    for p in sorted((ROOT/'runs').glob('*/train_summary.json')):
        j=json.loads(p.read_text());lines.append(f"- {p.parent.name}: {j['completed_steps']:,} updates, {j['symbolic_exposure'].get('graph',0):,} GraphNav examples, {j['status']}; [summary]({p}).")
    lines+=['','## Acceptance and component results','']
    for row in summary.get('decisions',[]):
        if 'evidence_qualified' in row:lines.append(f"- {row['stage']}: evidence-qualified={row['evidence_qualified']}, skill change={row['change']}; continued as an exploratory trajectory.")
        else:lines.append(f"- {row['stage']}: retention-qualified={row.get('retention',{}).get('pass')}, coverage-qualified={row.get('coverage_pass','not applicable')}, research-only={row.get('research_only')}.")
    for pattern in ('*/acceptance.json','*/coverage.json','D_*_summary.json','E_search/summary.json'):
        for p in sorted((ROOT/'runs').glob(pattern)):
            j=json.loads(p.read_text());status=j.get('pass',j.get('coverage_gate',j.get('status',None)));lines.append(f"- {p.parent.name}/{p.name}: {status if status is not None else 'see phase decisions'}; [evidence]({p}).")
    lines+=['','## Partial and skipped work','']
    for row in summary['skipped']:lines.append(f"- {row['stage']}: {row['reason']}.")
    for row in summary.get('partial',[]):lines.append(f"- Partial {row['stage']}: {row['reason']}.")
    lines+=['','## Sealed endpoint measurements','']
    for p in sorted((ROOT/'runs').glob('*_sealed_test/summary.json')):
        j=json.loads(p.read_text());std=j.get('symbolic',{}).get('standard',{});lines.append(f"- {p.parent.name}: {j.get('status')}; standard {std.get('successes','?')}/{std.get('n','?')}, interval {std.get('interval')}; [all task/text results]({p}).")
    lines+=['','The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, enlarged hard-context training, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Development thresholds do not establish superiority; sealed paired results and intervals must support any claim.','','Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. The earlier stopped run remains in `v8_tiny`. No early low-score gate terminated this revision.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':write_report(json.loads((ROOT/'PRIMARY_RESULT.json').read_text()))
