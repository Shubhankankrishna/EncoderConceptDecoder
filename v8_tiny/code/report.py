"""A report may state negative/partial evidence, never convert execution to scientific success."""
import json
from pathlib import Path
from runtime import ROOT

def write_report(summary):
    lines=['# V8 tiny experiment','',summary['message'],'',f"Status: **{summary['status']}**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.",'','## Foundation evidence','','| Arm | Parameters | Standard dev | Lookup dev | Competence |','|---|---:|---:|---:|---|']
    for name,v in summary['foundations'].items():lines.append(f"| {name} | {v['active_parameters']:,} | {v['standard']:.1%} | {v['lookup']:.1%} | {v['competence_pass']} |")
    lines+=['','## Training exposure','']
    for p in sorted((ROOT/'runs').glob('*/train_summary.json')):
        j=json.loads(p.read_text());lines.append(f"- {p.parent.name}: {j['completed_steps']:,} updates, {j['symbolic_exposure'].get('graph',0):,} GraphNav examples, {j['status']}; [summary]({p}).")
    lines+=['','## Acceptance and component results','']
    for pattern in ('*/acceptance.json','*/coverage.json','D_*_summary.json','E_search/summary.json'):
        for p in sorted((ROOT/'runs').glob(pattern)):
            j=json.loads(p.read_text());status=j.get('pass',j.get('coverage_gate',j.get('status',None)));lines.append(f"- {p.parent.name}/{p.name}: {status if status is not None else 'see phase decisions'}; [evidence]({p}).")
    lines+=['','## Partial and skipped work','']
    for row in summary['skipped']:lines.append(f"- {row['stage']}: {row['reason']}.")
    lines+=['','The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Operational development gates do not establish superiority; sealed paired results and intervals must support any claim.','','Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. Historical V6 replay is an inspected development diagnostic, not V8 confirmation.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':write_report(json.loads((ROOT/'PRIMARY_RESULT.json').read_text()))
