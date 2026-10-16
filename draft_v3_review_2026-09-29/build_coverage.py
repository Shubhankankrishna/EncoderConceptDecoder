from pathlib import Path
import json, collections

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent
inv = json.loads((out / 'inventory.json').read_text())
scan = {x['path']: x for x in json.loads((out / 'text_scan.json').read_text())}
critical = {
    'draft_v3.md', 'proposed_v3.md', 'PROMPT.md',
    'v3_design/v3_ref.py', 'v3_design/check_engram2.py', 'v3_design/v3_budget.py',
    'v3_design/v3_budget.json', 'v3_design/REVIEW_v3.md',
    'run_m2_25M/code/hrm_encoder.py', 'run_m2_25M/code/hrm_decoder.py',
    'run_m2_25M/code/concept_core_v2.py', 'run_m2_25M/code/model.py',
    'run_m2_25M/code/train.py', 'run_m2_25M/code/data.py',
    'run_m2_25M/code/evaluate.py', 'run_m2_25M/code/build_data.py',
    'run_m2_25M/code/tasks.py', 'run_m2_25M/data/MANIFEST.json',
    'run_v2_19M/code/train.py', 'run_v2_19M/code/model.py',
    'run_v2_19M/code/concept_core_d6.py',
    'Concept Design/RESULTS.txt', 'Concept Design/RESULTS_v2.txt',
    'Concept Design/RESULTS_E2E.txt', 'Concept Design/RESULTS_H1_H2.txt',
    'Concept Design/RESULTS_review.txt', 'Decoder Design/RESULTS_DECODER.txt',
    'Encoder Design/HRM_THINKER_ENCODER_v1_SPEC.md',
    'Decoder Design/HRM_THINKER_DECODER_v1_SPEC.md',
    'audit_v2_2026-09-28/V2_REVIEW.md',
    'v2_underperformance_review_2026-09-28/REPORT.md',
    'v3_design_review_2026-09-29/RESULTS.md',
}
papers = {'Imp Paper/LoopFormer.pdf', 'Imp Paper/Intern-S2-Mobius.pdf', 'Imp Paper/Diffusion Timestep Reasoning Google.pdf', 'Next-Latent Prediction Transformer.pdf'}
def status(r):
    p = r['path']; ext = r['extension']; s = scan.get(p, {})
    if p in critical: return 'Deep review of task-relevant content; source/results cross-check'
    if p in papers: return 'PDF text extracted; draft-relevant scientific claims checked'
    if p.startswith(('run_m2_25M/runs/main/', 'run_m2_25M/runs/cont/', 'run_v2_19M/runs/v2_48h/')) and ext in ('.json', '.jsonl'):
        return 'Parsed all records; frozen run evidence and selected metric comparisons'
    if 'duplicate_of' in s: return 'Hash-identical text to ' + s['duplicate_of'] + '; not independently reinterpreted'
    if p.startswith('run_m2_25M/webapp/') and r['kind'] == 'text': return 'Text/structure indexed; source provenance checked; presentation, not primary model evidence'
    if r['kind'] == 'text': return 'Full text machine-scanned/indexed (' + s.get('parsed_as', 'text') + '); supporting context'
    if ext == '.pdf': return 'PDF text extracted and catalogued; not a full paper-level scientific review'
    if ext == '.npy': return 'Array header/shape/dtype inspected; corpus semantics traced through manifest and loader'
    if ext in ('.pt', '.pth'): return 'Checkpoint/archive metadata inspected; no new full-weight evaluation'
    if ext in ('.zip', '.docx'): return 'Archive entry inventory; DOCX text extracted where applicable'
    if ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp'): return 'Asset inventoried; not newly visually reviewed for this draft task'
    return 'Filesystem metadata only (cache, runtime, or ancillary binary)'

lines = ['# Repository coverage for the draft V3 review', '',
         'Snapshot: ' + inv['created'] + '. This appendix describes inspection depth, not a claim that every file received the same kind of review.', '',
         '**1,723 files / 222 directories:** 862 files outside `.venv`, plus 861 environment/dependency files. All 720 non-dependency text files were scanned; 346 distinct text hashes were indexed. Original paths and hashes remain in `inventory.tsv` / `inventory.json`.', '',
         'The complete directory list is in `directories.tsv`. The generated review directory was excluded from the original inventory. Symlinks were inventoried without recursively following linked directories.', '',
         '## Folder-level assessment', '',
         '| Folder | Role in this review |', '|---|---|',
         '| Root documents | Run brief, paper index, competing V3 proposals; draft is the review target |',
         '| Encoder Design | L/G streams, causal/local reach, pooling, original specification and implementation lineage |',
         '| Concept Design | Isolated versus end-to-end recurrence, state initialization, branching and steering evidence |',
         '| Decoder Design | Confinement, plan alignment, NextLat comparisons, multi-value bottlenecks; bundle duplicates identified |',
         '| Imp Paper | Detailed comparison of LoopFormer, Mobius and diffusion-curriculum claims |',
         '| Loop Papers | Extracted research library; not treated as 41 independent endorsements of the draft |',
         '| run_m2_25M/code | v1 mechanics, loss masks/reductions, training clocks, evaluation and data building |',
         '| run_m2_25M/data | Corpus metadata, array shapes, tokenizer/config structure and deduplicated evaluation provenance |',
         '| run_m2_25M/runs | Screens, main run, continuation, calibration, test records and final metrics |',
         '| run_m2_25M/concept_v2 and model_v2 | Historical design decisions; superseded statuses distinguished from live implementation |',
         '| run_m2_25M/webapp | Displays and generated data; proposed-V3 source identity checked |',
         '| run_m2_25M/.venv | Environment inventory only; torch used for low-priority CPU probes |',
         '| run_v2_19M | Implemented D6 model, active training evidence, gradient windows, routing and operational history |',
         '| audit_v2_2026-09-28 | Earlier v1/full-data audit and its limitations |',
         '| v2_underperformance_review_2026-09-28 | Historical checkpoint interventions and gradient/routing diagnosis, not newly rerun |',
         '| v3_design_review_2026-09-29 | Earlier proposal investigation, later v2 diagnostics, immutable snapshots and duplicate content |',
         '| v3_design | Current draft reference, budget exploration, Engram test and prior review; new reference/probe execution |',
         '', '## Every non-dependency file', '', '| File | Bytes | Inspection |', '|---|---:|---|']
for r in inv['files']:
    if '/.venv/' in r['path']: continue
    rel = r['path']; link = '[' + rel.replace('|', '\\|') + '](<' + str(root/rel) + '>)'
    lines.append(f"| {link} | {r['bytes']} | {status(r).replace('|', '/')} |")
lines += ['', '## Dependency files', '', 'Every dependency path is listed in `inventory.tsv`; these 861 files were not individually audited. Treating third-party implementation internals as project research evidence would not strengthen this review.', '']
(out/'FILE_COVERAGE.md').write_text('\n'.join(lines))
(out/'directories.tsv').write_text('directory\n'+'\n'.join(inv['directories'])+'\n')
print('Wrote coverage for', sum('/.venv/' not in r['path'] for r in inv['files']), 'files')
