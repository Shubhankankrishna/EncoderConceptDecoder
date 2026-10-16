"""Render audit snapshots into human-readable tables; does not read live run state."""
from pathlib import Path
from collections import Counter
import json

A = Path(__file__).resolve().parent
R = A.parent / 'run_m2_25M'
a = json.loads((A / 'audit_results.json').read_text())
f = json.loads((A / 'main_final_snapshot.json').read_text())
inv = json.loads((A / 'inventory.json').read_text())
P = ['lr_recall1', 'lr_carry3', 'lr_multi4', 'lr_hop', 'lr_bind']

def fmt(x):
    if x is None: return '—'
    if isinstance(x, float): return f'{x:.6f}'
    return str(x).replace('|', '\\|').replace('\n', ' ')

def table(out, headers, rows):
    out.extend(['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)])
    out.extend('| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in rows)
    out.append('')

out = ['# Results appendix', '',
       f"Snapshot: {a['snapshot_time']}. Values below are generated from the saved audit snapshots, not refreshed from live training.", '',
       'Exact match is a fraction in these tables (0.97 means 97%). Nats are per valid next-token target. Training token counts are allocated token slots. Intact final evaluation uses R=4. Main/continuation quick probes use 128 items unless explicitly labeled otherwise.', '',
       f"All original final-evaluation values, including generations, are preserved in [main_final_snapshot.json]({A/'main_final_snapshot.json'}). All captured hourly evaluations and their telemetry are preserved in [audit_results.json]({A/'audit_results.json'}).", '',
       '## Training trajectories', '']
for run, v in a['runs'].items():
    out += [f'### {run}', '',
            f"Training-log records: {v['train_records']}; metric records: {v['metric_records']}; median logged throughput: {v['throughput_median']:.2f} tokens/s; range: {v['throughput_range'][0]:.2f}–{v['throughput_range'][1]:.2f}; guard records: {len(v['guards'])}.", '']
    table(out, ['Step', 'Global step', 'Wall h', 'Slots M', 'TS val nats', 'Wiki val nats', 'Mean', *[p[3:] for p in P], 'Swapped recall'],
          [[e['step'], e['step'] + (4050 if run == 'cont' else 0), e['wall_h'], e['tokens']/1e6,
            e['val_tinystories']['nats'], e['val_wikitext']['nats'], e['val_lang_mean'],
            *[e['probes'][p]['exact'] for p in P], e['recall1_plan_swap']['exact']] for e in v['evals']])
    table(out, ['Step', 'Min code usage', 'Mean code perplexity', 'PQ variance explained', 'NCP CE', 'Marginal code H', 'Eval seconds'],
          [[e['step'], *[e['codebook'][k] for k in ['codes_used_frac_min','perplexity_mean','var_explained','ncp_ce','code_entropy']], e.get('eval_sec')] for e in v['evals']])
    extra = [e for e in v['evals'] if 'probes_256' in e]
    if extra:
        out += ['Additional final-screen 256-item probe results (distinct from the 128-item quick probes above):', '']
        table(out, ['Step', *P], [[e['step'], *[e['probes_256'][p]['exact'] for p in P]] for e in extra])

out += ['## Final language test', '']
table(out, ['Condition', 'TinyStories nats', 'TinyStories bpb', 'WikiText nats', 'WikiText bpb'],
      [[c, *[f['lang'][n][c][k] for n in ['tinystories','wikitext'] for k in ['nats','bpb']]] for c in f['lang']['tinystories']])
out += ['## Context truncation as originally measured', '', 'The r72 comparison is confounded; see the review before treating it as a beyond-local-context gain.', '']
table(out, ['Dataset','r72','r128','r256','r512','r1024','Swap at 1024','Rows with ≥512 context'],
      [[n, *[v['summary']['curve'][f'r{r}'] for r in [72,128,256,512,1024]],v['swap_r1024']['nats'],v['summary']['rows_ctx_ge_512']] for n,v in f['trunc'].items()])

for split, cond in f['items'].items():
    out += [f'## Final {split}: every family and condition', '']
    conditions = list(cond)
    families = sorted(k for k in cond['intact'] if not k.startswith('_'))
    table(out, ['Family','Items',*conditions],
          [[fam,cond['intact'][fam]['n'],*[cond[c].get(fam,{}).get('exact') for c in conditions]] for fam in families]
          + [['Synthetic macro','—',*[cond[c].get('_macro_synthetic') for c in conditions]]])
    out += ['Per-offset and per-hop-depth results:', '']
    rows=[]
    for c in conditions:
        for fam in ['lr_carry3','lr_multi4']:
            for j,x in enumerate(cond[c][fam]['by_offset']): rows.append([c,fam,f'offset {j}',x])
        for k,x in cond[c]['lr_hop'].items():
            if k.startswith('exact_h'): rows.append([c,'lr_hop',k,x])
    table(out, ['Condition','Family','Submetric','Accuracy'], rows)

out += ['## Linear representation probes', '']
table(out, ['Probe','Target','Top-1','Majority','Train examples','Test examples','Classes'],
      [[n,k,v['acc'],v['majority_baseline'],v['n_train'],v['n_test'],v['classes']] for n,ps in f['probes'].items() for k,v in ps.items()])
out += ['## Final codebook and loop dynamics', '']
table(out,['Metric','Value'],list(f['health']['codebook'].items()))
loop=f['health']['loop_dynamics_lr_hop']
table(out,['Pass','State norm','Update norm','Angle from prior update, degrees','State turn, degrees','Gradient norm'],
      [[i+1,loop['state_norm'][i],loop['step'][i],loop['consecutive_step_angle_deg'][i-1] if i else None,
        loop['turn_deg'][i-1] if i else None,loop['grad'][i]] for i in range(8)])
out += [f"Pass-1/pass-4 gradient ratio: {loop['grad_ratio_pass1_pass4']:.6f}.", '',
        'Other final telemetry (temperature arrays retained in full):', '', '```json',json.dumps(f['health']['telemetry'],indent=2),'```','']

out += ['## Data validation', '']
table(out,['Split','Rows','Nonpad','NTP targets','NCP targets','All-pad NCP targets','All-pad fraction','Bad boundaries','Hash matches'],
      [[s,*[v[k] for k in ['rows','nonpad','ntp_targets','ncp_targets','all_pad_ncp_targets','all_pad_ncp_fraction','bad_chunk_boundary_starts','matches_manifest']]] for s,v in a['data_checks'].items()])
for s,v in a['item_checks'].items():
    out += [f'### Scored {s} items', '',f"Items: {v['n']}; unique: {v['unique']}; exact matches to training: {v['overlap_with_train']}.", '']
    table(out,['Family','N','Min tokens','Max tokens','Answer spans','Answer-token lengths','Minimum evidence distance','Invalid spans'],
          [[fam,*[vv[k] for k in ['n','tokens_min','tokens_max','spans','answer_tokens','min_evidence_distance','invalid_span_count']]] for fam,vv in v['families'].items()])

out += ['## All six proposed parameter/compute budget rows', '',
        'These are analytical proposed designs, not trained models. Parameter recount agrees with each stored table. Published MACs below omit the padded expert-slot correction discussed in the review.', '']
table(out,['Variant','Encoder params','Core params','Decoder params','Compute params','Stored params','NextLat params','Eval fwd MACs','Train MAC equivalents'],
      [[k,*[v['params'][p] for p in ['encoder','core','decoder','compute','stored','nextlat_psi_training_only']],v['macs']['fwd_eval'],v['macs']['train_equiv']] for k,v in a['budget_recount'].items()])
out += ['Specified row-C expert-capacity arithmetic:', '', '```json',json.dumps(a['moe_capacity_arithmetic'],indent=2),'```','',
        '## CPU diagnostic results', '', '```json',(A/'diagnostics.json').read_text().strip(),'```','',
        '## Saved generation samples', '', 'These samples are qualitative evidence, not a controlled generation benchmark. The source JSON retains all prompts and completions verbatim.', '',
        '```json',json.dumps(f['generation'],indent=2,ensure_ascii=False),'```','']
(A/'RESULTS_APPENDIX.md').write_text('\n'.join(out))

roles={
 'DECISIONS.md':'Chronological deviations, calibration, operations, and experiment decisions; checked against executable code and raw logs.',
 'ENV.txt':'Recorded runtime and hardware environment.',
 'INVENTORY.txt':'Historical file inventory; not treated as current experiment evidence.',
 'PAPERS_READ.md':'Project literature summary; not an independent experimental result.',
 'STATUS.md':'Operational status summary; live logs take precedence.',
 'code/hrm_encoder.py':'Encoder implementation, local/global streams, Engram, chunk pooling, exact structural tests.',
 'code/concept_core_v2.py':'Implemented one-layer recurrent core, PQ targets/readout, NCP/VQ losses; not proposed D6.',
 'code/hrm_decoder.py':'Confined/stack decoder, plan indexing, optional NextLat, parameter groups and tests.',
 'code/model.py':'Main model configuration, interventions, parameter and MAC accounting.',
 'code/train.py':'Optimizer, accumulation, sampled R, loss/masks, checkpoint/resume, LR and evaluation schedule.',
 'code/data.py':'Batch construction, Engram CPU indices, target masking, packing and data cursor.',
 'code/build_data.py':'Tokenizer fitting, mixture, packing, eval rebuild and hashes.',
 'code/tasks.py':'Synthetic generators and actual ID/OOD definitions; three families have unchanged distributions.',
 'code/check_data.py':'Existing data assertions; independently checked full packed rows and scored items.',
 'code/evaluate.py':'Language, truncation, answer-span metrics, codebook and intervention telemetry; r72 issue identified.',
 'code/final_eval.py':'Final test conditions, probe/generation orchestration and output schema.',
 'code/probes.py':'Linear probes and loop dynamics; reviewed what the metrics can and cannot establish.',
 'code/probe_multi.py':'Earlier representation probes; useful diagnostic context, not final-model accuracy.',
 'code/compat.py':'Workaround for EncoderLayer._apply collision with torch Module machinery.',
 'code/test_compat.py':'Compatibility test coverage.',
 'code/test_p5.py':'Causality/reach/doc-isolation assertions; exact-max-reach assertion causes screen aborts.',
 'code/calibrate.py':'MPS batch/memory/speed calibration.',
 'code/main_steps.py':'Main step budget derivation.',
 'code/extend_ckpt.py':'Continuation checkpoint preparation and preserved state.',
 'code/e2e.py':'Model interface, original end-to-end synthetic harness and pipeline tests.',
 'code/harness.py':'Earlier synthetic encoder/test harness.',
 'code/dmix_build_reference.py':'Earlier D-MIX construction reference; actual manifest/code take precedence.',
 'code/plots.py':'Presentation of recorded curves; no additional experiment.',
 'code/report_tables.py':'Derived tables, including separate 256-item screen probes; raw JSON is authoritative.',
 'code/OVERNIGHT_POLICY.md':'Existing runtime policy and operational thresholds; no new job launched by this review.',
 'concept_v2/DESIGNS.md':'Evolving candidate designs and earlier reasoning, distinguished from final proposals.',
 'concept_v2/DESIGNS_FINAL.md':'Candidate definitions, tests, comparisons and selection logic.',
 'concept_v2/DESIGN_D6.md':'D6 structure/training rationale; carry-necessity claim and R6 window note challenged.',
 'concept_v2/LOOP_PAPERS_NOTES.md':'Project notes for 42 papers plus three residual references; selected originals independently checked.',
 'concept_v2/ADVISOR_LOG.md':'Recorded advice/review history; not independent measurements.',
 'concept_v2/PROGRESS.md':'Design-work progress and provenance.',
 'concept_v2/core_budget.py':'Analytical core sizing, not an implemented/trainable D6 core.',
 'concept_v2/budgets.json':'Candidate analytical core budgets.',
 'concept_v2/scores.json':'Subjective weighted design scores; not measured model quality.',
 'model_v2/MODEL_V2_19M.md':'Current whole-model proposal; explicitly unimplemented/untrained; reviewed architecture, graph, budget and evidence.',
 'model_v2/v2_budget.py':'Parameter/compute estimator; all six parameter rows reproduced, padding arithmetic checked.',
 'model_v2/v2_budget.json':'Six analytical variants; row C is selected proposal.',
 'model_v2/make_arch_v2.py':'Diagram/data generator; not neural-network implementation.',
 'model_v2/arch_v2.json':'Generated diagram definition, not model weights/results.',
 'data/MANIFEST.json':'Actual data mixture, seeds, hashes, counts and duplicate drops.',
 'data/MANIFEST.before_eval_rebuild.json':'Earlier manifest; retained as provenance, not current scored-set truth.',
 'data/build.log':'Data-construction and eval-rebuild provenance.',
 'data/tokenizer.json':'8,192-ID byte-level BPE; hash checked, bytes/canonicalization inspected.',
 'webapp/server.py':'Dashboard serving/reading and analysis endpoints; not executed during audit.',
 'webapp/js/final.js':'Derived final-results/gate view; PASS-line counting does not fully represent aborted screen tests.',
 'webapp/data/analysis.json':'Existing generated analysis; not an independent model evaluation.',
 'runs/calibration_B4_watermark.txt':'Failed B4 experiment; distinguishes the 8.38 GiB figure from successful B2.',
 'runs/calibration.json':'Recorded B1/B2/B4 calibration data.',
 'runs/calibration.log':'Calibration process and failure context.',
 'runs/data_checks.json':'Previously recorded data invariants; independently cross-checked.',
 'runs/data_samples.md':'Human-readable generated samples and family examples.',
 'runs/report_tables.md':'Derived report tables; traced to raw JSON and distinguished validation/test sample sizes.',
 'runs/main/eval_final.json':'Authoritative main step4050 final matrix; copied verbatim into immutable audit snapshot.',
 'runs/main/eval_best.log':'Best/last checkpoint identity explains skipped duplicate evaluation.',
 'runs/main/figs/curves.png':'Rendered main learning curves; supporting display of the underlying logs.',
}
def role(p):
    if p in roles:return roles[p]
    if p.endswith('.pt') and p.startswith('data/eval/'):
        return 'Every item checked: train exact-overlap, within/cross-split duplicates, lengths, answer spans, distances and family counts.'
    if p.endswith('.pt'):
        return 'Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically.'
    if p.endswith('.npy'):
        if p.endswith('/tok.npy') or p.endswith('/doc.npy'):return 'Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries.'
        if 'synthetic_doc_sha1' in p:return 'Training document hash index used for every scored-item overlap check.'
        if 'token_bytes' in p:return 'Byte denominator table checked against ByteLevel symbol counts.'
        if 'canon' in p:return 'Canonical-ID lookup shape/range checked and traced to Engram indexing.'
        return 'Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts.'
    if p.endswith('/log.jsonl') and p.startswith('runs/'):
        return 'All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot.'
    if p.endswith('/train_log.jsonl'):
        return 'All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected.'
    if p.endswith('/GUARDS.jsonl'):return 'All guard records parsed; empty at audit snapshot.'
    if p.endswith('/config.json'):return 'Run arguments/configuration inspected to distinguish main, controls and continuation.'
    if p.endswith('/p5_trained.txt'):return 'All assertion output checked; main passes, screen tests abort at exact reach equality.'
    if p.startswith('runs/tests/'):return 'Recorded test output reviewed with compatibility/structural-test interpretation.'
    if '.bak_' in p:return 'Historical backup indexed and used as provenance; current executable source takes precedence.'
    if p.startswith('webapp/'):
        if p.endswith(('.js','.css','.html')):return 'Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior.'
        if p.endswith(('.json','.jsonl')):return 'Derived/cached visualization data parsed; raw experiment logs take precedence.'
        return 'Dashboard runtime log indexed; no new model evidence.'
    if p.endswith('.sh'):return 'Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed.'
    if p.endswith('/live.json'):return 'Volatile heartbeat snapshot; kept separate from completed evaluation results.'
    if p.endswith(('.log','.stdout','.pid')):return 'Operational/training process record indexed; failures/resumes contextualized with decisions and metrics.'
    return 'Experiment control/provenance file indexed and contextualized; no independent benchmark result.'

rows=[x for x in inv if x['category']=='first-party']
fo=['# File and folder review coverage','',
    'This appendix lists every project file in the snapshot. “Indexed” means bytes/text were read and recorded; it does not imply line-by-line semantic review of every operational script or display asset. Deep tracing focused on model/data/training/evaluation code, experiments, and the v2 proposal. Binary coverage is specified separately.', '',
    f"Total inventory: {len(inv)} files; substantive project files: {len(rows)}; dependencies/caches/OS files: {len(inv)-len(rows)}. The complete inventory, sizes, inodes, modification times and project text hashes are in [inventory.json]({A/'inventory.json'}).",'',
    '## Folder-level conclusions','',
    '| Folder | Conclusion |','|---|---|',
    '| Root records | Explain decisions and historical state; raw executable code and result records take precedence. |',
    '| code | Implements the d384 model with ConceptCoreV2; no D6/MoE training module exists. Evaluation has the identified truncation/padding caveats. |',
    '| data | Packed rows and tokenizer pass recorded hashes/alignment; final item sets are exact-disjoint from training. OOD definitions need qualification. |',
    '| runs/screens | Early schedule-matched confined/stack/fixed-R controls are within tie band. Trained structural checks abort on observed reach below the bound. |',
    '| runs/main | Completed final matrix supports distant recall and plan use, but not useful depth scaling or robust OOD composition. |',
    '| runs/cont | Additional training of the same model; improving validation skills, no completed final matrix at snapshot. |',
    '| runs/tests | Initial CPU structural evidence and MPS compatibility fix; distinct from every trained-screen assertion passing. |',
    '| concept_v2 | Literature synthesis, candidate specifications and subjective scores; design evidence, not trained candidate comparisons. |',
    '| model_v2 | Whole-model proposal, analytical budgets and diagrams; explicitly unimplemented/untrained. |',
    '| webapp | Presentation and cached summaries. Source metrics remain authoritative; animated expert routing does not show observed routing. |',
    '| .venv / __pycache__ / vendor / .DS_Store | Inventoried; not treated as original architecture or experimental evidence. |','',
    '## Every project file','']
for folder in ['root','code','data','runs','concept_v2','model_v2','webapp']:
    subset=[x for x in rows if (x['path'].split('/')[0] if '/' in x['path'] else 'root')==folder]
    fo += [f'### {folder} — {len(subset)} files','']
    table(fo,['File','Bytes','Lines / records','Role and coverage'],
          [[f"[{x['path']}]({R/x['path']})",x['bytes'],f"{x.get('lines','—')} / {x.get('records','—')}",role(x['path'])] for x in subset])
fo += ['## Supporting material outside the requested run folder','',
       'The parent PROMPT.md supplied the precommitted architecture, data, and evaluation gates. Earlier Concept Design and Decoder Design result files were used to check whether proposed explanations transfer from isolated symbolic tests to end-to-end training. Relevant method sections of the local NextLat, Sparse Layers, LoopFormer, and Diffusion curriculum PDFs were checked. This is a targeted cross-check, not an assertion that every parent-folder PDF was reread.','']
(A/'FILE_REVIEW.md').write_text('\n'.join(fo))
print('Wrote',A/'RESULTS_APPENDIX.md',A/'FILE_REVIEW.md')
