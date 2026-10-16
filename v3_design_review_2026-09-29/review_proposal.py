"""Self-review key proposal claims against frozen evidence; does not touch runs."""
from pathlib import Path
import datetime, hashlib, json, re, statistics

A = Path(__file__).resolve().parent
ROOT = A.parent
S = A / 'source_snapshot'
snap = json.loads((A / 'snapshot.json').read_text())
depth = json.loads((A / 'checkpoint_001703_diagnostics.json').read_text())
design = json.loads((A / 'design_checks.json').read_text())
inventory = json.loads((A / 'inventory.json').read_text())
proposal = (ROOT / 'proposed_v3.md').read_text()
checks = []

def check(name, passed, evidence=None):
    checks.append(dict(name=name, passed=bool(passed), evidence=evidence))

def displayed(name, value, fmt):
    text = format(value, fmt)
    check(name, text in proposal, dict(value=value, expected_display=text))

runs = snap['runs']
v1 = {x['step']: x for x in runs['v1']['eval']}
v2 = {x['step']: x for x in runs['v2']['eval']}
latest = runs['v2']['train'][-1]
matched = snap['matched_train'][-1]
check('declared evidence cutoff', snap['time'].startswith('2026-09-29T10:20:29'))
check('declared inventory count', len(inventory) == 382)
check('text parse errors absent', not [x for x in inventory if 'parse_error' in x])
check('official latest evaluation', max(v2) == depth['step'] == 1703)
check('last complete logged training window', latest['step'] == matched['step'] == 1750)

for run, step in [('v1', 365), ('v1', 2559), ('v1', 1593), ('v1', 1792), ('v2', 267), ('v2', 1703)]:
    row = (v1 if run == 'v1' else v2)[step]
    displayed(f'{run} step {step} validation', row['val_lang_mean'], '.4f')
displayed('nearly equal-time language deficit', v2[1703]['val_lang_mean']-v1[2559]['val_lang_mean'], '.4f')
check('equal-time comparison within two minutes', abs(v2[1703]['wall_h']-v1[2559]['wall_h'])*60 < 2)
for run in ['v1', 'v2']:
    displayed(f'{run} matched NTP', matched[run+'_ntp'], '.4f')
    displayed(f'{run} time to 1750', matched[run+'_h'], '.3f')
displayed('v2 time penalty percent', 100*(matched['v2_h']/matched['v1_h']-1), '.1f')
displayed('latest 25-step slots/s', latest['tok_s'], ',.0f')
displayed('matched 50-step slots/s', matched['v2_tps'], ',.0f')
for run, step in [('v1',2559), ('v2',1703)]:
    displayed(f'{run} matched-time recall', (v1 if run=='v1' else v2)[step]['probes']['lr_recall1']['exact']*100, '.2f')

for language, rows in runs['v1']['final']['lang'].items():
    for mode in ['intact','core_off','u_off','c_off','R1']:
        displayed(f'v1 final {language} {mode}', rows[mode]['nats'], '.4f')

for r in [1,2,4,6,8,12]:
    for lang in ['tinystories','wikitext']:
        displayed(f'latest sampled {lang} R{r}', depth['language_summary'][lang][str(r)], '.4f')
    displayed(f'latest sampled NCP R{r}', statistics.mean(x['ncp'][r-1] for x in depth['language_batches']), '.4f')
improved = [a>b for x in depth['language_batches'] for a,b in zip(x['depths']['1']['nats'],x['depths']['6']['nats'])]
check('R6 beats R1 in all sixteen sampled rows', len(improved)==16 and all(improved))
means = {r:statistics.mean(depth['language_summary'][l][str(r)] for l in ['tinystories','wikitext']) for r in [1,6,12]}
displayed('mean sampled R1 to R6 gain', means[1]-means[6], '.4f')
displayed('mean sampled R6 to R12 gain', means[6]-means[12], '.5f')
check('latest c_off hurts both languages', all(x['c_off']>x['6'] for x in depth['language_summary'].values()))
displayed('codebook usage percent', v2[1703]['codebook']['codes_used_frac_mean']*100, '.2f')
displayed('codebook explained variance', v2[1703]['codebook']['var_explained'], '.4f')
displayed('mature v1 pass gradient ratio', runs['v1']['final']['health']['loop_dynamics_lr_hop']['grad_ratio_pass1_pass4'], '.2f')

params=design['params']
for key in ['encoder_non_embedding','decoder','compute_params','engram_table','token_embedding','codebook','stored_total']:
    displayed('parameter count '+key, params[key], ',d')
displayed('core without book', params['concept_module']-params['codebook'], ',d')
check('parameter totals reconcile', params['encoder_non_embedding']+params['concept_module']-params['codebook']+params['decoder']==params['compute_params'] and params['compute_params']+params['engram_table']+params['token_embedding']+params['codebook']==params['stored_total'])
displayed('scratch token-slot budget', 4050*2*16*1024, ',d')
check('depth expectation', .75*2+.25*4==2.5 and '2.5 + 1.5p' in proposal)
for r in ['2','2.5','4']:
    displayed(f'MAC proxy R{r} million', design['analytical_macs_per_token'][r]['total']/1e6, '.2f')
displayed('R4 versus mean-R2.5 MAC percent', 100*(design['analytical_macs_per_token']['4']['total']/design['analytical_macs_per_token']['2.5']['total']-1), '.1f')
displayed('dense parameter state GB', params['stored_total']*4*4/1e9, '.2f')
p1=runs['v1']['config']['params'];p2=runs['v2']['config']['params']
displayed('combined token module shrink percent', 100*(1-(p2['encoder_non_embedding']+p2['decoder'])/(p1['encoder_non_embedding']+p1['decoder'])), '.1f')
check('mechanical design checks passed', design['passed'])

snapshot_mismatches=[];live_code_mismatches=[];frozen_count=0;code_count=0
for row in inventory:
    if 'sha256' not in row:
        continue
    frozen_count+=1
    p=S/row['path']
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        snapshot_mismatches.append(row['path'])
    if re.match(r'run_(m2_25M|v2_19M)/code/',row['path']):
        code_count+=1
        p=ROOT/row['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            live_code_mismatches.append(row['path'])
check('all captured text hashes agree with inventory', not snapshot_mismatches, dict(count=frozen_count,mismatches=snapshot_mismatches))
check('active v1/v2 code unchanged since snapshot', not live_code_mismatches, dict(count=code_count,mismatches=live_code_mismatches))

missing=[];volatile=[];bad_lines=[];bad_spaces=[];count=0
for p in [ROOT/'proposed_v3.md',A/'RESULTS.md',A/'FILE_COVERAGE.md']:
    txt=p.read_text()
    check(f'{p.name} no unresolved placeholders', '{{' not in txt)
    for match in re.finditer(r'\]\(<([^>]+)>\)|\]\(([^)]+)\)',txt):
        target=match[1] or match[2]
        if not target.startswith('/'):
            continue
        count+=1
        pathline=re.match(r'^(.*?)(?::(\d+))?$',target)
        path=Path(pathline[1]);line=pathline[2]
        if ' ' in target and match[1] is None:
            bad_spaces.append(target)
        if path==A/'proposal_review_checks.json':
            continue  # This script writes that artifact below.
        if not path.exists():
            if path.suffix=='.pt' and '/runs/' in str(path):
                volatile.append(str(path))
            else:
                missing.append(str(path))
        elif line and int(line)>len(path.read_text().splitlines()):
            bad_lines.append(target)
check('nonvolatile local references resolve', not missing and not bad_lines, dict(link_count=count,missing=missing,bad_lines=bad_lines,rotated_checkpoint_links=volatile))
check('space-containing file links are bracketed', not bad_spaces,bad_spaces)
check('evidence and future-work distinctions present', all(x in proposal for x in ['v3 has not been trained','not the full official','not a controlled single-variable','sidecars','not an independent second reviewer','Still required before launch','corrected v1 control','separately trained fixed-R1']))

out=dict(time=datetime.datetime.now().astimezone().isoformat(),scope='Self-review of key arithmetic/displayed values, file links, frozen source integrity, existing mechanics results and explicit caveats. Not a proof of learning quality or an independent review.',evidence_cutoff=snap['time'],checks=checks,passed=all(x['passed'] for x in checks),proposal_sha256=hashlib.sha256((ROOT/'proposed_v3.md').read_bytes()).hexdigest(),corrections_from_review=['Corrected last-interval throughput from 1,208 to 1,217 slots/s.','Made AdamW hyperparameters, auxiliary normalization and codebook restart semantics explicit.','Reformatted local paths containing spaces.','Kept fresh depth/branch results distinct from historical gradient/routing diagnostics.'],limitations=['No v3 has been trained.','Trainer, sidecars, deterministic resume, actual full-batch equivalence and MPS profiling remain implementation work.','Historical corpus/paper audits were reused with explicit coverage caveats.','Current training may have advanced after the frozen cutoff.'])
(A/'proposal_review_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],frozen_text_files=frozen_count,active_code_files=code_count,local_links=count,rotated_checkpoint_links=volatile),indent=2))
raise SystemExit(0 if out['passed'] else 1)
