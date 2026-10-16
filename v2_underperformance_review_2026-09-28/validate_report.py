"""Check the written review's central numbers against saved evidence, without running a model."""
from pathlib import Path
import datetime,hashlib,json,math,re,statistics
A=Path(__file__).resolve().parent
s=json.loads((A/'snapshot.json').read_text())
g=json.loads((A/'gradient_replication.json').read_text())
r=json.loads((A/'routing_replication.json').read_text())
inv=json.loads((A/'inventory.json').read_text())
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
def near(a,b,tol=1e-6):return abs(a-b)<tol
v1=next(e for e in s['runs']['v1_main']['eval'] if e['step']==265)
v2=next(e for e in s['runs']['v2']['eval'] if e['step']==267)
gap=v2['val_lang_mean']-v1['val_lang_mean']
check('Matched-update validation gap',near(gap,.5733243279705378),gap)
check('Geometric mean perplexity ratio',near(math.exp(gap),1.774154,1e-5),math.exp(gap))
v1time=next(e for e in s['runs']['v1_main']['eval'] if e['step']==365)
check('Matched-time validation gap',near(v2['val_lang_mean']-v1time['val_lang_mean'],1.0141712888))
w=s['matched_train_windows'][-1]
check('Equal-window training gap',near(w['delta'],.5195977020263678))
check('Throughput reduction',near(100*(1-w['v2_throughput_harmonic']/w['v1_throughput']),30.4722,.001))
check('Same-step time penalty',near(100*(w['v2_wall_h']/w['v1_wall_h']-1),38.1245,.001))
p1=s['runs']['v1_main']['config']['params'];p2=s['runs']['v2']['config']['params']
reduction=100*(1-(p2['encoder_non_embedding']+p2['decoder'])/(p1['encoder_non_embedding']+p1['decoder']))
check('Token-module parameter reduction',near(reduction,52.659,.001),reduction)
perpass=s['ncp_depth_observations']
check('All 11 final NCP values exceed pass 1',len(perpass)==11 and all(p['final']>p['pass1'] for p in perpass))
check('Nine best NCP passes at pass 2',sum(p['best_pass']==2 for p in perpass)==9)
for ck,expected in [(138,16),(267,15)]:
    d=json.loads((A/f'checkpoint_{ck:06d}_diagnostics.json').read_text())
    improved=sum(b['depths']['6']['nats'][i]<b['depths']['1']['nats'][i] for b in d['language_batches'] for i in [0,1])
    check(f'R6 vs R1 row count at checkpoint {ck}',improved==expected,improved)
    check(f'Both language averages improve with R6 at checkpoint {ck}',all(v['6']<v['1'] for v in d['language_summary'].values()))
    check(f'Dropless batch-order invariance at checkpoint {ck}',d['batch_order_test']['no_drop']['max_logit_difference']==0)
    if ck==267:
        improvement=statistics.mean(v['6']-v['12'] for v in d['language_summary'].values())
        check('Small R6 to R12 improvement',near(improvement,.0019930495919946),improvement)
        pooled=lambda x:sum(x['sums'])/sum(x['counts'])
        diff=pooled(d['capacity_sweeps']['capacity_1.25']['depths']['6'])-pooled(d['capacity_sweeps']['no_drop']['depths']['6'])
        check('Small instantaneous capacity loss',near(diff,.0004589,1e-6),diff)
for key in ['core_W_in','expert_w2','decoder_Wu','carry_gain']:
    check(f'NextLat gradients oppose NTP in all four batches: {key}',len(g['batches'])==4 and all(b['aggregate']['nextlat'][key]['cosine']<0 for b in g['batches']))
check('Initial-state projection has no final-objective gradient',all(all(b['initial_projection_gradients'][k] is None for k in ['NTP','NextLat_h','NextLat_KL','consistency_weighted']) for b in g['batches']))
check('Initial-state projection has NCP and routing gradients',all(all(b['initial_projection_gradients'][k]>0 for k in ['NCP_weighted','MoE_aux_weighted']) for b in g['batches']))
first=[];second=[];both=[]
for b in r['batches']:
    first.append(100*statistics.mean(z[0]['real_drop'] for z in b['dispatch_records']))
    second.append(100*statistics.mean(z[1]['real_drop'] for z in b['dispatch_records']))
    both.append(100*statistics.mean(z[1]['real_losing_both'] for z in b['dispatch_records']))
check('Second row routing disadvantage in all four batches',all(a<b for a,b in zip(first,second)))
check('Real second-row assignment-drop range',near(min(second),17.659722,.00001) and near(max(second),29.237805,.00001),second)
check('Real second-row both-experts-loss range',near(min(both),4.194445,.00001) and near(max(both),8.180894,.00001),both)
bridge=json.loads((A/'gradient_bridge_check.json').read_text())
check('K12/accum16/drop backward reference passed',bridge['passed'] and bridge['maximum_relative_gradient_difference']<1e-12 and bridge['drop']>0 and bridge['carry_norm_gradient_norm']>0)
check('All shared data metadata unchanged',len(s['shared_data_unchanged_since_prior_audit'])==30 and all(v['same_inode'] and v['same_size'] and v['same_mtime'] for v in s['shared_data_unchanged_since_prior_audit']))
check('Expected inventory count',len(inv)==99 and sum(e['category']=='project' for e in inv)==72)
check('No source snapshot parse errors',not any('parse_error' in e for e in inv))
check('Empty frozen guard log',s['runs']['v2']['guards']==[])
check('Score arithmetic',near(.30*4+.20*4+.20*5+.15*6+.15*8,5.1))
missing=[];badlines=[]
for fn in ['REPORT.md','RESULTS_APPENDIX.md','FILE_COVERAGE.md','SELF_REVIEW.md']:
    p=A/fn
    if not p.exists():missing.append(str(p));continue
    text=p.read_text()
    check(f'No unresolved placeholders in {fn}','{{' not in text and '}}' not in text)
    for target in re.findall(r'\]\((/[^)]+)\)',text):
        m=re.match(r'(.*?):(\d+)$',target);path=Path(m[1] if m else target)
        if not path.exists():missing.append(target)
        elif m and int(m[2])>len(path.read_text().splitlines()):badlines.append(target)
check('All local artifact/source links exist',not missing,missing)
check('All source-line citations in range',not badlines,badlines)
unchanged=[];changed=[]
for e in inv:
    if not e['path'].startswith('code/') or 'sha256' not in e:continue
    p=A.parent/'run_v2_19M'/e['path']
    (unchanged if hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'] else changed).append(e['path'])
check('Active v2 code still matches audit source hashes',not changed,dict(unchanged=len(unchanged),changed=changed))
out=dict(time=datetime.datetime.now().astimezone().isoformat(),passed=all(x['passed'] for x in checks),checks=checks)
(A/'review_checks.json').write_text(json.dumps(out,indent=2))
for c in checks:
    print(('PASS' if c['passed'] else 'FAIL'),c['name'],c['detail'] if not c['passed'] else '')
assert out['passed'], 'Review checks failed; inspect review_checks.json before finalizing.'
