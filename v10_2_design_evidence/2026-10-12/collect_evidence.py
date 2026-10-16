"""Read-only capture of a running experiment; never loads weights or GPU tensors."""
from pathlib import Path
from collections import deque
from datetime import datetime,timezone
import hashlib,json,math,statistics,urllib.request
ROOT=Path(__file__).resolve().parents[2];DEST=Path(__file__).resolve().parent
if (DEST/'telemetry_summary.json').exists():
 raise SystemExit('Evidence already captured. Copy this script to a new sibling evidence directory before capturing again.')
run=ROOT/'v10_1/runs/pretrain_16h'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with urllib.request.urlopen('http://127.0.0.1:8767/api/v1/overview') as response: overview=json.load(response)
(DEST/'overview.json').write_text(json.dumps(overview,indent=2)+'\n')
watermark=overview['run']['status']['step'];recent=deque(maxlen=5000);initial=[];count=0;bad=[];missing=[];previous=0
with (run/'train.jsonl').open() as f:
 for line in f:
  try:e=json.loads(line)
  except json.JSONDecodeError:continue
  if e['step']>watermark:break
  if e['step']!=previous+1:missing.append([previous,e['step']])
  previous=e['step'];count+=1;recent.append(e)
  if len(initial)<1000:initial.append(e)
  if not all(math.isfinite(v) for v in [*e['loss'].values(),e['dense_grad_norm'],e['engram_grad_norm']]):bad.append(e['step'])
recent=list(recent)
def quantile(a,p):return sorted(a)[round((len(a)-1)*p)]
def panel(rows):
 output={}
 for family in ['text','math','sudoku_easy','sudoku_hard']:
  a=[e for e in rows if e['family']==family];n=sum(e['supervised_tokens'] for e in a)
  codes=[sum(e['code_counts'][k] for e in a) for k in range(8)];total=sum(codes)
  output[family]=dict(updates=len(a),supervised_tokens=n,token_nll=sum(e['loss']['token']*e['supervised_tokens'] for e in a)/n,
   chunk_loss=statistics.mean(e['loss']['chunk'] for e in a),latent_loss=statistics.mean(e['loss']['latent'] for e in a),
   code_counts=codes,effective_codes=math.exp(-sum((x/total)*math.log(x/total) for x in codes if x)),
   candidate_fraction=sum(e['candidate_requested'] for e in a)/len(a))
 return output
history=[]
for p in sorted(run.glob('evaluation_*.json')):
 e=json.loads(p.read_text())
 if e['step']>watermark:continue
 (DEST/p.name).write_bytes(p.read_bytes())
 history.append(dict(file=p.name,sha256=sha(p),step=e['step'],wall_time=e['wall_time'],families=e['families'],
  generation={k:{a:b for a,b in v.items() if a!='examples'} for k,v in e['generation'].items()},memory_writes=e['memory_writes']))
history.sort(key=lambda x:x['step'])
norms=[e['dense_grad_norm'] for e in recent]
exposure=recent[-1]['family_supervised_tokens'];total=sum(exposure.values())
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),watermark=watermark,source_sha256=recent[-1]['source_sha256'],data_sha256=recent[-1]['data_sha256'],
 records=count,step_gaps=missing,nonfinite_records=bad,elapsed_hours=recent[-1]['elapsed_hours'],input_tokens=recent[-1]['tokens'],
 first1000=panel(initial),last5000=panel(recent),last5000_dense_grad=dict(median=quantile(norms,.5),p95=quantile(norms,.95),maximum=max(norms),clipped_fraction=sum(x>1 for x in norms)/len(norms)),
 exposure=exposure,exposure_fraction={k:v/total for k,v in exposure.items()},sustained_tps_last5000=sum(e['input_tokens'] for e in recent[1:])/((recent[-1]['elapsed_hours']-recent[0]['elapsed_hours'])*3600),
 evaluation_history=history,latest_training=recent[-1],files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'V10_1_RCDL_Architecture.md',ROOT/'v10_1/model.py',ROOT/'v10_1/memory.py',ROOT/'v10_1/evaluate.py',ROOT/'v10_1/train.py',ROOT/'v10_1/data/manifest.json']})
(DEST/'telemetry_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['evaluation_history','latest_training','files']},indent=2))
print('Latest evaluation',history[-1]['file'],json.dumps(history[-1]['families']))
