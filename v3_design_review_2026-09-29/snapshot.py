"""Freeze evidence for the v3 proposal. Does not modify either training run."""
from pathlib import Path
import os,json,hashlib,datetime,ast,statistics
A=Path(__file__).resolve().parent;ROOT=A.parent
SKIP={'.venv','vendor','__pycache__','.git','.matplotlib','.claude',A.name}
inventory=[];excluded=[]
for base,dirs,files in os.walk(ROOT):
    for d in list(dirs):
        if d in SKIP or d in ['audit_v2_2026-09-28','v2_underperformance_review_2026-09-28']:
            excluded.append(str((Path(base)/d).relative_to(ROOT)));dirs.remove(d)
    for name in files:
        p=Path(base)/name;rel=str(p.relative_to(ROOT));st=p.stat()
        row=dict(path=rel,bytes=st.st_size,inode=st.st_ino,mtime=st.st_mtime)
        if p.suffix in ['.py','.md','.json','.jsonl','.txt','.log','.sh','.js','.html','.css'] and st.st_size<8_000_000:
            raw=p.read_bytes();row['sha256']=hashlib.sha256(raw).hexdigest()
            try:
                txt=raw.decode();row['lines']=len(txt.splitlines())
                if p.suffix=='.py':ast.parse(txt)
                if p.suffix=='.json':json.loads(txt)
                if p.suffix=='.jsonl':row['records']=len([json.loads(x) for x in txt.splitlines() if x.strip()])
            except Exception as exc:row['parse_error']=str(exc)
            dest=A/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        inventory.append(row)
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
runs={}
for key,p in [('v1',ROOT/'run_m2_25M/runs/main'),('v1_cont',ROOT/'run_m2_25M/runs/cont'),('v2',ROOT/'run_v2_19M/runs/v2_48h')]:
    runs[key]=dict(config=json.loads((p/'config.json').read_text()),train=jl(p/'train_log.jsonl'),eval=jl(p/'log.jsonl'),guards=jl(p/'GUARDS.jsonl'))
    if (p/'live.json').exists():runs[key]['live']=json.loads((p/'live.json').read_text())
    if (p/'eval_final.json').exists():runs[key]['final']=json.loads((p/'eval_final.json').read_text())
matched=[]
t1={x['step']:x for x in runs['v1']['train'] if 'ntp' in x}
t2={x['step']:x for x in runs['v2']['train'] if 'ntp' in x}
for st,x in t1.items():
    if st not in t2 or st-25 not in t2:continue
    y,z=t2[st-25],t2[st]
    matched.append(dict(step=st,v1_ntp=x['ntp'],v2_ntp=(y['ntp']+z['ntp'])/2,
                        v1_lr=x['lr'],v2_lr=z['lr'],v1_h=x['wall_h'],v2_h=z['wall_h'],
                        v1_tps=x['tok_s'],v2_tps=2/(1/y['tok_s']+1/z['tok_s'])))
out=dict(time=datetime.datetime.now().astimezone().isoformat(),runs=runs,matched_train=matched,inventory_count=len(inventory),excluded_folders=excluded)
(A/'snapshot.json').write_text(json.dumps(out,indent=2))
(A/'inventory.json').write_text(json.dumps(inventory,indent=2))
print('frozen',out['time'],'files',len(inventory),'parse_errors',[e['path'] for e in inventory if 'parse_error' in e])
print('latest_v2_eval',runs['v2']['eval'][-1]['step'],'live',runs['v2']['live']['step'])
for x in matched:
    if x['step']%250==0 or x==matched[-1]:print(x)
