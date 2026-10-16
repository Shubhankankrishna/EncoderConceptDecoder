"""Read-only snapshots and comparisons. Writes only inside this review directory."""
from pathlib import Path
import ast,datetime,hashlib,json,math,sys
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
ROOT=A.parent
V1=ROOT/'run_m2_25M'; V2=ROOT/'run_v2_19M'
now=datetime.datetime.now().astimezone().isoformat()

def dump(name,x): (A/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def jl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

inv=[]
for p in sorted(V2.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(V2);st=p.stat()
    excluded=any(x in rel.parts for x in ['__pycache__','vendor','.venv']) or p.name=='.DS_Store'
    row=dict(path=str(rel),bytes=st.st_size,inode=st.st_ino,mtime=st.st_mtime,category='dependency/cache/OS' if excluded else 'project')
    if not excluded and p.suffix!='.pt':
        raw=p.read_bytes(); row['sha256']=hashlib.sha256(raw).hexdigest()
        try:
            txt=raw.decode();row['lines']=len(txt.splitlines())
            dest=A/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            if p.suffix=='.py':row['functions']=[n.name for n in ast.walk(ast.parse(txt)) if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
            if p.suffix=='.json':json.loads(txt)
            if p.suffix=='.jsonl':row['records']=len([json.loads(l) for l in txt.splitlines() if l.strip()])
        except Exception as e:row['parse_error']=str(e)
    inv.append(row)
dump('inventory.json',inv)

runs={}
for label,path in [('v1_main',V1/'runs/main'),('v1_cont',V1/'runs/cont'),('v2',V2/'runs/v2_48h')]:
    runs[label]={'config':json.loads((path/'config.json').read_text()),'train':jl(path/'train_log.jsonl'),'eval':jl(path/'log.jsonl'),'guards':jl(path/'GUARDS.jsonl')}
    if (path/'live.json').exists():runs[label]['live']=json.loads((path/'live.json').read_text())
matched=[]
one=[x for x in runs['v1_main']['train'] if 'ntp' in x]
two=[x for x in runs['v2']['train'] if 'ntp' in x]
for x in one:
    s=x['step'];rs=[r for r in two if s-50<r['step']<=s]
    if len(rs)!=2 or rs[-1]['step']!=s:continue
    n2=sum(r['ntp'] for r in rs)/2
    matched.append(dict(steps=f'{s-49}–{s}',step=s,tokens=s*32768,v1_ntp=x['ntp'],v2_ntp=n2,delta=n2-x['ntp'],perplexity_ratio=math.exp(n2-x['ntp']),
                        v1_wall_h=x['wall_h'],v2_wall_h=rs[-1]['wall_h'],v1_lr=x['lr'],v2_lr=rs[-1]['lr'],
                        v1_throughput=x['tok_s'],v2_throughput_harmonic=2/sum(1/r['tok_s'] for r in rs)))
perpass=[]
for x in two:
    p=x['ncp_pass'];best=min(range(len(p)),key=lambda i:p[i])
    perpass.append(dict(step=x['step'],K=x['R_last'],pass1=p[0],pass2=p[1],pass6=p[5],final=p[-1],best_pass=best+1,final_minus_best=p[-1]-p[best],final_minus_first=p[-1]-p[0],drop=x['drop'],route_div=x['route_div']))

prior=json.loads((ROOT/'audit_v2_2026-09-28/inventory.json').read_text())
data=[]
for x in prior:
    if not x['path'].startswith('data/'):continue
    p=V1/x['path'];s=p.stat()
    data.append(dict(path=x['path'],same_inode=s.st_ino==x['inode'],same_size=s.st_size==x['bytes'],same_mtime=s.st_mtime==x['mtime']))
code_compare=[]
for p in sorted((V2/'code').glob('*')):
    if not p.is_file():continue
    q=V1/'code'/p.name
    if q.exists():code_compare.append(dict(file=p.name,identical=p.read_bytes()==q.read_bytes()))
dump('snapshot.json',dict(time=now,runs=runs,matched_train_windows=matched,ncp_depth_observations=perpass,
                          shared_data_unchanged_since_prior_audit=data,v1_v2_code_identity=code_compare,
                          inventory_summary=dict(total=len(inv),project=sum(x['category']=='project' for x in inv),by_folder=dict(Counter(x['path'].split('/')[0] for x in inv)))))
print(now)
print('matched windows',json.dumps(matched,indent=1))
print('data unchanged',all(x['same_inode'] and x['same_size'] and x['same_mtime'] for x in data),'files',len(data))
print('inventory',len(inv),sum(x['category']=='project' for x in inv))
