from pathlib import Path
import os,json,hashlib,collections,datetime
root=Path(__file__).resolve().parents[1]; out=Path(__file__).resolve().parent
rows=[]; dirs=[]; summaries=collections.defaultdict(lambda:collections.Counter()); hashes=collections.defaultdict(list)
text_ext={'.py','.md','.txt','.json','.jsonl','.log','.stdout','.sh','.html','.css','.js','.toml','.yaml','.yml','.csv','.tsv','.rtf','.svg'}
for base,ds,fs in os.walk(root,followlinks=False):
 ds.sort();fs.sort()
 ds[:]=[x for x in ds if root/Path(base).relative_to(root)/x != out]
 relbase=Path(base).relative_to(root);dirs.append(str(relbase))
 for name in fs:
  p=Path(base)/name;rel=str(p.relative_to(root));st=p.stat();kind='binary/metadata'
  if p.suffix.lower() in text_ext or (st.st_size<2_000_000 and p.suffix=='' and name not in {'.DS_Store'}):
   try:
    b=p.read_bytes();s=b.decode('utf-8');kind='text';digest=hashlib.sha256(b).hexdigest();hashes[digest].append(rel)
   except (UnicodeError,OSError):s=None
  else:s=None
  r=dict(path=rel,bytes=st.st_size,mtime_ns=st.st_mtime_ns,kind=kind,extension=p.suffix.lower(),symlink=p.is_symlink())
  if s is not None:r.update(sha256=digest,lines=len(s.splitlines()))
  rows.append(r);summaries[rel.split('/')[0]]['files']+=1;summaries[rel.split('/')[0]]['bytes']+=st.st_size;summaries[rel.split('/')[0]][kind]+=1
payload=dict(created=datetime.datetime.now().astimezone().isoformat(),root=str(root),files=rows,directories=dirs,summary={k:dict(v) for k,v in summaries.items()},unique_text_hashes=len(hashes),duplicates=[v for v in hashes.values() if len(v)>1])
(out/'inventory.json').write_text(json.dumps(payload,indent=2))
(out/'inventory.tsv').write_text('path\tbytes\tkind\tsha256\n'+'\n'.join(f"{r['path']}\t{r['bytes']}\t{r['kind']}\t{r.get('sha256','')}" for r in rows)+'\n')
for k,v in summaries.items():print(k,dict(v))
print('TOTAL',len(rows),'files',len(dirs),'directories',len(hashes),'unique text contents')
for rel in ['draft_v3.md','v3_design/v3_ref.py','v3_design/check_engram2.py','v3_design/v3_budget.py','v3_design/REVIEW_v3.md']:
 p=out/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/rel).read_bytes())
