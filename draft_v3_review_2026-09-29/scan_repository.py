from pathlib import Path
import json,ast,hashlib,collections,zipfile,re,datetime
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
inv=json.loads((out/'inventory.json').read_text()); records=[]; seen={}; jserrs=[]
for r in inv['files']:
 p=root/r['path'];parts=p.parts
 if r['kind']!='text' or '.venv' in parts:continue
 rec={k:r[k] for k in ('path','bytes','sha256','lines')};h=r['sha256']
 if h in seen:rec['duplicate_of']=seen[h];records.append(rec);continue
 seen[h]=r['path'];s=p.read_text();ext=p.suffix
 try:
  if ext=='.json':
   j=json.loads(s);rec['parsed_as']='json';rec['structure']=list(j)[:30] if isinstance(j,dict) else {'type':type(j).__name__,'len':len(j) if isinstance(j,list) else None}
  elif ext=='.jsonl':
   n=0;keys=collections.Counter();bad=[]
   for i,l in enumerate(s.splitlines(),1):
    if not l.strip():continue
    try:
     j=json.loads(l);n+=1
     if isinstance(j,dict):keys.update(j.keys())
    except Exception as e:bad.append([i,str(e)])
   rec.update(parsed_as='jsonl',records=n,fields=dict(keys),bad_lines=bad)
  elif ext=='.py':
   tree=ast.parse(s);rec.update(parsed_as='python_ast',definitions=[{'name':n.name,'line':n.lineno} for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))])
  elif ext=='.md':rec.update(parsed_as='markdown',headings=re.findall(r'^#{1,6} .+$',s,re.M))
  else:rec.update(parsed_as='text',keywords={k:len(re.findall(k,s,re.I)) for k in ['NONFINITE','Traceback','FAIL','PASS']})
 except Exception as e:rec['parse_error']=str(e);jserrs.append([r['path'],str(e)])
 records.append(rec)
(out/'text_scan.json').write_text(json.dumps(records,indent=2))
print('Non-dependency text files',len(records),'unique',len(seen),'parse errors',jserrs)
for top in sorted({r['path'].split('/')[0] for r in records}):
 a=[r for r in records if r['path'].split('/')[0]==top];print(top,'files',len(a),'unique',sum('duplicate_of' not in x for x in a))
# Source-artifact archive metadata without executing payloads.
z=[]
for r in inv['files']:
 if r['extension'] in ('.zip','.docx') and '.venv/' not in r['path']:
  with zipfile.ZipFile(root/r['path']) as a:
   z.append({'path':r['path'],'entries':[{'path':i.filename,'bytes':i.file_size} for i in a.infolist()]})
   if r['extension']=='.docx':(out/'encoder_docx_text.txt').write_text(re.sub('<[^>]+>',' ',a.read('word/document.xml').decode()))
(out/'archive_inventory.json').write_text(json.dumps(z,indent=2))
