from pathlib import Path
import json,hashlib
from pypdf import PdfReader
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent/'papers';out.mkdir(exist_ok=True)
inv=json.loads((out.parent/'inventory.json').read_text());rs=[];seen={}
for r in inv['files']:
 if r['extension']!='.pdf' or '.venv/' in r['path']:continue
 p=root/r['path'];h=hashlib.sha256(p.read_bytes()).hexdigest();rec={'path':r['path'],'sha256':h}
 if h in seen:rec['duplicate_of']=seen[h];rs.append(rec);continue
 seen[h]=r['path']
 try:
  a=PdfReader(p);parts=[page.extract_text() or '' for page in a.pages]
  dest=out/(h[:12]+'.txt');dest.write_text('\n\n'.join(f'=== PAGE {i+1} ===\n{s}' for i,s in enumerate(parts)))
  rec.update(pages=len(parts),text_file=str(dest.relative_to(root)),chars=sum(map(len,parts)),metadata=str(a.metadata))
 except Exception as e:rec['error']=str(e)
 rs.append(rec)
(out.parent/'paper_inventory.json').write_text(json.dumps(rs,indent=2))
print('PDF files',len(rs),'unique',len(seen),'pages',sum(r.get('pages',0) for r in rs),'errors',[r for r in rs if 'error' in r])
for r in rs:
 if r['path'].startswith('Imp Paper'):print(r)
