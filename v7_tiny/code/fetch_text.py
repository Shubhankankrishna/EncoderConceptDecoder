"""Fetch a bounded fresh educational-text sample; no training is performed here."""
import gzip,json,hashlib,time
from pathlib import Path
from datasets import load_dataset

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data'/'fineweb_edu.jsonl.gz'
    if out.exists():
        print('Existing complete sample retained',flush=True);return
    temp=out.with_suffix(out.suffix+'.tmp');n=0;chars=0
    stream=load_dataset('HuggingFaceFW/fineweb-edu','sample-10BT',split='train',streaming=True)
    started=time.time()
    with gzip.open(temp,'wt') as f:
        for row in stream:
            text=row['text']
            if len(text)<300:continue
            record={k:row.get(k) for k in ('url','id','score','language','dump')}
            record['text']=text;record['normalized_sha256']=hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
            f.write(json.dumps(record,ensure_ascii=False)+'\n');n+=1;chars+=len(text)
            if n%1000==0:print(json.dumps(dict(documents=n,characters=chars,elapsed_s=time.time()-started)),flush=True)
            if n>=30000:break
    if n<10000:raise RuntimeError(f'Insufficient fresh text: {n} documents')
    temp.replace(out)
    (ROOT/'data'/'fresh_text_source.json').write_text(json.dumps(dict(dataset='HuggingFaceFW/fineweb-edu',configuration='sample-10BT',
        source='https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu',license='ODC-By; see upstream attribution',documents=n,
        characters=chars,seconds=time.time()-started,selection='first 30,000 streaming records with >=300 characters',
        content_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
    print('Fresh text fetch complete',flush=True)

if __name__=='__main__':main()
