"""Boundary-correct text packing and harder, independently generated symbolic tasks."""
import gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';T=384;K=4
PAD,EOS,GNAV,SEP,TO,DONE=range(6);NODE0=6;COL0=30;N_COL=3;N_LABELS=24

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def prepare_text():
    if (DATA/'manifest.json').exists():return json.loads((DATA/'manifest.json').read_text())
    shutil.copyfile(ROOT.parent/'v5_tiny/data/tokenizer.json',DATA/'tokenizer.json');tok=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    rows=[];exact=set();near_bands={};removed=0
    with gzip.open(DATA/'fineweb_edu.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);txt=r['text'];canonical=' '.join(txt.lower().split());h=hashlib.sha256(canonical.encode()).hexdigest()
            if h in exact:removed+=1;continue
            # Explicit lexical near-duplicate filter: bottom-32 hashes of 5-word shingles.
            words=re.findall(r'\w+',canonical)[:2000]
            sh={hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest() for i in range(max(1,len(words)-4))}
            signature=tuple(sorted(sh)[:32]);keys=[(b,signature[b*4:(b+1)*4]) for b in range(8)]
            possible=set()
            for key in keys:possible.update(near_bands.get(key,()))
            if any(len(set(signature)&set(rows[i]['signature']))/max(1,len(set(signature)|set(rows[i]['signature'])))>=.8 for i in possible):
                removed+=1;continue
            r['signature']=signature;r['hash']=h;exact.add(h)
            idx=len(rows)
            for key in keys:near_bands.setdefault(key,[]).append(idx)
            rows.append(r)
    rng=np.random.RandomState(71005);order=rng.permutation(len(rows));n_dev=1500;n_test=1500
    split_ids={'dev':order[:n_dev],'test':order[n_dev:n_dev+n_test],'train':order[n_dev+n_test:]}
    manifest={'fresh_source':json.loads((DATA/'fresh_text_source.json').read_text()),'splits':{},'removed_exact_or_lexical_near_duplicates':removed,
        'near_duplicate_filter':'bottom-32 5-word-shingle hashes, 8x4 candidate bands, signature Jaccard >=.8; first 2000 words; not semantic deduplication',
        'tokenizer_sha256':digest(DATA/'tokenizer.json'),'tiny_stories':'legacy training corpus only; its inspected validation corpus is development diagnostic only',
        'context':T,'symbolic_training':'streaming tasks with evaluation canonical instances excluded','split_seed':71005}
    for split,ids in split_ids.items():
        flat=[];offsets=[0];hashes=[];urls=[]
        for begin in range(0,len(ids),128):
            sub=ids[begin:begin+128];encoded=tok.encode_batch([rows[i]['text'] for i in sub])
            for i,e in zip(sub,encoded):
                seq=e.ids+[EOS];flat.extend(seq);offsets.append(len(flat));hashes.append(rows[i]['hash']);urls.append(rows[i].get('url'))
        np.save(DATA/f'edu_{split}.npy',np.asarray(flat,np.uint16));np.save(DATA/f'edu_{split}_offsets.npy',np.asarray(offsets,np.int64))
        (DATA/f'edu_{split}_documents.json').write_text(json.dumps(dict(hashes=hashes,urls=urls)))
        manifest['splits'][split]=dict(documents=len(ids),tokens=len(flat),sha256=digest(DATA/f'edu_{split}.npy'))
        print(json.dumps({'text_split':split,**manifest['splits'][split]}),flush=True)
    ts=np.load(ROOT.parent/'v5_tiny/data/ts_train.npy',mmap_mode='r');offsets=np.r_[0,np.flatnonzero(ts==EOS)+1]
    np.save(DATA/'ts_train_offsets.npy',offsets);manifest['legacy_train_tokens']=len(ts);manifest['legacy_train_documents']=len(offsets)-1
    manifest['legacy_train_sha256']=digest(ROOT.parent/'v5_tiny/data/ts_train.npy')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def parse_prompt(prompt):
    p=list(map(int,prompt));marker,a,b,c=p[:4];edges={}
    for j in range(4,len(p),4):
        src,col,dst,sep=p[j:j+4]
        if sep!=SEP:raise ValueError('Malformed public header')
        edges[(src-NODE0,col-COL0)]=dst-NODE0
    if marker==GNAV:return 'graph',edges,b-NODE0,a-NODE0,None
    if marker==TO:return 'lookup',edges,a-NODE0,None,b-COL0
    if marker==DONE:return 'pointer',edges,a-NODE0,c-NODE0,b-COL0
    if marker==EOS:return 'compose',edges,a-NODE0,c-COL0,b-COL0
    raise ValueError('Unknown symbolic instruction')

def verify(prompt,response):
    task,edges,start,goal,colour=parse_prompt(prompt);response=list(map(int,response));cur=start;hops=0;reached=False
    if task in ('lookup','compose'):
        value=edges[(start,colour)]
        if task=='compose':value=edges[(value,goal)]
        expect=[NODE0+value,DONE,SEP,SEP]
        ok=response==expect
        return dict(success=ok,reason='accepted' if ok else 'wrong_lookup',format_ok=len(response)==4,legal_prefix=0.,hops=0)
    for j in range(0,len(response),4):
        chunk=response[j:j+4]
        if len(chunk)!=4:return dict(success=False,reason='missing_terminator',format_ok=False,legal_prefix=hops/(hops+1),hops=hops)
        if chunk[0]==DONE:
            if chunk!=[DONE,SEP,SEP,SEP]:reason='malformed_terminator'
            elif task=='graph' and (hops==0 or cur!=goal):reason='wrong_goal'
            elif task=='pointer' and hops!=goal:reason='wrong_hop_count'
            elif j+4!=len(response):reason='trailing_tokens'
            else:reason='accepted'
            return dict(success=reason=='accepted',reason=reason,format_ok=reason not in ('malformed_terminator','trailing_tokens'),legal_prefix=hops/(hops+1),hops=hops)
        col,to,dst,sep=chunk
        if not (COL0<=col<COL0+N_COL and to==TO and NODE0<=dst<NODE0+N_LABELS and sep==SEP):reason='malformed_walk'
        elif task=='pointer' and col-COL0!=colour:reason='wrong_colour'
        elif edges.get((cur,col-COL0))!=dst-NODE0:reason='illegal_transition'
        elif task=='pointer' and hops>=goal:reason='too_many_hops'
        else:reason=None
        if reason:return dict(success=False,reason=reason,format_ok=reason!='malformed_walk',legal_prefix=hops/(hops+1),hops=hops)
        cur=dst-NODE0;hops+=1;reached|=task=='graph' and cur==goal
    return dict(success=False,reason='missing_terminator',format_ok=True,legal_prefix=hops/(hops+1),hops=hops)

def canonical(prompt):
    a=np.asarray(prompt,dtype=np.int64);edges=a[4:].reshape(-1,4);order=np.argsort(edges[:,0]*100+edges[:,1])
    return hashlib.sha256(a[:4].tobytes()+edges[order,:3].tobytes()).hexdigest()

def shortest(f,start):
    dist={start:0};paths={start:[]};q=[start]
    for v in q:
        for c in range(N_COL):
            u=int(f[c,v])
            if u not in dist:dist[u]=dist[v]+1;paths[u]=paths[v]+[(c,u)];q.append(u)
    return dist,paths

def gen_symbolic(rng,n=10,task='graph',hard=False,min_dist=4,layout='shuffle'):
    for _ in range(2000):
        labels=rng.choice(N_LABELS,n,replace=False);f=rng.randint(n-1,size=(N_COL,n));f+=f>=np.arange(n)[None];start=int(rng.randint(n))
        if task=='graph':
            if hard:
                dist,paths=shortest(f,start);far=[g for g,d in dist.items() if d>=min_dist]
                if not far:continue
                goal=int(rng.choice(far));walk=paths[goal]
            else:
                cur=start;walk=[];visited=[start]
                for _ in range(int(rng.randint(2,6))):
                    colour=int(rng.randint(N_COL));duration=int(rng.randint(1,5))
                    for _ in range(duration):cur=int(f[colour,cur]);walk.append((colour,cur));visited.append(cur)
                goal=cur
                if goal in visited[:-1]:continue
            instruction=[GNAV,NODE0+int(labels[goal]),NODE0+int(labels[start]),SEP]
        elif task=='pointer':
            colour=int(rng.randint(N_COL));hops=int(rng.randint(8,21) if hard else rng.randint(2,13));cur=start;walk=[]
            for _ in range(hops):cur=int(f[colour,cur]);walk.append((colour,cur))
            instruction=[DONE,NODE0+int(labels[start]),COL0+colour,NODE0+hops]
        elif task=='compose':
            colour=int(rng.randint(N_COL));second=int(rng.randint(N_COL));walk=[];instruction=[EOS,NODE0+int(labels[start]),COL0+colour,COL0+second]
        else:
            colour=int(rng.randint(N_COL));walk=[];instruction=[TO,NODE0+int(labels[start]),COL0+colour,SEP]
        edges=[(v,c) for v in range(n) for c in range(N_COL)]
        if layout=='shuffle':rng.shuffle(edges)
        elif layout=='reverse':edges=list(reversed(edges))
        prompt=instruction+[z for v,c in edges for z in [NODE0+int(labels[v]),COL0+c,NODE0+int(labels[f[c,v]]),SEP]]
        if task=='compose':response=[NODE0+int(labels[f[second,f[colour,start]]]),DONE,SEP,SEP]
        elif task=='lookup':response=[NODE0+int(labels[f[colour,start]]),DONE,SEP,SEP]
        else:response=[z for c,u in walk for z in [COL0+c,TO,NODE0+int(labels[u]),SEP]]+[DONE,SEP,SEP,SEP]
        if len(prompt)+len(response)>T:continue
        x=np.full(T,PAD,np.int64);docs=np.full(T,-1,np.int64);mask=np.zeros(T,np.float32)
        x[:len(prompt)+len(response)]=prompt+response;docs[:len(prompt)+len(response)]=0;mask[len(prompt)-1:len(prompt)+len(response)-1]=1
        meta=dict(task=task,nodes=n,prompt_len=len(prompt),response_len=len(response),canonical=canonical(prompt),hard=hard)
        return x,mask,docs,meta
    raise RuntimeError(f'Could not generate {task} n={n} hard={hard}')

def prepare_symbolic():
    seen=set();manifest={}
    suites=[('standard',10,'graph',False,4),('hard',14,'graph',True,4),('long',18,'graph',True,5),('pointer',14,'pointer',False,4),('pointer_long',18,'pointer',True,4),('lookup',18,'lookup',False,4),('layout',14,'graph',True,4),('compose',18,'compose',False,4)]
    for split in ('dev','test'):
        for i,(name,n,task,hard,dist) in enumerate(suites):
            rng=np.random.RandomState(73000+1000*(split=='test')+i);rows=[];count=128 if split=='dev' else (512 if name in ('standard','hard') else 256)
            while len(rows)<count:
                row=gen_symbolic(rng,n,task,hard,dist,'reverse' if name=='layout' else 'shuffle')
                if row[3]['canonical'] in seen:continue
                seen.add(row[3]['canonical']);rows.append(row)
            np.savez(DATA/f'{split}_{name}.npz',x=np.stack([r[0] for r in rows]),m=np.stack([r[1] for r in rows]),docs=np.stack([r[2] for r in rows]),prompt_len=np.array([r[3]['prompt_len'] for r in rows]))
            manifest[f'{split}_{name}']=dict(n=count,nodes=n,task=task,hard=hard,min_distance=dist,sha256=digest(DATA/f'{split}_{name}.npz'))
    (DATA/'evaluation_symbolic_hashes.txt').write_text('\n'.join(sorted(seen))+'\n')
    return manifest

class Text:
    def __init__(self,split='train'):
        self.split=split;self.edu=np.load(DATA/f'edu_{split}.npy',mmap_mode='r');self.off=np.load(DATA/f'edu_{split}_offsets.npy')
        if split=='train':self.ts=np.load(ROOT.parent/'v5_tiny/data/ts_train.npy',mmap_mode='r');self.ts_off=np.load(DATA/'ts_train_offsets.npy')
    def row(self,rng,edu_only=False,single=False):
        x=np.full(T,PAD,np.int64);docs=np.full(T,-1,np.int64);p=0;doc=0;identities=[]
        while p<T:
            use_edu=self.split!='train' or edu_only or rng.rand()<.5
            flat,off=(self.edu,self.off) if use_edu else (self.ts,self.ts_off)
            i=int(rng.randint(len(off)-1));start,end=map(int,off[i:i+2]);available=end-start
            if available>T-p:start+=K*int(rng.randint(max(1,(available-(T-p))//K+1)))
            n=min(end-start,T-p);x[p:p+n]=flat[start:start+n];docs[p:p+n]=doc
            identities.append(('edu' if use_edu else 'ts',i));p+=K*((n+K-1)//K);doc+=1
            if single:break
        m=np.zeros(T,np.float32);m[:-1]=((docs[:-1]>=0)&(docs[:-1]==docs[1:])).astype(np.float32)
        return x,m,docs,identities

class Mixer:
    def __init__(self,seed,batch=32):
        assert batch%2==0;self.text=Text();self.rng=np.random.RandomState(seed);self.batch_size=batch
        self.excluded=set((DATA/'evaluation_symbolic_hashes.txt').read_text().splitlines());self.seen_docs=set();self.seen_symbols=set();self.seen_tasks={};self.total_targets=0
    def batch(self,device='cpu',progress=1.,single_text=False):
        rows=[];metas=[]
        for _ in range(self.batch_size//2):
            x,m,d,ids=self.text.row(self.rng,single=single_text);rows.append((x,m,d));self.seen_docs.update(ids)
        for _ in range(self.batch_size//2):
            n=10 if progress<.15 else int(self.rng.choice([10,14,18],p=[.5,.3,.2]))
            task=str(self.rng.choice(['graph','pointer','lookup','compose'],p=[.6,.2,.1,.1]))
            while True:
                x,m,d,meta=gen_symbolic(self.rng,n,task)
                if meta['canonical'] not in self.excluded:break
            rows.append((x,m,d));metas.append(meta);self.seen_symbols.add(meta['canonical']);self.seen_tasks[task]=self.seen_tasks.get(task,0)+1
        xs,ms,ds=map(np.stack,zip(*rows));domain=np.array([0]*(self.batch_size//2)+[1]*(self.batch_size//2),np.int64)
        self.total_targets+=int(ms[:,:-1].sum())
        return dict(x=torch.as_tensor(xs,device=device),m=torch.as_tensor(ms,device=device),docs=torch.as_tensor(ds,device=device),domain=torch.as_tensor(domain,device=device),metas=metas)
    def state(self):return dict(rng=self.rng.get_state(),seen_docs=sorted(self.seen_docs),seen_symbols=sorted(self.seen_symbols),seen_tasks=self.seen_tasks,total_targets=self.total_targets)
    def restore(self,s):
        self.rng.set_state(s['rng']);self.seen_docs={tuple(v) for v in s['seen_docs']};self.seen_symbols=set(s['seen_symbols']);self.seen_tasks=s['seen_tasks'];self.total_targets=s['total_targets']

def fixed_text(split,n=128,seed=71200):
    t=Text(split);rng=np.random.RandomState(seed);r=[t.row(rng) for _ in range(n)]
    return {k:torch.tensor(v) for k,v in zip(('x','m','docs'),map(np.stack,zip(*[a[:3] for a in r])))}

if __name__=='__main__':
    prepare_text();j=json.loads((DATA/'manifest.json').read_text());j['symbolic']=prepare_symbolic();(DATA/'manifest.json').write_text(json.dumps(j,indent=2));print('Data preparation complete',flush=True)
