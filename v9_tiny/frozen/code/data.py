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

def split_tiny_stories(manifest):
    """Partition unique token-content groups, retaining one representative per group."""
    ts=np.load(ROOT.parent/'v5_tiny/data/ts_train.npy',mmap_mode='r');off=np.load(DATA/'ts_offsets.npy');groups={}
    for i,(start,end) in enumerate(zip(off[:-1],off[1:])):
        key=hashlib.sha256(ts[int(start):int(end)].tobytes()).hexdigest()
        if key not in groups:groups[key]=i
    rows=list(groups.items());order=np.random.RandomState(88889).permutation(len(rows));partitions={}
    for split,indices in [('dev',order[:1500]),('test',order[1500:3000]),('train',order[3000:])]:
        ids=np.array([rows[i][1] for i in indices],np.int64);np.save(DATA/f'ts_{split}_ids.npy',ids)
        hashes=[rows[i][0] for i in indices];(DATA/f'ts_{split}_hashes.txt').write_text('\n'.join(hashes)+'\n')
        partitions[split]=dict(documents=len(ids),ids_sha256=digest(DATA/f'ts_{split}_ids.npy'),content_hashes_sha256=digest(DATA/f'ts_{split}_hashes.txt'))
    manifest['tiny_stories']=dict(tokens=len(ts),sha256=digest(ROOT.parent/'v5_tiny/data/ts_train.npy'),split_policy='exact-token-content-groups-v1',original_documents=len(off)-1,unique_documents=len(rows),duplicates_removed=len(off)-1-len(rows),partitions=partitions,seed=88889,near_duplicate_policy='Exact token duplicates grouped here; no new near-duplicate guarantee')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def prepare_text():
    if (DATA/'manifest.json').exists():
        manifest=json.loads((DATA/'manifest.json').read_text())
        return manifest if manifest.get('tiny_stories',{}).get('split_policy')=='exact-token-content-groups-v1' else split_tiny_stories(manifest)
    source=ROOT.parent/'v7_tiny/data';shutil.copyfile(source/'tokenizer.json',DATA/'tokenizer.json')
    domains=('wikipedia.org','britannica.com','sciencedaily.com','phys.org','scientificamerican.com','newscientist.com','nps.gov','mathforum.org')
    groups={'base':[],'continual':[]};seen=set()
    for old in ('train','dev','test'):
        arr=np.load(source/f'edu_{old}.npy',mmap_mode='r');off=np.load(source/f'edu_{old}_offsets.npy');meta=json.loads((source/f'edu_{old}_documents.json').read_text())
        for i,h in enumerate(meta['hashes']):
            if h in seen:continue
            seen.add(h);url=meta['urls'][i] or '';family='continual' if any(d in url.lower() for d in domains) else 'base'
            groups[family].append((old,i,h,url))
    manifest=dict(protocol='V8 fresh model splits of local previously deduplicated historical corpus; NOT new independent external corpus',split_seed=88008,continual_domains=list(domains),text={},tokenizer_sha256=digest(DATA/'tokenizer.json'),source_manifest_sha256=digest(source/'manifest.json'),legacy_model_exposure='V6/V7 bridge may have seen these documents; bridge is a historical diagnostic only')
    arrays={old:np.load(source/f'edu_{old}.npy',mmap_mode='r') for old in ('train','dev','test')};offsets={old:np.load(source/f'edu_{old}_offsets.npy') for old in arrays}
    for family,rows in groups.items():
        rng=np.random.RandomState(88008+(family=='continual'));order=rng.permutation(len(rows));nd=min(1500,max(64,len(rows)//10));nt=nd
        for split,ids in [('dev',order[:nd]),('test',order[nd:nd+nt]),('train',order[nd+nt:])]:
            pieces=[];outoff=[0];outmeta=[]
            for idx in ids:
                old,i,h,url=rows[idx];start,end=offsets[old][i:i+2];piece=np.asarray(arrays[old][start:end]);pieces.append(piece);outoff.append(outoff[-1]+len(piece));outmeta.append(dict(sha256=h,url=url,source_split=old,source_document=i))
            name=f'{family}_{split}';np.save(DATA/f'{name}.npy',np.concatenate(pieces).astype(np.uint16));np.save(DATA/f'{name}_offsets.npy',np.array(outoff,np.int64));(DATA/f'{name}_documents.json').write_text(json.dumps(outmeta))
            manifest['text'][name]=dict(documents=len(ids),tokens=outoff[-1],sha256=digest(DATA/f'{name}.npy'))
    shutil.copyfile(source/'ts_train_offsets.npy',DATA/'ts_offsets.npy')
    return split_tiny_stories(manifest)

def parse_prompt(prompt):
    p=list(map(int,prompt));marker,a,b,c=p[:4];edges={};begin=8 if marker==EOS and len(p)%12==8 else 4
    shifted=marker==COL0+2
    for j in range(begin,len(p),4):
        row=p[j:j+4]
        src,col,dst,sep=(row[1],row[2],row[0],SEP if row[3]==TO else -1) if shifted else row
        if sep!=SEP:raise ValueError('Malformed public header')
        edges[(src-NODE0,col-COL0)]=dst-NODE0
    if marker==GNAV or shifted:return 'graph',edges,b-NODE0,a-NODE0,None
    if marker==TO:return 'lookup',edges,a-NODE0,None,b-COL0
    if marker==DONE:return 'pointer',edges,a-NODE0,c-NODE0,b-COL0
    if marker==EOS:return ('bind' if begin==8 else 'compose'),edges,a-NODE0,([b-COL0,c-COL0,p[5]-COL0,p[6]-COL0] if begin==8 else c-COL0),b-COL0
    raise ValueError('Unknown symbolic instruction')

def verify(prompt,response):
    task,edges,start,goal,colour=parse_prompt(prompt);response=list(map(int,response));cur=start;hops=0;reached=False
    if task in ('lookup','compose','bind'):
        value=start
        for cl in (goal if task=='bind' else [colour]):value=edges[(value,cl)]
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
    task,edges,start,goal,col=parse_prompt(prompt)
    return hashlib.sha256(json.dumps([task,start,goal,col,sorted((a,b,c) for (a,b),c in edges.items())],sort_keys=True).encode()).hexdigest()

def shortest(f,start):
    dist={start:0};paths={start:[]};q=[start]
    for v in q:
        for c in range(N_COL):
            u=int(f[c,v])
            if u not in dist:dist[u]=dist[v]+1;paths[u]=paths[v]+[(c,u)];q.append(u)
    return dist,paths

def gen_symbolic(rng,n=10,task='graph',hard=False,min_dist=4,layout='shuffle',context=192,label_style='random',labels_mode='walk'):
    for _ in range(2000):
        labels=(np.arange(n) if label_style=='ordered' else rng.choice(N_LABELS,n,replace=False));f=rng.randint(n-1,size=(N_COL,n));f+=f>=np.arange(n)[None];start=int(rng.randint(n))
        if task=='graph':
            if hard:
                dist,paths=shortest(f,start);far=[g for g,d in dist.items() if d>=min_dist]
                if not far:continue
                goal=int(rng.choice(far));walk=paths[goal]
            else:
                cur=start;walk=[];visited=[start]
                previous=-1
                for _ in range(int(rng.randint(2,5))):
                    colour=int(rng.randint(N_COL-1)) if previous>=0 else int(rng.randint(N_COL))
                    if previous>=0:colour+=colour>=previous
                    previous=colour;duration=int(rng.randint(2,5))
                    for _ in range(duration):cur=int(f[colour,cur]);walk.append((colour,cur));visited.append(cur)
                goal=cur
                if goal in visited[:-1]:continue
            instruction=[GNAV,NODE0+int(labels[goal]),NODE0+int(labels[start]),SEP]
        elif task=='pointer':
            colour=int(rng.randint(N_COL));hops=int(rng.randint(8,21) if hard else rng.randint(2,13));cur=start;walk=[]
            for _ in range(hops):cur=int(f[colour,cur]);walk.append((colour,cur))
            instruction=[DONE,NODE0+int(labels[start]),COL0+colour,NODE0+hops]
        elif task=='bind':
            colours=rng.randint(N_COL,size=4);walk=[];instruction=[EOS,NODE0+int(labels[start]),COL0+int(colours[0]),COL0+int(colours[1]),GNAV,COL0+int(colours[2]),COL0+int(colours[3]),SEP]
        elif task=='compose':
            colour=int(rng.randint(N_COL));second=int(rng.randint(N_COL));walk=[];instruction=[EOS,NODE0+int(labels[start]),COL0+colour,COL0+second]
        else:
            colour=int(rng.randint(N_COL));walk=[];instruction=[TO,NODE0+int(labels[start]),COL0+colour,SEP]
        edges=[(v,c) for v in range(n) for c in range(N_COL)]
        if layout=='shuffle':rng.shuffle(edges)
        elif layout=='reverse':edges=list(reversed(edges))
        if layout=='shifted':instruction[0]=COL0+2
        prompt=instruction+[z for v,c in edges for z in ([NODE0+int(labels[f[c,v]]),NODE0+int(labels[v]),COL0+c,TO] if layout=='shifted' else [NODE0+int(labels[v]),COL0+c,NODE0+int(labels[f[c,v]]),SEP])]
        if task=='bind':
            cur=start
            for colour in colours:cur=int(f[int(colour),cur])
            response=[NODE0+int(labels[cur]),DONE,SEP,SEP]
        elif task=='compose':response=[NODE0+int(labels[f[second,f[colour,start]]]),DONE,SEP,SEP]
        elif task=='lookup':response=[NODE0+int(labels[f[colour,start]]),DONE,SEP,SEP]
        else:response=[z for c,u in walk for z in [COL0+c,TO,NODE0+int(labels[u]),SEP]]+[DONE,SEP,SEP,SEP]
        if len(prompt)+len(response)>context:continue
        x=np.full(context,PAD,np.int64);docs=np.full(context,-1,np.int64);mask=np.zeros(context,np.float32)
        x[:len(prompt)+len(response)]=prompt+response;docs[:len(prompt)+len(response)]=0;mask[len(prompt)-1:len(prompt)+len(response)-1]=1
        meta=dict(task=task,nodes=n,prompt_len=len(prompt),response_len=len(response),canonical=canonical(prompt),hard=hard)
        return x,mask,docs,meta
    raise RuntimeError(f'Could not generate {task} n={n} hard={hard}')

def prepare_symbolic():
    seen=set();manifest={}
    suites=[('standard',10,'graph',False,192,'shuffle','random'),('hard10',10,'graph',True,192,'shuffle','random'),('hard14',14,'graph',True,384,'shuffle','random'),('long18',18,'graph',True,384,'shuffle','random'),('lookup',10,'lookup',False,192,'shuffle','random'),('compose',10,'compose',False,192,'shuffle','random'),('pointer',10,'pointer',False,192,'shuffle','random'),('bind',10,'bind',False,192,'shuffle','random'),('shifted',10,'graph',False,192,'shifted','ordered')]
    for split in ('dev','test'):
        for i,(name,n,task,hard,context,layout,labels) in enumerate(suites):
            count=2048 if name in ('standard','lookup') else 512
            rng=np.random.RandomState(88300+1000*(split=='test')+i);rows=[]
            while len(rows)<count:
                row=gen_symbolic(rng,n,task,hard,4,layout,context,labels)
                if row[3]['canonical'] in seen:continue
                seen.add(row[3]['canonical']);rows.append(row)
            np.savez(DATA/f'{split}_{name}.npz',x=np.stack([r[0] for r in rows]),m=np.stack([r[1] for r in rows]),docs=np.stack([r[2] for r in rows]),prompt_len=np.array([r[3]['prompt_len'] for r in rows]))
            manifest[f'{split}_{name}']=dict(n=count,nodes=n,task=task,hard=hard,context=context,layout=layout,label_style=labels,sha256=digest(DATA/f'{split}_{name}.npz'))
    (DATA/'evaluation_symbolic_hashes.txt').write_text('\n'.join(sorted(seen))+'\n');return manifest

class Text:
    def __init__(self,split='train',family='base',context=192):
        self.split=split;self.family=family;self.context=context
        self.edu=np.load(DATA/f'{family}_{split}.npy',mmap_mode='r');self.off=np.load(DATA/f'{family}_{split}_offsets.npy')
        self.ts=np.load(ROOT.parent/'v5_tiny/data/ts_train.npy',mmap_mode='r');self.ts_off=np.load(DATA/'ts_offsets.npy');self.ts_ids=np.load(DATA/f'ts_{split}_ids.npy')
    def row(self,rng,source='edu',single=False):
        t=self.context;x=np.full(t,PAD,np.int64);docs=np.full(t,-1,np.int64);p=0;doc=0;ids=[]
        flat,off=(self.edu,self.off) if source=='edu' else (self.ts,self.ts_off)
        while p<t:
            i=int(rng.randint(len(off)-1)) if source=='edu' else int(rng.choice(self.ts_ids));start,end=map(int,off[i:i+2]);available=end-start
            if available>t-p:start+=K*int(rng.randint(max(1,(available-(t-p))//K+1)))
            n=min(end-start,t-p);x[p:p+n]=flat[start:start+n];docs[p:p+n]=doc;ids.append((self.family if source=='edu' else 'ts',i));p+=K*((n+K-1)//K);doc+=1
            if single:break
        m=np.zeros(t,np.float32);m[:-1]=((docs[:-1]>=0)&(docs[:-1]==docs[1:])).astype(np.float32)
        return x,m,docs,ids

class Mixer:
    def __init__(self,seed,batch=64,mode='foundation',context=192):
        self.rng=np.random.RandomState(seed);self.batch_size=batch;self.mode=mode;self.context=context;self.text=Text(family='continual' if mode=='text' else 'base',context=context)
        self.excluded=set((DATA/'evaluation_symbolic_hashes.txt').read_text().splitlines());self.seen_docs=set();self.seen_symbols=set();self.seen_tasks={};self.targets={};self.slots=0;self.nonpad=0;self.source_targets={'edu':0,'ts':0}
    def batch(self,device='cpu',single_text=False):
        b=self.batch_size;t=self.context
        if self.mode in ('foundation','hard','refit','z'):tasks=['text']*(b//2)+['graph']*(b*3//8)+['lookup']*(b//16)+['compose']*(b//32)+['pointer']*(b//32)
        elif self.mode=='simple':tasks=['lookup']*(b//2)+['compose']*(b//2)
        elif self.mode=='text':tasks=['text']*b
        elif self.mode in ('bind','shifted','lookup'):tasks=[self.mode]*b
        else:raise ValueError(self.mode)
        assert len(tasks)==b
        rows=[];domains=[];metas=[];sources=[]
        for task in tasks:
            if task=='text':
                source='edu' if self.mode=='text' or self.source_targets['edu']<=self.source_targets['ts'] else 'ts'
                x,m,d,ids=self.text.row(self.rng,source,single_text);self.seen_docs.update(ids);self.source_targets[source]+=int(m.sum());dom=0;meta=dict(task='text',prompt_len=0,source=source)
            else:
                n=10
                if self.mode=='hard' and task=='graph':n=int(self.rng.choice([10,14,18],p=[.5,1/3,1/6]))
                while True:
                    x,m,d,meta=gen_symbolic(self.rng,n,'graph' if task=='shifted' else task,False,4,'shifted' if task=='shifted' else 'shuffle',t,'ordered' if task=='shifted' else 'random')
                    if meta['canonical'] not in self.excluded:break
                self.seen_symbols.add(meta['canonical']);self.seen_tasks[task]=self.seen_tasks.get(task,0)+1;dom=1
            rows.append((x,m,d));domains.append(dom);metas.append(meta);sources.append(meta.get('source','symbolic'));self.targets[task]=self.targets.get(task,0)+int(m.sum());self.nonpad+=int((d>=0).sum())
        xs,ms,ds=map(np.stack,zip(*rows));self.slots+=xs.size
        return dict(x=torch.as_tensor(xs,device=device),m=torch.as_tensor(ms,device=device),docs=torch.as_tensor(ds,device=device),domain=torch.tensor(domains,device=device),metas=metas,sources=sources)
    def state(self):return dict(rng=self.rng.get_state(),seen_docs=sorted(self.seen_docs),seen_symbols=sorted(self.seen_symbols),seen_tasks=self.seen_tasks,targets=self.targets,slots=self.slots,nonpad=self.nonpad,source_targets=self.source_targets)
    def restore(self,s):
        self.rng.set_state(s['rng']);self.seen_docs={tuple(v) for v in s['seen_docs']};self.seen_symbols=set(s['seen_symbols'])
        for k in ('seen_tasks','targets','slots','nonpad','source_targets'):setattr(self,k,s[k])

def fixed_text(split,n=128,seed=88200,family='base',source='edu',context=192):
    text=Text(split,family,context);rng=np.random.RandomState(seed);rows=[text.row(rng,source,single=True) for _ in range(n)]
    return {k:torch.tensor(v) for k,v in zip(('x','m','docs'),map(np.stack,zip(*[r[:3] for r in rows])))},[r[3][0] for r in rows]

if __name__=='__main__':
    prepare_text();j=json.loads((DATA/'manifest.json').read_text());j['symbolic']=prepare_symbolic();(DATA/'manifest.json').write_text(json.dumps(j,indent=2));print('V8 data ready',flush=True)
