"""Document-aware TinyStories packing and independent symbolic streams."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import K, VOCAB
from graphnav import (PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_NODE_TOK,
                      T, WALK_START, MAX_HOPS, gen_instance, gen_hardgoal, parse_header)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'v5_tiny' / 'data'
DATA = ROOT / 'data'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    train = np.load(SOURCE/'ts_train.npy', mmap_mode='r')
    val = np.load(SOURCE/'ts_val.npy', mmap_mode='r')
    offsets = {}
    for name, flat in [('train',train),('val',val)]:
        ends = np.flatnonzero(flat == EOS).astype(np.int64) + 1
        offsets[name] = np.concatenate([np.array([0]),ends])
        np.save(DATA/f'{name}_offsets.npy',offsets[name])
    # Token-content hashes detect exact duplicated documents even across source splits.
    def hashes(flat, off):
        return [hashlib.blake2b(flat[off[i]:off[i+1]].tobytes(),digest_size=16).digest() for i in range(len(off)-1)]
    tr_hashes = hashes(train,offsets['train']); seen = set(tr_hashes)
    va_hashes = hashes(val,offsets['val']); unique=[]; duplicate=0; val_seen=set()
    for i,h in enumerate(va_hashes):
        if h in seen or h in val_seen:
            duplicate+=1; continue
        val_seen.add(h); unique.append(i)
    rng=np.random.RandomState(60401); unique=np.asarray(unique,np.int64); rng.shuffle(unique)
    mid=len(unique)//2
    np.save(DATA/'dev_docs.npy',unique[:mid]); np.save(DATA/'test_docs.npy',unique[mid:])
    manifest=dict(source='Local TinyStories, V5 tokenizer; source files remain read-only',
                  train_tokens=len(train),val_tokens=len(val),train_documents=len(tr_hashes),
                  dev_documents=mid,test_documents=len(unique)-mid,removed_eval_duplicates=duplicate,
                  split_seed=60401, hashes={p.name:digest(p) for p in [SOURCE/'ts_train.npy',SOURCE/'ts_val.npy',SOURCE/'tokenizer.json']},
                  inference='New held-out document allocation inside the previously inspected V5 validation corpus; not independent new-corpus confirmation',
                  near_duplicates='Exact token-document deduplication only; semantic near-duplicates remain unmeasured')
    seen=set()
    for split in ('dev','test'):
        for name,hard,n in [('standard',False,256),('hard',True,512)]:
            seed=61000+(1000 if split=='test' else 0)+(10 if hard else 0)
            xs,ms,metas,new_seen=fixed_graph(n,seed,hard,seen);seen=new_seen
            np.savez(DATA/f'{split}_{name}_s0.npz',x=xs,m=ms)
    (DATA/'evaluation_graph_hashes.txt').write_text('\n'.join(sorted(seen))+'\n')
    manifest['symbolic_evaluation']=dict(dev_standard=256,dev_hard=512,test_standard=256,test_hard=512,
                                        unique_canonical_instances=len(seen),training_exclusion=True,
                                        fingerprints_sha256=digest(DATA/'evaluation_graph_hashes.txt'))
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


class Text:
    def __init__(self,split):
        source='train' if split=='train' else 'val'
        self.flat=np.load(SOURCE/f'ts_{source}.npy',mmap_mode='r')
        self.offsets=np.load(DATA/f'{source}_offsets.npy')
        self.ids=np.arange(len(self.offsets)-1) if split=='train' else np.load(DATA/f'{split}_docs.npy')

    def row(self,rng):
        x=np.full(T,PAD,np.int64); docs=np.full(T,-1,np.int64); p=0; doc=0; identities=[]
        while p<T:
            i=int(self.ids[rng.randint(len(self.ids))]); start,end=self.offsets[i:i+2]
            available=int(end-start)
            if available>T-p:
                # A long document is a chunk-aligned crop; the next target outside the row is unscored.
                skip=4*rng.randint(max(1,(available-(T-p))//4+1)); start+=skip
            n=min(int(end-start),T-p)
            x[p:p+n]=self.flat[start:start+n]; docs[p:p+n]=doc
            identities.append(i); p+=4*((n+3)//4); doc+=1
        m=np.zeros(T,np.float32)
        m[:-1]=((docs[:-1]>=0)&(docs[:-1]==docs[1:])).astype(np.float32)
        return x,m,docs,identities


class Mixer:
    def __init__(self,seed,split='train',batch=64):
        assert batch%2==0
        self.text=Text(split); self.rng=np.random.RandomState(seed); self.batch_size=batch
        self.seen_docs=set(); self.seen_graphs=set(); self.graphs=0
        self.excluded_graphs=set((DATA/'evaluation_graph_hashes.txt').read_text().splitlines()) if (DATA/'evaluation_graph_hashes.txt').exists() else set()

    def batch(self,device='cpu'):
        xs,ms,ds,metas=[],[],[],[]
        for _ in range(self.batch_size//2):
            x,m,d,ids=self.text.row(self.rng); xs.append(x);ms.append(m);ds.append(d);self.seen_docs.update(ids)
        for _ in range(self.batch_size//2):
            while True:
                x,m,meta=gen_instance(self.rng);identity=canonical(x)
                if identity not in self.excluded_graphs:break
            d=np.where(x!=PAD,0,-1).astype(np.int64)
            xs.append(x);ms.append(m);ds.append(d);metas.append(meta);self.graphs+=1;self.seen_graphs.add(identity)
        return dict(x=torch.tensor(np.stack(xs),device=device),m=torch.tensor(np.stack(ms),device=device),
                    docs=torch.tensor(np.stack(ds),device=device),domain=torch.tensor([0]*(self.batch_size//2)+[1]*(self.batch_size//2),device=device),metas=metas)

    def state(self):
        return dict(rng=self.rng.get_state(),seen_docs=sorted(self.seen_docs),seen_graphs=sorted(self.seen_graphs),graphs=self.graphs)

    def restore(self,state):
        self.rng.set_state(state['rng']);self.seen_docs=set(state['seen_docs']);self.seen_graphs=set(state.get('seen_graphs',[]));self.graphs=state['graphs']


def fixed_text(n,seed,split='dev'):
    stream=Text(split);rng=np.random.RandomState(seed)
    rows=[stream.row(rng) for _ in range(n)]
    return np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows]),np.stack([r[2] for r in rows])


def canonical(x):
    edges=np.asarray(x[4:WALK_START],dtype=np.int64).reshape(-1,4)
    order=np.argsort(edges[:,0]*VOCAB+edges[:,1])
    return hashlib.sha256(np.asarray(x[1:3],dtype=np.int64).tobytes()+edges[order,:3].tobytes()).hexdigest()


def fixed_graph(n,seed,hard=False,excluded=()):
    rng=np.random.RandomState(seed); xs=[];ms=[]; metas=[]; seen=set(excluded)
    while len(xs)<n:
        if hard:
            x,meta=gen_hardgoal(rng);m=np.zeros(T,np.float32)
        else:
            x,m,meta=gen_instance(rng)
        h=canonical(x)
        if h in seen:continue
        seen.add(h);xs.append(x);ms.append(m);metas.append(meta)
    return np.stack(xs),np.stack(ms),metas,seen


def verify(header,response):
    """Strict full-chunk terminator check. Never uses hidden trajectory labels."""
    edges,cur,goal=parse_header(header); reached=False; trajectory=[]
    for j in range(MAX_HOPS+1):
        chunk=list(map(int,response[4*j:4*j+4]))
        if len(chunk)!=4:return dict(success=False,format_ok=False,reached=reached,reason='missing_terminator',trajectory=trajectory)
        if chunk[0]==DONE:
            if chunk!=[DONE,SEP,SEP,SEP]:reason='malformed_terminator'
            elif not trajectory or cur!=goal:reason='wrong_goal'
            else:reason='accepted'
            return dict(success=reason=='accepted',format_ok=reason in ('accepted','wrong_goal'),reached=reached,reason=reason,trajectory=trajectory)
        c,to,u,sep=chunk
        if not (COL0<=c<COL0+N_COL and to==TO and NODE0<=u<NODE0+N_NODE_TOK and sep==SEP):
            return dict(success=False,format_ok=False,reached=reached,reason='malformed_walk',trajectory=trajectory)
        if edges.get((cur,c-COL0))!=u-NODE0:
            return dict(success=False,format_ok=True,reached=reached,reason='illegal_transition',trajectory=trajectory)
        cur=u-NODE0;trajectory.append((c-COL0,cur));reached |= cur==goal
    return dict(success=False,format_ok=True,reached=reached,reason='missing_terminator',trajectory=trajectory)


def oracle(header):
    edges,start,goal=parse_header(header);queue=[(start,[])];seen={start}
    for cur,path in queue:
        if cur==goal and path:
            return np.array([t for c,u in path for t in (COL0+c,TO,NODE0+u,SEP)]+[DONE,SEP,SEP,SEP],np.int64)
        for c in range(N_COL):
            u=edges[(cur,c)]
            if u not in seen:seen.add(u);queue.append((u,path+[(c,u)]))
    raise ValueError('Unreachable goal')
