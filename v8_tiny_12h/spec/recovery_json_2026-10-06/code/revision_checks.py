"""Current-revision content isolation and production cache checks, not quality scores."""
import argparse,copy,json
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import Incremental,FastIncremental

def run(source=None):
    torch.set_num_threads(4);seed_all(899);tests=[]
    def check(name,value,**details):
        if not value:raise AssertionError(name+' '+str(details))
        tests.append(dict(name=name,passed=True,**details))
    sets={split:set((DATA/f'ts_{split}_hashes.txt').read_text().splitlines()) for split in ('train','dev','test')}
    for a,b in (('train','dev'),('train','test'),('dev','test')):check('content-disjoint TinyStories '+a+'/'+b,not sets[a]&sets[b])
    manifest=json.loads((DATA/'manifest.json').read_text())
    check('external TinyStories corpus integrity',digest(ROOT.parent/'v5_tiny/data/ts_train.npy')==manifest['tiny_stories']['sha256'])
    for name,meta in manifest['text'].items():check('text array integrity '+name,digest(DATA/f'{name}.npy')==meta['sha256'])
    for name,meta in manifest['symbolic'].items():check('symbolic array integrity '+name,digest(DATA/f'{name}.npz')==meta['sha256'])
    for split,meta in manifest['tiny_stories']['partitions'].items():
        check('TinyStories ID integrity '+split,digest(DATA/f'ts_{split}_ids.npy')==meta['ids_sha256'])
        check('TinyStories content integrity '+split,digest(DATA/f'ts_{split}_hashes.txt')==meta['content_hashes_sha256'])
    for stage in ('MEM','SIMPLE','A','HARD'):
        s=System();opt=optimizer(s,stage=stage);group=next(g for g in opt.param_groups if any(p is s.foundation.Q for p in g['params']))
        check('consistent Q optimizer '+stage,group['lr_mult']==10 and group['weight_decay']==0)
    s=load_system(source)[0] if source else System().to(DEVICE);freeze(s)
    variants=[('actual_version',s)]
    fixture=copy.deepcopy(s)
    if not fixture.is_flat and fixture.adapters is None:fixture.adapters=Adapters().to(DEVICE)
    fixture.add_phase(force=None)
    for block in fixture.active_blocks():
        block.banks[-1].Bq.data.normal_(std=.01);block.banks[-1].Bv.data.normal_(std=.01)
        block.banks[-1].pos_var.fill_(10.);block.neg_var.fill_(100.)
    variants.append(('nonzero-gate-fixture',fixture))
    for label,model in variants:
        for nodes,p in ((14,172),(18,220)):
            row=gen_symbolic(np.random.RandomState(98000+nodes),nodes,hard=True,context=384);x=torch.tensor(row[0][None],device=DEVICE)
            for nonzero in (False,True) if model.adapters else (False,):
                z=ball(torch.randn(1,96,DZ,device=DEVICE)*.3) if nonzero else torch.zeros(1,96,DZ,device=DEVICE)
                for kind in ('dynamic','static') if not isinstance(model.core,Core) else ('dynamic',):
                    cache=(FastIncremental if kind=='static' else Incremental)(model,z)
                    with torch.no_grad():
                        inc=cache.prefill(x[:,:p]);full=model(x[:,:p],torch.zeros_like(x[:,:p]),z[:,:p//K])[:,-1]
                        check(f'{label}/{nodes}/{kind}/prefill/{nonzero}',torch.allclose(inc,full,atol=4e-4,rtol=1e-5),error=float((inc-full).abs().max()))
                        for pos in range(p,p+8):
                            inc=cache.step(x[:,pos]);full=model(x[:,:pos+1],torch.zeros_like(x[:,:pos+1]),z[:,: (pos+1)//K])[:,-1]
                            error=float((inc-full).abs().max());margin=full.topk(2).values.diff(dim=-1).abs().min().item()
                            check(f'{label}/{nodes}/{kind}/step{pos}/{nonzero}',torch.allclose(inc,full,atol=4e-4,rtol=1e-5),error=error,argmax_agreement=bool((inc.argmax(-1)==full.argmax(-1)).all()),minimum_reference_margin=margin)
    return dict(status='passed',source=source,device=DEVICE,implementation_sha256=implementation_hash(),checks=tests,scope='Execution/content contracts; synthetic bank is explicitly a fixture, not trained quality evidence')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source');a.add_argument('--out',required=True);args=a.parse_args()
    try:j=run(args.source);atomic_json(args.out,j);print(json.dumps({k:v for k,v in j.items() if k!='checks'}))
    except Exception as exc:atomic_json(args.out,dict(status='failed',error=repr(exc),implementation_sha256=implementation_hash()));raise
