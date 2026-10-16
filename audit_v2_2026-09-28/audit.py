"""Read-only inspection of run_m2_25M; writes findings only in this audit directory."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, datetime, hashlib, json, math, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUN = ROOT / 'run_m2_25M'
result = {'snapshot_time': datetime.datetime.now().astimezone().isoformat()}

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))

# Inventory every file, keeping first-party analysis distinct from dependency/cache bytes.
inventory, contents = [], {}
for f in sorted(RUN.rglob('*')):
    if not f.is_file():
        continue
    rel = str(f.relative_to(RUN))
    st = f.stat()
    excluded = any(x in f.parts for x in ('.venv', '__pycache__', 'vendor')) or f.name == '.DS_Store'
    ent = dict(path=rel, bytes=st.st_size, inode=st.st_ino, mtime=st.st_mtime,
               category='dependency/cache/OS' if excluded else 'first-party')
    if not excluded and f.suffix not in ('.pt', '.npy', '.png'):
        raw = f.read_bytes()
        ent['sha256'] = hashlib.sha256(raw).hexdigest()
        try:
            s = raw.decode()
            contents[rel] = s
            ent['lines'] = len(s.splitlines())
            if f.suffix == '.json':
                obj = json.loads(s)
                def count(x):
                    if isinstance(x, dict): return sum(count(v) for v in x.values())
                    if isinstance(x, list): return sum(count(v) for v in x)
                    return int(isinstance(x, (float, int)) and not isinstance(x, bool))
                ent['numeric_leaves'] = count(obj)
            elif f.suffix == '.jsonl':
                rows = [json.loads(l) for l in s.splitlines() if l.strip()]
                ent['records'] = len(rows)
            elif f.suffix == '.py':
                tree = ast.parse(s)
                ent['functions'] = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        except (UnicodeDecodeError, json.JSONDecodeError, SyntaxError) as e:
            ent['read_error'] = str(e)
    inventory.append(ent)
dump('inventory.json', inventory)
result['inventory'] = dict(files=len(inventory), first_party=sum(x['category']=='first-party' for x in inventory),
    readable_text_files=len(contents), text_lines=sum(x.get('lines',0) for x in inventory),
    json_numeric_leaves=sum(x.get('numeric_leaves',0) for x in inventory),
    jsonl_records=sum(x.get('records',0) for x in inventory),
    by_directory=dict(Counter(x['path'].split('/')[0] for x in inventory)))

# All arrays: shapes and complete streaming checks, including every training row.
arrays = {}
for f in sorted((RUN/'data').rglob('*.npy')):
    a = np.load(f, mmap_mode='r')
    arrays[str(f.relative_to(RUN))] = dict(shape=list(a.shape), dtype=str(a.dtype), count=int(a.size))
result['arrays'] = arrays
manifest = json.loads(contents['data/MANIFEST.json'])
data_checks = {}
for split in ('train','val','test','maintenance'):
    tok = np.load(RUN/'data'/split/'tok.npy', mmap_mode='r')
    doc = np.load(RUN/'data'/split/'doc.npy', mmap_mode='r')
    h = hashlib.sha256()
    nonpad = ntp = ncp = pad_ncp = partially_pad_ncp = spans = bad_starts = bad_lengths = 0
    lo, hi = 999999, -1
    for s in range(0,len(tok),512):
        t, d = tok[s:s+512], doc[s:s+512]
        h.update(t.tobytes())
        nonpad += int(np.count_nonzero(t))
        ntp += int(np.count_nonzero((d[:,1:]==d[:,:-1]) & (t[:,1:]!=0)))
        ok = d[:,4::4]==d[:,:-4:4]
        ch = t.reshape(len(t),-1,4)
        ncp += int(ok.sum())
        pad_ncp += int((ok & (ch[:,1:]==0).all(-1)).sum())
        partially_pad_ncp += int((ok & (ch[:,1:]==0).any(-1)).sum())
        lo,hi = min(lo,int(t.min())),max(hi,int(t.max()))
        # Because row boundaries are aligned, verifying every change verifies every span too.
        positions = np.nonzero(d[:,1:] != d[:,:-1])[1]+1
        bad_starts += int((positions%4!=0).sum())
        spans += len(t)+len(positions)
    data_checks[split] = dict(rows=len(tok), nonpad=nonpad, ntp_targets=ntp, ncp_targets=ncp,
        all_pad_ncp_targets=pad_ncp, any_pad_ncp_targets=partially_pad_ncp,
        all_pad_ncp_fraction=pad_ncp/max(ncp,1), bad_chunk_boundary_starts=bad_starts,
        doc_spans=spans, token_min=lo, token_max=hi,
        sha256=h.hexdigest(), matches_manifest=h.hexdigest()==manifest['splits'][split]['tok_sha256'])
    print('array check',split,data_checks[split],flush=True)
result['data_checks']=data_checks
result['tokenizer_sha256_match'] = hashlib.sha256((RUN/'data/tokenizer.json').read_bytes()).hexdigest()==manifest['tokenizer_sha256']

# Verify exact disjointness for every scored item against sorted train hashes.
train_hash=np.load(RUN/'data/train/synthetic_doc_sha1.npy',mmap_mode='r')
item_sets, item_hashes, item_checks = {},{},{}
for split in ('val','test','ood'):
    its=torch.load(RUN/'data/eval'/f'{split}.pt',map_location='cpu',weights_only=True)
    item_sets[split]=its
    hashes=np.array([hashlib.sha1(np.asarray(it['ids'],dtype=np.uint16).tobytes()).hexdigest() for it in its])
    loc=np.searchsorted(train_hash,hashes)
    overlap=int(((loc<len(train_hash)) & (train_hash[np.minimum(loc,len(train_hash)-1)]==hashes)).sum())
    item_hashes[split]=set(hashes.tolist())
    by={}
    for fam in sorted({it['family'] for it in its}):
        fs=[it for it in its if it['family']==fam]
        ds=[it['dist'] for it in fs if 'dist' in it]
        by[fam]=dict(n=len(fs), tokens_min=min(len(it['ids']) for it in fs),tokens_max=max(len(it['ids']) for it in fs),
            spans=sorted({len(it['spans']) for it in fs}),answer_tokens=sorted({b-a for it in fs for a,b,_ in it['spans']}),
            min_evidence_distance=min(ds) if ds else None,
            invalid_span_count=sum(not (0<=a<b<=len(it['ids'])) for it in fs for a,b,_ in it['spans']))
    item_checks[split]=dict(n=len(its),unique=len(set(hashes)),overlap_with_train=overlap,families=by)
result['item_checks']=item_checks
result['eval_split_intersections']={a+'_'+b:len(item_hashes[a]&item_hashes[b]) for a,b in [('val','test'),('test','ood'),('val','ood')]}

# Checkpoint metadata only: meta tensors avoid materializing multi-GB optimizer state.
ckmeta, inode_cache = [], {}
for f in sorted((RUN/'runs').rglob('*.pt')):
    inode=f.stat().st_ino
    rel=str(f.relative_to(RUN))
    if inode in inode_cache:
        ckmeta.append(dict(path=rel,hardlink_to=inode_cache[inode]))
        continue
    ck=torch.load(f,map_location='meta',weights_only=False,mmap=True)
    state=ck['model']
    shapes={k:list(v.shape) for k,v in state.items()}
    ent=dict(path=rel,step=ck.get('step'),cursor=ck.get('cursor'),elapsed=ck.get('elapsed'),
             best=ck.get('best'),args=ck.get('args'),tensor_count=len(state),
             embedding_shape=shapes.get('enc.embed.weight'),expert_tensor_keys=[k for k in state if 'expert' in k],
             state_shapes=shapes)
    ckmeta.append(ent);inode_cache[inode]=rel
    del ck
result['checkpoint_count']=len(ckmeta)
result['unique_checkpoint_inodes']=len(inode_cache)
dump('checkpoints.json',ckmeta)

# Recompute the full published architecture parameter budget on meta device.
sys.path.insert(0,str(RUN/'model_v2'))
import v2_budget
budget=json.loads(contents['model_v2/v2_budget.json'])
recount={}
with torch.device('meta'):
    for key,row in budget.items():
        cfg=row['cfg']
        b=v2_budget.build(cfg['enc_d_ff'],cfg['dec_d_ff'],cfg['E'],cfg['nl_hidden'],cfg['lam_kl'])
        recount[key]=dict(params_match=b['params']==row['params'],params=b['params'],macs=b['macs'])
result['budget_recount']=recount
c=budget['C_E16']
extra=(1.25-1)*2*3*256*512*4*10/4
result['moe_capacity_arithmetic']=dict(assignments_per_layer=2*256*2,capacity=80,allocated_slots=16*80,
        ideal_expert_forward_macs_token=2*3*256*512*4*10/4,
        padding_extra_fwd_macs_token=extra,reported_train_macs_equiv=c['macs']['train_equiv'],
        padded_train_macs_equiv=c['macs']['train_equiv']+3*extra)

# Complete logs, no trajectory cherry-picking.
runs={}
for name in ('main','cont','screens/S1','screens/S2','screens/S3'):
    evals=[json.loads(s) for s in contents[f'runs/{name}/log.jsonl'].splitlines() if s]
    tr=[json.loads(s) for s in contents[f'runs/{name}/train_log.jsonl'].splitlines() if s]
    metrics=[x for x in tr if 'ntp' in x]
    runs[name]=dict(evals=evals,train_records=len(tr),metric_records=len(metrics),
        throughput_range=[min(x['tok_s'] for x in metrics),max(x['tok_s'] for x in metrics)],
        throughput_median=float(np.median([x['tok_s'] for x in metrics])),
        max_logged_grad=max(x['gn_preclip'] for x in metrics),
        guards=[json.loads(s) for s in contents[f'runs/{name}/GUARDS.jsonl'].splitlines() if s],
        last_train=metrics[-1])
result['runs']=runs
final=json.loads(contents['runs/main/eval_final.json'])
result['final_eval_sections']=list(final)
result['final_eval_numeric_consistency']={}
for split,r in final['items'].items():
    vals=[v['exact'] for k,v in r['intact'].items() if isinstance(v,dict) and not k.startswith('lr_')]
    result['final_eval_numeric_consistency'][split]=dict(macro_recomputed=float(np.mean(vals)),
        macro_recorded=r['intact']['_macro_synthetic'],families=len(vals))

# Keep the exact immutable result and live evaluation snapshot alongside the report.
dump('main_final_snapshot.json',final)
dump('audit_results.json',result)
print('Audit completed',json.dumps(result['inventory']),flush=True)
print('Checkpoint metadata',result['checkpoint_count'],result['unique_checkpoint_inodes'],flush=True)
print('Eval intersections',result['eval_split_intersections'],flush=True)
print('Budget recount',{k:v['params_match'] for k,v in recount.items()},flush=True)
