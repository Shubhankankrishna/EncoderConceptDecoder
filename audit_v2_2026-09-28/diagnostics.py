"""Small CPU-only demonstrations of review findings; never changes training."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import torch
torch.set_num_threads(1)
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'run_m2_25M'; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN/'code'))
from model import build_model
from hrm_encoder import engram_indices
from tokenizers import Tokenizer
out={}

# Score exactly the same positions as evaluate.language(last=16).
torch.manual_seed(11)
torch.set_default_dtype(torch.float64)
m,ec,dc=build_model(d=32,V=64,S=4,N=8,n_heads=4,n_kv=2,head_dim=8,d_ff=64,
    engram_rows=101,engram_dim=4,gr_rank=8,rope_dims=0)
dc.rope_dims=0 # Remove floating point position-shift noise in this reach demonstration.
with torch.no_grad():
    for p in m.parameters(): p.add_(0.2*torch.randn_like(p))
    ids=torch.randint(2,64,(1,128))
    def f(t):
        d=torch.zeros_like(t)
        tok,c,_=m.enc(t,d,engram_indices(t,d,ec))
        return m.dec(tok,torch.zeros_like(c),torch.zeros_like(c),d,m.enc.embed.weight,core_off=True)[0]
    full=f(ids)[:,-17:-1]
    trunc={str(r):float((f(ids[:,-r:])[:,-17:-1]-full).abs().max()) for r in [72,88,128]}
out['core_off_truncation_max_logit_diff']=trunc
out['r72_scored_predictor_positions']=list(range(55,71))
out['r88_scored_predictor_positions']=list(range(71,87))

# The exact ordering written in v2 §6.3 makes a boundary-only learned carry norm untrainable.
x=torch.randn(3,8,requires_grad=True)
proj=torch.nn.Linear(8,8,bias=False)
carry=torch.nn.LayerNorm(8)
z=proj(x);z.square().mean().backward()
s=carry(z).detach()
read=torch.nn.Linear(8,1);read(s).square().mean().backward()
out['boundary_only_carry_norm_weight_grad_is_none']=carry.weight.grad is None

# ByteLevel single-token decoding can count replacement-character bytes rather than source bytes.
tokenizer=Tokenizer.from_file(str(RUN/'data/tokenizer.json'))
stored=np.load(RUN/'data/token_bytes.npy')
bytes_proxy=np.array([len(tokenizer.id_to_token(i)) if i>1 else 0 for i in range(8192)])
out['token_byte_table_differing_ids']=int((stored!=bytes_proxy).sum())
out['language_byte_denominators']={}
for name in ['tinystories','wikitext']:
    tok=np.load(RUN/'data/lang_eval'/f'test_{name}_tok.npy')
    doc=np.load(RUN/'data/lang_eval'/f'test_{name}_doc.npy')
    valid=(doc[:,1:]==doc[:,:-1]) & (tok[:,1:]!=0)
    old=stored.copy();old[1]=1
    new=bytes_proxy.copy();new[1]=1
    oldn=int((old[tok[:,1:]]*valid).sum());newn=int((new[tok[:,1:]]*valid).sum())
    out['language_byte_denominators'][name]=dict(recorded=oldn,bytelevel_symbol_count=newn,
        denominator_relative_excess=oldn/newn-1)

# Locate overlaps between evaluation splits; same unchanged generators can recur across OOD/ID.
by={}
for split in ['val','test','ood']:
    items=torch.load(RUN/'data/eval'/f'{split}.pt',weights_only=True)
    by[split]={hashlib.sha1(np.array(it['ids'],dtype=np.uint16).tobytes()).hexdigest():it['family'] for it in items}
out['eval_overlap_families']={a+'_'+b:[by[a][h] for h in by[a].keys()&by[b].keys()] for a,b in [('val','test'),('val','ood'),('test','ood')]}
(OUT/'diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
