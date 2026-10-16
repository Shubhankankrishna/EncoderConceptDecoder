"""Specification checks only. Does not implement/train V10.1 or touch active state."""
import hashlib
import json
import re
from pathlib import Path
import torch
from torch import nn
from v10_10M.config import ModelConfig
from v10_10M.model import V10, Block, RMSNorm, Decoder, geometry

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(1011)
result = {'scope': 'Specification arithmetic, interface and toy mathematics; no end-to-end V10.1 implementation validation'}
count = lambda module: sum(p.numel() for p in module.parameters())
c = ModelConfig(); base = V10(c)
bridge = nn.ModuleDict({
    'source_norm': nn.ModuleList([RMSNorm(128), RMSNorm(128)]),
    'source_proj': nn.ModuleList([nn.Linear(128,128,bias=False), nn.Linear(128,128,bias=False)]),
    'router_norm': RMSNorm(128), 'router': nn.Linear(128,2),
    'scale': nn.ParameterList([nn.Parameter(torch.full((128,),.1))])})
projection = nn.Linear(128,128,bias=False)
new_routes = nn.ModuleList([nn.Linear(128,3) for _ in range(4)])
aux = nn.ModuleDict({'block': Block(c), 'project': nn.Linear(128,128,bias=False),
                     'position': nn.Embedding(8,128), 'norm': RMSNorm(128),
                     'start': nn.ParameterList([nn.Parameter(torch.zeros(128))])})
b = base.counts()['dense']; ec = count(bridge)
cd = count(projection) + count(new_routes) - count(base.decoder.routes)
a = count(aux)
result['parameters'] = dict(base_dense=b, encoder_concept_crc=ec, concept_decoder_crc=cd,
                            crc_dense=b+ec+cd, auxiliary=a, full_dense=b+ec+cd+a,
                            engram=base.counts()['engram'], total=b+ec+cd+a+base.counts()['engram'])
assert result['parameters'] == dict(base_dense=5465890,encoder_concept_crc=33538,
    concept_decoder_crc=16900,crc_dense=5516328,auxiliary=215168,full_dense=5731496,
    engram=20000320,total=25731816)
del base

# Exact integer release test against the existing alignment convention.
checks = 0
for t in range(1,66):
    groups=t//8
    source=torch.arange(1,groups*2+1,dtype=torch.float32).reshape(1,groups,2,1)
    aligned=Decoder.align(source,t).flatten()
    for i in range(t):
        expected=0 if i<7 else ((i-7)//8)*2 + ((i-7)%8)//4 + 1
        assert aligned[i].item()==expected
        if i>=7:
            assert 8*((i-7)//8)+7 <= i
        checks+=1
result['causal_release_rows_checked'] = checks

# Mask unavailable continuous sources BEFORE softmax, not after.
logits=torch.tensor([[.2,-.1,-2.]])
masked=logits.clone(); masked[:,-1]=-torch.inf
assert torch.equal(masked.softmax(-1)[:,:2],logits[:,:2].softmax(-1))
result['startup_router_preserves_encoder_mixture'] = True

# Two concept slots in one completed group can see each other, never future groups.
mask,_=geometry(8,torch.tensor([8]),'cpu',32,group_slots=2)
for i in range(8):
    for j in range(8): assert bool(mask[0,0,i,j]) == (j//2<=i//2)
result['group_causal_mask_checked'] = True

# A real toy auxiliary block: perturb a target token, which can enter only the NEXT input.
aux.eval(); embedding=nn.Embedding(32,128); tokens=torch.arange(8)
forecast=torch.randn(2,128)
mask,rope=geometry(8,torch.tensor([8]),'cpu',32)
def auxiliary_hidden(target_tokens):
    prev=torch.cat((aux['start'][0][None],embedding(target_tokens[:-1])),0)
    x=prev+aux['position'].weight+aux['project'](forecast.repeat_interleave(4,0))
    return aux['norm'](aux['block'](x[None],mask,rope))[0]
with torch.no_grad():
    original=auxiliary_hidden(tokens)
    for j in range(8):
        changed=tokens.clone(); changed[j]=(changed[j]+10)%32
        other=auxiliary_hidden(changed)
        assert torch.allclose(other[:j+1],original[:j+1],atol=1e-7,rtol=1e-7)
result['auxiliary_current_and_future_token_perturbations'] = 8

# Partial target chunks retain terminal/short-answer supervision independently of latent pairs.
length=10  # 10 input rows; token 10 is EOS target for row 9.
group=0; first=8*(group+1)
target_mask=torch.zeros(length,dtype=torch.bool); target_mask[8:]=True
positions=[first+j for j in range(8) if first+j-1<length and target_mask[first+j-1]]
assert positions==[9,10]
assert length//8==1  # no complete following input group, hence no latent pair
result['partial_chunk_eos_target_indices'] = positions

# Finite differences through BOTH nonlinear downstream routes sharing the same correction.
torch.set_default_dtype(torch.float64)
u=torch.randn(4,8); r=(torch.randn(4,8)*.01).requires_grad_()
pq_logits=torch.randn(8,5); codebook=torch.randn(5,3); direct=torch.randn(8,3)
target=torch.randn(4,3)
def loss(value):
    z=u+value
    forecast=(z@pq_logits).softmax(-1)@codebook
    merged=forecast+z@direct
    return (torch.tanh(merged)-target).square().mean()+.1*forecast.square().mean()
gradient=torch.autograd.grad(loss(r),r)[0]
finite=torch.zeros_like(r); eps=1e-5
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        delta=torch.zeros_like(r);delta[i,j]=eps
        finite[i,j]=(loss(r.detach()+delta)-loss(r.detach()-delta))/(2*eps)
error=float((gradient-finite).abs().max())
assert error<1e-8
q=u/u.shape[-1]**.5; codes=torch.tensor([0,0,1,1])
kernel=torch.exp(-torch.cdist(q,q).square()/(2*.5**2))*(codes[:,None]==codes[None,:])
mineigen=float(torch.linalg.eigvalsh(kernel).min())
direction=-(kernel@gradient)
directional=float((gradient*direction).sum())
eta=1e-4; before=float(loss(r.detach()));after=float(loss(r.detach()+eta*direction))
assert mineigen>=-1e-10 and directional<0 and after<before
result['two_route_memory_toy']=dict(gradient_max_abs_error=error,kernel_min_eigenvalue=mineigen,
    first_order_directional_derivative=directional,eta=eta,before=before,after=after)

# Source and manifest must stay unchanged. No checkpoint hash: the live trainer advances it.
prior=json.loads((DEST/'data_evidence.json').read_text())
h=hashlib.sha256()
for path in sorted((ROOT/'v10_10M').glob('*.py')):
    h.update(path.name.encode());h.update(path.read_bytes())
assert h.hexdigest()==prior['active_source_sha256']
manifest=ROOT/'v10_10M/data/manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==prior['manifest_sha256']
result['active_source_and_manifest_hashes_unchanged']=True

doc=ROOT/'V10_1_RCDL_Architecture.md'; text=doc.read_text()
missing=[]
for target in re.findall(r'\]\((<[^>]+>|[^\)]+)\)',text):
    target=target.strip('<>')
    if target.startswith(('https://','http://','#')):continue
    if not (doc.parent/target).exists(): missing.append(target)
assert not missing,missing
assert text.count('```')%2==0
result['markdown_local_links_and_fences_checked']=True
result['document_sha256']=hashlib.sha256(doc.read_bytes()).hexdigest()
(DEST/'review_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
