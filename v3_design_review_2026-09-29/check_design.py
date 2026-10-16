"""CPU-only mechanical checks for the proposed v1-compatible v3 architecture.
No training, optimizer updates, or edits to model/run directories.
"""
from pathlib import Path
import sys,json,math,datetime
sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A/'source_snapshot/run_m2_25M/code'))
from model import build_model,param_table,macs_per_token
from hrm_encoder import engram_indices,build_masks
from hrm_decoder import plan_index

with torch.device('meta'):
    full,ec,dc=build_model()
params=param_table(full)
out={'time':datetime.datetime.now().astimezone().isoformat(),'params':params,
     'analytical_macs_per_token':{str(r):macs_per_token(ec,dc,R_mean=r) for r in [1,2,2.5,4,8]},
     'scope':'Mechanics of the retained v1 architecture and proposed masks/normalization of answer loss; no v3 training result.'}
del full
torch.set_default_dtype(torch.float64);torch.manual_seed(7)
m,ec,dc=build_model(d=64,V=128,n_heads=4,n_kv=2,head_dim=16,d_ff=176,engram_rows=31,engram_dim=8,S=8,N=16)
ids=torch.randint(2,128,(2,64));doc=torch.zeros_like(ids);doc[:,32:]=1
ei=engram_indices(ids,doc,ec)
tok,c,_=m.enc(ids,doc,ei);mask=build_masks(doc[:,::4],1)[1]
state=torch.zeros_like(c);states=[]
for r in range(4):
    state=m.core.layer(c,state,doc[:,::4],mask);state.retain_grad();states.append(state)
u=m.core.out_norm(state);_,cp=m.core.head(u)
logits,_=m.dec(tok,u,cp,doc,m.enc.embed.weight)
valid=doc[:,1:]==doc[:,:-1]
target=ids[:,1:].masked_fill(~valid,-100)
loss=F.cross_entropy(logits[:,:-1].reshape(-1,128),target.reshape(-1),ignore_index=-100)
loss.backward()
out['final_ntp_gradient_norm_by_state']=[s.grad.norm().item() for s in states]
out['initial_state_zero']=True
out['first_pass_is_nonzero']=bool(states[0].detach().abs().max()>0)
out['all_passes_receive_final_ntp']=all(x>0 and math.isfinite(x) for x in out['final_ntp_gradient_norm_by_state'])

@torch.no_grad()
def f(ids,doc):
    tok,c,_=m.enc(ids,doc,engram_indices(ids,doc,ec));u,lc,cp=m.core(c,doc[:,::4],4)
    return m.dec(tok,u,cp,doc,m.enc.embed.weight)[0]
m.eval();base=f(ids,doc)
mut=ids.clone();mut[0,17]=(mut[0,17]+1)%126+2
changed=f(mut,doc)
out['future_edit_max_change_to_earlier_logits']=(changed[0,:17]-base[0,:17]).abs().max().item()
mut2=ids.clone();mut2[0,:32]=((mut2[0,:32]+3)%126)+2
changed2=f(mut2,doc)
out['other_document_max_change']=(changed2[0,32:]-base[0,32:]).abs().max().item()
out['other_row_max_change']=(changed2[1]-base[1]).abs().max().item()
rev=f(ids.flip(0),doc.flip(0)).flip(0)
out['batch_permutation_max_change']=(rev-base).abs().max().item()
idx,ok,_=plan_index(doc)
t=torch.arange(ids.shape[1]).expand_as(ids)
out['plan_uses_only_completed_chunks']=bool(((4*idx+3<=t)|~ok).all())
out['null_at_each_document_first_three_positions']=bool((~ok[:,0:3]).all() and (~ok[:,32:35]).all())

# Proposed real/full-chunk masks. EOS=1 is real; PAD=0 is not.
test_ids=torch.tensor([[9,8,7,1,6,5,4,1,6,5,1,0,0,0,0,0,3,2,4,1]])
test_doc=torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2]])
chunks=test_ids.view(1,5,4);nonempty=(chunks!=0).any(-1);complete=(chunks!=0).all(-1)
cdoc=test_doc[:,::4]
ncp_mask=(cdoc[:,1:]==cdoc[:,:-1]) & nonempty[:,:-1] & complete[:,1:]
pad_logits=torch.randn(1,4,8,16,requires_grad=True)
pad_mask=torch.zeros(1,4,dtype=torch.bool)
pad_ce=F.cross_entropy(pad_logits.reshape(-1,16),torch.zeros(32,dtype=torch.long),reduction='none').view(1,4,8).mean(-1)
pad_loss=(pad_ce*pad_mask).sum()/pad_mask.sum().clamp_min(1)
pad_grad=torch.autograd.grad(pad_loss,pad_logits)[0]
out['mask_examples']={'complete_chunks':complete.tolist(),'nonempty_chunks':nonempty.tolist(),
                      'ncp_pairs':ncp_mask.tolist(),'all_pad_loss':pad_loss.item(),
                      'all_pad_gradient_max':pad_grad.abs().max().item()}

# Verify the document-balanced answer term and microbatch decomposition on arbitrary CE values.
# Two synthetic docs: doc A has two answer spans, doc B one. A long language doc has no spans.
ce=torch.tensor([1.,3.,5.,7.,9.,11.,13.,15.],requires_grad=True)
spans=[[(0,2),(2,3)],[(3,6)]]
def ans(x):return torch.stack([torch.stack([x[a:b].mean() for a,b in qs]).mean() for qs in spans]).mean()
la=ans(ce);ga=torch.autograd.grad(la,ce)[0]
micro1=(ce[0:2].mean()+ce[2:3].mean())/2/2
micro2=ce[3:6].mean()/2
gm=torch.autograd.grad(micro1+micro2,ce)[0]
out['answer_loss']={'value':la.item(),'per_token_gradient':ga.tolist(),
                    'accumulation_max_difference':(ga-gm).abs().max().item(),
                    'no_answer_tokens_have_zero_extra_weight':bool((ga[6:]==0).all())}
checks=[params['compute_params']==25052781,params['stored_total']==78578925,
        out['first_pass_is_nonzero'],out['all_passes_receive_final_ntp'],out['plan_uses_only_completed_chunks'],
        out['null_at_each_document_first_three_positions'],
        all(out[k]<1e-10 for k in ['future_edit_max_change_to_earlier_logits','other_document_max_change','other_row_max_change','batch_permutation_max_change']),
        out['answer_loss']['accumulation_max_difference']==0,out['answer_loss']['no_answer_tokens_have_zero_extra_weight'],
        ncp_mask.tolist()==[[True,False,False,False]],pad_loss.item()==0,pad_grad.abs().max().item()==0]
out['passed']=all(checks)
(A/'design_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
assert out['passed']
