import os,sys,json,math,copy
from pathlib import Path
import torch
import torch.nn.functional as F
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'v3_design'))
from v3_ref import HRMThinkerV3, param_table, SCHED
from hrm_encoder import engram_indices,rms
from check_engram2 import reach
torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(812)
out={}
def model():return HRMThinkerV3(d=64,V=64,enc_ff=128,dec_ff=128,core_ff=96,shared_ff=128,engram_rows=31,small=True)
m=model();ids=torch.randint(2,64,(2,128));doc=torch.zeros_like(ids);doc[1,64:]=1;ei=engram_indices(ids,doc,m.ecfg)
# Count actual calls to the shared embedding table.
count=[0]
def hook(mod,inp,res):count[0]+=1
h=m.enc.engram.table.register_forward_hook(hook)
tok,C,_=m.enc(ids,doc,ei);h.remove();out['engram_gathers_per_encoder_forward']=count[0]
C.retain_grad();S,st=m.core.trajectory(C,doc[:,::4],'3',return_states=True)
for x in st:x.retain_grad()
u=m.core.out_norm(S);_,cp=m.core.head(u);logits,_=m.dec(tok,u,cp,doc,m.enc.embed.weight)
loss=F.cross_entropy(logits[:,:-1].reshape(-1,64),ids[:,1:].reshape(-1));loss.backward()
out['true_init_ntp_grad_norms']={'C':C.grad.norm().item(),**{f'S{i+1}':s.grad.norm().item() for i,s in enumerate(st)}}
out['true_init_parameter_grads']={n:(None if p.grad is None else p.grad.norm().item()) for n,p in m.named_parameters() if n in ['enc.engram2.Wv.weight','enc.engram2.Wk.weight','core.shared.w1.weight','core.shared.w2.weight','core.shared.w3.weight','core.table','dec.plan.null']}
out['init_shared_alpha_grad_norm']=m.core.table.grad[:,:,6,:].norm().item()
# Show T4b's u.grad check does not detect an omitted stop-gradient: loss bypasses u.
m.zero_grad(set_to_none=True);tok,C,_=m.enc(ids,doc,ei);S=m.core.trajectory(C,doc[:,::4],'3');S.retain_grad();u=m.core.out_norm(S);u.retain_grad();Ss=m.core.trajectory(C,doc[:,::4],'1');C.retain_grad()
((rms(Ss)-rms(S))**2).mean().backward() # deliberately NO detach, local mutant only
out['teacher_detach_mutant']={'current_T4b_condition_passes':bool(C.grad.abs().sum()>0 and (u.grad is None or u.grad.abs().sum()==0)),'teacher_S_grad_norm':S.grad.norm().item()}
# A core can use distant evidence with the encoder's G contribution exactly zero.
m=model()
with torch.no_grad():
 m.enc.crc_router.weight.zero_();m.enc.crc_router.bias.copy_(torch.tensor([0.,-1000.,-1000.]))
logits,_=m(ids,doc,ei);ids2=ids.clone();ids2[0,4]=(ids2[0,4]+7)%64
logits2,_=m(ids2,doc,engram_indices(ids2,doc,m.ecfg))
out['global_path_with_CRC_G_zero']={'crc_mix':m.enc.crc_router.bias.softmax(-1).tolist(),'distant_logit_change_at_t100':(logits2[0,100]-logits[0,100]).abs().max().item(),'edit_position':4,'predictor_position':100}
# Neutral geometry alone cannot be a necessary condition for useful refinement.
updates=[torch.tensor([1.,0.]),torch.tensor([1.,.05]),torch.tensor([1.,.05])];states=[];s=torch.zeros(2)
for dx in updates:s=s+dx;states.append(s.clone())
rho=states[-1].norm()/sum(x.norm() for x in updates);angle=torch.acos(F.cosine_similarity(updates[1],updates[2],dim=0).clamp(-1,1))*180/math.pi
out['aligned_refinement_counterexample']={'angle_2_3_degrees':angle.item(),'rho':rho.item(),'second_direction_coordinate':[float(x[1]/x.norm()) for x in states],'threshold':.03,'passes_M2':bool(angle>=10 and rho<=.9),'decision_by_pass':[bool(x[1]/x.norm()>.03) for x in states]}
out['shortcut_row_optimizer_exposure']={'p_row_per_microbatch':.25,'accum':16,'p_row_used_at_least_once_per_optimizer_step':1-.75**16,'expected_optimizer_steps_used_of_7600':7600*(1-.75**16),'microbatch_exposures':7600*16*.25}
out['logit_tensor_bytes_B2_T1024_V8192_fp32']=2*1024*8192*4
out['NCP_CE_counterexample']=F.cross_entropy(torch.tensor([[-10.]+[0.]*127]),torch.tensor([0])).item()
# Retain model-defined counts, report read2's own parameters without alias table.
small_count=param_table(m);out['read2_part_count']={'reported_with_alias':small_count['parts']['eng2'],'own_only':sum(p.numel() for n,p in m.enc.engram2.named_parameters() if not n.startswith('table.'))}
(root/'draft_v3_review_2026-09-29/claim_probes.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
