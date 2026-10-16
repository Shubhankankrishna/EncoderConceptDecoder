"""The new tests must reject plausible implementation defects by assertion."""
import contextlib, json, sys, os
import torch
import model_v41 as M41
import hrm_decoder as HD
import test_v41 as T
import speed_v3
speed_v3.enable_block_attention(60)

@contextlib.contextmanager
def patch(obj,name,value):
    old=getattr(obj,name);setattr(obj,name,value)
    try:yield
    finally:setattr(obj,name,old)

def noncausal(self,doc,R_local=False):
    local,_=M41.build_masks(doc,self.cfg.window)
    return local,(doc[:,:,None]==doc[:,None,:]).unsqueeze(1)

def nodoc(self,doc,R_local=False):
    local,_=M41.build_masks(doc,self.cfg.window)
    t=torch.arange(doc.shape[1],device=doc.device)
    return local,(t[None,:]<=t[:,None])[None,None].expand(doc.shape[0],1,-1,-1)

def onlylocal(self,doc,R_local=False):
    local,_=M41.build_masks(doc,self.cfg.window);return local,local

original_init=M41.MemoryLayer.__init__
def leakinit(self,cfg,retrieval=False,stack=False):
    original_init(self,cfg,retrieval,stack or retrieval)

def wrongindex(doc,k=4):
    B,TT=doc.shape;t=torch.arange(TT,device=doc.device);j=t//k-1;jc=j.clamp_min(0)
    ok=(j>=0)[None,:]&(doc[:,jc*k]==doc)
    return jc[None,:].expand(B,TT),ok,(t+1)%k

MUTANTS=[("R is noncausal",M41.DecoderV41,"masks",noncausal,T.test_R_reach),
         ("R ignores document ids",M41.DecoderV41,"masks",nodoc,T.test_R_document_isolation),
         ("R reads D K/V",M41.MemoryLayer,"__init__",leakinit,T.test_private_plans),
         ("plan index off by one",HD,"plan_index",wrongindex,T.test_null_alignment),
         ("R is local only",M41.DecoderV41,"masks",onlylocal,T.test_R_reach)]

if __name__=="__main__":
    records=[]
    for name,obj,attr,value,test in MUTANTS:
        with patch(obj,attr,value):
            try:test();rec=dict(mutant=name,killed=False,error="survived");print("SURVIVED",name,flush=True)
            except AssertionError as e:rec=dict(mutant=name,killed=True,test=test.__name__,error=str(e));print("KILLED",name,"by",test.__name__,str(e)[:100],flush=True)
            except Exception as e:rec=dict(mutant=name,killed=False,error=repr(e));print("CRASHED",name,repr(e),flush=True)
        records.append(rec)
    with open(os.path.join(T.R.RUNS,"mutants_v41.json"),"w") as f:json.dump(records,f,indent=2)
    print(f"{sum(r['killed'] for r in records)}/{len(records)} killed by assertions")
    sys.exit(not all(r["killed"] for r in records))
