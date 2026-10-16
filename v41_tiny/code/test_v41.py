"""v4.1 invariants tested at input, concept, plan, memory and gradient sites."""
import argparse, copy, gc, json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F
import model_v4 as M4
import model_v41 as M41
import data_v4 as D
import pinpad as PP
import run_vs as R
import hrm_decoder as HD
from hrm_encoder import engram_indices, build_masks, rope_cache
import speed_v3

torch.set_num_threads(2)
BLOCK=60

def model(stack=False, seed=10):
    torch.manual_seed(seed)
    m=M41.build_model("test",stack=stack).double().eval()
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.ndim==1 or n.endswith(".w"):
                p.add_(.15*torch.randn_like(p))
    return m

def inputs(m,T=120,B=2):
    torch.manual_seed(7)
    doc=torch.zeros(B,T,dtype=torch.long)
    doc[0,T//2:]=1
    ids=torch.randint(5,64,(B,T))
    return ids,doc,engram_indices(ids,doc,m.ecfg)

def decoder_inputs(m,T=120):
    _,doc,_=inputs(m,T)
    torch.manual_seed(21)
    tok=torch.randn(2,T,64,dtype=torch.double)
    u=torch.randn(2,T//6,64,dtype=torch.double)
    cp=torch.randn_like(u)
    return tok,u,cp,doc,m.enc.embed.weight

def test_parameter_ledger():
    counts={}
    for arch in ("v4","v41","v41stack","llm"):
        m=R.build(arch,8192)
        counts[arch]=R.ptable(m)["compute"]
    assert counts==dict(v4=1212032,v41=1212036,v41stack=1212036,llm=1206864),counts
    m=M41.build_model("full")
    p=M4.param_table(m)
    assert p["compute"]==19175580,p
    assert sum(x.numel() for x in m.parameters())==101600284
    del m;gc.collect()
    return counts

def test_backbone_identity():
    torch.manual_seed(55);a=R.build("v4",64);rng=torch.get_rng_state()
    torch.manual_seed(55);b=R.build("v41",64)
    assert torch.equal(torch.get_rng_state(),rng),"conversion changed RNG"
    for n,p in a.state_dict().items():
        assert torch.equal(p,b.state_dict()[n]),n
    extra=set(b.state_dict())-set(a.state_dict())
    assert extra=={"dec.layers.2.log_tau"},extra
    assert torch.count_nonzero(b.dec.layers[2].log_tau)==0
    return "all inherited tensors bit-identical; only four zero temperatures added"

def test_retrieval_oracle():
    m=model();tok,u,cp,doc,E=decoder_inputs(m)
    layer=m.dec.layers[2];D0=tok+m.dec.plan(u,cp,doc)
    _,mask=build_masks(doc,8);rope=rope_cache(120,8,10000.,tok.device,tok.dtype)
    actual=layer(D0,None,tok,mask,rope)
    c=layer.cfg;B,T,_=tok.shape;H,Hk,dh=c.n_heads,c.n_kv_heads,c.head_dim
    x=layer.read_attn([D0])
    q=layer.q_norm(layer.q(x).view(B,T,H,dh).transpose(1,2))*layer.log_tau.exp()[None,:,None,None]
    k=layer.k_norm(layer.k(tok).view(B,T,Hk,dh).transpose(1,2)).repeat_interleave(H//Hk,1)
    v=layer.v(tok).view(B,T,Hk,dh).transpose(1,2).repeat_interleave(H//Hk,1)
    scores=(q@k.transpose(-1,-2))/(dh**.5)
    probs=scores.masked_fill(~mask,float("-inf")).softmax(-1)
    y=(probs@v).transpose(1,2).reshape(B,T,H*dh)
    expected=D0+layer.write_attn(x)*layer.o(y*torch.sigmoid(layer.gate(x)))
    x=layer.read_ffn([expected]);expected=expected+layer.write_ffn(x)*layer.ffn(x)
    err=(actual-expected).abs().max().item()
    assert err<1e-10,err
    return f"manual masked softmax oracle max error {err:.2e}"

def test_R_reach():
    m=model();args=list(decoder_inputs(m));tok=args[0]
    for local in (False,True):
        base=m.dec(*args,R_local=local)[0]
        for p in (0,5,29,53,60,91,119):
            new=tok.clone();new[0,p]+=torch.randn(64,dtype=torch.double)
            changed=(m.dec(new,*args[1:],R_local=local)[0]-base).abs().amax(-1)>1e-10
            assert not changed[1].any(),"cross-row leak"
            stop=min(p+8,(p//60+1)*60) if local else (p//60+1)*60
            assert not changed[0,:p].any(),"future leak"
            assert not changed[0,stop:].any(),"window/document leak"
            assert changed[0,p] and changed[0,stop-1],"missing expected reach"
    return "global reach to document end; R-local exactly window 8; no future/doc/row leak"

def test_R_document_isolation():
    m=model();args=list(decoder_inputs(m));base=m.dec(*args)[0]
    new=args[0].clone();new[0,:60]+=2*torch.randn_like(new[0,:60])
    out=m.dec(new,*args[1:])[0]
    assert torch.equal(out[0,60:],base[0,60:]),"cross-document R read"
    assert torch.equal(out[1],base[1]),"cross-row R read"
    assert not torch.equal(out[0,:60],base[0,:60]),"vacuous perturbation"
    return "changing one whole document cannot change another"

def test_private_plans():
    m=model();args=list(decoder_inputs(m));base=m.dec(*args)[0]
    for j in (0,2,7,9,10,18,19):
        u=args[1].clone();u[0,j]+=torch.randn(64,dtype=torch.double)
        out=m.dec(args[0],u,*args[2:])[0]
        changed=(out-base).abs().amax(-1)>1e-10
        expected=torch.zeros_like(changed)
        for t in range(6*j+5,min(6*j+11,120)):
            if args[3][0,6*j]==args[3][0,t]:expected[0,t]=True
        assert torch.equal(changed,expected),(j,torch.where(changed[0])[0],torch.where(expected[0])[0])
    return "u_j affects only legal positions 6j+5..6j+10, including document edges"

def test_stack_positive_control():
    m=model(stack=True);args=list(decoder_inputs(m));base=m.dec(*args)[0]
    u=args[1].clone();u[0,2]+=1.0
    changed=(m.dec(args[0],u,*args[2:])[0]-base).abs().amax(-1)>1e-10
    assert changed[0,59] and not changed[0,60:].any(),torch.where(changed[0])[0]
    assert not changed[0,:17].any(),"stack future leak"
    assert not changed[1].any()
    return "stack leaks plans forward through K/V while remaining causal and document-isolated"

def test_null_alignment():
    m=model();args=decoder_inputs(m);_,u,cp,doc,_=args
    idx,ok,r=HD.plan_index(doc,6)
    assert torch.where(~ok[0])[0].tolist()==list(range(5))+list(range(60,65))
    assert torch.where(~ok[1])[0].tolist()==list(range(5))
    t=torch.arange(120)
    assert ((6*idx+5<=t)|~ok).all(),"incomplete concept read"
    assert torch.equal(idx[0],((t+1)//6-1).clamp_min(0)),"plan off by one"
    return "first five positions null per document; every plan concept is complete"

def test_no_decoder_engram():
    m=model();assert len(m.dec.eng)==0 and not m.dec.reads_after
    assert not any("engram" in n or ".eng." in n for n,_ in m.dec.named_parameters())
    try:m.dec(*decoder_inputs(m),e=torch.randn(2,120,96,dtype=torch.double))
    except AssertionError:pass
    else:raise AssertionError("decoder accepted Engram rows")
    return "no decoder Engram parameters or handoff"

def test_full_causality_documents():
    m=model();ids,doc,ei=inputs(m);base=M41.forward(m,ids,doc,ei)[0]
    for p in (0,5,6,53,59,60,64,65,100,119):
        ii=ids.clone();ii[:,p]=(ii[:,p]+7-5)%59+5
        oo=M41.forward(m,ii,doc,engram_indices(ii,doc,m.ecfg))[0]
        assert torch.allclose(oo[:,:p],base[:,:p],atol=1e-10,rtol=0),f"future leak at {p}"
        assert (oo[:,p:]-base[:,p:]).abs().max()>1e-8
    ii=ids.clone();ii[0,:60]=5
    oo=M41.forward(m,ii,doc,engram_indices(ii,doc,m.ecfg))[0]
    assert torch.allclose(oo[0,60:],base[0,60:],atol=1e-10,rtol=0),"full graph doc leak"
    return "full graph causal at ten boundary/interior positions; document isolation"

def test_block_parity():
    m=model();ids,doc,ei=inputs(m)
    speed_v3.enable_block_attention(0);a=M41.forward(m,ids,doc,ei)[0]
    speed_v3.enable_block_attention(60);b=M41.forward(m,ids,doc,ei)[0]
    speed_v3.enable_block_attention(BLOCK)
    err=(a-b).abs().max().item();assert err<1e-9,err
    return f"block/full logits parity {err:.2e}"

def test_core_input_reach():
    m=model();args=decoder_inputs(m);tok,_,_,doc,E=args
    C=torch.randn(2,20,64,dtype=torch.double);cdoc=doc[:,::6]
    def predict(C):
        ss,_=m.core.run(C,cdoc);u=m.core.out_norm(ss);_,_,cp=m.core.head(u)
        return m.dec(tok,u,cp,doc,E)[0]
    a=predict(C);new=C.clone();new[0,2]+=1;changed=(predict(new)-a).abs().amax(-1)>1e-10
    assert not changed[0,:17].any() and changed[0,59] and not changed[0,60:].any()
    assert not changed[1].any()
    return "C_j can reach later plans throughout its document; u_j remains private"

def test_gradient_routes():
    m=model();b=inputs(m)
    lg,aux=M41.forward(m,*b)
    # Use language loss only: it must train R, core, pooler and encoder G.
    ntp=F.cross_entropy(lg[:,:-1].reshape(-1,64),b[0][:,1:].reshape(-1))
    m.zero_grad();ntp.backward()
    groups={"R_tau":[m.dec.layers[2].log_tau],"R_KV":[m.dec.layers[2].k.weight,m.dec.layers[2].v.weight],
            "core":[m.core.layers[0].attn.q.weight],"pool":[m.enc.pooler.q],
            "G":[m.enc.layers[3].attn.q.weight]}
    norms={}
    for name,params in groups.items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),name
        norm=sum(p.grad.abs().sum().item() for p in params);assert norm>1e-10,(name,norm);norms[name]=norm
    assert m.core.E0.grad is None and m.core.e.grad is None,"NTP trained codebooks"
    m.zero_grad();_,aux=M41.forward(m,*b)
    comp=M4.chunk_masks(b[0],b[1],6)[1]
    a,z,_,_=m.core.vq_terms(aux["c_hat"],comp,False);(a+z).backward()
    assert m.core.E0.grad is not None and m.core.e.grad is not None
    assert m.enc.embed.weight.grad is None,"VQ trained encoder"
    return norms

def test_padding_finite():
    m=model();ids,doc,ei=inputs(m)
    ids[0,55:60]=0;ids[0,95:]=0;ids[1]=0
    doc[0,96:]=2
    ei=engram_indices(ids,doc,m.ecfg)
    loss,terms,aux=M41.loss_terms(m,ids,doc,ei)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in terms.values()),terms
    assert torch.isfinite(aux["c_hat"]).all()
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    return "partial chunks, padding-only row and empty chunks have finite outputs and gradients"

def test_runner_rows():
    records={};cur=[None];orig=D.to_batch;orig_eval=R.final_eval
    def rec(t,d,canon,ecfg):
        records.setdefault(cur[0],[]).append((np.asarray(t).copy(),np.asarray(d).copy()))
        return orig(t,d,canon,ecfg)
    try:
        D.to_batch=rec;R.final_eval=lambda m,a,S:{}
        for arch in ("v4","v41","v41stack","llm"):
            cur[0]=arch;name="_rows_v41_"+arch
            R.main(["--name",name,"--arch",arch,"--data","pinpad","--seed","5","--max_steps","4", "--eval_every_tokens","1e12","--checkpoint_every","0"])
            shutil.rmtree(os.path.join(R.RUNS,name))
    finally:D.to_batch=orig;R.final_eval=orig_eval
    for arch,bs in records.items():
        training=bs if arch=="llm" else [bs[0]]+bs[3:]
        assert len(training)==4,(arch,len(training))
        for a,z in zip(training,records["llm"]):
            assert np.array_equal(a[0],z[0]) and np.array_equal(a[1],z[1]),arch
    return "all four actual training loops receive identical batches"

TESTS=[test_parameter_ledger,test_backbone_identity,test_retrieval_oracle,test_R_reach,
       test_R_document_isolation,test_private_plans,test_stack_positive_control,
       test_null_alignment,test_no_decoder_engram,test_full_causality_documents,
       test_block_parity,test_core_input_reach,test_gradient_routes,test_padding_finite,test_runner_rows]

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--block",type=int,default=60);ap.add_argument("--only",default="")
    a=ap.parse_args();BLOCK=a.block;speed_v3.enable_block_attention(BLOCK)
    failures=0;results=[]
    for test in TESTS:
        if a.only and a.only not in test.__name__:continue
        try:r=test();print(f"PASS {test.__name__}: {r}",flush=True);results.append(dict(test=test.__name__,pass_=True,detail=r))
        except Exception as e:failures+=1;print(f"FAIL {test.__name__}: {type(e).__name__}: {e}",flush=True);results.append(dict(test=test.__name__,pass_=False,error=str(e)))
    print(f"{len(results)-failures}/{len(results)} passed, {failures} failed",flush=True)
    with open(os.path.join(R.RUNS,f"test_v41_block{BLOCK}.json"),"w") as f:json.dump(results,f,indent=2)
    sys.exit(bool(failures))
