"""Calibrated plan controls estimated on independent training examples."""
import contextlib, json, sys
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import run_vs as R

@contextlib.contextmanager
def constant_contents(m,mu,mc,active):
    if not active:
        yield;return
    # Apply at the actual plan boundary, supporting both v4 and v4.1 without
    # altering any checkpoint or original source file.
    orig=m.dec.plan.forward
    def fixed(u,c_pred,doc,*args,**kw):
        return orig(mu.to(u).expand_as(u),mc.to(c_pred).expand_as(c_pred),doc,*args,**kw)
    m.dec.plan.forward=fixed
    try:yield
    finally:m.dec.plan.forward=orig

@torch.no_grad()
def run(name):
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    m,args=A.load(name)
    if args["data"] == "pinpad":
        st=PP.PinpadStream(480,6,4,seed=997);canon=PP.Ident()
    else:
        setup=R.text_setup(R.get_args(["--name","bank","--arch",args["arch"],"--data","text","--seed","997"]))
        st=setup["stream"];canon=setup["canon"]
    us=[];cs=[]
    for _ in range(16):
        b=D.to_batch(*st.next(),canon,m.ecfg)
        _,aux=R.fwd(m,*b);mask=M4.chunk_masks(b[0],b[1],6)[1]
        us.append(aux["u"][mask]);cs.append(aux["c_pred"][mask])
    u=torch.cat(us);c=torch.cat(cs);mu=u.mean(0);mc=c.mean(0)
    stats={}
    projected_u=m.dec.plan.Wu(u);projected_c=m.dec.plan.Wc(c)
    content=projected_u+projected_c
    for key,x in (("u",u),("c_pred",c),("projected_u",projected_u),
                  ("projected_c_pred",projected_c),("projected_content",content)):
        stats[key]=dict(total_mean_square=float(x.square().mean()),
                        mean_vector_mean_square=float(x.mean(0).square().mean()),
                        variation_fraction=float((x-x.mean(0)).square().mean()/x.square().mean().clamp_min(1e-12)))
    mean_content=content.mean(0);null=m.dec.plan.null
    stats["null_replacement"]=dict(null_rms=float(null.square().mean().sqrt()),
                                  mean_content_rms=float(mean_content.square().mean().sqrt()),
                                  rms_distance=float((null-mean_content).square().mean().sqrt()),
                                  cosine=float(torch.nn.functional.cosine_similarity(null[None],mean_content[None])))
    sets=R.pin_eval_sets() if args["data"]=="pinpad" else None
    out={"bank_seed":997,"bank_batches":16,"bank_complete_chunks":len(u),"plan_statistics":stats}
    units={}
    modes=(("intact","intact",{}),("constant_training_mean","constant",{}),
           ("zero_content_keep_nulls_offsets","zero_content",dict(use_u=False,use_c=False)),
           ("core_off_null_everywhere","null",dict(core_off=True)))
    for label,short,kw in modes:
        with constant_contents(m,mu,mc,short=="constant"):
            values=A.pin_units(m,sets["heldout"],**kw) if sets is not None else A.text_units(m,args,**kw)
        out[label]={k:float(np.nanmean(v)) for k,v in values.items() if v.dtype.kind in "fb"}
        for k,v in values.items():units[short+"/"+k]=v
    dest=Path(R.RUNS)/name
    (dest/"plan_diagnostics.json").write_text(json.dumps(out,indent=2)+"\n")
    np.savez(dest/"plan_diagnostic_units.npz",**units)
    print(json.dumps(out,indent=2))

if __name__=="__main__":run(sys.argv[1])
