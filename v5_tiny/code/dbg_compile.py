import torch, numpy as np
import data_v5 as dv
from model_v5 import *
from common_v5 import Mixer, domain_mean
dev="mps"
torch.manual_seed(0)
model = NCPModel(dv.VOCAB).to(dev).eval()
x, m, _ = Mixer(seed=3).batch()
def cmp(name, f):
    with torch.no_grad():
        a=f(x); b=torch.compile(f)(x)
    a = a if isinstance(a,(tuple,list)) else (a,)
    b = b if isinstance(b,(tuple,list)) else (b,)
    for i,(u,v) in enumerate(zip(a,b)):
        print(name, i, tuple(u.shape), float((u.float()-v.float()).abs().max()), float(u.float().abs().max()))
torch._dynamo.reset(); cmp("emb", lambda x: model.emb(x))
torch._dynamo.reset(); cmp("block0", lambda x: model.enc[0](model.emb(x), rope_cache(192,32,dev)))
torch._dynamo.reset(); cmp("encode", lambda x: model.encode(x))
torch._dynamo.reset(); cmp("logits", lambda x: model(x)["logits"])
torch._dynamo.reset(); cmp("nll", lambda x: token_nll(model(x)["logits"], x, m))
x2=x.cpu(); 
o = model(x)
nl = token_nll(o["logits"], x, m).detach()
torch._dynamo.reset(); cmp("domain_mean", lambda x: domain_mean(nl, m))
torch._dynamo.reset(); cmp("cmask", lambda x: concept_mask(m))
torch._dynamo.reset(); cmp("ncpmse", lambda x: ncp_mse(o["chat"].detach(), o["C"].detach(), concept_mask(m)))
torch._dynamo.reset(); cmp("sum", lambda x: (nl.sum(), m[:, :-1].sum(), m[:32,:-1].sum(), nl[:32].sum()))
