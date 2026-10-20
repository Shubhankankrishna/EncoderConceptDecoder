"""Real-checkpoint acceptance, MPS parity, and small-set overfit before long runs."""
import copy
import gc
import json
import time

import numpy as np
import torch

from .codec import ROOT,FrozenNCMCodec,state_hash
from .data import DATA,atomic_json
from .model import AstraHRM
from .tasks import parse_action,ACTION_TEXTS


def main():
    torch.set_num_threads(2);torch.manual_seed(1518);torch.mps.set_per_process_memory_fraction(.6)
    rows=dict(np.load(DATA/'samples.npz'));texts=json.loads((DATA/'texts.json').read_text())
    indices=np.concatenate([np.flatnonzero((rows['split']==0)&(rows['action']==a))[:4] for a in range(7)])
    codec=FrozenNCMCodec('cpu',family='lm1b');before=state_hash(codec.model)
    ids,_=codec.tokenize([texts[i] for i in rows['x'][indices]])
    x=codec.encode_ids(ids).clone();s=torch.from_numpy(rows['state'][indices].astype(np.int64))
    ai,_=codec.tokenize(ACTION_TEXTS);action_codes=codec.encode_ids(ai).clone();y=action_codes[rows['y'][indices]].clone()
    roundtrips=codec.decode_text(action_codes)
    action_gate=all(parse_action(t)==i for i,t in enumerate(roundtrips))
    report=dict(examples=len(indices),codec_action_roundtrip=action_gate,codec_identity=codec.identity)
    assert action_gate,roundtrips
    model=AstraHRM();loss,_=model(x[:2],s[:2],y[:2]);loss.backward()
    cpu_loss=float(loss.detach());names=['fast.attn.q.weight','slow.ffout.weight','code.0.weight']
    grads={n:dict(model.named_parameters())[n].grad.clone() for n in names}
    model.zero_grad(set_to_none=True);model.to('mps')
    xm,sm,ym=x.to('mps'),s.to('mps'),y.to('mps')
    loss,_=model(xm[:2],sm[:2],ym[:2]);loss.backward()
    report['cpu_mps_loss_abs_error']=abs(float(loss.detach())-cpu_loss)
    report['cpu_mps_grad_max_errors']={n:float((dict(model.named_parameters())[n].grad.cpu()-grads[n]).abs().max()) for n in names}
    assert report['cpu_mps_loss_abs_error']<1e-4
    assert max(report['cpu_mps_grad_max_errors'].values())<1e-4
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.,foreach=False)
    rng=np.random.default_rng(1518);history=[];start=time.perf_counter()
    for step in range(1,1501):
        ix=rng.choice(len(x),8,replace=False);model.train();opt.zero_grad(set_to_none=True)
        loss,stats=model(xm[ix],sm[ix],ym[ix],objective='per_code')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:
            model.eval()
            with torch.inference_mode():
                generated=torch.cat([model.generate(xm[i:i+4],sm[i:i+4]).cpu() for i in range(0,len(x),4)])
                decoded=codec.decode_text(generated)
                accuracy=float(np.mean([parse_action(t)==a for t,a in zip(decoded,rows['action'][indices])]))
                row=dict(step=step,loss=float(loss.detach()),code_accuracy=float((generated==y).float().mean()),
                         action_accuracy=accuracy,seconds=time.perf_counter()-start)
                history.append(row);print(json.dumps(row),flush=True)
                report['history']=history;atomic_json(ROOT/'reports/preflight.json',report)
                if accuracy>=.96 and row['code_accuracy']>.98:break
    report['codec_state_unchanged']=state_hash(codec.model)==before
    report['overfit_pass']=history[-1]['action_accuracy']>=.96 and history[-1]['code_accuracy']>.98
    report['samples']=[dict(action=int(a),decoded=t[:300]) for a,t in zip(rows['action'][indices],decoded)]
    atomic_json(ROOT/'reports/preflight.json',report)
    assert report['codec_state_unchanged'] and report['overfit_pass'], 'Overfit gate failed; investigate before training.'


if __name__=='__main__':main()
