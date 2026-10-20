"""Real-code GRAM prior-only overfit and CPU/MPS parity acceptance gate."""
import json
import time

import numpy as np
import torch

from .codec import ROOT,FrozenNCMCodec,state_hash
from .data import DATA,atomic_json
from .gram import AstraGRAM,GramConfig
from .tasks import ACTION_TEXTS,parse_action


def main():
    torch.set_num_threads(2);torch.manual_seed(1518);torch.mps.set_per_process_memory_fraction(.6)
    rows=dict(np.load(DATA/'samples.npz'));texts=json.loads((DATA/'texts.json').read_text())
    ix=np.concatenate([np.flatnonzero((rows['split']==0)&(rows['action']==a))[:4] for a in range(7)])
    codec=FrozenNCMCodec('cpu',family='lm1b');before=state_hash(codec.model)
    ids,_=codec.tokenize([texts[i] for i in rows['x'][ix]]);x=codec.encode_ids(ids).clone()
    ai,_=codec.tokenize(ACTION_TEXTS);targets=codec.encode_ids(ai).clone();y=targets[rows['y'][ix]].clone()
    s=torch.from_numpy(rows['state'][ix].astype(np.int64));labels=rows['action'][ix]
    m=AstraGRAM();noise=torch.randn(2,2,16,192)
    # Nonzero q-p difference exercises Gaussian gradients in the parity test.
    with torch.no_grad():m.posterior.out.bias[:192]=.001
    loss,_=m(x[:2],s[:2],y[:2],noise=noise);loss.backward();cpu_loss=float(loss.detach())
    names=['prior.out.weight','posterior.out.weight','slow.ffout.weight','code.0.weight']
    grads={n:dict(m.named_parameters())[n].grad.clone() for n in names}
    m.zero_grad(set_to_none=True);m.to('mps');xm,sm,ym=x.to('mps'),s.to('mps'),y.to('mps')
    loss,_=m(xm[:2],sm[:2],ym[:2],noise=noise.to('mps'));loss.backward()
    report=dict(examples=len(ix),parameters=m.parameter_count(),codec_id=codec.identity['codec_id'],
                cpu_mps_loss_abs_error=abs(float(loss.detach())-cpu_loss),
                cpu_mps_gradient_max_errors={n:float((dict(m.named_parameters())[n].grad.cpu()-grads[n]).abs().max()) for n in names},
                objective='full trajectory NELBO / 641; beta=1; no label corruption',history=[])
    assert report['cpu_mps_loss_abs_error']<1e-4 and max(report['cpu_mps_gradient_max_errors'].values())<1e-4,report
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=0.,foreach=False)
    rng=np.random.default_rng(1518);start=time.perf_counter();passed=False
    for step in range(1,2501):
        jj=rng.choice(len(x),8,replace=False);m.train();opt.zero_grad(set_to_none=True)
        loss,stats=m(xm[jj],sm[jj],ym[jj]);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:
            m.eval()
            with torch.inference_mode():
                pred=torch.cat([m.generate(xm[i:i+4],sm[i:i+4]).cpu() for i in range(0,len(x),4)])
                decoded=codec.decode_text(pred)
                accuracy=float(np.mean([parse_action(t)==a for t,a in zip(decoded,labels)]))
                row=dict(step=step,loss=float(loss.detach()),prior_action_accuracy=accuracy,
                         prior_code_accuracy=float((pred==y).float().mean()),kl_sum=float(stats['kl_sum']),
                         sigma_median=float(stats['sigma_median']),seconds=time.perf_counter()-start)
                if accuracy>=.96 and row['prior_code_accuracy']>.98:
                    # Remove q physically and compare with exactly fixed prior draws.
                    deployed=AstraGRAM(m.config,include_posterior=False).to('mps').eval()
                    deployed.load_state_dict(m.inference_state_dict(),strict=True)
                    fixed=torch.randn(2,4,16,192,device='mps')
                    report['posterior_removed_inference_equal']=bool(torch.equal(m.generate(xm[:4],sm[:4],noise=fixed),deployed.generate(xm[:4],sm[:4],noise=fixed)))
                    rates=[]
                    for seed in (7,17,27):
                        torch.mps.manual_seed(seed)
                        codes=torch.cat([deployed.generate(xm[i:i+4],sm[i:i+4]).cpu() for i in range(0,len(x),4)])
                        rates.append(float(np.mean([parse_action(t)==a for t,a in zip(codec.decode_text(codes),labels)])))
                    report['prior_action_accuracy_three_seeds']=rates
                    passed=min(rates)>=.90 and np.mean(rates)>=.96 and report['posterior_removed_inference_equal']
                    del deployed
                report['history'].append(row);print(json.dumps(row),flush=True)
                atomic_json(ROOT/'reports/gram_preflight.json',report)
                if passed:break
    report['codec_state_unchanged']=state_hash(codec.model)==before
    report['overfit_pass']=bool(passed);report['samples']=[dict(action=int(a),decoded=t[:300]) for a,t in zip(labels,decoded)]
    atomic_json(ROOT/'reports/gram_preflight.json',report)
    assert report['overfit_pass'] and report['codec_state_unchanged'],'GRAM prior-only overfit gate failed'


if __name__=='__main__':main()
