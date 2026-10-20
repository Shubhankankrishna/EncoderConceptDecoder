import argparse
import gc
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from .codec import FrozenNCMCodec, ROOT, area_matrix, state_hash
from .tasks import ACTION_TEXTS, parse_action, make_episodes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=2000);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1518)
    report={'n':args.n,'seed':1518,'purpose':'development-only codec acceptance; never final-test data'}
    texts=[]
    for i in range(args.n):
        k=i%5
        texts.append([
            f'The number is {i%127}. The goal is {(i*17)%127}. The modulus is {127}.',
            f'The current value is {-i}. The next operation is add {i%9+1}.',
            f'The path goes from node {i%81} to node {(i*19)%81}.',
            f'The stack contains {i%10}, {(i+1)%10}, and {(i+2)%10}.',
            f'A child found {i%30} stones and gave {i%7} stones to a friend.'
        ][k])
    for family in ('tinystories','lm1b'):
        c=FrozenNCMCodec('cpu',family=family); before=state_hash(c.model)
        x,lens=c.tokenize(texts)
        counts=dict(nonpad_correct=0,nonpad_total=0,payload_correct=0,payload_total=0,
                    all_correct=0,exact_blocks=0,number_fields_correct=0,number_fields_total=0)
        failures=[]; t=time.perf_counter();parts=[]
        for start in range(0,len(x),16):
            xx=x[start:start+16];codes=c.encode_ids(xx);parts.append(codes)
            yy=c.decode_logits(codes).argmax(-1).cpu()
            mask=xx!=c.pad
            payload=mask.clone()
            if family=='lm1b':
                payload &= (xx!=101)&(xx!=102)
            else:payload &= xx!=c.eos
            counts['nonpad_correct']+=int(((xx==yy)&mask).sum())
            counts['nonpad_total']+=int(mask.sum())
            counts['payload_correct']+=int(((xx==yy)&payload).sum())
            counts['payload_total']+=int(payload.sum())
            counts['all_correct']+=int((xx==yy).sum())
            counts['exact_blocks']+=int(((xx==yy)|~mask).all(-1).sum())
            for j in range(len(xx)):
                # Oracle reconstruction scoring only: original span is known.
                # The deployed action parser never uses target lengths.
                reconstructed=c.bpe.decode(yy[j,:int(lens[start+j])].tolist(),skip_special_tokens=True)
                expected=re.findall(r'-?\d+',texts[start+j]);actual=re.findall(r'-?\d+',reconstructed)
                counts['number_fields_correct']+=int(expected==actual)
                counts['number_fields_total']+=1
                if expected!=actual and len(failures)<20:
                    failures.append(dict(original=texts[start+j],reconstructed=reconstructed))
        elapsed=time.perf_counter()-t
        codes=torch.cat(parts)
        official=c.model.encode(x[:16]);assert torch.equal(official,codes[:16])
        assert torch.equal(c.encode_ids(x[:16]),codes[:16])
        ax,_=c.tokenize(ACTION_TEXTS);ac=c.encode_ids(ax);decoded=c.decode_text(ac)
        actions=[dict(expected=i,text=ACTION_TEXTS[i],decoded=s,parsed=parse_action(s)) for i,s in enumerate(decoded)]
        assert len(set(tuple(r) for r in ac.tolist()))==len(ac)
        batch_drift=float((c.encode_ids(x[:1])[0]!=codes[0]).float().mean())
        partial={}
        for k in (1,3,6,9,12):
            yy=c.decode_logits(codes[:16],k).argmax(-1).cpu()
            mask=x[:16]!=c.pad
            partial[k]=float(((x[:16]==yy)&mask).sum()/mask.sum())
        unchanged=before==state_hash(c.model);assert unchanged
        out=dict(**counts,nonpad_accuracy=counts['nonpad_correct']/counts['nonpad_total'],
                 payload_accuracy=counts['payload_correct']/counts['payload_total'],
                 all_token_accuracy=counts['all_correct']/(len(x)*128),
                 critical_field_accuracy=counts['number_fields_correct']/len(x),
                 failures=failures,action_roundtrips=actions,seconds=elapsed,
                 blocks_per_second=len(x)/elapsed,official_cpu_exact_parity=True,
                 single_vs_batch_code_fraction_changed=batch_drift,partial_nonpad=partial,
                 frozen_state_unchanged=unchanged,manifest=c.identity)
        report[family]=out
        np.save(ROOT/f'artifacts/{family}_action_codes.npy',ac.numpy().astype('<u2'))
        (ROOT/'reports/codec_audit.json').write_text(json.dumps(report,indent=2))
        print(family, {k:v for k,v in out.items() if k.endswith('accuracy') or k=='seconds'},flush=True)
        print('action matches', [a['parsed']==a['expected'] for a in actions],flush=True)
        del c;gc.collect()
    c=FrozenNCMCodec('mps',family='lm1b')
    xx,_=c.tokenize(texts[:32]);t=time.perf_counter();mc=c.encode_ids(xx);torch.mps.synchronize()
    report['mps']=dict(encode_32_seconds=time.perf_counter()-t)
    # Check fixed-size pooling semantics independently of quantization boundaries.
    z=torch.randn(3,128,384);errs=[]
    for n in c.levels:
        ref=F.interpolate(z.transpose(1,2),size=n,mode='area').transpose(1,2)
        got=torch.matmul(area_matrix(128,n,'mps'),z.to('mps')).cpu()
        errs.append(float((ref-got).abs().max()))
    report['mps']['area_max_abs_error']=max(errs)
    c.model.to('cpu');c.device=torch.device('cpu')
    # Reload unmodified source for CPU reference; do not reuse the MPS bound patch.
    del c;gc.collect();torch.mps.empty_cache()
    c=FrozenNCMCodec('cpu',family='lm1b');cc=c.encode_ids(xx)
    report['mps']['code_fraction_changed_from_cpu']=float((cc!=mc.cpu()).float().mean())
    report['decision']={
        'family':'lm1b',
        'exact_symbolic_text_only_accepted':report['lm1b']['critical_field_accuracy']==1.0,
        'state_conditioned_track':True,
        'action_codec_accepted':all(a['parsed']==a['expected'] for a in report['lm1b']['action_roundtrips']),
        'canonical_encoding_device':'cpu',
        'explanation':'Explicit current-state input preserves exact simulator state; action ends at first canonical sentence. No target length used in generation.'}
    (ROOT/'reports/codec_audit.json').write_text(json.dumps(report,indent=2))
    (ROOT/'artifacts/codec/manifest.json').write_text(json.dumps(report['lm1b']['manifest'],indent=2))
    print(report['mps'],report['decision'],flush=True)
    assert report['decision']['action_codec_accepted'], 'Action-code gate failed; do not launch training.'


if __name__=='__main__':main()
