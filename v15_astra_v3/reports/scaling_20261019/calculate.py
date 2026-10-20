"""CPU-only arithmetic estimator. Never imports torch or allocates a model.

Counts the present Astra modules exactly, then uses dense matrix multiply counts
as a FLOP proxy. Time estimates are planning scenarios, not measured benchmarks.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/20261018_24h_raw_gram'


def ledger(c):
    d,e,v,L,n,S,M,f,g,C,H=(c['width'],c['code_dim'],c['vocab'],len(c['levels']),max(c['levels']),
        c['slots'],c['state_length'],c['ff'],c['gaussian_hidden'],c['ctx_layers'],c['head_layers'])
    selfblock=4*d*d+3*d*f+4*d
    crossblock=8*d*d+3*d*f+8*d
    target=e*d+(n+L+1)*d+1
    groups=dict(code_tables=L*v*e,code_projections=2*e*d,level_position_pool_queries=(2*L+n)*d,
        pool_projections=2*d*d,typed_state=(128+M)*d,context=C*selfblock,initial_states=2*S*d+2*d*d,
        recurrent=2*crossblock,decoder=H*crossblock,final_norm=2*d,logit_scale_bias=L+L*v,
        prior=6*d+6*d*g,posterior=8*d+8*d*g,target_encoder=target)
    return dict(total=sum(groups.values()),deployment=sum(groups.values())-groups['posterior']-target,groups=groups)


def block_flops(n,m,d,f,cross):
    # A multiply-add counts as two FLOPs. Dense attention score/value products.
    out=8*n*d*d+4*n*n*d+6*n*d*f
    if cross:out+=4*n*d*d+4*m*d*d+4*n*m*d
    return out


def forward_proxy(c,history=2,pooling='pooled'):
    d,e,V,N,W,L,S,R,K,f,g=(c['width'],c['code_dim'],c['vocab'],sum(c['levels']),history,len(c['levels']),
        c['slots'],c['slow_steps'],c['fast_steps'],c['ff'],c['gaussian_hidden'])
    M=W*(L if pooling=='pooled' else L+c['levels'][-1])
    groups=dict(context_embedding_pooling=W*N*(2*e*d+4*d*d+4*d),
        context_blocks=c['ctx_layers']*block_flops(M,M,d,f,False),
        init_projection=4*d*d,
        recurrent_blocks=R*(K+1)*block_flops(S,M+S,d,f,True),
        gaussian_heads=R*28*S*d*g,
        target_encoder=2*N*e*d+4*N*d,
        target_level_summaries=N*(2*e*d+4*d*d+4*d),
        decoder_blocks=c['head_layers']*block_flops(N,M+S+L,d,f,True),
        output_projection=2*N*d*e,output_logits=2*N*e*V)
    if pooling=='hybrid':groups['extra_fine_context_embedding']=W*c['levels'][-1]*2*e*d
    return dict(total=sum(groups.values()),groups=groups,context_vectors=M)


def calculate():
    plan=json.loads((RUN/'plan.json').read_text())
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    assert actual==plan['sources'],'The registered training source changed'
    manifest=json.loads((RUN/'gram/manifest.json').read_text());base=manifest['config']
    base_ledger=ledger(base)
    assert base_ledger['total']==manifest['ledger']['trainable']
    assert base_ledger['deployment']==manifest['ledger']['deployment']
    assert base_ledger['groups']['code_tables']==manifest['ledger']['code_tables']
    rows=[json.loads(s) for s in (RUN/'gram/metrics.jsonl').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if isinstance(r,dict) and 'loss' in r][-60:]
    median_seconds=statistics.median(r['update_seconds'] for r in rows)
    configurations=[('10M current',base)]
    for target,width in [(20000000,384),(30000000,512),(50000000,640)]:
        possibilities=[]
        for ff in range(width,5*width+1,16):
            config={**base,'width':width,'heads':width//64,'ff':ff,'gaussian_hidden':width//2}
            possibilities.append((abs(ledger(config)['total']-target),config))
        config=min(possibilities,key=lambda x:x[0])[1]
        assert config['width']%config['heads']==0
        assert abs(ledger(config)['total']-target)/target<.01
        configurations.append((f'{target//1000000}M proposal',config))
    data=Path(plan['data']);docs=json.loads((data/'documents.json').read_text());examples=json.loads((data/'examples.json').read_text())
    lengths={i:n for doc in docs for i,n in zip(doc['block_ids'],doc['actual_lengths'])}
    train_docs=[doc for doc in docs if doc['split']=='train']
    target_ids={ex['target'] for ex in examples if ex['split']=='train'}
    target_tokens=sum(lengths[i] for i in target_ids)
    data_stats=dict(train_documents=len(train_docs),train_document_token_ids=sum(sum(d['actual_lengths']) for d in train_docs),
        training_windows=sum(e['split']=='train' for e in examples),unique_target_blocks=len(target_ids),
        unique_target_nonpad_token_positions=target_tokens)
    base_flops=forward_proxy(base)['total']
    # 24-hour total, final ten minutes reserved; 5% of the remainder set aside
    # for validation/checkpoint overhead. This is a budgeting assumption.
    update_budget=(24*3600-600)*.95
    effective_batch=manifest['batch']*manifest['accumulation']
    results=[]
    for name,c in configurations:
        parameters=ledger(c);forward=forward_proxy(c);ratio=forward['total']/base_flops
        seconds=median_seconds*ratio;fast=seconds*.65;slow=seconds*1.6
        updates=update_budget/seconds
        record=dict(name=name,config=c,parameters=parameters,forward_proxy=forward,
            fp32_adam_state_bytes=16*parameters['total'],checkpoint_verification_copy_bytes_approx=12*parameters['total'],
            flop_ratio_to_10m=ratio,estimated_step_seconds=seconds,planning_step_seconds_range=[fast,slow],
            central_updates_in_24h=int(updates),planning_updates_in_24h_range=[int(update_budget/slow),int(update_budget/fast)],
            central_target_block_examples=int(updates*effective_batch),
            central_target_window_revisits=updates*effective_batch/data_stats['training_windows'],
            central_nonpad_target_token_exposures=updates*effective_batch/data_stats['training_windows']*target_tokens,
            total_weights_with_frozen_codec=parameters['total']+111404672,
            hybrid_forward_proxy_ratio=forward_proxy(c,pooling='hybrid')['total']/base_flops)
        results.append(record)
    output=dict(created_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        method='Exact module parameter arithmetic; unprofiled dense matrix FLOP proxy; current logged throughput calibration',
        baseline_step_seconds=median_seconds,baseline_observed_step=rows[-1]['step'],baseline_logged_window=len(rows),
        batch=manifest['batch'],accumulation=manifest['accumulation'],device='MPS',dtype='FP32',
        planning_training_update_budget_seconds=update_budget,
        timing_uncertainty_factors=[.65,1.6],timing_range_kind='Subjective planning envelope; not a statistical confidence interval',
        data=data_stats,models=results,source_hashes_unchanged=True,
        exclusions=['No larger model allocated or benchmarked','No accuracy or quality prediction',
            'FLOP proxy omits nonlinearities, normalization, sampling and most elementwise work',
            'Dense attention mask computation assumed; kernel-specific sparse/fused behavior is unprofiled',
            'Forward FLOP ratios are used as a proxy for update cost; GPU utilization and sustained thermals can change ratios',
            'No peak RAM prediction: activation, allocator, checkpoint, codec and OS overhead require measurement',
            'Constant code_dim=64 leaves the output projection rank unchanged',
            'Unique raw pre-tokenizer IDs and correlated hierarchical code positions are different data measures'])
    HERE.mkdir(exist_ok=True)
    (HERE/'estimates.json').write_text(json.dumps(output,indent=2))
    configs=HERE/'candidate_configs';configs.mkdir(exist_ok=True)
    for row in results[1:]:(configs/(row['name'].split()[0].lower()+'.json')).write_text(json.dumps(row['config'],indent=2))
    for row in results:
        print(json.dumps(dict(name=row['name'],parameters=row['parameters']['total'],recurrent=row['parameters']['groups']['recurrent'],
            state_GiB=row['fp32_adam_state_bytes']/1024**3,flop_ratio=row['flop_ratio_to_10m'],seconds=row['estimated_step_seconds'],
            range=row['planning_step_seconds_range'],updates=row['central_updates_in_24h'],hybrid_ratio=row['hybrid_forward_proxy_ratio'])))
    return output


if __name__=='__main__':calculate()
