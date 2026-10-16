"""Small CPU checkpoint interventions; not retrained architecture comparisons."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.model import V10, losses

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(1191)
with (ROOT / 'v10_10M/runs/pretrain_24h/latest.pt').open('rb') as f:
    stat = os.fstat(f.fileno())
    ck = torch.load(f, map_location='cpu', weights_only=False)
model = V10(ModelConfig(**ck['model_config'])); model.load_state_dict(ck['model']); model.eval()
result = dict(step=ck['step'], captured_unix=time.time(), checkpoint_mtime=stat.st_mtime,
              source_sha256=ck['source_sha256'], device='cpu', threads=2,
              selection='Dev seed 2861, four batches of two per family with replacement',
              limitations='Small checkpoint interventions, not retrained controls; RBF probe does not write memory.',
              families={})
del ck
result['scales'] = dict(pool_alpha=float(model.pooler.alpha.detach()),
                       anchor_alpha=float(model.core.anchor_scale.detach()),
                       concept_fusion_rms=float(model.decoder.fusion.detach().square().mean().sqrt()))
with torch.no_grad():
    for family in ['text','math','sudoku_easy','sudoku_hard']:
        corpus=Corpus('dev', seed=2861); rows=[]; queries=[]; codes=[]; docs=[]
        for j in range(4):
            batch=corpus.batch(2, context=384, family=family, device='cpu')
            def ce(o): return float(losses(model,o,batch['targets'],batch['target_mask'])['token'])
            out=model(batch['ids'],batch['lengths']); base=ce(out)
            off=model(batch['ids'],batch['lengths'],codes=out['codes'],ablate=True)
            saved=model.decoder.scales.clone(); model.decoder.scales.zero_()
            try: no_crc=model(batch['ids'],batch['lengths'],codes=out['codes'])
            finally: model.decoder.scales.copy_(saved)
            model.c.concept_visits=1
            try: one=model(batch['ids'],batch['lengths'],codes=out['codes'])
            finally: model.c.concept_visits=2
            target=out['pooled'][:,1:]; valid=out['valid_groups'][:,1:]
            pooled=out['pooled'][:,:-1]
            running=pooled.cumsum(1)/torch.arange(1,pooled.shape[1]+1)[None,:,None,None]
            def mse(p): return float(((p-target).square().mean((-1,-2))*valid).sum()/valid.sum().clamp_min(1))
            rows.append(dict(tokens=int(batch['target_mask'].sum()),base_nll=base,
                             concepts_off_delta_nll=ce(off)-base,encoder_crc_off_delta_nll=ce(no_crc)-base,
                             one_concept_visit_delta_nll=ce(one)-base,
                             forecast_mse=mse(out['forecast'][:,:-1]),one_visit_forecast_mse=mse(one['forecast'][:,:-1]),
                             copy_current_mse=mse(pooled),running_mean_mse=mse(running)))
            for b in range(2):
                v=out['valid_groups'][b]; n=int(v.sum())
                queries.append(out['u'][b,v].flatten(-2)/16)
                codes.append(out['codes'][b,v]); docs.append(torch.full((n,),batch['chosen'][b]))
        weights=np.array([x['tokens'] for x in rows])
        summary={k:float(np.average([r[k] for r in rows],weights=weights)) for k in rows[0] if k!='tokens'}
        q=torch.cat(queries); c=torch.cat(codes); d=torch.cat(docs)
        pairs=(c[:,None]==c[None,:]) & (d[:,None]!=d[None,:]) & torch.triu(torch.ones(len(q),len(q),dtype=torch.bool),1)
        kernel=torch.exp(-torch.cdist(q,q).square()/2)[pairs]
        summary['different_document_same_code_sigma1_kernel_quantiles']=np.quantile(kernel.numpy(),[.1,.5,.9]).tolist() if len(kernel) else None
        summary['different_document_same_code_pairs']=len(kernel)
        result['families'][family]=dict(summary=summary,batches=rows)
(DEST/'interface_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
