"""Clean-data identity, scoring, telemetry persistence and production resume checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .evaluate import parse_answer,score_math,generation_indices
from v10_1.prepare_data import DATA,ORIGINAL,FAMILIES
from .telemetry import Store
from .train import verify_data
from .provenance import provenance


def main():
    checks={};verify_data(DATA)
    original=json.loads((ORIGINAL/'manifest.json').read_text());new=json.loads((DATA/'manifest.json').read_text())
    for split in ('train','dev'):
        for family in FAMILIES:
            name=f'{family}_{split}'
            before=[json.loads(x) for x in (ORIGINAL/f'{name}_meta.jsonl').read_text().splitlines()]
            after=[json.loads(x) for x in (DATA/f'{name}_meta.jsonl').read_text().splitlines()]
            assert {x['identity'] for x in before}=={x['identity'] for x in after},name
            if family=='math':
                assert all(not x['answer'].startswith(("b'",'b"')) and parse_answer(x['answer']) is not None for x in after)
            # Original active arrays and metadata must remain byte-identical.
            for ext,key in (('.npy','sha256'),('_meta.jsonl','meta_sha256')):
                assert hashlib.sha256((ORIGINAL/(name+ext)).read_bytes()).hexdigest()==original['datasets'][name][key]
    checks['all_train_dev_identities_preserved_and_original_arrays_unchanged']=True
    checks['all_math_answers_clean_and_supported_by_exact_grammar']=True
    checks['clean_tokenizer_and_arrays_hashes_verified']=new['tokenizer_sha256']
    for predicted,target,accepted in [('0.5','1/2',True),('-1.0','-1',True),('1, 2','1,2',True),
        ('2,1','1,2',False),('Answer: 2','2',False),('2 extra','2',False),('1/0','1',False),('nan','1',False),('2','2',True)]:
        assert score_math(predicted,target,True)['correct']==accepted
        assert not score_math(predicted,target,False)['correct']
    checks['strict_numeric_list_grammar_and_required_EOS']=True
    meta=[json.loads(x) for x in (DATA/'math_dev_meta.jsonl').read_text().splitlines()]
    chosen=generation_indices(meta,'math',np.random.default_rng(1927),4,2)
    counts={s:sum(meta[i]['source']==s for i in chosen) for s in {r['source'] for r in meta}}
    assert len(counts)==4 and set(counts.values())=={4} and len(set(chosen))==16
    checks['balanced_math_generation_selection']=counts
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp)/'run';run.mkdir()
        (run/'manifest.json').write_text(json.dumps(dict(start_unix=1,source_sha256='test')))
        (run/'status.json').write_text(json.dumps(dict(status='stopped',step=7)))
        event=dict(schema_version=1,step=7,elapsed_hours=.1,wall_time=1000,loss=dict(future=1.2),
            dense_grad_norm=2.,engram_grad_norm=.4,component_grad_norms=dict(core=.2),selector_regret=.03,
            objective_denominators=dict(token=12),future_visits=[1.3,1.1],future_offset_counts=[1]*8,utility_gap=.02,family_supervised_tokens=dict(text=12))
        (run/'train.jsonl').write_text(json.dumps(event)+'\n')
        store=Store(Path(tmp)/'test.db');store.ingest(run,dict(trainer_alive=False));store.ingest(run,dict(trainer_alive=False))
        with store.connect() as db:
            row=db.execute('SELECT * FROM events').fetchall();assert len(row)==1 and row[0]['wall_time']==1000 and row[0]['time_quality']=='trainer_clock'
            metrics={r['name']:r['value'] for r in db.execute('SELECT * FROM metrics')}
            for key in ('train.loss.future','train.gradient.engram_grad_norm','train.component_grad_norms.core','train.objective_denominators.token','train.future_visits.1','train.future_offset_counts.7','train.utility_gap'):assert key in metrics
    checks['new_telemetry_metrics_clock_and_idempotency']=True
    result=dict(**provenance(),status='passed',checks=checks)
    (Path(__file__).parent/'validation/pipeline_contracts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
