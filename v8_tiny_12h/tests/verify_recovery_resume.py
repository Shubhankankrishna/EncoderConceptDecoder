"""Resume must reuse completed work, retain its provenance, and keep the clock."""
import ast,json,sys,time,torch
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'))
from drive_12h import Queue,OriginalQueue
from runtime import implementation_hash,atomic_json
def main_ast(p):
    cl=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Queue')
    return ast.dump(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='main'),include_attributes=False)
assert main_ast(root/'spec/recovery_2026-10-06/code/drive_12h.py')==main_ast(root/'code/drive_12h.py')
with patch('drive.Log'):
    q=Queue(12,True)
    assert q.deadline==q.recovery['original_deadline_epoch'] and q.start==q.recovery['original_start_epoch']
    with patch.object(OriginalQueue,'train',return_value=Path('/fresh-foundation')) as inherited:
        assert q.train('A_alpha0','A',24000,alpha=0.,required=True)==root/'runs/A_alpha0/checkpoint.pt'
        assert q.train('A_delayed_ncp','A',24000,alpha=.1,required=True)==root/'runs/A_delayed_ncp/checkpoint.pt'
        assert inherited.call_count==0
        assert q.train('A_flat','A',24000,alpha=0.,flat=True,required=True)==Path('/fresh-foundation') and inherited.call_count==1
    for name,alpha in [('A_alpha0',0.),('A_delayed_ncp',.1)]:
        for kwargs in ({'alpha':1.},{'alpha':alpha,'context':384},{'alpha':alpha,'batch':32}):
            try:q.train(name,'A',24000,**kwargs);raise AssertionError('Wrong setting reuse was allowed')
            except RuntimeError:pass
        record=q.recovery['completed_foundations'][name]
        for field in ('checkpoint_sha256','summary_sha256','training_implementation_sha256'):
            old=record[field];record[field]='invalid'
            try:q.train(name,'A',24000,alpha=alpha);raise AssertionError('Changed provenance reuse was allowed')
            except RuntimeError:pass
            finally:record[field]=old
        checkpoint=torch.load(root/'runs'/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert checkpoint['step']==24000 and checkpoint['config']['alpha']==alpha and checkpoint['config']['implementation_sha256']==record['training_implementation_sha256']
        del checkpoint
    with patch.object(q,'run',side_effect=AssertionError('Completed evaluation was rerun')):
        assert q.eval('A_alpha0_dev',root/'runs/A_alpha0/checkpoint.pt')['status']=='completed'
        try:q.eval('A_alpha0_dev',root/'runs/A_delayed_ncp/checkpoint.pt');raise AssertionError('Wrong evaluation source reused')
        except RuntimeError:pass
out=dict(status='passed',implementation_sha256=implementation_hash(),both_completed_foundations_reused_without_training_dispatch=True,ordinary_transformer_still_fresh=True,wrong_settings_rejected=True,checkpoint_and_summary_integrity_enforced=True,training_provenance_enforced=True,completed_alpha_zero_evaluation_reused_with_original_provenance=True,original_budget_preserved=True,scientific_sequence_and_no_score_stop_policy_unchanged=True,remaining_hours=(q.deadline-time.time())/3600)
atomic_json(root/'tests/recovery_resume_contracts.json',out);print(json.dumps(out))
