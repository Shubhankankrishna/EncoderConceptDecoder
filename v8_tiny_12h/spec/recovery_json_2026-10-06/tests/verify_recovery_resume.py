"""Resume must reuse completed work, retain its provenance, and keep the clock."""
import ast,json,sys,time
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
        assert inherited.call_count==0
        assert q.train('A_delayed_ncp','A',24000,alpha=.1,required=True)==Path('/fresh-foundation') and inherited.call_count==1
    try:q.train('A_alpha0','A',24000,alpha=.1);raise AssertionError('Wrong-alpha reuse was allowed')
    except RuntimeError:pass
    old=q.recovery['completed_checkpoint_sha256'];q.recovery['completed_checkpoint_sha256']='invalid'
    try:q.train('A_alpha0','A',24000,alpha=0.);raise AssertionError('Changed checkpoint reuse was allowed')
    except RuntimeError:pass
    q.recovery['completed_checkpoint_sha256']=old
out=dict(status='passed',implementation_sha256=implementation_hash(),completed_training_reused_without_training_dispatch=True,NCP_still_fresh_training=True,wrong_alpha_rejected=True,checkpoint_integrity_enforced=True,original_budget_preserved=True,scientific_sequence_and_no_score_stop_policy_unchanged=True,remaining_hours=(q.deadline-time.time())/3600)
atomic_json(root/'tests/recovery_resume_contracts.json',out);print(json.dumps(out))
