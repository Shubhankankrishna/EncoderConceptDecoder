"""Exercise the actual failing evaluation flag and both prior-coverage outcomes."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'))
from evaluate import competence,symbolic_eval,wilson
from runtime import atomic_json,implementation_hash
from data import DATA
from model import VOCAB

torch.set_num_threads(4)
raw=(root/'spec/recovery_json_2026-10-06/NCP_unfinalized_evaluation.json.tmp').read_text()
prefix=raw[:raw.rfind('"competence_pass":')].rstrip().removesuffix(',')
recovered=json.loads(prefix+'\n}')
assert recovered['symbolic']['standard']['successes']==1648
recovered['symbolic']['standard']['interval']=wilson(1648,2048)
assert isinstance(recovered['symbolic']['standard']['interval'][0],np.floating)
alpha_zero=json.loads((root/'runs/A_alpha0_dev/summary.json').read_text())
for outcome,result in ((True,recovered),(False,alpha_zero)):
    flag=competence(result)
    assert type(flag) is bool and flag is outcome
    result=dict(result,competence_pass=flag)
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'evaluation.json';atomic_json(p,result)
        assert json.loads(p.read_text())['competence_pass'] is outcome
    json.dumps({k:v for k,v in result.items() if k not in ('symbolic','text')},allow_nan=False)

class Fixture(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
    def forward(self,x,docs):return torch.zeros(*x.shape,VOCAB)

fixture=Fixture();dataset=np.load(DATA/'dev_hard10.npz');coverage=[]
for expected in (True,False):
    calls=[0]
    def generated(system,prompt,z=None,max_new=None):
        batch_index=calls[0]//2;calls[0]+=1;first=batch_index*32;indices=np.arange(first,first+len(prompt))
        hits=(indices<16) if z is None else ((indices>=16)&(indices<48)&expected)
        return dict(x=dataset['x'][indices].copy(),lengths=np.full(len(prompt),prompt.shape[1]+4),verification=[dict(success=bool(hit),reason='fixture') for hit in hits])
    with patch('evaluate.generate',side_effect=generated):
        result=symbolic_eval(fixture,'hard10',n=128,latent_draws=1)
    assert type(result['coverage_gate']) is bool and result['coverage_gate'] is expected
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'coverage.json';atomic_json(p,result)
        assert json.loads(p.read_text())['coverage_gate'] is expected
    coverage.append(dict(expected=expected,baseline_failures=result['coverage']['baseline_failures'],rescued_problems=result['coverage']['rescued_problems']))

out=dict(status='passed',implementation_sha256=implementation_hash(),actual_failed_NCP_evaluation_serializes=True,competence_true_and_false_native_booleans=True,stdout_summary_serializes=True,prior_coverage_outcomes=coverage,scope='Serialization and resume engineering regression; fixtures are not model-quality evidence')
atomic_json(root/'tests/json_recovery_contracts.json',out);print(json.dumps(out))
