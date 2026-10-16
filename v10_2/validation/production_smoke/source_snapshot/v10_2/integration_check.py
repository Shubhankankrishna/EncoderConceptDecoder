"""Exercise the full-size evaluator, generation, and gradient probes on a smoke checkpoint."""
import argparse,json,time
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from .model import V102,Architecture
from .evaluate import evaluate
from .prepare_data import DATA

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    torch.set_num_threads(4);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=V102(ModelConfig(**ck['model_config']),Architecture(**ck['architecture']));m.load_state_dict(ck['model']);m.place('mps')
    result=evaluate(m,'mps',DATA,batches=1,math_per_module=1,sudoku_per_family=1,deadline=time.time()+180)
    assert len(result['families'])==4 and result['generation']['math']['n']==4
    assert all(result['generation'][f]['n']==1 for f in ('sudoku_easy','sudoku_hard'))
    assert 'objective_gradient_probe' in result and len(result['objective_gradient_probe']['reader_layers'])==4
    assert all('reader_ablation_delta_nll' in f and len(f['pq_segment_entropy'])==8 for f in result['families'].values())
    result['status']='passed';result['checkpoint_step']=ck['step']
    Path('v10_2/validation/integration_evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',seconds=result['seconds'],probe=result['objective_gradient_probe']),indent=2))
if __name__=='__main__':main()
