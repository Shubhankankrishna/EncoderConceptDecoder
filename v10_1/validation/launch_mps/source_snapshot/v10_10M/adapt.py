"""Explicit post-pretraining RBF adaptation on permitted training feedback only."""
import argparse
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .model import V10, losses
from .memory import RBFMemory, learn_episode
from .data import Corpus
from .prepare_data import DATA, digest


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--family',choices=['text','math','sudoku_easy','sudoku_hard'],default='math')
    p.add_argument('--device',default='mps');p.add_argument('--eta',type=float,default=.1);p.add_argument('--sigma',type=float,default=1.)
    a=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['data_sha256']!=digest(DATA/'manifest.json'):raise ValueError('Checkpoint data/tokenizer mismatch')
    m=V10(ModelConfig(**ck['model_config']));m.load_state_dict(ck['model']);m.place(a.device).eval().requires_grad_(False)
    era=digest(a.checkpoint)+':'+digest(DATA/'tokenizer.json')
    bank=RBFMemory(era,sigma=a.sigma);train=Corpus('train',seed=4411)
    log=[]
    for i in range(a.episodes):
        b=train.batch(1,family=a.family,device=a.device)
        try:
            event=learn_episode(m,bank,b['ids'],b['lengths'],b['targets'],b['target_mask'],
                                event_id=f'{a.family}:{i}:{b["chosen"][0]}',eta=a.eta)
        except OverflowError:
            log.append(dict(status='capacity_reached',episode=i));break
        log.append(event)
    # Same held-out development examples, scored without any adaptation writes.
    dev=Corpus('dev',seed=4412);scores=[]
    with torch.no_grad():
        for _ in range(8):
            b=dev.batch(2,family=a.family,device=a.device)
            base=m(b['ids'],b['lengths']);adapted=m(b['ids'],b['lengths'],bank=bank.snapshot())
            scores.append({name:float(losses(m,o,b['targets'],b['target_mask'])['token'])
                           for name,o in [('frozen',base),('rbf',adapted)]})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.save(bank.state_dict(),out/'memory.pt')
    (out/'report.json').write_text(json.dumps(dict(era=era,episodes=log,development=scores,
        caveat='Single-stream mechanism pilot, not a retention or general reasoning result. No sealed-test writes.'),indent=2))
    print(out/'report.json')


if __name__=='__main__':main()
