"""Run lock and immutable source snapshot; trainer owns deadline/checkpoints."""
import argparse
import fcntl
from pathlib import Path
import shutil
from .prepare_data import ROOT
from .train import main, parser


if __name__=='__main__':
    args=parser().parse_args()
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    with open(run/'run.lock','w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Run already locked by another trainer')
        if not args.resume:
            snap=run/'source_snapshot';snap.mkdir(exist_ok=True)
            for source in ROOT.glob('*.py'):shutil.copy2(source,snap/source.name)
            for name in ('README.md','ARCHITECTURE.md','DATASETS.md'):
                if (ROOT/name).exists():shutil.copy2(ROOT/name,snap/name)
        main(args)
