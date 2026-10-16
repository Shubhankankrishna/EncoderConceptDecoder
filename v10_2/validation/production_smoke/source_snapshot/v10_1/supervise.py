"""Exclusive run lock and reproducible source snapshot; no automatic restart."""
import fcntl
from pathlib import Path
import shutil
from .train import main,parser,ROOT


def supervise():
    args=parser().parse_args();run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    with (run/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Run already locked')
        if not args.resume:
            if (run/'latest.pt').exists():raise SystemExit('Existing run; use --resume')
            snap=run/'source_snapshot'
            if snap.exists():raise SystemExit('Source snapshot already exists; choose a new run directory')
            for package in ('v10_1','v10_10M'):
                dest=snap/package;dest.mkdir(parents=True)
                for source in (ROOT.parent/package).glob('*.py'):shutil.copy2(source,dest/source.name)
            shutil.copytree(ROOT.parent/'v10_10M/data_tools',snap/'v10_10M/data_tools',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT.parent/'V10_1_RCDL_Architecture.md',snap/'V10_1_RCDL_Architecture.md')
            for name in ('README.md','VALIDATION_REPORT.md'):
                if (ROOT/name).exists():shutil.copy2(ROOT/name,snap/name)
        main(args)


if __name__=='__main__':supervise()
