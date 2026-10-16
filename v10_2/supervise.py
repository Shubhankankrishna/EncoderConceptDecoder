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
            from .provenance import source_files
            for source in source_files():
                dest=snap/source.relative_to(ROOT.parent);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
            shutil.copy2(ROOT.parent/'V10_2_RCDL_Architecture.md',snap/'V10_2_RCDL_Architecture.md')
            for name in ('README.md','VALIDATION_REPORT.md'):
                if (ROOT/name).exists():shutil.copy2(ROOT/name,snap/name)
        main(args)


if __name__=='__main__':supervise()
