"""Read-only terminal summary; never starts or changes an experiment."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from .codec import ROOT


def main():
    runs=sorted((ROOT/'runs').glob('*/plan.json'))
    if not runs:
        print('No launched 48-hour experiment yet.');return
    run=runs[-1].parent;plan=json.loads(runs[-1].read_text())
    print('Run:',run)
    print('Deadline:',datetime.fromtimestamp(plan['deadline'],ZoneInfo('Asia/Kolkata')).isoformat())
    for p in [run/'status.json',ROOT/'artifacts/data/encoding_progress.json',run/'gram/status.json',run/'hrm/status.json']:
        if p.exists():print(p.name if p.parent==run else str(p.relative_to(ROOT)),json.dumps(json.loads(p.read_text())))


if __name__=='__main__':main()
