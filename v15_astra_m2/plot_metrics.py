"""Export learning and systems curves without changing any training state."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run):
    run=Path(run);fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={'gram':'#7c3aed','hrm':'#146c94'}
    for arm in ('gram','hrm'):
        p=run/arm/'metrics.jsonl'
        if not p.exists():continue
        rows=[]
        for line in p.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
        train=[r for r in rows if 'nll_per_code' in r and r.get('event')!='evaluation']
        val=[r for r in rows if r.get('event')=='evaluation']
        x=[r['training_seconds']/3600 for r in train]
        axes[0,0].plot(x,[r['nll_per_code'] for r in train],label=arm,color=colors[arm],alpha=.75)
        vx=[r['training_seconds']/3600 for r in val]
        axes[0,1].plot(vx,[r['optimal_action_rate'] for r in val],'.-',label=arm,color=colors[arm])
        axes[1,0].plot(x,[r['blocks_per_second'] for r in train],color=colors[arm],alpha=.7)
        axes[1,1].plot(x,[r['mps_driver_bytes']/2**30 for r in train],color=colors[arm],alpha=.7)
    for ax,title,y in zip(axes.flat,['Training code NLL','Free decoded validation actions','Throughput','MPS driver allocation'],['Nats / code','Optimal action fraction','Blocks / second','GiB']):
        ax.set_title(title);ax.set_ylabel(y);ax.set_xlabel('Optimizer wall time (hours)');ax.grid(alpha=.2)
    if axes[0,0].lines:axes[0,0].legend(frameon=False)
    axes[0,1].set_ylim(0,1)
    fig.suptitle('V15 Astra · 10M GRAM first / HRM control · frozen NCM · single-seed pilot')
    fig.savefig(run/'learning_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');plot(ap.parse_args().run)
