"""Render the saved review snapshot; no access to training state or GPU."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT/'diagnostics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
hist = d['validation_history']
x = [r['training_seconds']/3600 for r in hist]
ax[0].plot(x,[r['optimal_action_rate']*100 for r in hist],color='#3454d1',marker='o',ms=4,label='GRAM, fixed 128 validation examples')
b=d['baselines']['fixed_validation_128']
ax[0].axhline(b['fixed_tie_optimal_rate']*100,color='#d45828',ls='--',label='Legal distance heuristic (fixed ties)')
ax[0].axhline(b['uniform_legal_expected_optimal_rate']*100,color='#777777',ls=':',label='Uniform legal action, expected')
ax[0].set(xlabel='Recorded optimizer time (hours)',ylabel='Optimal next action (%)',ylim=(35,100),title='Learning is real; reasoning advantage is unproven')
ax[0].legend(loc='lower right',frameon=True,facecolor='white',edgecolor='none',framealpha=1.,fontsize=8)
ax[0].grid(alpha=.18)
levels=d['action_codes']['levels']
at=np.arange(1,13)
var=np.array([r['variable_positions'] for r in levels]);total=np.array([r['positions'] for r in levels])
ax[1].bar(at,total-var,color='#d5dce5',label='Same code in all seven actions')
ax[1].bar(at,var,bottom=total-var,color='#238a76',label='Varies between actions')
ax[1].axvline(4.5,color='#d45828',ls='--')
ax[1].text(4.7,124,'True levels 1–4 already\nidentify the action',va='top',fontsize=9,color='#a74320')
ax[1].set(xlabel='NCM refinement level',ylabel='Number of target code positions',xticks=at,title='493 / 641 positions are constant across actions')
ax[1].legend(loc='upper left',frameon=False,fontsize=8)
fig.suptitle('V15 Astra architecture review · validation snapshot, 18 October 2026 12:41 IST',fontsize=12)
fig.savefig(ROOT/'review_evidence.png',dpi=180)
fig.savefig(ROOT/'review_evidence.pdf')
print(ROOT/'review_evidence.png')
