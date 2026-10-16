"""Render the frozen audit measurements; no model loads or training operations."""
from pathlib import Path
import os, json
A = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(A / '.matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s = json.loads((A/'snapshot.json').read_text())
d = json.loads((A/'checkpoint_000267_diagnostics.json').read_text())
g = json.loads((A/'gradient_replication.json').read_text())
r = json.loads((A/'routing_replication.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#f7f8fa','axes.facecolor':'#ffffff'})
fig, ax = plt.subplots(2,2,figsize=(13.8,10),layout='constrained')
blue,orange,purple = '#1769aa','#d76524','#7043a2'
w=s['matched_train_windows']; x=[v['step'] for v in w]
for key,label,col in [('v1_ntp','v1',blue),('v2_ntp','v2',orange)]:
    ax[0,0].plot(x,[v[key] for v in w],'-o',label=label,color=col,lw=2)
ax[0,0].set(title='A. v2 learns more slowly on the same data',xlabel='Optimizer step (preceding 50-step mean)',ylabel='Training NTP, nats/token; lower is better')
ax[0,0].legend();ax[0,0].grid(alpha=.15)
depths=[1,2,4,6,8,10,12,16]
ntp=[np.mean([v[str(k)] for v in d['language_summary'].values()]) for k in depths]
ncp=[np.mean([b['ncp'][k-1] for b in d['language_batches']]) for k in depths]
ax[0,1].plot(depths,np.array(ntp)-ntp[0],'-o',color=blue,label='Token prediction (NTP)')
ax[0,1].plot(depths,np.array(ncp)-ncp[0],'-s',color=purple,label='Concept prediction (NCP)')
ax[0,1].axhline(0,color='#555',lw=.8)
ax[0,1].set(title='B. Loops help tokens, but hurt concept prediction',xlabel='Passes at fixed checkpoint 267',ylabel='Loss change from pass 1 (separate objectives)')
ax[0,1].legend(loc='upper left');ax[0,1].grid(alpha=.15)
ax[0,1].text(.98,.02,'16 validation rows; inference test, not retraining',ha='right',transform=ax[0,1].transAxes,fontsize=8,color='#555')
keys=['core_W_in','expert_w2','decoder_Wu','carry_gain']
labels=['Core input','Expert output','Decoder plan','Carry gain']
for i,b in enumerate(g['batches']):
    ax[1,0].scatter([b['aggregate']['nextlat'][k]['cosine'] for k in keys],np.arange(4)+(i-1.5)*.085,
                    color=[blue,orange,purple,'#328062'][i],s=36,label=f'Batch {i+1}')
ax[1,0].axvline(0,color='#555',lw=1)
ax[1,0].set(yticks=np.arange(4),yticklabels=labels,xlim=(-.7,.08),
            title='C. NextLat opposes NTP on these shared weights',xlabel='Cosine of raw gradients (negative = opposing)')
ax[1,0].invert_yaxis();ax[1,0].legend(ncol=2,fontsize=8,loc='lower right');ax[1,0].grid(axis='x',alpha=.15)
row1=[];row2=[]
for b in r['batches']:
    row1.append(100*np.mean([z[0]['real_drop'] for z in b['dispatch_records']]))
    row2.append(100*np.mean([z[1]['real_drop'] for z in b['dispatch_records']]))
x=np.arange(4)
ax[1,1].bar(x-.17,row1,width=.34,color=blue,label='First batch row')
ax[1,1].bar(x+.17,row2,width=.34,color=orange,label='Second batch row')
ax[1,1].set(xticks=x,xticklabels=[f'Batch {i+1}' for i in x],ylim=(0,35),
            title='D. Expert capacity favors earlier batch rows',ylabel='Dropped real-chunk expert assignments (%)')
ax[1,1].legend(fontsize=9);ax[1,1].grid(axis='y',alpha=.15)
fig.suptitle('v2 early-training diagnosis — frozen at 22:34 IST, 28 Sep 2026',fontsize=17,fontweight='bold')
fig.savefig(A/'evidence_summary.png',dpi=180)
fig.savefig(A/'evidence_summary.pdf')
print(A/'evidence_summary.png')
