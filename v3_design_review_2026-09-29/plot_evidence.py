from pathlib import Path
import os,json,numpy as np
A=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(A/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=json.loads((A/'snapshot.json').read_text());d=json.loads((A/'checkpoint_001703_diagnostics.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
for name,label,col in [('v1','v1','#1769aa'),('v2','v2','#d76524')]:
    rows=[v for v in s['runs'][name]['eval'] if v['step']<=1800]
    axs[0].plot([v['step'] for v in rows],[v['val_lang_mean'] for v in rows],'-o',label=label,color=col,ms=4)
    axs[1].plot([v['step'] for v in rows],[100*v['probes']['lr_recall1']['exact'] for v in rows],'-o',label=label,color=col,ms=4)
axs[0].set(title='Language gap persists at similar updates',xlabel='Optimizer steps (same slots per step)',ylabel='Official validation NTP (nats)')
axs[1].set(title='Retrieval learning is much later in v2',xlabel='Optimizer steps',ylabel='Recall exact match (%)',ylim=(-3,105))
rs=[1,2,4,6,8,12]
ntp=[np.mean([v[str(r)] for v in d['language_summary'].values()]) for r in rs]
ncp=[np.mean([b['ncp'][r-1] for b in d['language_batches']]) for r in rs]
axs[2].plot(rs,np.array(ntp)-ntp[0],'-o',label='Token loss',color='#1769aa')
axs[2].plot(rs,np.array(ncp)-ncp[0],'-s',label='Concept loss',color='#7043a2')
axs[2].axhline(0,color='#999',lw=.6)
axs[2].set(title='v2 loops help tokens, then gains flatten',xlabel='Passes, fixed checkpoint 1703',ylabel='Change from pass 1 (different objectives)')
for ax in axs:ax.legend();ax.grid(alpha=.15)
fig.suptitle('Evidence for the v3 proposal — frozen 29 Sep 2026 at 10:20 IST',fontsize=16,fontweight='bold')
fig.savefig(A/'evidence.png',dpi=170)
print(A/'evidence.png')
