"""Plot the frozen main evaluation; no model execution."""
from pathlib import Path
import os,json
A=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(A/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

f=json.loads((A/'main_final_snapshot.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.25,1]})
fig.patch.set_facecolor('#fcfcfd')
families=['lr_recall1','lr_carry3','lr_multi4','lr_hop','lr_bind']
labels=['One value','Three-token\nvalue','Four values','Link chain','Binding']
colors=['#245d96','#d88a39']
x=np.arange(5)
for i,split in enumerate(['test','ood']):
    v=[f['items'][split]['intact'][k]['exact']*100 for k in families]
    bars=axs[0].bar(x+(i-.5)*.34,v,width=.32,color=colors[i],label='Test' if i==0 else 'OOD')
    axs[0].bar_label(bars,labels=[f'{z:.1f}' for z in v],padding=3,fontsize=8)
axs[0].set_xticks(x,labels)
axs[0].set_ylim(0,112)
axs[0].set_yticks([0,25,50,75,100])
axs[0].set_ylabel('Complete-item accuracy (%)')
axs[0].set_title('Long-range tasks at R = 4',loc='left',fontweight='bold')
axs[0].legend(frameon=False,loc='upper right')
depths=[1,2,3,4,6,8]
for fam,split,label,color,style in [('lr_hop','test','Link chain · test','#245d96','-'),('lr_multi4','test','Four values · test','#7d519c','-'),('lr_hop','ood','Link chain · OOD','#d88a39','--')]:
    y=[f['items'][split]['intact' if r==4 else f'R{r}'][fam]['exact']*100 for r in depths]
    axs[1].plot(depths,y,style,marker='o',markersize=4,label=label,color=color)
axs[1].set_ylim(-.6,15)
axs[1].set_yticks([0,5,10,15])
axs[1].set_xticks(depths)
axs[1].set_xlabel('Core passes at inference (R)')
axs[1].set_ylabel('Complete-item accuracy (%)')
axs[1].set_title('Additional passes provide little gain',loc='left',fontweight='bold')
axs[1].legend(frameon=False,loc='upper left',fontsize=9)
for ax in axs:
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#e7e8eb',linewidth=.7)
    ax.set_facecolor('#fcfcfd')
fig.suptitle('Trained ConceptCoreV2 · main checkpoint, step 4,050',x=.075,ha='left',fontweight='bold',fontsize=14)
fig.text(.075,.025,'512 items per family and split. Proposed D6 / whole-model v2 has no trained results. Source: frozen eval_final.json.',fontsize=9,color='#565d68')
fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.81,wspace=.28)
fig.savefig(A/'evidence_summary.png',dpi=180,facecolor=fig.get_facecolor())
print(A/'evidence_summary.png')
