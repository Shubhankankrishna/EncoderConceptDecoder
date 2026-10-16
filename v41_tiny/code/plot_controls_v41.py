"""Static research figure: distinguish capability, reliance and control authority."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent;RUNS=ROOT/'runs'
BLUE='#2068b0';ORANGE='#dc872d';GREEN='#2d9374'
def get(name,file):
    p=RUNS/name/file
    return json.loads(p.read_text()) if p.exists() else None
def bars(ax,groups,labels,colors):
    for x,(vals,color) in enumerate(zip(groups,colors)):
        if not vals:continue
        ax.bar(x,np.mean(vals),color=color,width=.6,alpha=.8)
        if len(vals)>1:
            ax.vlines(x,min(vals),max(vals),color='#323942',lw=1.4)
            ax.scatter(x+np.linspace(-.1,.1,len(vals)),vals,color='#323942',s=23,zorder=4)
        ax.text(x,np.mean(vals)+.04,f'{np.mean(vals):.1%}',ha='center',fontsize=10)
    ax.set_xticks(range(len(labels)),labels)
def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    modes=['intact','constant_training_mean','zero_content_keep_nulls_offsets','core_off_null_everywhere']
    counts=[]
    for i,(prefix,label,color) in enumerate((('ref_vs_pin_v4','v4 / no R',GREEN),('v41_pin','v4.1',BLUE))):
        ps=[get(f'{prefix}_s{s}_6M','plan_diagnostics.json') for s in range(3)];ps=[p for p in ps if p];counts.append(len(ps))
        for k,mode in enumerate(modes):
            vals=[p[mode]['act_opt_b2'] for p in ps]
            if not vals:continue
            xx=k+(i-.5)*.32
            axs[0,0].bar(xx,np.mean(vals),.3,color=color,alpha=.85,label=label if k==0 else None)
            if len(vals)>1:
                axs[0,0].vlines(xx,min(vals),max(vals),color='#323942')
                axs[0,0].scatter(xx+np.linspace(-.04,.04,len(vals)),vals,color='#323942',s=16,zorder=4)
            axs[0,0].text(xx,max(vals)+.025,f'{np.mean(vals):.1%}',ha='center',fontsize=8)
    axs[0,0].set(xticks=range(4),xticklabels=['Intact','Constant','Zero content','Header null'],title=f'Calibrated plan contents (n={counts[0]}/{counts[1]} seeds)',ylabel='Pair-second accuracy',ylim=(0,1))
    axs[0,0].legend(fontsize=8)
    names=['exploratory_mps_plan_pin_s0_6M','exploratory_noplan_pin_s0_6M']
    normal,no=[get(n,'final.json') for n in names]
    nc,npc=[get(n,'counterfactual.json') for n in names]
    if normal and no and nc and npc:
        x=np.arange(2)
        a=[normal['heldout']['act_opt_b2'],no['heldout']['act_opt_b2']]
        b=[nc['intact']['changed_accuracy'],npc['intact']['changed_accuracy']]
        switched=[j['interventions']['core_off']['heldout']['act_opt_b2'] for j in (normal,no)]
        axs[0,1].bar(x-.24,a,.23,color=BLUE,label='Pair-second decisions')
        axs[0,1].bar(x,b,.23,color=ORANGE,label='Changed-layout decisions')
        axs[0,1].bar(x+.24,switched,.23,color='#8894a3',label='Header-null switch-off b2')
        for xx,y in zip(x-.24,a):axs[0,1].text(xx,y+.03,f'{y:.1%}',ha='center',fontsize=9)
        for xx,y in zip(x,b):axs[0,1].text(xx,y+.03,f'{y:.1%}',ha='center',fontsize=9)
        for xx,y in zip(x+.24,switched):axs[0,1].text(xx,y+.03,f'{y:.1%}',ha='center',fontsize=9)
        axs[0,1].set_xticks(x,['Normal plans','Trained without content']);axs[0,1].legend(fontsize=8)
    axs[0,1].set(title='Matched MPS training — exploratory, one seed',ylabel='Optimal-action accuracy',ylim=(0,1))
    colors=[BLUE,ORANGE];labels=['v4.1','Transformer'];counts=[]
    for i,prefix in enumerate(['v41_pin','ref_vs_pin_llm']):
        js=[get(f'{prefix}_s{s}_6M','counterfactual.json') for s in range(3)];js=[j for j in js if j];counts.append(len(js))
        for k,key in enumerate(['original_accuracy','changed_accuracy']):
            vals=[j['intact'][key] for j in js]
            if not vals:continue
            xx=k+(i-.5)*.32
            axs[1,0].bar(xx,np.mean(vals),.3,color=colors[i],alpha=.85,label=labels[i] if k==0 else None)
            if len(vals)>1:
                axs[1,0].vlines(xx,min(vals),max(vals),color='#323942')
                axs[1,0].scatter(xx+np.linspace(-.05,.05,len(vals)),vals,color='#323942',s=20,zorder=4)
    axs[1,0].set(xticks=[0,1],xticklabels=['Original facts','Changed facts'],ylim=(0,1),ylabel='Optimal-action accuracy',title=f'128 fixed legal counterfactual pairs (n={counts[0]}/{counts[1]})')
    axs[1,0].legend(fontsize=8)
    c=get('v41_pin_s0_6M','control.json')
    if c:
        rates=[.001,.01,.03]
        for site,color in zip(['C','S1','u'],[BLUE,GREEN,ORANGE]):
            vals=[c['sites'][f'{site}@{lr}']['oracle']['fraction_wrong_fixed'] for lr in rates]
            axs[1,1].plot(rates,vals,'o-',color=color,label=site)
        axs[1,1].plot(rates,[0]*3,'--',color='#8894a3',label='Random, matched proposal norm')
        axs[1,1].set_xscale('log');axs[1,1].set_xticks(rates,[str(x) for x in rates]);axs[1,1].legend(fontsize=8)
    axs[1,1].set(title='Frozen oracle offsets — seed 0, ten steps',xlabel='Learning rate',ylabel='Fraction of 52 initial errors fixed',ylim=(-.03,.65))
    for ax in axs.flat:ax.grid(axis='y',alpha=.18)
    fig.suptitle('v4.1: token capability, plan reliance and latent control are different measurements\nDots and lines show observed seed values/ranges, not confidence intervals',fontsize=13)
    fig.savefig(ROOT/'critical_controls.png',dpi=170)
    print(ROOT/'critical_controls.png')
if __name__=='__main__':main()
