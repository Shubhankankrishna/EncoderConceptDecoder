"""Illustrate a measured failure, with selection rule and aggregate rates shown."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pinpad as PP

ROOT=Path(__file__).resolve().parent.parent
RUNS=ROOT/'runs'
def main():
    fixtures=json.loads((RUNS/'counterfactual_fixtures.json').read_text())
    a=np.load(RUNS/'v41_pin_s0_6M/counterfactual_units.npz')
    b=np.load(RUNS/'ref_vs_pin_llm_s0_6M/counterfactual_units.npz')
    mask=a['intact/original_correct']&b['intact/original_correct']&b['intact/changed_correct']&~a['intact/changed_correct']
    mask &= a['intact/original_action']==a['intact/changed_action']
    index=int(np.flatnonzero(mask)[0]);f=fixtures[index]
    fig,axs=plt.subplots(1,2,figsize=(11,6.5),constrained_layout=True)
    cmap=plt.get_cmap('Pastel1')
    for ax,label,key in zip(axs,['Original layout','Target location changed'],['original','changed']):
        seq=f[key];colors=[seq[2*i+1]-PP.POS0 for i in range(8)]
        walls=[seq[i]-PP.POS0 for i in (17,19,21,23)]
        start=seq[25]-PP.POS0;current=seq[-1]-PP.POS0
        for cell in range(49):
            r,c=divmod(cell,7)
            fc='#e4e5e8' if cell in walls else 'white'
            if cell in colors:fc=cmap(colors.index(cell))
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=fc,edgecolor='#c7ccd2',linewidth=.6))
            if cell in colors:
                ci=colors.index(cell);ax.text(c,r,f'C{ci}',ha='center',va='center',fontsize=12,fontweight='bold')
            if cell in walls:ax.text(c,r,'wall',ha='center',va='center',fontsize=8,color='#757575')
        target=colors[f['target']];rr,cc=divmod(target,7)
        ax.add_patch(Rectangle((cc-.46,rr-.46),.92,.92,fill=False,edgecolor='#137341',linewidth=3))
        path=[start]+[seq[i]-PP.POS0 for i in range(28,len(seq),2)]
        xs=[p%7 for p in path];ys=[p//7 for p in path]
        ax.plot(xs,ys,color='#5f6874',linewidth=1.5,alpha=.7,zorder=3)
        ax.scatter([start%7],[start//7],marker='s',s=30,color='#5f6874',zorder=4)
        r,c=divmod(current,7)
        ax.scatter([c],[r],s=310,facecolors='none',edgecolors='#1a1a1a',linewidths=1.5,zorder=5)
        pred_key='intact/'+key+'_action'
        for u,offset,color in ((a,-.13,'#2068b0'),(b,.13,'#dc872d')):
            action=int(u[pred_key][index]);dr,dc=PP.MOVES[action]
            ax.annotate('',xy=(c+.75*dc+offset,r+.75*dr+offset),xytext=(c+offset,r+offset),
                        arrowprops=dict(arrowstyle='-|>',color=color,lw=2.4),zorder=7)
        opts=f[key+'_opt'];valid='/'.join(PP.ACT_NAMES[o] for o in opts)
        rp=PP.ACT_NAMES[int(a[pred_key][index])];lp=PP.ACT_NAMES[int(b[pred_key][index])]
        ax.set_title(f'{label}\nOptimal: {valid}   |   v4.1: {rp}   |   LLM: {lp}',fontsize=12)
        ax.set(xlim=(-.65,6.65),ylim=(6.65,-.65),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
    fig.suptitle(f'Same valid history; next target is C{f["target"]}\nBlue = v4.1, orange = transformer; green border = target',fontsize=14)
    ra=float(a['intact/changed_correct'].mean());rb=float(b['intact/changed_correct'].mean())
    fig.supxlabel(f'Seed 0, all 128 paired prefixes: changed-fact accuracy v4.1 {ra:.1%}, transformer {rb:.1%}.\nIllustration: first case where both start correct, only the transformer follows the moved fact.',fontsize=10)
    fig.savefig(ROOT/'counterfactual_case.png',dpi=170)
    (RUNS/'illustration_selection.json').write_text(json.dumps(dict(index=index,rule='first both-original-correct / LLM-changed-correct / v41-changed-wrong / v41-action-unchanged',fixture=f),indent=2)+'\n')
    print(ROOT/'counterfactual_case.png')
if __name__=='__main__':main()
