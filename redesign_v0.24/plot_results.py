"""Source-level scientific figures; no pseudo-replication confidence bands."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
    data=json.loads((out/'analysis.json').read_text());rows=data['seed_rows'];seeds=sorted({r['seed'] for r in rows});arms=['mean','attention']
    colors=['#326ca8','#ce6531'];get=lambda seed,arm:next(r for r in rows if r['seed']==seed and r['condition']==arm)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7),layout='constrained')
    for ax,group,title in zip(axes,['new12','old'],['A  Socially unseen resource combinations','B  Socially trained resource combinations']):
        y=np.array([[get(s,k)['scores'][group]['pooled']['J']*100 for k in arms] for s in seeds])
        for i,s in enumerate(seeds):ax.plot([0,1],y[i],color='#999999',alpha=.6,lw=1)
        for j in (0,1):
            ax.scatter(np.full(len(seeds),j),y[:,j],color=colors[j],s=35,zorder=3)
            ax.scatter(j,y[:,j].mean(),color='black',marker='_',s=400,lw=3,zorder=4)
        ax.set_xticks([0,1],['Mean: one key','Attention: two keys']);ax.set_xlim(-.3,1.3)
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.set_ylabel('Sequential-greedy joint success (%)');ax.grid(axis='y',alpha=.2);ax.set_title(title,fontsize=11)
    fig.suptitle('Communication from frozen private action predictions',fontsize=14)
    fig.supxlabel('Four inherited source seeds; linked dots are paired source means. Black marks show group means.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(folder/f'01_source_comparison.{ext}',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7),layout='constrained')
    for ax,group,title in zip(axes,['new12','old'],['12 socially unseen layouts','18 socially trained layouts']):
        for arm,color in zip(arms,colors):
            x=[r['update'] for r in get(seeds[0],arm)['curve']]
            y=np.array([[r['scores'][group]['pooled']['J']*100 for r in get(s,arm)['curve']] for s in seeds])
            ax.plot(x,y.mean(0),color=color,label=arm,marker='o',ms=4);ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.12)
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.set_xlabel('Communication updates');ax.set_ylabel('Joint success (%)');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Formation at a fixed social training budget',fontsize=14)
    fig.supxlabel('Shading: observed range across four sources, not a confidence interval. All private heads saw all 30 combinations.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(folder/f'02_learning_curves.{ext}',dpi=190)
    plt.close(fig);print('2 figures saved as PNG/PDF')
if __name__=='__main__':main()
