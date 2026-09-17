"""Two source-level figures from independent analysis; fixed primary/secondary."""
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
    a=json.loads((out/'analysis.json').read_text());rows=a['seed_rows'];seeds=a['seeds'];get=lambda s,k:next(r for r in rows if r['seed']==s and r['condition']==k)
    folder=out/'figures';folder.mkdir(exist_ok=True);colors={'mean':'#326ca8','attention':'#ca702d','joint':'#24876c'}
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7),layout='constrained')
    for ax,group,title in zip(axes,['old','new12'],['A  Primary: 18 socially trained layouts','B  Secondary: 12 socially unseen layouts']):
        y=np.array([[100*get(s,k)['scores'][group]['pooled']['J'] for k in ('mean','joint')] for s in seeds])
        for i in range(len(seeds)):ax.plot([0,1],y[i],color='#999999',alpha=.6,lw=1)
        for j,k in enumerate(('mean','joint')):
            ax.scatter(np.full(len(seeds),j),y[:,j],color=colors[k],s=35,zorder=3)
            ax.scatter(j,y[:,j].mean(),color='black',marker='_',s=400,lw=3,zorder=4)
        ax.set_xticks([0,1],['Mean (v24 reference)','Joint (new)']);ax.set_xlim(-.3,1.3);ax.set_ylim(-3,103)
        ax.set_yticks(range(0,101,20));ax.set_ylabel('Sequential-greedy joint success (%)');ax.grid(axis='y',alpha=.2);ax.set_title(title,fontsize=11)
    fig.suptitle('Learning from the same initial communication function',fontsize=14)
    fig.supxlabel('Four paired inherited sources. Joint can learn separate linear weights for the two resource roles.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(folder/f'01_source_comparison.{ext}',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7),layout='constrained')
    for ax,group,title in zip(axes,['old','new12'],['18 socially trained layouts','12 socially unseen layouts']):
        for k,label in [('mean','v24 mean'),('attention','v24 attention'),('joint','new joint')]:
            x=[r['update'] for r in get(seeds[0],k)['curve']];y=np.array([[100*r['scores'][group]['pooled']['J'] for r in get(s,k)['curve']] for s in seeds])
            ax.plot(x,y.mean(0),color=colors[k],label=label,marker='o',ms=4);ax.fill_between(x,y.min(0),y.max(0),color=colors[k],alpha=.10)
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.set_xlabel('Communication updates');ax.set_ylabel('Joint success (%)');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('Joint readout at the inherited fixed update budget',fontsize=14)
    fig.supxlabel('Shading is the observed source range, not a confidence interval. Prior mean/attention are reused references.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(folder/f'02_learning_curves.{ext}',dpi=190)
    plt.close(fig);print('Two PNG/PDF figures saved')
if __name__=='__main__':main()
