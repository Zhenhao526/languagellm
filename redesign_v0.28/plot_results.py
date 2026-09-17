"""Scientific figures from the independent source-level analysis only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/formation_001'
sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args();OUT=args.out.resolve()
    a=json.loads((OUT/'analysis.json').read_text());rows=a['seed_rows'];agg=a['aggregate'];seeds=sorted({r['seed'] for r in rows})
    get=lambda seed,system:next(r for r in rows if r['seed']==seed and r['condition']==system)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    colors=['#1876bd','#d06528','#458047'];labels=['Private18 / Comm18','Private30 / Comm18','Private30 / Comm30']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,systems,title,group in [(axes[0],['private_old','private_all'],'A  Private action on target12','new12'),
                                   (axes[1],['old_old','all_old','all_all'],'B  Communication on target12','new12'),
                                   (axes[2],['old_old','all_old','all_all'],'C  Communication on old18','old')]:
        values=np.asarray([[100*get(s,k)['scores'][group]['pooled']['J'] for k in systems] for s in seeds])
        for i,s in enumerate(seeds):ax.plot(np.arange(len(systems)),values[i],color='#919191',alpha=.55,lw=1,zorder=1)
        for j,k in enumerate(systems):
            ax.scatter(np.full(len(seeds),j),values[:,j],color=colors[j],s=28,zorder=3)
            ax.scatter(j,values[:,j].mean(),marker='_',s=320,linewidths=2.8,color='black',zorder=4)
        ax.set_xticks(range(len(systems)),['Private18','Private30'] if len(systems)==2 else ['A: 18 / 18','B: 30 / 18','C: 30 / 30'])
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.22);ax.set_title(title,fontsize=11);ax.set_ylabel('Greedy joint success (%)')
    fig.suptitle('New-material formation replication: private experience and communication',fontsize=15)
    fig.supxlabel('Four initialization seeds; one test-water image. Target12 is communication-unseen in A/B, trained in C.',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(figures/f'01_private_and_communication.{ext}',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,group,title in zip(axes,['new12','old'],['Target12: unseen in A/B, trained in C','Old18: trained in every communication arm']):
        for k,color,label in zip(['old_old','all_old','all_all'],colors,labels):
            times=[r['update'] for r in agg[k]['curve']]
            v=np.asarray([[100*r['scores'][group]['pooled']['J'] for r in get(s,k)['curve']] for s in seeds])
            ax.plot(times,v.mean(0),marker='o',ms=4,color=color,label=label)
            ax.fill_between(times,v.min(0),v.max(0),color=color,alpha=.12)
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.set_xlabel('Communication updates');ax.set_ylabel('Greedy joint success (%)');ax.set_title(title,fontsize=11);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best');fig.suptitle('New-material communication learning at matched budgets',fontsize=15)
    fig.supxlabel(f'Shading is the observed source range, not a confidence interval. {len(a["times"])} fixed evaluation checkpoints.',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(figures/f'02_learning_curves.{ext}',dpi=190)
    plt.close(fig)
    print('two scientific figures saved as PNG and PDF')
if __name__=='__main__':main()
