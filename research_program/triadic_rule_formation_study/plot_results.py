"""Two fixed scientific figures from the completed JSON-only summary."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .summarize import HERE,SEEDS,STEPS,CONDITIONS,TARGET,read,write,sha,require
except ImportError:
    from summarize import HERE,SEEDS,STEPS,CONDITIONS,TARGET,read,write,sha,require

COLORS=('#286090','#89abc6','#be6231','#ddb393')
LABELS=('Strict / live','Strict / silent','Reciprocal / live','Reciprocal / silent')
STYLES=('-','--','-','--')


def plotted_data(summary):
    require(summary.get('status')=='completed_json_only_summary' and summary.get('audit_status')=='passed' and
            summary['primary_exact_match'],'Audited completed summary required')
    require(summary['primary']['partition']==TARGET and summary['primary']['primary_measure']=='native_Q' and
            summary['primary']['primary_statistic']=='centered_AUC','Unique primary changed')
    trajectory={(r['condition'],r['update']):r for r in summary['target_trajectory_summaries']}
    entropy={(r['condition'],r['actor'],r['window']):r for r in summary['message_entropy_curves']}
    return dict(updates=list(STEPS),seeds=list(SEEDS),conditions=list(CONDITIONS),
        Q_percent={view:{c:[100*trajectory[c,t]['scoring'][view]['means']['Q'] for t in STEPS] for c in CONDITIONS}
                   for view in ('native','common_reciprocal')},
        centered_DiD_pp={name:dict(mean=(100*np.asarray(summary['trajectory_interactions'][name]['mean_centered_DiD'])).tolist(),
            by_seed=[dict(seed=r['seed'],values=(100*np.asarray(r['centered_DiD'])).tolist())
                     for r in summary['trajectory_interactions'][name]['by_seed']])
            for name in ('native_Q','common_reciprocal_Q')},
        native_AUC_pp=[dict(seed=r['seed'],value=100*r['measures']['native_Q']['centered_AUC']) for r in summary['primary']['by_seed']],
        primary_mean_pp=100*summary['primary']['statistics']['mean'],
        primary_interval_pp=[100*summary['primary']['statistics'][k] for k in ('ci95_lower','ci95_upper')],
        entropy_bits=[dict(condition=c,actor=a,window=w,values=entropy[c,a,w]['means']['conditional_entropy_bits'])
                      for c in CONDITIONS for a in range(3) for w in range(2)])


def render(data,destination):
    """Rendering only; caller checks completed-input provenance before entry."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
        'axes.labelsize':10,'savefig.dpi':190,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2))
    x=np.asarray(data['updates'])
    for ax,view,title in zip(axes[0],('native','common_reciprocal'),
                           ('A  Native-rule Q','B  Common reciprocal Q (auxiliary)')):
        for c,label,color,style in zip(CONDITIONS,LABELS,COLORS,STYLES):
            ax.plot(x,data['Q_percent'][view][c],label=label,color=color,linestyle=style,linewidth=2,marker='o',markersize=3)
        ax.set(title=title,xlabel='Training updates',ylabel='Q (%)');ax.legend(loc='best',frameon=False,fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_xlim(0,6000);ax.set_ylim(bottom=0)
    ax=axes[1,0]
    for name,label,color in (('native_Q','Native (primary path)','#286090'),
                            ('common_reciprocal_Q','Common reciprocal (auxiliary)','#be6231')):
        for row in data['centered_DiD_pp'][name]['by_seed']:
            ax.plot(x,row['values'],color=color,alpha=.13,linewidth=.7)
        ax.plot(x,data['centered_DiD_pp'][name]['mean'],color=color,label=label,linewidth=2.3,marker='o',markersize=3)
    ax.axhline(0,color='#707070',linewidth=.7)
    ax.set(title='C  Communication-by-rule DiD minus step 0',xlabel='Training updates',ylabel='Centered DiD (percentage points)')
    ax.set_xlim(0,6000);ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];values=[r['value'] for r in data['native_AUC_pp']]
    ax.scatter(np.arange(16),values,color='#286090',s=24,zorder=3,label='One paired initialization')
    mean=data['primary_mean_pp'];lo,hi=data['primary_interval_pp']
    ax.errorbar(17,mean,yerr=np.array([[mean-lo],[hi-mean]]),color='#111111',fmt='D',markersize=5,
                capsize=5,linewidth=1.6,zorder=4)
    ax.axvline(16,color='#dddddd',linewidth=1);ax.axhline(0,color='#707070',linewidth=.7)
    ax.set_xticks(list(range(16))+[17],labels=[str(s) for s in SEEDS]+['Mean\n95% t15'])
    for tick in ax.get_xticklabels()[:-1]:tick.set_rotation(60);tick.set_horizontalalignment('right');tick.set_fontsize(7)
    ax.get_xticklabels()[-1].set_fontsize(8)
    ax.set(title='D  Unique primary: centered native Q time AUC',ylabel='Time-averaged difference (percentage points)')
    ax.set_xlim(-.7,18.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Rule and communication during training — complete double holdout',fontsize=14,y=.982)
    fig.text(.055,.019,'Curves average 16 paired initializations. Thin lines in C are individual blocks. '
             'The only interval is the approximate society-level t15 interval in D.\n'
             'DiD = (reciprocal live − silent) − (strict live − silent). AUC uses update-count trapezoids / 6000. Q is a behavioral response.',
             fontsize=8,color='#444444',va='bottom')
    fig.subplots_adjust(left=.075,right=.982,bottom=.12,top=.92,wspace=.24,hspace=.38)
    for suffix in ('png','pdf'):fig.savefig(destination/f'01_formation_trajectory.{suffix}')
    plt.close(fig)

    fig,axes=plt.subplots(3,2,figsize=(11.5,9.3),sharex=True,sharey=True)
    rows={(r['condition'],r['actor'],r['window']):r for r in data['entropy_bits']}
    # Size the shared axis from every panel so a later high-entropy trajectory
    # cannot be clipped by the first panel's autoscale range.
    entropy_max=max(value for row in data['entropy_bits'] for value in row['values'])
    entropy_ylim=(0.0, max(1.0, entropy_max * 1.10 + 0.05))
    for actor in range(3):
        for window in range(2):
            ax=axes[actor,window]
            for c,label,color,style in zip(CONDITIONS,LABELS,COLORS,STYLES):
                ax.plot(x,rows[c,actor,window]['values'],label=label,color=color,linestyle=style,linewidth=1.8,marker='o',markersize=3)
            ax.set_title(f'Actor {"ABC"[actor]} · Window {window+1}')
            ax.grid(axis='y',alpha=.2);ax.set_xlim(0,6000);ax.set_ylim(*entropy_ylim)
            if window==0:ax.set_ylabel('H(packet | background), bits')
            if actor==2:ax.set_xlabel('Training updates')
    fig.suptitle('Natural packet diversity — six separate actor / window panels',fontsize=14,y=.979)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=9)
    fig.text(.074,.021,'Each curve averages 16 initializations; each entropy conditions on all 36 public backgrounds. '
        'Silent policies still generate natural packets.\n'
        'These are descriptions of packet diversity, not semantic content or language. Raw equality, Hamming and global / within-background ARI remain in JSON.',
        fontsize=8,color='#444444',va='bottom')
    fig.subplots_adjust(left=.09,right=.98,bottom=.1,top=.88,wspace=.16,hspace=.32)
    for suffix in ('png','pdf'):fig.savefig(destination/f'02_message_entropy.{suffix}')
    plt.close(fig)


def execute(summary_path,output='figures_001'):
    summary_path=Path(summary_path).resolve();source_receipt=summary_path.parent/'receipt.json'
    receipt=read(source_receipt);require(receipt['status']=='passed','Passed summary receipt required')
    require(receipt['outputs_sha256'][str(summary_path)]==sha(summary_path),'Summary hash changed')
    destination=Path(output);destination=destination if destination.is_absolute() else summary_path.parent.parent/destination
    require(not destination.exists(),'Refusing overwrite')
    data=plotted_data(read(summary_path));destination.mkdir(parents=True,exist_ok=False)
    write(destination/'plot_data.json',data);render(data,destination)
    files=[destination/name for name in ('plot_data.json','01_formation_trajectory.png','01_formation_trajectory.pdf',
                                       '02_message_entropy.png','02_message_entropy.pdf')]
    out=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(p):sha(p) for p in (summary_path,source_receipt,Path(__file__).resolve(),HERE/'summarize.py')},
        outputs_sha256={str(p):sha(p) for p in files},model_forwards=0,npz_reads=0,
        scope='Two predetermined static figures from completed JSON. No fitted curves,world-level intervals,posthoc seed selection or semantic score.',
        visual_review='Pending actual PNG inspection; PDF files use the same figure objects but were not separately rasterized.')
    write(destination/'receipt.json',out)
    return dict(status='passed',output=str(destination),outputs_sha256=out['outputs_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summary',type=Path,default=HERE/'results/formation_001/summary_001/summary.json')
    parser.add_argument('--output',default='figures_001');args=parser.parse_args()
    print(json.dumps(execute(args.summary,args.output),ensure_ascii=False))
