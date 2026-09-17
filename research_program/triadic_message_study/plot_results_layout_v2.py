"""Display-only layout revision of plot_results.py; same fixed data and metrics.

No weights, forward passes, partial results, or best-seed selection. Use the
project's .plotting_venv. The CLI only reads a completed formation summary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SEEDS=(47101,47102,47103,47104)
CONDITIONS=("FI_silent","FI_live","PI_silent","PI_live")
PARTITIONS=("train","new_needs","new_layouts","new_needs_and_layouts")
CHECKPOINTS=(0,100,500,1500,3000,6000)
COUNTS=dict(zip(PARTITIONS,(82836,24732,27612,8244)))
CONDITION_NAMES=("完整·静默","完整·开放","局部·静默","局部·开放")
PARTITION_NAMES=("训练需求／训练布局","新需求／训练布局","训练需求／新布局","新需求／新布局")
COLORS=("#0072B2","#D55E00","#009E73","#CC79A7")
MARKERS=("o","s","^","D")
FONT_PATH=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def validate_summary(data):
    """No plotting or filesystem access; checks every expected endpoint/monitor."""
    require(data['status']=='completed','Only a completed descriptive summary can be plotted')
    require(data['contract']['seeds']==list(SEEDS) and data['contract']['conditions']==list(CONDITIONS),
            'Wrong fixed study conditions/seeds')
    rows=data['evaluation_records']
    keys=[(x['seed'],x['condition'],x['partition'],x['stage'],x['update']) for x in rows]
    expected={(s,c,p,'monitor',u) for s in SEEDS for c in CONDITIONS for p in PARTITIONS for u in CHECKPOINTS}
    expected|={(s,c,p,'final',6000) for s in SEEDS for c in CONDITIONS for p in PARTITIONS}
    require(len(rows)==448 and len(set(keys))==448 and set(keys)==expected,'All384 monitor and64 final records required')
    index=dict(zip(keys,rows))
    for row in rows:
        n=1024 if row['stage']=='monitor' else COUNTS[row['partition']]
        counts=row['greedy_reward_counts']
        require(row['worlds']==n and set(counts)=={'0.0','0.5','1.0'} and sum(counts.values())==n,
                'Wrong monitor/final population or native reward counts')
        require(all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in counts.values()),'Invalid reward counts')
        require(row['greedy_full_success_rate']==counts['1.0']/n and
                row['greedy_reward_mean']==(counts['0.5']*.5+counts['1.0'])/n,'Stored score/count mismatch')
    primary=data['primary_comparison']
    require(primary['partition']=='new_needs_and_layouts' and primary['endpoint_update']==6000,
            'Wrong prespecified primary endpoint')
    require([x['seed'] for x in primary['seed_pairs']]==list(SEEDS),'Primary paired seed order changed')
    differences=[]
    for row in primary['seed_pairs']:
        rates={c:index[(row['seed'],c,'new_needs_and_layouts','final',6000)]['greedy_full_success_rate'] for c in CONDITIONS}
        pi=rates['PI_live']-rates['PI_silent'];fi=rates['FI_live']-rates['FI_silent']
        require(row['full_success_rates']==rates and row['PI_live_minus_silent']==pi
                and row['FI_live_minus_silent']==fi and row['difference_in_differences_PI_minus_FI']==pi-fi,
                'Primary paired contrast differs from endpoint worlds')
        differences.append(pi)
    require(primary['equal_weight_means']['PI_live_minus_silent']==sum(differences)/4,'Primary mean differs')
    return index,differences


def render(summary,output):
    source=Path(summary).resolve();output=Path(output).resolve()
    require(not output.exists(),'Refuse to overwrite figure outputs')
    source_sha=sha(source);script_sha=sha(__file__)
    data=json.loads(source.read_text());index,differences=validate_summary(data)
    require(FONT_PATH.is_file(),'Chinese figure font missing')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    font_manager.fontManager.addfont(str(FONT_PATH))
    family=font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.unicode_minus':False,
                         'svg.fonttype':'path','savefig.facecolor':'white'})
    output.mkdir(parents=True,exist_ok=False)
    figures=[]
    def score(seed,condition,partition,stage='final',update=6000):
        return 100*index[(seed,condition,partition,stage,update)]['greedy_full_success_rate']
    def style(ax):
        ax.set_ylim(0,100);ax.set_yticks((0,20,40,60,80,100));ax.set_ylabel('满分率（%）')
        ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    def save(fig,name):
        for ext in ('png','svg'):
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=190);figures.append(path)
        plt.close(fig)
    try:
        fig,axes=plt.subplots(2,2,figsize=(12.6,9.2))
        fig.subplots_adjust(left=.08,right=.985,top=.88,bottom=.17,hspace=.36,wspace=.2)
        fig.suptitle('三人符号共同学习：四条件完整终点',fontsize=17)
        offsets=(-.12,-.04,.04,.12)
        for p,name,ax in zip(PARTITIONS,PARTITION_NAMES,axes.flat):
            for j,c in enumerate(CONDITIONS):
                values=[score(s,c,p) for s in SEEDS]
                for v,offset,marker in zip(values,offsets,MARKERS):
                    ax.scatter(j+offset,v,s=37,marker=marker,color=COLORS[j],alpha=.9,zorder=3,clip_on=False)
                mean=sum(values)/4
                ax.plot([j-.2,j+.2],[mean,mean],color='#222222',linewidth=2.2,zorder=4)
                ax.annotate(f'{mean:.2f}%',(j,mean),xytext=(0,9 if mean<=92 else -16),textcoords='offset points',
                            ha='center',fontsize=9,color='#222222')
            ax.set_title(f'{name}（每运行{COUNTS[p]:,}世界）',fontsize=11)
            ax.set_xticks(range(4),CONDITION_NAMES);ax.set_xlim(-.5,3.5);style(ax)
        fig.legend(handles=[Line2D([],[],color='#666666',marker=m,linestyle='',label=str(s)) for s,m in zip(SEEDS,MARKERS)]
                           +[Line2D([],[],color='#222222',linewidth=2.2,label='四种子均值')],
                   loc='lower center',bbox_to_anchor=(.5,.067),ncol=5,frameon=False)
        fig.supxlabel('全部使用更新6000的完整四格；4个配对团队初始化、同一开发划分，无显著性检验。',fontsize=9,y=.016)
        save(fig,'complete_endpoints')

        fig,axes=plt.subplots(1,2,figsize=(12.6,5.9),layout='constrained')
        fig.suptitle('预定主比较：新需求／新布局的局部信息通信作用',fontsize=16)
        for seed,color,marker in zip(SEEDS,COLORS,MARKERS):
            axes[0].plot((0,1),[score(seed,'PI_silent','new_needs_and_layouts'),score(seed,'PI_live','new_needs_and_layouts')],
                         marker=marker,color=color,linewidth=1.6,markersize=6,label=str(seed),clip_on=False)
        axes[0].set_xticks((0,1),('局部·静默','局部·开放'));axes[0].set_xlim(-.25,1.25);style(axes[0])
        axes[0].set_title('每种子每条件完整8,244世界');axes[0].legend(title='配对种子',frameon=False)
        delta=[100*x for x in differences]
        axes[1].axhline(0,color='#555555',linewidth=1)
        for j,(value,color,marker) in enumerate(zip(delta,COLORS,MARKERS)):
            axes[1].vlines(j,0,value,color=color,alpha=.65,linewidth=2)
            axes[1].scatter(j,value,color=color,marker=marker,s=48,zorder=3)
            axes[1].annotate(f'{value:+.3f}',(j,value),xytext=(0,8 if value>=0 else -16),textcoords='offset points',ha='center',fontsize=10)
        mean=sum(delta)/4;axes[1].axhline(mean,color='#222222',linestyle='--',linewidth=1.2)
        axes[1].set_title(f'开放−静默：四配对均值 {mean:+.4f} 个百分点')
        axes[1].set_xticks(range(4),[str(s) for s in SEEDS]);axes[1].set_xlim(-.5,3.5)
        axes[1].set_ylim(-100,100);axes[1].set_yticks((-100,-50,0,50,100));axes[1].set_ylabel('满分率配对差（百分点）')
        axes[1].grid(axis='y',alpha=.22);axes[1].spines[['top','right']].set_visible(False)
        fig.supxlabel('四个配对差全部保留；负差同样展示。终点通道删除和内容移植由独立分析报告。',fontsize=9)
        save(fig,'paired_primary')

        fig,axes=plt.subplots(2,2,figsize=(12.6,9.2))
        fig.subplots_adjust(left=.08,right=.985,top=.88,bottom=.17,hspace=.36,wspace=.2)
        fig.suptitle('固定监测曲线：各格均为同一组1,024世界',fontsize=17)
        for p,name,ax in zip(PARTITIONS,PARTITION_NAMES,axes.flat):
            for c,color in zip(CONDITIONS,COLORS):
                curves=[[score(s,c,p,'monitor',u) for u in CHECKPOINTS] for s in SEEDS]
                for curve in curves:ax.plot(CHECKPOINTS,curve,color=color,alpha=.23,linewidth=.9,clip_on=False)
                mean=[sum(row[k] for row in curves)/4 for k in range(len(CHECKPOINTS))]
                ax.plot(CHECKPOINTS,mean,color=color,linewidth=2.2,marker='o',markersize=3.2,clip_on=False)
            ax.set_title(name,fontsize=11);style(ax);ax.set_xlabel('训练更新数')
            ax.set_xlim(0,6000);ax.set_xticks((0,1500,3000,4500,6000))
        fig.legend(handles=[Line2D([],[],color=color,linewidth=2.2,label=name) for color,name in zip(COLORS,CONDITION_NAMES)],
                   loc='lower center',bbox_to_anchor=(.5,.067),ncol=4,frameon=False)
        fig.supxlabel('细线：各种子；粗线：四种子均值。仅0／100／500／1500／3000／6000六点；连线不定位形成时刻。\n'
                      '监测子集与完整终点分母不同，不能把更新6000的监测值替代完整评价。',fontsize=9,y=.016)
        save(fig,'fixed_monitor_curves')
        require(sha(source)==source_sha and sha(__file__)==script_sha,'Plot inputs/source changed while rendering')
        receipt={'status':'rendered_pending_visual_review','created_at':datetime.now(timezone.utc).isoformat(),
                 'summary_path':str(source),'summary_sha256':source_sha,'script_sha256':script_sha,
                 'font_path':str(FONT_PATH),'font_sha256':sha(FONT_PATH),'matplotlib_version':matplotlib.__version__,
                 'figures_sha256':{p.name:sha(p) for p in figures},'endpoint_records':64,'monitor_records':384,
                 'paired_seeds':list(SEEDS),'rate_axis_limits_percent':[0,100],'difference_axis_limits_pp':[-100,100],
                 'best_seed_or_checkpoint_selection':False,'neural_forward_calls':0,'training_updates':0}
        write_new(output/'receipt.json',receipt)
        return {'status':receipt['status'],'output':str(output),'figures':[p.name for p in figures]}
    except BaseException as error:
        plt.close('all')
        write_new(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(render(args.summary,args.out),ensure_ascii=False))
