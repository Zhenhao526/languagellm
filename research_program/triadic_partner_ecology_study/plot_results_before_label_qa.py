"""Three static scientific figures from a complete weighted summary only.

No learner imports, network loading or training. Every seed and negative paired
contrast is retained. Figure display changes never change the saved estimands.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

SEEDS=(49101,49102,49103,49104)
ECOLOGIES=('unique','multiple')
CONDITIONS=('FI_silent','FI_live','PI_silent','PI_live')
PARTITIONS=('train','heldout_layouts')
CHECKPOINTS=(0,100,500,1500,3000,6000)
ECONAMES={'unique':'唯一兼容伙伴','multiple':'多个兼容伙伴'}
CONDNAMES=('完整·静默','完整·开放','局部·静默','局部·开放')
PARTNAMES={'train':'训练布局','heldout_layouts':'留出布局'}
METRICNAMES={'compatible_role_rate':'兼容角色率','full_success_rate':'任务满分率'}
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')
MARKERS=('o','s','^','D')
FONT_PATH=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')


def require(ok,message):
    if not ok:raise ValueError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def signed_display(value,digits):
    # Rounded zeros have no empirical direction; raw JSON values stay unchanged.
    return f'{0:.{digits}f}' if round(value,digits)==0 else f'{value:+.{digits}f}'

def write_new(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def key(x):return (x['seed'],x['ecology'],x['condition'],x['partition'],x['stage'],x['update'],x['mode'])


def validate(data):
    require(data['status']=='completed','Only a completed summary may be plotted')
    for name,values in [('seeds',SEEDS),('ecologies',ECOLOGIES),('conditions',CONDITIONS),('partitions',PARTITIONS),('checkpoints',CHECKPOINTS)]:
        require(data['contract'][name]==list(values),'Fixed contract mismatch: '+name)
    records=data['evaluations'];aliases=data['closed_aliases'];index={key(x):x for x in records}
    require(len(records)==len(index)==672 and len(aliases)==224,'Expected672 actual evaluations plus224 silent references')
    expected=set()
    for s in SEEDS:
        for e in ECOLOGIES:
            for c in CONDITIONS:
                for p in PARTITIONS:
                    for stage,updates in [('final',(6000,)),('monitor',CHECKPOINTS)]:
                        for u in updates:
                            expected.add((s,e,c,p,stage,u,'natural'))
                            if c.endswith('_live'):expected.add((s,e,c,p,stage,u,'closed'))
    require(set(index)==expected,'Incomplete actual grid')
    for alias in aliases:
        k=key(alias);require(k not in index and alias['condition'].endswith('_silent') and alias['mode']=='closed','Invalid silent alias')
        index[k]=index[(*k[:-1],'natural')]
    require(len(index)==896,'Incomplete logical grid')
    for row in records:
        for metric in METRICNAMES:
            value=row['weighted'][metric];require(math.isfinite(value) and -1e-12<=value<=1+1e-12,'Invalid probability')
        if row['stage']=='monitor':require(row['worlds']==672,'Incorrect monitor denominator')
    return index


def render(summary,out):
    source=Path(summary).resolve();out=Path(out).resolve();require(not out.exists(),'Refuse to overwrite figures')
    digest=sha(source);script_sha=sha(__file__);data=json.loads(source.read_text());index=validate(data)
    require(FONT_PATH.is_file(),'Chinese font missing')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    font_manager.fontManager.addfont(str(FONT_PATH));font=font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white'})
    out.mkdir(parents=True,exist_ok=False);files=[]
    def get(s,e,c,p,metric,mode='natural',stage='final',u=6000):
        return index[(s,e,c,p,stage,u,mode)]['weighted'][metric]*100
    def style(ax,metric):
        ax.set_ylim(0,100);ax.set_yticks((0,20,40,60,80,100));ax.set_ylabel(METRICNAMES[metric]+'（%）')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    def save(fig,name):
        for ext in ('png','svg'):
            p=out/f'{name}.{ext}';fig.savefig(p,dpi=180);files.append(p)
        plt.close(fig)
    try:
        # Full target distribution, not the fixed672 monitor subset.
        fig,axes=plt.subplots(2,4,figsize=(17,9.3))
        fig.subplots_adjust(left=.055,right=.985,top=.90,bottom=.17,hspace=.37,wspace=.28)
        fig.suptitle('完整加权终点：兼容角色与任务成功分别报告',fontsize=18)
        for ei,e in enumerate(ECOLOGIES):
            for pi,p in enumerate(PARTITIONS):
                for mi,m in enumerate(METRICNAMES):
                    ax=axes[ei,2*pi+mi]
                    for ci,c in enumerate(CONDITIONS):
                        vals=[get(s,e,c,p,m) for s in SEEDS]
                        for offset,marker,value in zip((-.12,-.04,.04,.12),MARKERS,vals):
                            ax.scatter(ci+offset,value,color=COLORS[ci],marker=marker,s=35,zorder=3,clip_on=False)
                        mean=sum(vals)/4;ax.plot((ci-.20,ci+.20),(mean,mean),color='#222222',lw=2.1,zorder=4)
                    gamma=index[(SEEDS[0],e,CONDITIONS[0],p,'final',6000,'natural')]['weighted']['best_fixed_pair_gamma']*100
                    ax.axhline(gamma,color='#555555',ls=':',lw=1.2)
                    ax.set_title(ECONAMES[e]+'／'+PARTNAMES[p],fontsize=10)
                    ax.set_xticks(range(4),CONDNAMES,rotation=15,ha='right',fontsize=8.5)
                    ax.set_xlim(-.5,3.5);style(ax,m)
        handles=[Line2D([],[],marker=m,color='#666666',linestyle='',label=str(s)) for s,m in zip(SEEDS,MARKERS)]
        handles += [Line2D([],[],color='#222222',lw=2,label='四种子均值'),Line2D([],[],color='#555555',ls=':',label='理想固定搭档上界')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.073),ncol=6,frameon=False)
        fig.text(.5,.025,'更新6000；每个目的地层等权。唯一／多个生态的固定搭档界分别为1/3、31/39。\n全部自然终点；完整信息与局部信息均保留，不以角色正确替代物理执行或任务成功。',ha='center',va='bottom',fontsize=9)
        save(fig,'complete_weighted_endpoints')

        # All four paired differences; zero/reference kept, no significance stars.
        fig,axes=plt.subplots(2,2,figsize=(12.8,9.2))
        fig.subplots_adjust(left=.095,right=.985,top=.88,bottom=.15,hspace=.38,wspace=.23)
        fig.suptitle('留出布局：生态对通信收益的影响',fontsize=18)
        differences={}
        for m in METRICNAMES:
            for info in ('PI','FI'):
                vals=[]
                for s in SEEDS:
                    gains={e:get(s,e,info+'_live','heldout_layouts',m)-get(s,e,info+'_silent','heldout_layouts',m) for e in ECOLOGIES}
                    vals.append(gains['unique']-gains['multiple'])
                differences[(m,info)]=vals
        maximum=max(abs(v) for values in differences.values() for v in values)
        bound=max(10,10*math.ceil(maximum*1.3/10))
        for mi,m in enumerate(METRICNAMES):
            for ii,info in enumerate(('PI','FI')):
                ax=axes[mi,ii];vals=differences[(m,info)];mean=sum(vals)/4
                ax.axhline(0,color='#555555',lw=1.2);ax.axhline(mean,color='#222222',ls='--',lw=1.2)
                for j,(s,value,color,marker) in enumerate(zip(SEEDS,vals,COLORS,MARKERS)):
                    ax.vlines(j,0,value,color=color,lw=2,alpha=.6);ax.scatter(j,value,color=color,marker=marker,s=48,zorder=3)
                    ax.annotate(signed_display(value,3),(j,value),xytext=(0,8 if value>=0 else -16),textcoords='offset points',ha='center',fontsize=10)
                qualifier='预定主比较' if (mi,ii)==(0,0) else '辅助比较'
                ax.set_title(f'{"局部" if info=="PI" else "完整"}信息·{METRICNAMES[m]}（{qualifier}）\n四配对均值 {signed_display(mean,4)} 个百分点',fontsize=11)
                ax.set_xticks(range(4),[str(s) for s in SEEDS]);ax.set_xlabel('配对团队初始化种子')
                ax.set_ylabel('差中差（百分点）');ax.set_ylim(-bound,bound);ax.set_xlim(-.5,3.5)
                ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        fig.text(.5,.035,'差中差＝唯一生态的（开放−静默）−多个生态的（开放−静默）。\n四个配对差全部保留；虚线为等权均值，无显著性检验。',ha='center',va='bottom',fontsize=9)
        save(fig,'paired_role_and_success')

        # Eight panels: every ecology×partition×metric. Thinseed lines; thickmean.
        fig,axes=plt.subplots(4,2,figsize=(14,15.5))
        fig.subplots_adjust(left=.07,right=.985,top=.93,bottom=.11,hspace=.40,wspace=.19)
        fig.suptitle('固定672世界的监测轨迹：自然与闭通道',fontsize=18)
        for ei,e in enumerate(ECOLOGIES):
            for pi,p in enumerate(PARTITIONS):
                row=ei*2+pi
                for mi,m in enumerate(METRICNAMES):
                    ax=axes[row,mi]
                    for ci,c in enumerate(CONDITIONS):
                        for mode in (('natural','closed') if c.endswith('_live') else ('natural',)):
                            ls='-' if mode=='natural' else '--'
                            curves=[[get(s,e,c,p,m,mode,'monitor',u) for u in CHECKPOINTS] for s in SEEDS]
                            for curve in curves:ax.plot(CHECKPOINTS,curve,color=COLORS[ci],ls=ls,lw=.75,alpha=.18,clip_on=False)
                            means=[sum(x[i] for x in curves)/4 for i in range(6)]
                            ax.plot(CHECKPOINTS,means,color=COLORS[ci],ls=ls,lw=2.0,marker='o' if mode=='natural' else None,ms=3,clip_on=False)
                    ax.set_title(ECONAMES[e]+'／'+PARTNAMES[p],fontsize=11)
                    ax.set_xlim(0,6000);ax.set_xticks((0,1500,3000,4500,6000));ax.set_xlabel('训练更新数');style(ax,m)
        handles=[Line2D([],[],color=color,lw=2,label=name) for color,name in zip(COLORS,CONDNAMES)]
        handles += [Line2D([],[],color='#444444',ls='-',label='自然路由'),Line2D([],[],color='#444444',ls='--',label='开放网络闭通道')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.063),ncol=6,frameon=False,fontsize=9)
        fig.text(.5,.022,'每个目的地层固定32世界；细线各代表一个种子，粗线是四种子均值。\n闭通道从第一窗重新计算；静默不重复前向。六个已保存时点的连线不定位精确形成时刻。\n672世界的监测不是全量目标分布终点，图中不使用监测样本的固定伙伴界冒充总体界。',ha='center',va='bottom',fontsize=9)
        save(fig,'fixed_monitor_curves')
        require(sha(source)==digest and sha(__file__)==script_sha,'Figure source changed during rendering')
        receipt=dict(status='rendered_pending_visual_review',created_at=datetime.now(timezone.utc).isoformat(),summary_path=str(source),summary_sha256=digest,
            script_sha256=script_sha,font_path=str(FONT_PATH),font_sha256=sha(FONT_PATH),matplotlib_version=matplotlib.__version__,
            files_sha256={p.name:sha(p) for p in files},actual_evaluations=672,rate_axis=[0,100],signed_difference_axis=[-bound,bound],
            independent_seeds=list(SEEDS),neural_forward_calls=0,training_updates=0)
        write_new(out/'receipt.json',receipt);return dict(status=receipt['status'],out=str(out),figures=[p.name for p in files])
    except BaseException as error:
        plt.close('all');write_new(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(render(args.summary,args.out),ensure_ascii=False))
