"""Three scientific figures from the completed semantic-summary JSON only."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

SEEDS=(49101,49102,49103,49104)
STEPS=(0,100,500,1500,3000,6000)
AXES=('kind_wood_fiber','length_short_long','destination_L_R')
WINDOWS=('w1','w2','both')
CONDITIONS=('FI_silent','FI_live','PI_silent','PI_live')
LABELS={'kind_wood_fiber':'对象种类','length_short_long':'长短属性','destination_L_R':'目的地','macro':'三轴等权'}
FONT=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def figure_data(summary):
    """Pure reshaping only: keep all four seeds and negative values."""
    assert summary['status']=='completed_read_only_summary'
    records=summary['compact_records']
    natural={(r['seed'],r['condition'],r['checkpoint'],r['partition']):r for r in records if r['kind']=='natural'}
    interventions={(r['seed'],r['condition'],r['partition'],r['mode']):r for r in records if r['kind']=='intervention'}
    remote={(r['seed'],r['condition'],r['partition'],r['mode']):r for r in records if r['kind']=='remote_contrast'}
    assert len(natural)==192 and len(interventions)==160 and len(remote)==48
    trajectory={p:{c:[[natural[s,c,t,p]['content']['macro']['both_endpoints_apt'] for t in STEPS] for s in SEEDS]
        for c in CONDITIONS} for p in ('train','heldout_layouts')}
    endpoint={a:{c:[(natural[s,c,6000,'heldout_layouts']['content']['macro']['both_endpoints_apt'] if a=='macro' else
        natural[s,c,6000,'heldout_layouts']['content']['by_axis'][a]['means']['both_endpoints_apt']) for s in SEEDS]
        for c in ('PI_silent','PI_live')} for a in (*AXES,'macro')}
    transfer={w:{c:dict(same=[interventions[s,c,'heldout_layouts','remote_same_'+w]['content']['macro']['direction_mean_counterfactual_apt'] for s in SEEDS],
        opposite=[interventions[s,c,'heldout_layouts','remote_opposite_'+w]['content']['macro']['direction_mean_counterfactual_apt'] for s in SEEDS],
        difference=[remote[s,c,'heldout_layouts',w]['content']['macro']['direction_mean_target_apt_opposite_minus_same'] for s in SEEDS])
        for c in ('PI_silent','PI_live')} for w in WINDOWS}
    return dict(trajectory=trajectory,endpoint=endpoint,transfer=transfer)


def plot(summary_path,output):
    summary_path,output=Path(summary_path).resolve(),Path(output).resolve()
    assert not output.exists(),'No figure overwrite'
    summary=json.loads(summary_path.read_text());data=figure_data(summary)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    assert FONT.is_file(),'Required Chinese font missing; refuse silent fallback'
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':180,'figure.dpi':120,'axes.titleweight':'normal'})
    output.mkdir(parents=True);saved=[]
    def finish(fig,name):
        fig.savefig(output/(name+'.png'),bbox_inches='tight')
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
        for suffix in ('.png','.pdf'):saved.append(dict(path=name+suffix,sha256=sha(output/(name+suffix))))
    fig,axs=plt.subplots(2,2,figsize=(12.8,8.4),sharex=True,sharey=True)
    colors={'silent':'#72757b','live':'#12679c'}
    for i,info in enumerate(('FI','PI')):
        for j,part in enumerate(('train','heldout_layouts')):
            ax=axs[i,j]
            for visibility in ('silent','live'):
                y=np.asarray(data['trajectory'][part][info+'_'+visibility])
                for row in y:ax.plot(STEPS,row,color=colors[visibility],alpha=.22,lw=.8)
                ax.plot(STEPS,y.mean(0),color=colors[visibility],marker='o',ms=4,lw=2,
                    label=('从头静默' if visibility=='silent' else '开放消息')+'：四种子均值')
            ax.axhline(1/3,color='#a27436',ls='--',lw=1,label='固定一对搭档的1/3参照')
            ax.set_title(('完整信息FI' if info=='FI' else '局部信息PI')+' · '+('训练布局' if part=='train' else '留出recipient布局'))
            ax.set_ylim(-.015,1.02);ax.set_xlim(-90,6090);ax.grid(axis='y',alpha=.15)
            ax.set_xticks(STEPS);ax.tick_params(axis='x',labelrotation=60,labelsize=8)
            if j==0:ax.set_ylabel('自然两端均适切比例\ncontent三轴等权')
            if i==1:ax.set_xlabel('保存检查点（更新次数）')
    axs[0,0].legend(fontsize=8,loc='upper right')
    fig.suptitle('自然内容适切性的六个保存时点',fontsize=15,y=.985)
    fig.text(.5,.015,'细线为全部四种子，粗线为等权均值；不是精确产生时刻。6000步引用旧完整终点。PI静默零是结构参照。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.96));finish(fig,'natural_content_trajectory')

    fig,axs=plt.subplots(2,4,figsize=(14.6,7.0),sharey='row')
    x=np.arange(4);labels=[str(s) for s in SEEDS]
    all_deltas=[]
    for j,axis in enumerate((*AXES,'macro')):
        values=data['endpoint'][axis];s=np.asarray(values['PI_silent']);l=np.asarray(values['PI_live']);delta=l-s;all_deltas.extend(delta)
        ax=axs[0,j]
        for i in range(4):ax.plot([i-.07,i+.07],[s[i],l[i]],color='#b6c0c7',lw=1)
        ax.scatter(x-.07,s,color=colors['silent'],marker='o',facecolors='none',label='PI静默',s=35)
        ax.scatter(x+.07,l,color=colors['live'],label='PI开放',s=35)
        ax.axhline(1/3,color='#a27436',ls='--',lw=1);ax.set_ylim(-.02,1.02)
        ax.set_title(LABELS[axis]);ax.grid(axis='y',alpha=.15)
        axs[1,j].axhline(0,color='#777',lw=.8)
        axs[1,j].vlines(x,0,delta*100,color=colors['live'],lw=1.2)
        axs[1,j].scatter(x,delta*100,color=colors['live'],s=35)
        for row in (0,1):axs[row,j].set_xticks(x,labels,rotation=35);axs[row,j].grid(axis='y',alpha=.15)
    delta_limit=max(.1,max(abs(float(v)) for v in all_deltas)*1.15)*100
    for ax in axs[1]:ax.set_ylim(-delta_limit,delta_limit)
    axs[0,0].set_ylabel('自然两端均适切比例');axs[1,0].set_ylabel('PI开放−静默（百分点）')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('6000步留出recipient：全部四种子与三个内容轴',fontsize=15,y=.99)
    fig.text(.5,.015,'主要量为最右侧三轴等权差；其余轴全部保留。虚线1/3只参照固定搭档政策，不是一般政策上界。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.96));finish(fig,'endpoint_content_all_seeds')

    fig,axs=plt.subplots(2,3,figsize=(13.5,7.6),sharey='row');difference=[]
    for j,window in enumerate(WINDOWS):
        for condition,color,offset in (('PI_silent','#72757b',-.12),('PI_live','#12679c',.12)):
            value=data['transfer'][window][condition];same=np.asarray(value['same']);opp=np.asarray(value['opposite']);delta=np.asarray(value['difference']);difference.extend(delta)
            prefix='静默' if condition=='PI_silent' else '开放'
            axs[0,j].plot(x+offset,same,color=color,ls=':',lw=1,marker='o',mfc='white',ms=5,label=prefix+'·同需求供体')
            axs[0,j].plot(x+offset,opp,color=color,lw=1,marker='o',ms=5,label=prefix+'·反需求供体')
            axs[1,j].plot(x+offset,delta*100,color=color,lw=1,marker='o',ms=5,label=prefix)
        axs[0,j].set_title({'w1':'只换第一窗','w2':'只换第二窗','both':'两窗替换（预定辅助）'}[window])
        axs[1,j].axhline(0,color='#777',lw=.8)
        for row in (0,1):axs[row,j].set_xticks(x,labels,rotation=35);axs[row,j].grid(axis='y',alpha=.15)
    axs[0,0].set_ylim(-.015,1.02);limit=max(.10,max(abs(float(v)) for v in difference)*1.15)*100
    axs[1,0].set_ylim(-limit,limit)
    axs[0,0].set_ylabel('反事实需求适切比例\n三轴等权、方向配对均值')
    axs[1,0].set_ylabel('反需求供体−同需求供体\n（百分点）')
    axs[0,0].legend(fontsize=7,loc='upper right',ncol=1)
    fig.suptitle('远背景消息：绝对适切率与同／反需求配对差',fontsize=15,y=.985)
    fig.text(.5,.02,'6000步、留出仅指recipient；5/6远背景供体来自训练布局。静默两模式重合是结构检查；正差不等于真实团队回报提高。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.955));finish(fig,'remote_transfer_all_windows')
    (output/'plot_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    receipt=dict(status='plotted_requires_visual_review',created_at=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        plot_source_sha256=sha(__file__),summary_path=str(summary_path),summary_sha256=sha(summary_path),
        font_path=str(FONT) if FONT.is_file() else None,font_sha256=sha(FONT) if FONT.is_file() else None,
        files=saved,plot_data_sha256=sha(output/'plot_data.json'),neural_forwards=0,training=0)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(plot(args.summary,args.out),ensure_ascii=False))
