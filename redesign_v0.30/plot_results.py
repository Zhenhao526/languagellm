"""Render the prespecified v0.30 support diagram and result figures."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import sys

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT));import support
sys.path.insert(0,str(PROJECT/'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch,Rectangle
from matplotlib.lines import Line2D

ARMS=support.ARMS
COLORS={'aligned_paths1':'#1B75BC','sparse_paths':'#D1612A'}
LABELS={'aligned_paths1':'每个目标1条三步路径','sparse_paths':'仅3/12目标有三步路径'}
MARKERS=('o','s','^','D')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(x):return 100*float(x)

def setup():
    font=font_manager.findfont('PingFang SC',fallback_to_default=False)
    plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'font.size':11,'axes.titlesize':13,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.unicode_minus':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    return font

def save(fig,directory,stem):
    out={}
    for ext in ('png','pdf'):
        p=directory/f'{stem}.{ext}';fig.savefig(p,dpi=200,bbox_inches='tight',pad_inches=.14);out[p.name]=sha(p)
    plt.close(fig);return out

def diagram(directory):
    design=read(ROOT/'support_design.json');target={tuple(x) for x in design['target12_pairs']}
    fig,axes=plt.subplots(1,2,figsize=(9.4,5.8));fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.19,wspace=.28)
    fig.suptitle('v0.30：多伙伴目标与两种训练支持',fontsize=17,y=.985)
    fig.text(.5,.925,'每个格子是“食物位置 × 水位置”；彩色格子进入训练，白色 T 格子是共同未训练目标',ha='center',fontsize=11)
    cells={}
    for ax,arm in zip(axes,ARMS):
        pairs={tuple(x) for x in design[f'{arm}_pairs']};paths=design['invariants'][f'{arm.split("_")[0]}_target_three_paths'] if arm.startswith('aligned') else design['invariants']['sparse_target_three_paths']
        for f in range(6):
            for w in range(6):
                if f==w: face,label,tc='#3D4249','×','white'
                elif (f,w) in pairs: face,label,tc=COLORS[arm],'','white'
                elif (f,w) in target: face,label,tc='white','T','#202932'
                else: face,label,tc='#E3E6EA','','#777'
                ax.add_patch(Rectangle((w-.5,f-.5),1,1,facecolor=face,edgecolor='white',linewidth=1.4))
                if label:ax.text(w,f,label,ha='center',va='center',fontsize=14,color=tc,weight='bold')
        ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5),xticks=range(6),yticks=range(6),xlabel='水位置',ylabel='食物位置');ax.set_aspect('equal');ax.tick_params(length=0)
        ax.set_title(f'{LABELS[arm]}\n目标路径分布：{paths}',color=COLORS[arm],fontsize=12.5,pad=9)
        cells[arm]={'training_pairs':sorted(map(list,pairs)),'target_pairs':sorted(map(list,target)),'path_distribution':paths}
    handles=[Patch(facecolor=COLORS[ARMS[0]],label=LABELS[ARMS[0]]),Patch(facecolor=COLORS[ARMS[1]],label=LABELS[ARMS[1]]),Patch(facecolor='white',edgecolor='#68717D',label='T：共同目标12'),Patch(facecolor='#E3E6EA',label='留出/未用布局')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=10.5)
    fig.text(.5,.015,'× 为不合法的同址布局；三步路径只描述训练支持图中的统计关系，不向主体显式提供。',ha='center',fontsize=10.5,color='#414954')
    return save(fig,directory,'01_support_design'),cells

def style(ax,title,ylabel='百分比'):
    ax.set_title(title,loc='left',pad=9);ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel(ylabel);ax.grid(axis='y',color='#C8CDD3',alpha=.55,linewidth=.7);ax.set_axisbelow(True)

def result_figure(directory,analysis):
    if analysis['status']!='complete' or not analysis['formal']:raise ValueError('formal independent analysis required')
    seeds=analysis['seeds'];rows={(r['seed'],r['condition']):r for r in analysis['seed_rows']};agg=analysis['aggregate'];times=[x['update'] for x in agg[ARMS[0]]['curve']]
    if set(rows)!={(s,a) for s in seeds for a in ARMS} or times!=[0,100,600,1200,2100,2400]:raise ValueError('incomplete formal matrix')
    fig,axes=plt.subplots(2,2,figsize=(12.1,9.8));fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.18,hspace=.43,wspace=.25)
    fig.suptitle('v0.30：共同目标的自然通信与多伙伴判据',fontsize=17,y=.99)
    fig.text(.5,.948,'所有4个初始化来源保留；细线为来源，粗线为来源均值；纵轴固定为0–100%',ha='center',fontsize=11.5)
    handles=[Line2D([0],[0],color=COLORS[a],marker='o',lw=2,label=LABELS[a]) for a in ARMS];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=11)
    a,b,c,d=axes.flat;evidence={'seeds':seeds,'times':times,'target_J':{},'pair_J':{},'endpoint_components':{},'recombination':{}}
    for arm in ARMS:
        vals=np.asarray([[pct(x['scores']['target12']['pooled']['J']) for x in rows[s,arm]['curve']] for s in seeds]);mean=np.asarray([pct(x['scores']['target12']['pooled']['J']) for x in agg[arm]['curve']])
        if not np.allclose(mean,vals.mean(0),atol=1e-9):raise ValueError('target curve aggregation mismatch')
        for v in vals:a.plot(times,v,color=COLORS[arm],alpha=.22,lw=.8)
        a.plot(times,mean,color=COLORS[arm],marker='o',ms=4,lw=2.2);evidence['target_J'][arm]={'source_percent':vals.tolist(),'mean_percent':mean.tolist()}
        vals2=np.asarray([[100*(x['scores']['target12']['pooled']['food_partner_pair_J']+x['scores']['target12']['pooled']['water_partner_pair_J'])/2 for x in rows[s,arm]['curve']] for s in seeds]);mean2=vals2.mean(0)
        for v in vals2:b.plot(times,v,color=COLORS[arm],alpha=.22,lw=.8)
        b.plot(times,mean2,color=COLORS[arm],marker='o',ms=4,lw=2.2);evidence['pair_J'][arm]={'source_percent':vals2.tolist(),'mean_percent':mean2.tolist()}
    style(a,'A  目标12：双资源自然成功 J','双资源成功率 J（%）');a.set_xlabel('共同通信更新次数');a.set_xlim(0,2400);a.set_xticks(times)
    style(b,'B  目标12：两伙伴同时区分','两伙伴同时正确率（%）');b.set_xlabel('共同通信更新次数');b.set_xlim(0,2400);b.set_xticks(times)
    endpoint_metrics=('J','food','water');x=np.arange(len(endpoint_metrics));width=.34
    for j,arm in enumerate(ARMS):
        vals=[]
        for k in endpoint_metrics:vals.append(pct(agg[arm]['scores']['target12']['pooled'][k]))
        pair=pct((agg[arm]['scores']['target12']['pooled']['food_partner_pair_J']+agg[arm]['scores']['target12']['pooled']['water_partner_pair_J'])/2)
        vals.append(pair);xx=np.arange(4)+(j-.5)*width;c.bar(xx,vals,width=width,color=COLORS[arm],alpha=.78,label=LABELS[arm]);evidence['endpoint_components'][arm]=dict(labels=['J','食物单边','水单边','伙伴对'],percent=vals)
    style(c,'C  终点12：自然结果分解','终点表现（%）');c.set_xticks(np.arange(4),['J','食物','水','伙伴对']);c.legend(frameon=False,fontsize=9,loc='upper right')
    for j,arm in enumerate(ARMS):
        evidence['recombination'][arm]={}
        for k,assignment in enumerate(('FW','WF')):
            metric=f'recombine_{assignment}_J';summary=agg[arm]['recombination_null']['pooled'][metric];xpos=k+(j-.5)*.36
            obs=pct(summary['observed']);null=pct(summary['null_mean']);d.plot([xpos,xpos],[null,obs],color=COLORS[arm],ls=':',lw=1.3);d.scatter(xpos,obs,color=COLORS[arm],s=75,edgecolor='white',zorder=3);d.scatter(xpos,null,color=COLORS[arm],marker='x',s=70,linewidth=2,zorder=4);evidence['recombination'][arm][assignment]={'observed_percent':obs,'null_mean_percent':null,'excess_percent':pct(summary['excess']),'upper_tail_fraction':summary['upper_tail_fraction']}
    style(d,'D  固定供体重组与整码重命名参照','目标12重组成功率（%）');d.set_xlim(-.45,1.45);d.set_xticks([0,1],['FW','WF']);d.set_xlabel('固定供体顺序；圆=原始，×=199次完整码参照');d.legend(handles=[Line2D([0],[0],marker='o',color='#555',lw=0,label='原始重组'),Line2D([0],[0],marker='x',color='#555',lw=0,label='整码重命名均值')],frameon=False,fontsize=9,loc='upper right')
    fig.text(.5,.105,'目标12中每个食物/水位置都有两个伙伴；伙伴对指标要求同一固定资源的两个目标边同时正确。',ha='center',fontsize=10.5)
    fig.text(.5,.065,'伙伴对成功率低于单边成功率时，单资源猜测不能被误读为共同符号已经稳定形成。',ha='center',fontsize=10.5)
    fig.text(.5,.025,'重组是离线固定干预，199项完整码双射是描述性参照，不是额外训练重复。',ha='center',fontsize=10.5,color='#414954')
    return save(fig,directory,'02_support_outcomes'),evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--diagram-only',action='store_true');ap.add_argument('--preview-dir',type=Path,default=ROOT/'figures_preview');args=ap.parse_args();font=setup()
    if args.diagram_only:
        directory=args.preview_dir.resolve();formal=ROOT/'results/support_001'
        if directory==formal or formal in directory.parents:ap.error('diagram-only cannot target formal output')
        directory.mkdir(parents=True,exist_ok=True);figures,cells=diagram(directory);record={'status':'diagram_preview','support_design_sha256':sha(ROOT/'support_design.json'),'plot_source_sha256':sha(__file__),'font_path':font,'figure_sha256':figures,'matrix_cells':cells,'model_results_read':False}
    else:
        if args.out is None:ap.error('--out is required');out=args.out.resolve()
        out=args.out.resolve()
        if not out.is_dir() or not (out/'analysis.json').is_file():ap.error('existing analyzed output required')
        analysis=read(out/'analysis.json');validation=read(out/'raw_validation.json')
        if not validation['passed'] or not analysis['formal'] or validation.get('analysis_sha256')!=sha(out/'analysis.json'):raise ValueError('independent formal analysis binding failed')
        directory=out/'figures';directory.mkdir(exist_ok=True);one,cells=diagram(directory);two,evidence=result_figure(directory,analysis);record={'status':'rendered_pending_visual_QA','support_design_sha256':sha(ROOT/'support_design.json'),'plot_source_sha256':sha(__file__),'analysis_sha256':sha(out/'analysis.json'),'raw_validation_sha256':sha(out/'raw_validation.json'),'font_path':font,'figure_sha256':{**one,**two},'matrix_cells':cells,'plotted_values':evidence,'axes_fixed_before_results':True,'source_and_condition_selection':False}
    write(directory/'figure_source.json',record);print(json.dumps({'status':record['status'],'directory':str(directory),'files':list(record['figure_sha256'])},ensure_ascii=False))

if __name__=='__main__':main()
