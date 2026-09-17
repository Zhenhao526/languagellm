"""Two fixed scientific figures from the completed JSON-only analysis.

No NPZ, checkpoint or neural model access. All four seeds remain visible.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'directed_message_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(59101,59102,59103,59104)
TARGET='new_needs_and_layouts'
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')
AXES=('kind','length','destination')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def limits(values):
    low=min(0.,float(np.min(values)));high=max(0.,float(np.max(values)))
    padding=max((high-low)*.15,.005)
    return low-padding,high+padding


def style(axis):
    axis.spines[['top','right']].set_visible(False)
    axis.axhline(0,color='#555555',linewidth=.8,linestyle='--',zorder=0)
    axis.grid(axis='y',color='#e5e5e5',linewidth=.7)
    axis.set_axisbelow(True)


def draw_checkpoints(summary):
    selected={(r['checkpoint'],r['partition']):r for r in summary['time_partition_summaries']}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7));offsets=np.linspace(-.055,.055,4)
    for ax,key,title in zip(axes,('L','D'),('L: final value is the primary estimand','D: auxiliary mean contrast')):
        values=[]
        for i,(seed,color) in enumerate(zip(SEEDS,COLORS)):
            y=[]
            for step in (0,6000):
                rows=selected[step,TARGET]['by_seed'];require([r['seed'] for r in rows]==list(SEEDS),'Seed order changed')
                y.append(rows[i][key]*100)
            values.extend(y)
            ax.plot(np.array([0,1])+offsets[i],y,'o-',color=color,markersize=5,linewidth=1.25,label=str(seed),alpha=.85)
        means=[selected[t,TARGET]['means'][key]*100 for t in (0,6000)]
        ax.plot([0,1],means,'D-',color='#111111',markersize=7,linewidth=2,label='Four-seed mean',zorder=6)
        ax.set_title(title,fontsize=11,fontweight='bold',loc='left')
        ax.set_xticks([0,1],['Initial (step 0)','Final (step 6000)'])
        ax.set_xlim(-.18,1.18);ax.set_ylim(*limits(values));ax.set_ylabel('Proposal probability contrast (percentage points)')
        ax.text(.98,.98,f'Final mean: {means[1]:+.4f} pp',transform=ax.transAxes,ha='right',va='top',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
        style(ax)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.10),fontsize=9)
    fig.suptitle('Complete double holdout: first-window message intervention',fontsize=13,fontweight='bold',y=.99)
    fig.text(.5,.02,'L takes the minimum of four context/recipient effects after averaging sender-own endpoints.\nD averages those four effects. Lines connect the same policy seed; four independent societies.',ha='center',fontsize=8.5,color='#444444')
    fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.86,wspace=.28)
    return fig


def draw_strata(summary):
    rows=[r for r in summary['all_time_partition_strata'] if r['checkpoint']==6000 and r['partition']==TARGET]
    require([(r['sender'],r['axis_index']) for r in rows]==[(s,a) for s in range(3) for a in range(3)],'Nine strata order changed')
    lookup={(r['sender'],r['axis_index']):r for r in rows}
    values=[s['L']*100 for r in rows for s in r['by_seed']];bounds=limits(values)
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.8),sharey=True);offsets=np.linspace(-.18,.18,4)
    for sender,ax in enumerate(axes):
        for axis in range(3):
            row=lookup[sender,axis]
            require([s['seed'] for s in row['by_seed']]==list(SEEDS),'Stratum seed order changed')
            for i,(seed,color) in enumerate(zip(SEEDS,COLORS)):
                ax.scatter(axis+offsets[i],row['by_seed'][i]['L']*100,s=29,color=color,alpha=.85,label=str(seed) if axis==0 else None,zorder=3)
            ax.scatter(axis,row['means']['L']*100,s=48,color='#111111',marker='D',label='Four-seed mean' if axis==0 else None,zorder=5)
        ax.set_xticks(range(3),['Kind','Length','Destination']);ax.set_xlim(-.5,2.5);ax.set_ylim(*bounds)
        ax.set_title(f'Sender {chr(65+sender)}',loc='left',fontsize=11,fontweight='bold');style(ax)
    axes[0].set_ylabel('L (percentage points)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.08),fontsize=9)
    fig.suptitle('Final complete double holdout: all nine sender / need-axis strata',fontsize=13,fontweight='bold',y=.99)
    fig.text(.5,.025,'Predefined auxiliary strata, with equal weight in the primary overall L. Points show every seed; diamonds show four-seed means.\nNo group, background, seed or stratum was selected by its outcome.',ha='center',fontsize=8.5,color='#444444')
    fig.subplots_adjust(left=.07,right=.99,bottom=.24,top=.85,wspace=.12)
    return fig


def execute(run,analysis='analysis_001',output='figures_001'):
    run=Path(run).resolve();folder=Path(analysis);folder=folder if folder.is_absolute() else run/folder
    destination=Path(output);destination=destination if destination.is_absolute() else run/destination
    require(not destination.exists(),'Refusing to overwrite figures')
    source=folder/'summary.json';receipt_path=folder/'receipt.json';result_path=run/'execution/results.json'
    summary=read(source);receipt=read(receipt_path);result=read(result_path)
    require(result.get('status')=='completed' and summary.get('status')=='completed_json_only_summary','Only completed data may be plotted')
    require(receipt['status']=='passed' and receipt['outputs_sha256'][str(source)]==sha(source),'Analysis hash mismatch')
    require(receipt['inputs_sha256'][str(result_path)]==sha(result_path),'Main result hash mismatch')
    require(summary['primary_exact_match'] and summary['primary']==result['primary'],'Primary mismatch')
    require(summary['counts']['partition_records']==32 and summary['counts']['independent_societies']==4,'Full fixed grid required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    destination.mkdir(parents=True,exist_ok=False);outputs={}
    for name,figure in (('01_checkpoint_L_D',draw_checkpoints(summary)),('02_final_nine_strata_L',draw_strata(summary))):
        for extension in ('png','pdf'):
            path=destination/f'{name}.{extension}';require(not path.exists(),'Refusing figure overwrite')
            figure.savefig(path,dpi=180,facecolor='white',metadata={'Title':name} if extension=='pdf' else None)
            outputs[str(path)]=sha(path)
        plt.close(figure)
    record=dict(status='generated_awaiting_visual_review',at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(p):sha(p) for p in (source,receipt_path,result_path,HERE/'plot_results.py')},outputs_sha256=outputs,
        figure_count=2,png_count=2,pdf_count=2,primary_exact_match=True,npz_reads=0,model_forwards=0,
        scale='Original probability contrasts multiplied by 100 and labeled percentage points, not percent success.',
        scope='Fixed complete double holdout only; all four seeds, both checkpoints, all nine final strata. Other partitions retained in source summary.',
        visual_review='Pending actual PNG inspection; PDF not separately rasterized.')
    with (destination/'receipt.json').open('x') as stream:json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return dict(status=record['status'],output=str(destination),outputs=outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/directed_001')
    parser.add_argument('--analysis',default='analysis_001');parser.add_argument('--output',default='figures_001');args=parser.parse_args()
    print(json.dumps(execute(args.run,args.analysis,args.output),ensure_ascii=False))
