"""One two-panel static figure from the completed audited JSON summary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'flow_study_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(59101,59102,59103,59104)
TARGET='new_needs_and_layouts'
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')
ARM_ORDER=('S00','S10','S01','S11')
EFFECT_ORDER=('I_given_O0','I_given_O1','O_given_I0','O_given_I1','interaction')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def limits(values):
    lo=min(0.,float(np.min(values)));hi=max(0.,float(np.max(values)));padding=max((hi-lo)*.13,.01)
    return lo-padding,hi+padding


def draw(summary):
    part=next(p for p in summary['partition_summaries'] if p['partition']==TARGET)
    response=part['responses']['S'];rows=response['by_seed']
    require([r['seed'] for r in rows]==list(SEEDS),'All four seeds in canonical order required')
    require(response['means']['I_given_O0']==summary['primary']['mean_S10_minus_S00'],'Primary summary mismatch')
    arms=np.asarray([[r[k] for k in ARM_ORDER] for r in rows])*100
    effects=np.asarray([[r[k] for k in EFFECT_ORDER] for r in rows])*100
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1,1.25]})
    offsets=np.linspace(-.18,.18,4)
    for axis,values,keys in zip(axes,(arms,effects),(ARM_ORDER,EFFECT_ORDER)):
        for i,(seed,color) in enumerate(zip(SEEDS,COLORS)):
            axis.scatter(np.arange(len(keys))+offsets[i],values[i],s=35,color=color,alpha=.88,label=str(seed),zorder=4)
        means=np.asarray([response['means'][k] for k in keys])*100
        axis.scatter(np.arange(len(keys)),means,s=60,marker='D',color='#111111',label='Four-seed mean',zorder=6)
        axis.spines[['top','right']].set_visible(False);axis.set_axisbelow(True)
        axis.grid(axis='y',color='#e5e5e5',linewidth=.7);axis.axhline(0,color='#555555',linestyle='--',linewidth=.8)
        axis.set_ylim(*limits(values));axis.set_xlim(-.5,len(keys)-.5)
    axes[0].set_title('Four intervention cells',loc='left',fontsize=12,fontweight='bold')
    axes[0].set_xticks(range(4),['00\nNatural replay','10\nIncoming only','01\nOutgoing only','11\nBoth'])
    axes[0].set_ylabel('Exact full-success probability S (%)')
    axes[1].set_title('Paired effects on S',loc='left',fontsize=12,fontweight='bold')
    axes[1].axvspan(-.38,.38,color='#e7f0f8',alpha=.5,zorder=0)
    axes[1].set_xticks(range(5),['I | O=0\nPrimary','I | O=1','O | I=0','O | I=1','I × O\nInteraction'])
    axes[1].set_ylabel('Signed probability difference (percentage points)')
    axes[1].text(.98,.98,f'Primary mean: {summary["primary"]["mean_S10_minus_S00"]*100:+.4f} pp',
                 transform=axes[1].transAxes,ha='right',va='top',fontsize=9,
                 bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,frameon=False,loc='lower center',bbox_to_anchor=(.5,.115),fontsize=9)
    fig.suptitle('Same-background W1 message intervention: complete double holdout',fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.032,'S integrates actions conditional on each cell’s greedy messages. The natural reference uses the same repeated case weights.\nPoints show all four existing societies; diamonds are means, not confidence intervals. Incoming and outgoing interventions differ in their information sources.',
             ha='center',fontsize=8.5,color='#444444')
    fig.subplots_adjust(left=.065,right=.99,bottom=.29,top=.85,wspace=.24)
    return fig


def execute(run,summary_folder='summary_001',output='figures_001'):
    run=Path(run).resolve();source_folder=Path(summary_folder);source_folder=source_folder if source_folder.is_absolute() else run/source_folder
    destination=Path(output);destination=destination if destination.is_absolute() else run/destination
    require(not destination.exists(),'Refusing to overwrite figures')
    summary_path=source_folder/'summary.json';receipt_path=source_folder/'receipt.json';main_path=run/'execution/results.json'
    summary=read(summary_path);receipt=read(receipt_path);main=read(main_path)
    require(main['status']=='completed' and summary['status']=='completed_json_only_summary' and summary['audit_status']=='passed',
            'Completed independently audited summary required')
    require(receipt['status']=='passed' and receipt['outputs_sha256'][str(summary_path)]==sha(summary_path) and
            receipt['inputs_sha256'][str(main_path)]==sha(main_path),'Summary/main hash mismatch')
    require(summary['primary_exact_match'] and summary['primary']==main['primary'],'Unique primary mismatch')
    require(summary['counts']['seed_partition_arms']==64 and summary['counts']['independent_societies']==4,'Incomplete fixed grid')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    figure=draw(summary);destination.mkdir(parents=True,exist_ok=False);outputs={}
    for extension in ('png','pdf'):
        path=destination/f'01_full_success_flow.{extension}'
        figure.savefig(path,dpi=180,facecolor='white',metadata={'Title':'Four-cell first-window flow intervention'} if extension=='pdf' else None)
        outputs[str(path)]=sha(path)
    plt.close(figure)
    receipt=dict(status='generated_awaiting_visual_review',at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(p):sha(p) for p in (summary_path,receipt_path,main_path,HERE/'plot_results.py')},
        outputs_sha256=outputs,figure_count=1,panels=2,png_count=1,pdf_count=1,npz_reads=0,model_forwards=0,
        scope='All four society seeds on complete double holdout. Other domains, nine strata, reward, execution and semantic responses remain in summary JSON.',
        units='Left: probability percentages; right: signed probability differences in percentage points. No inferred intervals or world-level uncertainty bars.',
        visual_review='Pending actual PNG inspection; PDF not separately rasterized.')
    with (destination/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return dict(status=receipt['status'],output=str(destination),outputs_sha256=outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/flow_001')
    parser.add_argument('--summary',default='summary_001');parser.add_argument('--output',default='figures_001');args=parser.parse_args()
    print(json.dumps(execute(args.run,args.summary,args.output),ensure_ascii=False))
