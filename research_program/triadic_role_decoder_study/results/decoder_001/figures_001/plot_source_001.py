"""Plot completed role-decoder aggregate JSON only; never load model artifacts.

Use research_program/.plotting_venv/bin/python -m
research_program.triadic_role_decoder_study.plot_results --run RUN_DIRECTORY.
The caller performs subsequent visual inspection. This file does not imply an
independent execution audit, model replay, or confidence interval estimation.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import re

import numpy as np

OLD_SEEDS=(53101,53102,53103,53104)
PROBE_SEEDS=(55101,55102,55103)
PAYOFFS=('a50','a10')
STEPS=(0,100,500,1500,3000,6000)
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
TARGET='new_needs_and_layouts'
WORLD_COUNTS=dict(zip(PARTS,(419904,160704,139968,53568)))
MONITOR_COUNTS=dict(zip(PARTS,(7776,2976,7776,2976)))
BOUND=23/62
ACTION_REFERENCE=1/3
SEED_COLORS=('#246B8E','#B15D3B','#597F4E','#8B6195')
PROBE_MARKERS=('o','s','^')
VIEW_COLORS={'Own':'#92999E','FI':'#483E63','Initial':'#8C6E44',
             'Final_a50':'#34799C','Final_a10':'#BA5D43'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def close(actual,expected,label):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=2e-12,rtol=0),'Numeric mismatch: '+label)


def compare(actual,expected,label=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'Mapping schema: '+label)
        for key,value in expected.items():compare(actual[key],value,label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'List schema: '+label)
        for index,value in enumerate(expected):compare(actual[index],value,label+'/'+str(index))
    elif isinstance(expected,float):close(actual,expected,label)
    else:require(actual==expected,'Value mismatch: '+label)


def number(value,label):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Nonfinite number: '+label)
    return float(value)


def verify_evaluation(row,worlds):
    require(type(row['worlds']) is int and row['worlds']==worlds,'Evaluation world denominator')
    for key in ('joint_accuracy','indiv_accuracy','expected_joint_correct_probability'):
        value=number(row[key],key);require(0<=value<=1,'Probability range: '+key)
    require(number(row['cross_entropy'],'cross_entropy')>=0,'Negative cross entropy')
    actors=row['per_actor_accuracy']
    require(len(actors)==3 and all(0<=number(v,'actor accuracy')<=1 for v in actors),'Per-actor accuracy')
    close(row['indiv_accuracy'],np.mean(actors),'Per-actor accuracy mean')
    require(row['joint_accuracy']<=min(actors)+2e-12,'Joint accuracy exceeds individual accuracy')
    require(abs(row['joint_accuracy']*worlds-round(row['joint_accuracy']*worlds))<1e-6,'Joint accuracy count')
    require(set(row['true_pair_strata'])=={'AB','AC','BC'},'True-pair strata')
    total=0;correct=0.
    for pair,value in row['true_pair_strata'].items():
        require(type(value['worlds']) is int and value['worlds']>0,'Nonempty pair stratum '+pair)
        accuracy=number(value['joint_accuracy'],'Pair accuracy')
        require(0<=accuracy<=1,'Pair accuracy range')
        total+=value['worlds'];correct+=value['worlds']*accuracy
    require(total==worlds,'Pair-stratum denominator')
    close(correct/worlds,row['joint_accuracy'],'Weighted true-pair accuracy')
    seen=set();total=0
    for value in row['predicted_joint_role_counts']:
        labels=tuple(value['labels']);count=value['worlds']
        require(len(labels)==3 and all(type(v) is int and 0<=v<3 for v in labels),'Predicted role domain')
        require(labels not in seen and type(count) is int and count>0,'Predicted role counts')
        seen.add(labels);total+=count
    require(total==worlds,'Predicted role denominator')
    for key in ('sha256','state_indices_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',row[key]) is not None,'Saved evaluation hash format')


def job_id(probe,view,old_seed=None,payoff=None):
    identity=f'probe_{probe}_{view}'
    if old_seed is not None:identity+=f'_protocol_{old_seed}'
    if payoff is not None:identity+='_'+payoff
    return identity


def extract(result):
    require(result.get('status')=='completed','Only a completed aggregate result may be plotted')
    require(re.fullmatch('[0-9a-f]{64}',result['plan_sha256']) is not None,'Plan identity format')
    budget=result['budget']
    for key,value in dict(actual_decoder_runs=42,logical_decoder_runs=96,aliased_decoder_runs=54,
        updates=252000,training_world_samples=64512000,training_forward_module_samples=193536000,
        checkpoints=252,actual_monitor_files=1008,actual_final_files=168,aliased_evaluation_records=1512).items():
        require(budget[key]==value,'Fixed budget: '+key)
    by={};actual=[]
    for row in result['runs']:
        job=row['job'];identity=job['id']
        require(identity not in by,'Duplicate actual job')
        require(job['probe_seed'] in PROBE_SEEDS and job['view'] in ('Own','FI','Initial','Final'),'Actual job identity')
        if job['view'] in ('Own','FI'):
            require(job['old_seed'] is None and job['payoff'] is None,'Own/FI must be shared jobs')
        else:
            require(job['old_seed'] in OLD_SEEDS,'Protocol seed')
            require((job['view']=='Initial' and job['payoff'] is None)
                    or (job['view']=='Final' and job['payoff'] in PAYOFFS),'Transcript stage/payoff')
        require(identity==job_id(job['probe_seed'],job['view'],job['old_seed'],job['payoff']),'Job ID arithmetic')
        require(set(row['final'])==set(PARTS),'Four full final partitions')
        for part in PARTS:verify_evaluation(row['final'][part],WORLD_COUNTS[part])
        require(tuple(m['update'] for m in row['monitor'])==STEPS,'Six fixed optimization checkpoints')
        for checkpoint in row['monitor']:
            require(set(checkpoint['evaluations'])==set(PARTS),'Four monitor partitions')
            for part in PARTS:verify_evaluation(checkpoint['evaluations'][part],MONITOR_COUNTS[part])
        by[identity]=row
        actual.append(dict(job=job,final={key:row['final'][TARGET][key] for key in
            ('worlds','joint_accuracy','indiv_accuracy','cross_entropy','path','sha256')},
            monitor=dict(worlds=MONITOR_COUNTS[TARGET],steps=list(STEPS),
                joint_accuracy=[m['evaluations'][TARGET]['joint_accuracy'] for m in row['monitor']],
                cross_entropy=[m['evaluations'][TARGET]['cross_entropy'] for m in row['monitor']])))
    expected=set()
    for probe in PROBE_SEEDS:
        expected.update(job_id(probe,view) for view in ('Own','FI'))
        for seed in OLD_SEEDS:
            expected.add(job_id(probe,'Initial',seed))
            expected.update(job_id(probe,'Final',seed,payoff) for payoff in PAYOFFS)
    require(len(result['runs'])==len(by)==42 and set(by)==expected,'Complete 42 actual decoder grid')
    logical=[]
    for seed in OLD_SEEDS:
        for payoff in PAYOFFS:
            for probe in PROBE_SEEDS:
                for view in ('Own','FI','Initial','Final'):
                    identity=job_id(probe,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None)
                    logical.append(dict(old_seed=seed,payoff=payoff,probe_seed=probe,view=view,actual_job_id=identity))
    require(len(logical)==96 and result['logical_results']==logical,'Logical alias map differs')
    primary_blocks=[]
    for seed in OLD_SEEDS:
        cells=[]
        for payoff in PAYOFFS:
            probes=[]
            for probe in PROBE_SEEDS:
                accuracy={view:by[job_id(probe,view,seed if view in ('Initial','Final') else None,
                    payoff if view=='Final' else None)]['final'][TARGET]['joint_accuracy'] for view in ('Own','FI','Initial','Final')}
                probes.append(dict(probe_seed=probe,accuracies=accuracy,final_minus_initial=accuracy['Final']-accuracy['Initial'],
                    final_minus_own=accuracy['Final']-accuracy['Own'],final_minus_bound=accuracy['Final']-BOUND,
                    initial_minus_bound=accuracy['Initial']-BOUND))
            cells.append(dict(payoff=payoff,probes=probes,mean_final_minus_initial=float(np.mean([p['final_minus_initial'] for p in probes]))))
        primary_blocks.append(dict(old_seed=seed,payoffs=cells,
            mean_final_minus_initial=float(np.mean([c['mean_final_minus_initial'] for c in cells]))))
    primary=dict(metric='joint_role_accuracy_final_minus_initial',partition=TARGET,protocol_seed_blocks=primary_blocks,
        mean_difference=float(np.mean([b['mean_final_minus_initial'] for b in primary_blocks])))
    compare(result['primary'],primary,'primary recomputation')
    return dict(actual_runs=actual,logical_aliases=logical,primary=primary,old_seeds=list(OLD_SEEDS),
        probe_seeds=list(PROBE_SEEDS),payoffs=list(PAYOFFS),steps=list(STEPS),target_partition=TARGET,
        full_final_worlds=WORLD_COUNTS[TARGET],monitor_worlds=MONITOR_COUNTS[TARGET],budget=budget,
        references=dict(silent_role_bound=BOUND,original_action_reference=ACTION_REFERENCE,
            scope='Fixed references requested for plotting: 23/62 silent-role bound and 1/3 original-action benchmark; not recomputed from this decoder result.'),
        units=dict(accuracies='Fractions in data, percent in figures',contrasts='Fractions in data, percentage points in figures'),
        scope=dict(independent_units='Four original protocol initializations. Three decoder seeds are repeated fits within each original society.',
            computation='42 actual decoder fits, 96 logical entries, 54 aliases. Own and FI are reused across all original seeds and payoffs; Initial is reused across both payoffs.',
            endpoint='All 53,568 double-holdout worlds at decoder update 6000.',
            monitoring='2,976 double-holdout worlds: every held-out need tuple at two fixed backgrounds. This is not the full final support.',
            dynamics='Optimization of newly supervised decoders on frozen transcripts. Decoder updates are not protocol generations or language-formation times.',
            uncertainty='All four society blocks shown; no world-level or logical-entry confidence interval.',
            validation='Aggregate JSON schema and arithmetic checks only; no NPZ, training-log, checkpoint or execution-audit files read.'))


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    output={}
    def save(fig,stem):
        for extension in ('.png','.pdf'):
            path=out/(stem+extension);require(not path.exists(),'Refuse figure overwrite')
            fig.savefig(path,bbox_inches='tight',facecolor='white')
            output[path.name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        plt.close(fig)
    blocks=data['primary']['protocol_seed_blocks']

    # Full-domain endpoint. Initial observations are shared across payoff rows.
    fig,axes=plt.subplots(2,4,figsize=(15.4,8.0),sharex=True,sharey=True)
    for column,block in enumerate(blocks):
        for row,cell in enumerate(block['payoffs']):
            ax=axes[row,column];values=np.asarray([[p['accuracies']['Initial'],p['accuracies']['Final']] for p in cell['probes']])*100
            for index,coordinates in enumerate(values):
                ax.plot([0,1],coordinates,color=SEED_COLORS[column],alpha=.45,lw=1.,
                    marker=PROBE_MARKERS[index],markersize=4.5,markerfacecolor='white',zorder=3)
            ax.plot([0,1],values.mean(0),color=SEED_COLORS[column],lw=2.6,marker='D',markersize=6,zorder=4)
            ax.axhline(BOUND*100,color='#777777',ls='--',lw=1.1,zorder=1)
            ax.axhline(ACTION_REFERENCE*100,color='#A1A1A1',ls=':',lw=1.1,zorder=1)
            ax.set_xticks([0,1],['Initial transcript','Final transcript'])
            ax.set_xlim(-.18,1.18);ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20))
            ax.grid(axis='y',alpha=.14)
            ax.set_title(f"Society {block['old_seed']}  |  alpha = {'.5' if cell['payoff']=='a50' else '.1'}",fontsize=11)
            if column==0:ax.set_ylabel('Joint role accuracy (%)')
            delta=cell['mean_final_minus_initial']*100
            ax.text(.5,.04,f'Mean change: {delta:+.2f} pp',transform=ax.transAxes,ha='center',fontsize=9,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    legend=[Line2D([0],[0],color='#596B75',lw=2.6,marker='D',label='Mean of 3 decoder seeds')]
    legend.extend(Line2D([0],[0],color='#596B75',lw=1,alpha=.5,marker=marker,markerfacecolor='white',label=f'Decoder {seed}')
                  for marker,seed in zip(PROBE_MARKERS,PROBE_SEEDS))
    legend.extend([Line2D([0],[0],color='#777777',ls='--',label='Silent-role upper bound: 23/62'),
                   Line2D([0],[0],color='#A1A1A1',ls=':',label='Original-action reference: 1/3')])
    fig.suptitle('Role information in frozen transcripts: initial versus learned protocol',fontsize=15,y=.99)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.81,bottom=.15,hspace=.34,wspace=.14)
    fig.text(.5,.055,'Full double holdout: 53,568 worlds per fit; fixed decoder update 6000. Initial fits are reused across both payoff rows.',ha='center',fontsize=10)
    fig.text(.5,.02,'Four independent societies; three decoder initializations are repeated fits. Reference lines are fixed benchmarks, not recomputed here.',ha='center',fontsize=9,color='#555555')
    save(fig,'01_initial_final_joint_accuracy')

    # Primary aggregation and complete payoff-specific values, including reversals.
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.9),gridspec_kw={'width_ratios':[1.05,1]})
    seed_values=np.asarray([b['mean_final_minus_initial'] for b in blocks])*100
    mean=data['primary']['mean_difference']*100
    payoff_values=np.asarray([[c['mean_final_minus_initial'] for c in b['payoffs']] for b in blocks])*100
    all_values=np.r_[seed_values,mean,payoff_values.ravel(),0]
    span=max(float(all_values.max()-all_values.min()),3.)
    limits=(min(float(all_values.min())-.12*span,-.75),max(float(all_values.max())+.17*span,.75))
    ax=axes[0]
    for index,(seed,value,color) in enumerate(zip(OLD_SEEDS,seed_values,SEED_COLORS)):
        ax.hlines(index,0,value,color=color,lw=2.,alpha=.7)
        ax.scatter(value,index,color=color,s=55,zorder=3)
        ax.annotate(f'{value:+.2f}',(value,index),xytext=(7 if value>=0 else -7,0),textcoords='offset points',
                    ha='left' if value>=0 else 'right',va='center',fontsize=9)
    ax.scatter(mean,4.25,marker='D',s=85,color='#252F37',zorder=4)
    ax.annotate(f'{mean:+.2f}',(mean,4.25),xytext=(7 if mean>=0 else -7,0),textcoords='offset points',
                ha='left' if mean>=0 else 'right',va='center',fontweight='bold',fontsize=10)
    ax.set_yticks([0,1,2,3,4.25],[str(s) for s in OLD_SEEDS]+['Four-society mean'])
    ax.set_ylim(4.75,-.55);ax.set_xlim(*limits);ax.axvline(0,color='#737373',lw=1)
    ax.grid(axis='x',alpha=.15);ax.set_xlabel('Final minus Initial joint accuracy (pp)')
    ax.set_title('Primary: average decoder seeds, then payoffs',fontsize=11,pad=12)
    ax=axes[1]
    for index,(seed,values,color) in enumerate(zip(OLD_SEEDS,payoff_values,SEED_COLORS)):
        ax.plot([0,1],values,marker=PROBE_MARKERS[index%3],color=color,lw=1.8,markersize=6,label=f'Society {seed}')
    ax.axhline(0,color='#737373',lw=1);ax.set_ylim(*limits);ax.set_xlim(-.15,1.15)
    ax.set_xticks([0,1],['alpha = .5','alpha = .1']);ax.set_ylabel('Final minus Initial joint accuracy (pp)')
    ax.grid(axis='y',alpha=.15);ax.set_title('Both payoffs retained for every society',fontsize=11,pad=12)
    ax.legend(loc='best',fontsize=9,frameon=True,framealpha=.9)
    fig.suptitle('Added decodable role information after protocol training',fontsize=15,y=.99)
    fig.subplots_adjust(top=.85,bottom=.20,wspace=.35)
    fig.text(.5,.07,'Each payoff-specific value averages 3 paired decoder fits; the primary averages the two payoffs within each of 4 societies.',ha='center',fontsize=10)
    fig.text(.5,.025,'42 actual decoder trainings support 96 logical entries. Own/FI and Initial aliases are not additional independent experiments.',ha='center',fontsize=9,color='#555555')
    save(fig,'02_final_minus_initial_contrasts')

    # Decoder optimization only, from monitoring support rather than full endpoint.
    records={r['job']['id']:r for r in data['actual_runs']}
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.0),sharex=True,sharey=True)
    for ax,seed in zip(axes.flat,OLD_SEEDS):
        for view in ('Own','FI','Initial','Final_a50','Final_a10'):
            curves=[]
            for probe in PROBE_SEEDS:
                if view.startswith('Final_'):identity=job_id(probe,'Final',seed,view.split('_',1)[1])
                elif view=='Initial':identity=job_id(probe,'Initial',seed)
                else:identity=job_id(probe,view)
                curves.append(records[identity]['monitor']['joint_accuracy'])
            curves=np.asarray(curves)*100
            color=VIEW_COLORS[view]
            style='--' if view in ('Own','FI') else '-'
            for curve in curves:ax.plot(STEPS,curve,color=color,ls=style,alpha=.18,lw=.8)
            ax.plot(STEPS,curves.mean(0),color=color,ls=style,lw=2,marker='o',markersize=3.5,label=view)
        ax.set_title(f'Society {seed}',fontsize=12)
        ax.set_ylim(0,100);ax.set_xlim(0,6000);ax.set_yticks(np.arange(0,101,20))
        ax.set_xticks([0,500,1500,3000,6000]);ax.tick_params(axis='x',labelsize=9)
        ax.grid(axis='y',alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Monitor joint role accuracy (%)')
    for ax in axes[-1]:ax.set_xlabel('Decoder optimization update (not a generation)')
    legend=[Line2D([0],[0],color=VIEW_COLORS[v],ls='--' if v in ('Own','FI') else '-',lw=2,
        label={'Own':'Own information (shared)','FI':'Full information (shared)','Initial':'Initial transcript',
               'Final_a50':'Final transcript, alpha = .5','Final_a10':'Final transcript, alpha = .1'}[v])
        for v in ('Own','FI','Initial','Final_a50','Final_a10')]
    fig.suptitle('Learning to decode roles from fixed inputs',fontsize=15,y=.99)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.15,hspace=.24,wspace=.13)
    fig.text(.5,.065,'All 6 saved updates: 0, 100, 500, 1500, 3000, 6000. Thick curves average 3 decoder seeds; thin curves show every fit.',ha='center',fontsize=10)
    fig.text(.5,.035,'Monitor support: 2,976 double-holdout worlds at two fixed backgrounds, not the 53,568-world full endpoint.',ha='center',fontsize=10)
    fig.text(.5,.008,'Own and FI curves are the same shared fits in every panel. These curves describe supervised decoder fitting, not language formation.',ha='center',fontsize=9,color='#555555')
    save(fig,'03_decoder_monitor_trajectories')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();run=Path(args.run).resolve()
    result_path=run/'execution/results.json'
    require(result_path.is_file(),'Completed aggregate results.json missing')
    digest=sha(result_path)
    with result_path.open(encoding='utf-8') as source:result=json.load(source)
    data=extract(result)
    out=Path(args.out).resolve() if args.out else run/'figures_001'
    require(not out.exists(),'Never overwrite a figure output directory');out.mkdir(parents=True)
    dump(out/'figure_data.json',data)
    try:
        outputs=draw(data,out)
        require(sha(result_path)==digest,'Aggregate result changed during plotting')
        import matplotlib
        receipt=dict(status='generated_visual_inspection_pending',generated_at=datetime.now(timezone.utc).isoformat(),
            source_result_path=str(result_path),source_result_sha256=digest,plan_sha256=result['plan_sha256'],
            plot_source_sha256=sha(__file__),figure_data_sha256=sha(out/'figure_data.json'),outputs=outputs,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
            source_scope='Only completed aggregate results.json; no model files, training logs, per-run outputs or execution-audit files read.',
            actual_decoder_runs=42,logical_decoder_entries=96,independent_protocol_seed_blocks=4,
            visual_inspection_performed=False,additional_model_forward_samples=0,training_updates=0)
        dump(out/'figure_receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],out=str(out),figures=list(outputs)),ensure_ascii=False))
    except BaseException as error:
        dump(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
             source_result_sha256=digest,plot_source_sha256=sha(__file__),automatic_retry=False))
        raise


if __name__=='__main__':main()
