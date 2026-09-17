"""Three descriptive figures from completed result JSON; no model imports.

Run only after formal main completion, using the plotting environment:
  research_program/.plotting_venv/bin/python -m \
    research_program.triadic_partial_payoff_study.plot_results --run RUN_DIRECTORY

Creates a new output directory, three PNG/PDF pairs, extracted data, and hashes.
Actual visual inspection is separate and must not be inferred from generation.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

import numpy as np

SEEDS = (53101, 53102, 53103, 53104)
PAYOFFS = (('a50', .5), ('a10', .1))
CONDITIONS = ('FI_silent', 'PL_silent', 'PL_live')
CELLS = tuple((name, alpha, condition) for name, alpha in PAYOFFS for condition in CONDITIONS)
STEPS = (0, 100, 500, 1500, 3000, 6000)
PARTS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
TARGET = 'new_needs_and_layouts'
METRICS = ('role_success_rate', 'full_success_rate')
METRIC_LABELS = {'role_success_rate':'Correct role (%)', 'full_success_rate':'Full success (%)'}
SHORT = {'FI_silent':'FI-S', 'PL_silent':'PL-S', 'PL_live':'PL-L'}
LONG = {'FI_silent':'Full info / silent', 'PL_silent':'Private needs / silent', 'PL_live':'Private needs / live'}
CONDITION_COLORS = {'FI_silent':'#667581', 'PL_silent':'#C2883C', 'PL_live':'#286F95'}
ALPHA_COLORS = {'a50':'#C17832', 'a10':'#176D91'}
SEED_COLORS = ('#216B8F', '#B45A39', '#5D8A51', '#895F96')
SEED_MARKERS = ('o', 's', '^', 'D')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=2e-12):
    return bool(np.allclose(a, b, atol=tolerance, rtol=0))


def number(value, label):
    require(isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value), label+' is not finite')
    return float(value)


def verify_record(record, worlds, condition, alpha, mode):
    require(record['worlds']==worlds and worlds>0, 'World denominator changed')
    require(record['information']==condition.split('_')[0], 'Information identity mismatch')
    require(record['partial_utility']==alpha, 'Payoff identity mismatch')
    require(record['live']==(condition.endswith('_live') and mode=='natural'), 'Message visibility mismatch')
    require(record['reused_natural']==(condition.endswith('_silent') and mode=='closed'), 'Alias identity mismatch')
    for key in (*METRICS, 'reward_mean', 'utility_mean', 'physical_execution_rate',
                'expected_reward_given_greedy_messages', 'expected_utility_given_greedy_messages',
                'full_probability_given_greedy_messages', 'full_posterior_mass_given_greedy_messages'):
        value=number(record[key],key)
        require(-2e-12<=value<=1+2e-12, key+' outside [0,1]')
    require(record['full_success_rate']<=record['role_success_rate']+2e-12, 'Full success exceeds correct roles')
    require(record['full_success_rate']<=record['physical_execution_rate']+2e-12, 'Full success exceeds execution')
    require(close(record['utility_mean'], record['full_success_rate']+2*alpha*(record['reward_mean']-record['full_success_rate'])),
            'Greedy utility/native arithmetic mismatch')
    require(close(record['expected_utility_given_greedy_messages'],
                  record['full_probability_given_greedy_messages']+2*alpha*(record['expected_reward_given_greedy_messages']-record['full_probability_given_greedy_messages'])),
            'Conditional expected utility/native arithmetic mismatch')
    for key in METRICS:
        count=record[key]*worlds
        require(abs(count-round(count))<1e-6,'Greedy rate does not have stated denominator')
    from_actions=Counter()
    actions_seen=set()
    for item in record['raw_joint_action_counts']:
        actions=tuple(item['action_indices']);count=item['worlds']
        require(len(actions)==3 and all(type(a) is int and 0<=a<17 for a in actions), 'Invalid action counts')
        require(actions not in actions_seen and type(count) is int and count>0,'Duplicate/invalid action count')
        actions_seen.add(actions)
        roles=tuple(-1 if action==0 else tuple(j for j in range(3) if j!=actor)[(action-1)%2]
                    for actor,action in enumerate(actions))
        from_actions[roles]+=count
    from_roles={}
    for item in record['raw_joint_role_counts']:
        roles=tuple(item['partner_indices']);count=item['worlds']
        require(roles not in from_roles and type(count) is int and count>0,'Duplicate/invalid role count')
        from_roles[roles]=count
    require(dict(from_actions)==from_roles and sum(from_roles.values())==worlds,'Action/role frequency mismatch')


def verify_modes(modes, worlds, condition, alpha):
    require(set(modes)=={'natural','closed'},'Both natural and closed modes required')
    for mode in ('natural','closed'):
        verify_record(modes[mode],worlds,condition,alpha,mode)
    natural,closed=modes['natural'],modes['closed']
    if condition.endswith('_silent'):
        require({k:v for k,v in natural.items() if k!='reused_natural'}==
                {k:v for k,v in closed.items() if k!='reused_natural'},'Silent alias differs from natural record')
    else:
        require(natural['path']!=closed['path'],'Live closure must be a separate evaluation')


def fixed_pair_description(run, mode):
    combined=Counter()
    for part in PARTS:
        combined.update({tuple(item['partner_indices']):item['worlds'] for item in run['final'][part][mode]['raw_joint_role_counts']})
    patterns=sorted(combined)
    fixed=False
    if len(patterns)==1:
        pattern=patterns[0]
        active=[i for i,p in enumerate(pattern) if p!=-1]
        fixed=len(active)==2 and pattern[active[0]]==active[1] and pattern[active[1]]==active[0]
    return dict(fixed_mutual_pair=bool(fixed), patterns=[dict(partners=list(p),worlds=combined[p]) for p in patterns],
                worlds=sum(combined.values()), scope='Union of every world in all four final partitions; includes failures')


def extract_data(result, prepared):
    require(result.get('status')=='completed','Only completed main results may be plotted')
    runs=result['runs']
    expected={(seed,payoff,condition) for seed in SEEDS for payoff,_,condition in CELLS}
    lookup={(row['seed'],row['payoff'],row['condition']):row for row in runs}
    require(len(runs)==24 and set(lookup)==expected,'All24 seed/payoff/condition cells required')
    require(result['budget']==prepared['budget'],'Main/prepared budget mismatch')
    rows=[]
    for seed in SEEDS:
        for payoff,alpha,condition in CELLS:
            run=lookup[seed,payoff,condition]
            require(run['partial_utility']==alpha and run['updates']==6000,'Run identity mismatch')
            require(set(run['final'])==set(PARTS),'All four final partitions required')
            for part in PARTS:
                verify_modes(run['final'][part],prepared['partitions'][part]['world_count'],condition,alpha)
            require(tuple(item['update'] for item in run['monitor'])==STEPS,'Exactly six saved checkpoints required')
            for checkpoint in run['monitor']:
                require(set(checkpoint['monitor'])==set(PARTS),'Incomplete monitoring partitions')
                for part in PARTS:
                    n=len(prepared['partitions'][part]['monitor_indices'])
                    verify_modes(checkpoint['monitor'][part],n,condition,alpha)
            final={mode:{key:run['final'][TARGET][mode][key] for key in
                         (*METRICS,'reward_mean','utility_mean','worlds','path','data_sha256','reused_natural')}
                   for mode in ('natural','closed')}
            monitor={mode:{key:[item['monitor'][TARGET][mode][key] for item in run['monitor']] for key in METRICS}
                     for mode in ('natural','closed')}
            rows.append(dict(seed=seed,payoff=payoff,alpha=alpha,condition=condition,final=final,monitor=monitor,
                monitor_worlds=len(prepared['partitions'][TARGET]['monitor_indices']),
                natural_role_patterns=fixed_pair_description(run,'natural'),
                closed_role_patterns=fixed_pair_description(run,'closed')))
    by={(row['seed'],row['payoff'],row['condition']):row for row in rows}
    primary=result['primary']
    require(primary['metric']=='role_success_rate' and primary['partition']==TARGET,'Primary estimand changed')
    stored={row['seed']:row for row in primary['paired_seeds']}
    require(len(primary['paired_seeds'])==4 and set(stored)==set(SEEDS),'All four paired primary values required')
    paired=[]
    for seed in SEEDS:
        cell={payoff+'_'+condition:by[seed,payoff,condition]['final']['natural'] for payoff,_,condition in CELLS}
        diff={key:(cell['a10_PL_live'][key]-cell['a10_PL_silent'][key])-
                  (cell['a50_PL_live'][key]-cell['a50_PL_silent'][key])
              for key in (*METRICS,'reward_mean')}
        for key,value in diff.items():
            require(close(stored[seed]['contrasts'][key],value),'Stored primary paired arithmetic mismatch')
        for identity,value in cell.items():
            for key in (*METRICS,'reward_mean','utility_mean'):
                require(close(stored[seed]['cells'][identity][key],value[key]),'Stored primary cell mismatch')
        paired.append(dict(seed=seed,role_DiD=diff['role_success_rate'],full_success_DiD=diff['full_success_rate']))
    mean=float(np.mean([row['role_DiD'] for row in paired]))
    require(close(primary['mean_difference'],mean),'Stored primary mean mismatch')
    return dict(rows=rows,primary=paired,primary_mean=mean,steps=list(STEPS),seeds=list(SEEDS),
        target_partition=TARGET,final_worlds=prepared['partitions'][TARGET]['world_count'],
        monitor_worlds=len(prepared['partitions'][TARGET]['monitor_indices']),
        units={'rates':'Fraction in data; plotted as percent by multiplying100.',
               'primary':'Difference of four role-success fractions; plotted in percentage points by multiplying100.'},
        scope={'monitor':'All held-out need relations in exactly two fixed layouts/owner backgrounds, not full final support.',
               'final':'All worlds in new_needs_and_layouts; no selection by success.',
               'silent_closed':'Alias of natural result, not an extra evaluation or independent observation.',
               'live_closed':'Same saved policy rerun with external messages closed; no retraining.',
               'fixed_pairs':'Unique mutual partner pattern over all four complete final partitions.',
               'uncertainty':'Four paired initializations; all shown. No confidence interval or significance claim.'})


def load_data(run):
    run=Path(run).resolve()
    status_path=run/'execution/status.json'
    require(status_path.is_file() and read(status_path).get('status')=='completed','Refuse incomplete execution')
    require(not (run/'execution/failure.json').exists(),'Refuse execution with a failure marker')
    result_path=run/'execution/results.json'
    result=read(result_path)
    plan_path,prepared_path,freeze_path=run/'plan.json',run/'prepared.json',run/'freeze.json'
    plan=read(plan_path);prepared=read(prepared_path);freeze=read(freeze_path)
    require(sha(plan_path)==result['plan_sha256']==freeze['plan_sha256'],'Plan identity mismatch')
    require(sha(prepared_path)==plan['prepared_sha256'],'Prepared support identity mismatch')
    require(tuple(plan['config']['seeds'])==SEEDS and tuple(plan['config']['conditions'])==CONDITIONS,'Configuration mismatch')
    source_paths=(result_path,status_path,plan_path,prepared_path,freeze_path)
    sources={str(p):sha(p) for p in source_paths}
    return extract_data(result,prepared),sources


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    rows={(row['seed'],row['payoff'],row['condition']):row for row in data['rows']}
    outputs={}
    def save(fig,stem):
        for ext in ('.png','.pdf'):
            path=out/(stem+ext);require(not path.exists(),'Never overwrite a figure')
            fig.savefig(path,bbox_inches='tight',facecolor='white')
            outputs[path.name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        plt.close(fig)
    limits={}
    for key in METRICS:
        values=[]
        for row in data['rows']:
            for mode in ('natural','closed'):
                values.append(row['final'][mode][key]);values.extend(row['monitor'][mode][key])
        limits[key]=min(100,max(40,10*math.ceil(max(values)*100/10+0.35)))
    def rate_axis(ax,key,horizontal=False):
        bound=limits[key]
        if horizontal:
            ax.set_xlim(-.025*bound,1.025*bound);ax.set_xticks(np.arange(0,bound+.1,20 if bound>60 else 10))
            ax.axvline(100/3,color='#999999',linestyle=':',linewidth=1,zorder=0)
            ax.grid(axis='x',alpha=.18)
        else:
            ax.set_ylim(0,bound);ax.set_yticks(np.arange(0,bound+.1,20 if bound>60 else 10))
            ax.axhline(100/3,color='#999999',linestyle=':',linewidth=1,zorder=0)
            ax.grid(axis='y',alpha=.18)

    # Figure1: every seed and cell; only the predeclared role DiD is emphasized.
    fig=plt.figure(figsize=(16.3,10.5))
    grid=fig.add_gridspec(3,4,height_ratios=(1,1,.78),left=.065,right=.985,top=.85,bottom=.145,hspace=.7,wspace=.28)
    xs=np.array([0,1,2,4,5,6])
    for mi,key in enumerate(METRICS):
        for si,seed in enumerate(SEEDS):
            ax=fig.add_subplot(grid[mi,si]);rate_axis(ax,key)
            values=[rows[seed,p,c]['final']['natural'][key]*100 for p,_,c in CELLS]
            bars=ax.bar(xs,values,width=.7,color=[CONDITION_COLORS[c] for _,_,c in CELLS],zorder=2)
            for bar in bars[3:]:bar.set_hatch('//');bar.set_edgecolor('white')
            ax.set_xticks(xs,[SHORT[c] for _,_,c in CELLS],fontsize=8)
            ax.set_xlim(-.6,6.6)
            ax.text(1,-.25,'alpha = 0.5',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
            ax.text(5,-.25,'alpha = 0.1',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
            if mi==0:ax.set_title(f'Seed {seed}',fontsize=11)
            if si==0:ax.set_ylabel(METRIC_LABELS[key])
            for x,value in zip(xs,values):
                inside=value>.93*limits[key]
                ax.annotate(f'{value:.1f}',(x,value),xytext=(0,-12 if inside else 4),textcoords='offset points',
                            ha='center',fontsize=7.4,color='white' if inside else '#333333')
    ax=fig.add_subplot(grid[2,:])
    difference=np.array([row['role_DiD'] for row in data['primary']])*100
    mean=data['primary_mean']*100
    maxabs=max(1,math.ceil(max(np.max(np.abs(difference)),abs(mean))*1.4))
    ax.set_ylim(-maxabs,maxabs);ax.set_xlim(-.55,3.55)
    ax.axhline(0,color='#999999',linewidth=1)
    ax.axhline(mean,color='#222222',linestyle='--',linewidth=1.4,label=f'Mean: {mean:+.2f} pp')
    for i,(seed,value) in enumerate(zip(SEEDS,difference)):
        ax.scatter(i,value,color=SEED_COLORS[i],s=55,zorder=3)
        ax.annotate(f'{value:+.2f}',(i,value),xytext=(0,9 if value>=0 else -14),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(range(4),[str(s) for s in SEEDS]);ax.set_ylabel('Role DiD (pp)');ax.set_xlabel('Paired initialization')
    ax.set_title('Primary: (live - silent) at alpha 0.1 minus (live - silent) at alpha 0.5',fontsize=11)
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='upper right')
    fig.suptitle('Partial-success payoff: complete held-out-world endpoint',fontsize=16,y=.975)
    fig.text(.065,.922,f'6000 updates | {data["final_worlds"]:,} worlds per policy | all four paired initializations',fontsize=11)
    fig.legend(handles=[Patch(color=CONDITION_COLORS[c],label=SHORT[c]+': '+LONG[c]) for c in CONDITIONS],
               loc='upper left',bbox_to_anchor=(.06,.908),ncol=3,frameon=False,fontsize=10)
    fig.text(.065,.052,'Rates are percentages; the primary contrast is in percentage points. Dotted line: fixed-pair bound of 33.3%.\n'
             'No success filtering. Full success also requires the correct object, site and destination. No significance test is shown.',fontsize=9,color='#4E4E4E')
    save(fig,'endpoint_cells_and_role_DiD')

    # Figure2: same monitoring support at all six points, never replace the last point with the full endpoint.
    fig,axes=plt.subplots(2,3,figsize=(15.4,8.9),squeeze=False)
    fig.subplots_adjust(left=.07,right=.985,top=.81,bottom=.17,hspace=.35,wspace=.25)
    for mi,key in enumerate(METRICS):
        for ci,condition in enumerate(CONDITIONS):
            ax=axes[mi,ci];rate_axis(ax,key)
            for payoff,alpha in PAYOFFS:
                values=np.array([rows[seed,payoff,condition]['monitor']['natural'][key] for seed in SEEDS])*100
                for si,seed in enumerate(SEEDS):
                    ax.plot(STEPS,values[si],color=ALPHA_COLORS[payoff],alpha=.38,linewidth=.9,
                            marker=SEED_MARKERS[si],markersize=3)
                ax.plot(STEPS,values.mean(axis=0),color=ALPHA_COLORS[payoff],linewidth=2.5,
                        label=f'alpha = {alpha:.1f}',zorder=4)
            ax.set_xlim(-100,6100);ax.set_xticks((0,1500,3000,6000));ax.set_xlabel('Training updates')
            if mi==0:ax.set_title(LONG[condition],fontsize=12)
            if ci==0:ax.set_ylabel(METRIC_LABELS[key])
    fig.suptitle('Formation monitoring: six saved checkpoints',fontsize=16,y=.97)
    fig.text(.07,.914,f'{data["monitor_worlds"]:,} held-out worlds per point | two fixed backgrounds | natural messages',fontsize=11)
    handles=[Line2D([],[],color=ALPHA_COLORS[p],linewidth=2.5,label=f'alpha = {a:.1f} (mean)') for p,a in PAYOFFS]
    handles += [Line2D([],[],color='#777777',linewidth=.8,marker=SEED_MARKERS[i],markersize=4,label=str(s)) for i,s in enumerate(SEEDS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.891),ncol=6,frameon=False,fontsize=9)
    fig.text(.07,.064,'Thin lines: each initialization; thick lines: equal-weight mean. Time spacing follows the actual update counts.\n'
             'All six points use the same monitoring worlds. The 6000-step monitoring point is not the complete-world endpoint.\n'
             'Dotted line: fixed-pair bound of 33.3%. Optimization checkpoints are not generations.',fontsize=9,color='#4E4E4E')
    save(fig,'six_checkpoint_role_and_success')

    # Figure3: paired message closure for every policy; silent aliases explicitly marked.
    fig,axes=plt.subplots(2,6,figsize=(17.2,8.6),squeeze=False)
    fig.subplots_adjust(left=.076,right=.988,top=.795,bottom=.19,hspace=.36,wspace=.2)
    for ci,(payoff,alpha,condition) in enumerate(CELLS):
        fixed=sum(rows[seed,payoff,condition]['natural_role_patterns']['fixed_mutual_pair'] for seed in SEEDS)
        alias=condition.endswith('_silent')
        for mi,key in enumerate(METRICS):
            ax=axes[mi,ci];rate_axis(ax,key,horizontal=True)
            for si,seed in enumerate(SEEDS):
                row=rows[seed,payoff,condition]
                natural=row['final']['natural'][key]*100;closed=row['final']['closed'][key]*100
                ax.plot([natural,closed],[si,si],color=SEED_COLORS[si],linewidth=1.6,alpha=.8,zorder=2)
                ax.scatter(natural,si,color=SEED_COLORS[si],s=28,zorder=3)
                ax.scatter(closed,si,facecolors='none',edgecolors=SEED_COLORS[si],s=68,marker='s',linewidths=1.2,zorder=4)
            ax.set_yticks(range(4),[str(s) for s in SEEDS] if ci==0 else ['']*4)
            ax.set_ylim(3.5,-.5)
            ax.set_xlabel('Percent')
            if ci==0:ax.set_ylabel(('Correct role' if mi==0 else 'Full success')+'\nSeed')
            if mi==0:ax.set_title(f'{SHORT[condition]} | alpha {alpha:.1f}\nFixed pairs: {fixed}/4',fontsize=10)
            if mi==1:
                ax.text(.5,-.3,'Closed = alias' if alias else 'Closed = rerun',transform=ax.transAxes,
                        ha='center',fontsize=9,color='#555555')
    fig.suptitle('Complete endpoint: natural messages versus closed channel',fontsize=16,y=.97)
    fig.text(.076,.913,f'6000 updates | {data["final_worlds"]:,} held-out worlds | each row is one paired policy',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='#333333',marker='o',linestyle='none',label='Natural'),
                        Line2D([],[],color='#333333',marker='s',markerfacecolor='none',markersize=8,linestyle='none',label='Closed')],
               loc='upper left',bbox_to_anchor=(.071,.886),ncol=2,frameon=False)
    fig.text(.076,.065,'FI-S: full information / silent; PL-S: private needs / silent; PL-L: private needs / live. All layouts are public.\n'
             'Silent closure reuses the natural record; it is not an extra evaluation. Live closure reruns the saved policy without retraining.\n'
             'Fixed pairs: one mutual partner pattern over all four full final partitions, including failures. Dotted line: 33.3% fixed-pair bound.',
             fontsize=9,color='#4E4E4E')
    save(fig,'natural_closed_all_policies')
    return outputs,matplotlib.__version__


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();run=Path(args.run).resolve();out=Path(args.out).resolve() if args.out else run/'figures_001'
    require(not out.exists(),'Refuse to overwrite an existing output directory')
    data,sources=load_data(run)
    out.mkdir(parents=True,exist_ok=False)
    try:
        path=out/'figure_data.json'
        with path.open('x',encoding='utf-8') as handle:json.dump(data,handle,ensure_ascii=False,indent=2)
        outputs,matplotlib_version=draw(data,out)
        outputs[path.name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        require(all(sha(path)==digest for path,digest in sources.items()),'Input changed while plotting')
        receipt=dict(status='figures_generated_pending_visual_inspection',at=datetime.now(timezone.utc).isoformat(),
            run=str(run),out=str(out),source_sha256=sources,plot_source_sha256=sha(__file__),outputs=outputs,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib_version),
            scope=dict(neural_forward_calls=0,training_updates=0,model_parameter_loads=0,npz_data_files_read=0,
                       training_logs_read=0,completed_policy_records=24,figures_png=3,figures_pdf=3,
                       visual_inspection_performed=False),primary_mean_role_DiD=data['primary_mean'])
        with (out/'receipt.json').open('x',encoding='utf-8') as handle:json.dump(receipt,handle,ensure_ascii=False,indent=2)
        print(json.dumps(dict(status=receipt['status'],out=str(out),outputs=list(outputs)),ensure_ascii=False))
    except BaseException as error:
        with (out/'failure.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(status='failed',error=repr(error),source_sha256=sources),handle,ensure_ascii=False,indent=2)
        raise


if __name__=='__main__':main()
