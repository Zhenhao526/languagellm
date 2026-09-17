"""Execution-ability figures from one completed aggregate result JSON only.

No model, environment or runner imports; no NPZ or training-log reads.
Use research_program/.plotting_venv/bin/python -m
research_program.triadic_reciprocal_execution_study.plot_results --run RUN.
"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,math,platform,re
import numpy as np

SEEDS=(57101,57102,57103,57104)
RULES=('strict','reciprocal')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
TARGET='new_needs_and_layouts'
WORLD_COUNTS=dict(zip(PARTS,(419904,160704,139968,53568)))
MONITOR_COUNTS=dict(zip(PARTS,(7776,2976,7776,2976)))
METRICS=('full_success_rate','executed_partner_correct_rate','proposal_role_success_rate','reward_mean')
SEED_COLORS=('#246B8E','#B36042','#64814E','#906197')
MARKERS=('o','s','^','D')
RULE_COLORS={'strict':'#637D90','reciprocal':'#BD623F'}


def require(ok,label):
    if not ok:raise ValueError(label)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def number(value,label):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Nonfinite '+label)
    return float(value)


def close(actual,expected,label):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=2e-12,rtol=0),'Arithmetic mismatch '+label)


def compare(actual,expected,label=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'Mapping '+label)
        for key,value in expected.items():compare(actual[key],value,label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'List '+label)
        for index,value in enumerate(expected):compare(actual[index],value,label+'/'+str(index))
    elif isinstance(expected,float):close(actual,expected,label)
    else:require(actual==expected,'Value '+label)


def verify_summary(row,worlds,rule):
    require(row['worlds']==worlds and row['rule']==rule,'Settlement identity/denominator')
    rates=(*METRICS,'physical_execution_rate','kind_correct_rate','length_correct_rate','destination_correct_rate',
           'material_identity_correct_rate','ignored_proposal_world_rate','ignored_proposal_agent_rate','unexecuted_proposal_agent_rate')
    for key in rates:require(0<=number(row[key],key)<=1,'Rate range '+key)
    reward_counts=row['reward_counts'];require(set(reward_counts)=={'0.0','0.5','1.0'},'Reward count schema')
    require(all(type(v) is int and v>=0 for v in reward_counts.values()) and sum(reward_counts.values())==worlds,'Reward count denominator')
    close(row['full_success_rate'],reward_counts['1.0']/worlds,'Full success count')
    close(row['reward_mean'],(.5*reward_counts['0.5']+reward_counts['1.0'])/worlds,'Native reward count')
    require(row['full_success_rate']<=row['executed_partner_correct_rate']+2e-12<=row['physical_execution_rate']+4e-12,'Full/partner/execution ordering')
    pairs=row['actual_pair_counts'];require(set(pairs)=={'none','AB','AC','BC'},'Executed pair count schema')
    require(all(type(v) is int and v>=0 for v in pairs.values()) and sum(pairs.values())==worlds,'Executed pair denominator')
    close(row['physical_execution_rate'],1-pairs['none']/worlds,'Actual physical executions')
    ignored=row['ignored_proposal_counts_by_actor']
    require(len(ignored)==3 and all(type(v) is int and 0<=v<=worlds for v in ignored),'Ignored actor counts')
    close(row['ignored_proposal_world_rate'],sum(ignored)/worlds,'At most one ignored proposal per executed world')
    close(row['ignored_proposal_agent_rate'],sum(ignored)/(3*worlds),'Ignored agent denominator')
    require(row['ignored_proposal_world_rate']<=row['physical_execution_rate']+2e-12,'Ignored proposal requires execution')
    if rule=='strict':
        require(sum(ignored)==0,'Strict rule cannot ignore a third proposal')
        require(row['full_success_rate']<=row['proposal_role_success_rate']+2e-12,'Strict full success requires all proposal roles')
    from_actions=Counter();seen=set();total=0
    for item in row['raw_joint_action_counts']:
        actions=tuple(item['action_indices']);count=item['worlds']
        require(len(actions)==3 and all(type(v) is int and 0<=v<17 for v in actions),'Proposal action domain')
        require(actions not in seen and type(count) is int and count>0,'Proposal action histogram')
        seen.add(actions);total+=count
        roles=tuple(-1 if value==0 else tuple(j for j in range(3) if j!=i)[(value-1)%2] for i,value in enumerate(actions))
        from_actions[roles]+=count
    require(total==worlds,'Proposal action denominator')
    from_roles={tuple(item['partner_indices']):item['worlds'] for item in row['raw_proposal_role_counts']}
    require(len(from_roles)==len(row['raw_proposal_role_counts']) and dict(from_actions)==from_roles,'Proposal role histogram agrees with actions')
    require(set(row['true_pair_strata'])=={'AB','AC','BC'},'True-pair strata')
    total=0;weighted={key:0. for key in rates}
    for value in row['true_pair_strata'].values():
        n=value['worlds'];require(type(n) is int and n>0,'Nonempty true-pair stratum');total+=n
        for key in rates:
            v=number(value[key],'True-pair '+key);require(0<=v<=1,'True-pair rate range');weighted[key]+=n*v
    require(total==worlds,'True-pair denominator')
    for key,value in weighted.items():close(value/worlds,row[key],'True-pair weighted '+key)


def verify_record(row,worlds,rule):
    verify_summary(row,worlds,rule)
    require(row['information']=='FI' and row['live'] is False,'FI-silent evaluation required')
    require(set(row['cross_settlement'])==set(RULES),'Both fixed settlement rules required')
    for settlement in RULES:verify_summary(row['cross_settlement'][settlement],worlds,settlement)
    own=row['cross_settlement'][rule]
    for key,value in own.items():require(row[key]==value,'Native settlement copy '+key)
    first,second=(row['cross_settlement'][r] for r in RULES)
    require(first['raw_joint_action_counts']==second['raw_joint_action_counts'],'Cross-settlement proposals must be identical')
    require(first['raw_proposal_role_counts']==second['raw_proposal_role_counts'],'Cross-settlement proposal roles must be identical')
    close(first['proposal_role_success_rate'],second['proposal_role_success_rate'],'Proposal-role success invariant across settlement')
    require(second['full_success_rate']+2e-12>=first['full_success_rate'],'Reciprocal settlement cannot remove a strict success')
    for key in ('expected_reward_given_greedy_messages','full_probability_given_greedy_messages',
                'execution_probability_given_greedy_messages','full_posterior_mass_given_greedy_messages'):
        require(0<=number(row[key],key)<=1,'Conditional probability range')
    for key in ('data_sha256','state_indices_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',row[key]) is not None,'Evaluation digest format')


def extract(result):
    require(result.get('status')=='completed','Only completed aggregate results may be plotted')
    require(re.fullmatch('[0-9a-f]{64}',result['plan_sha256']) is not None,'Plan hash format')
    for key,value in dict(runs=8,training_updates=48000,training_world_samples=12288000,
        message_trajectories=24576000,categorical_symbol_samples=589824000,
        training_forward_module_samples=221184000,checkpoints=48,actual_monitor_files=192,
        actual_final_files=32,aliased_evaluations=0,actual_monitor_worlds=1032192,
        actual_final_worlds=6193152).items():require(result['budget'][key]==value,'Frozen budget '+key)
    by={(r['seed'],r['rule']):r for r in result['runs']}
    require(len(result['runs'])==len(by)==8 and set(by)=={(s,r) for s in SEEDS for r in RULES},'All 8 paired policy runs required')
    rows=[]
    for seed in SEEDS:
        for rule in RULES:
            row=by[seed,rule]
            require(row['condition']=='FI_silent' and row['updates']==6000,'Fixed condition/update')
            require(set(row['final'])==set(PARTS),'All final partitions')
            for part in PARTS:verify_record(row['final'][part],WORLD_COUNTS[part],rule)
            require(tuple(m['update'] for m in row['monitor'])==STEPS,'Six checkpoint grid')
            for checkpoint in row['monitor']:
                require(set(checkpoint['monitor'])==set(PARTS),'All monitor partitions')
                for part in PARTS:verify_record(checkpoint['monitor'][part],MONITOR_COUNTS[part],rule)
            final=row['final'][TARGET]
            rows.append(dict(seed=seed,rule=rule,final={key:final[key] for key in (*METRICS,'worlds','path','data_sha256')},
                cross_settlement={r:{key:final['cross_settlement'][r][key] for key in METRICS} for r in RULES},
                monitor={key:[m['monitor'][TARGET][key] for m in row['monitor']]
                         for key in ('full_success_rate','executed_partner_correct_rate')}))
    paired=[]
    for seed in SEEDS:
        cells={r:by[seed,r]['final'][TARGET] for r in RULES}
        differences={key:cells['reciprocal'][key]-cells['strict'][key] for key in METRICS}
        matrix={trained:{settlement:cells[trained]['cross_settlement'][settlement]['full_success_rate'] for settlement in RULES} for trained in RULES}
        ss,sr=matrix['strict']['strict'],matrix['strict']['reciprocal']
        rs,rr=matrix['reciprocal']['strict'],matrix['reciprocal']['reciprocal']
        decomposition=dict(common_reciprocal_policy_difference=rr-sr,strict_policy_mechanical_release=sr-ss,
            common_strict_policy_difference=rs-ss,reciprocal_policy_mechanical_release=rr-rs)
        close(differences['full_success_rate'],decomposition['common_reciprocal_policy_difference']+decomposition['strict_policy_mechanical_release'],'Reciprocal reference arithmetic')
        close(differences['full_success_rate'],decomposition['common_strict_policy_difference']+decomposition['reciprocal_policy_mechanical_release'],'Strict reference arithmetic')
        paired.append(dict(seed=seed,contrasts=differences,cells={r:{key:c[key] for key in METRICS} for r,c in cells.items()},
            full_success_cross_settlement=matrix,decomposition=decomposition))
    primary=dict(metric='full_success_rate',partition=TARGET,paired_seeds=paired,
        mean_difference=float(np.mean([v['contrasts']['full_success_rate'] for v in paired])))
    compare(result['primary'],primary,'Primary and cross-settlement arithmetic')
    return dict(rows=rows,primary=primary,seeds=list(SEEDS),rules=list(RULES),steps=list(STEPS),
        target_partition=TARGET,final_worlds=53568,monitor_worlds=2976,budget=result['budget'],
        fixed_pair_reference=1/3,units=dict(rates='Fractions in data, percent in figures',differences='Fractions in data, percentage points in figures'),
        scope=dict(independent_units='Four paired initialization seeds; eight policy fits are repeated conditions within four blocks.',
            setting='Full needs and public layout, no cross-agent messages. These figures assess execution ability.',
            final='All 53,568 double-holdout worlds at fixed update 6000.',
            monitor='2,976 double-holdout worlds at two fixed backgrounds; not the full final evaluation.',
            cross_settlement='Each policy keeps exactly its saved joint proposals; only the settlement calculation changes. No new model forward.',
            decomposition='Two equivalent arithmetic identities, not a causal mechanism decomposition or separate independent evidence.',
            reference='1/3 is a fixed-pair reference on balanced true pairs, not a general FI-silent upper bound.',
            data_validation='Aggregate JSON schema and arithmetic only; no per-run artifact, checkpoint, NPZ or audit verification read.'))


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    outputs={}
    def save(fig,stem):
        for suffix in ('.png','.pdf'):
            path=out/(stem+suffix);require(not path.exists(),'No figure overwrite')
            fig.savefig(path,bbox_inches='tight',facecolor='white');outputs[path.name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        plt.close(fig)
    rows={(v['seed'],v['rule']):v for v in data['rows']};paired=data['primary']['paired_seeds']

    fig,ax=plt.subplots(figsize=(9.6,6.6))
    for seed,color,marker,offset in zip(SEEDS,SEED_COLORS,MARKERS,(np.arange(4)-1.5)*.035):
        y=[rows[seed,r]['final']['full_success_rate']*100 for r in RULES]
        ax.plot(np.asarray([0,1])+offset,y,color=color,marker=marker,lw=1.9,markersize=7,label=f'Seed {seed}')
    ax.axhline(100/3,color='#838383',ls='--',lw=1.1,label='Fixed-pair reference: 1/3')
    ax.set_xticks([0,1],['Strict settlement','Reciprocal settlement']);ax.set_xlim(-.18,1.18)
    ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('Full success (%)');ax.grid(axis='y',alpha=.18)
    ax.set_title('Execution ability under two physical rules',fontsize=15,pad=18)
    ax.legend(loc='best',framealpha=.92,fontsize=9)
    fig.subplots_adjust(bottom=.2,top=.86)
    fig.text(.5,.09,'Four paired seeds; fixed update 6000; all 53,568 double-holdout worlds. Horizontal offsets separate coincident seeds.',ha='center',fontsize=9.5)
    fig.text(.5,.045,'Full information, silent channel. The 1/3 fixed-pair reference is not an upper bound for general FI-silent policies.',ha='center',fontsize=9,color='#555555')
    save(fig,'01_execution_ability_paired')

    # Primary, all four 2x2 matrices, and the chosen arithmetic identity side by side.
    fig=plt.figure(figsize=(18.6,7.7));grid=fig.add_gridspec(1,3,width_ratios=[1.05,1.1,1.35],wspace=.4)
    primary_ax=fig.add_subplot(grid[0,0]);matrix_grid=grid[0,1].subgridspec(2,2,wspace=.24,hspace=.5)
    split_ax=fig.add_subplot(grid[0,2])
    differences=np.asarray([p['contrasts']['full_success_rate'] for p in paired])*100
    common=np.asarray([p['decomposition']['common_reciprocal_policy_difference'] for p in paired])*100
    release=np.asarray([p['decomposition']['strict_policy_mechanical_release'] for p in paired])*100
    mean=data['primary']['mean_difference']*100
    candidates=np.r_[differences,common,release,mean,0]
    span=max(float(candidates.max()-candidates.min()),5.)
    limits=(min(float(candidates.min())-.15*span,-1.),max(float(candidates.max())+.2*span,1.))
    for i,(seed,value,color,marker) in enumerate(zip(SEEDS,differences,SEED_COLORS,MARKERS)):
        primary_ax.hlines(i,0,value,color=color,lw=2,alpha=.7);primary_ax.scatter(value,i,color=color,marker=marker,s=60,zorder=3)
        primary_ax.annotate(f'{value:+.2f}',(value,i),xytext=(7,0),textcoords='offset points',
            va='center',ha='left',fontsize=9)
    primary_ax.scatter(mean,4.25,color='#28343D',marker='D',s=85)
    primary_ax.annotate(f'{mean:+.2f}',(mean,4.25),xytext=(7,0),textcoords='offset points',
        va='center',ha='left',fontsize=9,fontweight='bold')
    primary_ax.set_yticks([0,1,2,3,4.25],[str(s) for s in SEEDS]+['Mean'])
    primary_ax.set_ylim(4.8,-.55);primary_ax.set_xlim(*limits);primary_ax.axvline(0,color='#777777',lw=1)
    primary_ax.grid(axis='x',alpha=.16);primary_ax.set_xlabel('Reciprocal minus Strict (pp)')
    primary_ax.set_title('Primary full-success difference',fontsize=11,pad=14)
    for index,pair in enumerate(paired):
        ax=fig.add_subplot(matrix_grid[index//2,index%2])
        matrix=np.asarray([[pair['full_success_cross_settlement'][trained][settlement] for settlement in RULES] for trained in RULES])*100
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='equal')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=10,
                color='white' if matrix[i,j]>=58 else '#172831')
        ax.set_xticks([0,1],['S','R']);ax.set_yticks([0,1],['S','R']);ax.set_title(f'Seed {pair["seed"]}',fontsize=10,pad=7)
        if index//2==1:ax.set_xlabel('Settlement rule',fontsize=9)
        if index%2==0:ax.set_ylabel('Training rule',fontsize=9)
        for spine in ax.spines.values():spine.set_visible(False)
    for i,(color,marker) in enumerate(zip(SEED_COLORS,MARKERS)):
        split_ax.hlines(i-.13,0,common[i],color=color,lw=1.6)
        split_ax.scatter(common[i],i-.13,color=color,marker=marker,s=55,zorder=3)
        split_ax.hlines(i+.13,0,release[i],color=color,lw=1.2,ls='--')
        split_ax.scatter(release[i],i+.13,facecolors='white',edgecolors=color,marker=marker,s=55,zorder=3)
    split_ax.set_yticks(range(4),[str(s) for s in SEEDS]);split_ax.set_ylim(3.7,-.7);split_ax.set_xlim(*limits)
    split_ax.axvline(0,color='#777777',lw=1);split_ax.grid(axis='x',alpha=.16);split_ax.set_xlabel('Difference in full success (pp)')
    split_ax.set_title('Saved-proposal arithmetic, all signs retained',fontsize=11,pad=14)
    split_ax.legend(handles=[Line2D([0],[0],color='#526675',marker='o',lw=1.6,label='rr - sr: policies, same R settlement'),
        Line2D([0],[0],color='#526675',marker='o',markerfacecolor='white',ls='--',label='sr - ss: strict policy, rule release')],
        loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False,fontsize=9)
    fig.suptitle('Execution ability: paired differences and cross-settlement checks',fontsize=15,y=.99)
    fig.subplots_adjust(top=.83,bottom=.34)
    fig.text(.5,.17,'Middle: full success for each fixed policy under both settlement rules; S = strict, R = reciprocal; colors use the same 0-100% scale.',ha='center',fontsize=10)
    fig.text(.5,.115,'Identity: rr - ss = (rr - sr) + (sr - ss). Cross-settlement keeps the proposals fixed and uses no additional model forward.',ha='center',fontsize=10)
    fig.text(.5,.065,'These are descriptive arithmetic terms, not a causal mechanism decomposition. Four paired seeds are the independent units.',ha='center',fontsize=10,color='#555555')
    save(fig,'02_primary_and_cross_settlement')

    fig,axes=plt.subplots(2,4,figsize=(16.4,8.0),sharex=True,sharey=True)
    for column,seed in enumerate(SEEDS):
        for row,metric in enumerate(('full_success_rate','executed_partner_correct_rate')):
            ax=axes[row,column]
            for rule in RULES:ax.plot(STEPS,np.asarray(rows[seed,rule]['monitor'][metric])*100,color=RULE_COLORS[rule],
                marker='o' if rule=='strict' else 's',lw=1.8,markersize=3.5,label=rule.title())
            ax.set_ylim(0,100);ax.set_xlim(0,6000);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.15)
            ax.set_xticks([0,500,1500,3000,6000]);ax.tick_params(axis='x',labelsize=8)
            if row==0:ax.set_title(f'Seed {seed}',fontsize=11)
            if column==0:ax.set_ylabel('Full success (%)' if row==0 else 'Correct executed partner pair (%)')
            if row==1:ax.set_xlabel('Optimization update',fontsize=10)
    fig.suptitle('Execution ability during fixed-budget optimization',fontsize=15,y=.99)
    fig.legend(handles=[Line2D([0],[0],color=RULE_COLORS[rule],marker='o' if rule=='strict' else 's',lw=1.8,label=rule.title()) for rule in RULES],
        loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
    fig.subplots_adjust(top=.83,bottom=.18,hspace=.25,wspace=.14)
    fig.text(.5,.095,'All six saved updates: 0, 100, 500, 1500, 3000, 6000. Each policy is evaluated under its own training settlement rule.',ha='center',fontsize=10)
    fig.text(.5,.055,'Monitoring uses 2,976 worlds at two fixed double-holdout backgrounds; the full endpoint uses 53,568 worlds.',ha='center',fontsize=10)
    fig.text(.5,.02,'Executed partner correctness requires an actual matching pair; it is distinct from whether all three original proposals have the correct roles.',ha='center',fontsize=9,color='#555555')
    save(fig,'03_monitor_execution_ability')
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();run=Path(args.run).resolve();path=run/'execution/results.json'
    require(path.is_file(),'Completed aggregate result missing')
    digest=sha(path)
    with path.open(encoding='utf-8') as stream:result=json.load(stream)
    data=extract(result);out=Path(args.out).resolve() if args.out else run/'figures_001'
    require(not out.exists(),'Never overwrite figure output directory');out.mkdir(parents=True)
    write(out/'figure_data.json',data)
    try:
        outputs=draw(data,out);require(sha(path)==digest,'Result changed during plotting')
        import matplotlib
        receipt=dict(status='generated_visual_inspection_pending',at=datetime.now(timezone.utc).isoformat(),
            source_result_path=str(path),source_result_sha256=digest,plot_source_sha256=sha(__file__),
            plan_sha256=result['plan_sha256'],figure_data_sha256=sha(out/'figure_data.json'),outputs=outputs,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
            visual_inspection_performed=False,independent_seed_blocks=4,policy_fits=8,
            additional_model_forward_samples=0,training_updates=0,
            validation_scope='Only completed aggregate JSON. No NPZ, model, training log, prepared plan or audit file is read.')
        write(out/'figure_receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],out=str(out),outputs=list(outputs)),ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            source_result_sha256=digest,plot_source_sha256=sha(__file__),automatic_retry=False));raise


if __name__=='__main__':main()
