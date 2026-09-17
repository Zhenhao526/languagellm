"""Independent rule-by-communication formation audit, frozen before training.

Only pinned historical independent helpers are imported. Every saved evaluation
is freshly forwarded with independent PL features and legal two-window routing.
Q, exact shuffle, physical settlement and conditional events use the previously
independent audited definitions. Trajectory primary and message summaries are
implemented here without importing current production modules.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_key]='1'
import argparse,json,math,platform,re,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
from research_program.triadic_need_response_study import audit_response as response

prior=response.prior;env=prior.env;message=prior.message;old=prior.old
require,read,sha,array_sha,json_bytes,close,compare,finite_tree=(prior.require,prior.read,prior.sha,prior.array_sha,prior.json_bytes,prior.close,prior.compare,prior.finite_tree)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=tuple(range(60101,60117));RULES=('strict','reciprocal');LIVES=(True,False);PARTS=env.PARTS;STEPS=(0,100,500,1500,3000,6000);TARGET=PARTS[-1]
RESPONSE_SHA='589ca516ff962f56f3bda0e23d253bd48a02b2e513d03d84d310eae1d51527ac';ORIGINAL_SHA=response.ORIGINAL_SHA
# Protocol quantile independently checked by64-node Gaussian quadrature below.
T15=2.1314495455597715
CONDITIONAL=('conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability','conditional_full_posterior_mass')


def references():
    refs=response.references();p=Path(response.__file__).resolve();require(sha(p)==RESPONSE_SHA,'Pinned independent Q and settlement source');refs[str(p)]=RESPONSE_SHA;return refs


def condition(rule,live):return rule+'_PL_'+('live' if live else 'silent')
def name(seed,rule,live):return f'seed_{seed}_{condition(rule,live)}'


def adjusted_rand(left,right):
    """Integer contingency pair counts; degenerate identical partitions give1."""
    x=np.asarray(left);y=np.asarray(right);require(x.shape==y.shape and x.ndim==1 and len(x)>0,'ARI paired nonempty panel')
    _,xi,xc=np.unique(x,return_inverse=True,return_counts=True);_,yi,yc=np.unique(y,return_inverse=True,return_counts=True)
    _,cells=np.unique(xi.astype(np.int64)*len(yc)+yi,return_counts=True)
    choose=lambda counts:sum(int(k)*(int(k)-1)//2 for k in counts)
    a=choose(xc);b=choose(yc);observed=choose(cells);pairs=len(x)*(len(x)-1)//2
    # Multiply the standard ARI numerator and denominator by 2*pairs.
    numerator=2*(observed*pairs-a*b);denominator=(a+b)*pairs-2*a*b
    if denominator==0:return 1.
    return numerator/denominator


def entropy_bits(labels):
    _,count=np.unique(labels,return_counts=True);p=count/len(labels);return float(-np.dot(p,np.log2(p)))


def encode_packets(messages):
    v=np.asarray(messages);require(v.ndim==4 and v.shape[1:]==(2,3,4) and v.dtype.kind in 'iu' and np.all((v>=0)&(v<8)),'Complete legal packet panel')
    return ((v[...,0].astype(np.int64)*8+v[...,1])*8+v[...,2])*8+v[...,3]


def t15_cdf(value):
    """Independent smooth-density quadrature, used only in pure preflight."""
    nodes,weights=np.polynomial.legendre.leggauss(64);x=(nodes+1)*value/2
    coefficient=math.gamma(8)/(math.sqrt(15*math.pi)*math.gamma(7.5))
    return .5+float(np.dot(weights,coefficient*(1+x*x/15)**-8))*value/2


def statistics(values):
    x=np.asarray(values,dtype=np.float64);require(x.shape==(16,) and np.isfinite(x).all(),'Sixteen independent society values')
    mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/4;half=T15*se
    return dict(n=16,mean=mean,sample_sd=sd,standard_error=se,t_critical=T15,df=15,interval_level=.95,ci95_lower=mean-half,ci95_upper=mean+half)


def trajectory_contrast(cells):
    """One society's four six-point arrays, differences then centered time AUC."""
    require(set(cells)==set(product(RULES,LIVES)),'Complete four-cell society')
    v={k:np.asarray(x,dtype=np.float64) for k,x in cells.items()};require(all(x.shape==(6,) and np.isfinite(x).all() for x in v.values()),'Six actual checkpoint scores')
    s=v['strict',True]-v['strict',False];r=v['reciprocal',True]-v['reciprocal',False];did=r-s;centered=did-did[0]
    dt=np.diff(STEPS);auc=float(np.dot(dt,(centered[1:]+centered[:-1])*.5)/6000)
    return dict(communication_strict=s.tolist(),communication_reciprocal=r.tolist(),DiD=did.tolist(),centered_DiD=centered.tolist(),baseline_DiD=float(did[0]),raw_AUC=float(np.dot(dt,(did[1:]+did[:-1])*.5)/6000),centered_AUC=auc,endpoint_DiD=float(did[-1]),endpoint_centered_DiD=float(centered[-1]))


def training_row(row,seed,rule,live,update,expected):
    require((row['seed'],row['rule'],row['condition'],row['update'])==(seed,rule,condition(rule,live),update),'Four-cell training identity')
    prior.training_row(dict(row,condition='FI_silent'),seed,rule,update,expected)


def evaluate_saved(entry,path,states,rule,live,networks,cases):
    path=Path(path);n=len(states);ids=np.arange(n,dtype=np.int64)
    require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['data_sha256'],'Evaluation path/hash')
    value=env.load_npz(path)
    require(value['states'].dtype==np.int16 and np.array_equal(value['states'],states),'Full original state support')
    require(value['state_indices'].dtype==np.int64 and np.array_equal(value['state_indices'],ids),'Canonical complete evaluation indices')
    response.validate_partition(cases,value['states'],value['state_indices'])
    p=value['action_probabilities'];a=value['action_indices'];m=value['messages']
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Full probability array');close(p.sum(-1),np.ones((n,3)),'Probability normalization')
    require(a.dtype==np.int16 and a.shape==(n,3) and np.array_equal(a,p.argmax(-1)),'Greedy action identity')
    require(m.dtype==np.int8 and m.shape==(n,2,3,4) and np.all((m>=0)&(m<8)),'Two-window symbol domain')
    native=prior.settle(states,a,rule);common=prior.settle(states,a,'reciprocal');strict=prior.settle(states,a,'strict')
    for tag,settled in (('',native),('common_reciprocal__',common),('strict__',strict)):
        for key,v in settled.items():require(value[tag+key].dtype==v.dtype and np.array_equal(value[tag+key],v),'Independent same-action settlement '+tag+key)
    require(set(value)==set(native)|{'common_reciprocal__'+k for k in common}|{'strict__'+k for k in strict}|set(CONDITIONAL)|{'states','state_indices','messages','action_probabilities','action_indices'},'Exact full evaluation schema')
    for key in CONDITIONAL:require(value[key].shape==(n,) and value[key].dtype==np.float64 and np.isfinite(value[key]).all(),'Conditional array domain '+key)
    errors=defaultdict(float);ties=0
    for begin in range(0,n,1024):
        sl=slice(begin,begin+1024);chunk=states[sl];mm,pp,logs,tt=old.forward_final(networks,chunk,'PL',live);ties+=tt
        require(np.array_equal(mm,m[sl]) and np.array_equal(pp.argmax(-1),a[sl]),'Fresh complete two-window message/action replay')
        errors['probability']=max(errors['probability'],close(pp,p[sl],'Fresh full evaluation action probabilities'))
        exact=prior.conditional_statistics(p[sl],env.rewards(chunk),rule,logs)
        for key in CONDITIONAL:errors['conditional']=max(errors['conditional'],close(value[key][sl],exact[key],'Independent native conditional event '+key))
    calculated=dict(native=prior.summarize(states,a,rule),strict=prior.summarize(states,a,'strict'),common_reciprocal=prior.summarize(states,a,'reciprocal'),need_response=dict(native=response.metrics(cases,native['actual_pair_index']),common_reciprocal=response.metrics(cases,common['actual_pair_index'])))
    calculated.update(information='PL',live=live,rule=rule,scope='complete_partition',worlds=n,forward_module_samples=9*n,expected_reward_given_greedy_messages=float(value['conditional_exact_expected_reward'].mean()),full_probability_given_greedy_messages=float(value['conditional_exact_full_success_probability'].mean()),execution_probability_given_greedy_messages=float(value['conditional_exact_execution_probability'].mean()),full_posterior_mass_given_greedy_messages=float(value['conditional_full_posterior_mass'].mean()),state_indices_sha256=array_sha(ids))
    compare(entry,calculated,'Independent both-settlement summary/Q/')
    return value,dict(sha256=sha(path),worlds=n,independent_module_samples=9*n,max_errors=dict(errors),sender_ties=ties),calculated


def budget(specs,cases):
    worlds=sum(s['world_count'] for s in specs.values());target=specs[TARGET]['world_count'];runs=64;updates=384000;ev=runs*(6*target+worlds-target)
    require((worlds,target,ev*9)==(774144,53568,600182784),'Complete fixed evaluation support')
    return dict(runs=runs,independent_paired_seed_blocks=16,training_updates=updates,training_world_samples=updates*256,message_trajectories=updates*256*2,categorical_symbol_samples=updates*256*2*24,
        training_forward_module_samples=updates*256*2*9,checkpoints=384,training_log_rows=updates,full_target_trajectory_files=384,other_final_files=192,evaluation_files=576,final_target_aliases=64,
        additional_final_target_forward_samples=0,closed_files=0,monitor_subset_files=0,evaluation_worlds=ev,evaluation_forward_module_samples=ev*9,total_forward_module_samples=updates*256*2*9+ev*9,
        complete_need_response_records=1152,need_response_additional_forward_samples=0,generated_NPZ_files=960)


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution';refs=references();artifacts=dict(refs)
    require(re.fullmatch('[0-9a-f]{64}',expected_plan_sha) is not None,'Explicit frozen plan hash')
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        if digest is not None:require(actual==digest,'Changed artifact '+str(path))
        artifacts[str(path)]=actual;return actual
    for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json',HERE/'audit_freeze_001.json'):bind(path)
    plan=read(run/'plan.json');prepared=read(run/'prepared.json');result=read(execution/'results.json');status=read(execution/'status.json');finite_tree(result)
    require(status['status']==result['status']=='completed' and not (execution/'failure.json').exists(),'Only completed formal training')
    require(status['results_sha256']==sha(execution/'results.json'),'Completed results hash')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256']==result['plan_sha256']==read(execution/'started.json')['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==plan['prepared_sha256']==read(run/'freeze.json')['prepared_sha256'],'Prepared chain')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Exact numerical runtime')
    conditions=[condition(r,l) for r,l in product(RULES,LIVES)]
    compare(plan['config'],dict(seeds=list(SEEDS),conditions=conditions,execution_rules=list(RULES),partitions=list(PARTS),updates=6000,batch_size=256,checkpoints=list(STEPS),features=54,dtype='float64',
        learning_rate=.001,global_gradient_clip=5.,entropy_initial=.001,entropy_zero_after_updates=1000,sender_entropy_coefficient=0,trajectories_per_state=2,sender_windows=2,sender_tokens_per_window=4,actions_per_actor=17,
        worker_count=4,multiprocessing_start_method='spawn',dimensions=dict(sender1=[54,64,64,32],sender2=[153,64,64,32],action=[252,64,64,17]),
        primary='six_point_native_Q_communication_rule_DiD_minus_initial_trapezoid_over6000_then_mean16_blocks',information='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
        evaluation_checkpoints=list(STEPS),closed_channel_interventions=0,monitor_subsets=0,automatic_followon_experiment=False),'Frozen core config/')
    af=read(HERE/'audit_freeze_001.json');require(af['status']=='frozen_before_main_training' and af['plan_sha256']==expected_plan_sha,'Pretraining source freeze')
    require(af['at']<read(execution/'started.json')['at'],'Independent freeze precedes main')
    require(set(af['source_sha256'])=={str(HERE/'audit_execution.py'),str(HERE/'test_audit.py'),str(HERE/'audit_preflight_001.json')},'Independent freeze coverage')
    for p,d in af['source_sha256'].items():bind(p,d)
    for p,d in plan['source_sha256'].items():bind(p,d);bind(run/'source_snapshot'/Path(p).relative_to(ROOT),d)
    for p,d in plan['inputs_sha256'].items():bind(p,d)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json';bind(original,ORIGINAL_SHA);specs=read(original)['partitions'];independent,_,_=env.independent_specs()
    require(prepared['schema']=='triadic_rule_formation_v1' and prepared['source_prepared_sha256']==ORIGINAL_SHA and prepared['independent_initializations']==16 and prepared['partitions']==specs,'Anchored original support')
    for p in PARTS:compare(specs[p],independent[p],'Independent full partition/')
    cc={p:response.build_cases(specs[p]) for p in PARTS};compare(prepared['need_response_cases'],cc,'Independent Q support/')
    require(prepared['budget']==result['budget']==budget(specs,cc),'Exact fixed whole-study budget')
    require([(r['seed'],r['condition']) for r in result['runs']]==[(s,c) for s in SEEDS for c in conditions],'Canonical64 four-cell runs')
    states={p:env.packed(specs[p]) for p in PARTS};hashes={}
    for part,s in states.items():
        x=env.features(s,'PL');require(np.all(x[:,:,53]==0),'FI flag hidden')
        for who,other in product(range(3),repeat=2):
            if who!=other:require(np.all(x[:,who,7*other:7*other+7]==0),'Other needs fully hidden')
        hashes[part]=dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)),x_PL=array_sha(x))
    del x
    require(result['array_hashes']==hashes,'Independent full world/observation arrays')
    by={(r['seed'],r['rule'],r['live']):r for r in result['runs']};require(set(by)==set(product(SEEDS,RULES,LIVES)),'Unique64 run identity')
    counts=Counter();errors=defaultdict(float);evaluations={};computed_runs=[];paths=set();initial_hashes={}
    for seed in SEEDS:
        bind(execution/f'seed_{seed}_arrays.json');require(read(execution/f'seed_{seed}_arrays.json')['array_hashes']==hashes,'Worker full arrays')
        bind(execution/f'seed_{seed}_status.json');require(read(execution/f'seed_{seed}_status.json')['status']=='completed','Completed seed block')
        rng_at={};initial=[];initial_reference={};w1_reference=None
        for rule,live in product(RULES,LIVES):
            row=by[seed,rule,live];directory=execution/name(seed,rule,live);require(row['updates']==6000,'Exactly6000 updates')
            for file in ('result.json','training.jsonl','trajectory.jsonl','status.json'):bind(directory/file)
            require(read(directory/'result.json')==row,'Per-run result identity');status=read(directory/'status.json');require(status['status']=='completed' and status['result_sha256']==sha(directory/'result.json'),'Run completed result hash')
            require(sha(directory/'training.jsonl')==row['training_log_sha256'],'Training log hash');trajectory=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
            require(trajectory==row['trajectory'] and [q['update'] for q in trajectory]==list(STEPS),'Exact six checkpoint records')
            rng_at[rule,live]={};previous_messages=None;recomputed=[]
            for step,checkpoint in zip(STEPS,trajectory):
                cp=directory/f'checkpoint_{step:04d}.npz';require(Path(checkpoint['checkpoint_path'])==cp,'Checkpoint path identity');bind(cp,checkpoint['checkpoint_sha256']);paths.add(cp)
                networks,wr,mr=message.checkpoint(cp,step,seed);rng_at[rule,live][step]=(wr,mr);counts['checkpoints']+=1;parameter_hash=message.network_hash(networks)
                if step==0:require(parameter_hash==row['initial_parameter_sha256'],'Initial identity');initial.append(parameter_hash)
                if step==6000:require(parameter_hash==row['final_parameter_sha256'] and sha(cp)==row['final_checkpoint_sha256'],'Final parameters')
                path=directory/f'trajectory_{step:04d}_{TARGET}.npz';value,receipt,summary=evaluate_saved(checkpoint['evaluation'],path,states[TARGET],rule,live,networks,cc[TARGET]);bind(path,receipt['sha256']);paths.add(path);evaluations[str(path)]=receipt
                counts['trajectory_files']+=1;counts['evaluation_worlds']+=receipt['worlds'];counts['independent_module_samples']+=receipt['independent_module_samples'];counts['response_records']+=2
                for k,v in receipt['max_errors'].items():errors[k]=max(errors[k],v)
                if step==0:
                    if w1_reference is None:w1_reference=value['messages'][:,0].copy()
                    else:require(np.array_equal(w1_reference,value['messages'][:,0]),'Four-cell identical initial W1')
                    if live not in initial_reference:initial_reference[live]=(value['messages'].copy(),value['action_probabilities'].copy())
                    else:
                        require(np.array_equal(initial_reference[live][0],value['messages']) and np.array_equal(initial_reference[live][1],value['action_probabilities']),'Rules do not alter initial fixed policy rollout')
                snapshot=message_snapshot(value['messages'],specs[TARGET]);compare(checkpoint['message_snapshot'],snapshot,'Independent snapshot entropy/')
                counts['message_snapshots']+=1
                if step==0:require(checkpoint['message_transition'] is None,'No preceding t0 transition')
                else:
                    transition=message_transition(previous_messages,value['messages'],specs[TARGET]);compare(checkpoint['message_transition'],transition,'Independent consecutive message form/');counts['message_transitions']+=1
                previous_messages=value['messages'].copy();recomputed.append(dict(update=step,evaluation=summary))
                require(message.network_hash(networks)==parameter_hash,'Forward leaves parameters unchanged')
                print(json.dumps(dict(stage='trajectory_audited',seed=seed,rule=rule,live=live,update=step)),flush=True)
            require(set(row['final'])==set(PARTS),'All final partitions');alias=row['final'][TARGET];endpoint=trajectory[-1]['evaluation']
            require(alias.get('alias_of')=='trajectory_update_6000' and alias.get('additional_forward_module_samples')==0 and {k:v for k,v in alias.items() if k not in ('alias_of','additional_forward_module_samples')}==endpoint,'Exact6000 target alias without forward');counts['target_aliases']+=1
            for part in PARTS:
                if part==TARGET:continue
                path=directory/f'final_{part}.npz';value,receipt,summary=evaluate_saved(row['final'][part],path,states[part],rule,live,networks,cc[part]);bind(path,receipt['sha256']);paths.add(path);evaluations[str(path)]=receipt
                counts['other_final_files']+=1;counts['evaluation_worlds']+=receipt['worlds'];counts['independent_module_samples']+=receipt['independent_module_samples'];counts['response_records']+=2
                for k,v in receipt['max_errors'].items():errors[k]=max(errors[k],v)
            require(set(directory.glob('*.npz'))=={p for p in paths if p.parent==directory},'Exact15 NPZ per run')
            computed_runs.append(dict(seed=seed,rule=rule,live=live,condition=condition(rule,live),trajectory=recomputed))
        require(len(set(initial))==1,'Four independent cells share initial values');initial_hashes[seed]=initial[0]
        wr=np.random.default_rng(np.random.SeedSequence([seed,200]));mr={f't{t}_w{w}_a{a}':np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300])) for t,w,a in product(range(2),range(2),range(3))}
        def check_rng(step):
            current=(wr.bit_generator.state,{k:g.bit_generator.state for k,g in mr.items()})
            for cell in product(RULES,LIVES):require(rng_at[cell][step]==current,'All four checkpoint random stream states')
        streams={(r,l):(execution/name(seed,r,l)/'training.jsonl').open() for r,l in product(RULES,LIVES)}
        try:
            check_rng(0)
            for update in range(1,6001):
                uniform=wr.random((256,3));ids=env.sample(specs['train'],uniform);mu=np.empty((2,256,2,3,4))
                for t,w,a in product(range(2),range(2),range(3)):mu[t,:,w,a]=mr[f't{t}_w{w}_a{a}'].random((256,4))
                expected=dict(world_uniforms_sha256=array_sha(uniform),batch_indices_sha256=array_sha(ids),batch_states_sha256=array_sha(states['train'][ids]),sample_uniforms_sha256=array_sha(mu))
                for (r,l),stream in streams.items():
                    line=stream.readline();require(bool(line),'Missing training log row');training_row(json.loads(line),seed,r,l,update,expected);counts['training_rows']+=1
                if update in STEPS:check_rng(update)
            require(all(stream.read()=='' for stream in streams.values()),'No extra training rows')
        finally:
            for stream in streams.values():stream.close()
        print(json.dumps(dict(stage='paired_seed_audited',seed=seed)),flush=True)
    require(len(set(initial_hashes.values()))==16,'Sixteen distinct independent initializations')
    expected=dict(checkpoints=384,trajectory_files=384,other_final_files=192,target_aliases=64,evaluation_worlds=66686976,independent_module_samples=600182784,response_records=1152,message_snapshots=384,message_transitions=320,training_rows=384000)
    require(dict(counts)==expected,'Full planned independent audit scope');calculated=primary(computed_runs);compare(result['primary'],calculated,'Independent centered DiD time AUC and16-seed interval/')
    measured=dict(training_forward_module_samples=1769472000,evaluation_forward_module_samples=counts['independent_module_samples'],checkpoints=counts['checkpoints'],evaluation_files=counts['trajectory_files']+counts['other_final_files'],evaluation_worlds=counts['evaluation_worlds'],final_target_aliases=64)
    require(result['measured_budget']==measured,'Exact measured main accounting');require(len(paths)==960 and set(execution.rglob('*.npz'))==paths,'Every384 checkpoint and576 evaluation NPZ covered')
    require({p for p in execution.glob('seed_*_*') if p.is_dir()}=={execution/name(s,r,l) for s,r,l in product(SEEDS,RULES,LIVES)},'No extra run directories')
    for p,d in artifacts.items():require(sha(p)==d,'Artifact changed during audit '+p)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,scope=dict(counts),max_errors=dict(errors),artifacts_sha256=artifacts,historical_helpers_sha256=refs,evaluations=evaluations,primary=calculated,
        limits=['All576 complete evaluation files freshly replayed, including all384 trajectory checkpoints;600182784 independent evaluation module samples.',
        'All384 checkpoint optimizer/RNG schemas and384000 training log rows checked; paired world/message uniform streams reconstructed. No gradient or optimizer update replay; sampled message hashes are not regenerated.',
        'Same action strict/native/common reciprocal settlements, complete Q/exact shuffle, all message form summaries,16 paired centered time-AUC values and approximate t15 intervals independently recomputed.',
        'Native-rule scoring includes changed physical feasibility as well as learning. Common reciprocal scoring is auxiliary and cannot replace the fixed primary.',
        'Message entropy/stability/ARI describe symbol form, not semantics, composition, language formation, or human language origin.'])


def primary(runs):
    require(len(runs)==64,'Exactly64 society/condition trajectories')
    by={(r['seed'],r['rule'],r['live']):r for r in runs};require(set(by)==set(product(SEEDS,RULES,LIVES)) and len(by)==64,'Unique complete grid')
    measures=dict(native_Q=('native','Q'),native_Q_excess=('native','Q_excess'),common_reciprocal_Q=('common_reciprocal','Q'),common_reciprocal_Q_excess=('common_reciprocal','Q_excess'))
    rows=[]
    for seed in SEEDS:
        cells={}
        for rule,live in product(RULES,(False,True)):
            run=by[seed,rule,live];require([r['update'] for r in run['trajectory']]==list(STEPS),'Ordered six-checkpoint primary')
            values={}
            for key,(settlement,metric) in measures.items():
                v=[r['evaluation']['need_response'][settlement][metric] for r in run['trajectory']]
                require(all(type(x) in (float,int) and np.isfinite(x) and (-1 if metric=='Q_excess' else 0)<=x<=1 for x in v),'Valid complete Q/Q-excess')
                values[key]=v
            cells[condition(rule,live)]=dict(rule=rule,live=live,**values)
        summaries={key:trajectory_contrast({(v['rule'],v['live']):v[key] for v in cells.values()}) for key in measures}
        rows.append(dict(seed=seed,cells=cells,measures=summaries))
    stats=statistics([r['measures']['native_Q']['centered_AUC'] for r in rows]);fields=('raw_AUC','centered_AUC','baseline_DiD','endpoint_DiD','endpoint_centered_DiD')
    auxiliary={key:{field:statistics([r['measures'][key][field] for r in rows]) for field in fields} for key in measures}
    return dict(name='mean_baseline_centered_native_Q_communication_DiD_time_AUC',partition=TARGET,independent_societies=16,training_runs=64,checkpoints=list(STEPS),horizon_updates=6000,primary_measure='native_Q',primary_statistic='centered_AUC',statistics=stats,mean_centered_AUC=stats['mean'],by_seed=rows,auxiliary_statistics=auxiliary)


def panel_shape(messages,spec):
    n=spec['world_count'];N=len(spec['needs']);B=len(spec['layouts'])*len(spec['private_sites']);m=np.asarray(messages)
    require(n==N*B and spec['state_order']=='need-major, then layout, then owner','Complete canonical message panel')
    require(m.shape==(n,2,3,4),'Complete panel shape');return m,encode_packets(m),N,B


def message_snapshot(messages,spec):
    _,codes,N,B=panel_shape(messages,spec);owners=len(spec['private_sites']);panels=[]
    for actor,window in product(range(3),range(2)):
        v=codes[:,window,actor].reshape(N,B);backgrounds=[]
        for b in range(B):
            labels=v[:,b];h=entropy_bits(labels);count=len(np.unique(labels))
            backgrounds.append(dict(background_index=b,layout_index=b//owners,owner_index=b%owners,need_worlds=N,entropy_bits=h,effective_packet_count=float(2**h),observed_packet_count=count,constant_code=count==1))
        h=float(np.mean([bg['entropy_bits'] for bg in backgrounds]));panels.append(dict(actor=actor,window=window,conditional_entropy_bits=h,conditional_effective_packet_count=float(2**h),backgrounds=backgrounds,global_observed_packet_count=len(np.unique(v))))
    return dict(schema='natural_message_snapshot_v1',worlds=N*B,need_worlds=N,background_count=B,panel_order='actor,window',panels=panels)


def message_transition(before,after,spec):
    first,a,N,B=panel_shape(before,spec);last,z,NN,BB=panel_shape(after,spec);require((NN,BB)==(N,B),'Same complete panel at adjacent times');owners=len(spec['private_sites']);panels=[]
    for actor,window in product(range(3),range(2)):
        left=a[:,window,actor];right=z[:,window,actor];abg=left.reshape(N,B);zbg=right.reshape(N,B);backgrounds=[]
        for b in range(B):
            nc=len(np.unique(abg[:,b]));mc=len(np.unique(zbg[:,b]));backgrounds.append(dict(background_index=b,layout_index=b//owners,owner_index=b%owners,ARI=adjusted_rand(abg[:,b],zbg[:,b]),observed_packets_before=nc,observed_packets_after=mc,constant_before=nc==1,constant_after=mc==1,entropy_bits_before=entropy_bits(abg[:,b]),entropy_bits_after=entropy_bits(zbg[:,b])))
        nc=len(np.unique(left));mc=len(np.unique(right));panels.append(dict(actor=actor,window=window,packet_equal_rate=float(np.mean(left==right)),token_hamming_fraction=float(np.mean(first[:,window,actor]!=last[:,window,actor])),global_ARI=adjusted_rand(left,right),mean_background_ARI=float(np.mean([bg['ARI'] for bg in backgrounds])),global_observed_packets_before=nc,global_observed_packets_after=mc,global_constant_before=nc==1,global_constant_after=mc==1,conditional_entropy_bits_before=float(np.mean([bg['entropy_bits_before'] for bg in backgrounds])),conditional_entropy_bits_after=float(np.mean([bg['entropy_bits_after'] for bg in backgrounds])),backgrounds=backgrounds))
    return dict(schema='adjacent_natural_message_transition_v1',worlds=N*B,need_worlds=N,background_count=B,panel_order='actor,window',panels=panels)


def freeze(run,expected_plan_sha):
    run=Path(run).resolve();out=HERE/'audit_freeze_001.json';require(not out.exists(),'Never overwrite independent freeze');require(not (run/'execution').exists(),'Only freeze before training')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256'],'Exact prepared plan before training');plan=read(run/'plan.json')
    require(sha(run/'prepared.json')==plan['prepared_sha256'],'Prepared hash');require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Prepared runtime')
    receipt=read(HERE/'audit_preflight_001.json');require(receipt['status']=='passed' and receipt['real_neural_forward_samples']==0 and receipt['formal_output_files_read']==0,'Passed pure preflight')
    for path in (HERE/'audit_execution.py',HERE/'test_audit.py'):require(receipt['source_sha256'][str(path)]==sha(path),'Current tested audit source')
    for p,d in plan['source_sha256'].items():require(sha(p)==d,'Current main source identity')
    value=dict(status='frozen_before_main_training',at=datetime.now(timezone.utc).isoformat(),plan_sha256=expected_plan_sha,source_sha256={str(p):sha(p) for p in (HERE/'audit_execution.py',HERE/'test_audit.py',HERE/'audit_preflight_001.json')},historical_helpers_sha256=references(),real_neural_forward_samples=0,formal_output_files_read=0,expected_independent_evaluation_module_samples=600182784,training_gradient_replay=False)
    out.write_bytes(json_bytes(value));return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--plan-sha',required=True);parser.add_argument('--out');parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:
        require(args.out is None,'Freeze uses the fixed audit receipt path');print(json.dumps(freeze(args.run,args.plan_sha)));return
    require(args.out is not None,'Audit output path required');out=Path(args.out).resolve();require(not out.exists(),'Never overwrite formal audit');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),run=args.run,expected_plan_sha256=args.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        value=audit(args.run,args.plan_sha);value.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(value));print(json.dumps({k:value[k] for k in ('status','scope','max_errors','elapsed_seconds')}))
    except BaseException as error:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)));raise

if __name__=='__main__':main()
