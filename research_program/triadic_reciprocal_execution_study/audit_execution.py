"""Independent completed-run audit; no production reciprocal module imported.

Pinned old independent helpers supply states, FI observations, native structural
reward tables, silent routing, checkpoints and nine-head endpoint forward.
Settlement, event probabilities, summaries, cross-settlement and primary are
implemented here. Training RNGs are rebuilt; gradients/optimizer are not replayed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import json,math,platform,re,time,traceback
import numpy as np
from research_program.triadic_partial_payoff_study import audit_execution as old

env,message=old.env,old.message
require,read,sha,array_sha,json_bytes,close,compare,finite_tree=(old.require,old.read,old.sha,old.array_sha,old.json_bytes,old.close,old.compare,old.finite_tree)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=(57101,57102,57103,57104);RULES=('strict','reciprocal');PARTS=env.PARTS;STEPS=env.STEPS
PAIRS=((0,1),(0,2),(1,2));PAIR_NAMES=('AB','AC','BC');TOL=2e-12
OLD_AUDIT_SHA='5755f98d09b9499abd5a2b2cb178baca9c7ff5fb0733dcbffb13d4121107e244'
ORIGINAL_PREPARED_SHA='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'


def references():
    refs=old.references();path=Path(old.__file__).resolve()
    require(sha(path)==OLD_AUDIT_SHA,'Pinned independent forward helper changed')
    refs[str(path)]=OLD_AUDIT_SHA;return refs


def name(seed,rule):return f'seed_{seed}_{rule}_FI_silent'


def settle(states,actions,rule):
    """Physical execution using only submitted proposals and native acceptance."""
    require(rule in RULES,'Unknown execution rule');states=np.asarray(states);actions=np.asarray(actions)
    n=len(states)
    require(states.shape==(n,10) and n>0 and states.dtype.kind in 'iu','State shape/type')
    require(np.all((states[:,:3]>=0)&(states[:,:3]<24)),'Need domain')
    require(np.all(np.sort(states[:,3:7],axis=1)==np.arange(4)) and np.all(np.sort(states[:,7:],axis=1)==np.arange(1,4)),'Layout/owner permutations')
    require(actions.shape==(n,3) and actions.dtype.kind in 'iu' and np.all((actions>=0)&(actions<17)),'Action shape/domain')
    roles=old.role_indices(actions);raw=np.maximum(actions.astype(np.int64)-1,0)
    sites=raw//4;dest=(raw//2)%2;active=actions!=0
    executed=np.zeros((n,3),bool);pair=np.full(n,-1,np.int8)
    site=np.full(n,-1,np.int8);material=np.full(n,-1,np.int8);destination=np.full(n,-1,np.int8)
    for index,(i,j) in enumerate(PAIRS):
        mask=active[:,i]&active[:,j]&(roles[:,i]==j)&(roles[:,j]==i)&(sites[:,i]==sites[:,j])&(dest[:,i]==dest[:,j])
        if rule=='strict':mask &= active.sum(1)==2
        require(np.all(pair[mask]==-1),'Multiple reciprocal pairs')
        executed[mask,i]=True;executed[mask,j]=True;pair[mask]=index
        site[mask]=sites[mask,i];destination[mask]=dest[mask,i]
        material[mask]=states[np.flatnonzero(mask),3+sites[mask,i]]
    physical=pair>=0;satisfied=np.zeros((n,3),bool)
    for actor in range(3):
        rows=np.flatnonzero(executed[:,actor])
        satisfied[rows,actor]=env.ACCEPT[states[rows,actor],material[rows],destination[rows]]
    unexecuted=active&~executed
    return dict(executed=executed,satisfied=satisfied,greedy_reward=satisfied.sum(1)/2.,
        executed_roles=np.where(executed,roles,-1).astype(np.int8),actual_pair_index=pair,
        ignored_proposal=unexecuted&physical[:,None],proposal_roles=roles,unexecuted_proposal=unexecuted,
        executed_site=site,executed_material=material,executed_destination=destination)


def conditional_statistics(probabilities,native,rule,log_probabilities=None):
    """Independent weighted event log masses; ignored actor integrates to one."""
    require(rule in RULES,'Conditional rule');p=np.asarray(probabilities);r=np.asarray(native)
    require(p.ndim==3 and p.shape[1:]==(3,17) and r.shape==(len(p),24),'Conditional shapes')
    require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Conditional probabilities')
    close(p.sum(-1),np.ones(p.shape[:2]),'Probability normalization')
    require(np.isin(r,(0.,.5,1.)).all() and np.all((r==1).sum(1)==1),'Native reward support')
    with np.errstate(divide='ignore'):lp=np.log(p) if log_probabilities is None else np.asarray(log_probabilities)
    require(lp.shape==p.shape and not np.isnan(lp).any() and np.all(lp<=TOL),'Log probability domain')
    if log_probabilities is not None:
        require(np.isfinite(lp).all(),'Stable log probabilities finite');close(np.exp(lp),p,'Log/probability consistency')
    event=np.ones(r.shape);log_event=np.zeros(r.shape)
    for actor in range(3):
        use=np.ones(24,bool) if rule=='strict' else env.JOINT[:,actor]!=0
        choices=env.JOINT[use,actor]
        selected_lp=lp[:,actor,choices]
        if log_probabilities is None:
            require(np.isfinite(selected_lp[r[:,use]>0]).all(),'Monitor posterior cannot be verified after positive-event probability underflow')
        event[:,use]*=p[:,actor,choices];log_event[:,use]+=selected_lp
    log_mass=np.full(r.shape,-np.inf);positive=r>0
    log_mass[positive]=log_event[positive]+np.log(r[positive])
    log_J=np.logaddexp.reduce(log_mass,axis=1);require(np.isfinite(log_J).all(),'Finite expected-reward log')
    posterior=np.exp(log_mass-log_J[:,None]);close(posterior.sum(1),np.ones(len(p)),'Reward posterior normalization')
    require(np.all(event.sum(1)<=1+TOL),'Disjoint physical event probability')
    return dict(conditional_exact_expected_reward=(event*r).sum(1),
        conditional_exact_full_success_probability=(event*(r==1)).sum(1),
        conditional_exact_execution_probability=event.sum(1),conditional_full_posterior_mass=(posterior*(r==1)).sum(1),
        posterior=posterior,log_expected_reward=log_J)


METRIC_SCOPE={
 'content_axes':'Actual execution and equality to the unique global correct-plan axis; no correct-partner gate; denominator all worlds.',
 'proposal_roles':'All three original partner/wait proposals equal the original unique-plan roles, regardless of physical execution.',
 'executed_partner':'An actual mutual pair exists and equals the unique correct pair; location/destination need not be correct.',
 'ignored':'Nonwait third proposal while another pair executes; failed unmatched proposals separately counted as unexecuted.',
 'actor_rates':'Denominator3 times all worlds; never conditional on having proposed.'}


def summarize(states,actions,rule,native=None):
    data=settle(states,actions,rule);r=env.rewards(states) if native is None else np.asarray(native)
    require(r.shape==(len(states),24) and np.all((r==1).sum(1)==1),'One correct plan for summaries')
    target=(r==1).argmax(1);target_pair=target//8;target_site=(target%8)//2;target_dest=target%2
    target_material=states[np.arange(len(states)),3+target_site];target_roles=old.role_indices(env.JOINT[target])
    physical=data['actual_pair_index']>=0;actual=data['executed_material']
    events=dict(full_success_rate=data['greedy_reward']==1,physical_execution_rate=physical,
        executed_partner_correct_rate=physical&(data['actual_pair_index']==target_pair),
        proposal_role_success_rate=np.all(data['proposal_roles']==target_roles,axis=1),
        kind_correct_rate=physical&(actual//2==target_material//2),length_correct_rate=physical&(actual%2==target_material%2),
        destination_correct_rate=physical&(data['executed_destination']==target_dest),
        material_identity_correct_rate=physical&(actual==target_material),ignored_proposal_world_rate=data['ignored_proposal'].any(1))
    require(np.all(~events['full_success_rate']|events['executed_partner_correct_rate']),'Full success has correct pair')
    def group(mask):
        n=int(mask.sum());row=dict(worlds=n,reward_mean=float(data['greedy_reward'][mask].mean()) if n else None)
        row.update({k:float(v[mask].mean()) if n else None for k,v in events.items()})
        row.update(ignored_proposal_agent_rate=float(data['ignored_proposal'][mask].mean()) if n else None,
            unexecuted_proposal_agent_rate=float(data['unexecuted_proposal'][mask].mean()) if n else None)
        return row
    def histogram(values,key):
        u,c=np.unique(values,axis=0,return_counts=True)
        return [{key:x.tolist(),'worlds':int(count)} for x,count in zip(u,c)]
    result=group(np.ones(len(states),bool));result.update(rule=rule,
        reward_counts={str(v):int((data['greedy_reward']==v).sum()) for v in (0.,.5,1.)},
        raw_joint_action_counts=histogram(actions,'action_indices'),raw_proposal_role_counts=histogram(data['proposal_roles'],'partner_indices'),
        raw_executed_role_counts=histogram(data['executed_roles'],'partner_indices'),
        actual_pair_counts={key:int((data['actual_pair_index']==i).sum()) for i,key in ((-1,'none'),(0,'AB'),(1,'AC'),(2,'BC'))},
        ignored_proposal_counts_by_actor=data['ignored_proposal'].sum(0).astype(int).tolist(),
        true_pair_strata={pair:group(target_pair==i) for i,pair in enumerate(PAIR_NAMES)},metric_scope=METRIC_SCOPE)
    return result


def evaluate_saved(entry,path,states,indices,rule,networks=None):
    path=Path(path);ids=np.asarray(indices,dtype=np.int64);n=len(ids)
    require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['data_sha256'],'Evaluation path/hash')
    require(entry['rule']==rule and entry['information']=='FI' and entry['live'] is False,'FI silent evaluation identity')
    value=env.load_npz(path);expected_settle=settle(states[ids],value['action_indices'],rule)
    conditional_keys=('conditional_exact_expected_reward','conditional_exact_full_success_probability',
        'conditional_exact_execution_probability','conditional_full_posterior_mass')
    require(set(value)==set(expected_settle)|set(conditional_keys)|{'states','state_indices','messages','action_indices','action_probabilities'},'Evaluation exact NPZ schema')
    require(value['states'].dtype==np.int16 and np.array_equal(value['states'],states[ids]),'Evaluation ordered worlds')
    require(value['state_indices'].dtype==np.int64 and np.array_equal(value['state_indices'],ids),'Evaluation ordered ids')
    for key,v in expected_settle.items():
        require(value[key].dtype==v.dtype and np.array_equal(value[key],v),'Independent physical settlement '+key)
    p=value['action_probabilities'];a=value['action_indices'];m=value['messages']
    require(a.dtype==np.int16 and a.shape==(n,3),'Action dtype/shape')
    require(m.dtype==np.int8 and m.shape==(n,2,3,4) and np.all((m>=0)&(m<8)),'Message dtype/domain')
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Probability dtype/domain')
    require(np.array_equal(p.argmax(-1),a),'Argmax action identity')
    for key in conditional_keys:require(value[key].dtype==np.float64 and value[key].shape==(n,) and np.isfinite(value[key]).all(),'Conditional dtype/domain '+key)
    max_p=max_stat=0.;ties=0
    for begin in range(0,n,1024):
        sl=slice(begin,begin+1024);chunk=states[ids[sl]];native=env.rewards(chunk);logs=None
        if networks is not None:
            mm,pp,logs,tt=old.forward_final(networks,chunk,'FI',False);ties+=tt
            require(np.array_equal(mm,m[sl]) and np.array_equal(pp.argmax(-1),a[sl]),'Complete final independent discrete replay')
            max_p=max(max_p,close(pp,p[sl],'Complete final probability replay'))
        expected=conditional_statistics(p[sl],native,rule,logs)
        for key in conditional_keys:max_stat=max(max_stat,close(value[key][sl],expected[key],'Independent conditional '+key))
    native=env.rewards(states[ids]);metrics=summarize(states[ids],a,rule,native)
    metrics.update(information='FI',live=False,
        expected_reward_given_greedy_messages=float(value['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(value['conditional_exact_full_success_probability'].mean()),
        execution_probability_given_greedy_messages=float(value['conditional_exact_execution_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(value['conditional_full_posterior_mass'].mean()),state_indices_sha256=array_sha(ids))
    cross={other:summarize(states[ids],a,other,native) for other in RULES}
    require(cross['reciprocal']['full_success_rate']+TOL>=cross['strict']['full_success_rate'],'Fixed-policy mechanical full release is negative')
    require(cross['reciprocal']['reward_mean']+TOL>=cross['strict']['reward_mean'],'Fixed-policy mechanical reward release is negative')
    metrics['cross_settlement']=cross;compare(entry,metrics,'Evaluation summary/')
    return value,dict(sha256=sha(path),worlds=n,max_probability_error=max_p,max_conditional_error=max_stat,
        independent_forward_worlds=n if networks is not None else 0,independent_network_samples=9*n if networks is not None else 0,
        independent_final_sender_ties=ties if networks is not None else 0,metrics=metrics)


def training_row(row,seed,rule,update,expected):
    finite_tree(row)
    require((row['seed'],row['rule'],row['condition'],row['update'])==(seed,rule,'FI_silent',update),'Training identity')
    require(row['settlement_rule']==rule and row['partial_success_utility']==.5,'Training rule/reward identity')
    for key,v in expected.items():require(row[key]==v,'Independent random stream '+key)
    beta=.001*max(0.,1-(update-1)/1000)
    close(row['entropy_coefficient'],beta,'Original entropy schedule')
    close(row['mean_F'],row['mean_log_expected_utility']+beta*row['mean_actor_entropy'],'Training objective identity')
    close(row['receiver_loss'],-row['mean_F'],'Training loss sign')
    for key in ('mean_expected_utility','mean_native_expected_reward','mean_full_success_probability','mean_partial_success_probability',
                'mean_full_success_posterior_mass','mean_partial_success_posterior_mass','mean_execution_probability'):
        require(-TOL<=row[key]<=1+TOL,'Training probability/reward range '+key)
    close(row['mean_expected_utility'],row['mean_native_expected_reward'],'Native utility identity')
    close(row['mean_expected_utility'],row['mean_full_success_probability']+.5*row['mean_partial_success_probability'],'Reward event decomposition')
    require(row['mean_full_success_probability']+row['mean_partial_success_probability']<=row['mean_execution_probability']+TOL,'Rewarding events execute')
    close(row['mean_full_success_posterior_mass']+row['mean_partial_success_posterior_mass'],1.,'Posterior normalization')
    require(row['min_log_expected_utility']<=row['mean_log_expected_utility']+TOL<=row['max_log_expected_utility']+2*TOL<=3*TOL,'Log reward range')
    require(type(row['zero_float_expected_utility_states']) is int and 0<=row['zero_float_expected_utility_states']<=512,'Underflow count')
    require(-TOL<=row['mean_actor_entropy']<=math.log(17)+TOL,'Actor entropy range')
    close(row['sender_advantage_mean'],0.,'Paired LOO antisymmetry')
    require(0<=row['sender_advantage_abs_mean']<=row['sender_advantage_max_abs']+TOL,'Advantage moments')
    require(row['sender_advantage_abs_mean']**2<=row['sender_advantage_squared_mean']+TOL,'Advantage second moment')
    require(row['sender_mean_complete_log_score']<=TOL,'Log score sign')
    require(re.fullmatch('[0-9a-f]{64}',row['sampled_messages_sha256']) is not None,'Sampled message hash syntax, not sampled-message replay')
    norm=row['gradient_norm'];require(norm>=0,'Gradient norm')
    close(row['gradient_clip_scale'],min(1.,5./max(norm,1e-300)),'Gradient clipping')


def primary(final_metrics):
    require(set(final_metrics)=={(s,r) for s in SEEDS for r in RULES},'Eight final primary cells')
    blocks=[];keys=('full_success_rate','executed_partner_correct_rate','proposal_role_success_rate','reward_mean')
    for seed in SEEDS:
        cells={rule:final_metrics[seed,rule] for rule in RULES}
        contrasts={k:cells['reciprocal'][k]-cells['strict'][k] for k in keys}
        matrix={t:{e:cells[t]['cross_settlement'][e]['full_success_rate'] for e in RULES} for t in RULES}
        ss=matrix['strict']['strict'];sr=matrix['strict']['reciprocal'];rs=matrix['reciprocal']['strict'];rr=matrix['reciprocal']['reciprocal']
        d=dict(common_reciprocal_policy_difference=rr-sr,strict_policy_mechanical_release=sr-ss,
            common_strict_policy_difference=rs-ss,reciprocal_policy_mechanical_release=rr-rs)
        close(contrasts['full_success_rate'],rr-ss,'Diagonal primary identity')
        close(contrasts['full_success_rate'],d['common_reciprocal_policy_difference']+d['strict_policy_mechanical_release'],'Reciprocal decomposition')
        close(contrasts['full_success_rate'],d['common_strict_policy_difference']+d['reciprocal_policy_mechanical_release'],'Strict decomposition')
        require(sr+TOL>=ss and rr+TOL>=rs,'Nonnegative mechanical releases')
        blocks.append(dict(seed=seed,contrasts=contrasts,cells={r:{k:cells[r][k] for k in keys} for r in RULES},
            full_success_cross_settlement=matrix,decomposition=d))
    return dict(metric='full_success_rate',partition='new_needs_and_layouts',paired_seeds=blocks,
        mean_difference=float(np.mean([b['contrasts']['full_success_rate'] for b in blocks])))


def budget(specs):
    worlds=sum(s['world_count'] for s in specs.values());mon=sum(len(s['monitor_indices']) for s in specs.values())
    require(worlds==774144 and mon==21504,'World coverage')
    return dict(runs=8,training_updates=48000,training_world_samples=48000*256,message_trajectories=48000*256*2,
        categorical_symbol_samples=48000*256*2*24,training_forward_module_samples=48000*256*2*9,checkpoints=48,
        actual_monitor_files=192,actual_final_files=32,aliased_evaluations=0,actual_monitor_worlds=8*6*mon,
        actual_final_worlds=8*worlds,monitor_forward_module_samples=8*6*mon*9,final_forward_module_samples=8*worlds*9,
        evaluation_forward_module_samples=(8*6*mon+8*worlds)*9)


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution';refs=references()
    require(re.fullmatch('[0-9a-f]{64}',expected_plan_sha) is not None,'Explicit frozen plan hash required')
    require(read(execution/'status.json')['status']=='completed' and not (execution/'failure.json').exists(),'No incomplete-run audit')
    result=read(execution/'results.json');finite_tree(result);plan=read(run/'plan.json');prepared=read(run/'prepared.json')
    require(result['status']=='completed' and len(result['runs'])==8,'Completed eight-run result')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256']==result['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==plan['prepared_sha256'] and read(execution/'started.json')['plan_sha256']==expected_plan_sha,'Prepared/started chain')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Audit numerical runtime')
    compare(plan['config'],dict(seeds=list(SEEDS),conditions=['FI_silent'],execution_rules=list(RULES),partitions=list(PARTS),
        updates=6000,batch_size=256,checkpoints=list(STEPS),features=54,dtype='float64',learning_rate=.001,global_gradient_clip=5.,
        entropy_initial=.001,entropy_zero_after_updates=1000,sender_entropy_coefficient=0,trajectories_per_state=2,
        sender_windows=2,sender_tokens_per_window=4,actions_per_actor=17,worker_count=4,multiprocessing_start_method='spawn',
        dimensions=dict(sender1=[54,64,64,32],sender2=[153,64,64,32],action=[252,64,64,17]),
        primary='6000_double_holdout_full_success_reciprocal_minus_strict',world_sampling='three_uniforms_uniform_need_layout_owner',
        information='FI_all_needs_and_public_layout; no_cross_agent_messages',native_reward='number_of_satisfied_actual_executors_divided_by_two'),
        'Frozen configuration/')
    artifacts=dict(refs)
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:require(digest==expected,'Changed artifact '+str(path))
        artifacts[str(path)]=digest;return digest
    for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json',
                 Path(__file__),HERE/'tests/test_audit.py',HERE/'audit_preflight_001.json'):bind(path)
    freeze=read(HERE/'audit_freeze_001.json');bind(HERE/'audit_freeze_001.json')
    require(freeze['status']=='frozen_before_formal_audit','Audit freeze status')
    require(freeze['plan_sha256']==expected_plan_sha,'Audit bound to the frozen main plan')
    require(set(freeze['source_sha256'])=={str(Path(__file__).resolve()),str(HERE/'tests/test_audit.py'),str(HERE/'audit_preflight_001.json')},'Audit source coverage')
    for path,digest in freeze['source_sha256'].items():bind(path,digest)
    for path,digest in plan['source_sha256'].items():
        bind(path,digest);bind(run/'source_snapshot'/Path(path).relative_to(ROOT),digest)
    for path,digest in plan['inputs_sha256'].items():bind(path,digest)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001'
    bind(original/'prepared.json',ORIGINAL_PREPARED_SHA)
    require(prepared['source_prepared_sha256']==ORIGINAL_PREPARED_SHA,'Original static source')
    specs,_,_=env.independent_specs()
    require(prepared['partitions']==read(original/'prepared.json')['partitions'],'Anchored full partition metadata')
    for part in PARTS:compare(prepared['partitions'][part],specs[part],'Independent partition core/')
    require(prepared['schema']=='triadic_reciprocal_execution_fi_v1' and prepared['independent_initializations']==4,'Prepared schema')
    expected_budget=budget(specs);require(prepared['budget']==result['budget']==expected_budget,'Fixed budget')
    require([(r['seed'],r['rule']) for r in result['runs']]==[(s,r) for s in SEEDS for r in RULES],'Canonical paired grid')
    states={p:env.packed(specs[p]) for p in PARTS};hashes={}
    for part,s in states.items():hashes[part]=dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)),x_FI=array_sha(env.features(s,'FI')))
    require(result['array_hashes']==hashes,'Independent complete input arrays')
    for seed in SEEDS:
        path=execution/f'seed_{seed}_arrays.json';bind(path);require(read(path)['array_hashes']==hashes,'Worker arrays')
    by={(r['seed'],r['rule']):r for r in result['runs']};counts=Counter();errors=defaultdict(float)
    evaluations={};final_metrics={};initial_hashes={};ckpt_paths=set();evaluation_paths=set()
    for seed in SEEDS:
        rng_at={};final_nets={};initial=[]
        for rule in RULES:
            row=by[seed,rule];directory=execution/name(seed,rule)
            require(row['condition']=='FI_silent' and row['updates']==6000,'Run condition/budget')
            for file in ('result.json','training.jsonl','monitor.jsonl'):bind(directory/file)
            require(read(directory/'result.json')==row,'Per-run result')
            require(sha(directory/'training.jsonl')==row['training_log_sha256'],'Training log hash')
            monitor=[json.loads(line) for line in (directory/'monitor.jsonl').read_text().splitlines()]
            require(monitor==row['monitor'] and [m['update'] for m in monitor]==list(STEPS),'Monitor/checkpoint grid')
            rng_at[rule]={}
            for step,m in zip(STEPS,monitor):
                path=directory/f'checkpoint_{step:04d}.npz';bind(path,m['checkpoint_sha256']);ckpt_paths.add(path)
                nets,wr,mr=message.checkpoint(path,step,seed);rng_at[rule][step]=(wr,mr);counts['checkpoints']+=1
                if step==0:
                    h=message.network_hash(nets);require(h==row['initial_parameter_sha256'],'Initial parameter identity');initial.append(h)
                if step==6000:
                    require(message.network_hash(nets)==row['final_parameter_sha256'] and sha(path)==row['final_checkpoint_sha256'],'Final parameter identity')
                    final_nets[rule]=nets
        require(len(set(initial))==1,'Paired initialization');initial_hashes[seed]=initial[0]
        wr=np.random.default_rng(np.random.SeedSequence([seed,200]))
        mr={f't{t}_w{w}_a{a}':np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300])) for t,w,a in product(range(2),range(2),range(3))}
        def rng_check(step):
            now=(wr.bit_generator.state,{k:g.bit_generator.state for k,g in mr.items()})
            for rule in RULES:require(rng_at[rule][step]==now,'Checkpoint random stream state')
        streams=[(execution/name(seed,r)/'training.jsonl').open() for r in RULES]
        try:
            rng_check(0)
            for update in range(1,6001):
                u=wr.random((256,3));ids=env.sample(specs['train'],u);mu=np.empty((2,256,2,3,4),np.float64)
                for t,w,a in product(range(2),range(2),range(3)):mu[t,:,w,a]=mr[f't{t}_w{w}_a{a}'].random((256,4))
                expected=dict(world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),
                    batch_states_sha256=array_sha(states['train'][ids]),sample_uniforms_sha256=array_sha(mu))
                for rule,stream in zip(RULES,streams):
                    line=stream.readline();require(bool(line),'Missing training row')
                    training_row(json.loads(line),seed,rule,update,expected);counts['training_rows']+=1
                if update in STEPS:rng_check(update)
            require(all(stream.read()=='' for stream in streams),'Excess training row')
        finally:
            for stream in streams:stream.close()
        initial_evaluations={}
        for rule in RULES:
            row=by[seed,rule];directory=execution/name(seed,rule);end_monitor={}
            for checkpoint in row['monitor']:
                step=checkpoint['update'];require(set(checkpoint['monitor'])==set(PARTS),'Monitor partitions')
                for part in PARTS:
                    ids=np.asarray(specs[part]['monitor_indices'],np.int64);path=directory/f'monitor_{step:04d}_{part}.npz'
                    value,receipt=evaluate_saved(checkpoint['monitor'][part],path,states[part],ids,rule)
                    bind(path,receipt['sha256']);evaluation_paths.add(path);evaluations[str(path)]=receipt
                    counts['monitor_files']+=1;counts['monitor_worlds']+=len(ids)
                    errors['conditional']=max(errors['conditional'],receipt['max_conditional_error'])
                    if step==0:
                        for key in ('states','state_indices','messages','action_indices','action_probabilities'):
                            if rule=='strict':initial_evaluations[part,key]=value[key]
                            else:require(np.array_equal(value[key],initial_evaluations[part,key]),'Initial proposals must match across rules')
                    if step==6000:end_monitor[part]=value
            require(set(row['final'])==set(PARTS),'Final partitions')
            for part in PARTS:
                ids=np.arange(specs[part]['world_count'],dtype=np.int64);path=directory/f'final_{part}.npz'
                value,receipt=evaluate_saved(row['final'][part],path,states[part],ids,rule,final_nets[rule])
                bind(path,receipt['sha256']);evaluation_paths.add(path);evaluations[str(path)]=receipt;counts['final_files']+=1
                for key in ('independent_forward_worlds','independent_network_samples','independent_final_sender_ties'):counts[key]+=receipt[key]
                for key in ('conditional','probability'):errors[key]=max(errors[key],receipt['max_'+key+'_error'])
                for key,v in end_monitor[part].items():
                    selected=value[key][specs[part]['monitor_indices']]
                    if v.dtype.kind=='f':close(v,selected,'Final monitor subset '+key)
                    else:require(np.array_equal(v,selected),'Final monitor subset '+key)
                if part=='new_needs_and_layouts':final_metrics[seed,rule]=receipt['metrics']
                print(json.dumps(dict(stage='final_audited',seed=seed,rule=rule,partition=part)),flush=True)
            require(set(directory.glob('*.npz'))=={p for p in ckpt_paths|evaluation_paths if p.parent==directory},'Unexpected NPZ')
        print(json.dumps(dict(stage='paired_seed_audited',seed=seed)),flush=True)
    require(len(set(initial_hashes.values()))==4,'Four distinct initialization blocks')
    compare(dict(counts),dict(training_rows=48000,checkpoints=48,monitor_files=192,final_files=32,monitor_worlds=1032192,
        independent_forward_worlds=6193152,independent_network_samples=55738368),'Audit scope/')
    require({p for p in execution.glob('seed_*_*') if p.is_dir()}=={execution/name(s,r) for s in SEEDS for r in RULES},'Unexpected run directories')
    calculated=primary(final_metrics);compare(result['primary'],calculated,'Paired primary and cross-settlement/')
    for path,digest in artifacts.items():require(sha(path)==digest,'Artifact changed during audit '+path)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,
        audit_authorship='Independent from new production environment/kernel/runner; pinned prior independent observation/checkpoint/forward helpers reused.',
        runtime=dict(python=platform.python_version(),numpy=np.__version__),historical_helpers_sha256=refs,
        artifacts_sha256=artifacts,scope=dict(counts),max_errors=dict(errors),evaluations=evaluations,primary=calculated,
        limits=['No optimizer replay, gradient replay, sampled-message regeneration or intermediate-checkpoint forward.',
            'All training world and message uniform streams reconstructed; training objective rows only algebraically/range checked.',
            'All saved settlements, both cross-settlements, role/content summaries and conditional probabilities reimplemented; only full endpoints independently forwarded.',
            'Individual positive-event action-probability underflow prevents monitor-posterior verification; fail instead of adding epsilon.',
            'Primary is total rule effect; cross-settlement algebra is not causal isolation of neutral paths or evidence of language.',
            'PL no-message information bounds do not apply to this FI experiment; four independent societies are the replication units.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args();out=Path(args.out).resolve();require(not out.exists(),'Never overwrite an audit');out.mkdir(parents=True)
    start=time.perf_counter();(out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),
        run=str(Path(args.run).resolve()),expected_plan_sha256=args.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        v=audit(args.run,args.plan_sha);v.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(v));print(json.dumps({k:v[k] for k in ('status','elapsed_seconds','scope','max_errors')},ensure_ascii=False))
    except BaseException as e:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(e),traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)));raise


if __name__=='__main__':main()
