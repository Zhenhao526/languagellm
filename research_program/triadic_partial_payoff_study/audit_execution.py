"""Read-only reimplementation audit of the partial-payoff experiment.

The utility module and this auditor have the same author. Their numerical
implementations are separate; neither the production runner nor utility module
is imported here. Older independent auditors supply pinned environment,
observation, routing, checkpoint and forward helpers. Only final evaluations
receive fresh nine-network forward passes. No optimizer is replayed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import json
import math
import platform
import re
import time
import traceback
import numpy as np

from research_program.triadic_action_dependency_study import audit_execution as env
from research_program.triadic_message_study import audit_execution as message

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (53101,53102,53103,53104)
PAYOFFS = (('a50',.5),('a10',.1))
CONDITIONS = ('FI_silent','PL_silent','PL_live')
CELLS = tuple((p,a,c) for p,a in PAYOFFS for c in CONDITIONS)
PARTS, STEPS = env.PARTS, env.STEPS
TOL = 2e-12
REFERENCE_SHA = {
    'research_program/triadic_action_dependency_study/audit_execution.py': '20f1be8c017a53874346d1e8865620586d56de4128580b1e428c27899a30e96d',
    'research_program/triadic_message_study/audit_execution.py': 'f09f30612ccb7e27b9ec7e3b38b9fa217d133f1df1185e2712fd899254151bd3',
    'research_program/triadic_learning_baseline/audit_execution.py': '739175596c4182068bbfbb578439189208cae72812068b2127d0006393be2281',
}
ORIGINAL_PREPARED_SHA='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'
require,read,sha,array_sha,json_bytes,close,finite_tree,compare = (
    env.require,env.read,env.sha,env.array_sha,env.json_bytes,env.close,env.finite_tree,env.compare)


def references():
    result = {}
    for relative,digest in REFERENCE_SHA.items():
        path = ROOT/relative
        require(sha(path)==digest,'Frozen independent helper changed '+relative)
        result[str(path)] = digest
    return result


def run_name(seed,payoff,condition):
    return f'seed_{seed}_{payoff}_{condition}'


def role_indices(actions):
    actions = np.asarray(actions)
    require(actions.ndim==2 and actions.shape[1]==3 and actions.dtype.kind in 'iu'
            and ((actions>=0)&(actions<17)).all(),'Role action domain')
    output = np.full(actions.shape,-1,dtype=np.int8)
    for who in range(3):
        active = actions[:,who]!=0
        other = [j for j in range(3) if j!=who]
        output[active,who] = np.asarray(other)[(actions[active,who]-1)%2]
    return output


def conditional_statistics(probabilities,native,alpha,log_probabilities=None):
    """Direct weighted joint log masses, without production formula calls.

Saved monitors contain probabilities, not logits. Their posterior can be
reconstructed only when positive-reward plans retain finite single-action log
probabilities. Fail explicitly on lost information instead of adding epsilon.
    """
    p,r = np.asarray(probabilities),np.asarray(native)
    require(p.ndim==3 and p.shape[1:]==(3,17) and len(p)>0 and r.shape==(len(p),24),'Conditional shapes')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Conditional probability domain')
    close(p.sum(-1),np.ones(p.shape[:2]),'Conditional policy normalization')
    require(np.isin(r,(0,.5,1)).all() and ((r==1).sum(-1)==1).all(),'Unique full native support')
    require(alpha in (.5,.1) and not isinstance(alpha,(bool,np.bool_)),'Fixed payoff alpha')
    with np.errstate(divide='ignore'):
        lp = np.log(p) if log_probabilities is None else np.asarray(log_probabilities)
    require(lp.shape==p.shape and not np.isnan(lp).any() and (lp<=TOL).all(),'Conditional log probabilities')
    if log_probabilities is not None:
        require(np.isfinite(lp).all(),'Explicit log probabilities finite')
        close(np.exp(lp),p,'Explicit log/probability consistency')
    # Advanced indexing builds [B,24,3]; sum/product reduce actor dimension.
    selected_p = np.stack([p[:,who,env.JOINT[:,who]] for who in range(3)],axis=-1)
    selected_lp = np.stack([lp[:,who,env.JOINT[:,who]] for who in range(3)],axis=-1)
    if log_probabilities is None:
        require(np.isfinite(selected_lp[r>0]).all(),
                'Saved positive-reward action probability underflow prevents monitor posterior proof')
    joint_probability = np.prod(selected_p,axis=-1)
    log_weight = np.full(r.shape,-np.inf)
    log_weight[r==1] = 0.
    log_weight[r==.5] = math.log(alpha)
    log_mass = selected_lp.sum(-1)+log_weight
    log_expected_utility = np.logaddexp.reduce(log_mass,axis=1)
    require(np.isfinite(log_expected_utility).all(),'Finite log utility support')
    posterior = np.exp(log_mass-log_expected_utility[:,None])
    close(posterior.sum(-1),np.ones(len(p)),'Direct log-mass posterior normalization')
    full = r==1;partial = r==.5
    full_p = (joint_probability*full).sum(-1)
    partial_p = (joint_probability*partial).sum(-1)
    return {
        'conditional_exact_expected_reward':full_p+.5*partial_p,
        'conditional_exact_full_success_probability':full_p,
        'conditional_exact_execution_probability':joint_probability.sum(-1),
        'conditional_exact_expected_utility':full_p+alpha*partial_p,
        'conditional_full_posterior_mass':posterior[full],
        'posterior':posterior,'log_expected_utility':log_expected_utility,
    }


def forward_final(networks,states,information,live):
    """Nine heads once; retain receiver log probabilities for underflow cases."""
    x=env.features(states,information);messages=[];inputs=x;ties=0
    for window in (0,1):
        probabilities=np.stack([message.probabilities(networks[3*a+window],inputs[:,a]) for a in range(3)],axis=1)
        ties+=int(((probabilities==probabilities.max(-1,keepdims=True)).sum(-1)>1).sum())
        messages.append(probabilities.argmax(-1).astype(np.int8))
        if window==0:inputs=np.concatenate((x,message.route(messages[0],live)),axis=-1)
    inputs=np.concatenate((x,message.route(messages[0],live),message.route(messages[1],live)),axis=-1)
    probabilities=[];logs=[]
    for actor in range(3):
        network=networks[3*actor+2];local=inputs[:,actor]
        first=np.tanh(local@network['W1']+network['b1'])
        second=np.tanh(first@network['W2']+network['b2'])
        logits=second@network['W3']+network['b3']
        require(np.isfinite(logits).all(),'Independent final logits finite')
        centered=logits-logits.max(-1,keepdims=True)
        mass=np.exp(centered);normalizer=mass.sum(-1,keepdims=True)
        probabilities.append(mass/normalizer);logs.append(centered-np.log(normalizer))
    return np.stack(messages,axis=1),np.stack(probabilities,axis=1),np.stack(logs,axis=1),ties


def evaluate_saved(entry,path,states,indices,information,live,alpha,networks=None):
    path = Path(path);indices = np.asarray(indices,dtype=np.int64)
    require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['data_sha256'],'Evaluation path/hash')
    require(entry['information']==information and entry['live'] is live and entry['reused_natural'] is False,'Evaluation mode')
    require(entry['partial_utility']==alpha,'Evaluation alpha')
    values = env.load_npz(path);n = len(indices)
    scalar_keys = ('greedy_reward','greedy_utility','conditional_exact_expected_reward',
        'conditional_exact_full_success_probability','conditional_exact_execution_probability',
        'conditional_exact_expected_utility','conditional_full_posterior_mass')
    require(set(values)==set(scalar_keys)|{'states','state_indices','messages','action_indices','action_probabilities','executed','satisfied'},'Evaluation exact schema')
    require(values['state_indices'].dtype==np.int64 and np.array_equal(values['state_indices'],indices),'Ordered evaluation indices')
    require(values['states'].dtype==np.int16 and np.array_equal(values['states'],states[indices]),'Exact evaluation states')
    for key in scalar_keys:
        require(values[key].shape==(n,) and values[key].dtype==np.float64 and np.isfinite(values[key]).all(),'Scalar dtype/finite '+key)
    tokens = values['messages'];actions = values['action_indices'];p = values['action_probabilities']
    require(tokens.shape==(n,2,3,4) and tokens.dtype==np.int8 and ((tokens>=0)&(tokens<8)).all(),'Saved message domain')
    require(actions.shape==(n,3) and actions.dtype==np.int16 and ((actions>=0)&(actions<17)).all(),'Saved actions')
    require(p.shape==(n,3,17) and p.dtype==np.float64 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Saved policy domain')
    close(p.sum(-1),np.ones((n,3)),'Saved policy normalization')
    require(np.array_equal(p.argmax(-1),actions),'Saved greedy actions')
    reward,executed,satisfied = env.native(values['states'],actions)
    for key,expected in (('greedy_reward',reward),('executed',executed),('satisfied',satisfied)):
        require(np.array_equal(values[key],expected),'Native settlement '+key)
    for key in ('executed','satisfied'):
        require(values[key].shape==(n,3) and values[key].dtype==bool,'Settlement boolean dtype')
    utility = (reward==1).astype(np.float64)+alpha*(reward==.5)
    require(np.array_equal(values['greedy_utility'],utility),'Greedy utility mapping')
    target_roles = np.empty((n,3),dtype=np.int8);maximum_probability=maximum_conditional=0.;ties=0
    for begin in range(0,n,1024):
        sl = slice(begin,min(begin+1024,n));chunk = values['states'][sl]
        truth = env.rewards(chunk)
        target = env.JOINT[(truth==1).argmax(-1)]
        target_roles[sl] = role_indices(target)
        independent_logs=None
        if networks is not None:
            messages,new_p,independent_logs,new_ties = forward_final(networks,chunk,information,live)
            ties += new_ties
            require(np.array_equal(messages,tokens[sl]),'Final replay message identity')
            require(np.array_equal(new_p.argmax(-1),actions[sl]),'Final replay action identity')
            maximum_probability=max(maximum_probability,close(new_p,p[sl],'Final replay probability'))
        expected = conditional_statistics(p[sl],truth,alpha,independent_logs)
        for key,value in expected.items():
            if key in values:
                maximum_conditional=max(maximum_conditional,close(values[key][sl],value,'Conditional '+key))
    roles = role_indices(actions)
    unique_actions,action_counts = np.unique(actions,axis=0,return_counts=True)
    unique_roles,role_counts = np.unique(roles,axis=0,return_counts=True)
    metrics = dict(worlds=n,information=information,live=live,partial_utility=alpha,
        reward_mean=float(reward.mean()),utility_mean=float(utility.mean()),full_success_rate=float((reward==1).mean()),
        role_success_rate=float(np.all(roles==target_roles,axis=1).mean()),physical_execution_rate=float(executed.any(1).mean()),
        expected_reward_given_greedy_messages=float(values['conditional_exact_expected_reward'].mean()),
        expected_utility_given_greedy_messages=float(values['conditional_exact_expected_utility'].mean()),
        full_probability_given_greedy_messages=float(values['conditional_exact_full_success_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(values['conditional_full_posterior_mass'].mean()),
        raw_joint_action_counts=[dict(action_indices=a.tolist(),worlds=int(c)) for a,c in zip(unique_actions,action_counts)],
        raw_joint_role_counts=[dict(partner_indices=a.tolist(),worlds=int(c)) for a,c in zip(unique_roles,role_counts)],
        state_indices_sha256=array_sha(indices),reused_natural=False)
    compare(entry,metrics,'evaluation/')
    return values,dict(sha256=sha(path),worlds=n,max_probability_error=maximum_probability,max_conditional_error=maximum_conditional,
        independent_forward_worlds=n if networks is not None else 0,
        independent_network_samples=9*n if networks is not None else 0,
        independent_final_sender_ties=ties if networks is not None else 0,metrics=metrics)


def training_row(row,seed,payoff,alpha,condition,update,expected):
    finite_tree(row)
    require((row['seed'],row['payoff'],row['condition'],row['update'])==(seed,payoff,condition,update),'Training row identity')
    require(row['partial_utility']==row['partial_success_utility']==alpha,'Training payoff identity')
    for key,value in expected.items():require(row[key]==value,'Independent exogenous stream '+key)
    beta=.001*max(0.,1-(update-1)/1000)
    close(row['entropy_coefficient'],beta,'Entropy schedule')
    close(row['mean_F'],row['mean_log_expected_utility']+beta*row['mean_actor_entropy'],'Expected utility plus entropy')
    close(row['receiver_loss'],-row['mean_F'],'Receiver loss sign')
    for key in ('mean_expected_utility','mean_native_expected_reward','mean_full_success_probability','mean_partial_success_probability',
                'mean_full_success_posterior_mass','mean_partial_success_posterior_mass'):
        require(-TOL<=row[key]<=1+TOL,'Training probability/reward range '+key)
    close(row['mean_expected_utility'],row['mean_full_success_probability']+alpha*row['mean_partial_success_probability'],'Utility decomposition')
    close(row['mean_native_expected_reward'],row['mean_full_success_probability']+.5*row['mean_partial_success_probability'],'Native reward decomposition')
    close(row['mean_full_success_posterior_mass']+row['mean_partial_success_posterior_mass'],1.,'Training posterior normalization')
    require(row['min_log_expected_utility']<=row['mean_log_expected_utility']+TOL<=row['max_log_expected_utility']+2*TOL<=3*TOL,'Log utility range')
    require(type(row['zero_float_expected_utility_states']) is int and 0<=row['zero_float_expected_utility_states']<=512,'Underflow state count')
    require(-TOL<=row['mean_actor_entropy']<=math.log(17)+TOL,'Entropy range')
    close(row['sender_advantage_mean'],0.,'Antisymmetric LOO advantage')
    for key in ('sender_advantage_abs_mean','sender_advantage_squared_mean','sender_advantage_max_abs'):
        require(row[key]>=0,'Sender advantage moment range')
    require(row['sender_advantage_abs_mean']<=row['sender_advantage_max_abs']+TOL,'Sender advantage moment ordering')
    require(row['sender_advantage_abs_mean']**2<=row['sender_advantage_squared_mean']+TOL,'Sender second moment')
    require(row['sender_mean_complete_log_score']<=TOL,'Sample log score sign')
    require(re.fullmatch('[0-9a-f]{64}',row['sampled_messages_sha256']) is not None,'Sampled message digest')
    norm=row['gradient_norm'];require(norm>=0,'Gradient norm range')
    close(row['gradient_clip_scale'],min(1.,5./max(norm,1e-300)),'Gradient clip scale')


def budget(specs):
    worlds=sum(s['world_count'] for s in specs.values())
    monitor=sum(len(s['monitor_indices']) for s in specs.values())
    return dict(runs=24,training_updates=144000,training_world_samples=144000*256,message_trajectories=144000*256*2,
        categorical_symbol_samples=144000*256*2*24,checkpoints=144,actual_monitor_files=32*6*4,actual_final_files=32*4,
        silent_closed_aliases=16*7*4,actual_monitor_worlds=32*6*monitor,actual_final_worlds=32*worlds,
        evaluation_network_samples=(32*6*monitor+32*worlds)*9,training_forward_module_samples=144000*256*2*9)


def validate_partitions(recorded,independent,inherited):
    """Eight domain fields are regenerated; five metadata fields retain provenance.

The inherited semantic-pair summary is not a target of this world-weighted
experiment and is not advertised as an independently regenerated pair table.
    """
    core={'needs','layouts','private_sites','world_count','need_split','layout_split','monitor_backgrounds','monitor_indices'}
    extra={'monitor_scope','partition','semantic_pair_summary','state_order','weighting'}
    require(set(recorded)==set(independent)==set(inherited)==set(PARTS),'Partition names')
    for part in PARTS:
        require(set(independent[part])==core,'Independent domain schema')
        require(set(recorded[part])==core|extra and set(inherited[part])==core|extra,'Inherited partition schema')
        require(recorded[part]==inherited[part],'Exact frozen partition copy '+part)
        for key in core:require(recorded[part][key]==independent[part][key],'Independent domain field '+part+'/'+key)
        require(recorded[part]['partition']==part,'Partition metadata identity')
        require(recorded[part]['state_order']=='need-major, then layout, then owner','World-order metadata')
        require(recorded[part]['weighting']=='uniform needs × layouts × owners','World-weight metadata')
    return dict(independently_reconstructed_fields=sorted(core),inherited_fields_checked_against_fixed_source=sorted(extra),
                source_prepared_sha256=ORIGINAL_PREPARED_SHA)


def primary(final_metrics):
    values=[]
    for seed in SEEDS:
        cells={p+'_'+c:final_metrics[f'{seed}/{p}/{c}/new_needs_and_layouts/natural'] for p,_,c in CELLS}
        contrasts={m:(cells['a10_PL_live'][m]-cells['a10_PL_silent'][m])-(cells['a50_PL_live'][m]-cells['a50_PL_silent'][m])
                   for m in ('role_success_rate','full_success_rate','reward_mean')}
        values.append(dict(seed=seed,contrasts=contrasts,
            cells={key:{m:v[m] for m in ('role_success_rate','full_success_rate','reward_mean','utility_mean')} for key,v in cells.items()}))
    return dict(metric='role_success_rate',partition='new_needs_and_layouts',paired_seeds=values,
        mean_difference=float(np.mean([v['contrasts']['role_success_rate'] for v in values])))


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution'
    require(re.fullmatch('[0-9a-f]{64}',expected_plan_sha) is not None,'Explicit expected plan SHA required')
    require(read(execution/'status.json')['status']=='completed','No audit of incomplete batch')
    require(not (execution/'failure.json').exists(),'Main failure artifact present')
    results=read(execution/'results.json');finite_tree(results)
    require(results['status']=='completed' and len(results['runs'])==24,'Completed full grid')
    plan,prepared,freeze=read(run/'plan.json'),read(run/'prepared.json'),read(run/'freeze.json')
    require(sha(run/'plan.json')==expected_plan_sha==freeze['plan_sha256']==results['plan_sha256'],'Frozen plan chain')
    require(sha(run/'prepared.json')==plan['prepared_sha256'],'Prepared artifact chain')
    require(read(execution/'started.json')['plan_sha256']==expected_plan_sha,'Started plan hash')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Exact numerical runtime')
    compare(plan['config'],dict(seeds=list(SEEDS),conditions=list(CONDITIONS),payoffs=[dict(name=p,partial_utility=a) for p,a in PAYOFFS],
        partitions=list(PARTS),updates=6000,batch_size=256,checkpoints=list(STEPS),features=54,dtype='float64',
        learning_rate=.001,global_gradient_clip=5.,entropy_initial=.001,entropy_zero_after_updates=1000,
        sender_entropy_coefficient=0,trajectories_per_state=2,sender_windows=2,sender_tokens_per_window=4,
        actions_per_actor=17,worker_count=4,multiprocessing_start_method='spawn',
        dimensions=dict(sender1=[54,64,64,32],sender2=[153,64,64,32],action=[252,64,64,17]),
        primary='6000_double_holdout_world_role_success_(a10_PL_live-a10_PL_silent)-(a50_PL_live-a50_PL_silent)',
        native_reward='satisfied_needs/2_always_0_0.5_1',training_utility='none0_exactly_one_alpha_both1',
        information='FI_all_needs_or_PL_private_own_need; public_layout_both',
        world_sampling='three_uniforms_uniform_need_layout_owner'),'Frozen config/')
    artifacts={str(p):sha(p) for p in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json')}
    for path,digest in plan['source_sha256'].items():
        path=Path(path)
        for target in (path,run/'source_snapshot'/path.relative_to(ROOT)):
            require(sha(target)==digest,'Frozen production source '+str(target));artifacts[str(target)]=digest
    for path,digest in plan['inputs_sha256'].items():
        require(sha(path)==digest,'Frozen input '+path);artifacts[path]=digest
    helpers=references();artifacts.update(helpers)
    artifacts[str(Path(__file__))]=sha(__file__)
    testpath=HERE/'tests/test_audit.py';artifacts[str(testpath)]=sha(testpath)
    specs,_,_=env.independent_specs()
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001'
    require(prepared['source_prepared_sha256']==sha(original/'prepared.json')==ORIGINAL_PREPARED_SHA,'Static source prepared hash')
    partition_proof=validate_partitions(prepared['partitions'],specs,read(original/'prepared.json')['partitions'])
    require(sha(original/'plan.json')=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba','Static original plan')
    require(prepared['schema']=='triadic_partial_payoff_v1' and prepared['new_initializations']==4,'Prepared schema')
    expected_budget=budget(specs)
    require(prepared['budget']==results['budget']==expected_budget,'Fixed budget')
    require([(r['seed'],r['payoff'],r['condition']) for r in results['runs']]==[(s,p,c) for s in SEEDS for p,_,c in CELLS],'Canonical 24-run grid')
    states={part:env.packed(specs[part]) for part in PARTS};hashes={}
    for part,s in states.items():
        hashes[part]=dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)))
        for info in ('FI','PL'):hashes[part]['x_'+info]=array_sha(env.features(s,info))
    require(results['array_hashes']==hashes,'Independent static array hashes')
    for seed in SEEDS:
        path=execution/f'seed_{seed}_arrays.json';require(read(path)['array_hashes']==hashes,'Per-worker arrays');artifacts[str(path)]=sha(path)
    by={(r['seed'],r['payoff'],r['condition']):r for r in results['runs']}
    stats=Counter();errors=defaultdict(float);evaluations={};final_metrics={};initial_hashes={}
    checkpoint_files=set();evaluation_files=set()
    for seed in SEEDS:
        rng_at={};final_nets={};initial=[]
        for payoff,alpha,condition in CELLS:
            row=by[seed,payoff,condition];directory=execution/run_name(seed,payoff,condition)
            require(row['partial_utility']==alpha and row['updates']==6000,'Run fixed utility/budget')
            for file in ('result.json','training.jsonl','monitor.jsonl'):
                path=directory/file;artifacts[str(path)]=sha(path)
            require(read(directory/'result.json')==row,'Per-run and aggregate results')
            require(sha(directory/'training.jsonl')==row['training_log_sha256'],'Training log chain')
            monitor=[json.loads(line) for line in (directory/'monitor.jsonl').read_text().splitlines()]
            require(monitor==row['monitor'] and [m['update'] for m in monitor]==list(STEPS),'Monitor log and checkpoint grid')
            rng_at[payoff,condition]={}
            for step,m in zip(STEPS,monitor):
                path=directory/f'checkpoint_{step:04d}.npz'
                require(sha(path)==m['checkpoint_sha256'],'Checkpoint chain');artifacts[str(path)]=sha(path);checkpoint_files.add(path)
                networks,world_rng,message_rng=message.checkpoint(path,step,seed)
                rng_at[payoff,condition][step]=(world_rng,message_rng);stats['checkpoints']+=1
                if step==0:
                    digest=message.network_hash(networks);require(digest==row['initial_parameter_sha256'],'Initial parameter hash');initial.append(digest)
                if step==6000:
                    require(message.network_hash(networks)==row['final_parameter_sha256'] and sha(path)==row['final_checkpoint_sha256'],'Final parameter chain')
                    final_nets[payoff,condition]=networks
            require(set(directory.glob('checkpoint_*.npz'))=={directory/f'checkpoint_{step:04d}.npz' for step in STEPS},'Unexpected checkpoint file')
        require(len(set(initial))==1,'Six cells share seed initialization');initial_hashes[seed]=initial[0]
        world_rng=np.random.default_rng(np.random.SeedSequence([seed,200]))
        message_rng={f't{t}_w{w}_a{a}':np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300])) for t,w,a in product(range(2),range(2),range(3))}
        streams=[(execution/run_name(seed,p,c)/'training.jsonl').open() for p,_,c in CELLS]
        def check_rng(step):
            current=(world_rng.bit_generator.state,{k:r.bit_generator.state for k,r in message_rng.items()})
            for p,_,c in CELLS:require(rng_at[p,c][step]==current,'Checkpoint exogenous RNG state')
        try:
            check_rng(0)
            for update in range(1,6001):
                uniforms=world_rng.random((256,3));ids=env.sample(specs['train'],uniforms)
                message_uniforms=np.empty((2,256,2,3,4),dtype=np.float64)
                for t,w,a in product(range(2),range(2),range(3)):
                    message_uniforms[t,:,w,a]=message_rng[f't{t}_w{w}_a{a}'].random((256,4))
                expected=dict(world_uniforms_sha256=array_sha(uniforms),batch_indices_sha256=array_sha(ids),
                    batch_states_sha256=array_sha(states['train'][ids]),sample_uniforms_sha256=array_sha(message_uniforms))
                for (payoff,alpha,condition),stream in zip(CELLS,streams):
                    line=stream.readline();require(bool(line),'Missing training update')
                    training_row(json.loads(line),seed,payoff,alpha,condition,update,expected);stats['training_rows']+=1
                if update in STEPS:check_rng(update)
            require(all(stream.read()=='' for stream in streams),'Excess training update')
        finally:
            for stream in streams:stream.close()
        for payoff,alpha,condition in CELLS:
            row=by[seed,payoff,condition];directory=execution/run_name(seed,payoff,condition)
            info,visibility=condition.split('_');live=visibility=='live';end_monitor={}
            for step,m in zip(STEPS,row['monitor']):
                require(set(m['monitor'])==set(PARTS),'Monitor partition grid')
                for part in PARTS:
                    modes=m['monitor'][part];require(set(modes)=={'natural','closed'},'Monitor channel modes')
                    ids=np.asarray(specs[part]['monitor_indices'],dtype=np.int64)
                    for mode in ('natural','closed'):
                        if mode=='closed' and not live:
                            require(modes[mode]==dict(modes['natural'],reused_natural=True),'Silent monitor alias');stats['silent_closed_aliases']+=1
                            if step==6000:end_monitor[part,mode]=end_monitor[part,'natural']
                            continue
                        path=directory/f'monitor_{step:04d}_{part}_{mode}.npz'
                        values,receipt=evaluate_saved(modes[mode],path,states[part],ids,info,live and mode=='natural',alpha)
                        artifacts[str(path)]=receipt['sha256'];evaluations[str(path)]=receipt;evaluation_files.add(path)
                        stats['monitor_files']+=1;stats['monitor_worlds']+=len(ids)
                        errors['conditional']=max(errors['conditional'],receipt['max_conditional_error'])
                        if step==6000:end_monitor[part,mode]=values
            require(set(row['final'])==set(PARTS),'Final partition grid')
            for part in PARTS:
                modes=row['final'][part];require(set(modes)=={'natural','closed'},'Final channel modes')
                ids=np.arange(specs[part]['world_count'],dtype=np.int64)
                for mode in ('natural','closed'):
                    key=f'{seed}/{payoff}/{condition}/{part}/{mode}'
                    if mode=='closed' and not live:
                        require(modes[mode]==dict(modes['natural'],reused_natural=True),'Silent final alias');stats['silent_closed_aliases']+=1
                        final_metrics[key]=final_metrics[f'{seed}/{payoff}/{condition}/{part}/natural'];continue
                    path=directory/f'final_{part}_{mode}.npz'
                    values,receipt=evaluate_saved(modes[mode],path,states[part],ids,info,live and mode=='natural',alpha,final_nets[payoff,condition])
                    artifacts[str(path)]=receipt['sha256'];evaluations[str(path)]=receipt;evaluation_files.add(path);stats['final_files']+=1
                    for name in ('independent_forward_worlds','independent_network_samples','independent_final_sender_ties'):stats[name]+=receipt[name]
                    for name in ('conditional','probability'):errors[name]=max(errors[name],receipt['max_'+name+'_error'])
                    selected=np.asarray(specs[part]['monitor_indices'])
                    for name,value in end_monitor[part,mode].items():
                        expected=values[name][selected]
                        if value.dtype.kind=='f':close(value,expected,'Final monitor subset '+name)
                        else:require(np.array_equal(value,expected),'Final monitor subset '+name)
                    final_metrics[key]=receipt['metrics']
                    print(json.dumps(dict(audit_stage='final_checked',seed=seed,payoff=payoff,condition=condition,partition=part,mode=mode)),flush=True)
            require(set(directory.glob('*.npz'))=={x for x in checkpoint_files|evaluation_files if x.parent==directory},'Unexpected NPZ artifact')
        print(json.dumps(dict(audit_stage='seed_complete',seed=seed)),flush=True)
    require(len(set(initial_hashes.values()))==4,'Independent seeds differ')
    compare(dict(stats),dict(training_rows=144000,checkpoints=144,monitor_files=768,final_files=128,
        monitor_worlds=expected_budget['actual_monitor_worlds'],independent_forward_worlds=expected_budget['actual_final_worlds'],
        independent_network_samples=expected_budget['actual_final_worlds']*9,silent_closed_aliases=448),'Audit scope/')
    require({p for p in execution.glob('seed_*_*') if p.is_dir()}=={execution/run_name(s,p,c) for s in SEEDS for p,_,c in CELLS},'Unexpected run directories')
    recomputed_primary=primary(final_metrics);compare(results['primary'],recomputed_primary,'Primary role DiD/')
    for path,digest in artifacts.items():require(sha(path)==digest,'Artifact changed during audit '+path)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,
        audit_authorship='Same author as utility.py; separate implementation, with pinned older independently authored helpers.',
        runtime=dict(python=platform.python_version(),numpy=np.__version__),source_sha256=plan['source_sha256'],
        historical_helpers_sha256=helpers,artifacts_sha256=artifacts,scope=dict(stats),max_errors=dict(errors),
        evaluations=evaluations,final_metrics=final_metrics,primary=recomputed_primary,partition_proof=partition_proof,
        limits=['No optimizer replay, retraining, or intermediate checkpoint forward.',
            'Training random streams and saved checkpoint state are fully reconstructed; training loss rows receive algebraic/range checks, not regeneration of gradients or sampled messages.',
            'All monitor settlements, probability-conditional statistics and role/action histograms are recomputed from saved arrays; only complete final records receive independent nine-network forward.',
            'Positive-reward probabilities that have individually underflowed to zero prevent proof of monitor posterior; the audit fails rather than using an epsilon.',
            'Full-plan utility posterior mechanically increases when alpha decreases at fixed action probabilities; it alone is not evidence of learning.',
            'Four team initialization seeds are independent units. Worlds, checkpoints, payoffs and observation/channel cells are repeated measurements.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args();out=Path(args.out).resolve()
    require(not out.exists(),'Never overwrite or retry an audit in place');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        run=str(Path(args.run).resolve()),expected_plan_sha256=args.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        result=audit(args.run,args.plan_sha)
        result.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(result))
        text=(f"复核通过：144,000 条训练记录、144 个检查点、768 个监测文件、128 个完整末点文件及 448 个静默关闭引用。\n\n"
              f"末点重新前向 {result['scope']['independent_forward_worlds']:,} 个世界，{result['scope']['independent_network_samples']:,} 个网络样本；概率最大误差 {result['max_errors']['probability']:.3g}，条件统计最大误差 {result['max_errors']['conditional']:.3g}。\n\n"
              "审计与效用模块由同一作者分别实现；环境、观察、路由、检查点和末点前向沿用已固定哈希的旧独立审计函数。未重放优化器，未重算中间梯度或抽样消息。\n\n"
              "效用与原始奖励分开核对，角色直方图和双留出角色成功率的四种子 DiD 已重新计算。降低部分成功效用会机械性提高固定策略的完整计划后验，不能单凭这一指标声称学习改善。\n")
        (out/'独立核验.md').write_text(text)
        print(json.dumps(dict(status='passed',out=str(out),scope=result['scope'],max_errors=result['max_errors']),ensure_ascii=False),flush=True)
    except BaseException as error:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)))
        raise


if __name__=='__main__':main()
