"""Fresh paired rule x communication learning trajectories; frozen complete evaluation support."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from itertools import zip_longest
from copy import deepcopy
import argparse,json,platform,shutil,time,multiprocessing
import numpy as np
from research_program.triadic_private_partner_study import runner as previous
from research_program.triadic_action_dependency_study import runner as old, dataset
from . import metrics
from research_program.triadic_reciprocal_execution_study import environment, kernel
from research_program.triadic_need_response_study import cases

core=old.core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
PHYSICS=HERE.parent/'triadic_reciprocal_execution_study/results/rules_001'
SEEDS=tuple(range(60101,60117))
CONDITIONS=('strict_PL_live','strict_PL_silent','reciprocal_PL_live','reciprocal_PL_silent')
PARTS=old.PARTS;STEPS=old.STEPS
PRIVATE=HERE.parent/'triadic_private_partner_study/results/private_001'
TARGET='new_needs_and_layouts'
CONFIG=deepcopy(old.CONFIG)
for _key in ('monitor_worlds_per_partition','secondary_task_contrast'):CONFIG.pop(_key,None)
CONFIG.update(seeds=list(SEEDS),conditions=list(CONDITIONS),execution_rules=['strict','reciprocal'],
    task='execution_rule_by_communication_learning_trajectory',
    information='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='six_point_native_Q_communication_rule_DiD_minus_initial_trapezoid_over6000_then_mean16_blocks',
    training_objective='mean_log_exact_expected_native_reward_plus_original_action_entropy; sender_two_trajectory_LOO',
    native_reward='number_of_satisfied_actual_executors_divided_by_two',
    evaluation='complete_double_holdout_at_all_six_checkpoints; other_three_complete_partitions_only6000',
    evaluation_checkpoints=list(STEPS),target_final_alias='reuse_6000_full_target_trajectory_file_no_new_forward',
    scoring='same_greedy_actions_native_rule_and_common_reciprocal; Q_all_nine_strata_complete_support',
    evaluation_probability='conditional_on_greedy_messages; native_rule_exact_full_success_expected_reward_execution',
    closed_channel_interventions=0,monitor_subsets=0,
    message_panel='target_only_six_snapshots_five_adjacent_transitions; same_evaluation_messages; zero_extra_forward',
    pairing='same_seed_initial_parameters_and_world_and_message_uniform_streams_across_four_independent_trainings',
    run_order='four_spawn_workers_one_seed_block_each; strict_live,strict_silent,reciprocal_live,reciprocal_silent',
    worker_count=4,selection='fixed16_blocks_6000_updates_no_early_stop_or_result_dependent_changes',
    automatic_followon_experiment=False)

sha,read,write,json_hash,json_bytes,array_sha,now=old.sha,old.read,old.write,old.json_hash,old.json_bytes,old.array_sha,old.now


def require(ok,message):
    if not ok:raise ValueError(message)


def split_condition(condition):
    require(condition in CONDITIONS,'Unknown condition')
    return condition.split('_',1)[0],condition.endswith('_live')


def name(seed,condition):
    require(seed in SEEDS and condition in CONDITIONS,'Unknown seed/condition')
    return f'seed_{seed}_{condition}'


def prepared():
    source=read(ORIGINAL/'prepared.json');freeze=read(ORIGINAL/'freeze.json')
    require(sha(ORIGINAL/'prepared.json')==freeze['prepared_sha256']=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299','Original prepared hash mismatch')
    require(sha(ORIGINAL/'plan.json')==freeze['plan_sha256']=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba','Original plan hash mismatch')
    parts=source['partitions'];worlds=sum(s['world_count'] for s in parts.values());target_worlds=parts[TARGET]['world_count']
    require((worlds,target_worlds)==(774144,53568),'Unexpected complete world support')
    cc={part:cases.build_cases(parts[part]) for part in PARTS}
    require(all(len(c['strata'])==9 and not c['empty_strata'] and all(not s['empty'] and s['need_edges']>0 for s in c['strata']) for c in cc.values()),'All nine complete nonempty strata required')
    runs=len(SEEDS)*len(CONDITIONS);updates=runs*6000;eval_worlds=runs*(len(STEPS)*target_worlds+worlds-target_worlds)
    budget=dict(runs=runs,independent_paired_seed_blocks=len(SEEDS),training_updates=updates,training_world_samples=updates*256,
        message_trajectories=updates*256*2,categorical_symbol_samples=updates*256*2*24,
        training_forward_module_samples=updates*256*2*9,checkpoints=runs*len(STEPS),training_log_rows=updates,
        full_target_trajectory_files=runs*len(STEPS),other_final_files=runs*3,evaluation_files=runs*9,
        final_target_aliases=runs,additional_final_target_forward_samples=0,closed_files=0,monitor_subset_files=0,
        evaluation_worlds=eval_worlds,evaluation_forward_module_samples=eval_worlds*9,
        total_forward_module_samples=updates*256*2*9+eval_worlds*9,
        complete_need_response_records=runs*9*2,need_response_additional_forward_samples=0,
        generated_NPZ_files=runs*(6+9))
    require(budget['training_forward_module_samples']==1769472000 and budget['evaluation_forward_module_samples']==600182784,'Fixed module budget mismatch')
    return dict(schema='triadic_rule_formation_v1',partitions=parts,need_response_cases=cc,
        source_prepared_sha256=sha(ORIGINAL/'prepared.json'),independent_initializations=len(SEEDS),budget=budget)


def sources():
    require(sha(PRIVATE/'plan.json')=='b5126ca468e16a41db4ee9e96c77f0a7e465c3c2123589d2f423911fc5bf5555','Frozen private policy design changed')
    out={}
    for path,digest in read(PRIVATE/'plan.json')['source_sha256'].items():
        require(sha(path)==digest,f'Frozen dependency changed: {path}');out[path]=digest
    # previous.make_arrays is reused directly; bind its already frozen implementation.
    require(sha(previous.__file__)==read(PRIVATE/'plan.json')['source_sha256'][str(Path(previous.__file__).resolve())],'Frozen PL array helper changed')
    for filename in ('__init__.py','runner.py','test_runner.py','runner_preflight_001.json','metrics.py','test_metrics.py',
        'metrics_preflight_001.json','measure_design.md','plan.md','design_review.md','main_preflight.json',
        'literature_sources_001.json','literature/近期约定性研究与本轮定位.md'):
        path=HERE/filename;out[str(path)]=sha(path)
    return out


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Never overwrite prepared output')
    static=prepared();ss=sources();out.mkdir(parents=True)
    for path in ss:
        target=out/'source_snapshot'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write(out/'prepared.json',static)
    inputs=[ORIGINAL/f for f in ('plan.json','prepared.json','freeze.json')]
    inputs += [PHYSICS/f for f in ('plan.json','freeze.json')]
    inputs += [PRIVATE/f for f in ('plan.json','freeze.json')]
    write(out/'plan.json',dict(status='prepared_without_training',at=now(),config=CONFIG,
        source_sha256=ss,inputs_sha256={str(p):sha(p) for p in inputs},
        prepared_sha256=sha(out/'prepared.json'),runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json'),prepared_sha256=sha(out/'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training',plan_sha256=sha(out/'plan.json'),budget=static['budget'])


def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json');static=read(out/'prepared.json');freeze=read(out/'freeze.json')
    require(sha(out/'plan.json')==freeze['plan_sha256'] and plan['config']==CONFIG,'Plan/config mismatch')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Runtime mismatch')
    require(sha(out/'prepared.json')==plan['prepared_sha256']==freeze['prepared_sha256'],'Prepared hash mismatch')
    require(static==prepared() and plan['source_sha256']==sources(),'Static source/config mismatch')
    for path,digest in plan['inputs_sha256'].items():require(sha(path)==digest,f'Changed input: {path}')
    for path,digest in plan['source_sha256'].items():
        require(sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest,f'Changed snapshot: {path}')
    return plan,static


make_arrays=previous.make_arrays
process_status=previous.process_status


def evaluate(networks,arrays,live,rule,indices,path,case_spec):
    """One complete partition, one rollout, two physical scorings, no new sampling."""
    raw=np.asarray(indices);path=Path(path)
    require(rule in ('strict','reciprocal') and isinstance(live,(bool,np.bool_)),'Invalid evaluation rule/live')
    require(raw.ndim==1 and raw.dtype.kind in 'iu','Indices must be one-dimensional integers')
    ids=raw.astype(np.int64);n=len(ids)
    require(n>0 and np.all((ids>=0)&(ids<len(arrays['packed_states']))),'Invalid state indices')
    require(not path.exists(),'Never overwrite evaluation output')
    states=arrays['packed_states'][ids];cases.validate_partition_arrays(case_spec,states,ids)
    data=dict(states=states,state_indices=ids,messages=np.empty((n,2,3,4),np.int8),
        action_indices=np.empty((n,3),np.int16),action_probabilities=np.empty((n,3,17)),
        conditional_exact_expected_reward=np.empty(n),conditional_exact_full_success_probability=np.empty(n),
        conditional_exact_execution_probability=np.empty(n),conditional_full_posterior_mass=np.empty(n))
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));ix=ids[sl]
        trace=core.rollout(networks,arrays['x_PL'][ix],bool(live))
        terms=kernel.objective_terms(trace['action_logits'],arrays['rewards'][ix],rule)
        probs=terms['probabilities'];data['messages'][sl]=trace['messages']
        data['action_probabilities'][sl]=probs;data['action_indices'][sl]=probs.argmax(-1)
        data['conditional_exact_expected_reward'][sl]=terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl]=terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl]=terms['execution_probability']
        data['conditional_full_posterior_mass'][sl]=terms['full_success_posterior_mass']
    actions=data['action_indices'];strict=environment.settle(states,actions,'strict')
    common=environment.settle(states,actions,'reciprocal');native=strict if rule=='strict' else common
    data.update(native)
    data.update({f'strict__{k}':v for k,v in strict.items()})
    data.update({f'common_reciprocal__{k}':v for k,v in common.items()})
    truth=environment.truth_from_rewards(states,arrays['rewards'][ids])
    record=dict(native=environment.metrics(native,truth,actions,rule),
        strict=environment.metrics(strict,truth,actions,'strict'),
        common_reciprocal=environment.metrics(common,truth,actions,'reciprocal'),
        need_response=dict(native=cases.metrics(case_spec,native['actual_pair_index']),
            common_reciprocal=cases.metrics(case_spec,common['actual_pair_index'])))
    with path.open('xb') as stream:np.savez_compressed(stream,**data)
    record.update(path=str(path),data_sha256=sha(path),information='PL',live=bool(live),rule=rule,
        scope='complete_partition',worlds=n,forward_module_samples=9*n,
        expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
        execution_probability_given_greedy_messages=float(data['conditional_exact_execution_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(data['conditional_full_posterior_mass'].mean()),
        state_indices_sha256=array_sha(ids))
    return record,data['messages']


def evaluate_other_final(networks,arrays,condition,response_cases,directory,target_record):
    """The target endpoint is an existing trajectory record, not another rollout."""
    rule,live=split_condition(condition);directory=Path(directory)
    require(target_record['rule']==rule and target_record['live']==live,'Target alias policy mismatch')
    result={}
    for part in PARTS:
        if part==TARGET:
            result[part]=dict(target_record,alias_of='trajectory_update_6000',additional_forward_module_samples=0)
        else:
            ids=np.arange(len(arrays[part]['packed_states']),dtype=np.int64)
            record,_=evaluate(networks,arrays[part],live,rule,ids,directory/f'final_{part}.npz',response_cases[part])
            result[part]=record
    return result


def make_random_streams(seed):
    require(seed in SEEDS,'Unknown paired seed')
    return np.random.default_rng(np.random.SeedSequence([seed,200])),core.make_message_rngs(seed)


def train_run(seed,condition,static,arrays,execution):
    directory=Path(execution)/name(seed,condition);directory.mkdir(exist_ok=False)
    rule,live=split_condition(condition);nets=core.make_networks(seed);initial=core.parameter_hash(nets)
    optimizer=core.base.make_adam(nets);wr,mr=make_random_streams(seed)
    trajectory=[];start=time.perf_counter();previous_messages=None
    def checkpoint(step):
        nonlocal previous_messages
        ckpt=directory/f'checkpoint_{step:04d}.npz';digest=core.save_checkpoint(ckpt,nets,optimizer,step,wr,mr)
        evaluation,messages=evaluate(nets,arrays[TARGET],live,rule,np.arange(len(arrays[TARGET]['packed_states']),dtype=np.int64),
            directory/f'trajectory_{step:04d}_{TARGET}.npz',static['need_response_cases'][TARGET])
        snapshot=metrics.message_snapshot(messages,static['partitions'][TARGET])
        transition=None if previous_messages is None else metrics.message_transition(previous_messages,messages,static['partitions'][TARGET])
        previous_messages=messages
        record=dict(update=step,checkpoint_path=str(ckpt),checkpoint_sha256=digest,evaluation=evaluation,
            message_snapshot=snapshot,message_transition=transition,elapsed_seconds=time.perf_counter()-start)
        trajectory.append(record)
        with (directory/'trajectory.jsonl').open('a') as stream:stream.write(json_bytes(record).decode())
        process_status(directory/'status.json',status='running',seed=seed,condition=condition,rule=rule,update=step,
            checkpoint_sha256=digest,elapsed_seconds=time.perf_counter()-start)
    checkpoint(0)
    with (directory/'training.jsonl').open('x') as stream:
        for update in range(1,6001):
            u=wr.random((256,3));ids=dataset.sample_indices(static['partitions']['train'],u)
            uniforms=core.draw_uniforms(mr,256)
            gradients,row=kernel.training_gradients(nets,arrays['train']['x_PL'][ids],
                arrays['train']['rewards'][ids],live,uniforms,update,rule)
            norm,scale=core.base.adam_step(nets,gradients,optimizer,update)
            row.update(update=update,seed=seed,rule=rule,condition=condition,
                world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),
                batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                sample_uniforms_sha256=array_sha(uniforms),gradient_norm=norm,gradient_clip_scale=scale)
            stream.write(json_bytes(row).decode())
            if update in STEPS:stream.flush();checkpoint(update)
    final=evaluate_other_final(nets,arrays,condition,static['need_response_cases'],directory,trajectory[-1]['evaluation'])
    result=dict(seed=seed,condition=condition,rule=rule,live=live,updates=6000,initial_parameter_sha256=initial,
        final_parameter_sha256=core.parameter_hash(nets),final_checkpoint_sha256=sha(directory/'checkpoint_6000.npz'),
        training_log_sha256=sha(directory/'training.jsonl'),trajectory=trajectory,final=final,
        elapsed_seconds=time.perf_counter()-start)
    write(directory/'result.json',result)
    process_status(directory/'status.json',status='completed',seed=seed,condition=condition,update=6000,
        result_sha256=sha(directory/'result.json'),elapsed_seconds=time.perf_counter()-start)
    return result


def worker(payload):
    seed,static,execution=payload;execution=Path(execution);start=time.perf_counter()
    process_status(execution/f'seed_{seed}_status.json',status='building_arrays',seed=seed)
    arrays={part:make_arrays(static['partitions'][part]) for part in PARTS}
    hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()}
    write(execution/f'seed_{seed}_arrays.json',dict(array_hashes=hashes,build_seconds=time.perf_counter()-start))
    print(json.dumps(dict(stage='arrays_prepared',seed=seed)),flush=True);runs=[]
    for condition in CONDITIONS:
        process_status(execution/f'seed_{seed}_status.json',status='running',seed=seed,condition=condition)
        runs.append(train_run(seed,condition,static,arrays,execution))
        print(json.dumps(dict(stage='run_completed',seed=seed,condition=condition,elapsed_seconds=time.perf_counter()-start)),flush=True)
    require(hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()},'Training mutated world arrays')
    process_status(execution/f'seed_{seed}_status.json',status='completed',seed=seed)
    return runs


def verify_pairing(execution,runs):
    execution=Path(execution)
    for seed in SEEDS:
        cells=[r for r in runs if r['seed']==seed]
        require(len(cells)==4 and {r['condition'] for r in cells}==set(CONDITIONS),'Incomplete paired conditions')
        require(len({r['initial_parameter_sha256'] for r in cells})==1,'Unpaired initialization')
        streams=[(execution/name(seed,c)/'training.jsonl').open() for c in CONDITIONS]
        try:
            count=0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines),'Unequal training log lengths')
                rows=[json.loads(line) for line in lines];count+=1
                require(all(r['update']==count and r['seed']==seed and r['condition']==c and r['rule']==split_condition(c)[0]
                    for r,c in zip(rows,CONDITIONS)),'Training identity/update mismatch')
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256','entropy_coefficient'):
                    require(len({r[key] for r in rows})==1,f'Unpaired {key}')
            require(count==6000,'Training must contain exactly6000 updates')
        finally:
            for stream in streams:stream.close()


primary=metrics.primary


def measured_budget(runs):
    require(len(runs)==len(SEEDS)*4,'Incomplete run grid')
    actual=[]
    for run in runs:
        require(run['updates']==6000 and [v['update'] for v in run['trajectory']]==list(STEPS),'Incomplete fixed trajectory')
        trajectory=run['trajectory'];endpoint=trajectory[-1]['evaluation'];alias=run['final'][TARGET]
        require(alias['path']==endpoint['path'] and alias['data_sha256']==endpoint['data_sha256']
            and alias['alias_of']=='trajectory_update_6000' and alias['additional_forward_module_samples']==0,'Target final must alias existing full trajectory endpoint')
        require({k:v for k,v in alias.items() if k not in ('alias_of','additional_forward_module_samples')}==endpoint,'Alias evaluation fields differ')
        actual.extend(v['evaluation'] for v in trajectory)
        actual.extend(run['final'][part] for part in PARTS if part!=TARGET)
    require(len({r['path'] for r in actual})==len(actual),'Repeated actual evaluation file')
    return dict(training_forward_module_samples=sum(r['updates'] for r in runs)*256*2*9,
        evaluation_forward_module_samples=sum(r['forward_module_samples'] for r in actual),
        checkpoints=sum(len(r['trajectory']) for r in runs),evaluation_files=len(actual),
        evaluation_worlds=sum(r['worlds'] for r in actual),final_target_aliases=len(runs))


def execute(out):
    out=Path(out).resolve();plan,static=verify(out);execution=out/'execution'
    execution.mkdir(exist_ok=False);start=time.perf_counter()
    write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    process_status(execution/'status.json',status='running',plan_sha256=sha(out/'plan.json'),worker_count=4)
    try:
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            groups=pool.map(worker,[(s,static,str(execution)) for s in SEEDS])
        runs=[r for group in groups for r in group]
        require([(r['seed'],r['condition']) for r in runs]==[(s,c) for s in SEEDS for c in CONDITIONS],'Noncanonical run grid')
        verify_pairing(execution,runs)
        hashes=[read(execution/f'seed_{s}_arrays.json')['array_hashes'] for s in SEEDS]
        require(all(v==hashes[0] for v in hashes),'Different worker world arrays');verify(out)
        measured=measured_budget(runs)
        require(all(static['budget'][k]==v for k,v in measured.items()),'Measured budget mismatch')
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),elapsed_seconds=time.perf_counter()-start,
            budget=static['budget'],measured_budget=measured,runs=runs,array_hashes=hashes[0],primary=primary(runs))
        write(execution/'results.json',result)
        process_status(execution/'status.json',status='completed',results_sha256=sha(execution/'results.json'))
        return dict(status='completed',elapsed_seconds=result['elapsed_seconds'],primary=result['primary'])
    except BaseException as error:
        failure=dict(status='failed',at=now(),error=repr(error),elapsed_seconds=time.perf_counter()-start,automatic_retry=False)
        write(execution/'failure.json',failure);process_status(execution/'status.json',status='failed',error=repr(error));raise


run=execute
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','verify','execute','run'))
    parser.add_argument('--out',required=True);args=parser.parse_args();result=globals()[args.command](args.out)
    print(json.dumps(result if args.command!='verify' else dict(status='verified'),ensure_ascii=False))
