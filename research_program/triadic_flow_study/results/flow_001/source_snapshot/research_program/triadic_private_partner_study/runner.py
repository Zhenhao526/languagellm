"""Fresh paired PL policies under the frozen reciprocal physical rule.

Import has no training or model loading side effects. Preparation freezes the
complete case support before any policy runs; Q is evaluated only on full final
partitions. Four live-policy closed-channel rollouts are planned interventions.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from itertools import zip_longest
from copy import deepcopy
import argparse,json,platform,shutil,time,multiprocessing
import numpy as np
from research_program.triadic_action_dependency_study import runner as old, dataset
from research_program.triadic_reciprocal_execution_study import environment, kernel
from research_program.triadic_need_response_study import cases

core=old.core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
PHYSICS=HERE.parent/'triadic_reciprocal_execution_study/results/rules_001'
SEEDS=(59101,59102,59103,59104)
CONDITIONS=('PL_live','PL_silent')
RULE='reciprocal';PARTS=old.PARTS;STEPS=old.STEPS
TARGET='new_needs_and_layouts'
CONFIG=deepcopy(old.CONFIG)
CONFIG.update(seeds=list(SEEDS),conditions=list(CONDITIONS),execution_rule=RULE,
    task='private_need_partner_response_under_reciprocal_execution',
    information='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='6000_full_double_holdout_Q_PL_live_minus_PL_silent_then_mean_four_paired_seeds',
    secondary_task_contrast='four_partition_native_metrics_and_live_policy_closed_channel_intervention',
    training_objective='mean_log_exact_expected_native_reward_plus_original_action_entropy; sender_two_trajectory_LOO',
    native_reward='number_of_satisfied_actual_executors_divided_by_two',
    need_response='final_full_partitions_only; frozen_cases; equal_actor_axis_background_weights',
    monitor='six_checkpoints_natural_only; original_two_backgrounds; no_Q_or_closed_intervention',
    final='all_four_full_partitions_natural_for_eight_policies; complete_closed_rollout_for_four_live_policies',
    channel_closure='from_window_1_cross_actor_messages_and_visibility_bits_zero; own_messages_retained',
    run_order='four_spawn_workers_one_seed_each_PL_live_then_PL_silent',
    selection='fixed_6000_updates_no_early_stop_extra_seeds_or_score_dependent_changes',
    automatic_followon_experiment=False)
sha,read,write,json_hash,json_bytes,array_sha,now=old.sha,old.read,old.write,old.json_hash,old.json_bytes,old.array_sha,old.now


def require(ok,message):
    if not ok:raise ValueError(message)


def name(seed,condition):
    require(seed in SEEDS and condition in CONDITIONS,'Unknown seed/condition')
    return f'seed_{seed}_{condition}'


def prepared():
    source=read(ORIGINAL/'prepared.json');freeze=read(ORIGINAL/'freeze.json')
    require(sha(ORIGINAL/'prepared.json')==freeze['prepared_sha256']=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299','Original prepared hash mismatch')
    require(sha(ORIGINAL/'plan.json')==freeze['plan_sha256']=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba','Original plan hash mismatch')
    parts=source['partitions'];worlds=sum(s['world_count'] for s in parts.values())
    mon=sum(len(s['monitor_indices']) for s in parts.values())
    require(worlds==774144 and mon==21504,'Unexpected world/monitor support')
    response_cases={part:cases.build_cases(parts[part]) for part in PARTS}
    require(all(len(c['strata'])==9 and not c['empty_strata'] and
        all(not s['empty'] and s['need_edges']>0 for s in c['strata']) for c in response_cases.values()),
        'Frozen original support must retain all nine nonempty structural strata')
    return dict(schema='triadic_private_partner_v1',partitions=parts,
        need_response_cases=response_cases,
        source_prepared_sha256=sha(ORIGINAL/'prepared.json'),independent_initializations=4,
        budget=dict(runs=8,training_updates=48000,training_world_samples=48000*256,
            message_trajectories=48000*256*2,categorical_symbol_samples=48000*256*2*24,
            training_forward_module_samples=48000*256*2*9,checkpoints=48,
            natural_monitor_files=8*6*4,natural_final_files=8*4,closed_final_files=4*4,
            actual_monitor_files=8*6*4,actual_final_files=12*4,aliased_evaluations=0,
            actual_monitor_worlds=8*6*mon,natural_final_worlds=8*worlds,closed_final_worlds=4*worlds,
            actual_final_worlds=12*worlds,monitor_forward_module_samples=8*6*mon*9,
            final_forward_module_samples=12*worlds*9,
            evaluation_forward_module_samples=(8*6*mon+12*worlds)*9,
            need_response_full_partition_evaluations=48,
            need_response_edge_background_evaluations=12*sum(c['state_edges'] for c in response_cases.values()),
            need_response_additional_forward_samples=0))


def sources():
    require(sha(PHYSICS/'plan.json')==read(PHYSICS/'freeze.json')['plan_sha256']==
        '59e9f8d3b2056538b339072867fde4c58b1478274d982e3158d1b4268edd0e53','Frozen reciprocal plan mismatch')
    out={}
    for path,digest in read(PHYSICS/'plan.json')['source_sha256'].items():
        require(sha(path)==digest,f'Frozen dependency changed: {path}');out[path]=digest
    files=('__init__.py','runner.py','tests/test_runner.py','plan.md','design_review.md','runner_preflight_001.json','main_preflight.json')
    response=HERE.parent/'triadic_need_response_study'
    paths=[HERE/f for f in files]+[response/f for f in
        ('__init__.py','cases.py','test_cases.py','static_availability.json','cases_preflight_001.json','plan.md','design_review.md')]
    for path in paths:out[str(path)]=sha(path)
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


def make_arrays(spec):
    arrays=dataset.make_arrays(spec,information='PL');del arrays['states']
    x=arrays['x_PL'];require(x.shape==(len(arrays['packed_states']),3,54),'Bad PL feature shape')
    require(np.all(x[:,:,53]==0),'PL must not expose the FI flag')
    for actor in range(3):
        for other in range(3):
            if actor!=other:require(np.all(x[:,actor,7*other:7*other+7]==0),'Other private needs exposed')
    return arrays


def evaluate(networks,arrays,live,indices,path,case_spec=None,mode='natural'):
    raw=np.asarray(indices);path=Path(path)
    require(isinstance(live,(bool,np.bool_)) and mode in ('natural','closed'),'Invalid evaluation mode')
    require(mode!='closed' or not live,'Closed intervention must disable cross-agent routing')
    require(raw.ndim==1 and raw.dtype.kind in 'iu','Indices must be one-dimensional integers')
    ids=raw.astype(np.int64);n=len(ids)
    require(n>0 and len(np.unique(ids))==n and np.all((ids>=0)&(ids<len(arrays['packed_states']))),'Invalid state indices')
    require(not path.exists(),'Never overwrite evaluation output')
    states=arrays['packed_states'][ids]
    if case_spec is not None:
        # Reject monitor subsets and reordered worlds before any neural call.
        cases.validate_partition_arrays(case_spec,states,ids)
    data=dict(states=states,state_indices=ids,messages=np.empty((n,2,3,4),np.int8),
        action_indices=np.empty((n,3),np.int16),action_probabilities=np.empty((n,3,17)),
        conditional_exact_expected_reward=np.empty(n),conditional_exact_full_success_probability=np.empty(n),
        conditional_exact_execution_probability=np.empty(n),conditional_full_posterior_mass=np.empty(n))
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));ix=ids[sl]
        trace=core.rollout(networks,arrays['x_PL'][ix],bool(live))
        terms=kernel.objective_terms(trace['action_logits'],arrays['rewards'][ix],RULE)
        probs=terms['probabilities'];data['messages'][sl]=trace['messages']
        data['action_probabilities'][sl]=probs;data['action_indices'][sl]=probs.argmax(-1)
        data['conditional_exact_expected_reward'][sl]=terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl]=terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl]=terms['execution_probability']
        data['conditional_full_posterior_mass'][sl]=terms['full_success_posterior_mass']
    data.update(environment.settle(data['states'],data['action_indices'],RULE))
    record=environment.metrics(data,environment.truth_from_rewards(states,arrays['rewards'][ids]),data['action_indices'],RULE)
    if case_spec is not None:record['need_response']=cases.metrics(case_spec,data['actual_pair_index'])
    # Opening in exclusive mode also protects against a concurrent duplicate writer.
    with path.open('xb') as stream:np.savez_compressed(stream,**data)
    record.update(path=str(path),data_sha256=sha(path),information='PL',live=bool(live),mode=mode,
        intervention='close_cross_agent_channel_from_window_1' if mode=='closed' else None,
        expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
        execution_probability_given_greedy_messages=float(data['conditional_exact_execution_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(data['conditional_full_posterior_mass'].mean()),
        state_indices_sha256=array_sha(ids))
    return record


def evaluate_final(networks,arrays,condition,case_spec,prefix):
    require(condition in CONDITIONS,'Unknown condition');live=condition=='PL_live'
    ids=np.arange(len(arrays['packed_states']),dtype=np.int64)
    output={'natural':evaluate(networks,arrays,live,ids,Path(str(prefix)+'_natural.npz'),case_spec)}
    if live:
        output['closed']=evaluate(networks,arrays,False,ids,Path(str(prefix)+'_closed.npz'),case_spec,mode='closed')
    return output


def process_status(path,**fields):
    """Replace only the mutable process-status file, never scientific artifacts."""
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:stream.write(json_bytes(dict(at=now(),pid=os.getpid(),**fields)))
    os.replace(temporary,path)


def train_run(seed,condition,static,arrays,execution):
    directory=Path(execution)/name(seed,condition);directory.mkdir(exist_ok=False)
    live=condition=='PL_live';nets=core.make_networks(seed);initial=core.parameter_hash(nets)
    optimizer=core.base.make_adam(nets);wr=np.random.default_rng(np.random.SeedSequence([seed,200]))
    mr=core.make_message_rngs(seed);monitors=[];start=time.perf_counter()
    def checkpoint(step):
        ckpt=directory/f'checkpoint_{step:04d}.npz';digest=core.save_checkpoint(ckpt,nets,optimizer,step,wr,mr)
        evaluations={part:evaluate(nets,arrays[part],live,static['partitions'][part]['monitor_indices'],
            directory/f'monitor_{step:04d}_{part}.npz') for part in PARTS}
        record=dict(update=step,checkpoint_sha256=digest,monitor=evaluations,elapsed_seconds=time.perf_counter()-start)
        monitors.append(record)
        with (directory/'monitor.jsonl').open('a') as stream:stream.write(json_bytes(record).decode())
        process_status(directory/'status.json',status='running',seed=seed,condition=condition,update=step,
            checkpoint_sha256=digest,elapsed_seconds=time.perf_counter()-start)
    checkpoint(0)
    with (directory/'training.jsonl').open('x') as stream:
        for update in range(1,6001):
            u=wr.random((256,3));ids=dataset.sample_indices(static['partitions']['train'],u)
            uniforms=core.draw_uniforms(mr,256)
            gradients,row=kernel.training_gradients(nets,arrays['train']['x_PL'][ids],
                arrays['train']['rewards'][ids],live,uniforms,update,RULE)
            norm,scale=core.base.adam_step(nets,gradients,optimizer,update)
            row.update(update=update,seed=seed,rule=RULE,condition=condition,
                world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),
                batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                sample_uniforms_sha256=array_sha(uniforms),gradient_norm=norm,gradient_clip_scale=scale)
            stream.write(json_bytes(row).decode())
            if update in STEPS:stream.flush();checkpoint(update)
    final={part:evaluate_final(nets,arrays[part],condition,static['need_response_cases'][part],
        directory/f'final_{part}') for part in PARTS}
    result=dict(seed=seed,condition=condition,rule=RULE,updates=6000,initial_parameter_sha256=initial,
        final_parameter_sha256=core.parameter_hash(nets),final_checkpoint_sha256=sha(directory/'checkpoint_6000.npz'),
        training_log_sha256=sha(directory/'training.jsonl'),monitor=monitors,final=final,elapsed_seconds=time.perf_counter()-start)
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
        require(len(cells)==2 and {r['condition'] for r in cells}==set(CONDITIONS),'Incomplete paired conditions')
        require(len({r['initial_parameter_sha256'] for r in cells})==1,'Unpaired initialization')
        streams=[(execution/name(seed,c)/'training.jsonl').open() for c in CONDITIONS]
        try:
            count=0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines),'Unequal training log lengths')
                rows=[json.loads(line) for line in lines];count+=1
                require(all(r['update']==count and r['seed']==seed and r['condition']==c and r['rule']==RULE
                    for r,c in zip(rows,CONDITIONS)),'Training identity/update mismatch')
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256','entropy_coefficient'):
                    require(len({r[key] for r in rows})==1,f'Unpaired {key}')
            require(count==6000,'Training must contain exactly 6000 updates')
        finally:
            for stream in streams:stream.close()


def primary(runs):
    require(len(runs)==8,'Expected eight runs')
    by={(r['seed'],r['condition']):r for r in runs}
    require(set(by)=={(s,c) for s in SEEDS for c in CONDITIONS},'Missing/duplicate/extra paired seed')
    paired=[]
    for seed in SEEDS:
        values={c:by[seed,c]['final'][TARGET]['natural']['need_response']['Q'] for c in CONDITIONS}
        require(all(isinstance(q,(float,int)) and not isinstance(q,bool) and np.isfinite(q) and 0<=q<=1
            for q in values.values()),'Q must be defined on all nine strata')
        paired.append(dict(seed=seed,Q_live=values['PL_live'],Q_silent=values['PL_silent'],
            difference=values['PL_live']-values['PL_silent']))
    return dict(metric='Q',partition=TARGET,contrast='PL_live_minus_PL_silent',
        independent_paired_seed_blocks=4,paired_seeds=paired,
        mean_difference=float(np.mean([p['difference'] for p in paired])))


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
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),elapsed_seconds=time.perf_counter()-start,
            budget=static['budget'],runs=runs,array_hashes=hashes[0],primary=primary(runs))
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
