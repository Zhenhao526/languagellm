"""Destination-stratified partner ecology; unchanged discrete-message learner.

Prepare validates and snapshots only. Formal execution is exclusive, four spawn
workers, one per seed. All randomness and networks remain process-local.
"""
from __future__ import annotations
import os
THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
for _key in THREAD_KEYS:os.environ[_key]='1'

import argparse
from copy import deepcopy
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from research_program.triadic_message_study import runner as r
from research_program.triadic_partner_ecology_study import design,metrics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS,ECOLOGIES,CONDITIONS,PARTITIONS=design.SEEDS,design.ECOLOGIES,design.CONDITIONS,design.PARTITIONS
CHECKPOINTS=r.CHECKPOINTS
MESSAGE_RUNNER_SHA='5299c99bb92f3e18f8fae969347084c631a66f78e635d2abc4609c476cb9ae60'
require,finite,sha=r.require,r.finite,r.sha
read,write_new,json_hash,json_bytes,array_sha,now=r.read,r.write_new,r.json_hash,r.json_bytes,r.array_sha,r.now
CONFIG=deepcopy(r.CONFIG)
CONFIG.update(seeds=list(SEEDS),ecologies=list(ECOLOGIES),partitions=list(PARTITIONS),
    monitor_worlds_per_partition=672,monitor_per_destination_stratum=32,
    world_sampling='four_uniforms: destination_stratum, conditional_needs, layout, private_owner',
    population_weighting='equal21destination_strata_uniform_within_each_stratum',
    monitor_weighting='32frozen_worlds_per_stratum_equal672weights',
    run_order='four_spawn_workers_one_seed_each_ecology_then_condition_serial',
    same_initialization_and_message_uniforms_across_eight_arms=True,
    same_world_uniforms_destinations_layout_owners_across_eight_arms=True,
    same_actual_needs_states_only_within_ecology=True,
    endpoint_channel_control='live_networks_rerun_from_window1_cross_channels_closed',
    demand_generalization='all_supported_demands_trained_no_unseen_demand_partition',
    layout_generalization='reused18_6_development_split_not_independent_confirmation')


def make_prepared():
    p=design.make_prepared()
    p['config']=deepcopy(CONFIG)
    p['learner_source_sha256']=MESSAGE_RUNNER_SHA
    domain_worlds=sum(p['partitions'][e][part]['world_count'] for e in ECOLOGIES for part in PARTITIONS)
    p.update(categorical_message_samples_total=32*6000*256*2*24,
        weighted_structural_action_contributions=32*6000*256*2*24,
        offline_reward_table_entries_per_worker=domain_worlds*24,
        offline_reward_table_entries_actual=4*domain_worlds*24,
        planned_checkpoint_files=32*6,planned_natural_monitor_files=32*6*2,
        planned_closed_monitor_forward_files=4*2*2*6*2,
        planned_natural_final_files=32*2,planned_closed_final_forward_files=4*2*2*2,
        monitor_forward_worlds_total=(32+16)*6*2*672,
        all_final_forward_worlds_total=p['full_natural_worlds_total']+p['full_closed_worlds_total'],
        independent_team_initializations=4,automatic_followon_experiment=False)
    return p


def sources():
    require(sha(r.__file__)==MESSAGE_RUNNER_SHA,'Frozen message learner changed')
    require(sha(r.base.__file__)==r.BASE_SHA,'Frozen base learner changed')
    require(sha(r.coordination.__file__)==r.COORDINATION_SHA,'Frozen coordination learner changed')
    paths=[Path(__file__),HERE/'design.py',HERE/'metrics.py',HERE/'plan.md',HERE/'运行说明.md',
        HERE/'tests/test_runner.py',HERE/'tests/test_design.py',HERE/'tests/test_metrics.py',
        HERE/'seed_selection_receipt.json',HERE/'design_test_receipt.json',
        HERE/'design_static_001/enumerate_design.py',HERE/'design_static_001/results.json',
        HERE/'design_static_001/marginals.json',HERE/'design_static_001/weighted_semantic_support.json',
        Path(r.__file__),Path(r.base.__file__),Path(r.coordination.__file__),Path(r.base.env.__file__)]
    require(all(p.is_file() for p in paths),'Missing source/plan/test/static receipt before freezing')
    return {str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}


def prepare(output):
    output=Path(output).resolve();require(not output.exists(),'Refuse to overwrite preparation')
    hashes=sources();prepared=make_prepared()
    plan=dict(schema='triadic_partner_ecology_v1',prepared_at=now(),config=deepcopy(CONFIG),
        prepared_sha256=json_hash(prepared),sources=hashes,
        runtime={'python':platform.python_version(),'numpy':np.__version__},
        no_training_or_network_initialization_by_prepare=True)
    output.mkdir(parents=True,exist_ok=False)
    for relative in hashes:
        target=output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    write_new(output/'prepared.json',prepared);write_new(output/'plan.json',plan)
    write_new(output/'freeze.json',{'plan_sha256':sha(output/'plan.json'),'prepared_sha256':sha(output/'prepared.json')})
    verify(output)
    return dict(status='prepared_not_trained',output=str(output),plan_sha256=sha(output/'plan.json'),runs=32)


def verify(output):
    output=Path(output).resolve();plan,prepared,freeze=[read(output/n) for n in ('plan.json','prepared.json','freeze.json')]
    require(sha(output/'plan.json')==freeze['plan_sha256'],'Plan changed')
    require(sha(output/'prepared.json')==freeze['prepared_sha256']==plan['prepared_sha256'],'Prepared input changed')
    require(json_hash(make_prepared())==plan['prepared_sha256'],'Current prepared values differ')
    require(plan['config']==CONFIG and plan['runtime']=={'python':platform.python_version(),'numpy':np.__version__},'Configuration or runtime changed')
    require(plan['sources']==sources(),'Current sources changed')
    for relative,digest in plan['sources'].items():
        require(sha(output/'source_snapshot'/relative)==digest,'Frozen source snapshot changed')
    return plan,prepared


def save_weights(path,spec,execution):
    require(not path.exists(),'Weight output exists')
    ids=np.asarray(spec['monitor_indices'],dtype=np.int64)
    full=design.evaluation_weights(spec);monitor=design.evaluation_weights(spec,ids)
    np.savez_compressed(path,full_weights=full,monitor_weights=monitor,monitor_indices=ids)
    return dict(file=str(path.relative_to(execution)),file_sha256=sha(path),
        full_weights_sha256=array_sha(full),monitor_weights_sha256=array_sha(monitor),
        monitor_indices_sha256=array_sha(ids))


def evaluate_weighted(networks,arrays,condition,spec,indices,save_path,weight_receipt,execution):
    """Same greedy evaluator; keep uniform diagnostics apart from target weights."""
    uniform=r.evaluate(networks,arrays,condition,indices,save_path=save_path)
    weights=design.evaluation_weights(spec,indices)
    expected_key='full_weights_sha256' if indices is None else 'monitor_weights_sha256'
    require(array_sha(weights)==weight_receipt[expected_key],'Evaluation weights differ from frozen worker file')
    with np.load(save_path,allow_pickle=False) as data:
        result=metrics.summarize(data['states'],data['action_indices'],data['greedy_reward'],weights)
        # Explicitly conditional on greedy messages, just as the old evaluator.
        result['weighted'].update({
            'conditional_exact_expected_reward_mean':float(weights@data['conditional_exact_expected_reward']),
            'conditional_exact_full_success_probability_mean':float(weights@data['conditional_exact_full_success_probability']),
            'conditional_exact_physical_execution_probability_mean':float(weights@data['conditional_exact_execution_probability'])})
    return dict(uniform=uniform,weighted=result['weighted'],raw=result['raw'],
        data_file=str(save_path.relative_to(execution)),data_sha256=sha(save_path),
        weights_file=weight_receipt['file'],weights_file_sha256=weight_receipt['file_sha256'],
        weight_array='full_weights' if indices is None else 'monitor_weights',weights_sha256=array_sha(weights),
        actual_rollout_condition=condition,reused_natural=False,
        population_weighting=spec['population_weighting'] if indices is None else spec['monitor_weighting'])


def evaluate_modes(networks,arrays,condition,spec,indices,prefix,weight_receipt,execution):
    natural=evaluate_weighted(networks,arrays,condition,spec,indices,
                              Path(str(prefix)+'_natural.npz'),weight_receipt,execution)
    if condition.endswith('_live'):
        closed_condition=condition.replace('_live','_silent')
        closed=evaluate_weighted(networks,arrays,closed_condition,spec,indices,
                                 Path(str(prefix)+'_closed.npz'),weight_receipt,execution)
        closed['control']='same_parameters_from_window1_cross_channels_closed_own_messages_retained'
    else:
        closed=deepcopy(natural);closed.update(reused_natural=True,
            control='already_silent_identical_route_no_duplicate_forward')
    return dict(natural=natural,closed=closed)


def train_run(seed,ecology,condition,prepared,arrays,weight_receipts,output,execution):
    full,live=r.condition_settings(condition);output.mkdir(exist_ok=False)
    networks=r.make_networks(seed);initial_sha=r.parameter_hash(networks);optimizer=r.base.make_adam(networks)
    batch_rng=np.random.default_rng(np.random.SeedSequence([seed,200]));message_rngs=r.make_message_rngs(seed)
    started=time.perf_counter();monitor=[];specs=prepared['partitions'][ecology]
    def checkpoint(update):
        digest=r.save_checkpoint(output/f'checkpoint_{update:04d}.npz',networks,optimizer,update,batch_rng,message_rngs)
        scores={part:evaluate_modes(networks,arrays[part],condition,specs[part],np.asarray(specs[part]['monitor_indices'],dtype=np.int64),
            output/f'monitor_{update:04d}_{part}',weight_receipts[part],execution) for part in PARTITIONS}
        row=dict(update=update,checkpoint_sha256=digest,monitor=scores,elapsed_seconds=time.perf_counter()-started)
        monitor.append(row)
        with (output/'monitor.jsonl').open('a',encoding='utf-8') as stream:stream.write(json_bytes(row).decode())
    checkpoint(0)
    with (output/'training.jsonl').open('x',encoding='utf-8') as stream:
        for update in range(1,CONFIG['updates']+1):
            world_uniforms=batch_rng.random((CONFIG['batch_size'],4))
            ids=design.sample_indices(specs['train'],world_uniforms)
            paired=design.pairing_fields(specs['train'],ids)
            uniforms=r.draw_uniforms(message_rngs,len(ids))
            gradients,row=r.training_gradients(networks,arrays['train']['x_FI' if full else 'x_PI'][ids],
                arrays['train']['rewards'][ids],live,uniforms,update)
            norm,scale=r.base.adam_step(networks,gradients,optimizer,update)
            row.update(update=update,seed=seed,ecology=ecology,condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms),batch_indices_sha256=array_sha(ids),
                batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                destinations_sha256=array_sha(paired['destinations']),layout_indices_sha256=array_sha(paired['layout_indices']),
                owner_indices_sha256=array_sha(paired['owner_indices']),sample_uniforms_sha256=array_sha(uniforms),
                gradient_norm=norm,gradient_clip_scale=scale,elapsed_seconds=time.perf_counter()-started)
            stream.write(json_bytes(row).decode())
            if update in CHECKPOINTS:stream.flush();checkpoint(update)
    final={part:evaluate_modes(networks,arrays[part],condition,specs[part],None,
             output/f'final_{part}',weight_receipts[part],execution) for part in PARTITIONS}
    counts={}
    for part in PARTITIONS:
        for row in final[part]['natural']['uniform']['joint_argmax_action_counts']:
            key=tuple(row['action_indices']);counts[key]=counts.get(key,0)+row['worlds']
    domain=dict(worlds=sum(counts.values()),distinct_joint_argmax_actions=len(counts),
        joint_argmax_action_counts=[dict(action_indices=list(a),worlds=n) for a,n in sorted(counts.items())],
        weighting='raw_complete_domain_counts_only_not_target_distribution')
    require(domain['worlds']==sum(specs[p]['world_count'] for p in PARTITIONS),'Incomplete domain actions')
    result=dict(seed=seed,ecology=ecology,condition=condition,updates=CONFIG['updates'],
        training_world_samples=CONFIG['updates']*CONFIG['batch_size'],sampled_complete_message_trajectories=CONFIG['updates']*CONFIG['batch_size']*2,
        initial_parameter_sha256=initial_sha,final_parameter_sha256=r.parameter_hash(networks),final=final,
        full_domain_actions=domain,monitor=monitor,
        candidate_threshold_met=final['heldout_layouts']['natural']['weighted']['full_success_rate']>=.99,
        training_log_sha256=sha(output/'training.jsonl'),final_checkpoint_sha256=sha(output/f"checkpoint_{CONFIG['updates']:04d}.npz"),
        elapsed_seconds=time.perf_counter()-started)
    write_new(output/'result.json',result);return result


def check_pairing(execution,results):
    index={(row['seed'],row['ecology'],row['condition']):row for row in results}
    expected={(s,e,c) for s in SEEDS for e in ECOLOGIES for c in CONDITIONS}
    require(len(results)==32 and set(index)==expected,'Missing or duplicate ecology/condition run')
    checks=[]
    common=('update','world_uniforms_sha256','destinations_sha256','layout_indices_sha256',
            'owner_indices_sha256','sample_uniforms_sha256','entropy_coefficient')
    within=('batch_indices_sha256','batch_states_sha256')
    for seed in SEEDS:
        keys=[(seed,e,c) for e in ECOLOGIES for c in CONDITIONS]
        require(len({index[key]['initial_parameter_sha256'] for key in keys})==1,'Cross-arm initial parameters differ')
        streams=[(execution/f'seed_{s}_{e}_{c}'/'training.jsonl').open() for s,e,c in keys]
        try:
            count=0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines),'Unequal paired training log length')
                group=[json.loads(line) for line in lines]
                require(all(all(row[k]==group[0][k] for k in common) for row in group),'Paired common uniform/context differs')
                require(group[0]['update']==count+1,'Nonsequential paired update')
                for i,key in enumerate(keys):require((group[i]['seed'],group[i]['ecology'],group[i]['condition'])==key,'Training row identity mismatch')
                for ei in range(2):
                    rows=group[ei*4:(ei+1)*4]
                    require(all(all(row[k]==rows[0][k] for k in within) for row in rows),'Within-ecology actual states differ')
                count+=1
            require(count==CONFIG['updates'],'Missing paired updates')
        finally:
            for stream in streams:stream.close()
        checks.append(dict(seed=seed,same_initial_parameters=True,common_uniform_context_updates=count,
            actual_states_matched_within_ecology=True,actual_states_not_required_equal_across_ecologies=True))
    return checks


def seed_worker(output,seed):
    output=Path(output).resolve();_,prepared=verify(output);require(seed in SEEDS,'Unknown worker seed')
    execution=output/'execution';require((execution/'started.json').is_file(),'Worker requires started batch')
    worker=execution/f'worker_seed_{seed}';worker.mkdir(exist_ok=False);started=time.perf_counter()
    write_new(worker/'started.json',dict(seed=seed,pid=os.getpid(),parent_pid=os.getppid(),started_at=now(),
        plan_sha256=sha(output/'plan.json'),thread_environment={key:os.environ[key] for key in THREAD_KEYS}))
    try:
        # Rebuild both domains once per worker; never share arrays or RNG objects.
        arrays={e:{p:r.build_arrays(prepared['partitions'][e][p]) for p in PARTITIONS} for e in ECOLOGIES}
        receipts={e:{p:save_weights(worker/f'weights_{e}_{p}.npz',prepared['partitions'][e][p],execution)
                     for p in PARTITIONS} for e in ECOLOGIES}
        inputs={e:{p:dict(worlds=len(a['states']),full_information_features_sha256=array_sha(a['x_FI']),
            private_information_features_sha256=array_sha(a['x_PI']),native_rewards_sha256=array_sha(a['rewards']),
            states_sha256=array_sha(a['packed_states']),weights=receipts[e][p]) for p,a in pa.items()} for e,pa in arrays.items()}
        write_new(worker/'input_arrays.json',inputs)
        results=[]
        for run in prepared['runs']:
            if run['seed']!=seed:continue
            e,c=run['ecology'],run['condition']
            result=train_run(seed,e,c,prepared,arrays[e],receipts[e],execution/run['directory'],execution)
            results.append(result)
            print(json.dumps(dict(completed_run=run,weighted_full_success_rates={p:v['natural']['weighted']['full_success_rate'] for p,v in result['final'].items()},
                worker_elapsed_seconds=time.perf_counter()-started)),flush=True)
        verify(output)
        require([(row['ecology'],row['condition']) for row in results]==[(e,c) for e in ECOLOGIES for c in CONDITIONS],'Incomplete worker sequence')
        write_new(worker/'results.json',dict(status='completed',seed=seed,runs=results,
            offline_reward_table_entries=prepared['offline_reward_table_entries_per_worker'],elapsed_seconds=time.perf_counter()-started))
        write_new(worker/'status.json',dict(status='completed',completed_at=now()))
    except BaseException as error:
        write_new(worker/'failure.json',dict(status='failed',seed=seed,failed_at=now(),error_type=type(error).__name__,
            error=str(error),elapsed_seconds=time.perf_counter()-started))
        write_new(worker/'status.json',dict(status='failed',failed_at=now()));raise


stop_own_workers,wait_for_workers=r.stop_own_workers,r.wait_for_workers


def normalized_inputs(inputs):
    value=deepcopy(inputs)
    # Worker-relative paths/ZIP metadata may differ; all numerical arrays must match.
    for e in ECOLOGIES:
        for p in PARTITIONS:
            value[e][p]['weights'].pop('file')
            value[e][p]['weights'].pop('file_sha256')
    return value


def execute(output):
    output=Path(output).resolve();_,prepared=verify(output);execution=output/'execution'
    require(not execution.exists(),'Never resume or overwrite started execution')
    execution.mkdir(exist_ok=False);started=time.perf_counter();processes=[]
    write_new(execution/'started.json',dict(started_at=now(),plan_sha256=sha(output/'plan.json'),device='cpu_numpy',pid=os.getpid(),
        thread_environment={key:os.environ[key] for key in THREAD_KEYS}))
    try:
        context=multiprocessing.get_context('spawn')
        for seed in SEEDS:
            process=context.Process(target=seed_worker,args=(str(output),seed),name=f'partner_ecology_seed_{seed}')
            processes.append(process);process.start()
        wait_for_workers(processes);results=[];inputs=[]
        for seed in SEEDS:
            worker=execution/f'worker_seed_{seed}';record=read(worker/'results.json')
            require(record['status']=='completed' and record['seed']==seed,'Wrong or partial worker result')
            require([(row['ecology'],row['condition']) for row in record['runs']]==[(e,c) for e in ECOLOGIES for c in CONDITIONS],'Wrong worker arm order')
            for row in record['runs']:
                require(row==read(execution/f"seed_{seed}_{row['ecology']}_{row['condition']}"/'result.json'),'Worker aggregate differs from run')
            results.extend(record['runs']);inputs.append(read(worker/'input_arrays.json'))
        require(all(normalized_inputs(x)==normalized_inputs(inputs[0]) for x in inputs),'Worker input arrays differ')
        write_new(execution/'input_arrays.json',dict(worker_count=4,all_worker_arrays_and_weights_identical=True,
            arrays_by_worker={str(s):x for s,x in zip(SEEDS,inputs)},actual_offline_reward_table_entries=prepared['offline_reward_table_entries_actual']))
        pairing=check_pairing(execution,results);verify(output)
        result=dict(status='completed',completed_at=now(),plan_sha256=sha(output/'plan.json'),runs=results,
            completed_run_count=32,paired_seed_count=4,pairing_checks=pairing,primary_comparison=metrics.primary_comparison(results),
            updates_total=32*6000,training_world_samples_total=prepared['training_state_samples_total'],
            sampled_complete_message_trajectories_total=prepared['sampled_complete_message_trajectories_total'],
            weighted_structural_action_contributions=prepared['weighted_structural_action_contributions'],
            offline_reward_table_entries_actual=prepared['offline_reward_table_entries_actual'],
            worker_processes=[dict(name=p.name,pid=p.pid,exitcode=p.exitcode) for p in processes],
            all_seed_candidates_by_ecology_condition={e:{c:all(row['candidate_threshold_met'] for row in results if row['ecology']==e and row['condition']==c)
                for c in CONDITIONS} for e in ECOLOGIES},experiment_type='discrete_symbol_message_co_learning_partner_ecology',
            messages_generated=True,automatic_followon_experiment=False,language_or_convention_claim_automatically_supported=False,
            elapsed_seconds=time.perf_counter()-started,
            scope='Four paired initializations; destination-stratified ecological supports; development layout holdout only; exact counterfactual-rich receiver training.')
        write_new(execution/'results.json',result);write_new(execution/'status.json',dict(status='completed',completed_at=now(),completed_runs=32))
    except BaseException as error:
        stop_own_workers(processes)
        write_new(execution/'failure.json',dict(status='failed',failed_at=now(),error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started))
        write_new(execution/'status.json',dict(status='failed',failed_at=now()));raise
    return dict(status='completed',output=str(execution),primary_comparison=result['primary_comparison'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','execute'))
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.out)
    elif args.command=='verify':verify(args.out);result=dict(status='verified_not_trained',output=str(args.out.resolve()))
    else:result=execute(args.out)
    print(json.dumps(result,ensure_ascii=False))
