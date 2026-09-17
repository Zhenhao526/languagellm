"""Fresh paired FI-silent policies; only the physical execution rule changes."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from itertools import zip_longest
from copy import deepcopy
import argparse,json,platform,shutil,time,multiprocessing
import numpy as np
from research_program.triadic_action_dependency_study import runner as old, dataset
from . import environment, kernel

core=old.core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
SEEDS=(57101,57102,57103,57104)
RULES=('strict','reciprocal')
PARTS=old.PARTS;STEPS=old.STEPS
CONFIG=deepcopy(old.CONFIG)
CONFIG.update(seeds=list(SEEDS),conditions=['FI_silent'],execution_rules=list(RULES),
    task='same_unique_execution_plan_with_two_physical_settlement_rules',
    information='FI_all_needs_and_public_layout; no_cross_agent_messages',
    primary='6000_double_holdout_full_success_reciprocal_minus_strict',
    secondary_task_contrast='actual_executed_partner_correct_and_proposal_role_correct_separately',
    training_objective='mean_log_exact_expected_native_reward_plus_original_action_entropy; sender_two_trajectory_LOO',
    native_reward='number_of_satisfied_actual_executors_divided_by_two',
    cross_settlement='both_rules_on_each_saved_proposal; no_new_forward; fixed_secondary_decomposition',
    run_order='four_spawn_workers_one_seed_each_strict_then_reciprocal',
    selection='fixed_budget_no_early_stop_or_extra_seeds',automatic_followon_experiment=False)
sha,read,write,json_hash,json_bytes,array_sha,now=old.sha,old.read,old.write,old.json_hash,old.json_bytes,old.array_sha,old.now

def name(seed,rule):return f'seed_{seed}_{rule}_FI_silent'

def prepared():
    source=read(ORIGINAL/'prepared.json');freeze=read(ORIGINAL/'freeze.json')
    assert sha(ORIGINAL/'prepared.json')==freeze['prepared_sha256']=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'
    assert sha(ORIGINAL/'plan.json')==freeze['plan_sha256']=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba'
    parts=source['partitions'];worlds=sum(s['world_count'] for s in parts.values());mon=sum(len(s['monitor_indices']) for s in parts.values())
    assert worlds==774144 and mon==21504
    return dict(schema='triadic_reciprocal_execution_fi_v1',partitions=parts,source_prepared_sha256=sha(ORIGINAL/'prepared.json'),
        independent_initializations=4,budget=dict(runs=8,training_updates=48000,training_world_samples=48000*256,
        message_trajectories=48000*256*2,categorical_symbol_samples=48000*256*2*24,
        training_forward_module_samples=48000*256*2*9,checkpoints=48,actual_monitor_files=8*6*4,actual_final_files=8*4,
        aliased_evaluations=0,actual_monitor_worlds=8*6*mon,actual_final_worlds=8*worlds,
        monitor_forward_module_samples=8*6*mon*9,final_forward_module_samples=8*worlds*9,
        evaluation_forward_module_samples=(8*6*mon+8*worlds)*9))

def sources():
    out={}
    for relative,digest in read(ORIGINAL/'plan.json')['sources'].items():
        path=ROOT/relative;assert sha(path)==digest;out[str(path)]=digest
    # The strict reference module is imported by the new kernel for compatibility.
    ref=HERE.parent/'triadic_partial_payoff_study/utility.py'
    old_payoff=read(HERE.parent/'triadic_partial_payoff_study/results/payoff_001/plan.json')
    assert sha(ref)==old_payoff['source_sha256'][str(ref)];out[str(ref)]=sha(ref)
    files=('__init__.py','runner.py','environment.py','kernel.py','plan.md','design_review.md','literature_sources.json',
        'tests/test_runner.py','tests/test_environment.py','tests/test_kernel.py',
        'main_preflight.json','environment_preflight_001.json','kernel_preflight_001.json')
    for f in files:
        path=HERE/f;out[str(path)]=sha(path)
    for f in ('候选规则静态审查.md','review.md','candidate_static_003.json','independent_bounds_review_001.json',
              'static_execution_history_001.json','enumerate_candidate.py','independent_bounds_review.py'):
        path=HERE/'design_audit'/f;out[str(path)]=sha(path)
    return out

def prepare(out):
    out=Path(out).resolve();assert not out.exists();static=prepared();ss=sources();out.mkdir(parents=True)
    for path in ss:
        target=out/'source_snapshot'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write(out/'prepared.json',static)
    write(out/'plan.json',dict(status='prepared_without_training',at=now(),config=CONFIG,source_sha256=ss,
        inputs_sha256={str(ORIGINAL/f):sha(ORIGINAL/f) for f in ('plan.json','prepared.json','freeze.json')},
        prepared_sha256=sha(out/'prepared.json'),runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')));verify(out)
    return dict(status='prepared_without_training',plan_sha256=sha(out/'plan.json'),budget=static['budget'])

def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json');static=read(out/'prepared.json')
    assert sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256'] and plan['config']==CONFIG
    assert plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__)
    assert sha(out/'prepared.json')==plan['prepared_sha256'] and static==prepared() and plan['source_sha256']==sources()
    for path,digest in plan['inputs_sha256'].items():assert sha(path)==digest
    for path,digest in plan['source_sha256'].items():assert sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest
    return plan,static

def make_arrays(spec):
    a=dataset.make_arrays(spec,information='FI');del a['states'];return a

def evaluate(networks,arrays,rule,indices,path):
    raw=np.asarray(indices);assert raw.ndim==1 and np.issubdtype(raw.dtype,np.integer)
    ids=raw.astype(np.int64);n=len(ids);path=Path(path)
    assert rule in RULES and n and len(np.unique(ids))==n and np.all((ids>=0)&(ids<len(arrays['packed_states']))) and not path.exists()
    data=dict(states=arrays['packed_states'][ids],state_indices=ids,messages=np.empty((n,2,3,4),np.int8),
        action_indices=np.empty((n,3),np.int16),action_probabilities=np.empty((n,3,17)),
        conditional_exact_expected_reward=np.empty(n),conditional_exact_full_success_probability=np.empty(n),
        conditional_exact_execution_probability=np.empty(n),conditional_full_posterior_mass=np.empty(n))
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));ix=ids[sl]
        trace=core.rollout(networks,arrays['x_FI'][ix],False)
        terms=kernel.objective_terms(trace['action_logits'],arrays['rewards'][ix],rule);p=terms['probabilities']
        data['messages'][sl]=trace['messages'];data['action_probabilities'][sl]=p;data['action_indices'][sl]=p.argmax(-1)
        data['conditional_exact_expected_reward'][sl]=terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl]=terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl]=terms['execution_probability']
        data['conditional_full_posterior_mass'][sl]=terms['full_success_posterior_mass']
    data.update(environment.settle(data['states'],data['action_indices'],rule))
    np.savez_compressed(path,**data)
    record=environment.summarize(data['states'],data['action_indices'],rule,arrays['rewards'][ids])
    own=deepcopy(record)
    record['cross_settlement']={r:own if r==rule else environment.summarize(data['states'],data['action_indices'],r,arrays['rewards'][ids]) for r in RULES}
    record.update(path=str(path),data_sha256=sha(path),information='FI',live=False,
        expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
        execution_probability_given_greedy_messages=float(data['conditional_exact_execution_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(data['conditional_full_posterior_mass'].mean()),state_indices_sha256=array_sha(ids))
    return record

def train_run(seed,rule,static,arrays,execution):
    directory=execution/name(seed,rule);directory.mkdir(exist_ok=False)
    nets=core.make_networks(seed);initial=core.parameter_hash(nets);optimizer=core.base.make_adam(nets)
    wr=np.random.default_rng(np.random.SeedSequence([seed,200]));mr=core.make_message_rngs(seed)
    monitors=[];start=time.perf_counter()
    def checkpoint(step):
        ckpt=directory/f'checkpoint_{step:04d}.npz';digest=core.save_checkpoint(ckpt,nets,optimizer,step,wr,mr)
        evaluations={part:evaluate(nets,arrays[part],rule,static['partitions'][part]['monitor_indices'],directory/f'monitor_{step:04d}_{part}.npz') for part in PARTS}
        rec=dict(update=step,checkpoint_sha256=digest,monitor=evaluations,elapsed_seconds=time.perf_counter()-start);monitors.append(rec)
        with (directory/'monitor.jsonl').open('a') as f:f.write(json_bytes(rec).decode())
    checkpoint(0)
    with (directory/'training.jsonl').open('x') as f:
        for update in range(1,6001):
            u=wr.random((256,3));ids=dataset.sample_indices(static['partitions']['train'],u);mu=core.draw_uniforms(mr,256)
            gradients,row=kernel.training_gradients(nets,arrays['train']['x_FI'][ids],arrays['train']['rewards'][ids],False,mu,update,rule)
            norm,scale=core.base.adam_step(nets,gradients,optimizer,update)
            row.update(update=update,seed=seed,rule=rule,condition='FI_silent',world_uniforms_sha256=array_sha(u),
                batch_indices_sha256=array_sha(ids),batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                sample_uniforms_sha256=array_sha(mu),gradient_norm=norm,gradient_clip_scale=scale)
            f.write(json_bytes(row).decode())
            if update in STEPS:f.flush();checkpoint(update)
    final={part:evaluate(nets,arrays[part],rule,np.arange(len(arrays[part]['packed_states'])),directory/f'final_{part}.npz') for part in PARTS}
    result=dict(seed=seed,rule=rule,condition='FI_silent',updates=6000,initial_parameter_sha256=initial,
        final_parameter_sha256=core.parameter_hash(nets),final_checkpoint_sha256=sha(directory/'checkpoint_6000.npz'),
        training_log_sha256=sha(directory/'training.jsonl'),monitor=monitors,final=final,elapsed_seconds=time.perf_counter()-start)
    write(directory/'result.json',result);return result

def worker(payload):
    seed,static,execution=payload;execution=Path(execution);start=time.perf_counter()
    arrays={part:make_arrays(static['partitions'][part]) for part in PARTS}
    hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI')} for p,a in arrays.items()}
    write(execution/f'seed_{seed}_arrays.json',dict(array_hashes=hashes,build_seconds=time.perf_counter()-start))
    print(json.dumps(dict(stage='arrays_prepared',seed=seed)),flush=True);runs=[]
    for rule in RULES:
        runs.append(train_run(seed,rule,static,arrays,execution))
        print(json.dumps(dict(stage='run_completed',seed=seed,rule=rule,elapsed_seconds=time.perf_counter()-start)),flush=True)
    assert hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI')} for p,a in arrays.items()}
    return runs

def verify_pairing(execution,runs):
    for seed in SEEDS:
        cells=[r for r in runs if r['seed']==seed];assert len(cells)==2 and len({r['initial_parameter_sha256'] for r in cells})==1
        streams=[(execution/name(seed,rule)/'training.jsonl').open() for rule in RULES]
        try:
            count=0
            for lines in zip_longest(*streams):
                assert all(line is not None for line in lines);rows=[json.loads(line) for line in lines];count+=1
                assert all(r['update']==count for r in rows)
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256','entropy_coefficient'):
                    assert len({r[key] for r in rows})==1
            assert count==6000
        finally:
            for stream in streams:stream.close()

def primary(runs):
    assert len(runs)==8;by={(r['seed'],r['rule']):r for r in runs};assert len(by)==8;values=[]
    for seed in SEEDS:
        cells={rule:by[seed,rule]['final']['new_needs_and_layouts'] for rule in RULES}
        metrics=('full_success_rate','executed_partner_correct_rate','proposal_role_success_rate','reward_mean')
        differences={m:cells['reciprocal'][m]-cells['strict'][m] for m in metrics}
        matrix={trained:{settlement:cells[trained]['cross_settlement'][settlement]['full_success_rate'] for settlement in RULES} for trained in RULES}
        ss=matrix['strict']['strict'];sr=matrix['strict']['reciprocal'];rs=matrix['reciprocal']['strict'];rr=matrix['reciprocal']['reciprocal']
        decomposition=dict(common_reciprocal_policy_difference=rr-sr,strict_policy_mechanical_release=sr-ss,
            common_strict_policy_difference=rs-ss,reciprocal_policy_mechanical_release=rr-rs)
        assert abs(differences['full_success_rate']-(decomposition['common_reciprocal_policy_difference']+decomposition['strict_policy_mechanical_release']))<1e-12
        values.append(dict(seed=seed,contrasts=differences,cells={r:{m:c[m] for m in metrics} for r,c in cells.items()},
            full_success_cross_settlement=matrix,decomposition=decomposition))
    return dict(metric='full_success_rate',partition='new_needs_and_layouts',paired_seeds=values,
        mean_difference=float(np.mean([r['contrasts']['full_success_rate'] for r in values])))

def execute(out):
    out=Path(out).resolve();plan,static=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False);start=time.perf_counter()
    write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    try:
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            groups=pool.map(worker,[(s,static,str(execution)) for s in SEEDS])
        runs=[r for group in groups for r in group];assert [(r['seed'],r['rule']) for r in runs]==[(s,r) for s in SEEDS for r in RULES]
        verify_pairing(execution,runs);hashes=[read(execution/f'seed_{s}_arrays.json')['array_hashes'] for s in SEEDS];assert all(v==hashes[0] for v in hashes)
        verify(out)
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),elapsed_seconds=time.perf_counter()-start,
            budget=static['budget'],runs=runs,array_hashes=hashes[0],primary=primary(runs))
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return dict(status='completed',elapsed_seconds=result['elapsed_seconds'],primary=result['primary'])
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),elapsed_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','verify','execute'));p.add_argument('--out',required=True)
    a=p.parse_args();result=globals()[a.command](a.out)
    print(json.dumps(result if a.command!='verify' else dict(status='verified'),ensure_ascii=False))
