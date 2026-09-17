"""One frozen matched-world experiment; reuse the audited discrete learner."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from copy import deepcopy
from itertools import product, zip_longest
from pathlib import Path
import argparse
import json
import multiprocessing
import platform
import shutil
import time
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset, environment as env

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SEEDS=(51101,51102,51103,51104)
CONDITIONS=('FI_silent','FI_live','PL_silent','PL_live','LL_silent','LL_live')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
OLD_CORE_SHA='5299c99bb92f3e18f8fae969347084c631a66f78e635d2abc4609c476cb9ae60'
CONFIG=deepcopy(core.CONFIG)
CONFIG.update(seeds=list(SEEDS),conditions=list(CONDITIONS),partitions=list(PARTS),
    task='eight_resource_predicates_single_full_plan',information='FI_or_private_needs_with_public_or_local_layout',
    feature_semantics='54: need-known/kind-acceptance/length-acceptance/destination-acceptance; official views only',
    primary='6000_both_holdouts_content_both_apt_PL_live_minus_LL_live',
    secondary_task_contrast='both_holdouts_full_success_(PL_live-PL_silent)-(LL_live-LL_silent)',
    world_sampling='three_uniforms_uniform_need_layout_owner',monitor_worlds_per_partition='all_partition_needs_times_two_fixed_backgrounds',
    demand_generalization='unseen_joint_need_relation_orbits_not_unseen_individual_attribute_values',
    layout_generalization='new_sha_fixed_18_6_split',automatic_followon_experiment=False,
    run_order='four_spawn_workers_one_seed_each_six_conditions_serial',worker_count=4)
CONFIG.pop('candidate_min_full_success_rate_each_heldout_partition_each_seed',None)
sha,read,write,json_hash,json_bytes,array_sha,now=(core.sha,core.read,core.write_new,core.json_hash,core.json_bytes,core.array_sha,core.now)


def sources():
    assert sha(core.__file__)==OLD_CORE_SHA
    assert sha(core.base.__file__)==core.BASE_SHA
    assert sha(core.coordination.__file__)==core.COORDINATION_SHA
    paths=[HERE/n for n in ('__init__.py','runner.py','environment.py','dataset.py','metrics.py','plan.md',
        'tests/test_environment.py','tests/test_dataset.py','tests/test_runner.py','tests/test_metrics.py',
        'environment_dataset_preflight_001.json','metrics_test_receipt.json','main_preflight.json')]
    paths += [Path(core.__file__),Path(core.base.__file__),Path(core.coordination.__file__),Path(core.base.env.__file__)]
    return {str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}


def make_prepared():
    prepared=dataset.make_prepared()
    assert tuple(prepared['partitions'])==PARTS or set(prepared['partitions'])==set(PARTS)
    worlds=sum(s['world_count'] for s in prepared['partitions'].values())
    assert worlds==774144
    prepared['execution_budget']=dict(runs=24,training_updates=24*6000,
        training_world_samples=24*6000*256,message_trajectories=24*6000*256*2,
        categorical_symbol_samples=24*6000*256*2*24,
        checkpoints=24*6,natural_monitor_files=24*6*4,closed_monitor_files=12*6*4,
        natural_final_files=24*4,closed_final_files=12*4,
        natural_final_worlds=24*worlds,closed_final_worlds=12*worlds,
        final_network_samples=36*worlds*9,
        monitor_worlds=36*6*sum(len(s['monitor_indices']) for s in prepared['partitions'].values()))
    return prepared


def prepare(out):
    out=Path(out).resolve();assert not out.exists()
    prepared=make_prepared(); ss=sources()
    out.mkdir(parents=True)
    for relative in ss:
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    write(out/'prepared.json',prepared)
    write(out/'plan.json',dict(status='prepared_without_training',created_at=now(),config=CONFIG,
        runtime=dict(python=platform.python_version(),numpy=np.__version__),sources=ss,
        prepared_sha256=sha(out/'prepared.json'),independent_team_initializations=4))
    write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json'),prepared_sha256=sha(out/'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training',plan_sha256=sha(out/'plan.json'),budget=prepared['execution_budget'])


def verify(out):
    out=Path(out).resolve();p=read(out/'plan.json');v=read(out/'prepared.json');f=read(out/'freeze.json')
    assert sha(out/'plan.json')==f['plan_sha256']
    assert sha(out/'prepared.json')==f['prepared_sha256']==p['prepared_sha256']
    assert json_hash(v)==json_hash(make_prepared())
    assert p['config']==CONFIG and p['runtime']==dict(python=platform.python_version(),numpy=np.__version__)
    assert p['sources']==sources()
    for relative,digest in p['sources'].items():assert sha(out/'source_snapshot'/relative)==digest
    return p,v


def native(packed,choices):
    """Exact original atomic matching and new resource acceptance; no plan mask."""
    n=len(packed)
    matched=np.any(np.all(choices[:,None,:]==core.base.JOINT_ACTIONS[None,:,:],axis=-1),axis=1)
    executed=matched[:,None] & (choices!=0)
    site=np.maximum(choices-1,0)//4;destination=(np.maximum(choices-1,0)//2)%2
    material=packed[np.arange(n)[:,None],3+site]
    acceptance=np.asarray([[[env.accepts(need,m,d) for d in range(2)] for m in range(4)] for need in range(24)])
    satisfied=executed & acceptance[packed[:,:3],material,destination]
    return dict(greedy_reward=satisfied.sum(1)/2,executed=executed,satisfied=satisfied)


def evaluate(networks,arrays,information,live,indices,path):
    indices=np.asarray(indices,dtype=np.int64); n=len(indices)
    assert n and len(np.unique(indices))==n and not path.exists()
    data=dict(states=arrays['packed_states'][indices],state_indices=indices,
        messages=np.empty((n,2,3,4),dtype=np.int8),action_indices=np.empty((n,3),dtype=np.int16),
        action_probabilities=np.empty((n,3,17),dtype=np.float64),
        conditional_exact_expected_reward=np.empty(n),conditional_exact_full_success_probability=np.empty(n),
        conditional_exact_execution_probability=np.empty(n))
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));ids=indices[sl]
        trace=core.rollout(networks,arrays['x_'+information][ids],live)
        probs,_=core.base.policy_distribution(trace['action_logits'])
        data['messages'][sl]=trace['messages'];data['action_probabilities'][sl]=probs
        data['action_indices'][sl]=np.argmax(probs,axis=-1)
        e,f,x=core.base.exact_statistics(probs,arrays['rewards'][ids])
        data['conditional_exact_expected_reward'][sl]=e
        data['conditional_exact_full_success_probability'][sl]=f
        data['conditional_exact_execution_probability'][sl]=x
    data.update(native(data['states'],data['action_indices']))
    np.savez_compressed(path,**data)
    targets=core.base.JOINT_ACTIONS[np.argmax(arrays['rewards'][indices],axis=1)]
    assert np.all(np.sum(arrays['rewards'][indices]==1,axis=1)==1)
    # Role includes actual stated partner, not merely a nonzero action.
    partner=np.full((3,17),-1,dtype=np.int8)
    for a in range(3):
        for i,action in enumerate(core.base.ACTIONS[a]):
            if i:partner[a,i]=core.base.AGENTS.index(action['partner'])
    actual_roles=partner[np.arange(3),data['action_indices']]
    target_roles=partner[np.arange(3),targets]
    actions,counts=np.unique(data['action_indices'],axis=0,return_counts=True)
    return dict(path=str(path),data_sha256=sha(path),worlds=n,information=information,live=live,
        reward_mean=float(data['greedy_reward'].mean()),full_success_rate=float((data['greedy_reward']==1).mean()),
        role_success_rate=float(np.all(actual_roles==target_roles,axis=1).mean()),
        physical_execution_rate=float(np.any(data['executed'],axis=1).mean()),
        expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
        raw_joint_action_counts=[dict(action_indices=a.tolist(),worlds=int(c)) for a,c in zip(actions,counts)],
        state_indices_sha256=array_sha(indices),reused_natural=False)


def evaluate_modes(networks,arrays,condition,indices,prefix):
    information,visibility=condition.split('_');live=visibility=='live'
    natural=evaluate(networks,arrays,information,live,indices,Path(str(prefix)+'_natural.npz'))
    if live:closed=evaluate(networks,arrays,information,False,indices,Path(str(prefix)+'_closed.npz'))
    else:
        closed=deepcopy(natural);closed['reused_natural']=True
    return dict(natural=natural,closed=closed)


def train_run(seed,condition,prepared,arrays,execution):
    output=execution/f'seed_{seed}_{condition}';output.mkdir(exist_ok=False)
    networks=core.make_networks(seed);initial=core.parameter_hash(networks)
    optimizer=core.base.make_adam(networks)
    batch_rng=np.random.default_rng(np.random.SeedSequence([seed,200]));message_rngs=core.make_message_rngs(seed)
    started=time.perf_counter();monitors=[]
    information,visibility=condition.split('_');live=visibility=='live'
    def checkpoint(step):
        ckpt=output/f'checkpoint_{step:04d}.npz'
        digest=core.save_checkpoint(ckpt,networks,optimizer,step,batch_rng,message_rngs)
        results={part:evaluate_modes(networks,arrays[part],condition,prepared['partitions'][part]['monitor_indices'],output/f'monitor_{step:04d}_{part}') for part in PARTS}
        row=dict(update=step,checkpoint_sha256=digest,monitor=results,elapsed_seconds=time.perf_counter()-started)
        monitors.append(row)
        with (output/'monitor.jsonl').open('a') as f:f.write(json_bytes(row).decode())
    checkpoint(0)
    with (output/'training.jsonl').open('x') as f:
        for update in range(1,6001):
            world_uniforms=batch_rng.random((256,3))
            ids=dataset.sample_indices(prepared['partitions']['train'],world_uniforms)
            uniforms=core.draw_uniforms(message_rngs,256)
            gradients,row=core.training_gradients(networks,arrays['train']['x_'+information][ids],arrays['train']['rewards'][ids],live,uniforms,update)
            norm,scale=core.base.adam_step(networks,gradients,optimizer,update)
            row.update(update=update,seed=seed,condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms),batch_indices_sha256=array_sha(ids),
                batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                sample_uniforms_sha256=array_sha(uniforms),gradient_norm=norm,gradient_clip_scale=scale)
            f.write(json_bytes(row).decode())
            if update in STEPS:f.flush();checkpoint(update)
    final={part:evaluate_modes(networks,arrays[part],condition,np.arange(len(arrays[part]['packed_states'])),output/f'final_{part}') for part in PARTS}
    result=dict(seed=seed,condition=condition,updates=6000,initial_parameter_sha256=initial,
        final_parameter_sha256=core.parameter_hash(networks),training_log_sha256=sha(output/'training.jsonl'),
        final_checkpoint_sha256=sha(output/'checkpoint_6000.npz'),monitor=monitors,final=final,
        elapsed_seconds=time.perf_counter()-started)
    write(output/'result.json',result)
    return result


def worker(payload):
    seed,prepared,execution=payload;execution=Path(execution)
    t=time.perf_counter()
    arrays={part:dataset.make_arrays(prepared['partitions'][part]) for part in PARTS}
    data_hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI','x_PL','x_LL')} for p,a in arrays.items()}
    write(execution/f'seed_{seed}_arrays.json',dict(array_hashes=data_hashes,build_seconds=time.perf_counter()-t))
    print(json.dumps(dict(stage='arrays_prepared',seed=seed,elapsed_seconds=time.perf_counter()-t)),flush=True)
    result=[]
    for condition in CONDITIONS:
        result.append(train_run(seed,condition,prepared,arrays,execution))
        print(json.dumps(dict(completed_seed=seed,condition=condition,elapsed_seconds=time.perf_counter()-t)),flush=True)
    assert data_hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI','x_PL','x_LL')} for p,a in arrays.items()}
    return result


def verify_pairing(execution,runs):
    for seed in SEEDS:
        selected=[v for v in runs if v['seed']==seed]
        assert len(selected)==6 and len({v['initial_parameter_sha256'] for v in selected})==1
        streams=[(execution/f'seed_{seed}_{c}'/'training.jsonl').open() for c in CONDITIONS]
        try:
            n=0
            for lines in zip_longest(*streams):
                assert all(x is not None for x in lines)
                rows=[json.loads(x) for x in lines];n+=1
                assert all(r['update']==n for r in rows)
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256','entropy_coefficient'):
                    assert len({r[key] for r in rows})==1
            assert n==6000
        finally:
            for s in streams:s.close()


def execute(out):
    out=Path(out).resolve();plan,prepared=verify(out)
    execution=out/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter()
    write(execution/'started.json',dict(started_at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    try:
        ctx=multiprocessing.get_context('spawn')
        with ctx.Pool(4) as pool: groups=pool.map(worker,[(s,prepared,str(execution)) for s in SEEDS])
        runs=[v for group in groups for v in group]
        assert [(v['seed'],v['condition']) for v in runs]==list(product(SEEDS,CONDITIONS))
        hashes=[read(execution/f'seed_{s}_arrays.json')['array_hashes'] for s in SEEDS]
        assert all(v==hashes[0] for v in hashes)
        verify_pairing(execution,runs);verify(out)
        result=dict(status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-started,
            completed_run_count=24,plan_sha256=sha(out/'plan.json'),budget=prepared['execution_budget'],runs=runs,
            pairing='same initial parameters, sampled worlds, uniforms and training schedule across all six arms',
            array_hashes=hashes[0])
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return dict(status='completed',runs=24,elapsed_seconds=result['elapsed_seconds'])
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','execute'));p.add_argument('--out',required=True);a=p.parse_args()
    answer=prepare(a.out) if a.command=='prepare' else execute(a.out) if a.command=='execute' else verify(a.out)[0]
    print(json.dumps(answer,ensure_ascii=False))
