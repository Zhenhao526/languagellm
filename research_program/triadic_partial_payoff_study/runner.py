"""Fresh paired policies; only utility for exactly one satisfied need changes."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
from itertools import product,zip_longest
from copy import deepcopy
import argparse,json,platform,shutil,time,multiprocessing
import numpy as np
from research_program.triadic_action_dependency_study import runner as previous,dataset,metrics as native_metrics
from . import utility
core=previous.core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
SEEDS=(53101,53102,53103,53104)
PAYOFFS=(('a50',.5),('a10',.1))
CONDITIONS=('FI_silent','PL_silent','PL_live')
CELLS=tuple((name,alpha,c) for name,alpha in PAYOFFS for c in CONDITIONS)
PARTS=previous.PARTS;STEPS=previous.STEPS
CONFIG=deepcopy(previous.CONFIG)
CONFIG.update(seeds=list(SEEDS),conditions=list(CONDITIONS),payoffs=[dict(name=n,partial_utility=a) for n,a in PAYOFFS],
    primary='6000_double_holdout_world_role_success_(a10_PL_live-a10_PL_silent)-(a50_PL_live-a50_PL_silent)',
    secondary_task_contrast='same_DiD_full_success_rate',experiment='partial_completion_payoff_with_fixed_worlds_and_observations',
    training_objective='mean_log_exact_expected_utility_plus_original_action_entropy; sender_two_trajectory_LOO',
    native_reward='satisfied_needs/2_always_0_0.5_1',training_utility='none0_exactly_one_alpha_both1',
    information='FI_all_needs_or_PL_private_own_need; public_layout_both',
    run_order='four_spawn_workers_one_seed_each_payoff_then_condition',automatic_followon_experiment=False)
sha,read,write,json_hash,json_bytes,array_sha,now=previous.sha,previous.read,previous.write,previous.json_hash,previous.json_bytes,previous.array_sha,previous.now

def name(seed,payoff,condition):return f'seed_{seed}_{payoff}_{condition}'

def prepared():
    old=read(ORIGINAL/'plan.json');freeze=read(ORIGINAL/'freeze.json');static=read(ORIGINAL/'prepared.json')
    assert sha(ORIGINAL/'plan.json')==freeze['plan_sha256']=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba'
    assert sha(ORIGINAL/'prepared.json')==freeze['prepared_sha256']==old['prepared_sha256']
    parts=static['partitions'];worlds=sum(s['world_count'] for s in parts.values());mon=sum(len(s['monitor_indices']) for s in parts.values())
    assert worlds==774144 and mon==21504
    return dict(schema='triadic_partial_payoff_v1',partitions=parts,source_prepared_sha256=sha(ORIGINAL/'prepared.json'),
        budget=dict(runs=24,training_updates=144000,training_world_samples=144000*256,message_trajectories=144000*256*2,
        categorical_symbol_samples=144000*256*2*24,checkpoints=144,actual_monitor_files=32*6*4,actual_final_files=32*4,
        silent_closed_aliases=16*7*4,actual_monitor_worlds=32*6*mon,actual_final_worlds=32*worlds,
        evaluation_network_samples=(32*6*mon+32*worlds)*9,
        training_forward_module_samples=144000*256*2*9),new_initializations=4)

def sources():
    old=read(ORIGINAL/'plan.json');out={}
    for rel,digest in old['sources'].items():
        path=ROOT/rel;assert sha(path)==digest;out[str(path)]=digest
    for f in ('__init__.py','runner.py','utility.py','plan.md','tests/test_utility.py','tests/test_runner.py','main_preflight.json',
              'utility_preflight.json','test_runner_receipt_001.json','test_runner_receipt_002.json'):
        path=HERE/f;out[str(path)]=sha(path)
    for f in ('支付操纵与伙伴表达的静态审查.md','enumerate_payoff.py','payoff_static_001.json','native_crosscheck_001.json',
              '机制与近邻审查.md','literature_sources.json'):
        path=HERE.parent/'triadic_role_expression_study/design_audit'/f;out[str(path)]=sha(path)
    path=Path(native_metrics.__file__);out[str(path)]=sha(path)
    return out

def prepare(out):
    out=Path(out).resolve();assert not out.exists();static=prepared();ss=sources();out.mkdir(parents=True)
    for path in ss:
        dest=out/'source_snapshot'/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
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
    a=dataset.make_arrays(spec,information='FI')
    a['x_PL']=a['x_FI'].copy();a['x_PL'][:,:,53]=0
    for viewer in range(3):
        for speaker in range(3):
            if viewer!=speaker:a['x_PL'][:,viewer,7*speaker:7*speaker+7]=0
    del a['states']
    return a

def evaluate(networks,arrays,information,live,alpha,indices,path):
    raw=np.asarray(indices);assert raw.ndim==1 and np.issubdtype(raw.dtype,np.integer)
    ids=raw.astype(np.int64);n=len(ids)
    assert n and len(np.unique(ids))==n and np.all((ids>=0)&(ids<len(arrays['packed_states']))) and not path.exists()
    assert information in ('FI','PL') and isinstance(live,(bool,np.bool_)) and alpha in (.5,.1)
    data=dict(states=arrays['packed_states'][ids],state_indices=ids,messages=np.empty((n,2,3,4),np.int8),
        action_indices=np.empty((n,3),np.int16),action_probabilities=np.empty((n,3,17)),
        conditional_exact_expected_reward=np.empty(n),conditional_exact_full_success_probability=np.empty(n),
        conditional_exact_execution_probability=np.empty(n),conditional_exact_expected_utility=np.empty(n),
        conditional_full_posterior_mass=np.empty(n))
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));ix=ids[sl];trace=core.rollout(networks,arrays['x_'+information][ix],live)
        terms=utility.objective_terms(trace['action_logits'],arrays['rewards'][ix],alpha);p=terms['probabilities']
        native,full,physical=core.base.exact_statistics(p,arrays['rewards'][ix])
        data['messages'][sl]=trace['messages'];data['action_probabilities'][sl]=p;data['action_indices'][sl]=p.argmax(-1)
        data['conditional_exact_expected_reward'][sl]=native;data['conditional_exact_full_success_probability'][sl]=full
        data['conditional_exact_execution_probability'][sl]=physical;data['conditional_exact_expected_utility'][sl]=terms['J']
        data['conditional_full_posterior_mass'][sl]=terms['full_success_posterior_mass']
    data.update(previous.native(data['states'],data['action_indices']))
    data['greedy_utility']=np.where(data['greedy_reward']==.5,alpha,data['greedy_reward'])
    np.savez_compressed(path,**data)
    truth=core.base.JOINT_ACTIONS[np.argmax(arrays['rewards'][ids],axis=1)]
    partner=native_metrics.PARTNER;roles=partner[np.arange(3),data['action_indices']];correct=partner[np.arange(3),truth]
    acts,counts=np.unique(data['action_indices'],axis=0,return_counts=True)
    rp,rc=np.unique(roles,axis=0,return_counts=True)
    return dict(path=str(path),data_sha256=sha(path),worlds=n,information=information,live=live,partial_utility=alpha,
        reward_mean=float(data['greedy_reward'].mean()),utility_mean=float(data['greedy_utility'].mean()),
        full_success_rate=float((data['greedy_reward']==1).mean()),role_success_rate=float(np.all(roles==correct,axis=1).mean()),
        physical_execution_rate=float(np.any(data['executed'],axis=1).mean()),
        expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
        expected_utility_given_greedy_messages=float(data['conditional_exact_expected_utility'].mean()),
        full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(data['conditional_full_posterior_mass'].mean()),
        raw_joint_action_counts=[dict(action_indices=a.tolist(),worlds=int(c)) for a,c in zip(acts,counts)],
        raw_joint_role_counts=[dict(partner_indices=a.tolist(),worlds=int(c)) for a,c in zip(rp,rc)],
        state_indices_sha256=array_sha(ids),reused_natural=False)

def evaluate_modes(networks,arrays,condition,alpha,indices,prefix):
    information,visibility=condition.split('_');live=visibility=='live'
    natural=evaluate(networks,arrays,information,live,alpha,indices,Path(str(prefix)+'_natural.npz'))
    if live:closed=evaluate(networks,arrays,information,False,alpha,indices,Path(str(prefix)+'_closed.npz'))
    else:closed=deepcopy(natural);closed['reused_natural']=True
    return dict(natural=natural,closed=closed)

def train_run(seed,payoff,alpha,condition,static,arrays,execution):
    directory=execution/name(seed,payoff,condition);directory.mkdir(exist_ok=False)
    nets=core.make_networks(seed);initial=core.parameter_hash(nets);optimizer=core.base.make_adam(nets)
    wr=np.random.default_rng(np.random.SeedSequence([seed,200]));mr=core.make_message_rngs(seed)
    information,visibility=condition.split('_');live=visibility=='live';monitors=[];start=time.perf_counter()
    def checkpoint(step):
        ckpt=directory/f'checkpoint_{step:04d}.npz';digest=core.save_checkpoint(ckpt,nets,optimizer,step,wr,mr)
        evaluations={part:evaluate_modes(nets,arrays[part],condition,alpha,static['partitions'][part]['monitor_indices'],directory/f'monitor_{step:04d}_{part}') for part in PARTS}
        rec=dict(update=step,checkpoint_sha256=digest,monitor=evaluations,elapsed_seconds=time.perf_counter()-start);monitors.append(rec)
        with (directory/'monitor.jsonl').open('a') as f:f.write(json_bytes(rec).decode())
    checkpoint(0)
    with (directory/'training.jsonl').open('x') as f:
        for update in range(1,6001):
            u=wr.random((256,3));ids=dataset.sample_indices(static['partitions']['train'],u);mu=core.draw_uniforms(mr,256)
            gradients,row=utility.training_gradients(nets,arrays['train']['x_'+information][ids],arrays['train']['rewards'][ids],live,mu,update,alpha)
            norm,scale=core.base.adam_step(nets,gradients,optimizer,update)
            row.update(update=update,seed=seed,payoff=payoff,partial_utility=alpha,condition=condition,world_uniforms_sha256=array_sha(u),
                batch_indices_sha256=array_sha(ids),batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                sample_uniforms_sha256=array_sha(mu),gradient_norm=norm,gradient_clip_scale=scale)
            f.write(json_bytes(row).decode())
            if update in STEPS:f.flush();checkpoint(update)
    final={part:evaluate_modes(nets,arrays[part],condition,alpha,np.arange(len(arrays[part]['packed_states'])),directory/f'final_{part}') for part in PARTS}
    result=dict(seed=seed,payoff=payoff,partial_utility=alpha,condition=condition,updates=6000,initial_parameter_sha256=initial,
        final_parameter_sha256=core.parameter_hash(nets),final_checkpoint_sha256=sha(directory/'checkpoint_6000.npz'),
        training_log_sha256=sha(directory/'training.jsonl'),monitor=monitors,final=final,elapsed_seconds=time.perf_counter()-start)
    write(directory/'result.json',result);return result

def worker(payload):
    seed,static,execution=payload;execution=Path(execution);start=time.perf_counter()
    arrays={part:make_arrays(static['partitions'][part]) for part in PARTS}
    hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI','x_PL')} for p,a in arrays.items()}
    write(execution/f'seed_{seed}_arrays.json',dict(array_hashes=hashes,build_seconds=time.perf_counter()-start))
    print(json.dumps(dict(stage='arrays_prepared',seed=seed,elapsed_seconds=time.perf_counter()-start)),flush=True)
    result=[]
    for payoff,alpha,condition in CELLS:
        result.append(train_run(seed,payoff,alpha,condition,static,arrays,execution))
        print(json.dumps(dict(stage='run_completed',seed=seed,payoff=payoff,condition=condition,elapsed_seconds=time.perf_counter()-start)),flush=True)
    assert hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_FI','x_PL')} for p,a in arrays.items()}
    return result

def verify_pairing(execution,runs):
    for seed in SEEDS:
        cells=[r for r in runs if r['seed']==seed];assert len(cells)==6 and len({r['initial_parameter_sha256'] for r in cells})==1
        streams=[(execution/name(seed,p,c)/'training.jsonl').open() for p,_,c in CELLS]
        try:
            count=0
            for lines in zip_longest(*streams):
                assert all(line is not None for line in lines);rows=[json.loads(line) for line in lines];count+=1
                assert all(r['update']==count for r in rows)
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256','entropy_coefficient'):assert len({r[key] for r in rows})==1
            assert count==6000
        finally:
            for stream in streams:stream.close()

def primary(runs):
    assert len(runs)==24
    by={(r['seed'],r['payoff'],r['condition']):r for r in runs};assert len(by)==24
    values=[]
    for s in SEEDS:
        cells={p+'_'+c:by[s,p,c]['final']['new_needs_and_layouts']['natural'] for p,_,c in CELLS}
        diff={}
        for metric in ('role_success_rate','full_success_rate','reward_mean'):
            diff[metric]=(cells['a10_PL_live'][metric]-cells['a10_PL_silent'][metric])-(cells['a50_PL_live'][metric]-cells['a50_PL_silent'][metric])
        values.append(dict(seed=s,contrasts=diff,cells={key:{m:v[m] for m in ('role_success_rate','full_success_rate','reward_mean','utility_mean')} for key,v in cells.items()}))
    return dict(metric='role_success_rate',partition='new_needs_and_layouts',paired_seeds=values,
        mean_difference=float(np.mean([r['contrasts']['role_success_rate'] for r in values])))

def execute(out):
    out=Path(out).resolve();plan,static=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False);start=time.perf_counter()
    write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    try:
        ctx=multiprocessing.get_context('spawn')
        with ctx.Pool(4) as pool:groups=pool.map(worker,[(s,static,str(execution)) for s in SEEDS])
        runs=[r for g in groups for r in g];assert [(r['seed'],r['payoff'],r['condition']) for r in runs]==[(s,p,c) for s in SEEDS for p,_,c in CELLS]
        verify_pairing(execution,runs);hashes=[read(execution/f'seed_{s}_arrays.json')['array_hashes'] for s in SEEDS];assert all(v==hashes[0] for v in hashes)
        verify(out)
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),elapsed_seconds=time.perf_counter()-start,
            budget=static['budget'],runs=runs,array_hashes=hashes[0],primary=primary(runs))
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return dict(status='completed',elapsed_seconds=result['elapsed_seconds'],primary=result['primary'])
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),elapsed_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','verify','execute'));p.add_argument('--out',required=True);a=p.parse_args();r=globals()[a.command](a.out)
    print(json.dumps(r if a.command!='verify' else dict(status='verified'),ensure_ascii=False))
