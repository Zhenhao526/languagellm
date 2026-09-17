"""One fixed decoder experiment; prepare performs no network computation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from itertools import zip_longest
import argparse,json,multiprocessing,platform,shutil,time
import numpy as np
from research_program.triadic_partial_payoff_study import runner as old
from . import dataset,learner

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROBE_SEEDS=(55101,55102,55103);OLD_SEEDS=(53101,53102,53103,53104)
PAYOFFS=('a50','a10');PARTS=old.PARTS;STEPS=old.STEPS
sha,read,write,array_sha,json_bytes,now=old.sha,old.read,old.write,old.array_sha,old.json_bytes,old.now
CONFIG=dict(probe_seeds=list(PROBE_SEEDS),protocol_seeds=list(OLD_SEEDS),payoffs=list(PAYOFFS),
    updates=6000,batch_size=256,checkpoints=list(STEPS),dimensions=[252,64,64,3],heads=3,
    objective='mean_supervised_role_cross_entropy_over_worlds_and_actors; no entropy',
    primary='double_holdout_joint_role_accuracy_final_minus_initial; mean_probe_then_payoff_then_protocol_seed',
    auxiliary_absolute_bound=23/62,views=['Own','FI','Initial','Final'],
    world_sampling='original_three_uniform_need_layout_owner',checkpoint_selection='fixed_6000_no_early_stopping',
    worker_count=3,run_order='one_worker_per_probe_seed; Own,FI,then_each_old_seed_Initial,a50_Final,a10_Final',
    learning_rate=.001,global_gradient_clip=5.,dtype='float64',automatic_followon=False)

def jobs(probe_seed):
    rows=[dict(probe_seed=probe_seed,view=v,old_seed=None,payoff=None) for v in ('Own','FI')]
    for s in OLD_SEEDS:
        rows.append(dict(probe_seed=probe_seed,view='Initial',old_seed=s,payoff=None))
        rows.extend(dict(probe_seed=probe_seed,view='Final',old_seed=s,payoff=p) for p in PAYOFFS)
    for row in rows:row['id']=job_id(**row)
    return rows

def job_id(probe_seed,view,old_seed=None,payoff=None):
    result=f'probe_{probe_seed}_{view}'
    if old_seed is not None:result+=f'_protocol_{old_seed}'
    if payoff is not None:result+='_'+payoff
    return result

def budget(parts):
    worlds=sum(s['world_count'] for s in parts.values());monitor=sum(len(s['monitor_indices']) for s in parts.values())
    assert worlds==774144 and monitor==21504
    return dict(actual_decoder_runs=42,logical_decoder_runs=96,aliased_decoder_runs=54,
        updates=42*6000,training_world_samples=42*6000*256,training_forward_module_samples=42*6000*256*3,
        checkpoints=42*6,actual_monitor_files=42*6*4,actual_final_files=42*4,aliased_evaluation_records=54*7*4,
        actual_monitor_worlds=42*6*monitor,actual_final_worlds=42*worlds,
        decoder_evaluation_module_samples=(42*6*monitor+42*worlds)*3,
        initial_transcript_files=4*4,initial_sender_worlds=4*worlds,initial_sender_module_samples=4*worlds*6,
        old_parameter_loads=4,reused_final_transcript_files=8*4,reused_final_transcript_worlds=8*worlds)

def sources():
    names=('__init__.py','runner.py','dataset.py','learner.py','plan.md','design_review.md','literature_sources.json',
           'tests/test_dataset.py','tests/test_learner.py','tests/test_runner.py',
           'dataset_preflight_001.json','learner_preflight_001.json','main_preflight.json')
    paths=[HERE/n for n in names]+[Path(old.core.__file__),Path(old.core.base.__file__),Path(old.dataset.__file__)]
    return {str(p):sha(p) for p in paths}

def prepare(out):
    out=Path(out).resolve();assert not out.exists()
    source=dataset.source_manifest();prepared=dict(schema='frozen_transcript_role_decoder_v1',source=source,
        budget=budget(source['partitions']),jobs=[j for seed in PROBE_SEEDS for j in jobs(seed)])
    ss=sources();out.mkdir(parents=True)
    for path in ss:
        target=out/'source_snapshot'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write(out/'prepared.json',prepared)
    write(out/'plan.json',dict(status='prepared_without_inference_or_training',at=now(),config=CONFIG,sources=ss,
        inputs_sha256=source['source_sha256'],prepared_sha256=sha(out/'prepared.json'),
        runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')));verify(out)
    return dict(status='prepared_without_inference_or_training',plan_sha256=sha(out/'plan.json'),budget=prepared['budget'])

def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json');prepared=read(out/'prepared.json')
    assert sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256'] and plan['config']==CONFIG
    assert sha(out/'prepared.json')==plan['prepared_sha256'] and plan['sources']==sources()
    assert plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__)
    assert prepared['budget']==budget(prepared['source']['partitions'])
    assert prepared['jobs']==[j for seed in PROBE_SEEDS for j in jobs(seed)]
    assert plan['inputs_sha256']==prepared['source']['source_sha256']
    for path,digest in plan['inputs_sha256'].items():assert sha(path)==digest
    for path,digest in plan['sources'].items():assert sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest
    return plan,prepared

def make_initial(source,execution):
    """Only six frozen sender heads; old action heads are never evaluated."""
    directory=execution/'initial_transcripts';directory.mkdir(exist_ok=False);records=[]
    initial={row['seed']:row for row in source['initial_sources']}
    reference=read(Path(source['source_run'])/'execution/results.json')['array_hashes']
    # Build one domain at a time; avoid a large feature cache in the parent.
    nets={s:old.core.load_networks(initial[s]['checkpoint0']['path']) for s in OLD_SEEDS}
    for part,spec in source['partitions'].items():
        arrays=dataset.make_arrays(spec);n=len(arrays['packed_states']);ids=np.arange(n,dtype=np.int64)
        for key in ('packed_states','x_PL','x_FI'):assert array_sha(arrays[key])==reference[part][key]
        for seed in OLD_SEEDS:
            messages=np.empty((n,2,3,4),np.int8)
            for start in range(0,n,1024):
                result=dataset.initial_messages(nets[seed],arrays['x_PL'][start:start+1024])
                assert result['action_forward_samples']==0
                messages[start:start+1024]=result['messages']
            path=directory/f'protocol_{seed}_{part}.npz'
            np.savez_compressed(path,states=arrays['packed_states'],state_indices=ids,messages=messages)
            for policy in source['policies']:
                if policy['seed']!=seed:continue
                anchor=policy['monitor0'][part]
                assert sha(anchor['path'])==anchor['sha256']
                with np.load(anchor['path'],allow_pickle=False) as z:
                    monitor=np.asarray(spec['monitor_indices'])
                    assert np.array_equal(z['state_indices'],monitor)
                    assert np.array_equal(z['states'],arrays['packed_states'][monitor])
                    assert np.array_equal(z['messages'],messages[monitor])
            records.append(dict(seed=seed,partition=part,path=str(path),sha256=sha(path),worlds=n,
                initial_checkpoint_sha256=initial[seed]['checkpoint0']['sha256'],module_samples=n*6,
                monitor_anchor_payoffs=['a50','a10']))
        del arrays
    write(directory/'results.json',dict(status='completed',records=records,module_samples=sum(r['module_samples'] for r in records),old_parameter_loads=4))
    return records

def evaluate(nets,arrays,job,messages,indices,path):
    ids=np.asarray(indices)
    assert ids.ndim==1 and ids.dtype.kind in 'iu' and len(ids) and len(np.unique(ids))==len(ids)
    assert np.all((ids>=0)&(ids<len(arrays['packed_states']))) and not path.exists()
    ids=ids.astype(np.int64);n=len(ids)
    p=np.empty((n,3,3));nll=np.empty(n);view=job['view'] if job['view'] in ('Own','FI') else 'Transcript'
    for start in range(0,n,1024):
        sl=slice(start,min(start+1024,n));batch=ids[sl]
        x=dataset.build_inputs(arrays,batch,view,messages)
        pp,lp=learner.prediction_terms(nets,x);p[sl]=pp
        nll[sl]=-np.take_along_axis(lp,arrays['labels'][batch,:,None],axis=-1)[:,:,0].mean(1)
    target=arrays['labels'][ids];pred=p.argmax(-1).astype(np.int8);correct=pred==target;joint=correct.all(1)
    chosen=np.take_along_axis(p,target[:,:,None],axis=-1)[:,:,0]
    np.savez_compressed(path,states=arrays['packed_states'][ids],state_indices=ids,labels=target,
        probabilities=p,predictions=pred,negative_log_likelihood=nll,joint_correct=joint)
    return dict(path=str(path),sha256=sha(path),worlds=n,joint_accuracy=float(joint.mean()),
        indiv_accuracy=float(correct.mean()),cross_entropy=float(nll.mean()),
        expected_joint_correct_probability=float(np.prod(chosen,axis=1).mean()),
        per_actor_accuracy=correct.mean(0).tolist(),
        true_pair_strata={pair:dict(worlds=int((target[:,waiting]==0).sum()),
            joint_accuracy=float(joint[target[:,waiting]==0].mean())) for waiting,pair in ((2,'AB'),(1,'AC'),(0,'BC'))},
        predicted_joint_role_counts=[dict(labels=v.tolist(),worlds=int(c)) for v,c in zip(*np.unique(pred,axis=0,return_counts=True))],
        state_indices_sha256=array_sha(ids))

def train(job,source,arrays,messages,execution):
    directory=execution/job['id'];directory.mkdir(exist_ok=False);start=time.perf_counter()
    nets=learner.make_networks(job['probe_seed']);initial=learner.parameter_hash(nets);opt=learner.base.make_adam(nets)
    rng=np.random.default_rng(np.random.SeedSequence([job['probe_seed'],800]));monitor=[]
    def checkpoint(step):
        file=directory/f'checkpoint_{step:04d}.npz';digest=learner.save_checkpoint(file,nets,opt,step,rng)
        evaluations={part:evaluate(nets,arrays[part],job,messages.get(part),source['partitions'][part]['monitor_indices'],
            directory/f'monitor_{step:04d}_{part}.npz') for part in PARTS}
        row=dict(update=step,checkpoint_sha256=digest,evaluations=evaluations);monitor.append(row)
        with (directory/'monitor.jsonl').open('a') as f:f.write(json_bytes(row).decode())
    checkpoint(0)
    with (directory/'training.jsonl').open('x') as f:
        for step in range(1,6001):
            u=rng.random((256,3));ids=old.dataset.sample_indices(source['partitions']['train'],u)
            view=job['view'] if job['view'] in ('Own','FI') else 'Transcript'
            x=dataset.build_inputs(arrays['train'],ids,view,messages.get('train'));labels=arrays['train']['labels'][ids]
            grad,row=learner.training_gradients(nets,x,labels);norm,scale=learner.base.adam_step(nets,grad,opt,step)
            row.update(update=step,job_id=job['id'],world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),
                batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),labels_sha256=array_sha(labels),
                input_features_sha256=array_sha(x),gradient_norm=norm,gradient_clip_scale=scale)
            f.write(json_bytes(row).decode())
            if step in STEPS:f.flush();checkpoint(step)
    final={part:evaluate(nets,arrays[part],job,messages.get(part),np.arange(len(arrays[part]['packed_states'])),
        directory/f'final_{part}.npz') for part in PARTS}
    result=dict(job=job,initial_parameter_sha256=initial,final_parameter_sha256=learner.parameter_hash(nets),monitor=monitor,
        final=final,training_log_sha256=sha(directory/'training.jsonl'),elapsed_seconds=time.perf_counter()-start)
    write(directory/'result.json',result);return result

def worker(payload):
    probe_seed,source,initial,execution=payload;execution=Path(execution)
    arrays={p:dataset.make_arrays(source['partitions'][p]) for p in PARTS}
    hashes={p:{k:array_sha(v) for k,v in a.items()} for p,a in arrays.items()}
    write(execution/f'probe_{probe_seed}_arrays.json',hashes);rows=[]
    for job in jobs(probe_seed):
        messages={}
        if job['view']=='Initial':
            for record in initial:
                if record['seed']!=job['old_seed']:continue
                assert sha(record['path'])==record['sha256']
                with np.load(record['path'],allow_pickle=False) as z:
                    p=record['partition'];assert np.array_equal(z['states'],arrays[p]['packed_states'])
                    messages[p]=z['messages'].copy()
        elif job['view']=='Final':
            policy=next(p for p in source['policies'] if (p['seed'],p['payoff'])==(job['old_seed'],job['payoff']))
            for part,entry in policy['final'].items():
                messages[part]=dataset.load_final_messages(entry['path'],arrays[part]['packed_states'],expected_sha256=entry['sha256'])
        rows.append(train(job,source,arrays,messages,execution))
        print(json.dumps(dict(stage='decoder_completed',job=job['id'])),flush=True)
    assert hashes=={p:{k:array_sha(v) for k,v in a.items()} for p,a in arrays.items()}
    return rows

def logical_results(runs):
    by={r['job']['id']:r for r in runs};assert len(by)==42 and len(runs)==42
    logical=[]
    for old_seed in OLD_SEEDS:
        for payoff in PAYOFFS:
            for probe_seed in PROBE_SEEDS:
                for view in ('Own','FI','Initial','Final'):
                    actual=job_id(probe_seed,view,old_seed if view in ('Initial','Final') else None,payoff if view=='Final' else None)
                    assert actual in by
                    logical.append(dict(old_seed=old_seed,payoff=payoff,probe_seed=probe_seed,view=view,actual_job_id=actual))
    assert len(logical)==96
    return logical

def primary(runs):
    by={r['job']['id']:r for r in runs};logical_results(runs);values=[]
    for seed in OLD_SEEDS:
        cells=[]
        for payoff in PAYOFFS:
            probes=[]
            for ps in PROBE_SEEDS:
                get=lambda view:by[job_id(ps,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None)]['final']['new_needs_and_layouts']['joint_accuracy']
                v={view:get(view) for view in ('Own','FI','Initial','Final')}
                probes.append(dict(probe_seed=ps,accuracies=v,final_minus_initial=v['Final']-v['Initial'],
                    final_minus_own=v['Final']-v['Own'],final_minus_bound=v['Final']-23/62,initial_minus_bound=v['Initial']-23/62))
            cells.append(dict(payoff=payoff,probes=probes,mean_final_minus_initial=float(np.mean([p['final_minus_initial'] for p in probes]))))
        values.append(dict(old_seed=seed,payoffs=cells,mean_final_minus_initial=float(np.mean([c['mean_final_minus_initial'] for c in cells]))))
    return dict(metric='joint_role_accuracy_final_minus_initial',partition='new_needs_and_layouts',protocol_seed_blocks=values,
        mean_difference=float(np.mean([v['mean_final_minus_initial'] for v in values])))

def verify_pairing(execution,runs):
    for seed in PROBE_SEEDS:
        subset=[r for r in runs if r['job']['probe_seed']==seed]
        assert len(subset)==14 and len({r['initial_parameter_sha256'] for r in subset})==1
        streams=[(execution/r['job']['id']/'training.jsonl').open() for r in subset]
        try:
            count=0
            for lines in zip_longest(*streams):
                assert all(v is not None for v in lines);rows=[json.loads(v) for v in lines];count+=1
                assert all(r['update']==count for r in rows)
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','labels_sha256'):
                    assert len({r[key] for r in rows})==1
            assert count==6000
        finally:
            for stream in streams:stream.close()

def execute(out):
    out=Path(out).resolve();plan,prepared=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False);start=time.perf_counter()
    write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    try:
        initial=make_initial(prepared['source'],execution)
        ctx=multiprocessing.get_context('spawn')
        with ctx.Pool(3) as pool:groups=pool.map(worker,[(s,prepared['source'],initial,str(execution)) for s in PROBE_SEEDS])
        runs=[r for g in groups for r in g];assert [r['job'] for r in runs]==prepared['jobs']
        verify_pairing(execution,runs)
        hashes=[read(execution/f'probe_{s}_arrays.json') for s in PROBE_SEEDS];assert all(h==hashes[0] for h in hashes)
        verify(out)
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),elapsed_seconds=time.perf_counter()-start,
            budget=prepared['budget'],initial_transcripts=initial,array_hashes=hashes[0],runs=runs,
            logical_results=logical_results(runs),primary=primary(runs))
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return {k:result[k] for k in ('status','elapsed_seconds','primary')}
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),elapsed_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','verify','execute'));p.add_argument('--out',required=True);args=p.parse_args()
    result=globals()[args.command](args.out)
    print(json.dumps(result if args.command!='verify' else dict(status='verified'),ensure_ascii=False))
