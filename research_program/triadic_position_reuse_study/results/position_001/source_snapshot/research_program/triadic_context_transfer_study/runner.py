"""Run one bounded frozen-policy diagnostic, never train or retry a policy."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import multiprocessing
import platform
import shutil
import time
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset as task_data
from research_program.triadic_action_dependency_study import runner as task_run
from research_program.triadic_context_transfer_study import channel_intervention as ch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE.parent/'triadic_action_dependency_study/results/context_001'
DATA=HERE/'dataset_001'
SEEDS=(51101,51102,51103,51104)
CONDITIONS=('PL_silent','PL_live','LL_silent','LL_live')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
BATCH=1024
MAIN_RESULT_SHA='021cef0cbbf18a1532100d3c6058f04520ffa66a83a18aa10b8b9c53e2e6d2b0'
CORE_SHA='5299c99bb92f3e18f8fae969347084c631a66f78e635d2abc4609c476cb9ae60'
CONFIG=dict(seeds=list(SEEDS),conditions=list(CONDITIONS),partitions=list(PARTS),checkpoint=6000,
    batch_size=BATCH,worker_count=4,windows=[0,1],training_updates=0,new_initializations=0,
    primary='double_holdout_eligible_raw_counterfactual_apt_gain_PLlive_minus_LLlive',
    remote_modes=['remote_same_both','remote_opposite_both'],controls=['sham_both','local_opposite_both'],
    donor_support='all_other_layouts_in_same_partition_owner_unchanged',
    primary_support='both_endpoint_target_materials_change_site',
    weighting='axis_equal_then_six_sender_listener_then_cases_then_recipient_background_then_eligible_donor_then_two_directions',
    silent='logical_alias_of_saved_natural_after_exact_route_checks_no_neural_forward',
    actual_intervention_files=1536,silent_alias_records=1536,reused_natural_files=64,
    new_intervention_worlds=184135680,new_network_samples=1104814080,
    reused_natural_worlds=12386304,new_natural_forward_worlds=0,
    execution_order='four_seed_workers_each_condition_then_partition_then_mode_shift_direction',
    automatic_followon=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def modes(spec):
    yield 'sham_both',-1
    yield 'local_opposite_both',-1
    for shift in range(len(spec['donor_endpoint_indices'])):
        yield 'remote_same_both',shift
        yield 'remote_opposite_both',shift
def donor_indices(spec,mode,shift,direction):
    assert direction in (0,1)
    if mode=='sham_both':assert shift==-1;return spec['endpoint_indices'][:,direction]
    if mode=='local_opposite_both':assert shift==-1;return spec['endpoint_indices'][:,1-direction]
    assert mode in ('remote_same_both','remote_opposite_both') and shift>=0
    return spec['donor_endpoint_indices'][shift,:,direction if mode=='remote_same_both' else 1-direction]

def collect_inputs():
    task_run.verify(SOURCE)
    assert sha(SOURCE/'execution/results.json')==MAIN_RESULT_SHA
    result=read(SOURCE/'execution/results.json');assert result['status']=='completed' and result['completed_run_count']==24
    assert read(SOURCE/'audit_execution_001/verification.json')['status']=='passed'
    manifest=read(DATA/'manifest.json')
    paths=[SOURCE/'plan.json',SOURCE/'prepared.json',SOURCE/'freeze.json',SOURCE/'execution/results.json',
           SOURCE/'audit_execution_001/verification.json',DATA/'manifest.json']
    for name,digest in manifest['source_sha256'].items():
        path=Path(name);assert sha(path)==digest;paths.append(path)
    for name,record in manifest['outputs'].items():
        path=DATA/name;assert sha(path)==record['sha256'];paths.append(path)
    policies=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            r=next(v for v in result['runs'] if (v['seed'],v['condition'])==(seed,condition))
            directory=SOURCE/'execution'/f'seed_{seed}_{condition}'
            checkpoint=directory/'checkpoint_6000.npz';assert sha(checkpoint)==r['final_checkpoint_sha256'];paths.append(checkpoint)
            endpoints={}
            for part in PARTS:
                e=r['final'][part]['natural'];path=Path(e['path']);assert sha(path)==e['data_sha256'];paths.append(path)
                endpoints[part]=str(path)
            policies.append(dict(seed=seed,condition=condition,checkpoint=str(checkpoint),endpoints=endpoints))
    return policies,{str(p):sha(p) for p in paths}

def sources():
    assert sha(core.__file__)==CORE_SHA
    files=[HERE/name for name in ('__init__.py','runner.py','channel_intervention.py','dataset.py','metrics.py','plan.md',
        'tests/test_runner.py','tests/test_channel_intervention.py','tests/test_dataset.py','tests/test_metrics.py',
        'main_preflight.json','静态识别审查.md')]
    # Imported production sources are covered individually, not only by an old receipt.
    files += [ROOT/name for name in read(SOURCE/'plan.json')['sources']]
    return {str(p):sha(p) for p in files}

def prepare(out):
    out=Path(out).resolve();assert not out.exists()
    policies,inputs=collect_inputs();ss=sources();out.mkdir(parents=True)
    for path in ss:
        target=out/'source_snapshot'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    plan=dict(status='prepared_without_new_forward',prepared_at=now(),config=CONFIG,
        runtime=dict(python=platform.python_version(),numpy=np.__version__),sources_sha256=ss,
        inputs_sha256=inputs,policies=policies,dataset_directory=str(DATA),source_directory=str(SOURCE))
    write(out/'plan.json',plan);write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')))
    return dict(status=plan['status'],plan_sha256=sha(out/'plan.json'),config=CONFIG)

def verify(out):
    out=Path(out).resolve();p=read(out/'plan.json');assert sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256']
    assert p['config']==CONFIG and p['runtime']==dict(python=platform.python_version(),numpy=np.__version__)
    assert sources()==p['sources_sha256']
    for path,h in p['sources_sha256'].items():assert sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==h
    for path,h in p['inputs_sha256'].items():assert sha(path)==h
    return p

def execute_cell(networks,x,pool,spec,mode,shift,direction,live,path):
    ids=spec['endpoint_indices'][:,direction];cfids=spec['endpoint_indices'][:,1-direction]
    di=donor_indices(spec,mode,shift,direction);sender=spec['sender'];n=len(ids)
    data=None
    if live:
        data=dict(dataset_rows=np.arange(n,dtype=np.int64),recipient_indices=ids,counterfactual_recipient_indices=cfids,
            donor_indices=di,donor_packets=np.empty((n,2,4),dtype=np.int8),
            messages=np.empty((n,2,3,4),dtype=np.int8),action_indices=np.empty((n,3),dtype=np.int16),
            action_probabilities=np.empty((n,3,17),dtype=np.float64),
            action_input_equals_donor=np.empty((n,3,2),dtype=bool))
    batches=[];identity_error=0.
    for start in range(0,n,BATCH):
        stop=min(start+BATCH,n);sl=slice(start,stop);r=ids[sl];d=di[sl];s=sender[sl];rows=np.arange(stop-start)
        generated=pool['messages'][r];packets=pool['messages'][d,:,s,:]
        if not live:
            # A foreign replacement has no place in an invisible route. Same deterministic
            # downstream functions consequently have the old outputs; no neural call here.
            route1=ch.replace_outward(generated[:,0],s,packets[:,0],False)
            route2=ch.replace_outward(generated[:,1],s,packets[:,1],False)
            assert np.array_equal(route1,core.routed_window(generated[:,0],False))
            assert np.array_equal(route2,core.routed_window(generated[:,1],False))
            action_inputs=np.concatenate((x[r],route1,route2),axis=-1)
            assert np.array_equal(action_inputs,ch.natural_action_inputs(x[r],generated,False))
            trace=dict(first_routes=route1,second_routes=route2,action_inputs=action_inputs)
        else:
            trace=ch.intervene(networks,x[r],generated,s,packets,True)
            for key in ('messages','action_indices','action_probabilities'):data[key][sl]=trace[key]
            data['donor_packets'][sl]=packets
            assert np.array_equal(trace['messages'][rows,1,s],generated[rows,1,s])
            if mode=='sham_both':
                assert np.array_equal(trace['messages'],generated)
                assert np.array_equal(trace['action_indices'],pool['action_indices'][r])
                err=float(np.max(np.abs(trace['action_probabilities']-pool['action_probabilities'][r])))
                assert err<=2e-12;identity_error=max(identity_error,err)
            if mode=='local_opposite_both':
                expected=ch.natural_action_inputs(x[d],pool['messages'][d],True)
                foreign=np.arange(3)[None,:]!=s[:,None]
                assert np.array_equal(trace['action_inputs'][foreign],expected[foreign])
                err=float(np.max(np.abs(trace['action_probabilities'][foreign]-pool['action_probabilities'][d][foreign])))
                assert err<=2e-12;identity_error=max(identity_error,err)
            for endpoint in (0,1):
                donor_ids=spec['donor_endpoint_indices'][shift,sl,endpoint] if shift>=0 else spec['endpoint_indices'][sl,endpoint]
                expected=ch.natural_action_inputs(x[donor_ids],pool['messages'][donor_ids],True)
                data['action_input_equals_donor'][sl,:,endpoint]=np.all(trace['action_inputs']==expected,axis=-1)
        batches.append(dict(start=start,stop=stop,first_routes_sha256=core.array_sha(trace['first_routes']),
            second_routes_sha256=core.array_sha(trace['second_routes']),action_inputs_sha256=core.array_sha(trace['action_inputs']),
            donor_packets_sha256=core.array_sha(packets)))
    record=dict(mode=mode,shift_index=shift,direction=direction,worlds=n,
        recipient_indices_sha256=core.array_sha(ids),counterfactual_recipient_indices_sha256=core.array_sha(cfids),
        donor_indices_sha256=core.array_sha(di),batches=batches,max_identity_error=identity_error,
        new_forward_worlds=n if live else 0,new_network_samples=6*n if live else 0,
        is_silent_alias=not live)
    if live:
        data.update(ch.settle(pool['states'][ids],data['action_indices']))
        cf=ch.settle(pool['states'][cfids],data['action_indices'])
        data['counterfactual_reward']=cf['greedy_reward'];data['counterfactual_satisfied']=cf['satisfied']
        assert np.array_equal(cf['executed'],data['executed'])
        assert np.array_equal(cf['greedy_reward']==1,np.all(data['action_indices']==spec['correct_actions'][:,1-direction],axis=1))
        assert not path.exists();np.savez_compressed(path,**data)
        record.update(path=str(path),data_sha256=sha(path))
    else:
        record.update(path=None,data_sha256=None,alias_recipe='gather saved natural by endpoint_indices[:,direction]; outgoing replacements invisible; actual inputs checked for every row')
    return record

def worker(args):
    seed,policies,execution,specs,task_specs=args;execution=Path(execution);started=time.perf_counter()
    features={};hashes={}
    for part in PARTS:
        states=task_data.pack_states(task_specs[part])
        features[part]={info:ch.observations(states,info) for info in ('PL','LL')}
        hashes[part]={info:core.array_sha(x) for info,x in features[part].items()}
    write(execution/f'seed_{seed}_features.json',dict(seed=seed,hashes=hashes,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(seed=seed,stage='features_ready',elapsed_seconds=time.perf_counter()-started)),flush=True)
    records=[];natural=[];loads=0
    for policy in policies:
        condition=policy['condition'];info,visibility=condition.split('_');live=visibility=='live'
        networks=core.load_networks(policy['checkpoint']) if live else None
        loads+=int(live);directory=execution/f'seed_{seed}_{condition}';directory.mkdir()
        for part in PARTS:
            pool=load_npz(policy['endpoints'][part]);x=features[part][info];spec=specs[part]
            assert np.array_equal(pool['states'],task_data.pack_states(task_specs[part]))
            assert np.array_equal(pool['state_indices'],np.arange(len(x)))
            natural.append(dict(seed=seed,condition=condition,partition=part,path=policy['endpoints'][part],
                data_sha256=sha(policy['endpoints'][part]),worlds=len(x),new_forward_worlds=0))
            for mode,shift in modes(spec):
                for direction in (0,1):
                    name=f'{part}_{mode}_shift{shift:02d}_direction{direction}'
                    record=execute_cell(networks,x,pool,spec,mode,shift,direction,live,directory/(name+'.npz'))
                    record.update(seed=seed,condition=condition,partition=part,natural_source=policy['endpoints'][part])
                    write(directory/(name+'.json'),record);records.append(record)
                if mode=='remote_opposite_both':
                    print(json.dumps(dict(seed=seed,condition=condition,partition=part,completed_shift=shift,elapsed_seconds=time.perf_counter()-started)),flush=True)
            del pool
    assert hashes=={part:{info:core.array_sha(x) for info,x in views.items()} for part,views in features.items()}
    result=dict(seed=seed,records=records,natural_references=natural,parameter_loads=loads,feature_hashes=hashes,elapsed_seconds=time.perf_counter()-started)
    write(execution/f'seed_{seed}_result.json',result);return result

def execute(out):
    out=Path(out).resolve();plan=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter();write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    try:
        specs={part:load_npz(DATA/(part+'.npz')) for part in PARTS}
        task_specs=read(SOURCE/'prepared.json')['partitions']
        tasks=[(seed,[p for p in plan['policies'] if p['seed']==seed],str(execution),specs,task_specs) for seed in SEEDS]
        with multiprocessing.get_context('spawn').Pool(4) as pool:groups=pool.map(worker,tasks)
        records=[r for g in groups for r in g['records']];natural=[r for g in groups for r in g['natural_references']]
        assert len(records)==3072 and len(natural)==64
        actual=[r for r in records if not r['is_silent_alias']]
        assert len(actual)==1536 and sum(r['new_forward_worlds'] for r in records)==CONFIG['new_intervention_worlds']
        assert sum(r['new_network_samples'] for r in records)==CONFIG['new_network_samples']
        assert sum(r['worlds'] for r in natural)==CONFIG['reused_natural_worlds']
        assert sum(g['parameter_loads'] for g in groups)==8
        assert all(g['feature_hashes']==groups[0]['feature_hashes'] for g in groups)
        verify(out)
        result=dict(status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-started,
            plan_sha256=sha(out/'plan.json'),records=records,natural_references=natural,
            actual_intervention_files=len(actual),silent_aliases=len(records)-len(actual),
            new_intervention_worlds=sum(r['new_forward_worlds'] for r in records),
            new_network_samples=sum(r['new_network_samples'] for r in records),
            parameter_loads=8,training_updates=0,feature_hashes=groups[0]['feature_hashes'])
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return {k:v for k,v in result.items() if k not in ('records','natural_references','feature_hashes')}
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','execute'));parser.add_argument('--out',required=True);args=parser.parse_args()
    result=prepare(args.out) if args.command=='prepare' else execute(args.out) if args.command=='execute' else verify(args.out)
    print(json.dumps(result,ensure_ascii=False))
