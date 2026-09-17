"""Frozen six-checkpoint formation diagnostic and cross-time compatibility."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import json
import platform
import shutil
import time
import numpy as np
from research_program.triadic_position_reuse_study import runner as pr, discovery, metrics as pm
from research_program.triadic_action_dependency_study import dataset as td
from . import dataset, channel as ch, metrics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'triadic_position_reuse_study/results/position_001'
STEPS=metrics.STEPS
SEEDS=(51101,51102,51103,51104)
CONDITIONS=('PL_silent','PL_live','LL_silent','LL_live')
BATCH=1024
OLD_RESULT_SHA='c9f5ce31c8329db29c7afb2ca124dba006a257beb65d781c76b612ce2914391c'
OLD_AUDIT_SHA='b5f0b8f2400c8c9208d89d0232d920667956d67a2939d5341dd9954ea70f1e6d'
CONFIG=dict(steps=list(STEPS),seeds=list(SEEDS),conditions=list(CONDITIONS),batch_size=BATCH,
    primary='four_paired_seed_PLlive_minus_LLlive_normalized_trapezoid_area_of_time_local_selected_S',
    same_time_discovery='complete_train_at_each_checkpoint_exact_Fraction',
    retrospective='fixed6000_selected_positions_applied_at_all_times_explicit_future_information',
    cross_time='complete6x6_whole_same_opposite_with_receiver_row_same_time_reference',
    positions=list(range(8)),arms=['same','opposite'],shams=[0,4],directions=[0,1],
    validation_rows=144,validation_pool_worlds=554,train_worlds=419904,
    new_train_message_worlds=33592320,new_train_message_module_samples=201553920,
    new_validation_natural_worlds=44320,new_validation_natural_module_samples=398880,
    new_position_worlds=207360,new_position_module_samples=933120,
    new_cross_time_worlds=161280,new_cross_time_module_samples=967680,
    total_new_module_samples=203853600,parameter_loads=88,
    train_records=96,natural_records=96,position_records=3456,cross_time_records=2304,
    new_position_npz=1440,new_cross_time_npz=1120,
    reused_position_records=576,reused_cross_time_records=64,
    training_updates=0,new_initializations=0,execution_order='serial_policy_all_checkpoint_natural_then_positions_then_cross_time',
    positive_score_filter=False,force_distinct_positions=False,onset_threshold=None,automatic_retry=False)
sha,read,write,now,load_npz,require=pr.sha,pr.read,pr.write,pr.now,pr.load_npz,pr.require

def tag(policy):return pr.tag(policy)

def sources():
    old=read(PRIOR/'plan.json')
    files=[HERE/name for name in ('__init__.py','runner.py','dataset.py','channel.py','metrics.py','plan.md',
        'tests/test_runner.py','tests/test_dataset.py','tests/test_channel.py','tests/test_metrics.py','main_preflight.json')]
    for path,digest in old['sources_sha256'].items():
        require(sha(path)==digest,'Prior production source changed: '+path);files.append(Path(path))
    return {str(p):sha(p) for p in files}

def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Refuse preparation overwrite')
    pr.verify(PRIOR);ss=sources()
    require(sha(PRIOR/'execution/results.json')==OLD_RESULT_SHA,'Prior result identity')
    require(sha(PRIOR/'audit_execution_001/verification.json')==OLD_AUDIT_SHA,'Prior audit identity')
    old=read(PRIOR/'execution/results.json');old_plan=read(PRIOR/'plan.json')
    require(old['status']=='completed','Prior run incomplete')
    out.mkdir(parents=True)
    write(out/'frozen_spec.json',dict(at=now(),config=CONFIG,sources_sha256=ss,new_checkpoint_outputs_read=False))
    for path in ss:
        dest=out/'source_snapshot'/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    dataset.prepare(out/'dataset')
    manifest=read(out/'dataset/manifest.json');checkpoints=read(out/'dataset/checkpoints.json')
    inputs=dict(manifest['source_sha256'])
    inputs.update({str(PRIOR/'execution/results.json'):OLD_RESULT_SHA,str(PRIOR/'audit_execution_001/verification.json'):OLD_AUDIT_SHA})
    policies=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            previous=next(p for p in old_plan['policies'] if p['seed']==seed and p['condition']==condition)
            ps=next(p for p in old['policy_summaries'] if p['seed']==seed and p['condition']==condition)
            pos=[r for r in old['records'] if r['seed']==seed and r['condition']==condition]
            whole=[r for r in old['references'] if r['seed']==seed and r['condition']==condition and r['kind']=='whole']
            require(len(pos)==36 and len(whole)==4,'Missing endpoint anchors')
            for path in [*previous['endpoints'].values(),previous['selection'],previous['train_packet_codes']]:
                digest=old_plan['inputs_sha256'].get(path,old_plan['prepared_files_sha256'].get(path))
                require(digest is not None and sha(path)==digest,'Endpoint anchor changed');inputs[path]=digest
            for r in [*pos,*whole]:
                if r['path']:
                    require(sha(r['path'])==r['data_sha256'],'Reused raw result changed');inputs[r['path']]=r['data_sha256']
            require(sha(ps['path'])==ps['sha256'],'Prior summary changed');inputs[ps['path']]=ps['sha256']
            policies.append(dict(seed=seed,condition=condition,checkpoints=[c for c in checkpoints if c['seed']==seed and c['condition']==condition],
                prior_policy=previous,prior_position_records=pos,prior_whole_references=whole,prior_summary=ps))
    require(sources()==ss,'Source changed during preparation')
    artifacts={str(p):sha(p) for p in out.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts}
    plan=dict(status='prepared_without_new_forward',at=now(),config=CONFIG,policies=policies,sources_sha256=ss,
        inputs_sha256=inputs,prepared_files_sha256=artifacts,runtime=dict(python=platform.python_version(),numpy=np.__version__))
    write(out/'plan.json',plan);write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')))
    return dict(status=plan['status'],plan_sha256=sha(out/'plan.json'),config=CONFIG)

def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json')
    require(sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256'],'Plan changed')
    require(plan['config']==CONFIG and plan['sources_sha256']==sources(),'Frozen configuration or source changed')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Runtime changed')
    for path,digest in {**plan['inputs_sha256'],**plan['prepared_files_sha256']}.items():require(sha(path)==digest,'Input changed: '+path)
    for path,digest in plan['sources_sha256'].items():require(sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest,'Snapshot changed')
    return plan

def local_spec(validation,pool_index):
    spec=dict(validation);spec['endpoint_indices']=pool_index['recipient_pool_rows'];spec['donor_endpoint_indices']=pool_index['donor_pool_rows']
    ids=pool_index['validation_pool_endpoint_indices']
    require(np.array_equal(ids[spec['endpoint_indices']],validation['endpoint_indices']),'Recipient local/full mapping')
    require(np.array_equal(ids[spec['donor_endpoint_indices']],validation['donor_endpoint_indices']),'Donor local/full mapping')
    return spec

def generate_train(networks,x,live):
    messages=np.empty((len(x),2,3,4),np.int8);samples=calls=0
    for start in range(0,len(x),BATCH):
        stop=min(start+BATCH,len(x));trace=ch.generate_messages(networks,x[start:stop],live)
        messages[start:stop]=trace['messages'];samples+=trace['neural_forward_samples'];calls+=trace['neural_forward_calls']
    require(samples==6*len(x),'Train sender-only accounting')
    return messages,dict(new_forward_worlds=len(x),new_network_samples=samples,neural_forward_calls=calls,actions_generated=False)

def build_natural(networks,x,states,ids,live):
    trace=ch.natural_rollout(networks,x,live)
    data={k:trace[k] for k in ('messages','action_indices','action_probabilities')}
    data.update(states=states,state_indices=ids);data.update(ch.settle(states,data['action_indices']))
    return data,dict(new_forward_worlds=len(x),new_network_samples=trace['neural_forward_samples'],
        neural_forward_calls=trace['neural_forward_calls'],routing_sha256=trace['routing_sha256'])

def save_array(path,data):
    require(not path.exists(),'Refuse raw output overwrite');np.savez_compressed(path,**data)
    return dict(path=str(path),data_sha256=sha(path))

def position_data(policy,update,networks,x,pool,spec,full_ids,unit,arm,direction,live):
    reused=update==6000
    if reused and live:
        old=next(r for r in policy['prior_position_records'] if (r['unit'],r['arm'],r['direction'])==(unit,arm,direction))
        data=load_npz(old['path']);record=dict(old)
        require(np.array_equal(data['recipient_indices'],full_ids[spec['endpoint_indices'][:,direction]]),'Reused position recipient')
        _,di=pr.indices(spec,arm,direction)
        require(np.array_equal(data['donor_indices'],full_ids[di]),'Reused position donor')
        require(np.array_equal(data['donor_packets'],pool['messages'][di,:,spec['sender'],:]),'Reused position packets')
        record.update(new_forward_worlds=0,new_network_samples=0,neural_forward_calls=0,old_actual_forward_worlds=old['new_forward_worlds'])
    else:
        data,record=pr.execute_cell(networks,x,pool,spec,unit,arm,direction,live)
        if reused:
            old=next(r for r in policy['prior_position_records'] if (r['unit'],r['arm'],r['direction'])==(unit,arm,direction))
            require(record['routing_sha256']==old['routing_sha256'],'Reused silent position routes')
    record.update(kind='position',receiver_update=update,donor_update=update,is_reused=reused,
        index_scope='prior_full_domain' if reused and live else 'local554_pool',
        recipient_full_indices_sha256=pr.core.array_sha(full_ids[spec['endpoint_indices'][:,direction]]))
    return data,record

def whole_data(policy,rtime,stime,networks,x,receiver,donor_pool,spec,arm,direction,live):
    ids,di=pr.indices(spec,arm,direction);sender=spec['sender'];packets=donor_pool['messages'][di,:,sender,:]
    if rtime==stime==6000:
        old=next(r for r in policy['prior_whole_references'] if r['arm']==arm and r['direction']==direction)
        data=load_npz(old['path']);require(np.array_equal(data['patched_outward_packets'],packets),'Whole endpoint anchor packet')
        record=dict(path=old['path'],data_sha256=old['data_sha256'],is_reused=True,new_forward_worlds=0,new_network_samples=0,neural_forward_calls=0)
    else:
        trace=ch.cross_time_whole(networks,x,receiver['messages'][ids],sender,packets,live)
        data={k:trace[k] for k in ('messages','patched_outward_packets')}
        for key in ('action_indices','action_probabilities'):data[key]=trace[key] if live else receiver[key][ids].copy()
        data.update(donor_packets=packets,recipient_indices=ids,donor_indices=di,
            counterfactual_recipient_indices=spec['endpoint_indices'][:,1-direction],dataset_rows=np.arange(len(ids)))
        pr.settle_data(data,receiver,spec,direction)
        record=dict(is_reused=False,new_forward_worlds=len(ids) if live else 0,new_network_samples=int(trace['neural_forward_samples']),
            neural_forward_calls=int(trace['neural_forward_calls']),routing_sha256=trace['routing_sha256'])
    record.update(kind='cross_time_whole',receiver_update=rtime,donor_update=stime,arm=arm,direction=direction,
        worlds=len(ids),is_silent_alias=not live,outward_patch_visible=live,
        donor_packets_sha256=pr.core.array_sha(packets),index_scope='prior_reference' if record['is_reused'] else 'local554_pool')
    return data,record

def combine(values):
    return {k:np.stack([values[d][k] for d in (0,1)]) for k in values[0]}

def execute(out):
    out=Path(out).resolve();plan=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False)
    start=time.perf_counter();write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    records=[];train_records=[];natural_records=[];policies_done=[];loads=0
    try:
        meta=read(out/'dataset/pool.json');pool_index=load_npz(out/'dataset/pool.npz')
        validation=load_npz(out/'dataset/validation.npz');ds=load_npz(out/'dataset/discovery.npz');spec=local_spec(validation,pool_index)
        train_states=td.pack_states(meta['training_spec']);states=pool_index['validation_pool_states'];full_ids=pool_index['validation_pool_endpoint_indices']
        require(len(train_states)==419904 and len(states)==554,'Frozen natural supports')
        train_x={info:ch.observations(train_states,info) for info in ('PL','LL')}
        val_x={info:ch.observations(states,info) for info in ('PL','LL')}
        features={info:dict(train=pr.core.array_sha(train_x[info]),validation=pr.core.array_sha(val_x[info])) for info in train_x}
        write(execution/'features.json',features)
        for policy in plan['policies']:
            directory=execution/tag(policy);directory.mkdir();condition=policy['condition'];info=condition.split('_')[0];live=condition.endswith('_live')
            networks={c['update']:pr.core.load_networks(c['path']) for c in policy['checkpoints'] if c['update']!=6000 or live}
            loads+=len(networks);pools={};messages={};profiles={};codes={};points={};changes=[];cross={}
            final_profile=read(policy['prior_policy']['selection'])
            for update in STEPS:
                stepdir=directory/f'checkpoint_{update:04d}';stepdir.mkdir()
                if update==6000:
                    with np.load(policy['prior_policy']['endpoints']['train'],allow_pickle=False) as z:tm=z['messages']
                    with np.load(policy['prior_policy']['endpoints'][pr.PART],allow_pickle=False) as z:
                        pool={k:z[k][full_ids] for k in ('states','messages','action_indices','action_probabilities','greedy_reward','executed','satisfied')}
                    pool['state_indices']=full_ids.copy();require(np.array_equal(pool['states'],states),'Reused natural pool alignment')
                    trecord=dict(new_forward_worlds=0,new_network_samples=0,neural_forward_calls=0,actions_generated=False,
                        source_path=policy['prior_policy']['endpoints']['train'])
                    nrecord=dict(new_forward_worlds=0,new_network_samples=0,neural_forward_calls=0,
                        source_path=policy['prior_policy']['endpoints'][pr.PART])
                else:
                    tm,trecord=generate_train(networks[update],train_x[info],live)
                    pool,nrecord=build_natural(networks[update],val_x[info],states,full_ids,live)
                profile=discovery.select_positions(ds,tm);sets=discovery.training_code_sets(tm)
                if update==6000:require(profile==final_profile,'Endpoint discovery differs from frozen previous result')
                trecord.update(save_array(stepdir/'train_messages.npz',dict(messages=tm)))
                nrecord.update(save_array(stepdir/'natural_validation.npz',pool))
                for record,worlds in ((trecord,419904),(nrecord,554)):
                    record.update(seed=policy['seed'],condition=condition,update=update,is_reused=update==6000,worlds=worlds)
                write(stepdir/'train_messages.json',trecord);write(stepdir/'natural_validation.json',nrecord)
                write(stepdir/'selection.json',profile);save_array(stepdir/'train_packet_codes.npz',{str(a):sets[a] for a in range(3)})
                train_records.append(trecord);natural_records.append(nrecord)
                messages[update]=tm;pools[update]=pool;profiles[update]=profile;codes[update]=sets
                if update!=STEPS[0]:
                    earlier=STEPS[STEPS.index(update)-1]
                    change=metrics.temporal_change(messages[earlier],tm,profiles[earlier],profile)
                    change.update(earlier=earlier,later=update);changes.append(change)
                print(json.dumps(dict(stage='natural_completed',policy=tag(policy),update=update,elapsed_seconds=time.perf_counter()-start)),flush=True)
            for update in STEPS:
                stepdir=directory/f'checkpoint_{update:04d}';pool=pools[update];cells={};shams={}
                for unit,arm,direction in pr.cell_specs():
                    x=val_x[info][spec['endpoint_indices'][:,direction]]
                    data,record=position_data(policy,update,networks.get(update),x,pool,spec,full_ids,unit,arm,direction,live)
                    name=f'position_unit{unit}_{arm}_direction{direction}'
                    if not record['is_reused'] and live:record.update(save_array(stepdir/(name+'.npz'),data))
                    elif not live:record.update(path=None,data_sha256=None,alias_recipe='Current receiver natural output; invisible external single-position proposal; routes exactly checked.')
                    record.update(seed=policy['seed'],condition=condition,receiver_natural_source=str(stepdir/'natural_validation.npz'))
                    write(stepdir/(name+'.json'),record);records.append(record)
                    value=pr.measure(spec,direction,pool,data,codes[update])
                    if arm=='sham':shams[(unit,direction)]=value
                    else:cells[(unit,arm,direction)]=value
                positions={}
                for unit in range(8):
                    arms={arm:combine({d:cells[(unit,arm,d)] for d in (0,1)}) for arm in ('same','opposite')}
                    arms['contrast']=pm.contrast(arms['same'],arms['opposite']);positions[unit]=arms
                current=pm.selection_summary(spec,profiles[update],positions)
                current['all_positions']={str(u):{arm:pm.aggregate_rows(spec,v) for arm,v in arms.items()} for u,arms in positions.items()}
                retrospective=pm.selection_summary(spec,final_profile,positions)
                natural_values=combine({d:pr.measure(spec,d,pool,pr.natural_data(pool,spec,d),codes[update]) for d in (0,1)})
                point=dict(update=update,current=current,retrospective_final=retrospective,
                    discovery=metrics.discovery_description(profiles[update]),natural=pm.aggregate_rows(spec,natural_values),
                    shams={str(u):pm.aggregate_rows(spec,combine({d:shams[(u,d)] for d in (0,1)})) for u in (0,4)})
                write(stepdir/'summary.json',point);points[update]=point
            for rtime in STEPS:
                for stime in STEPS:
                    values={};cell_dir=directory/f'cross_receiver{rtime:04d}_donor{stime:04d}';cell_dir.mkdir()
                    for arm in ('same','opposite'):
                        for direction in (0,1):
                            x=val_x[info][spec['endpoint_indices'][:,direction]]
                            data,record=whole_data(policy,rtime,stime,networks.get(rtime),x,pools[rtime],pools[stime],spec,arm,direction,live)
                            name=f'{arm}_direction{direction}'
                            if not record['is_reused'] and live:record.update(save_array(cell_dir/(name+'.npz'),data))
                            elif not record['is_reused']:record.update(path=None,data_sha256=None,alias_recipe='Receiver-time natural output; proposed two-window donor-time packet is invisible.')
                            record.update(seed=policy['seed'],condition=condition,
                                receiver_natural_source=str(directory/f'checkpoint_{rtime:04d}/natural_validation.npz'),
                                donor_natural_source=str(directory/f'checkpoint_{stime:04d}/natural_validation.npz'))
                            write(cell_dir/(name+'.json'),record);records.append(record)
                            values[(arm,direction)]=pr.measure(spec,direction,pools[rtime],data,codes[rtime])
                    arms={arm:combine({d:values[(arm,d)] for d in (0,1)}) for arm in ('same','opposite')}
                    arms['contrast']=pm.contrast(arms['same'],arms['opposite'])
                    cross[(rtime,stime)]={arm:pm.aggregate_rows(spec,v) for arm,v in arms.items()}
                    write(cell_dir/'summary.json',cross[(rtime,stime)])
            summary=metrics.trajectory(points,cross)
            summary.update(seed=policy['seed'],condition=condition,temporal_changes=changes,
                discovery_at_checkpoints={str(t):points[t]['discovery'] for t in STEPS},
                point_summaries={str(t):dict(path=str(directory/f'checkpoint_{t:04d}/summary.json'),sha256=sha(directory/f'checkpoint_{t:04d}/summary.json')) for t in STEPS},
                cross_summaries=[dict(receiver_update=r,donor_update=s,path=str(directory/f'cross_receiver{r:04d}_donor{s:04d}/summary.json'),sha256=sha(directory/f'cross_receiver{r:04d}_donor{s:04d}/summary.json')) for r in STEPS for s in STEPS])
            write(directory/'summary.json',summary)
            policies_done.append(dict(seed=policy['seed'],condition=condition,path=str(directory/'summary.json'),sha256=sha(directory/'summary.json'),normalized_areas=summary['normalized_areas']))
            del networks,pools,messages
            print(json.dumps(dict(stage='policy_completed',policy=tag(policy),elapsed_seconds=time.perf_counter()-start)),flush=True)
        pos=[r for r in records if r['kind']=='position'];cross_records=[r for r in records if r['kind']=='cross_time_whole']
        totals=dict(new_train_message_worlds=sum(r['new_forward_worlds'] for r in train_records),
            new_train_message_module_samples=sum(r['new_network_samples'] for r in train_records),
            new_validation_natural_worlds=sum(r['new_forward_worlds'] for r in natural_records),
            new_validation_natural_module_samples=sum(r['new_network_samples'] for r in natural_records),
            new_position_worlds=sum(r['new_forward_worlds'] for r in pos),new_position_module_samples=sum(r['new_network_samples'] for r in pos),
            new_cross_time_worlds=sum(r['new_forward_worlds'] for r in cross_records),new_cross_time_module_samples=sum(r['new_network_samples'] for r in cross_records),
            total_new_module_samples=sum(r['new_network_samples'] for r in [*train_records,*natural_records,*records]),parameter_loads=loads,
            train_records=len(train_records),natural_records=len(natural_records),position_records=len(pos),cross_time_records=len(cross_records),
            new_position_npz=sum(not r['is_reused'] and not r['is_silent_alias'] for r in pos),
            new_cross_time_npz=sum(not r['is_reused'] and not r['is_silent_alias'] for r in cross_records),
            reused_position_records=sum(r['is_reused'] for r in pos),reused_cross_time_records=sum(r['is_reused'] for r in cross_records))
        require(all(totals[k]==CONFIG[k] for k in totals),'Execution accounting mismatch')
        paired=[]
        for seed in SEEDS:
            p=next(p for p in policies_done if p['seed']==seed and p['condition']=='PL_live')['normalized_areas']['selectivity']
            l=next(p for p in policies_done if p['seed']==seed and p['condition']=='LL_live')['normalized_areas']['selectivity']
            paired.append(dict(seed=seed,PL=p,LL=l,difference=p-l))
        verify(out)
        result=dict(status='completed',at=now(),elapsed_seconds=time.perf_counter()-start,plan_sha256=sha(out/'plan.json'),
            config=CONFIG,totals=totals,train_records=train_records,natural_records=natural_records,records=records,policies=policies_done,
            primary=dict(paired_seeds=paired,mean_difference=float(np.mean([p['difference'] for p in paired]))),training_updates=0)
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return {k:v for k,v in result.items() if k not in ('records','train_records','natural_records','policies','config')}
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),completed_records=len(records),completed_policies=len(policies_done)))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','execute','verify'));parser.add_argument('--out',required=True)
    args=parser.parse_args();result=globals()[args.command](args.out)
    if args.command=='verify':result=dict(status='verified',plan_sha256=sha(Path(args.out)/'plan.json'))
    print(json.dumps(result,ensure_ascii=False))
