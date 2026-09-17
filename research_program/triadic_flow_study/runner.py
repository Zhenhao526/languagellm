"""Same-background incoming/outgoing W1 substitutions in four final policies; no training."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse,json,multiprocessing,platform,shutil,time
import numpy as np
from research_program.triadic_private_partner_study import runner as previous
from research_program.triadic_reciprocal_execution_study import environment
from . import intervention,cases

core=intervention.core
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=HERE.parent/'triadic_private_partner_study/results/private_001'
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
AUDIT=SOURCE/'audit_execution_001/verification.json'
SEEDS=(59101,59102,59103,59104);CHECKPOINTS=(6000,)
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts');TARGET=PARTS[-1]
ANCHORS={'plan.json':'b5126ca468e16a41db4ee9e96c77f0a7e465c3c2123589d2f423911fc5bf5555',
    'execution/results.json':'4d80f8395a94a1bedbbbd27a4b287d011db88dbe9cf97ecef4b1e9540fe19f0e',
    'audit_execution_001/verification.json':'3bb1d4894094ec75fcb4ccc3c5187e188bea03c1c834260aa43dd38f6e7084f2'}
CONFIG=dict(schema='same_background_W1_flow_v1',seeds=list(SEEDS),checkpoints=list(CHECKPOINTS),
    partitions=list(PARTS),condition='PL_live',execution_rule='reciprocal',
    primary='mean_four_seed_complete_double_holdout_exact_S10_minus_S00',
    primary_response='S',primary_contrast='I_given_O0',
    case_order='group/background/context/self_endpoint/incoming/outgoing',
    donor='same_background; incoming:(1-context,self_endpoint); outgoing:(context,1-self_endpoint)',
    natural_bank='saved_audited_final_natural_banks_only; zero_new_native_forward',
    sham='all_00_cells; native_messages_actions_exact_and_probabilities_atol1e-13_rtol0',
    training_updates=0,worker_count=4,run_order='one_spawn_worker_per_seed; canonical_four_partitions',
    chunk_size=1024,automatic_followon_experiment=False,selection='all_frozen_cases_no_success_or_message_filter',
    responses=dict(actor_order=['focal','u','v'],u_v='other identities ascending',
        partner=['wait','focal','u','v'],site=['wait',0,1,2,3],material=['wait',0,1,2,3],
        kind=['wait',0,1],length=['wait',0,1],destination=['wait',0,1],
        execution=['none','AB','AC','BC'],changed_W1_symbols=['incoming','outgoing']),
    raw_action_order='physical_actor_A_B_C; all17 actions; semantic summaries retain wait',
    success='unique full-success reciprocal event probability; spectator integrated over all17 actions')

sha,read,write,json_bytes,array_sha,now=previous.sha,previous.read,previous.write,previous.json_bytes,previous.array_sha,previous.now
require=intervention.require


def prepared():
    path=ORIGINAL/'prepared.json'
    require(sha(path)=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299','Original partition hash')
    parts=read(path)['partitions'];cc={part:cases.build_cases(parts[part]) for part in PARTS}
    require(all(len(c['strata'])==9 and all(s['group_count']>0 for s in c['strata']) for c in cc.values()),'Nine nonempty case strata required')
    worlds=sum(p['world_count'] for p in parts.values())
    flow=sum(c['flow_cells'] for c in cc.values());sham=sum(c['sham_cells'] for c in cc.values())
    require((worlds,flow,sham)==(774144,2018304,504576),'Unexpected frozen support budget')
    budget=dict(policy_states=4,trained_policies=4,training_updates=0,checkpoint_files_read=4,
        new_natural_bank_files=0,reused_final_natural_bank_files=16,new_natural_worlds=0,
        reused_final_natural_worlds=4*worlds,new_natural_module_samples=0,
        flow_files=16,group_metric_files=16,flow_rows=4*flow,sham_rows=4*sham,
        intervention_rows=4*flow,intervention_module_samples=4*flow*6,
        total_new_module_samples=4*flow*6,generated_data_files=32,formal_random_draws=0)
    return dict(schema=CONFIG['schema'],partitions=parts,cases=cc,budget=budget,
        original_prepared_sha256=sha(path),independent_societies=4)


def input_manifest():
    manifest={str(SOURCE/name):digest for name,digest in ANCHORS.items()}
    for path,digest in manifest.items():require(sha(path)==digest,f'Changed source anchor: {path}')
    source=read(SOURCE/'execution/results.json');audit=read(AUDIT)
    require(source['status']=='completed' and audit['status']=='passed','Source run/audit incomplete')
    require(source['plan_sha256']==audit['plan_sha256']==ANCHORS['plan.json'],'Source plan mismatch')
    source_plan=read(SOURCE/'plan.json');source_freeze=read(SOURCE/'freeze.json')
    require(source_freeze['plan_sha256']==ANCHORS['plan.json'] and
        source_freeze['prepared_sha256']==source_plan['prepared_sha256']==sha(SOURCE/'prepared.json'),'Source static freeze mismatch')
    require(sha(ORIGINAL/'prepared.json')=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299','Original prepared anchor')
    for path in (SOURCE/'freeze.json',SOURCE/'prepared.json',ORIGINAL/'prepared.json'):
        manifest[str(path)]=sha(path)
    runs={(r['seed'],r['condition']):r for r in source['runs']}
    require(len(runs)==8 and set(runs)=={(s,c) for s in SEEDS for c in ('PL_live','PL_silent')},'Unexpected source policy grid')
    states=[]
    for seed in SEEDS:
        run=runs[seed,'PL_live'];directory=SOURCE/'execution'/f'seed_{seed}_PL_live'
        for step in CHECKPOINTS:
            checkpoint=directory/f'checkpoint_{step:04d}.npz'
            stored=next(v for v in run['monitor'] if v['update']==step)['checkpoint_sha256']
            require(audit['artifacts_sha256'][str(checkpoint)]==stored,'Checkpoint not bound by completed audit')
            manifest[str(checkpoint)]=stored
            meta=dict(seed=seed,checkpoint=step,checkpoint_path=str(checkpoint),checkpoint_sha256=stored,
                parameter_sha256=run['initial_parameter_sha256'] if step==0 else run['final_parameter_sha256'],natural_banks={})
            if step==6000:
                require(stored==run['final_checkpoint_sha256'],'Final checkpoint identity')
                for part in PARTS:
                    bank=run['final'][part]['natural'];path=directory/f'final_{part}_natural.npz'
                    require(Path(bank['path'])==path and bank['live'] is True and bank['information']=='PL','Wrong source natural bank')
                    require(audit['artifacts_sha256'][str(path)]==bank['data_sha256'],'Bank not bound by completed audit')
                    manifest[str(path)]=bank['data_sha256']
                    meta['natural_banks'][part]=dict(path=str(path),sha256=bank['data_sha256'],worlds=bank['worlds'])
            states.append(meta)
    require(len(states)==4 and len({v['checkpoint_path'] for v in states})==4,'Expected four final source checkpoints')
    return manifest,states


def sources():
    out=dict(intervention.frozen_sources())
    for path,digest in read(SOURCE/'plan.json')['source_sha256'].items():
        require(sha(path)==digest,f'Frozen production dependency changed: {path}');out[path]=digest
    directed=HERE.parent/'triadic_directed_message_study/cases.py'
    require(sha(directed)=='1963d501318869158837f4b04e24394b14ed7c7bbb6bd9cbd2678f1ef5479ccf','Frozen mirrored group source changed')
    out[str(directed)]=sha(directed)
    for name in ('__init__.py','runner.py','test_runner.py','intervention.py','test_intervention.py',
        'intervention_preflight_001.json','cases.py','test_cases.py','static_availability.json','cases_preflight_001.json',
        'plan.md','support_review.md','design_review.md','literature_sources_001.json',
        'literature/近邻与创新性约束.md','runner_preflight_001.json','main_preflight.json'):
        path=HERE/name;out[str(path)]=sha(path)
    return out


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Never overwrite preparation')
    static=prepared();inputs,states=input_manifest();ss=sources()
    for path,digest in inputs.items():require(sha(path)==digest,f'Changed frozen input: {path}')
    out.mkdir(parents=True)
    for path in ss:
        destination=out/'source_snapshot'/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    write(out/'prepared.json',static)
    write(out/'plan.json',dict(status='prepared_without_policy_forward',at=now(),config=CONFIG,
        inputs_sha256=inputs,source_sha256=ss,policy_states=states,prepared_sha256=sha(out/'prepared.json'),
        runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json'),prepared_sha256=sha(out/'prepared.json')))
    verify(out)
    return dict(status='prepared_without_policy_forward',plan_sha256=sha(out/'plan.json'),budget=static['budget'])


def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json');static=read(out/'prepared.json');freeze=read(out/'freeze.json')
    require(sha(out/'plan.json')==freeze['plan_sha256'] and plan['config']==CONFIG,'Frozen plan mismatch')
    require(sha(out/'prepared.json')==plan['prepared_sha256']==freeze['prepared_sha256'],'Frozen prepared mismatch')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Frozen runtime mismatch')
    inputs,states=input_manifest()
    require(plan['inputs_sha256']==inputs and plan['policy_states']==states,'Frozen input manifest mismatch')
    require(plan['source_sha256']==sources() and static==prepared(),'Frozen source/static support mismatch')
    for path,digest in plan['inputs_sha256'].items():require(sha(path)==digest,f'Changed frozen input: {path}')
    for path,digest in plan['source_sha256'].items():
        require(sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest,f'Changed source snapshot: {path}')
    return plan,static


def save_npz(path,values):
    path=Path(path)
    with path.open('xb') as stream:np.savez_compressed(stream,**values)
    return dict(path=str(path),sha256=sha(path),arrays={k:dict(shape=list(np.shape(v)),dtype=str(np.asarray(v).dtype)) for k,v in values.items()})


def check_bank(arrays,bank):
    n=len(arrays['packed_states'])
    require(np.array_equal(bank['states'],arrays['packed_states']),'Natural bank state packing mismatch')
    require(np.array_equal(bank['state_indices'],np.arange(n)),'Natural bank must use complete canonical domain')
    messages=bank['messages'];probabilities=bank['action_probabilities'];actions=bank['action_indices']
    require(messages.shape==(n,2,3,4) and messages.dtype.kind in 'iu' and np.all((messages>=0)&(messages<8)),'Bad native messages')
    require(probabilities.shape==(n,3,17) and np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)), 'Bad native probabilities')
    require(np.allclose(probabilities.sum(-1),1,atol=1e-14,rtol=0),'Unnormalized native probabilities')
    require(actions.shape==(n,3) and actions.dtype.kind in 'iu' and np.array_equal(actions,probabilities.argmax(-1)),'Bad native greedy actions')


def saved_bank(arrays,meta):
    path=Path(meta['path']);require(sha(path)==meta['sha256'],'Source bank hash mismatch')
    with np.load(path,allow_pickle=False) as saved:
        bank={k:saved[k] for k in ('states','state_indices','messages','action_probabilities','action_indices')}
    check_bank(arrays,bank)
    return bank,dict(path=str(path),sha256=meta['sha256'],worlds=len(bank['states']),reused=True,new_module_samples=0)


def check_context_packets(case_spec,bank):
    """The exact same sender W1 must hold across the mirrored other-needs context."""
    groups=np.asarray(case_spec['group_need_indices'],dtype=np.int64)
    senders=np.asarray(case_spec['sender'],dtype=np.int64);backgrounds=case_spec['n_backgrounds']
    checked=0
    for background in range(backgrounds):
        before=groups[:,0,:]*backgrounds+background;after=groups[:,1,:]*backgrounds+background
        first=bank['messages'][before,0,senders[:,None]]
        second=bank['messages'][after,0,senders[:,None]]
        require(np.array_equal(first,second),'Sender W1 changed when only other private needs were swapped')
        checked+=2*len(groups)
    other_checked=0
    for background in range(backgrounds):
        before=groups[:,:,0]*backgrounds+background;after=groups[:,:,1]*backgrounds+background
        for gi,sender in enumerate(senders):
            others=[a for a in range(3) if a!=sender]
            require(np.array_equal(bank['messages'][before[gi],0][:,others],bank['messages'][after[gi],0][:,others]),
                'Other W1 changed when only focal private need changed')
            other_checked+=4
    return dict(compared_sender_packets=checked,compared_other_packets=other_checked,all_equal=True,neural_forward_samples=0)


def verify_sham(result,bank,receiver_indices):
    ids=np.asarray(receiver_indices)
    require(np.array_equal(result['generated_messages'],bank['messages'][ids]),'Sham generated messages do not reproduce native')
    require(np.array_equal(result['action_indices'],bank['action_indices'][ids]),'Sham actions do not reproduce native')
    error=float(np.max(np.abs(result['action_probabilities']-bank['action_probabilities'][ids])))
    require(np.isfinite(error) and error<=1e-13,'Sham probabilities do not reproduce native')
    return error


def exact_statistics(probabilities,rewards):
    """Researcher-only integration of all17 actions, conditional on greedy messages."""
    p=np.asarray(probabilities);r=np.asarray(rewards);n=len(p)
    require(p.shape==(n,3,17) and n>0 and p.dtype.kind in 'fiu' and np.isfinite(p).all()
        and np.all((p>=0)&(p<=1)) and np.allclose(p.sum(-1),1,atol=1e-13,rtol=0),'Invalid action probabilities')
    require(r.shape==(n,24) and r.dtype.kind in 'fiu' and np.isin(r,(0,.5,1)).all()
        and np.all((r==1).sum(-1)==1),'A unique full-success structural event is required')
    events=np.ones((n,24),dtype=np.float64)
    for actor in range(3):
        actions=environment.STRUCTURAL_ACTIONS[:,actor];active=actions>0
        events[:,active]*=p[:,actor,actions[active]]
    pairs=events.reshape(n,3,8).sum(-1);none=1-pairs.sum(-1)
    require(np.all(none>=-1e-13),'Overlapping or unnormalized reciprocal execution events')
    # Preserve floating arithmetic without clipping/renormalizing the defined complement.
    return dict(exact_full_success_probability=(events*(r==1)).sum(-1),
        expected_native_reward=(events*r).sum(-1),execution_probabilities=np.column_stack((none,pairs)))


def semantic_responses(probabilities,states,focal):
    p=np.asarray(probabilities);states=environment._states(states);n=len(states)
    require(p.shape==(n,3,17),'Semantic probability shape')
    focal=intervention._focal(focal,n)
    order=np.asarray([[int(s)]+[a for a in range(3) if a!=s] for s in focal],dtype=np.int8)
    ordered=p[np.arange(n)[:,None],order];wait=ordered[:,:,0:1]
    nonwait=ordered[:,:,1:].reshape(n,3,4,2,2)
    site=nonwait.sum(axis=(3,4));destination=nonwait.sum(axis=(2,4))
    materials=states[:,3:7]
    material=np.stack([(site*(materials[:,None,:]==m)).sum(-1) for m in range(4)],axis=-1)
    kind=np.stack((material[:,:,:2].sum(-1),material[:,:,2:].sum(-1)),axis=-1)
    length=np.stack((material[:,:,::2].sum(-1),material[:,:,1::2].sum(-1)),axis=-1)
    roles=environment.PROPOSAL_ROLES[order]
    partner=np.stack([(ordered*(roles==order[:,j,None,None])).sum(-1) for j in range(3)],axis=-1)
    out={key:np.concatenate((wait,value),axis=-1) for key,value in
        dict(partner=partner,site=site,material=material,kind=kind,length=length,destination=destination).items()}
    require(all(np.allclose(v.sum(-1),1,atol=1e-13,rtol=0) for v in out.values()),'Semantic marginal lost probability mass')
    require(np.all(out['partner'][:,np.arange(3),1+np.arange(3)]==0),'Self-partner mass must be zero')
    return order,out


def changed_symbols(delivered,native_messages,focal,flags_I,flags_O):
    n=len(delivered);rows=np.arange(n);focus=np.asarray(focal)
    native=np.asarray(native_messages)[:,0]
    incoming=(delivered[rows,0,focus]!=native).sum(axis=(1,2))
    outgoing=(delivered[rows,0,:,focus]!=native[rows,focus,None,:]).sum(axis=(1,2))
    result=np.column_stack((incoming,outgoing)).astype(np.int8)
    require(np.all((result>=0)&(result<=8)),'Changed-symbol bounds')
    require(np.all(result[~np.asarray(flags_I,dtype=bool),0]==0) and
        np.all(result[~np.asarray(flags_O,dtype=bool),1]==0),'Off edge set changed symbols')
    return result


def evaluate_cells(networks,arrays,bank,case_spec,path):
    path=Path(path);require(not path.exists(),'Never overwrite flow output')
    meta=cases.expand(case_spec);receiver=meta['receiver_state_indices'];focal=meta['focus_actor'];n=len(receiver)
    require(n==case_spec['flow_cells'] and n>0,'Unexpected complete flow count')
    require(np.all((receiver>=0)&(receiver<len(bank['states']))),'Receiver indices out of range')
    data={k:np.asarray(v) for k,v in meta.items()}
    data.update(receiver_states=bank['states'][receiver],
        generated_messages=np.empty((n,2,3,4),np.int8),delivered_tokens=np.empty((n,2,3,3,4),np.int8),
        delivery_visibility=np.empty((n,2,3,3),bool),incoming_donor_W1=np.empty((n,3,4),np.int8),
        outgoing_donor_W1=np.empty((n,4),np.int8),action_probabilities=np.empty((n,3,17)),
        action_indices=np.empty((n,3),np.int16),exact_full_success_probability=np.empty(n),
        expected_native_reward=np.empty(n),execution_probabilities=np.empty((n,4)),
        changed_W1_symbols=np.empty((n,2),np.int8),response_actor_agents=np.empty((n,3),np.int8))
    maximum_error=0.;checked=0;module_samples=0
    for start in range(0,n,CONFIG['chunk_size']):
        sl=slice(start,min(start+CONFIG['chunk_size'],n));ri=receiver[sl];s=focal[sl]
        incoming=bank['messages'][meta['incoming_donor_state_indices'][sl],0]
        outgoing=bank['messages'][meta['outgoing_donor_state_indices'][sl],0,s]
        result=intervention.intervene(networks,arrays['x_PL'][ri],bank['messages'][ri],s,incoming,outgoing,
            meta['incoming'][sl],meta['outgoing'][sl])
        require(result['neural_forward_samples']==6*len(ri),'Intervention module budget mismatch')
        module_samples+=result['neural_forward_samples']
        for key in ('generated_messages','delivered_tokens','delivery_visibility','incoming_donor_W1','outgoing_donor_W1','action_probabilities','action_indices'):
            data[key][sl]=result[key]
        for key,value in exact_statistics(result['action_probabilities'],arrays['rewards'][ri]).items():data[key][sl]=value
        data['response_actor_agents'][sl]=np.asarray([[int(a)]+[b for b in range(3) if b!=a] for a in s],dtype=np.int8)
        data['changed_W1_symbols'][sl]=changed_symbols(result['delivered_tokens'],bank['messages'][ri],s,meta['incoming'][sl],meta['outgoing'][sl])
        mask=meta['sham'][sl]
        if np.any(mask):
            selected={k:result[k][mask] for k in ('generated_messages','action_indices','action_probabilities')}
            maximum_error=max(maximum_error,verify_sham(selected,bank,ri[mask]));checked+=int(mask.sum())
    require(checked==case_spec['sham_cells'],'Incomplete sham replay')
    data.update(environment.settle(data['receiver_states'],data['action_indices'],'reciprocal'))
    record=save_npz(path,data);record.update(rows=n,new_module_samples=module_samples,
        physical_summary_scope='All repeated flow cells; descriptive case-weighted settlement, not the natural full-world task score.',
        physical_summary=environment.metrics(data,environment.truth_from_rewards(data['receiver_states'],arrays['rewards'][receiver]),data['action_indices'],'reciprocal'),
        native_replay=dict(all_messages_equal=True,all_actions_equal=True,max_probability_absolute_error=maximum_error,checked_rows=checked))
    return record,data


def save_metrics(case_spec,data,path):
    order,semantic=semantic_responses(data['action_probabilities'],data['receiver_states'],data['focus_actor'])
    require(np.array_equal(order,data['response_actor_agents']),'Semantic actor order mismatch')
    values=dict(S=data['exact_full_success_probability'],expected_reward=data['expected_native_reward'],
        execution=data['execution_probabilities'],**semantic,changed_W1_symbols=data['changed_W1_symbols'])
    result={};details={}
    for response,value in values.items():
        block=cases.metrics(case_spec,value);raw=block.pop('per_group_background')
        for key,array in raw.items():
            numeric=np.asarray(array)
            require(numeric.dtype.kind in 'fiu' and np.isfinite(numeric).all(),'Nonfinite/nonnumeric detailed metric')
            details[response+'__'+key]=numeric
        block['per_group_background_key_prefix']=response+'__';result[response]=block
    file=save_npz(path,details)
    for block in result.values():block['per_group_background_file']=dict(path=file['path'],sha256=file['sha256'])
    return result,file


def worker(payload):
    seed,meta,static,execution=payload;directory=Path(execution)/f'seed_{seed}'
    directory.mkdir(exist_ok=False);started=time.perf_counter()
    previous.process_status(directory/'status.json',status='building_arrays',seed=seed)
    arrays={part:previous.make_arrays(static['partitions'][part]) for part in PARTS}
    array_hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()}
    write(directory/'array_hashes.json',array_hashes)
    require(meta['seed']==seed and meta['checkpoint']==6000,'Incorrect frozen policy state')
    require(sha(meta['checkpoint_path'])==meta['checkpoint_sha256'],'Checkpoint changed before load')
    with np.load(meta['checkpoint_path'],allow_pickle=False) as saved:require(int(saved['update'])==6000,'Checkpoint update mismatch')
    networks=core.load_networks(meta['checkpoint_path'])
    require(core.parameter_hash(networks)==meta['parameter_sha256'],'Policy parameter identity mismatch')
    parts={}
    for part in PARTS:
        previous.process_status(directory/'status.json',status='running',seed=seed,checkpoint=6000,part=part)
        part_dir=directory/part;part_dir.mkdir(exist_ok=False);case_spec=static['cases'][part]
        bank,natural_record=saved_bank(arrays[part],meta['natural_banks'][part])
        cases.validate_partition_arrays(case_spec,bank['states'],bank['state_indices'])
        context_check=check_context_packets(case_spec,bank)
        flow,data=evaluate_cells(networks,arrays[part],bank,case_spec,part_dir/'flow.npz')
        metrics,metric_file=save_metrics(case_spec,data,part_dir/'flow_metrics.npz');del data,bank
        record=dict(natural_bank=natural_record,context_packet_check=context_check,flow=flow,metrics=metrics,metric_file=metric_file)
        write(part_dir/'result.json',record);parts[part]=record
        print(json.dumps(dict(stage='partition_complete',seed=seed,checkpoint=6000,part=part)),flush=True)
    require(core.parameter_hash(networks)==meta['parameter_sha256'],'Intervention changed policy weights')
    require(array_hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()},'World arrays changed')
    record=dict(seed=seed,checkpoint=6000,checkpoint_sha256=meta['checkpoint_sha256'],parameter_sha256=meta['parameter_sha256'],
        parts=parts,elapsed_seconds=time.perf_counter()-started)
    write(directory/'result.json',record)
    previous.process_status(directory/'status.json',status='completed',seed=seed,elapsed_seconds=time.perf_counter()-started)
    return record


def primary(records):
    require(len(records)==4,'Expected four complete final policy states');by={r['seed']:r for r in records}
    require(set(by)==set(SEEDS) and all(r['checkpoint']==6000 for r in records),'Missing/duplicate final policy states')
    rows=[]
    for seed in SEEDS:
        block=by[seed]['parts'][TARGET]['metrics']['S'];delta=block['I_given_O0']
        require(type(delta) in (int,float) and np.isfinite(delta) and -1<=delta<=1,'Invalid primary exact probability contrast')
        require(all(type(block[k]) in (int,float) and np.isfinite(block[k]) and 0<=block[k]<=1 for k in ('S00','S01','S10','S11')),'Invalid exact cell probability')
        require(abs(delta-(block['S10']-block['S00']))<=1e-13,'Primary contrast sign or pairing mismatch')
        rows.append(dict(seed=seed,S10_minus_S00=delta,**{k:block[k] for k in ('S00','S01','S10','S11')}))
    return dict(name=CONFIG['primary'],partition=TARGET,response='S',contrast='I_given_O0',independent_societies=4,
        by_seed=rows,mean_S10_minus_S00=float(np.mean([r['S10_minus_S00'] for r in rows])),
        interpretation='Conditional exact action full-success probability given recomputed greedy messages; signed incoming effect with outgoing unmodified.')


def execute(out):
    out=Path(out).resolve();plan,static=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False)
    start=time.perf_counter();write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    previous.process_status(execution/'status.json',status='running',worker_count=4)
    try:
        payloads=[(s,next(r for r in plan['policy_states'] if r['seed']==s),static,str(execution)) for s in SEEDS]
        with multiprocessing.get_context('spawn').Pool(4) as pool:records=pool.map(worker,payloads)
        require([(r['seed'],r['checkpoint']) for r in records]==[(s,6000) for s in SEEDS],'Noncanonical completed grid')
        hashes=[read(execution/f'seed_{s}'/'array_hashes.json') for s in SEEDS]
        require(all(h==hashes[0] for h in hashes),'Workers disagree on full domain');verify(out)
        measured=dict(new_natural_module_samples=0,
            intervention_module_samples=sum(p['flow']['new_module_samples'] for r in records for p in r['parts'].values()))
        measured['total_new_module_samples']=sum(measured.values())
        require(all(static['budget'][k]==v for k,v in measured.items()),'Measured forward budget mismatch')
        require(sum(p['flow']['rows'] for r in records for p in r['parts'].values())==static['budget']['flow_rows'],'Measured row count mismatch')
        require(sum(p['flow']['native_replay']['checked_rows'] for r in records for p in r['parts'].values())==static['budget']['sham_rows'],'Measured sham count mismatch')
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),policy_states=records,
            budget=static['budget'],measured_module_samples=measured,array_hashes=hashes[0],primary=primary(records),elapsed_seconds=time.perf_counter()-start)
        write(execution/'results.json',result)
        previous.process_status(execution/'status.json',status='completed',results_sha256=sha(execution/'results.json'))
        return dict(status='completed',primary=result['primary'],elapsed_seconds=result['elapsed_seconds'])
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),elapsed_seconds=time.perf_counter()-start,automatic_retry=False))
        previous.process_status(execution/'status.json',status='failed',error=repr(error));raise


run=execute
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','verify','execute','run'));parser.add_argument('--out',required=True)
    args=parser.parse_args();result=globals()[args.command](args.out)
    print(json.dumps(result if args.command!='verify' else dict(status='verified'),ensure_ascii=False))
