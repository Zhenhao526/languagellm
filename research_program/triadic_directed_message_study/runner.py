"""Fixed full-support W1 substitutions in eight frozen policy states; no training."""
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
SEEDS=(59101,59102,59103,59104);CHECKPOINTS=(0,6000)
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts');TARGET=PARTS[-1]
ANCHORS={'plan.json':'b5126ca468e16a41db4ee9e96c77f0a7e465c3c2123589d2f423911fc5bf5555',
    'execution/results.json':'4d80f8395a94a1bedbbbd27a4b287d011db88dbe9cf97ecef4b1e9540fe19f0e',
    'audit_execution_001/verification.json':'3bb1d4894094ec75fcb4ccc3c5187e188bea03c1c834260aa43dd38f6e7084f2'}
CONFIG=dict(schema='directed_W1_context_reversal_v1',seeds=list(SEEDS),checkpoints=list(CHECKPOINTS),
    partitions=list(PARTS),condition='PL_live',execution_rule='reciprocal',overwrite_windows=[True,False],
    donor='canonical_context_0_both_sender_needs; half_cycle_other_layout_same_owner',
    primary='mean_four_seed_final_complete_double_holdout_L',
    auxiliary='initial_L; final_minus_initial_L; D; strata; physical_settlement',
    case_order='cross:group/background/context/self_endpoint/packet_endpoint; sham:group/background/context/self_endpoint',
    natural_bank='checkpoint0_full_native_9B; checkpoint6000_saved_audited_natural_bank_0B',
    sham='all_four_cells_per_group_background; native_messages_actions_exact_and_probabilities_atol1e-13_rtol0',
    training_updates=0,worker_count=4,run_order='one_spawn_worker_per_seed; checkpoint0_then6000; canonical_four_partitions',
    chunk_size=1024,automatic_followon_experiment=False,selection='all_frozen_cases_no_success_or_message_filter')
sha,read,write,json_bytes,array_sha,now=previous.sha,previous.read,previous.write,previous.json_bytes,previous.array_sha,previous.now
require=intervention.require


def prepared():
    path=ORIGINAL/'prepared.json'
    require(sha(path)=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299','Original partition hash')
    parts=read(path)['partitions'];cc={part:cases.build_cases(parts[part]) for part in PARTS}
    require(all(len(c['strata'])==9 and all(s['group_count']>0 for s in c['strata']) for c in cc.values()),'Nine nonempty case strata required')
    worlds=sum(p['world_count'] for p in parts.values())
    cross=sum(c['cross_cells'] for c in cc.values());sham=sum(c['sham_cells'] for c in cc.values())
    require((worlds,cross,sham)==(774144,1009152,504576),'Unexpected frozen support budget')
    budget=dict(policy_states=8,trained_policies=4,training_updates=0,checkpoint_files_read=8,
        initial_natural_bank_files=16,reused_final_natural_bank_files=16,
        new_natural_worlds=4*worlds,reused_final_natural_worlds=4*worlds,
        new_natural_module_samples=4*worlds*9,
        cross_files=32,sham_files=32,group_metric_files=32,cross_rows=8*cross,sham_rows=8*sham,
        intervention_rows=8*(cross+sham),intervention_module_samples=8*(cross+sham)*6,
        total_new_module_samples=4*worlds*9+8*(cross+sham)*6,
        generated_data_files=16+32+32+32,formal_random_draws=0)
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
    require(len(states)==8 and len({v['checkpoint_path'] for v in states})==8,'Expected eight source checkpoints')
    return manifest,states


def sources():
    out=dict(intervention.frozen_sources())
    for path,digest in read(SOURCE/'plan.json')['source_sha256'].items():
        require(sha(path)==digest,f'Frozen production dependency changed: {path}');out[path]=digest
    for name in ('__init__.py','runner.py','test_runner.py','intervention.py','test_intervention.py',
        'intervention_preflight_001.json','cases.py','test_cases.py','static_availability.json','cases_preflight_001.json',
        'plan.md','support_review.md','design_review.md','static_identification_proof.py','static_identification_proof_001.json',
        'literature_sources_001.json','runner_preflight_001.json','main_preflight.json'):
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


def native_bank(networks,arrays,path):
    n=len(arrays['packed_states']);path=Path(path);require(not path.exists(),'Never overwrite a native bank')
    bank=dict(states=arrays['packed_states'].copy(),state_indices=np.arange(n,dtype=np.int64),
        messages=np.empty((n,2,3,4),np.int8),action_probabilities=np.empty((n,3,17)),action_indices=np.empty((n,3),np.int16))
    for start in range(0,n,CONFIG['chunk_size']):
        sl=slice(start,min(start+CONFIG['chunk_size'],n));trace=core.rollout(networks,arrays['x_PL'][sl],True)
        probabilities,_=core.base.policy_distribution(trace['action_logits'])
        bank['messages'][sl]=trace['messages'];bank['action_probabilities'][sl]=probabilities;bank['action_indices'][sl]=probabilities.argmax(-1)
    check_bank(arrays,bank);bank.update(environment.settle(bank['states'],bank['action_indices'],'reciprocal'))
    record=save_npz(path,bank);record.update(worlds=n,reused=False,new_module_samples=9*n,
        physical_summary=environment.metrics(bank,environment.truth_from_rewards(bank['states'],arrays['rewards']),bank['action_indices'],'reciprocal'))
    return bank,record


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
    return dict(compared_sender_packets=checked,all_equal=True,neural_forward_samples=0)


def recipient_probabilities(action_probabilities,senders,recipients):
    p=np.asarray(action_probabilities);senders=np.asarray(senders);recipients=np.asarray(recipients);n=len(p)
    require(p.shape==(n,3,17) and recipients.shape==(n,2) and senders.shape==(n,),'Recipient probability shape')
    require(np.all((senders>=0)&(senders<3)) and np.all((recipients>=0)&(recipients<3)) and np.all(recipients!=senders[:,None]),'Recipient identity mismatch')
    roles=environment.PROPOSAL_ROLES[recipients]
    selected=p[np.arange(n)[:,None],recipients]
    mask=roles==senders[:,None,None]
    require(np.all(mask.sum(-1)==8),'Partner event must retain eight of seventeen proposals')
    return (selected*mask).sum(-1)


def verify_sham(result,bank,receiver_indices):
    ids=np.asarray(receiver_indices)
    require(np.array_equal(result['generated_messages'],bank['messages'][ids]),'Sham generated messages do not reproduce native')
    require(np.array_equal(result['action_indices'],bank['action_indices'][ids]),'Sham actions do not reproduce native')
    error=float(np.max(np.abs(result['action_probabilities']-bank['action_probabilities'][ids])))
    require(np.isfinite(error) and error<=1e-13,'Sham probabilities do not reproduce native')
    return error


def evaluate_cells(networks,arrays,bank,case_spec,mode,path):
    require(mode in ('cross','sham'),'Unknown intervention mode');path=Path(path)
    require(not path.exists(),'Never overwrite intervention output')
    meta=cases.expand(case_spec,mode=mode)
    receiver=np.asarray(meta['receiver_state_indices']);donor=np.asarray(meta['donor_state_indices'])
    senders=np.asarray(meta['sender']);recipients=np.asarray(meta['recipient_agents']);n=len(receiver)
    require(n==case_spec[mode+'_cells'],'Unexpected complete case count')
    require(receiver.dtype.kind in 'iu' and donor.dtype.kind in 'iu' and
        np.all((receiver>=0)&(receiver<len(bank['states']))) and np.all((donor>=0)&(donor<len(bank['states']))),'Case state indices out of range')
    data={k:np.asarray(v) for k,v in meta.items()}
    data.update(receiver_states=bank['states'][receiver],
        generated_messages=np.empty((n,2,3,4),np.int8),delivered_tokens=np.empty((n,2,3,3,4),np.int8),
        delivery_visibility=np.empty((n,2,3,3),bool),donor_packets=np.empty((n,2,4),np.int8),
        action_probabilities=np.empty((n,3,17)),action_indices=np.empty((n,3),np.int16),
        recipient_partner_probs=np.empty((n,2)),overwrite_windows=np.array([True,False],bool))
    maximum_error=0.
    for start in range(0,n,CONFIG['chunk_size']):
        sl=slice(start,min(start+CONFIG['chunk_size'],n));ri=receiver[sl];di=donor[sl];sender=senders[sl]
        packets=bank['messages'][di,:,sender]
        result=intervention.intervene(networks,arrays['x_PL'][ri],bank['messages'][ri],sender,packets,overwrite_windows=(True,False))
        require(result['neural_forward_samples']==6*len(ri),'Intervention module budget mismatch')
        for key in ('generated_messages','delivered_tokens','delivery_visibility','donor_packets','action_probabilities','action_indices'):
            data[key][sl]=result[key]
        data['recipient_partner_probs'][sl]=recipient_probabilities(result['action_probabilities'],sender,recipients[sl])
        if mode=='sham':
            maximum_error=max(maximum_error,verify_sham(result,bank,ri))
    data.update(environment.settle(data['receiver_states'],data['action_indices'],'reciprocal'))
    record=save_npz(path,data);record.update(mode=mode,rows=n,new_module_samples=6*n,
        physical_summary_scope='All repeated intervention cells; descriptive case-weighted settlement, not the natural full-world task score.',
        physical_summary=environment.metrics(data,environment.truth_from_rewards(data['receiver_states'],arrays['rewards'][receiver]),data['action_indices'],'reciprocal'))
    if mode=='sham':record['native_replay']=dict(all_messages_equal=True,all_actions_equal=True,max_probability_absolute_error=maximum_error,checked_rows=n)
    return record,data['recipient_partner_probs']


def save_metrics(case_spec,probabilities,path):
    result=cases.metrics(case_spec,probabilities)
    details=result.pop('per_group_background')
    require(isinstance(details,dict),'Expected parallel per-group/background metric arrays')
    arrays={key:np.asarray(value) for key,value in details.items()}
    require(all(a.dtype.kind in 'fiu' and np.isfinite(a).all() for a in arrays.values()),'Metric details must be finite numeric arrays')
    result['per_group_background_file']=save_npz(path,arrays)
    return result


def worker(payload):
    seed,policy_states,static,execution=payload;execution=Path(execution);directory=execution/f'seed_{seed}'
    directory.mkdir(exist_ok=False);started=time.perf_counter()
    previous.process_status(directory/'status.json',status='building_arrays',seed=seed)
    arrays={part:previous.make_arrays(static['partitions'][part]) for part in PARTS}
    array_hashes={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()}
    write(directory/'array_hashes.json',array_hashes);output=[]
    require([s['checkpoint'] for s in policy_states]==list(CHECKPOINTS),'Incorrect policy-state order')
    for meta in policy_states:
        step=meta['checkpoint'];step_dir=directory/f'checkpoint_{step:04d}';step_dir.mkdir(exist_ok=False)
        require(sha(meta['checkpoint_path'])==meta['checkpoint_sha256'],'Checkpoint changed before load')
        with np.load(meta['checkpoint_path'],allow_pickle=False) as saved:
            require(int(saved['update'])==step,'Checkpoint update mismatch')
        networks=core.load_networks(meta['checkpoint_path'])
        require(core.parameter_hash(networks)==meta['parameter_sha256'],'Policy parameter identity mismatch')
        parts={}
        for part in PARTS:
            previous.process_status(directory/'status.json',status='running',seed=seed,checkpoint=step,part=part)
            part_dir=step_dir/part;part_dir.mkdir(exist_ok=False)
            case_spec=static['cases'][part]
            bank,natural_record=native_bank(networks,arrays[part],part_dir/'natural_bank.npz') if step==0 else saved_bank(arrays[part],meta['natural_banks'][part])
            context_check=check_context_packets(case_spec,bank)
            cross,probs=evaluate_cells(networks,arrays[part],bank,case_spec,'cross',part_dir/'cross.npz')
            metrics=save_metrics(case_spec,probs,part_dir/'cross_metrics.npz');del probs
            sham,_=evaluate_cells(networks,arrays[part],bank,case_spec,'sham',part_dir/'sham.npz')
            record=dict(natural_bank=natural_record,context_packet_check=context_check,cross=cross,sham=sham,metrics=metrics)
            write(part_dir/'result.json',record);parts[part]=record;del bank
            print(json.dumps(dict(stage='partition_complete',seed=seed,checkpoint=step,part=part)),flush=True)
        require(core.parameter_hash(networks)==meta['parameter_sha256'],'Intervention changed policy weights')
        record=dict(seed=seed,checkpoint=step,checkpoint_sha256=meta['checkpoint_sha256'],parameter_sha256=meta['parameter_sha256'],
            parts=parts,elapsed_seconds=time.perf_counter()-started)
        write(step_dir/'result.json',record);output.append(record)
    require(array_hashes=={p:{k:array_sha(a[k]) for k in ('packed_states','rewards','x_PL')} for p,a in arrays.items()},'World arrays changed')
    previous.process_status(directory/'status.json',status='completed',seed=seed,elapsed_seconds=time.perf_counter()-started)
    return output


def primary(records):
    require(len(records)==8,'Expected eight complete policy states');by={(r['seed'],r['checkpoint']):r for r in records}
    require(set(by)=={(s,t) for s in SEEDS for t in CHECKPOINTS},'Missing/duplicate policy states')
    rows=[]
    for seed in SEEDS:
        first=by[seed,0]['parts'][TARGET]['metrics'];final=by[seed,6000]['parts'][TARGET]['metrics']
        require(all(type(v[k]) in (int,float) and np.isfinite(v[k]) and -1<=v[k]<=1 for v in (first,final) for k in ('L','D')),'Invalid L or D')
        rows.append(dict(seed=seed,L_initial=first['L'],L_final=final['L'],L_final_minus_initial=final['L']-first['L'],
            D_initial=first['D'],D_final=final['D']))
    return dict(name=CONFIG['primary'],partition=TARGET,independent_societies=4,by_seed=rows,
        mean_L_final=float(np.mean([r['L_final'] for r in rows])),
        auxiliary=dict(mean_L_initial=float(np.mean([r['L_initial'] for r in rows])),
            mean_L_final_minus_initial=float(np.mean([r['L_final_minus_initial'] for r in rows])),
            mean_D_final=float(np.mean([r['D_final'] for r in rows])),mean_D_initial=float(np.mean([r['D_initial'] for r in rows]))))


def execute(out):
    out=Path(out).resolve();plan,static=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False)
    start=time.perf_counter();write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    previous.process_status(execution/'status.json',status='running',worker_count=4)
    try:
        payloads=[(s,[r for r in plan['policy_states'] if r['seed']==s],static,str(execution)) for s in SEEDS]
        with multiprocessing.get_context('spawn').Pool(4) as pool:groups=pool.map(worker,payloads)
        records=[r for g in groups for r in g]
        require([(r['seed'],r['checkpoint']) for r in records]==[(s,t) for s in SEEDS for t in CHECKPOINTS],'Noncanonical completed grid')
        hashes=[read(execution/f'seed_{s}'/'array_hashes.json') for s in SEEDS]
        require(all(h==hashes[0] for h in hashes),'Workers disagree on full domain');verify(out)
        measured=dict(new_natural_module_samples=sum(p['natural_bank']['new_module_samples'] for r in records for p in r['parts'].values()),
            intervention_module_samples=sum(p[m]['new_module_samples'] for r in records for p in r['parts'].values() for m in ('cross','sham')))
        measured['total_new_module_samples']=sum(measured.values())
        require(all(static['budget'][k]==v for k,v in measured.items()),'Measured forward budget mismatch')
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
