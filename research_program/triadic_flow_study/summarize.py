"""JSON-only descriptions of a completed, independently audited flow study.

No NPZ, checkpoint, model, environment or production scoring module is opened.
Stored response arithmetic is checked; the independent audit replays its origin.
"""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import argparse,hashlib,json

import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(59101,59102,59103,59104)
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
TARGET=PARTS[-1]
ARMS=('S00','S01','S10','S11')
CONTRASTS=('I_given_O0','I_given_O1','O_given_I0','O_given_I1','interaction')
VALUES=ARMS+CONTRASTS
SHAPES=dict(S=(),expected_reward=(),execution=(4,),partner=(3,4),site=(3,5),material=(3,5),
            kind=(3,3),length=(3,3),destination=(3,3),changed_W1_symbols=(2,))
SEMANTIC=('partner','site','material','kind','length','destination')
PRIMARY='mean_four_seed_complete_double_holdout_exact_S10_minus_S00'
AXES=('kind','length','destination')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def number(value):
    value=np.asarray(value)
    return float(value) if value.ndim==0 else value.tolist()


def average(values):return number(np.mean(values,axis=0))


def close_tree(left,right):
    if isinstance(left,dict):
        require(isinstance(right,dict) and set(left)==set(right),'Audit primary keys mismatch')
        for key in left:close_tree(left[key],right[key])
    elif isinstance(left,list):
        require(isinstance(right,list) and len(left)==len(right),'Audit primary rows mismatch')
        for a,b in zip(left,right):close_tree(a,b)
    elif type(left) in (int,float):
        require(type(right) in (int,float) and np.isfinite(right) and abs(left-right)<=2e-12,'Audit primary number mismatch')
    else:require(left==right,'Audit primary label mismatch')


def primary(records):
    by={r['seed']:r for r in records};rows=[]
    for seed in SEEDS:
        block=by[seed]['parts'][TARGET]['metrics']['S']
        rows.append(dict(seed=seed,S10_minus_S00=block['I_given_O0'],**{k:block[k] for k in ARMS}))
    return dict(name=PRIMARY,partition=TARGET,response='S',contrast='I_given_O0',independent_societies=4,
        by_seed=rows,mean_S10_minus_S00=float(np.mean([r['S10_minus_S00'] for r in rows])),
        interpretation='Conditional exact action full-success probability given recomputed greedy messages; signed incoming effect with outgoing unmodified.')


def validate_block(block,response):
    shape=SHAPES[response];strata=block['strata']
    require(block['response_shape']==list(shape),'Response axes changed')
    require(block['complete_nine_strata'] and block['empty_strata']==[] and len(strata)==9,'Nine complete strata required')
    require([(r['sender'],r['axis_index'],r['axis']) for r in strata]==[(s,a,AXES[a]) for s,a in product(range(3),range(3))],
            'Canonical strata missing')
    maximum=8 if response=='changed_W1_symbols' else 1
    for row in (block,*strata):
        for key in VALUES:
            value=np.asarray(row[key]);low=0 if key in ARMS else -maximum*(2 if key=='interaction' else 1)
            high=maximum*(2 if key=='interaction' else 1)
            require(value.shape==shape and value.dtype.kind in 'fiu' and np.isfinite(value).all() and
                    np.all(value>=low-2e-13) and np.all(value<=high+2e-13),'Invalid response value/shape')
        pairs=dict(I_given_O0=np.asarray(row['S10'])-row['S00'],I_given_O1=np.asarray(row['S11'])-row['S01'],
                   O_given_I0=np.asarray(row['S01'])-row['S00'],O_given_I1=np.asarray(row['S11'])-row['S10'])
        pairs['interaction']=np.asarray(row['I_given_O1'])-row['I_given_O0']
        for key,value in pairs.items():require(np.allclose(row[key],value,rtol=0,atol=2e-13),'Stored paired contrast arithmetic mismatch')
    for key in VALUES:
        require(np.array_equal(np.asarray(block[key]),np.mean([s[key] for s in strata],axis=0)),
                'Stored top response is not exact nine-stratum mean')
    require(all(not s['empty'] and s['group_count']>0 for s in strata),'Unexpected empty stratum')
    require(sum(s['group_count'] for s in strata)==block['group_count'],'Incomplete group counts')
    require(all(s['group_background_count']==s['group_count']*block['n_backgrounds'] for s in strata),'Stratum background count mismatch')
    require(block['flow_cells']==16*block['group_background_count']==16*block['group_count']*block['n_backgrounds'] and
            block['arm_cells']==4*block['group_background_count'],'Flow and arm count mismatch')
    if response in SEMANTIC or response=='execution':
        for key in ARMS:require(np.allclose(np.sum(block[key],axis=-1),1,rtol=0,atol=2e-13),'Probability mass lost')
    if response=='partner':
        for key in ARMS:require(np.all(np.asarray(block[key])[np.arange(3),1+np.arange(3)]==0),'Self-partner probability must remain zero')
    if response=='changed_W1_symbols':
        for key in ARMS:
            require((key[1]!='0' or block[key][0]==0) and (key[2]!='0' or block[key][1]==0),
                    'An inactive direction cannot change symbols')


def summarize(result):
    require(result.get('status')=='completed','Only completed main results can be summarized')
    records=result['policy_states']
    require([(r['seed'],r['checkpoint']) for r in records]==[(s,6000) for s in SEEDS],'Four canonical final policies required')
    by={r['seed']:r for r in records}
    for policy in records:
        require(set(policy['parts'])==set(PARTS),'All four partitions required')
        for part in PARTS:
            record=policy['parts'][part];blocks=record['metrics']
            require(set(blocks)==set(SHAPES),'Missing or unexpected scientific response')
            for response,block in blocks.items():validate_block(block,response)
            reference=blocks['S']
            require(all((b['group_count'],b['n_backgrounds'],b['flow_cells'])==
                        (reference['group_count'],reference['n_backgrounds'],reference['flow_cells']) for b in blocks.values()),
                    'Responses do not use the same complete case grid')
            require(record['flow']['rows']==reference['flow_cells'] and record['flow']['new_module_samples']==6*reference['flow_cells'],
                    'Incomplete flow forward budget')
            sham=record['flow']['native_replay']
            require(sham['all_messages_equal'] and sham['all_actions_equal'] and sham['checked_rows']==reference['arm_cells'] and
                    0<=sham['max_probability_absolute_error']<=1e-13,'Sham did not reproduce the natural trace')
            require(record['natural_bank']['reused'] and record['natural_bank']['new_module_samples']==0,'Natural bank must be reused')
    main=primary(records);require(main==result['primary'],'Unique primary differs from main result')
    partitions=[];strata_rows=[];arm_records=[];contrast_records=[]
    for part in PARTS:
        comparison={}
        for response in SHAPES:
            selected=[by[seed]['parts'][part]['metrics'][response] for seed in SEEDS]
            comparison[response]=dict(response_shape=list(SHAPES[response]),
                by_seed=[dict(seed=seed,**{k:deepcopy(block[k]) for k in VALUES}) for seed,block in zip(SEEDS,selected)],
                means={k:average([block[k] for block in selected]) for k in VALUES})
        partitions.append(dict(partition=part,responses=comparison))
        for index,(sender,axis) in enumerate(product(range(3),range(3))):
            responses={}
            for response in SHAPES:
                chosen=[by[s]['parts'][part]['metrics'][response]['strata'][index] for s in SEEDS]
                responses[response]=dict(by_seed=[dict(seed=s,**deepcopy(b)) for s,b in zip(SEEDS,chosen)],
                    means={k:average([b[k] for b in chosen]) for k in VALUES})
            strata_rows.append(dict(partition=part,sender=sender,axis=AXES[axis],axis_index=axis,responses=responses))
        for seed in SEEDS:
            blocks=by[seed]['parts'][part]['metrics']
            for arm in ARMS:
                responses={name:deepcopy(block[arm]) for name,block in blocks.items()}
                arm_records.append(dict(seed=seed,partition=part,arm=arm,incoming=int(arm[1]),outgoing=int(arm[2]),
                    responses=responses,any_execution_probability=1-responses['execution'][0]))
            for contrast in CONTRASTS:
                contrast_records.append(dict(seed=seed,partition=part,contrast=contrast,
                    responses={name:deepcopy(block[contrast]) for name,block in blocks.items()}))
    measured=dict(new_natural_module_samples=0,
        intervention_module_samples=sum(p['flow']['new_module_samples'] for r in records for p in r['parts'].values()))
    measured['total_new_module_samples']=sum(measured.values())
    require(measured==result['measured_module_samples'],'Measured budget mismatch')
    require(all(result['budget'][k]==v for k,v in measured.items()),'Frozen budget mismatch')
    cells=sum(p['flow']['rows'] for r in records for p in r['parts'].values())
    sham_cells=sum(p['flow']['native_replay']['checked_rows'] for r in records for p in r['parts'].values())
    require(cells==result['budget']['flow_rows'] and sham_cells==result['budget']['sham_rows'],'Case row budget mismatch')
    return dict(status='completed_json_only_summary',primary=main,primary_exact_match=True,
        counts=dict(independent_societies=4,policy_states=4,partition_records=16,seed_partition_arms=64,
            seed_partition_contrasts=80,response_types=10,seed_partition_response_strata=1440,flow_rows=cells,sham_rows=sham_cells),
        response_schema=dict(actor_order=['focal','u','v'],u_v='Other actor identities in ascending order',
            partner=['wait','focal','u','v'],site=['wait',0,1,2,3],material=['wait',0,1,2,3],
            kind=['wait',0,1],length=['wait',0,1],destination=['wait',0,1],execution=['none','AB','AC','BC'],
            changed_W1_symbols=['incoming','outgoing']),
        partition_summaries=partitions,all_partition_strata=strata_rows,arm_records=arm_records,contrast_records=contrast_records,
        policy_states=deepcopy(records),budget=deepcopy(result['budget']),measured_module_samples=measured,
        source_elapsed_seconds=result.get('elapsed_seconds'),
        scope=dict(input='Completed main JSON plus plan/prepared/freeze and passed audit JSON only. No array or checkpoint files read.',
            validation='Stored arithmetic, response axes, full grid, exact main primary and source hashes; independent audit handles model/NPZ replay.',
            primary='Case-weighted original-world conditional exact full-success probability S, I_given_O0, complete double holdout; the other responses never replace it.',
            S='Conditional on each intervention cell greedy messages, integrates all action proposals; unmatched third actor integrates to one.',
            weights='Paired within world before means; contexts/endpoints, backgrounds, within-stratum groups, nine strata and four societies have prescribed weights.',
            natural_reference='S00 uses repeated case weights; it is not the old uniform full-world natural success rate.',
            semantic='Six proposal probability distributions retain wait; actors are focal/u/v, self-partner mass zero. Not semantic correctness or word meaning.',
            symbols='Changed W1 symbols count delivered directed positions; outgoing duplicates the same packet on two edges. Not comparable intervention dose or a divisor for effects.',
            physical_summary='Stored flow.physical_summary is pooled over all sixteen cells, including all four arms; it is not an arm-specific greedy contrast or uniform native-world score.',
            units='Probabilities and signed probability differences remain on their original scale; plotted probability levels use percent, differences use percentage points.',
            interpretation='Intervention consequences locate functional paths; I/O effects do not identify centralized control, demand semantics or language formation.'))


def execute(run,output='summary_001',audit='audit_execution_001/verification.json'):
    run=Path(run).resolve();destination=Path(output);destination=destination if destination.is_absolute() else run/destination
    audit_path=Path(audit);audit_path=audit_path if audit_path.is_absolute() else run/audit_path
    require(not destination.exists(),'Refusing to overwrite summary')
    result_path=run/'execution/results.json';result=read(result_path);verification=read(audit_path)
    require(result.get('status')=='completed' and verification.get('status')=='passed','Completed main and passed audit required')
    paths=[run/'plan.json',run/'prepared.json',run/'freeze.json'];plan,prepared,freeze=map(read,paths)
    require(sha(paths[0])==freeze['plan_sha256']==result['plan_sha256']==verification['plan_sha256'],'Plan binding mismatch')
    require(sha(paths[1])==freeze['prepared_sha256']==plan['prepared_sha256'],'Prepared binding mismatch')
    require(result['budget']==prepared['budget'],'Prepared budget mismatch')
    require(plan['config']['primary']==PRIMARY and plan['config']['seeds']==list(SEEDS) and
            plan['config']['checkpoints']==[6000] and plan['config']['partitions']==list(PARTS),'Unexpected fixed design')
    summary=summarize(result)
    require(verification['artifacts_sha256'][str(result_path)]==sha(result_path),'Audit main-result binding mismatch')
    close_tree({k:v for k,v in summary['primary'].items() if k!='interpretation'},verification['primary'])
    summary['audit_status']='passed';summary['audit_primary']=deepcopy(verification['primary'])
    destination.mkdir(parents=True,exist_ok=False);write(destination/'summary.json',summary)
    receipt=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(p):sha(p) for p in (*paths,result_path,audit_path,HERE/'summarize.py',HERE/'test_summarize.py')},
        outputs_sha256={str(destination/'summary.json'):sha(destination/'summary.json')},counts=summary['counts'],
        primary_exact_match=True,npz_reads=0,checkpoint_loads=0,model_forwards=0,training_updates=0,scope=summary['scope'])
    write(destination/'receipt.json',receipt)
    return dict(status='passed',output=str(destination),summary_sha256=sha(destination/'summary.json'),primary=summary['primary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/flow_001')
    parser.add_argument('--output',default='summary_001');parser.add_argument('--audit',default='audit_execution_001/verification.json')
    args=parser.parse_args();print(json.dumps(execute(args.run,args.output,args.audit),ensure_ascii=False))
