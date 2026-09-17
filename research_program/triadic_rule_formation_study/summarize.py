"""Completed JSON-only formation summaries; no model, NPZ or metric imports."""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import argparse,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=tuple(range(60101,60117));STEPS=(0,100,500,1500,3000,6000)
CONDITIONS=('strict_PL_live','strict_PL_silent','reciprocal_PL_live','reciprocal_PL_silent')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts');TARGET=PARTS[-1]
VIEWS=('native','common_reciprocal');RESPONSES=('Q','Q_shuffle','Q_excess')
TASK=('reward_mean','full_success_rate','physical_execution_rate','executed_partner_correct_rate','proposal_role_success_rate',
      'kind_correct_rate','length_correct_rate','destination_correct_rate','material_identity_correct_rate')
MEASURES={'native_Q':('native','Q'),'native_Q_excess':('native','Q_excess'),
          'common_reciprocal_Q':('common_reciprocal','Q'),'common_reciprocal_Q_excess':('common_reciprocal','Q_excess')}
SCALARS=('raw_AUC','centered_AUC','baseline_DiD','endpoint_DiD','endpoint_centered_DiD')
T_CRITICAL=2.1314495455597715
TRANSITIONS=('packet_equal_rate','token_hamming_fraction','global_ARI','mean_background_ARI',
             'conditional_entropy_bits_before','conditional_entropy_bits_after')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def stats(values):
    values=np.asarray(values,float);require(values.shape==(16,) and np.isfinite(values).all(),'Sixteen finite societies required')
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1));se=sd/4
    return dict(n=16,mean=mean,sample_sd=sd,standard_error=se,df=15,t_critical=T_CRITICAL,interval_level=.95,
        ci95_lower=mean-T_CRITICAL*se,ci95_upper=mean+T_CRITICAL*se,
        interval_method='Approximate two-sided Student-t interval across16 paired initializations;not a world-level interval or equivalence proof.')


def paired_primary(runs):
    """Independent stored-JSON arithmetic, preserving the frozen reduction order."""
    by={(r['seed'],r['condition']):r for r in runs};rows=[]
    widths=np.diff(np.asarray(STEPS,dtype=float))
    for seed in SEEDS:
        cells={};measures={}
        for rule,live in product(('strict','reciprocal'),(False,True)):
            condition=f'{rule}_PL_{"live" if live else "silent"}';run=by[seed,condition]
            cells[condition]=dict(rule=rule,live=live,**{name:[r['evaluation']['need_response'][view][field] for r in run['trajectory']]
                for name,(view,field) in MEASURES.items()})
        for name in MEASURES:
            strict=np.asarray(cells['strict_PL_live'][name])-cells['strict_PL_silent'][name]
            reciprocal=np.asarray(cells['reciprocal_PL_live'][name])-cells['reciprocal_PL_silent'][name]
            did=reciprocal-strict;centered=did-did[0]
            measures[name]=dict(communication_strict=strict.tolist(),communication_reciprocal=reciprocal.tolist(),DiD=did.tolist(),
                centered_DiD=centered.tolist(),baseline_DiD=float(did[0]),
                raw_AUC=float(np.sum(widths*(did[:-1]+did[1:])*.5)/6000),
                centered_AUC=float(np.sum(widths*(centered[:-1]+centered[1:])*.5)/6000),
                endpoint_DiD=float(did[-1]),endpoint_centered_DiD=float(centered[-1]))
        rows.append(dict(seed=seed,cells=cells,measures=measures))
    main=stats([r['measures']['native_Q']['centered_AUC'] for r in rows])
    return dict(statistics=main,mean_centered_AUC=main['mean'],by_seed=rows,
        auxiliary_statistics={name:{field:stats([r['measures'][name][field] for r in rows]) for field in SCALARS} for name in MEASURES})


def evaluation_group(evaluations):
    out={}
    for view in VIEWS:
        rows=[]
        for seed,e in zip(SEEDS,evaluations):
            response=e['need_response'][view]
            require(response['complete_nine_strata'] and not response['empty_strata'] and len(response['strata'])==9,'Incomplete Q strata')
            require([(s['changed_person'],s['axis_index']) for s in response['strata']]==list(product(range(3),range(3))),'Q stratum order changed')
            rows.append(dict(seed=seed,**{k:response[k] for k in RESPONSES},**{k:e[view][k] for k in TASK}))
        keys=RESPONSES+TASK
        out[view]=dict(by_seed=rows,means={k:float(np.mean([r[k] for r in rows])) for k in keys},
            strata=[dict(changed_person=person,axis_index=axis,
                by_seed=[dict(seed=seed,**deepcopy(e['need_response'][view]['strata'][person*3+axis])) for seed,e in zip(SEEDS,evaluations)],
                means={k:float(np.mean([e['need_response'][view]['strata'][person*3+axis][k] for e in evaluations])) for k in RESPONSES})
                for person,axis in product(range(3),range(3))])
    return out


def summarize(result):
    require(result.get('status')=='completed','Completed main results required')
    runs=result['runs'];require([(r['seed'],r['condition']) for r in runs]==list(product(SEEDS,CONDITIONS)),'Canonical64-run grid required')
    by={(r['seed'],r['condition']):r for r in runs};actual=[]
    for run in runs:
        require(run['rule']==run['condition'].split('_')[0] and run['live']==run['condition'].endswith('_live'),'Condition metadata mismatch')
        require([r['update'] for r in run['trajectory']]==list(STEPS) and set(run['final'])==set(PARTS),'Incomplete evaluation grid')
        final=run['final'][TARGET];endpoint=run['trajectory'][-1]['evaluation']
        require(final['alias_of']=='trajectory_update_6000' and final['additional_forward_module_samples']==0 and
                {k:v for k,v in final.items() if k not in ('alias_of','additional_forward_module_samples')}==endpoint,'Final target alias changed')
        actual.extend(r['evaluation'] for r in run['trajectory']);actual.extend(run['final'][p] for p in PARTS if p!=TARGET)
        for index,row in enumerate(run['trajectory']):
            snapshot=row['message_snapshot'];transition=row['message_transition']
            require(snapshot['background_count']==36 and len(snapshot['panels'])==6,'Missing target message snapshot panels')
            require([(p['actor'],p['window']) for p in snapshot['panels']]==list(product(range(3),range(2))),'Message panel order changed')
            require(all(len(p['backgrounds'])==36 for p in snapshot['panels']),'Missing snapshot backgrounds')
            if index==0:require(transition is None,'Step0 cannot have an adjacent transition')
            else:
                require(transition['background_count']==36 and len(transition['panels'])==6 and
                    [(p['actor'],p['window']) for p in transition['panels']]==list(product(range(3),range(2))) and
                    all(len(p['backgrounds'])==36 for p in transition['panels']),'Missing adjacent transition panels/backgrounds')
    require(len(actual)==576 and len({e['path'] for e in actual})==576,'Duplicate or missing actual evaluations')
    computed=paired_primary(runs)
    for key,value in computed.items():require(value==result['primary'][key],'Independent primary arithmetic differs from main')
    require(result['primary']['partition']==TARGET and result['primary']['primary_measure']=='native_Q' and
            result['primary']['primary_statistic']=='centered_AUC','Unique primary changed')
    trajectory=[];endpoints=[];entropy=[];transitions=[]
    for condition in CONDITIONS:
        for index,update in enumerate(STEPS):
            trajectory.append(dict(condition=condition,update=update,partition=TARGET,
                scoring=evaluation_group([by[s,condition]['trajectory'][index]['evaluation'] for s in SEEDS])))
        for part in PARTS:endpoints.append(dict(condition=condition,partition=part,
            scoring=evaluation_group([by[s,condition]['final'][part] for s in SEEDS])))
        for panel,(actor,window) in enumerate(product(range(3),range(2))):
            erows=[];trows=[]
            for seed in SEEDS:
                saved=by[seed,condition]['trajectory']
                erows.append(dict(seed=seed,conditional_entropy_bits=[r['message_snapshot']['panels'][panel]['conditional_entropy_bits'] for r in saved],
                    conditional_effective_packet_count=[r['message_snapshot']['panels'][panel]['conditional_effective_packet_count'] for r in saved]))
                trows.append(dict(seed=seed,**{k:[r['message_transition']['panels'][panel][k] for r in saved[1:]] for k in TRANSITIONS}))
            entropy.append(dict(condition=condition,actor=actor,window=window,updates=list(STEPS),by_seed=erows,
                means={k:np.mean([r[k] for r in erows],axis=0).tolist() for k in ('conditional_entropy_bits','conditional_effective_packet_count')}))
            transitions.append(dict(condition=condition,actor=actor,window=window,time_pairs=[list(p) for p in zip(STEPS[:-1],STEPS[1:])],
                by_seed=trows,means={k:np.mean([r[k] for r in trows],axis=0).tolist() for k in TRANSITIONS}))
    measured=dict(training_forward_module_samples=sum(r['updates'] for r in runs)*256*2*9,
        evaluation_forward_module_samples=sum(e['forward_module_samples'] for e in actual),checkpoints=384,evaluation_files=576,
        evaluation_worlds=sum(e['worlds'] for e in actual),final_target_aliases=64)
    require(measured==result['measured_budget'] and all(result['budget'][k]==v for k,v in measured.items()),'Budget mismatch')
    interactions={name:dict(by_seed=[dict(seed=r['seed'],**deepcopy(r['measures'][name])) for r in computed['by_seed']],
        mean_DiD=np.mean([r['measures'][name]['DiD'] for r in computed['by_seed']],axis=0).tolist(),
        mean_centered_DiD=np.mean([r['measures'][name]['centered_DiD'] for r in computed['by_seed']],axis=0).tolist()) for name in MEASURES}
    return dict(status='completed_json_only_summary',primary=deepcopy(result['primary']),primary_exact_match=True,
        counts=dict(independent_societies=16,runs=64,target_trajectory_records=384,other_final_records=192,actual_evaluations=576,
            target_final_aliases=64,message_snapshots=384,adjacent_message_transitions=320,panels_per_message_record=6),
        target_trajectory_summaries=trajectory,complete_endpoint_summaries=endpoints,trajectory_interactions=interactions,
        message_entropy_curves=entropy,message_transition_curves=transitions,runs=runs,
        budget=result['budget'],measured_budget=measured,source_elapsed_seconds=result.get('elapsed_seconds'),
        scope=dict(input='Completed JSON only; no NPZ, checkpoint or model access.',
            raw='All64 original runs and their complete saved records retained, including every message panel/background and full task/content/Q stratum.',
            primary='Native-rule Q baseline-centered communication-by-rule DiD time AUC; raw/common-reciprocal/Qexcess/endpoints auxiliary.',
            inference='One approximate t15 interval across16 paired initialization blocks; no world-level p-values or inferred effect threshold.',
            messages='Six actor/window panels remain separate. Entropy/effective code count/ARI/raw changes describe natural packets,not meaning or language.',
            effective_mean='Across-seed effective packet curve averages each seed conditional effective count; it is not2 raised to the across-seed mean entropy.',
            aliases='Target final reuses existing6000 full evaluation,not an extra forward or independent sample.'))


def compare_audit(left,right):
    if isinstance(left,dict):
        require(isinstance(right,dict),'Audit structure mismatch')
        for key,value in left.items():
            if key in ('interval_method','definition','unit','scope'):continue
            require(key in right,'Missing audit field:'+key)
            compare_audit(value,right[key])
    elif isinstance(left,list):
        require(isinstance(right,list) and len(left)==len(right),'Audit array mismatch')
        for a,b in zip(left,right):compare_audit(a,b)
    elif type(left) in (int,float):require(type(right) in (int,float) and np.isfinite(right) and abs(left-right)<=2e-12,'Audit number mismatch')
    else:require(left==right,'Audit label mismatch')


def execute(run,output='summary_001',audit='audit_execution_001/verification.json'):
    run=Path(run).resolve();destination=Path(output);destination=destination if destination.is_absolute() else run/destination
    audit_path=Path(audit);audit_path=audit_path if audit_path.is_absolute() else run/audit_path
    require(not destination.exists(),'Refusing overwrite')
    main_path=run/'execution/results.json';main=read(main_path);verification=read(audit_path)
    require(main.get('status')=='completed' and verification.get('status')=='passed','Completed main and passed audit required')
    paths=[run/'plan.json',run/'prepared.json',run/'freeze.json'];plan,prepared,freeze=map(read,paths)
    require(sha(paths[0])==main['plan_sha256']==verification['plan_sha256']==freeze['plan_sha256'],'Plan hash mismatch')
    require(sha(paths[1])==plan['prepared_sha256']==freeze['prepared_sha256'],'Prepared hash mismatch')
    require(verification['artifacts_sha256'][str(main_path)]==sha(main_path),'Audit result binding mismatch')
    require(prepared['budget']==main['budget'],'Frozen budget mismatch')
    summary=summarize(main);compare_audit(summary['primary'],verification['primary'])
    summary.update(audit_status='passed',audit_primary=verification['primary'])
    destination.mkdir(parents=True,exist_ok=False);write(destination/'summary.json',summary)
    receipt=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),inputs_sha256={str(p):sha(p) for p in
        (*paths,main_path,audit_path,HERE/'summarize.py',HERE/'test_summarize.py')},
        outputs_sha256={str(destination/'summary.json'):sha(destination/'summary.json')},primary_exact_match=True,
        counts=summary['counts'],npz_reads=0,checkpoint_loads=0,model_forwards=0,scope=summary['scope'])
    write(destination/'receipt.json',receipt)
    return dict(status='passed',output=str(destination),summary_sha256=sha(destination/'summary.json'),statistics=summary['primary']['statistics'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/formation_001')
    parser.add_argument('--output',default='summary_001');parser.add_argument('--audit',default='audit_execution_001/verification.json')
    args=parser.parse_args();print(json.dumps(execute(args.run,args.output,args.audit),ensure_ascii=False))
