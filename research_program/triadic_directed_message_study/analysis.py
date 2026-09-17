"""JSON-only descriptions of a completed directed-message experiment.

No NPZ, checkpoint, model or production scoring module is imported or opened.
This summarizes stored records; the separate audit replays their derivation.
"""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import argparse,hashlib,json,math

import numpy as np

SEEDS=(59101,59102,59103,59104)
STEPS=(0,6000)
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
AXES=('kind','length','destination')
TARGET=PARTS[-1]
PRIMARY='mean_four_seed_final_complete_double_holdout_L'
METRICS=('L','D','mean_compatible_probability','mean_incompatible_probability','strict_positive_L_fraction')
COUNTS=('positive_L','zero_L','negative_L')
HERE=Path(__file__).resolve().parent


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def mean(values):return float(np.mean(values))


def primary(records):
    """Independent arithmetic in the frozen runner's declared seed order."""
    by={(r['seed'],r['checkpoint']):r for r in records}
    rows=[]
    for seed in SEEDS:
        initial=by[seed,0]['parts'][TARGET]['metrics'];final=by[seed,6000]['parts'][TARGET]['metrics']
        rows.append(dict(seed=seed,L_initial=initial['L'],L_final=final['L'],
            L_final_minus_initial=final['L']-initial['L'],D_initial=initial['D'],D_final=final['D']))
    return dict(name=PRIMARY,partition=TARGET,independent_societies=4,by_seed=rows,
        mean_L_final=mean([r['L_final'] for r in rows]),
        auxiliary=dict(mean_L_initial=mean([r['L_initial'] for r in rows]),
            mean_L_final_minus_initial=mean([r['L_final_minus_initial'] for r in rows]),
            mean_D_final=mean([r['D_final'] for r in rows]),mean_D_initial=mean([r['D_initial'] for r in rows])))


def validate_metric(record):
    metric=record['metrics'];strata=metric['strata']
    require(metric['complete_nine_strata'] and metric['empty_strata']==[],'Nine complete strata required')
    require([(s['sender'],s['axis_index'],s['axis']) for s in strata]==[(s,a,AXES[a]) for s,a in product(range(3),range(3))],
            'Unexpected stratum order')
    require(all(not s['empty'] and s['group_count']>0 for s in strata),'Empty structural stratum')
    for row in (metric,*strata):
        for key in METRICS:
            low=-1 if key in ('L','D') else 0
            require(type(row[key]) in (int,float) and math.isfinite(row[key]) and low-2e-12<=row[key]<=1+2e-12,
                    'Invalid stored probability or contrast')
    for key in METRICS:require(metric[key]==mean([s[key] for s in strata]),'Stored overall metric is not equal-nine-stratum mean')
    b=metric['n_backgrounds'];g=metric['group_count'];gb=g*b
    require(metric['group_background_count']==gb and metric['cross_cells']==gb*8,'Invalid metric counts')
    require(sum(s['group_count'] for s in strata)==g,'Stratum group counts incomplete')
    for s in strata:
        count=s['group_count']*b
        require(s['group_background_count']==count,'Stratum background count mismatch')
        require(sum(s[k+'_group_backgrounds'] for k in COUNTS)==count,'Stratum sign counts incomplete')
        require(math.isclose(s['strict_positive_L_fraction'],s['positive_L_group_backgrounds']/count,rel_tol=0,abs_tol=2e-15),
                'Invalid stratum positive fraction')
    for key in COUNTS:
        require(metric['raw_group_background_counts'][key]==sum(s[key+'_group_backgrounds'] for s in strata),
                'Raw sign counts do not match strata')
    require(record['cross']['rows']==gb*8 and record['sham']['rows']==gb*4,'Incomplete cross/sham cells')
    require(record['context_packet_check']['all_equal'] and record['context_packet_check']['compared_sender_packets']==gb*2,
            'Missing sender context invariance')
    sham=record['sham']['native_replay']
    require(sham['all_messages_equal'] and sham['all_actions_equal'] and sham['checked_rows']==gb*4 and
            0<=sham['max_probability_absolute_error']<=1e-13,'Natural sham replay did not pass')


def summarize(result):
    require(result.get('status')=='completed','Only completed results may be summarized')
    records=result['policy_states']
    require([(r['seed'],r['checkpoint']) for r in records]==list(product(SEEDS,STEPS)),
            'Expected eight policy states in frozen order, without duplicate or missing cells')
    by={(r['seed'],r['checkpoint']):r for r in records}
    for row in records:
        require(set(row['parts'])==set(PARTS),'All four partitions required')
        for part in PARTS:validate_metric(row['parts'][part])
    main=primary(records)
    require(main==result['primary'],'Unique primary differs from completed main result')
    partitions=[];strata_rows=[];paired=[];marginals=[]
    for step,part in product(STEPS,PARTS):
        selected=[by[s,step]['parts'][part]['metrics'] for s in SEEDS]
        seed_rows=[dict(seed=s,**{k:m[k] for k in METRICS},raw_group_background_counts=deepcopy(m['raw_group_background_counts']),
            raw_pooled_positive_L_fraction=m['raw_group_background_counts']['positive_L']/m['group_background_count'])
            for s,m in zip(SEEDS,selected)]
        partitions.append(dict(checkpoint=step,partition=part,by_seed=seed_rows,
            means={k:mean([m[k] for m in selected]) for k in METRICS},
            negative_L_seeds=[r['seed'] for r in seed_rows if r['L']<0],
            positive_L_seeds=[r['seed'] for r in seed_rows if r['L']>0],
            positive_D_but_nonpositive_L_seeds=[r['seed'] for r in seed_rows if r['D']>0 and r['L']<=0]))
        for index,(sender,axis) in enumerate(product(range(3),range(3))):
            selected_strata=[m['strata'][index] for m in selected]
            strata_rows.append(dict(checkpoint=step,partition=part,sender=sender,axis=AXES[axis],axis_index=axis,
                by_seed=[dict(seed=seed,**deepcopy(s)) for seed,s in zip(SEEDS,selected_strata)],
                means={k:mean([s[k] for s in selected_strata]) for k in METRICS},
                mean_four_deltas=np.mean([s['mean_four_deltas'] for s in selected_strata],axis=0).tolist()))
    for seed,part in product(SEEDS,PARTS):
        initial=by[seed,0]['parts'][part]['metrics'];final=by[seed,6000]['parts'][part]['metrics']
        paired.append(dict(seed=seed,partition=part,initial={k:initial[k] for k in METRICS},final={k:final[k] for k in METRICS},
            final_minus_initial={k:final[k]-initial[k] for k in METRICS}))
    for step,dimension,level in product(STEPS,('sender','axis'),range(3)):
        seed_rows=[]
        for seed in SEEDS:
            selected=[s for s in by[seed,step]['parts'][TARGET]['metrics']['strata'] if s[dimension if dimension=='sender' else 'axis_index']==level]
            require(len(selected)==3,'Three equally weighted strata in each marginal')
            seed_rows.append(dict(seed=seed,**{k:mean([s[k] for s in selected]) for k in METRICS}))
        marginals.append(dict(checkpoint=step,partition=TARGET,dimension=dimension,level=level,
            label=AXES[level] if dimension=='axis' else str(level),by_seed=seed_rows,
            means={k:mean([s[k] for s in seed_rows]) for k in METRICS}))
    costs=dict(new_natural_module_samples=sum(p['natural_bank']['new_module_samples'] for r in records for p in r['parts'].values()),
        intervention_module_samples=sum(p[m]['new_module_samples'] for r in records for p in r['parts'].values() for m in ('cross','sham')))
    costs['total_new_module_samples']=sum(costs.values())
    require(costs==result['measured_module_samples'],'Stored measured forward budget mismatch')
    require(all(result['budget'][k]==v for k,v in costs.items()),'Measured budget differs from prepared budget')
    return dict(status='completed_json_only_summary',primary=main,primary_exact_match=True,
        counts=dict(independent_societies=4,policy_states=8,partition_records=32,stratum_records=288,
            time_partition_summaries=8,paired_seed_partitions=16,target_time_marginals=12),
        time_partition_summaries=partitions,all_time_partition_strata=strata_rows,
        paired_seed_partitions=paired,target_sender_and_axis_marginals=marginals,
        policy_states=deepcopy(records),budget=deepcopy(result['budget']),measured_module_samples=costs,
        source_elapsed_seconds=result.get('elapsed_seconds'),
        scope=dict(input='Completed JSON and bound plan/prepared/freeze JSON only; no NPZ or model access.',
            validation='Stored JSON arithmetic and completeness checks. Does not replay neural outputs, physical settlement or individual group/background arrays.',
            units='Probabilities and probability differences on their original 0..1 or -1..1 scales. Multiply differences by 100 only to display percentage points.',
            main='Final complete double-holdout L, mean of four independent policy seeds; D and final-minus-initial L remain auxiliary.',
            positive_fraction='Equal-nine-stratum average of positive-L group/background fractions; raw pooled fractions are explicitly separate.',
            marginals='Within seed, equal mean over the three sender or axis strata; then equal mean of four seeds.',
            repeated_measurements='Groups, backgrounds, recipients and both checkpoints are not new independent societies.',
            physical_records='Cross/sham physical summaries use repeated intervention cells, not natural full-world task denominators. Reused final natural records may omit a physical summary.',
            interpretation='Positive aggregate L does not imply all four effects positive in every group; negative L limits this stronger W1 cross-layout recipient-selectivity test, not all communication or language.'))


def execute(run,output='analysis_001'):
    run=Path(run).resolve();destination=Path(output)
    if not destination.is_absolute():destination=run/destination
    require(not destination.exists(),'Refusing to overwrite analysis output')
    paths={name:run/name for name in ('plan.json','prepared.json','freeze.json','execution/results.json')}
    result=read(paths['execution/results.json'])
    require(result.get('status')=='completed','Only completed results may be opened for analysis')
    plan=read(paths['plan.json']);prepared=read(paths['prepared.json']);freeze=read(paths['freeze.json'])
    require(sha(paths['plan.json'])==freeze['plan_sha256']==result['plan_sha256'],'Completed plan hash mismatch')
    require(sha(paths['prepared.json'])==freeze['prepared_sha256']==plan['prepared_sha256'],'Prepared hash mismatch')
    require(result['budget']==prepared['budget'],'Frozen prepared budget mismatch')
    require(plan['config']['primary']==PRIMARY and plan['config']['seeds']==list(SEEDS) and
            plan['config']['checkpoints']==list(STEPS) and plan['config']['partitions']==list(PARTS),'Unexpected frozen design')
    summary=summarize(result)
    source_hashes={str(p):sha(p) for p in (*paths.values(),HERE/'analysis.py',HERE/'test_analysis.py')}
    destination.mkdir(parents=True,exist_ok=False);write(destination/'summary.json',summary)
    receipt=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),inputs_sha256=source_hashes,
        outputs_sha256={str(destination/'summary.json'):sha(destination/'summary.json')},primary_exact_match=True,
        counts=summary['counts'],npz_reads=0,checkpoint_loads=0,model_forwards=0,training_updates=0,
        scope=summary['scope'])
    write(destination/'receipt.json',receipt)
    return dict(status='passed',output=str(destination),summary_sha256=sha(destination/'summary.json'),primary=summary['primary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/directed_001')
    parser.add_argument('--output',default='analysis_001');args=parser.parse_args()
    print(json.dumps(execute(args.run,args.output),ensure_ascii=False))
