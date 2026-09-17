"""Descriptive aggregation of completed decoder JSON records; no model imports.

This is not a probability replay or independent accuracy audit. It validates
stored metric arithmetic, complete coverage, aliases, and source identities.
All actual evaluations and every seed/probe/payoff layer remain in the output.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

import numpy as np

OLD_SEEDS=(53101,53102,53103,53104)
PROBE_SEEDS=(55101,55102,55103)
PAYOFFS=('a50','a10')
VIEWS=('Own','FI','Initial','Final')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
PATTERNS=tuple(product(range(3),repeat=3))
TARGET='new_needs_and_layouts'
SCALARS=('joint_accuracy','indiv_accuracy','cross_entropy','expected_joint_correct_probability')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a,b):return bool(np.allclose(a,b,atol=2e-12,rtol=0))


def equal_tree(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        return isinstance(b,(int,float)) and not isinstance(b,bool) and close(a,b)
    return a==b


def job_id(probe,view,seed=None,payoff=None):
    value=f'probe_{probe}_{view}'
    if seed is not None:value+=f'_protocol_{seed}'
    if payoff is not None:value+='_'+payoff
    return value


def logical_mapping():
    return [dict(old_seed=seed,payoff=payoff,probe_seed=probe,view=view,
                 actual_job_id=job_id(probe,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None))
            for seed in OLD_SEEDS for payoff in PAYOFFS for probe in PROBE_SEEDS for view in VIEWS]


def bounds(part):
    return 23/62 if part in ('new_needs','new_needs_and_layouts') else 185/486


def checked_record(record,worlds):
    require(type(worlds) is int and worlds>0 and record['worlds']==worlds,'Evaluation world count mismatch')
    out={key:float(record[key]) for key in SCALARS}
    require(all(math.isfinite(x) and x>=0 for x in out.values()),'Invalid evaluation metric')
    for key in SCALARS:
        if key!='cross_entropy':require(out[key]<=1,'Probability/accuracy exceeds1')
    actor=np.asarray(record['per_actor_accuracy'],dtype=float)
    require(actor.shape==(3,) and np.isfinite(actor).all() and np.all((actor>=0)&(actor<=1)),'Invalid per-actor accuracy')
    require(close(actor.mean(),out['indiv_accuracy']) and out['joint_accuracy']<=actor.min()+2e-12,'Individual/joint accuracy mismatch')
    joint_count=out['joint_accuracy']*worlds
    require(abs(joint_count-round(joint_count))<1e-6,'Joint accuracy does not use stated denominator')
    strata=record['true_pair_strata'];require(set(strata)=={'AB','AC','BC'},'Missing true-pair stratum')
    total=0;correct=0
    out_strata={}
    for pair in ('AB','AC','BC'):
        n=strata[pair]['worlds'];v=float(strata[pair]['joint_accuracy'])
        require(type(n) is int and n>0 and math.isfinite(v) and 0<=v<=1,'Invalid true-pair stratum')
        require(abs(n*v-round(n*v))<1e-6,'Stratum accuracy is not an integer count')
        total+=n;correct+=n*v;out_strata[pair]=dict(worlds=n,joint_accuracy=v)
    require(total==worlds and close(correct/worlds,out['joint_accuracy']),'True-pair weighted accuracy mismatch')
    counts={pattern:0 for pattern in PATTERNS};seen=set()
    for row in record['predicted_joint_role_counts']:
        pattern=tuple(row['labels']);n=row['worlds']
        require(len(pattern)==3 and all(type(v) is int for v in pattern) and pattern in counts and
                pattern not in seen and type(n) is int and n>0,'Invalid/duplicate predicted role pattern')
        counts[pattern]=n;seen.add(pattern)
    require(sum(counts.values())==worlds,'Predicted role count total mismatch')
    out.update(worlds=worlds,per_actor_accuracy=actor.tolist(),true_pair_strata=out_strata,
        predicted_joint_role_distribution=[dict(labels=list(p),worlds=counts[p],proportion=counts[p]/worlds) for p in PATTERNS],
        source_path=record['path'],source_sha256=record['sha256'],state_indices_sha256=record['state_indices_sha256'])
    return out


def primary(runs):
    """Same fixed reduction order as the frozen main, implemented independently."""
    by={row['job']['id']:row for row in runs}
    require(len(runs)==42 and len(by)==42,'All42 actual jobs required')
    values=[]
    for seed in OLD_SEEDS:
        cells=[]
        for payoff in PAYOFFS:
            probes=[]
            for probe in PROBE_SEEDS:
                accuracies={view:by[job_id(probe,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None)]['final'][TARGET]['joint_accuracy'] for view in VIEWS}
                probes.append(dict(probe_seed=probe,accuracies=accuracies,
                    final_minus_initial=accuracies['Final']-accuracies['Initial'],
                    final_minus_own=accuracies['Final']-accuracies['Own'],
                    final_minus_bound=accuracies['Final']-23/62,initial_minus_bound=accuracies['Initial']-23/62))
            cells.append(dict(payoff=payoff,probes=probes,
                              mean_final_minus_initial=float(np.mean([p['final_minus_initial'] for p in probes]))))
        values.append(dict(old_seed=seed,payoffs=cells,
                           mean_final_minus_initial=float(np.mean([c['mean_final_minus_initial'] for c in cells]))))
    return dict(metric='joint_role_accuracy_final_minus_initial',partition=TARGET,protocol_seed_blocks=values,
                mean_difference=float(np.mean([v['mean_final_minus_initial'] for v in values])))


def average_records(records):
    """Equal-record descriptive means; role frequencies are proportions, not new counts."""
    require(len(records)>0 and len({r['worlds'] for r in records})==1,'Incomparable averaging denominators')
    world_count=records[0]['worlds']
    out={key:float(np.mean([r[key] for r in records])) for key in SCALARS}
    out.update(worlds_per_actual_evaluation=world_count,averaged_records=len(records),
        per_actor_accuracy=np.mean([r['per_actor_accuracy'] for r in records],axis=0).tolist(),
        true_pair_strata={pair:dict(worlds_per_actual_evaluation=records[0]['true_pair_strata'][pair]['worlds'],
            joint_accuracy=float(np.mean([r['true_pair_strata'][pair]['joint_accuracy'] for r in records]))) for pair in ('AB','AC','BC')},
        predicted_joint_role_distribution=[dict(labels=list(pattern),mean_proportion=float(np.mean([
            r['predicted_joint_role_distribution'][i]['proportion'] for r in records]))) for i,pattern in enumerate(PATTERNS)])
    require(close(sum(row['mean_proportion'] for row in out['predicted_joint_role_distribution']),1),'Mean role proportions do not sum1')
    return out


def mean_summaries(summaries):
    """Average already averaged seed/payoff summaries without pooling repeats."""
    out={key:float(np.mean([r[key] for r in summaries])) for key in SCALARS}
    out.update(worlds_per_actual_evaluation=summaries[0]['worlds_per_actual_evaluation'],
        averaged_blocks=len(summaries),per_actor_accuracy=np.mean([r['per_actor_accuracy'] for r in summaries],axis=0).tolist(),
        true_pair_strata={pair:dict(worlds_per_actual_evaluation=summaries[0]['true_pair_strata'][pair]['worlds_per_actual_evaluation'],
            joint_accuracy=float(np.mean([r['true_pair_strata'][pair]['joint_accuracy'] for r in summaries]))) for pair in ('AB','AC','BC')},
        predicted_joint_role_distribution=[dict(labels=list(pattern),mean_proportion=float(np.mean([
            r['predicted_joint_role_distribution'][i]['mean_proportion'] for r in summaries]))) for i,pattern in enumerate(PATTERNS)])
    return out


def derived(views,part):
    reference=bounds(part)
    return dict(final_minus_initial=views['Final']['joint_accuracy']-views['Initial']['joint_accuracy'],
                final_minus_own=views['Final']['joint_accuracy']-views['Own']['joint_accuracy'],
                no_communication_role_upper_bound=reference,
                per_view_minus_no_communication_bound={view:views[view]['joint_accuracy']-reference for view in VIEWS},
                bound_scope='Own-information upper bound; FI and transcript views contain additional information.',
                **({'per_view_minus_23_over_62':{view:views[view]['joint_accuracy']-23/62 for view in VIEWS}}
                   if part in ('new_needs',TARGET) else {}))


def native_reference(old_result):
    require(old_result.get('status')=='completed','Old policy result must be complete')
    by={(r['seed'],r['payoff']):r for r in old_result['runs'] if r['condition']=='PL_live'}
    require(set(by)==set(product(OLD_SEEDS,PAYOFFS)),'All8 old live policies required')
    rows=[]
    for seed in OLD_SEEDS:
        for payoff in PAYOFFS:
            run=by[seed,payoff];patterns=Counter();parts={}
            for part in PARTS:
                record=run['final'][part]['natural'];counts={tuple(row['partner_indices']):row['worlds'] for row in record['raw_joint_role_counts']}
                require(sum(counts.values())==record['worlds'],'Old actual role histogram denominator mismatch')
                require(all(type(n) is int and n>0 for n in counts.values()),'Invalid old role count')
                patterns.update(counts)
                parts[part]={key:record[key] for key in ('worlds','role_success_rate','full_success_rate','reward_mean','utility_mean','path','data_sha256')}
                parts[part]['role_accuracy_equals_one_third']=close(record['role_success_rate'],1/3)
            fixed=False
            if len(patterns)==1:
                pattern=next(iter(patterns));active=[i for i,p in enumerate(pattern) if p!=-1]
                fixed=len(active)==2 and pattern[active[0]]==active[1] and pattern[active[1]]==active[0]
            rows.append(dict(old_seed=seed,payoff=payoff,partitions=parts,
                fixed_mutual_pair_over_all_worlds=bool(fixed),
                actual_role_patterns=[dict(partner_indices=list(p),worlds=n) for p,n in sorted(patterns.items())],
                all_partitions_role_accuracy_equals_one_third=all(v['role_accuracy_equals_one_third'] for v in parts.values())))
    return rows


def extract_summary(result,prepared,old_result):
    require(result.get('status')=='completed','Refuse unfinished decoder results')
    runs=result['runs'];expected=logical_mapping();by={row['job']['id']:row for row in runs}
    require(len(runs)==42 and len(by)==42 and set(by)=={r['actual_job_id'] for r in expected},'Actual job coverage mismatch')
    require(result['logical_results']==expected,'Logical aliases changed')
    require(result['budget']==prepared['budget'],'Prepared/main budget mismatch')
    recomputed_primary=primary(runs)
    require(recomputed_primary==result['primary'],'Primary is not exactly equal to the fixed main reduction')
    parts=prepared['source']['partitions'];actual={};record_count=0
    for identity,run in by.items():
        job=run['job']
        require(identity==job_id(job['probe_seed'],job['view'],job['old_seed'],job['payoff']),'Job naming mismatch')
        require(set(run['final'])==set(PARTS) and tuple(row['update'] for row in run['monitor'])==STEPS,'Incomplete evaluations')
        final={part:checked_record(run['final'][part],parts[part]['world_count']) for part in PARTS}
        monitor={}
        for checkpoint in run['monitor']:
            require(set(checkpoint['evaluations'])==set(PARTS),'Missing monitor partition')
            monitor[str(checkpoint['update'])]={part:checked_record(checkpoint['evaluations'][part],len(parts[part]['monitor_indices'])) for part in PARTS}
        actual[identity]=dict(job=job,final=final,monitor=monitor)
        record_count+=len(final)+sum(len(p) for p in monitor.values())
    require(record_count==1176,'Actual evaluation count changed')
    def record(identity,phase,step,part):
        return actual[identity]['final'][part] if phase=='final' else actual[identity]['monitor'][str(step)][part]
    cells=[]
    for seed in OLD_SEEDS:
        for payoff in PAYOFFS:
            row=dict(old_seed=seed,payoff=payoff,final={},monitor={})
            for phase,step in [('final',6000)]+[('monitor',s) for s in STEPS]:
                destination=row['final'] if phase=='final' else row['monitor'].setdefault(str(step),{})
                for part in PARTS:
                    probes=[]
                    for probe in PROBE_SEEDS:
                        identities={view:job_id(probe,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None) for view in VIEWS}
                        values={view:record(identities[view],phase,step,part) for view in VIEWS}
                        probes.append(dict(probe_seed=probe,actual_job_ids=identities,
                            accuracies={v:values[v]['joint_accuracy'] for v in VIEWS},
                            final_minus_initial=values['Final']['joint_accuracy']-values['Initial']['joint_accuracy'],
                            final_minus_own=values['Final']['joint_accuracy']-values['Own']['joint_accuracy'],
                            per_view_minus_no_communication_bound={v:values[v]['joint_accuracy']-bounds(part) for v in VIEWS}))
                    means={view:average_records([record(p['actual_job_ids'][view],phase,step,part) for p in probes]) for view in VIEWS}
                    destination[part]=dict(probes=probes,mean_over_probe_seeds=means,**derived(means,part))
            cells.append(row)
    blocks=[]
    for seed in OLD_SEEDS:
        block=dict(old_seed=seed,final={},monitor={})
        selected=[c for c in cells if c['old_seed']==seed]
        for phase,step in [('final',6000)]+[('monitor',s) for s in STEPS]:
            dest=block['final'] if phase=='final' else block['monitor'].setdefault(str(step),{})
            for part in PARTS:
                sources=[c['final'][part] if phase=='final' else c['monitor'][str(step)][part] for c in selected]
                views={view:mean_summaries([s['mean_over_probe_seeds'][view] for s in sources]) for view in VIEWS}
                dest[part]=dict(mean_over_payoffs_after_probe_seeds=views,**derived(views,part))
        blocks.append(block)
    grand=dict(final={},monitor={})
    for phase,step in [('final',6000)]+[('monitor',s) for s in STEPS]:
        dest=grand['final'] if phase=='final' else grand['monitor'].setdefault(str(step),{})
        for part in PARTS:
            sources=[b['final'][part] if phase=='final' else b['monitor'][str(step)][part] for b in blocks]
            views={view:mean_summaries([s['mean_over_payoffs_after_probe_seeds'][view] for s in sources]) for view in VIEWS}
            dest[part]=dict(mean_over_protocol_seeds_after_payoff_and_probe=views,**derived(views,part))
    # The prespecified primary averages paired differences before averaging arms.
    # Its exact result is retained above; subtracting aggregate absolute means can
    # differ by floating-point roundoff, so the descriptive copy is tolerance-checked.
    require(close(grand['final'][TARGET]['final_minus_initial'],recomputed_primary['mean_difference']),'Descriptive/primary mismatch')
    native=native_reference(old_result)
    return dict(schema='triadic_role_decoder_descriptive_v1',status='summarized',primary=recomputed_primary,
        actual_jobs=actual,logical_results=expected,protocol_payoff_cells=cells,protocol_seed_blocks=blocks,
        grand_descriptive_means=grand,old_native_policies=native,
        old_policy_checks=dict(policies=len(native),fixed_pairs=sum(r['fixed_mutual_pair_over_all_worlds'] for r in native),
            all_partition_one_third=sum(r['all_partitions_role_accuracy_equals_one_third'] for r in native)),
        counts=dict(actual_jobs=42,logical_jobs=96,aliased_jobs=54,actual_evaluation_records=1176,
                    logical_evaluation_references=2688,aliased_evaluation_references=1512,
                    protocol_seed_blocks=4,probe_initializations_per_protocol=3,payoff_repeated_conditions=2,
                    joint_role_patterns=27),
        scope=dict(summary_only=True,probability_arrays_recomputed=False,neural_forward_calls=0,training_updates=0,
            primary_order='Within old seed and payoff: mean three paired Final-Initial probe differences; mean two payoffs; mean four old seeds.',
            aliases='Own/FI each trained once per probe seed; Initial once per old seed/probe; logical repetitions are not independent runs.',
            monitoring='All four original monitor supports at six saved decoder updates; separate from full endpoint support.',
            endpoint='Full four partitions, fixed6000 decoder update; no early stopping or selection by accuracy.',
            readout='Supervised auxiliary role decoders. Their scores are not the original actors joint behavior and do not establish language or composition.',
            reference='Original policy role behavior is read from frozen independently audited source records and their bound NPZ hashes; no source rollout is repeated.',
            uncertainty='All seeds, payoffs and probes retained. Four source initialization blocks; no significance claim or world-level independence.',
            units='Accuracies/probabilities are fractions; differences are fraction differences (times100 gives percentage points). Cross entropy is nats per actor.'))


def load_completed(run):
    run=Path(run).resolve();sources={}
    status_path=run/'execution/status.json'
    require(status_path.is_file() and read(status_path)['status']=='completed','Refuse active/incomplete execution')
    require(not (run/'execution/failure.json').exists(),'Execution has a failure marker')
    paths=[status_path,run/'execution/results.json',run/'plan.json',run/'prepared.json',run/'freeze.json']
    for path in paths:sources[str(path)]=sha(path)
    result,plan,prepared,freeze=(read(path) for path in paths[1:])
    require(sources[str(run/'plan.json')]==result['plan_sha256']==freeze['plan_sha256'],'Plan identity mismatch')
    require(sources[str(run/'prepared.json')]==plan['prepared_sha256'],'Prepared source mismatch')
    require(plan['inputs_sha256']==prepared['source']['source_sha256'],'Frozen source manifest mismatch')
    source=prepared['source'];old_root=Path(source['source_run'])
    old_paths=[old_root/'execution/results.json',old_root/'plan.json',old_root/'audit_execution_001/verification.json']
    for path in old_paths:
        actual=sha(path);require(actual==source['source_sha256'][str(path)],'Changed frozen original source')
        sources[str(path)]=actual
    old_result,old_plan,old_audit=(read(path) for path in old_paths)
    require(old_audit['status']=='passed' and old_audit['plan_sha256']==old_result['plan_sha256']==sources[str(old_paths[1])],
            'Old source independent audit not bound')
    by={(r['seed'],r['payoff']):r for r in old_result['runs'] if r['condition']=='PL_live'}
    for policy in source['policies']:
        original=by[policy['seed'],policy['payoff']]
        for part,entry in policy['final'].items():
            record=original['final'][part]['natural'];path=Path(entry['path'])
            digest=sha(path)
            require(record['path']==str(path) and digest==entry['sha256']==record['data_sha256']==source['source_sha256'][str(path)],
                    'Old final NPZ identity mismatch')
            require(old_audit['evaluations'][str(path)]['sha256']==digest and
                    equal_tree(old_audit['evaluations'][str(path)]['metrics'],
                               {k:v for k,v in record.items() if k not in ('path','data_sha256')}),
                    'Old audit/native record mismatch')
            sources[str(path)]=digest
    return result,prepared,old_result,sources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();run=Path(args.run).resolve();out=Path(args.out).resolve() if args.out else run/'summary_001'
    require(not out.exists(),'Refuse to overwrite summary output')
    result,prepared,old_result,sources=load_completed(run)
    summary=extract_summary(result,prepared,old_result)
    out.mkdir(parents=True,exist_ok=False)
    path=out/'descriptive_summary.json'
    with path.open('x',encoding='utf-8') as handle:json.dump(summary,handle,ensure_ascii=False,indent=2)
    require(all(sha(path)==digest for path,digest in sources.items()),'Summary input changed')
    receipt=dict(status='completed',at=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        script_sha256=sha(__file__),outputs={str(path):sha(path)},runtime=dict(python=platform.python_version(),numpy=np.__version__),
        scope=dict(neural_forward_calls=0,training_updates=0,npz_arrays_loaded=0,
                   bound_old_final_npz_files_hashed=32,main_actual_evaluation_records_checked=1176,
                   main_probabilities_recomputed=False,primary_exact_equals_main=True),
        primary=summary['primary'],old_policy_checks=summary['old_policy_checks'])
    with (out/'receipt.json').open('x',encoding='utf-8') as handle:json.dump(receipt,handle,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status='completed',out=str(out),primary=summary['primary']['mean_difference'],
                         old_policy_checks=summary['old_policy_checks']),ensure_ascii=False))


if __name__=='__main__':main()
