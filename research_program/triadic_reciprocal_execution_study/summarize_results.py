"""Describe completed settlement-study JSON records without loading any NPZ.

This checks stored-record coverage and arithmetic, not model behavior or an
independent replay. No production environment, kernel, or runner is imported.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

import numpy as np

SEEDS=(57101,57102,57103,57104)
RULES=('strict','reciprocal')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
TARGET='new_needs_and_layouts'
PAIRS=('AB','AC','BC')
PAIR_ROLES={'none':(-1,-1,-1),'AB':(1,0,-1),'AC':(2,-1,0),'BC':(-1,2,1)}
ROLE_PATTERNS=tuple(product(*[tuple(v for v in (-1,0,1,2) if v!=actor) for actor in range(3)]))
PRIMARY_METRICS=('full_success_rate','executed_partner_correct_rate','proposal_role_success_rate','reward_mean')
SCALARS=('reward_mean','full_success_rate','physical_execution_rate','executed_partner_correct_rate',
         'proposal_role_success_rate','kind_correct_rate','length_correct_rate','destination_correct_rate',
         'material_identity_correct_rate','ignored_proposal_world_rate','ignored_proposal_agent_rate','unexecuted_proposal_agent_rate')
CONDITIONAL=('expected_reward_given_greedy_messages','full_probability_given_greedy_messages',
             'execution_probability_given_greedy_messages','full_posterior_mass_given_greedy_messages')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        digest=hashlib.sha256()
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):return abs(a-b)<2e-12
def mean(values):return float(np.mean(values))


def histogram(rows,key,worlds,valid):
    counts={}
    for row in rows:
        pattern=tuple(row[key]);count=row['worlds']
        require(len(pattern)==3 and all(type(v) is int for v in pattern) and valid(pattern), 'Invalid histogram pattern')
        require(pattern not in counts and type(count) is int and count>0,'Duplicate/invalid histogram count')
        counts[pattern]=count
    require(sum(counts.values())==worlds,'Histogram denominator mismatch')
    return counts


def checked_settlement(row,worlds,rule):
    require(type(worlds) is int and worlds>0 and row['worlds']==worlds and row['rule']==rule,'Wrong settlement world count/rule')
    require(all(isinstance(row[k],(int,float)) and not isinstance(row[k],bool) and
                math.isfinite(row[k]) and 0<=row[k]<=1 for k in SCALARS),'Invalid settlement scalar')
    rewards=row['reward_counts']
    require(set(rewards)=={'0.0','0.5','1.0'} and all(type(n) is int and n>=0 for n in rewards.values()) and
            sum(rewards.values())==worlds,'Invalid reward counts')
    require(close(row['reward_mean'],(.5*rewards['0.5']+rewards['1.0'])/worlds) and
            close(row['full_success_rate'],rewards['1.0']/worlds),'Reward count arithmetic mismatch')
    histogram(row['raw_joint_action_counts'],'action_indices',worlds,lambda p:all(0<=v<17 for v in p))
    histogram(row['raw_proposal_role_counts'],'partner_indices',worlds,lambda p:p in ROLE_PATTERNS)
    roles=histogram(row['raw_executed_role_counts'],'partner_indices',worlds,lambda p:p in PAIR_ROLES.values())
    pairs=row['actual_pair_counts']
    require(set(pairs)==set(PAIR_ROLES) and all(type(n) is int and n>=0 for n in pairs.values()) and
            sum(pairs.values())==worlds,'Invalid executed-pair counts')
    require(all(pairs[p]==roles.get(pattern,0) for p,pattern in PAIR_ROLES.items()),'Executed roles/pair counts disagree')
    require(close(row['physical_execution_rate'],1-pairs['none']/worlds),'Physical execution/count mismatch')
    require(row['full_success_rate']<=row['executed_partner_correct_rate']+2e-12 and
            row['executed_partner_correct_rate']<=row['physical_execution_rate']+2e-12,'Execution metric ordering mismatch')
    ignored=row['ignored_proposal_counts_by_actor']
    require(len(ignored)==3 and all(type(n) is int and 0<=n<=worlds for n in ignored),'Invalid ignored counts')
    require(close(sum(ignored)/(3*worlds),row['ignored_proposal_agent_rate']) and
            close(sum(ignored)/worlds,row['ignored_proposal_world_rate']),'Ignored proposal denominator mismatch')
    if rule=='strict':require(sum(ignored)==0,'Strict cannot ignore a third proposal')
    strata=row['true_pair_strata'];require(set(strata)==set(PAIRS),'Missing true-pair strata')
    require(sum(s['worlds'] for s in strata.values())==worlds,'True-pair denominator mismatch')
    for stratum in strata.values():
        n=stratum['worlds'];require(type(n) is int and n>=0,'Invalid stratum size')
        require(all(stratum[k] is None if n==0 else isinstance(stratum[k],(int,float)) and
                    math.isfinite(stratum[k]) and 0<=stratum[k]<=1 for k in SCALARS),'Invalid stratum scalars')
    for key in SCALARS:
        weighted=sum(s['worlds']*s[key] for s in strata.values() if s['worlds'])/worlds
        require(close(row[key],weighted),'True-pair weighted metric mismatch: '+key)


def checked_record(record,worlds,rule):
    checked_settlement(record,worlds,rule)
    require(record['information']=='FI' and record['live'] is False,'Unexpected information condition')
    require(all(math.isfinite(record[k]) and -2e-12<=record[k]<=1+2e-12 for k in CONDITIONAL),'Invalid conditional metric')
    cross=record['cross_settlement'];require(set(cross)==set(RULES),'Missing cross-settlement condition')
    for settlement in RULES:
        checked_settlement(cross[settlement],worlds,settlement)
        require(cross[settlement]['raw_joint_action_counts']==record['raw_joint_action_counts'] and
                cross[settlement]['raw_proposal_role_counts']==record['raw_proposal_role_counts'],
                'Cross settlement changed proposals')
    require(all(record[k]==v for k,v in cross[rule].items()),'Native and same-rule cross summary disagree')
    require(cross['reciprocal']['full_success_rate']+2e-12>=cross['strict']['full_success_rate'],
            'Mechanical release must be nonnegative')
    for key in ('path','data_sha256','state_indices_sha256'):require(isinstance(record[key],str) and record[key], 'Missing saved record identity')


def contrast(cells,metrics=SCALARS):
    differences={m:cells['reciprocal'][m]-cells['strict'][m] for m in metrics}
    matrix={trained:{settlement:cells[trained]['cross_settlement'][settlement]['full_success_rate']
                     for settlement in RULES} for trained in RULES}
    ss=matrix['strict']['strict'];sr=matrix['strict']['reciprocal'];rs=matrix['reciprocal']['strict'];rr=matrix['reciprocal']['reciprocal']
    decomposition=dict(common_reciprocal_policy_difference=rr-sr,strict_policy_mechanical_release=sr-ss,
                       common_strict_policy_difference=rs-ss,reciprocal_policy_mechanical_release=rr-rs)
    require(close(differences['full_success_rate'],decomposition['common_reciprocal_policy_difference']+decomposition['strict_policy_mechanical_release']) and
            close(differences['full_success_rate'],decomposition['common_strict_policy_difference']+decomposition['reciprocal_policy_mechanical_release']),
            'Two decompositions do not recover primary difference')
    return dict(contrasts=differences,cells={r:{m:c[m] for m in metrics} for r,c in cells.items()},
                full_success_cross_settlement=matrix,decomposition=decomposition)


def primary(runs):
    """Independent copy of the frozen reduction, with identical arithmetic order."""
    by={(r['seed'],r['rule']):r for r in runs}
    require(len(runs)==8 and len(by)==8 and set(by)==set(product(SEEDS,RULES)),'Incomplete eight-run coverage')
    values=[]
    for seed in SEEDS:
        cells={rule:by[seed,rule]['final'][TARGET] for rule in RULES}
        values.append(dict(seed=seed,**contrast(cells,PRIMARY_METRICS)))
    return dict(metric='full_success_rate',partition=TARGET,paired_seeds=values,
                mean_difference=mean([r['contrasts']['full_success_rate'] for r in values]))


def mean_records(records):
    require(len(records)==4 and len({r['worlds'] for r in records})==1,'Four equal-domain seed records required')
    return dict(worlds_per_policy=records[0]['worlds'],seed_count=4,
        **{k:mean([r[k] for r in records]) for k in (*SCALARS,*CONDITIONAL)},
        cross_settlement={rule:{k:mean([r['cross_settlement'][rule][k] for r in records]) for k in SCALARS} for rule in RULES},
        true_pair_strata={pair:dict(worlds_per_policy=records[0]['true_pair_strata'][pair]['worlds'],
            **{k:(mean([r['true_pair_strata'][pair][k] for r in records])
                  if records[0]['true_pair_strata'][pair]['worlds'] else None) for k in SCALARS}) for pair in PAIRS})


def policy_behavior(run):
    """Fixedness comes from observed histograms, with failed worlds kept separate."""
    pair_counts=Counter({p:0 for p in PAIR_ROLES});proposal=Counter();executed=Counter();partition_counts={}
    for part in PARTS:
        record=run['final'][part];pair_counts.update(record['actual_pair_counts'])
        proposal.update({tuple(r['partner_indices']):r['worlds'] for r in record['raw_proposal_role_counts']})
        executed.update({tuple(r['partner_indices']):r['worlds'] for r in record['raw_executed_role_counts']})
        partition_counts[part]={k:deepcopy(record[k]) for k in ('worlds','actual_pair_counts','raw_proposal_role_counts','raw_executed_role_counts')}
    total=sum(pair_counts.values());nonempty=[p for p in PAIRS if pair_counts[p]>0];active=total-pair_counts['none']
    fixed=len(nonempty)==1
    return dict(seed=run['seed'],trained_rule=run['rule'],worlds=total,partitions=partition_counts,
        actual_pair_counts=dict(pair_counts),executing_worlds=active,no_execution_worlds=pair_counts['none'],
        observed_executed_pairs=nonempty,one_pair_when_execution_occurs=fixed,
        same_pair_executes_in_every_world=fixed and pair_counts['none']==0,
        fixed_executed_pair=nonempty[0] if fixed else None,
        raw_proposal_role_pattern_count=len(proposal),raw_executed_role_pattern_count=len(executed),
        raw_proposal_role_counts=[dict(partner_indices=list(p),worlds=proposal[p]) for p in ROLE_PATTERNS],
        raw_executed_role_counts=[dict(partner_indices=list(p),worlds=executed[p]) for p in ROLE_PATTERNS],
        scope='Complete four-domain saved native-settlement histograms; fixedness is not inferred from a one-third score.')


def extract_summary(result,prepared):
    require(result.get('status')=='completed','Refuse unfinished main results')
    require(result['budget']==prepared['budget'],'Prepared/main budget mismatch')
    calculated=primary(result['runs']);require(calculated==result['primary'],'Primary is not exactly equal to main')
    by={(r['seed'],r['rule']):r for r in result['runs']};spec=prepared['partitions']
    require(set(spec)==set(PARTS),'Partition coverage mismatch')
    records=0;paths=set()
    for run in result['runs']:
        require(run['updates']==6000 and run['condition']=='FI_silent','Unexpected training endpoint/condition')
        require(set(run['final'])==set(PARTS) and tuple(c['update'] for c in run['monitor'])==STEPS,'Incomplete saved points')
        for phase,step in [('final',6000)]+[('monitor',s) for s in STEPS]:
            values=run['final'] if phase=='final' else next(c['monitor'] for c in run['monitor'] if c['update']==step)
            require(set(values)==set(PARTS),'Incomplete evaluation domains')
            for part in PARTS:
                record=values[part];worlds=spec[part]['world_count'] if phase=='final' else len(spec[part]['monitor_indices'])
                checked_record(record,worlds,run['rule']);records+=1
                require(record['path'] not in paths,'Repeated physical evaluation path');paths.add(record['path'])
    require(records==224,'Expected224 actual records')
    def record(seed,rule,phase,step,part):
        run=by[seed,rule]
        return run['final'][part] if phase=='final' else next(c['monitor'][part] for c in run['monitor'] if c['update']==step)
    endpoints={};monitors={}
    for phase,step in [('final',6000)]+[('monitor',s) for s in STEPS]:
        output=endpoints if phase=='final' else monitors.setdefault(str(step),{})
        for part in PARTS:
            pairs=[dict(seed=seed,**contrast({rule:record(seed,rule,phase,step,part) for rule in RULES})) for seed in SEEDS]
            output[part]=dict(paired_seeds=pairs,
                means_by_trained_rule={rule:mean_records([record(seed,rule,phase,step,part) for seed in SEEDS]) for rule in RULES},
                mean_paired_contrasts={k:mean([p['contrasts'][k] for p in pairs]) for k in SCALARS},
                mean_decomposition={k:mean([p['decomposition'][k] for p in pairs]) for k in pairs[0]['decomposition']},
                mean_full_success_cross_settlement={r:{s:mean([p['full_success_cross_settlement'][r][s] for p in pairs]) for s in RULES} for r in RULES})
    require(endpoints[TARGET]['mean_paired_contrasts']['full_success_rate']==calculated['mean_difference'],'Endpoint primary reduction mismatch')
    behaviors=[policy_behavior(by[seed,rule]) for seed in SEEDS for rule in RULES]
    return dict(schema='triadic_reciprocal_execution_descriptive_v1',status='summarized',primary=calculated,
        actual_runs=deepcopy(result['runs']),full_endpoints=endpoints,monitor_subsets_by_update=monitors,
        final_policy_behavior=behaviors,budget=deepcopy(result['budget']),
        counts=dict(training_runs=8,independent_paired_seeds=4,actual_evaluation_records=224,
                    monitor_records=192,full_endpoint_records=32,cross_settlement_summary_references=448,
                    additional_cross_settlement_forward_calls=0,proposal_role_patterns=27),
        scope=dict(json_records_only=True,npz_files_opened=0,model_loads=0,neural_forward_calls=0,training_updates=0,
            primary_order='For each of four fixed seeds: reciprocal minus strict full endpoint double-holdout success; arithmetic mean in frozen seed order.',
            cross_settlement='Same saved proposals under both rules;448 summaries describe224 evaluations, not448 policy forwards.',
            decomposition='Two exact algebraic paths, not a causal identification of a neutral transition mechanism.',
            monitoring='Six saved actor-training updates on original fixed monitor subsets; full6000 endpoint kept separate. No best checkpoint selection.',
            histograms='All four domains and all seeds retained. none denotes no executing pair; proposal and actual execution patterns are separate.',
            validation='Checks completed JSON coverage and arithmetic only. Stored NPZ identities are declarations; no NPZ hash or array is read here.',
            uncertainty='Four paired independent initializations; worlds and cross-settlement references are not independent replications. No significance claim.',
            language='FI with cross-agent messages closed; success or changing partners does not demonstrate language formation.',
            units='Rates and probabilities are fractions; multiplying a difference by100 gives percentage points.'))


def load_completed(run):
    run=Path(run).resolve();status=run/'execution/status.json'
    require(status.is_file() and read(status).get('status')=='completed','Refuse active/incomplete execution')
    require(not (run/'execution/failure.json').exists(),'Execution has a failure marker')
    files=[status,run/'execution/results.json',run/'plan.json',run/'prepared.json',run/'freeze.json']
    sources={str(p):sha(p) for p in files}
    result,plan,prepared,freeze=(read(p) for p in files[1:])
    require(result.get('status')=='completed','Main result is incomplete')
    require(sources[str(run/'plan.json')]==result['plan_sha256']==freeze['plan_sha256'],'Plan identity mismatch')
    require(sources[str(run/'prepared.json')]==plan['prepared_sha256'],'Prepared identity mismatch')
    require(plan['config']['seeds']==list(SEEDS) and plan['config']['execution_rules']==list(RULES),'Frozen condition set changed')
    return result,prepared,sources


def execute(run,out=None):
    run=Path(run).resolve();out=Path(out).resolve() if out else run/'summary_001'
    require(not out.exists(),'Refuse to overwrite summary output')
    result,prepared,sources=load_completed(run);summary=extract_summary(result,prepared)
    require(all(sha(p)==digest for p,digest in sources.items()),'Summary input changed')
    out.mkdir(parents=True,exist_ok=False);path=out/'descriptive_summary.json'
    with path.open('x',encoding='utf-8') as handle:json.dump(summary,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    receipt=dict(status='completed',at=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        script_sha256=sha(__file__),outputs={str(path):sha(path)},runtime=dict(python=platform.python_version(),numpy=np.__version__),
        scope=dict(npz_files_opened=0,model_loads=0,neural_forward_calls=0,training_updates=0,
                   actual_evaluation_records=224,cross_settlement_summary_references=448,primary_exact_equals_main=True),
        primary=summary['primary'])
    with (out/'receipt.json').open('x',encoding='utf-8') as handle:json.dump(receipt,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    return dict(status='completed',out=str(out),primary=summary['primary']['mean_difference'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();print(json.dumps(execute(args.run,args.out),ensure_ascii=False))
