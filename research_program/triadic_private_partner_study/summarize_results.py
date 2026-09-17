"""JSON-only descriptive summary of completed private-partner experiments.

No policy output arrays, model, production metric function, or other study is
loaded. Stored endpoint and monitor records remain available without selection.
"""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import argparse,hashlib,json,math,platform

import numpy as np

SEEDS=(59101,59102,59103,59104)
CONDITIONS=('PL_live','PL_silent')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
AXES=('kind','length','destination')
TARGET='new_needs_and_layouts'
VIEWS=('PL_live_natural','PL_silent_natural','PL_live_closed')
TASK=('reward_mean','full_success_rate','physical_execution_rate','executed_partner_correct_rate',
      'proposal_role_success_rate','kind_correct_rate','length_correct_rate','destination_correct_rate',
      'material_identity_correct_rate','ignored_proposal_world_rate','ignored_proposal_agent_rate','unexecuted_proposal_agent_rate')
CONDITIONAL=('expected_reward_given_greedy_messages','full_probability_given_greedy_messages',
             'execution_probability_given_greedy_messages','full_posterior_mass_given_greedy_messages')
RESPONSE=('Q','Q_shuffle','Q_excess','both_correct_rate','only_first_correct_rate','only_second_correct_rate',
          'one_correct_rate','neither_correct_rate','partner_change_rate','both_executed_pair_change_rate','execution_status_change_rate')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def mean(values):return float(np.mean(values))
def close(a,b):return abs(a-b)<=2e-12


def primary(runs):
    """Same seed order and paired-difference reduction as the frozen main."""
    by={(r['seed'],r['condition']):r for r in runs}
    require(len(runs)==8 and set(by)==set(product(SEEDS,CONDITIONS)),'Expected eight complete paired runs')
    paired=[]
    for seed in SEEDS:
        values={c:by[seed,c]['final'][TARGET]['natural']['need_response']['Q'] for c in CONDITIONS}
        require(all(isinstance(q,(float,int)) and not isinstance(q,bool) and math.isfinite(q) and 0<=q<=1 for q in values.values()),'Undefined Q')
        paired.append(dict(seed=seed,Q_live=values['PL_live'],Q_silent=values['PL_silent'],difference=values['PL_live']-values['PL_silent']))
    return dict(metric='Q',partition=TARGET,contrast='PL_live_minus_PL_silent',independent_paired_seed_blocks=4,
                paired_seeds=paired,mean_difference=mean([p['difference'] for p in paired]))


def checked_response(row,spec):
    require(row['worlds']==spec['world_count'] and row['need_edges']==spec['need_edges'] and
            row['state_edges']==spec['state_edges'],'Response support mismatch')
    require(row['complete_nine_strata'] is True and row['empty_strata']==[] and len(row['strata'])==9,'Response must retain nine nonempty strata')
    require([(r['changed_person'],r['axis']) for r in row['strata']]==list(product(range(3),AXES)),'Response stratum order mismatch')
    for value in [row,*row['strata']]:
        require(all(isinstance(value[k],(int,float)) and math.isfinite(value[k]) and -1<=value[k]<=1 for k in RESPONSE),'Invalid response scalar')
        require(close(value['Q'],value['both_correct_rate']) and close(value['Q_excess'],value['Q']-value['Q_shuffle']),'Response Q identity mismatch')
        require(close(value['both_correct_rate']+value['one_correct_rate']+value['neither_correct_rate'],1),'Response outcome rates do not sum1')
    for actual,frozen in zip(row['strata'],spec['strata']):
        require(actual['need_edges']==frozen['need_edges'] and actual['state_edges']==frozen['state_edges'] and
                len(actual['backgrounds'])==spec['n_backgrounds'],'Response edge/background count mismatch')
        require([b['background_index'] for b in actual['backgrounds']]==list(range(spec['n_backgrounds'])),'Response background order mismatch')
        for k in RESPONSE:require(close(actual[k],mean([b[k] for b in actual['backgrounds']])),'Response background reduction mismatch')
    for k in RESPONSE:require(close(row[k],mean([r[k] for r in row['strata']])),'Nine-stratum equal weighting mismatch')


def checked_record(row,worlds,condition,mode,response_spec=None):
    require(row['worlds']==worlds and row['information']=='PL' and row['rule']=='reciprocal','Wrong record support/information/rule')
    require(row['mode']==mode and row['live']==(condition=='PL_live' and mode=='natural'),'Unexpected channel condition')
    require(all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) and -2e-12<=row[k]<=1+2e-12 for k in (*TASK,*CONDITIONAL)),'Invalid task scalar')
    require(set(row['actual_pair_counts'])=={'none','AB','AC','BC'} and
            all(type(v) is int and v>=0 for v in row['actual_pair_counts'].values()) and
            sum(row['actual_pair_counts'].values())==worlds,'Executed-pair denominator mismatch')
    for field in ('raw_joint_action_counts','raw_proposal_role_counts','raw_executed_role_counts'):
        require(sum(r['worlds'] for r in row[field])==worlds,'Joint histogram denominator mismatch')
    require(close(row['physical_execution_rate'],1-row['actual_pair_counts']['none']/worlds),'Physical/count inconsistency')
    if response_spec is None:require('need_response' not in row,'Monitor may not contain full-support Q')
    else:
        require('need_response' in row,'Missing complete endpoint Q');checked_response(row['need_response'],response_spec)


def means(records,with_response):
    require(len(records)==4 and len({r['worlds'] for r in records})==1,'Equal-domain four-seed records required')
    out=dict(seed_count=4,worlds_per_evaluation=records[0]['worlds'],
        **{k:mean([r[k] for r in records]) for k in (*TASK,*CONDITIONAL)},
        mean_actual_pair_proportions={p:mean([r['actual_pair_counts'][p]/r['worlds'] for r in records]) for p in ('none','AB','AC','BC')},
        true_pair_strata={p:{k:mean([r['true_pair_strata'][p][k] for r in records]) for k in TASK} for p in ('AB','AC','BC')})
    if with_response:
        out['need_response']={k:mean([r['need_response'][k] for r in records]) for k in RESPONSE}
        out['need_response']['strata']=[dict(changed_person=person,axis=axis,
            **{k:mean([r['need_response']['strata'][index][k] for r in records]) for k in RESPONSE})
            for index,(person,axis) in enumerate(product(range(3),AXES))]
    return out


def contrast(left,right,with_response):
    row={k:left[k]-right[k] for k in TASK}
    if with_response:
        row['need_response']={k:left['need_response'][k]-right['need_response'][k] for k in RESPONSE}
        row['need_response']['strata']=[dict(changed_person=p,axis=a,
            **{k:left['need_response']['strata'][i][k]-right['need_response']['strata'][i][k] for k in RESPONSE})
            for i,(p,a) in enumerate(product(range(3),AXES))]
    return row


def mean_contrasts(rows,with_response):
    out={k:mean([r[k] for r in rows]) for k in TASK}
    if with_response:
        out['need_response']={k:mean([r['need_response'][k] for r in rows]) for k in RESPONSE}
        out['need_response']['strata']=[dict(changed_person=p,axis=a,
            **{k:mean([r['need_response']['strata'][i][k] for r in rows]) for k in RESPONSE})
            for i,(p,a) in enumerate(product(range(3),AXES))]
    return out


def extract_summary(result,prepared):
    require(result.get('status')=='completed','Refuse unfinished results')
    require(result['budget']==prepared['budget'],'Budget mismatch')
    calculated=primary(result['runs']);require(calculated==result['primary'],'Primary must exactly equal main')
    by={(r['seed'],r['condition']):r for r in result['runs']};paths=set();record_count=0
    for run in result['runs']:
        require(run['updates']==6000 and run['rule']=='reciprocal','Unexpected endpoint/rule')
        require(set(run['final'])==set(PARTS) and tuple(r['update'] for r in run['monitor'])==STEPS,'Missing saved evaluations')
        for part in PARTS:
            modes=('natural','closed') if run['condition']=='PL_live' else ('natural',)
            require(set(run['final'][part])==set(modes),'Unexpected final mode or alias')
            for mode in modes:
                row=run['final'][part][mode]
                checked_record(row,prepared['partitions'][part]['world_count'],run['condition'],mode,prepared['need_response_cases'][part])
                require(row['path'] not in paths,'Repeated actual evaluation path');paths.add(row['path']);record_count+=1
        for checkpoint in run['monitor']:
            require(set(checkpoint['monitor'])==set(PARTS),'Incomplete monitor domains')
            for part,row in checkpoint['monitor'].items():
                checked_record(row,len(prepared['partitions'][part]['monitor_indices']),run['condition'],'natural')
                require(row['path'] not in paths,'Repeated actual evaluation path');paths.add(row['path']);record_count+=1
    require(record_count==240,'Expected240 actual evaluations')
    final={};monitor={}
    for part in PARTS:
        cells={view:[] for view in VIEWS};paired=[]
        for seed in SEEDS:
            records=dict(PL_live_natural=by[seed,'PL_live']['final'][part]['natural'],
                PL_silent_natural=by[seed,'PL_silent']['final'][part]['natural'],PL_live_closed=by[seed,'PL_live']['final'][part]['closed'])
            for view in VIEWS:cells[view].append(records[view])
            paired.append(dict(seed=seed,live_minus_silent=contrast(records['PL_live_natural'],records['PL_silent_natural'],True),
                live_minus_closed=contrast(records['PL_live_natural'],records['PL_live_closed'],True)))
        final[part]=dict(paired_seeds=paired,means_by_policy_view={v:means(cells[v],True) for v in VIEWS},
            mean_live_minus_silent=mean_contrasts([r['live_minus_silent'] for r in paired],True),
            mean_live_minus_closed=mean_contrasts([r['live_minus_closed'] for r in paired],True))
    for update in STEPS:
        output=monitor[str(update)]={}
        for part in PARTS:
            cells={condition:[next(c['monitor'][part] for c in by[seed,condition]['monitor'] if c['update']==update) for seed in SEEDS] for condition in CONDITIONS}
            paired=[dict(seed=seed,live_minus_silent=contrast(cells['PL_live'][i],cells['PL_silent'][i],False)) for i,seed in enumerate(SEEDS)]
            output[part]=dict(paired_seeds=paired,means_by_condition={c:means(cells[c],False) for c in CONDITIONS},
                mean_live_minus_silent=mean_contrasts([p['live_minus_silent'] for p in paired],False))
    require(final[TARGET]['mean_live_minus_silent']['need_response']['Q']==calculated['mean_difference'],'Primary descriptive reduction mismatch')
    return dict(schema='private_partner_descriptive_v1',status='summarized',primary=calculated,budget=deepcopy(result['budget']),
        actual_runs=deepcopy(result['runs']),full_endpoints=final,monitor_subsets_by_update=monitor,
        counts=dict(training_runs=8,paired_seed_blocks=4,actual_evaluation_records=240,natural_monitor_records=192,
                    natural_final_records=32,live_policy_closed_final_records=16,full_need_response_records=48,aliased_records=0),
        scope=dict(json_only=True,npz_files_opened=0,neural_forward_calls=0,model_loads=0,training_updates=0,
            primary='Full6000 double-holdout Q_live-Q_silent,paired within seed then equal mean of all4 seeds.',
            weights='Q retains equal9 structural strata and equal backgrounds;raw pooled edge counts are descriptive only.',
            monitor='Natural policy only,six fixed saved updates on original monitor subsets;no monitor Q or closed-channel records.',
            closure='Closed is a new recorded full rollout of an already trained live policy,not a separately trained silent policy or alias.',
            attribution='Within reciprocal rule only;does not identify an execution-rule×communication interaction.',
            interpretation='Q/task gains indicate cooperative response,not by themselves language,word meaning,composition or convention formation.',
            replication='Four paired initializations;edges,worlds,backgrounds and closure conditions are not new independent societies.',
            validation='CompletedJSON coverage and arithmetic checks,not independent NPZ replay;NPZ identities retained as saved declarations.',
            units='Rates/probabilities are fractions;fraction differences times100 are percentage points.'))


def load_completed(run):
    run=Path(run).resolve();status_path=run/'execution/status.json'
    require(status_path.is_file() and read(status_path).get('status')=='completed','Refuse active or incomplete execution')
    require(not (run/'execution/failure.json').exists(),'Execution has a failure marker')
    files=[status_path,run/'execution/results.json',run/'plan.json',run/'prepared.json',run/'freeze.json'];sources={str(p):sha(p) for p in files}
    status,result,plan,prepared,freeze=[read(p) for p in files]
    require(status['results_sha256']==sources[str(run/'execution/results.json')],'Completed result hash mismatch')
    require(result['plan_sha256']==freeze['plan_sha256']==sources[str(run/'plan.json')],'Plan identity mismatch')
    require(plan['prepared_sha256']==freeze['prepared_sha256']==sources[str(run/'prepared.json')],'Prepared identity mismatch')
    require(plan['config']['seeds']==list(SEEDS) and plan['config']['conditions']==list(CONDITIONS) and plan['config']['execution_rule']=='reciprocal','Frozen condition set changed')
    return result,prepared,sources


def execute(run,out=None):
    run=Path(run).resolve();out=Path(out).resolve() if out else run/'summary_001'
    require(not out.exists(),'Refuse to overwrite summary output')
    result,prepared,sources=load_completed(run);summary=extract_summary(result,prepared)
    require(all(sha(p)==h for p,h in sources.items()),'Input changed during summary')
    out.mkdir(parents=True,exist_ok=False);path=out/'summary.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    receipt=dict(status='completed',at=datetime.now(timezone.utc).isoformat(),source_sha256=sources,script_sha256=sha(__file__),
        outputs={str(path):sha(path)},primary=summary['primary'],counts=summary['counts'],
        scope=dict(npz_files_opened=0,neural_forward_calls=0,training_updates=0,primary_exact_equals_main=True),
        runtime=dict(python=platform.python_version(),numpy=np.__version__))
    with (out/'receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return dict(status='completed',out=str(out),primary=summary['primary']['mean_difference'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();print(json.dumps(execute(args.run,args.out),ensure_ascii=False))
