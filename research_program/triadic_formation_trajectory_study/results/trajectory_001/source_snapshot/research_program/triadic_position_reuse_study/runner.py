"""One frozen, bounded position-splice experiment; no training or retries."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import shutil
import time
import numpy as np
from research_program.triadic_message_study import runner as core
from . import dataset, discovery, metrics, channel_intervention as ch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent / 'triadic_context_transfer_study/results/transfer_001'
PART = 'new_needs_and_layouts'
PRIOR_PLAN_SHA = 'af051fa46aedb4653210c47587010d0b0e0e6d971a92f81bc61f2a98d96245c5'
PRIOR_RESULT_SHA = '5c610cd4f2e79d479ca3859e7e7628d93fbfa270415c539b282407baece914bc'
CONFIG = dict(seeds=[51101,51102,51103,51104], conditions=['PL_silent','PL_live','LL_silent','LL_live'],
    checkpoint=6000, discovery='complete_train_content_pairs', validation=PART,
    selected_cases_per_axis_ordered_pair=4, recipient_backgrounds_per_case=2,
    donor_rule='one_hash_ranked_eligible_donor_each_background', validation_rows=144,
    directions=[0,1], positions=list(range(8)), arms=['same','opposite'], shams=[0,4],
    discovery_ties='exact_fraction_score_then_lowest_position',
    primary='mean_four_paired_seed_PLlive_minus_LLlive_selectivity',
    selectivity='mean_diagonal_minus_mean_six_offdiagonal_of_training_selected_3x3_target_apt_gain',
    uniform_reference='exact_mean_all_eight_positions_on_identical_rows',
    actual_worlds=41472, actual_module_samples=186624, actual_files=288,
    silent_alias_records=288, silent_logical_worlds=41472, parameter_loads=8,
    reference_cells=96, reference_row_occurrences=13824,
    training_updates=0, new_natural_forward_worlds=0, execution='serial_policy_then_unit_arm_direction',
    positivity_filter=False, force_distinct_positions=False, validation_selection=False)

def require(ok, message):
    if not ok: raise AssertionError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')
def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def tag(policy): return f"seed_{policy['seed']}_{policy['condition']}"

def sources():
    files = [HERE / p for p in ('__init__.py','runner.py','dataset.py','channel_intervention.py',
        'discovery.py','metrics.py','plan.md','main_preflight.json','tests/test_runner.py',
        'tests/test_dataset.py','tests/test_channel_intervention.py','tests/test_discovery.py','tests/test_metrics.py')]
    old = read(PRIOR / 'plan.json')
    require(sha(PRIOR/'plan.json') == PRIOR_PLAN_SHA, 'Prior plan identity')
    for path, digest in old['sources_sha256'].items():
        require(sha(path) == digest, 'Prior frozen source changed: '+path)
        files.append(Path(path))
    return {str(p):sha(p) for p in files}

def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Refuse overwrite of preparation')
    ss = sources()
    require(sha(PRIOR/'execution/results.json') == PRIOR_RESULT_SHA, 'Prior result identity')
    old_plan, old_result = read(PRIOR/'plan.json'), read(PRIOR/'execution/results.json')
    require(old_result['status']=='completed', 'Prior execution incomplete')
    out.mkdir(parents=True)
    # This immutable record is written before static preparation and before any
    # policy-message discovery. Outcome-derived selections are bound afterwards.
    write(out/'frozen_spec.json', dict(at=now(), config=CONFIG, sources_sha256=ss,
        runtime=dict(python=platform.python_version(),numpy=np.__version__), discovery_started=False))
    for path in ss:
        target=out/'source_snapshot'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    dataset.prepare(out/'dataset')
    inputs = {str(PRIOR/'plan.json'):PRIOR_PLAN_SHA, str(PRIOR/'execution/results.json'):PRIOR_RESULT_SHA}
    dm=read(out/'dataset/manifest.json')
    inputs.update(dm['source_sha256'])
    ds=load_npz(out/'dataset/discovery.npz')
    vs=load_npz(out/'dataset/validation.npz')
    policies=[]
    for original in old_plan['policies']:
        p=dict(seed=original['seed'],condition=original['condition'],checkpoint=original['checkpoint'],
            endpoints={part:original['endpoints'][part] for part in ('train',PART)})
        require(p['seed'] in CONFIG['seeds'] and p['condition'] in CONFIG['conditions'], 'Unexpected policy')
        for path in [p['checkpoint'],*p['endpoints'].values()]:
            digest=old_plan['inputs_sha256'][path]
            require(sha(path)==digest, 'Changed original policy input')
            inputs[path]=digest
        directory=out/'discovery'/tag(p); directory.mkdir(parents=True)
        with np.load(p['endpoints']['train'],allow_pickle=False) as z: messages=z['messages']
        require(len(messages)==419904, 'Use the complete training endpoint domain')
        profile=discovery.select_positions(ds,messages)
        codes=discovery.training_code_sets(messages)
        write(directory/'selection.json',profile)
        np.savez_compressed(directory/'train_packet_codes.npz',**{str(a):codes[a] for a in range(3)})
        p.update(selection=str(directory/'selection.json'),train_packet_codes=str(directory/'train_packet_codes.npz'))
        refs=[r for r in old_result['records'] if r['seed']==p['seed'] and r['condition']==p['condition']
              and r['partition']==PART and r['mode'] in ('remote_same_both','remote_opposite_both')
              and r['shift_index'] in set(map(int,vs['donor_shift_index']))]
        require(len(refs)==4*len(set(vs['donor_shift_index'])), 'Missing old whole-message records')
        for r in refs:
            if r['path']:
                require(sha(r['path'])==r['data_sha256'], 'Changed whole-message reference')
                inputs[r['path']]=r['data_sha256']
        p['whole_reference_records']=refs
        policies.append(p)
        print(json.dumps(dict(stage='prepared_discovery',policy=tag(p))),flush=True)
    require(len(policies)==16 and len({tag(p) for p in policies})==16, 'Policy cohort')
    require(sources()==ss, 'Sources changed during preparation')
    prepared_files={str(p):sha(p) for p in out.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts}
    plan=dict(status='prepared_no_new_forward',at=now(),config=CONFIG,sources_sha256=ss,
        inputs_sha256=inputs,prepared_files_sha256=prepared_files,policies=policies,
        runtime=dict(python=platform.python_version(),numpy=np.__version__))
    write(out/'plan.json',plan);write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')))
    return dict(status=plan['status'],plan_sha256=sha(out/'plan.json'),config=CONFIG)

def verify(out):
    out=Path(out).resolve();p=read(out/'plan.json')
    require(sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256'], 'Plan changed')
    require(p['config']==CONFIG and p['sources_sha256']==sources(), 'Configuration/source changed')
    require(p['runtime']==dict(python=platform.python_version(),numpy=np.__version__), 'Runtime changed')
    for path,digest in {**p['inputs_sha256'],**p['prepared_files_sha256']}.items():
        require(sha(path)==digest,'Frozen input changed: '+path)
    for path,digest in p['sources_sha256'].items():
        require(sha(out/'source_snapshot'/Path(path).relative_to(ROOT))==digest,'Snapshot changed')
    return p

def cell_specs():
    for unit in range(8):
        for arm in ('same','opposite'):
            for direction in (0,1): yield unit,arm,direction
    for unit in (0,4):
        for direction in (0,1): yield unit,'sham',direction

def indices(spec,arm,direction):
    require(direction in (0,1) and arm in ('same','opposite','sham','natural'), 'Cell identity')
    ids=spec['endpoint_indices'][:,direction]
    if arm in ('sham','natural'): return ids,ids
    return ids,spec['donor_endpoint_indices'][:,direction if arm=='same' else 1-direction]

def settle_data(data,pool,spec,direction):
    ids=spec['endpoint_indices'][:,direction];cfids=spec['endpoint_indices'][:,1-direction]
    native=ch.settle(pool['states'][ids],data['action_indices'])
    cf=ch.settle(pool['states'][cfids],data['action_indices'])
    for key,value in native.items():
        if key in data: require(np.array_equal(data[key],value),'Reused settlement mismatch: '+key)
    if 'counterfactual_reward' in data:
        require(np.array_equal(data['counterfactual_reward'],cf['greedy_reward']), 'Reused counterfactual mismatch')
    data.update(native,counterfactual_reward=cf['greedy_reward'],counterfactual_satisfied=cf['satisfied'])
    require(np.array_equal(native['executed'],cf['executed']),'Demand flip changed physics')
    require(np.array_equal(cf['greedy_reward']==1,np.all(data['action_indices']==spec['correct_actions'][:,1-direction],axis=1)), 'Counterfactual truth mismatch')
    return data

def natural_data(pool,spec,direction):
    ids=spec['endpoint_indices'][:,direction];s=spec['sender'];rows=np.arange(len(s))
    data={k:pool[k][ids].copy() for k in ('messages','action_indices','action_probabilities')}
    data['patched_outward_packets']=data['messages'][rows,:,s,:].copy()
    return settle_data(data,pool,spec,direction)

def execute_cell(networks,x,pool,spec,unit,arm,direction,live):
    ids,di=indices(spec,arm,direction);s=spec['sender'];n=len(s);rows=np.arange(n)
    natural=pool['messages'][ids];donor=pool['messages'][di,:,s,:]
    trace=ch.intervene(networks,x,natural,s,donor,unit//4,unit%4,live=live)
    data={k:trace[k] for k in ('messages','patched_outward_packets','replacement_symbols')}
    for key in ('action_indices','action_probabilities'):
        data[key]=trace[key] if live else pool[key][ids].copy()
    data.update(dataset_rows=rows,recipient_indices=ids,counterfactual_recipient_indices=spec['endpoint_indices'][:,1-direction],
        donor_indices=di,donor_packets=donor)
    require(np.array_equal(data['messages'][:,0],natural[:,0]),'Own generated W1 was overwritten')
    require(np.array_equal(data['messages'][rows,1,s],natural[rows,1,s]),'Focal sender self history changed')
    if unit>=4 or not live: require(np.array_equal(data['messages'],natural),'Forbidden message recomputation')
    error=0.
    if arm=='sham':
        require(np.array_equal(data['messages'],natural),'Sham changed messages')
        require(np.array_equal(data['action_indices'],pool['action_indices'][ids]),'Sham changed action')
        error=float(np.max(np.abs(data['action_probabilities']-pool['action_probabilities'][ids])))
        require(error<=2e-12,'Sham probability mismatch')
    settle_data(data,pool,spec,direction)
    record=dict(unit=unit,window=unit//4,position=unit%4,arm=arm,direction=direction,worlds=n,
        is_silent_alias=not live,new_forward_worlds=n if live else 0,
        new_network_samples=int(trace['neural_forward_samples']), neural_forward_calls=int(trace['neural_forward_calls']),
        routing_sha256=trace['routing_sha256'],max_identity_error=error,
        donor_packets_sha256=core.array_sha(donor),outward_patch_visible=live)
    return data,record

def whole_reference(policy,pool,spec,arm,direction):
    ids,di=indices(spec,arm,direction);s=spec['sender'];n=len(s)
    data=natural_data(pool,spec,direction)
    data['patched_outward_packets']=pool['messages'][di,:,s,:].copy()
    used=[]
    if policy['condition'].endswith('_live'):
        fields=('messages','action_indices','action_probabilities','greedy_reward','executed','satisfied','counterfactual_reward','counterfactual_satisfied')
        for shift in sorted(set(map(int,spec['donor_shift_index']))):
            take=np.flatnonzero(spec['donor_shift_index']==shift);oldrows=spec['source_spec_row'][take]
            record=next(r for r in policy['whole_reference_records'] if r['shift_index']==shift and r['direction']==direction and r['mode']==f'remote_{arm}_both')
            with np.load(record['path'],allow_pickle=False) as z:
                require(np.array_equal(z['recipient_indices'][oldrows],ids[take]),'Whole recipient alignment')
                require(np.array_equal(z['donor_indices'][oldrows],di[take]),'Whole donor alignment')
                require(np.array_equal(z['donor_packets'][oldrows],data['patched_outward_packets'][take]),'Whole donor packet alignment')
                for key in fields: data[key][take]=z[key][oldrows]
            used.append(dict(path=record['path'],sha256=record['data_sha256'],shift=shift,selected_rows=len(take)))
    settle_data(data,pool,spec,direction)
    return data,used

def measure(spec,direction,pool,data,codes):
    ids=spec['endpoint_indices'][:,direction];cfids=spec['endpoint_indices'][:,1-direction]
    natural={k:pool[k][ids] for k in ('messages','action_indices','action_probabilities')}
    return metrics.row_values(spec,direction,pool['states'][ids],natural,data,codes,
                              counterfactual_states=pool['states'][cfids])

def execute(out):
    out=Path(out).resolve();plan=verify(out);execution=out/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter();write(execution/'started.json',dict(at=now(),pid=os.getpid(),plan_sha256=sha(out/'plan.json')))
    records=[];references=[];summaries=[];loads=0
    try:
        spec=load_npz(out/'dataset/validation.npz')
        for policy in plan['policies']:
            directory=execution/tag(policy);directory.mkdir()
            condition=policy['condition'];information=condition.split('_')[0];live=condition.endswith('_live')
            pool=load_npz(policy['endpoints'][PART]);networks=core.load_networks(policy['checkpoint']) if live else None
            loads+=int(live);codes_dict=load_npz(policy['train_packet_codes']);codes=[codes_dict[str(a)] for a in range(3)]
            x={d:ch.observations(pool['states'][spec['endpoint_indices'][:,d]],information) for d in (0,1)}
            cells={};ref_values={};shams={}
            for unit,arm,direction in cell_specs():
                data,record=execute_cell(networks,x[direction],pool,spec,unit,arm,direction,live)
                record.update(seed=policy['seed'],condition=condition,natural_source=policy['endpoints'][PART])
                name=f'unit{unit}_{arm}_direction{direction}'
                if live:
                    path=directory/(name+'.npz');np.savez_compressed(path,**data)
                    record.update(path=str(path),data_sha256=sha(path))
                else:
                    record.update(path=None,data_sha256=None,alias_recipe='Gather natural output by recipient endpoint; proposed one-symbol splice is invisible; exact input equality checked.')
                write(directory/(name+'.json'),record);records.append(record)
                values=measure(spec,direction,pool,data,codes)
                if arm=='sham': shams[(unit,direction)]=values
                else: cells[(unit,arm,direction)]=values
            position_reports={}
            for unit in range(8):
                arms={a:{k:np.stack([cells[(unit,a,d)][k] for d in (0,1)]) for k in cells[(unit,a,0)]} for a in ('same','opposite')}
                arms['contrast']=metrics.contrast(arms['same'],arms['opposite']);position_reports[unit]=arms
            summary=metrics.selection_summary(spec,read(policy['selection']),position_reports)
            summary['all_positions']={str(u):{arm:metrics.aggregate_rows(spec,values) for arm,values in arms.items()} for u,arms in position_reports.items()}
            for arm in ('natural','same','opposite'):
                for direction in (0,1):
                    if arm=='natural': data=natural_data(pool,spec,direction);used=[]
                    else: data,used=whole_reference(policy,pool,spec,arm,direction)
                    path=directory/f'reference_{arm}_direction{direction}.npz';np.savez_compressed(path,**data)
                    record=dict(seed=policy['seed'],condition=condition,kind='natural' if arm=='natural' else 'whole',
                        arm=arm,direction=direction,worlds=len(spec['axis']),path=str(path),data_sha256=sha(path),
                        source_natural=policy['endpoints'][PART],whole_sources=used,new_forward_worlds=0,new_network_samples=0,
                        is_silent_alias=not live)
                    references.append(record);ref_values[(arm,direction)]=measure(spec,direction,pool,data,codes)
            refs={a:{k:np.stack([ref_values[(a,d)][k] for d in (0,1)]) for k in ref_values[(a,0)]} for a in ('natural','same','opposite')}
            refs['contrast']=metrics.contrast(refs['same'],refs['opposite'])
            summary['matched_references']={a:metrics.aggregate_rows(spec,v) for a,v in refs.items()}
            summary['shams']={str(u):metrics.aggregate_rows(spec,{k:np.stack([shams[(u,d)][k] for d in (0,1)]) for k in shams[(u,0)]}) for u in (0,4)}
            summary.update(seed=policy['seed'],condition=condition,selection=read(policy['selection']))
            write(directory/'summary.json',summary);summaries.append(summary)
            print(json.dumps(dict(stage='policy_completed',policy=tag(policy),elapsed_seconds=time.perf_counter()-started)),flush=True)
        totals=dict(actual_worlds=sum(r['new_forward_worlds'] for r in records),actual_module_samples=sum(r['new_network_samples'] for r in records),
            actual_files=sum(not r['is_silent_alias'] for r in records),silent_alias_records=sum(r['is_silent_alias'] for r in records),
            silent_logical_worlds=sum(r['worlds'] for r in records if r['is_silent_alias']),parameter_loads=loads,
            reference_cells=len(references),reference_row_occurrences=sum(r['worlds'] for r in references))
        require(all(totals[k]==CONFIG[k] for k in totals),'Execution budget mismatch')
        paired=[]
        for seed in CONFIG['seeds']:
            pl=next(s for s in summaries if s['seed']==seed and s['condition']=='PL_live')
            ll=next(s for s in summaries if s['seed']==seed and s['condition']=='LL_live')
            paired.append(dict(seed=seed,PL=pl['selectivity'],LL=ll['selectivity'],difference=pl['selectivity']-ll['selectivity']))
        verify(out)
        result=dict(status='completed',at=now(),elapsed_seconds=time.perf_counter()-started,plan_sha256=sha(out/'plan.json'),
            records=records,references=references,totals=totals,training_updates=0,new_natural_forward_worlds=0,
            primary=dict(paired_seeds=paired,mean_difference=float(np.mean([p['difference'] for p in paired]))),
            policy_summaries=[dict(seed=s['seed'],condition=s['condition'],path=str(execution/tag(s)/'summary.json'),sha256=sha(execution/tag(s)/'summary.json')) for s in summaries])
        write(execution/'results.json',result);write(execution/'status.json',dict(status='completed',at=now()))
        return {k:v for k,v in result.items() if k not in ('records','references','policy_summaries')}
    except BaseException as error:
        write(execution/'failure.json',dict(status='failed',at=now(),error=repr(error),completed_records=len(records)))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','execute','verify'));parser.add_argument('--out',required=True)
    args=parser.parse_args();fn=globals()[args.command];result=fn(args.out)
    if args.command=='verify': result=dict(status='verified',plan_sha256=sha(Path(args.out)/'plan.json'))
    print(json.dumps(result,ensure_ascii=False))
