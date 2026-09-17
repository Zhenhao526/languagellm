"""Read-only complete-batch task/message description; never loads model weights.

Message frequencies and temporal changes describe generated strings. They do
not identify semantics, compositionality, or a precise formation time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from research_program.triadic_message_study import runner as r

HERE=Path(__file__).resolve().parent
SEEDS=(47101,47102,47103,47104)
CONDITIONS=("FI_silent","FI_live","PI_silent","PI_live")
PARTITIONS=("train","new_needs","new_layouts","new_needs_and_layouts")
CHECKPOINTS=(0,100,500,1500,3000,6000)
CONTRACT={"schema":"triadic_message_description_v1","seeds":list(SEEDS),"conditions":list(CONDITIONS),
    "partitions":list(PARTITIONS),"checkpoints":list(CHECKPOINTS),"monitor_records":384,"final_records":64,
    "frequency_population":"All generated messages, including undelivered other-agent messages in silent conditions",
    "code_space":"4096 fixed four-position eight-symbol strings; base8 position0 most significant",
    "transition_pairs":[list(x) for x in zip(CHECKPOINTS,CHECKPOINTS[1:])],
    "examples":"Every condition/seed/partition/checkpoint: first5 ascending fixed monitor state indices, all outcomes retained",
    "example_world_records":1920,"no_best_string_or_success_selection":True,
    "statistical_unit":"four paired team initializations on one reused development split",
    "scope":"Descriptive generated-message frequencies and changes; no semantic/formation-onset inference",
    "weight_loading":False,"neural_forward":False,"parameter_replay":False}


def frequency_counts(messages):
    m=np.asarray(messages)
    r.require(m.ndim==4 and m.shape[1:]==(2,3,4) and m.dtype.kind in 'iu'
              and ((m>=0)&(m<8)).all(),"Invalid generated messages")
    codes=(m.astype(np.int64)*np.array([512,64,8,1])).sum(-1)
    code_counts=np.empty((2,3,4096),dtype=np.int64)
    token_counts=np.empty((2,3,4,8),dtype=np.int64)
    for w in range(2):
        for a in range(3):
            code_counts[w,a]=np.bincount(codes[:,w,a],minlength=4096)
            for p in range(4):token_counts[w,a,p]=np.bincount(m[:,w,a,p],minlength=8)
    r.require((code_counts.sum(-1)==len(m)).all() and (token_counts.sum(-1)==len(m)).all(),"Frequency denominator changed")
    return code_counts,token_counts


def transition_counts(before,after):
    r.require(before.shape==after.shape and before.ndim==4 and before.shape[1:]==(2,3,4),"Unpaired message arrays")
    different=before!=after
    return {"worlds":len(before),"changed_tokens_by_window_agent_position":different.sum(0).tolist(),
            "changed_complete_messages_by_window_agent":different.any(-1).sum(0).tolist(),
            "worlds_with_any_generated_message_change":int(different.any(axis=(1,2,3)).sum())}


def endpoint_contrasts(rows):
    index={(x['seed'],x['condition'],x['partition']):x for x in rows}
    expected={(s,c,p) for s in SEEDS for c in CONDITIONS for p in PARTITIONS}
    r.require(len(rows)==64 and set(index)==expected,"All64 endpoint cells required")
    contrasts=[];averages=[]
    for part in PARTITIONS:
        pair_rows=[]
        for seed in SEEDS:
            rates={c:index[(seed,c,part)]['greedy_full_success_rate'] for c in CONDITIONS}
            rewards={c:index[(seed,c,part)]['greedy_reward_mean'] for c in CONDITIONS}
            pi=rates['PI_live']-rates['PI_silent'];fi=rates['FI_live']-rates['FI_silent']
            row={"seed":seed,"partition":part,"full_success_rates":rates,"native_reward_means":rewards,
                 "PI_live_minus_silent":pi,"FI_live_minus_silent":fi,"difference_in_differences_PI_minus_FI":pi-fi}
            pair_rows.append(row);contrasts.append(row)
        averages.append({"partition":part,"paired_seed_count":4,
            "contrasts":{k:{"seed_values":[row[k] for row in pair_rows],"mean":float(np.mean([row[k] for row in pair_rows])),
                             "min":min(row[k] for row in pair_rows),"max":max(row[k] for row in pair_rows)} for k in
                         ("PI_live_minus_silent","FI_live_minus_silent","difference_in_differences_PI_minus_FI")},
            "conditions":{c:{metric:{"seed_values":[index[(s,c,part)][metric] for s in SEEDS],
                                       "mean":float(np.mean([index[(s,c,part)][metric] for s in SEEDS])),
                                       "min":min(index[(s,c,part)][metric] for s in SEEDS),
                                       "max":max(index[(s,c,part)][metric] for s in SEEDS)}
                               for metric in ("greedy_full_success_rate","greedy_reward_mean")} for c in CONDITIONS}})
    return contrasts,averages


def describe_state(packed):
    state=r.base.env.State(tuple(map(int,packed[:3])),tuple(map(int,packed[3:7])),tuple(map(int,packed[7:])))
    full_views={a:r.base.env.observe(state,a,shared_needs=True,full_information=True) for a in r.base.AGENTS}
    return state,{"researcher_global_materials":full_views['A']['visible_materials'],
                  "researcher_global_needs":full_views['A']['shared_needs'],
                  "public_private_view_owners":full_views['A']['private_view_owners']}


def examples_for_record(record,data):
    indices=data['state_indices'];order=np.argsort(indices,kind='stable')[:5]
    full=record['condition'].startswith('FI_');live=record['condition'].endswith('_live')
    result=[]
    for offset in order:
        state,semantics=describe_state(data['states'][offset])
        m=data['messages'][offset]
        views={a:r.base.env.observe(state,a,shared_needs=full,full_information=full) for a in r.base.AGENTS}
        result.append({"seed":record['seed'],"condition":record['condition'],"partition":record['partition'],
            "update":record['update'],"state_index":int(indices[offset]),"source_npz":record['source_npz'],
            "selection":"first5 fixed monitor state indices, regardless of reward/messages",
            "researcher_global_semantics":semantics,"actual_agent_observations":views,
            "researcher_generated_messages":{a:["".join(r.ALPHABET[int(v)] for v in m[w,i]) for w in range(2)]
                                               for i,a in enumerate(r.base.AGENTS)},
            "delivery_visibility_by_listener":{a:{b:(live or a==b) for b in r.base.AGENTS} for a in r.base.AGENTS},
            "delivery_time":"Window1 broadcasts after all W1 senders, window2 broadcasts after all W2 senders",
            "actual_actions":{a:r.base.ACTIONS[i][int(data['action_indices'][offset,i])] for i,a in enumerate(r.base.AGENTS)},
            "actual_reward":float(data['greedy_reward'][offset]),
            "executed":dict(zip(r.base.AGENTS,map(bool,data['executed'][offset]))),
            "satisfied":dict(zip(r.base.AGENTS,map(bool,data['satisfied'][offset])))} )
    return result


def analyze(run,output):
    run=Path(run).resolve();output=Path(output).resolve()
    analysis_source_sha=r.sha(__file__)
    r.require(not output.exists(),"Refuse to overwrite descriptive analysis")
    status=r.read(run/'execution/status.json')
    r.require(status['status']=='completed' and status['completed_runs']==16,"Complete sixteen-run batch required before analysis")
    plan,prepared=r.verify(run)
    final_result=r.read(run/'execution/results.json')
    r.require(final_result['status']=='completed' and final_result['completed_run_count']==16,"Incomplete root aggregate")
    r.require([(x['seed'],x['condition']) for x in final_result['runs']]==[(s,c) for s in SEEDS for c in CONDITIONS],
              "Wrong canonical run sequence")
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        inputs={str(run/rel):r.sha(run/rel) for rel in ('plan.json','freeze.json','prepared.json','execution/results.json','execution/status.json')}
        records=[];endpoints=[];transitions=[];code_arrays=[];token_arrays=[];example_count=0
        example_path=output/'fixed_examples.jsonl'
        with example_path.open('x',encoding='utf-8') as examples:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    directory=run/'execution'/f'seed_{seed}_{condition}'
                    result=r.read(directory/'result.json');inputs[str(directory/'result.json')]=r.sha(directory/'result.json')
                    aggregate=next(x for x in final_result['runs'] if (x['seed'],x['condition'])==(seed,condition))
                    r.require(result==aggregate,"Run/root aggregate mismatch")
                    monitors={row['update']:row['monitor'] for row in result['monitor']}
                    r.require(set(monitors)==set(CHECKPOINTS),"Missing fixed checkpoint")
                    previous={}
                    for stage,update in [('monitor',u) for u in CHECKPOINTS]+[('final',6000)]:
                        for part in PARTITIONS:
                            filename=f'monitor_{update:04d}_{part}.npz' if stage=='monitor' else f'final_{part}.npz'
                            path=directory/filename;digest=r.sha(path);inputs[str(path)]=digest
                            saved_score=monitors[update][part] if stage=='monitor' else result['final'][part]
                            r.require(digest==saved_score['data_sha256'],"Saved evaluation NPZ changed")
                            with np.load(path,allow_pickle=False) as saved:
                                data={k:saved[k] for k in ('states','state_indices','messages','action_indices','action_probabilities',
                                                          'greedy_reward','executed','satisfied')}
                            spec=prepared['partitions'][part]
                            expected=np.array(spec['monitor_indices'],dtype=np.int64) if stage=='monitor' else np.arange(spec['world_count'],dtype=np.int64)
                            r.require(np.array_equal(data['state_indices'],expected),"Evaluation state selection/order differs")
                            n=len(expected);m=data['messages'];reward=data['greedy_reward']
                            r.require(m.shape==(n,2,3,4) and data['action_indices'].shape==(n,3)
                                      and data['action_probabilities'].shape==(n,3,17),"Incomplete saved world arrays")
                            r.require(r.array_sha(data['states'])==saved_score['packed_states_sha256']
                                      and r.array_sha(m)==saved_score['messages_sha256']
                                      and r.array_sha(data['action_indices'])==saved_score['action_indices_sha256'],"Saved array SHA changed")
                            r.require(np.isin(reward,[0,.5,1]).all() and saved_score['worlds']==n,"Invalid native reward/world count")
                            counts={str(v):int((reward==v).sum()) for v in (0.,.5,1.)}
                            r.require(counts==saved_score['greedy_reward_counts']
                                      and int((reward==1).sum())==saved_score['greedy_full_successes']
                                      and float((reward==1).mean())==saved_score['greedy_full_success_rate']
                                      and float(reward.mean())==saved_score['greedy_reward_mean'],"Reward aggregate mismatch")
                            codes,tokens=frequency_counts(m);code_arrays.append(codes);token_arrays.append(tokens)
                            record={"frequency_array_row":len(records),"seed":seed,"condition":condition,"partition":part,
                                "stage":stage,"update":update,"worlds":n,"source_npz":str(path),"source_sha256":digest,
                                "generated_full_string_types_by_window_agent":np.count_nonzero(codes,axis=-1).tolist(),
                                "greedy_full_success_rate":saved_score['greedy_full_success_rate'],
                                "greedy_reward_mean":saved_score['greedy_reward_mean'],"greedy_reward_counts":counts,
                                "greedy_non_full_success_worlds":int((reward!=1).sum()),
                                "greedy_failure_categories":saved_score['greedy_failure_categories'],
                                "greedy_executed_pair_worlds":saved_score['greedy_executed_pair_worlds'],
                                "conditional_exact_expected_reward_mean":saved_score['conditional_exact_expected_reward_mean'],
                                "conditional_exact_full_success_probability_mean":saved_score['conditional_exact_full_success_probability_mean']}
                            records.append(record)
                            if stage=='monitor':
                                if part in previous:
                                    old=previous[part]
                                    r.require(np.array_equal(old['states'],data['states']),"Checkpoint world states changed")
                                    transitions.append({"seed":seed,"condition":condition,"partition":part,
                                        "before_update":old['update'],"after_update":update,**transition_counts(old['messages'],m)})
                                previous[part]={'update':update,'states':data['states'],'messages':m.copy()}
                                for example in examples_for_record(record,data):examples.write(r.json_bytes(example).decode());example_count+=1
                            else:endpoints.append(record)
        r.require(len(records)==448 and len(endpoints)==64 and len(transitions)==320 and example_count==1920,
                  "Incomplete descriptive populations")
        contrasts,means=endpoint_contrasts(endpoints)
        computed=r.primary_comparison(final_result['runs'])
        r.require(computed==final_result['primary_comparison'],"Original main contrast mismatch")
        by_seed={row['seed']:row for row in contrasts if row['partition']=='new_needs_and_layouts'}
        for row in computed['seed_pairs']:
            r.require(all(row[k]==by_seed[row['seed']][k] for k in ('PI_live_minus_silent','FI_live_minus_silent',
                'difference_in_differences_PI_minus_FI')),"Description/main paired contrast differs")
        freq_path=output/'message_frequencies.npz'
        np.savez_compressed(freq_path,full_string_counts=np.stack(code_arrays),position_symbol_counts=np.stack(token_arrays))
        # Reverify byte sources and whole run before publishing completed metrics.
        r.require(all(r.sha(path)==digest for path,digest in inputs.items()),"Analysis input changed while reading")
        r.require(r.sha(__file__)==analysis_source_sha,"Analysis source changed while running")
        r.verify(run)
        summary={"status":"completed","completed_at":r.now(),"contract":CONTRACT,
            "run_directory":str(run),"plan_sha256":r.sha(run/'plan.json'),"input_sha256":inputs,
            "analysis_source_sha256":analysis_source_sha,"evaluation_records":records,"endpoint_rows":endpoints,
            "adjacent_checkpoint_message_changes":transitions,"paired_endpoint_contrasts":contrasts,
            "four_seed_endpoint_means_and_ranges":means,"primary_comparison":computed,
            "outputs_sha256":{"message_frequencies.npz":r.sha(freq_path),'fixed_examples.jsonl':r.sha(example_path)},
            "example_world_records":example_count,"weight_loads":0,"neural_forward_calls":0,"parameter_replay":False,
            "elapsed_seconds":time.perf_counter()-started}
        r.write_new(output/'summary.json',summary)
        (output/'读取说明.md').write_text(
            '# 固定消息描述与完整任务汇总\n\n'
            '本输出使用全部384个固定监测记录及64个完整终点。summary.json保留所有四种子的四格配对差、失败与得分，未作显著性检验。\n\n'
            'message_frequencies.npz的第一轴由evaluation_records中的frequency_array_row对应：full_string_counts为[448,2窗,3人,4096完整码]，'
            'position_symbol_counts为[448,2窗,3人,4位置,8符号]。码号按8进制、首位置为高位；字母表依次为@ # $ % & * + ~。'
            '频数涵盖silent组实际生成但未跨人投递的内容，不能称全部都是主体听到的广播。\n\n'
            'fixed_examples.jsonl逐监测记录固定取排序最前5个世界，共1920条，不挑成功或可读字符串。全局语义明确标为研究者信息，'
            '各主体原观察及投递可见性另列。不能把字符串看似有规律直接解释成语义。\n\n'
            '相邻检查点变动仅比较固定相同状态，并保留全部5个预定区间。它不提供未保存更新的准确形成时刻；'
            '频数、变动、消息类别和本页例子均为描述，不单独证明消息因果、组合性或新语言。没有加载权重或重放优化。\n',encoding='utf-8')
        return {"status":"completed","output":str(output),"records":len(records),"example_world_records":example_count,
                "summary_sha256":r.sha(output/'summary.json')}
    except BaseException as error:
        r.write_new(output/'failure.json',{"status":"failed","failed_at":r.now(),'error_type':type(error).__name__,
                    'error':str(error),'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(analyze(args.run,args.out),ensure_ascii=False))
