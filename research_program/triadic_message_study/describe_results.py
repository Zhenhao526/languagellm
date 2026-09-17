"""Read-only post-completion message/partner description, with all rows retained.

Only existing JSON counts are used; no model, NumPy, environment or optimizer.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

SEEDS=(47101,47102,47103,47104)
CONDITIONS=('FI_silent','FI_live','PI_silent','PI_live')
PARTITIONS=('train','new_needs','new_layouts','new_needs_and_layouts')
NAMES=dict(zip(PARTITIONS,('训练／训练','新需求／训练布局','训练需求／新布局','双留出')))
AGENTS=('A','B','C')


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def proposal_description(run,actions):
    domain=run['full_domain_actions'];entries=domain['joint_argmax_action_counts']
    require(domain['worlds']==143424 and sum(x['worlds'] for x in entries)==143424,'Incomplete domain proposal denominator')
    require(len(entries)==domain['distinct_joint_argmax_actions']==len({tuple(x['action_indices']) for x in entries}),
            'Duplicate/missing full-domain joint action types')
    stats={a:{'wait_worlds':0,'transport_worlds':0,'partners':Counter(),'sites':Counter(),'destinations':Counter()} for a in AGENTS}
    active_counts=Counter()
    for row in entries:
        n=row['worlds'];ids=row['action_indices']
        require(isinstance(n,int) and n>0 and len(ids)==3 and all(isinstance(x,int) and 0<=x<17 for x in ids),
                'Invalid actual joint action count')
        active=0
        for i,a in enumerate(AGENTS):
            choice=actions[i][ids[i]]
            if choice['kind']=='wait':stats[a]['wait_worlds']+=n
            else:
                require(choice['kind']=='transport','Unexpected original action')
                active+=1;stats[a]['transport_worlds']+=n
                for key,field in (('partners','partner'),('sites','site'),('destinations','destination')):
                    stats[a][key][choice[field]]+=n
        active_counts[active]+=n
    wait=[a for a in AGENTS if stats[a]['wait_worlds']==143424]
    active=[a for a in AGENTS if stats[a]['transport_worlds']==143424]
    fixed=None
    if len(wait)==1 and len(active)==2 and all(set(stats[a]['partners'])==set(active)-{a} for a in active):
        fixed=''.join(active)
    executed={pair:sum(run['final'][p]['greedy_executed_pair_worlds'][pair] for p in PARTITIONS) for pair in ('AB','AC','BC')}
    require(sum(executed.values())<=143424,'Execution counts exceed worlds')
    return {'seed':run['seed'],'condition':run['condition'],'worlds':143424,
            'distinct_joint_argmax_actions':len(entries),'active_agent_count_worlds':dict(sorted(active_counts.items())),
            'always_waiting_agents':wait,'always_transporting_agents':active,'fixed_mutual_partner_pair':fixed,
            'proposal_counts_by_agent':{a:{k:dict(v) if isinstance(v,Counter) else v for k,v in info.items()} for a,info in stats.items()},
            'actual_executed_pair_worlds':executed,
            'all_domain_semantic_action_fields_observed':'wait/transport, partner, site, destination directly tallied from every stored joint argmax type'}


def describe(run,summary,output):
    run,summary,output=map(lambda p:Path(p).resolve(),(run,summary,output))
    require(not output.exists(),'Description output already exists')
    main_path=run/'execution/results.json';prepared_path=run/'prepared.json'
    inputs={str(p):sha(p) for p in (main_path,prepared_path,summary)}
    data=json.loads(main_path.read_text());prepared=json.loads(prepared_path.read_text());s=json.loads(summary.read_text())
    require(data['status']==s['status']=='completed' and data['completed_run_count']==16,'Completed entire batch required')
    require(s['input_sha256'][str(main_path)]==inputs[str(main_path)],'Summary bound to different root results')
    require([(x['seed'],x['condition']) for x in data['runs']]==[(seed,c) for seed in SEEDS for c in CONDITIONS],
            'Wrong run population')
    behavior=[proposal_description(row,prepared['actions']) for row in data['runs']]
    require(len(behavior)==16,'All16 proposal records required')
    index={(x['seed'],x['condition'],x['partition']):x for x in s['endpoint_rows']}
    last={(x['seed'],x['condition'],x['partition']):x for x in s['adjacent_checkpoint_message_changes']
          if (x['before_update'],x['after_update'])==(3000,6000)}
    messages=[]
    for seed in SEEDS:
        for part in PARTITIONS:
            endpoint=index[(seed,'PI_live',part)];change=last[(seed,'PI_live',part)]
            require(change['worlds']==1024,'Wrong last-interval monitor population')
            counts=change['changed_complete_messages_by_window_agent'];flat=[n for window in counts for n in window]
            messages.append({'seed':seed,'condition':'PI_live','partition':part,'full_endpoint_worlds':endpoint['worlds'],
                'complete_message_type_counts_by_window_agent':endpoint['generated_full_string_types_by_window_agent'],
                'transition_updates':[3000,6000],'transition_fixed_worlds':1024,
                'changed_complete_messages_by_window_agent':counts,
                'changed_complete_message_fractions_by_window_agent':[[n/1024 for n in window] for window in counts],
                'six_agent_window_change_fraction_min':min(flat)/1024,'six_agent_window_change_fraction_max':max(flat)/1024,
                'worlds_with_any_message_change':change['worlds_with_any_generated_message_change'],
                'actual_executed_pair_worlds':endpoint['greedy_executed_pair_worlds'],
                'greedy_full_success_rate':endpoint['greedy_full_success_rate'],
                'greedy_reward_counts':endpoint['greedy_reward_counts'],
                'greedy_failure_categories':endpoint['greedy_failure_categories']})
    output.mkdir(parents=True,exist_ok=False)
    payload={'status':'completed','recorded_at':datetime.now(timezone.utc).isoformat(),'input_sha256':inputs,
             'source_sha256':sha(__file__),'whole_domain_proposals':behavior,'PI_live_message_rows':messages,
             'primary_comparison':data['primary_comparison'],
             'scope':'All saved full-domain joint argmax counts checked, not only executed pairs. PI message diversity/stability are descriptions, not semantics.',
             'new_model_loads':0,'neural_forward_calls':0,'new_training_updates':0}
    lines=['# 消息描述与全域伙伴提议核对','',
        '本补充仅读完整批次的原结果与一次固定汇总；未读取权重、前向或重放优化。消息频率／变化沿预定固定记录，'
        '伙伴结论额外遍历全部联合动作频数，不从成功执行的搭档反推所有提议。','',
        '## 全16运行的完整143424世界提议','',
        '| 种子 | 条件 | 固定互选对 | 始终等待 | 联合动作种类 | 实际执行 AB／AC／BC |','| --- | --- | --- | --- | ---: | --- |']
    for row in behavior:
        pair=row['fixed_mutual_partner_pair'] or '非固定';waiting='/'.join(row['always_waiting_agents']) or '无'
        counts='/'.join(str(row['actual_executed_pair_worlds'][p]) for p in ('AB','AC','BC'))
        lines.append(f"| {row['seed']} | {row['condition']} | {pair} | {waiting} | {row['distinct_joint_argmax_actions']} | {counts} |")
    all_fixed=all(row['fixed_mutual_partner_pair'] for row in behavior)
    lines+=['',f'全部16运行是否均有同一固定互选搭档、第三人始终等待：{all_fixed}。每人的等待／运输、伙伴、位置、目的地计数均保存在JSON，覆盖失败提议。',
        '固定伙伴不能解释成没有输入变化：位置／目的地仍可能随状态变化。相反，联合动作种类多也包括双方位置或目的地不一致的组合，不能等同于更多成功合作方式。','',
        '## PI_live全部种子与分区的消息种类、末段变动','',
        '每组A/B/C按同一顺序；“种类”是一窗4符号完整消息的实际种类数，来自更新6000的完整分区。'
        '末段数值为同一1024监测世界中，3000→6000时完整消息至少一个符号变化的次数；范围是六个主体×窗口比例的最小值–最大值。'
        '完整终点和监测子集分母不同，不能合并解释。','',
        '| 种子 | 分区 | W1种类 A/B/C | W2种类 A/B/C | W1变化数／1024 A/B/C | W2变化数／1024 A/B/C | 六项变化率范围 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in messages:
        types=row['complete_message_type_counts_by_window_agent'];changes=row['changed_complete_messages_by_window_agent']
        vals=['/'.join(map(str,v)) for v in [types[0],types[1],changes[0],changes[1]]]
        lines.append(f"| {row['seed']} | {NAMES[row['partition']]} | "+' | '.join(vals)+
                     f" | {row['six_agent_window_change_fraction_min']*100:.3f}%–{row['six_agent_window_change_fraction_max']*100:.3f}% |")
    lines+=['','## PI_live各格实际执行和失败','',
        '| 种子 | 分区 | 世界数 | 执行 AB／AC／BC | 两人不匹配 | 匹配后无人满足 | 只满足一人 | 满分 |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in messages:
        f=row['greedy_failure_categories'];counts='/'.join(str(row['actual_executed_pair_worlds'][p]) for p in ('AB','AC','BC'))
        lines.append(f"| {row['seed']} | {NAMES[row['partition']]} | {row['full_endpoint_worlds']} | {counts} | {f['two_unmatched']} | {f['matched_no_need_satisfied']} | {f['matched_one_need_satisfied']} | {row['greedy_reward_counts']['1.0']} |")
    lines+=['','## 配对方向与解释范围','',
        '| 种子 | PI开放−静默／百分点 | FI开放−静默／百分点 | 两者之差／百分点 |',
        '| --- | ---: | ---: | ---: |']
    for row in data['primary_comparison']['seed_pairs']:
        lines.append(f"| {row['seed']} | {row['PI_live_minus_silent']*100:+.4f} | {row['FI_live_minus_silent']*100:+.4f} | {row['difference_in_differences_PI_minus_FI']*100:+.4f} |")
    means=data['primary_comparison']['equal_weight_means']
    lines.append(f"| 四种子均值 | {means['PI_live_minus_silent']*100:+.4f} | {means['FI_live_minus_silent']*100:+.4f} | {means['difference_in_differences_PI_minus_FI']*100:+.4f} |")
    lines+=['','PI四个差值均正；FI47104和差中差47103的反向结果保留。四个初始化共用一套开发划分，不作显著性检验，差中差不单独隔离信息机制。',
        '这些消息可能稳定也可能继续变化，仅据频数不能认定词义、共享语法、组合表达或约定已经完成。'
        '各窗字符串由固定四位置分类头产生，窗口、身份、长度与顺序是外设机制。形成时点不能从两个检查点之间的连线确定。',
        '消息的任务内容使用与因果作用须另看冻结末点通道删除／内容移植；本描述不代替那些读数。'
        '全部1920个透明示例固定选各监测记录排序最前5世界，并区分研究者全局语义与主体实际观察，未挑选漂亮或成功字符串。','']
    require(all(sha(p)==digest for p,digest in inputs.items()),'Input changed during description')
    write_new(output/'description.json',payload)
    with (output/'消息与伙伴描述.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    return {'status':'completed','output':str(output),'all16_fixed_mutual_pair':all_fixed,
            'description_sha256':sha(output/'description.json'),'markdown_sha256':sha(output/'消息与伙伴描述.md')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);a=parser.parse_args()
    print(json.dumps(describe(a.run,a.summary,a.out),ensure_ascii=False))
