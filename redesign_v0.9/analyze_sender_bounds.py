"""Post-hoc sender distinguishability bounds from saved v0.8 codebook counts.

Read-only with respect to all source experiments; no model or training imports.
The constrained bound is an analyst lookup-table oracle, not a learned receiver.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product, permutations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=(27101,27102,27103,27104)
KINDS=('additive','mixed','joint')
MAPS=list(permutations(range(6),2))
CODES=list(product(range(7),repeat=2))


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def count_rate(n,d):return {'numerator':int(n),'denominator':int(d),'rate':float(n/d) if d else None}


def constrained_dp(train_max,held_max,threshold):
    """Exact per-code choice frontier. Each code decodes into train OR held.

    Any nonmaximal map inside a chosen support is dominated: it changes no other
    code and cannot improve the other support's count. Thus two choices suffice.
    Keep every attainable exact training count and its largest held count.
    """
    require(len(train_max)==len(held_max),'Mismatched code counts')
    require(0<=threshold<=sum(train_max),'Infeasible original training threshold')
    states={0:0};parents=[]
    for t,h in zip(train_max,held_max):
        require(t>=0 and h>=0,'Negative count')
        nxt={};parent={}
        for old_t,old_h in sorted(states.items()):
            for choice,dt,dh in ((0,int(t),0),(1,0,int(h))):
                new_t,new_h=old_t+dt,old_h+dh
                if new_t not in nxt or new_h>nxt[new_t]:
                    nxt[new_t]=new_h;parent[new_t]=(old_t,choice)
        states=nxt;parents.append(parent)
    eligible=[(h,t) for t,h in states.items() if t>=threshold]
    best_h,best_t=max(eligible)  # among maximal held scores, prefer more train successes
    choices=[];cursor=best_t
    for parent in reversed(parents):
        cursor,choice=parent[cursor];choices.append(choice)
    choices.reverse();require(cursor==0,'DP backtracking failed')
    require(sum(t for t,c in zip(train_max,choices) if c==0)==best_t,'DP train witness')
    require(sum(h for h,c in zip(held_max,choices) if c==1)==best_h,'DP held witness')
    return {'train_correct':int(best_t),'held_correct':int(best_h),'choices':choices,
            'frontier':[{'train_correct':int(t),'held_correct':int(h)} for t,h in sorted(states.items())],
            'states_at_final_code':len(states)}


def self_test():
    checked=0
    # Exhaustive thresholds and 2^k assignments are independent of the DP update.
    cases=[([3,3],[2,0]),([0,2,1],[3,0,2]),([2,1,3,0],[1,2,0,4]),([0,0],[0,0])]
    for ts,hs in cases:
        for threshold in range(sum(ts)+1):
            brute=[]
            for choices in product((0,1),repeat=len(ts)):
                t=sum(v for v,c in zip(ts,choices) if c==0)
                h=sum(v for v,c in zip(hs,choices) if c==1)
                if t>=threshold:brute.append((h,t))
            result=constrained_dp(ts,hs,threshold)
            require((result['held_correct'],result['train_correct'])==max(brute),'Brute-force DP mismatch')
            checked+=1
    reused=constrained_dp([3,3],[2,0],3)
    require(reused['choices']==[1,0] and reused['held_correct']==2,'Occupied-code reassignment example')
    return {'status':'passed','exhaustive_toy_thresholds':checked,
        'scope':'Integer optimization only; no model inference or new experimental data.',
        'occupied_code_example':'An old-correct code can be reassigned to heldout while another code gains equal train successes.'}


def message_counts(entries):
    counts={}
    for entry in entries:
        code=tuple(entry['message']);n=entry['count']
        require(code in CODES and isinstance(n,int) and n>0,'Invalid codebook frequency')
        require(code not in counts,'Duplicate codebook message entry')
        counts[code]=n
    return counts


def analyze_direction(run,direction,photos_per_map):
    phase=direction['phases']['validation'];book=phase['codebook'];menu=direction['menu_audit']
    require(menu['all_menu_permutations_physically_equivalent'] and menu['menus']==720,
            'Fixed physical decoder requires verified menu invariance')
    require(menu['enumerated_messages']==49 and menu['goals']==2,'Decoder support')
    decoder={tuple(row['message']):tuple(row['actions_by_goal']) for row in direction['receiver_decoder_table']}
    require(set(decoder)==set(CODES),'Incomplete 49-code decoder')
    require(all(len(v)==2 and all(isinstance(x,int) and 0<=x<6 for x in v) for v in decoder.values()),'Decoder place range')
    indexed={(r['map_id'],r['sender_goal']):r for r in book}
    require(len(book)==len(indexed)==60 and set(indexed)==set(product(range(30),(0,1))),'Codebook map/source-goal coverage')
    matrix=np.zeros((49,30),dtype=np.int64);code_index={c:i for i,c in enumerate(CODES)}
    baseline_per_map=np.zeros(30,dtype=np.int64)
    for mid,pair in enumerate(MAPS):
        a,b=indexed[mid,0],indexed[mid,1]
        for r in (a,b):
            require((r['food_location'],r['water_location'])==pair,'Map ID convention changed')
            require(r['both']['denominator']==photos_per_map,'Unequal per-map photo weighting')
        f0=message_counts(a['delivered_messages']);f1=message_counts(b['delivered_messages'])
        require(f0==f1 and a['both']==b['both'],'Source-goal compatibility rows are not duplicates')
        require(message_counts(a['emitted_messages'])==message_counts(b['emitted_messages']),'Emitted source-goal dependence')
        require(sum(f0.values())==photos_per_map,'Delivered frequency sum')
        food=water=both=0
        for code,n in f0.items():
            matrix[code_index[code],mid]+=n
            food+=n*(decoder[code][0]==pair[0]);water+=n*(decoder[code][1]==pair[1]);both+=n*(decoder[code]==pair)
        require((food,water,both)==(a['native']['numerator'],a['switched']['numerator'],a['both']['numerator']),
                'Reconstructed natural F/W/both mismatch')
        require((water,food,both)==(b['native']['numerator'],b['switched']['numerator'],b['both']['numerator']),
                'Duplicate source-goal natural counts mismatch')
        baseline_per_map[mid]=both
    train=np.array(run['train_map_ids'],dtype=int);held=np.array(run['heldout_map_ids'],dtype=int)
    require(len(train)==24 and len(held)==6 and set(train).isdisjoint(held) and set(train)|set(held)==set(range(30)),
            'Expected 24/6 map partition')
    np.testing.assert_array_equal(matrix.sum(0),np.full(30,photos_per_map))
    metrics={};maximizing_maps={}
    for label,ids in (('train24',train),('held6',held),('full30',np.arange(30))):
        maxima=matrix[:,ids].max(1);denominator=int(matrix[:,ids].sum());natural=int(baseline_per_map[ids].sum())
        bound=int(maxima.sum());require(natural<=bound<=denominator,'Sender bound versus actual performance')
        metrics[label]={'C_S':count_rate(bound,denominator),'natural_J':count_rate(natural,denominator),
            'unrecoverable_conflict_rate':1-bound/denominator,'oracle_minus_natural':(bound-natural)/denominator}
        maximizing_maps[label]=ids[matrix[:,ids].argmax(1)].tolist()
        if label!='full30':
            recorded=phase['cross_goal_groups']['seen' if label=='train24' else 'unseen']['same_message_correct_for_both_goals']
        else:recorded=phase['cross_goal']['same_message_correct_for_both_goals']
        require(recorded['numerator']==2*natural and recorded['denominator']==2*denominator,
                'Compatibility aggregate must contain exactly two duplicated source-goal indices')
    ts=matrix[:,train].max(1).tolist();hs=matrix[:,held].max(1).tolist()
    threshold=metrics['train24']['natural_J']['numerator'];dp=constrained_dp(ts,hs,threshold)
    new_maps=[maximizing_maps['train24' if choice==0 else 'held6'][i] for i,choice in enumerate(dp['choices'])]
    witness_t=witness_h=0;assignment=[]
    for i,(code,mid) in enumerate(zip(CODES,new_maps)):
        t=int(matrix[i,mid]) if mid in train else 0;h=int(matrix[i,mid]) if mid in held else 0
        witness_t+=t;witness_h+=h
        old_pair=decoder[code];old_mid=MAPS.index(old_pair) if old_pair in MAPS else None
        assignment.append({'message':list(code),'current_decoder_F_W':list(old_pair),
            'oracle_map_id':int(mid),'oracle_F_W':list(MAPS[mid]),
            'current_train_correct':int(matrix[i,old_mid]) if old_mid is not None and old_mid in train else 0,
            'current_held_correct':int(matrix[i,old_mid]) if old_mid is not None and old_mid in held else 0,
            'oracle_train_correct':t,'oracle_held_correct':h,
            'train_emissions':int(matrix[i,train].sum()),'held_emissions':int(matrix[i,held].sum())})
    require((witness_t,witness_h)==(dp['train_correct'],dp['held_correct']),'Recounted oracle witness disagrees with DP')
    require(dp['train_correct']>=threshold and dp['held_correct']>=metrics['held6']['natural_J']['numerator'],'Current decoder must be feasible/dominated')
    require(dp['held_correct']<=metrics['held6']['C_S']['numerator'],'Constrained bound exceeds independent held oracle')
    constraint={'original_train_correct':threshold,'train_denominator':24*photos_per_map,
        'max_held':count_rate(dp['held_correct'],6*photos_per_map),
        'witness_train':count_rate(dp['train_correct'],24*photos_per_map),
        'improvement_over_current_held':(dp['held_correct']-metrics['held6']['natural_J']['numerator'])/(6*photos_per_map),
        'constraint_cost_vs_unconstrained_held':(metrics['held6']['C_S']['numerator']-dp['held_correct'])/(6*photos_per_map),
        'frontier':dp['frontier'],'states_at_final_code':dp['states_at_final_code'],'oracle_assignment':assignment}
    return {'seed':run['seed'],'condition':run['condition'],'reward_kind':run['plan']['reward_kind'],
        'lambda':run['plan']['complementarity'],'split':run['plan']['split'],
        'scout':direction['scout'],'collector':direction['collector'],
        'train_map_ids':train.tolist(),'heldout_map_ids':held.tolist(),
        'metrics':metrics,'no_aggregate_train_loss_oracle':constraint,
        'frequency_matrix_axes':'49 lexicographic complete messages by 30 ordered resource maps',
        'delivered_code_by_map_counts':matrix.tolist(),
        'independent_subset_argmax_map_ids':maximizing_maps,
        'final_checkpoint':run['final_checkpoint'],'final_sha256_from_protocol':run['final_sha256'],
        'checks':{'goal_compatibility_rows_deduplicated':30,'natural_map_counts_reconstructed':30,
                  'unique_validation_cases':30*photos_per_map,'dp_assignment_recounted':True}}


def describe(values):
    return {'n':len(values),'mean':float(np.mean(values)),'min':float(min(values)),
            'max':float(max(values)),'values':[float(x) for x in values]}


def aggregate(directions):
    keys=('train_C_S','held_C_S','full_C_S','natural_train_J','natural_held_J','natural_full_J',
          'held_C_S_without_train_loss','witness_train_J','oracle_held_gain','train_retention_cost')
    def value(row,key):
        maps={'train_C_S':('train24','C_S'),'held_C_S':('held6','C_S'),'full_C_S':('full30','C_S'),
              'natural_train_J':('train24','natural_J'),'natural_held_J':('held6','natural_J'),'natural_full_J':('full30','natural_J')}
        if key in maps:
            group,field=maps[key];return row['metrics'][group][field]['rate']
        d=row['no_aggregate_train_loss_oracle']
        return {'held_C_S_without_train_loss':d['max_held']['rate'],'witness_train_J':d['witness_train']['rate'],
            'oracle_held_gain':d['improvement_over_current_held'],'train_retention_cost':d['constraint_cost_vs_unconstrained_held']}[key]
    out={}
    for kind in KINDS:
        local=[d for d in directions if d['reward_kind']==kind];require(len(local)==24,'Expected 24 directions per utility condition')
        by_seed=[]
        for seed in SEEDS:
            rows=[d for d in local if d['seed']==seed]
            require(len(rows)==6 and {(r['split'],r['scout']) for r in rows}==set(product((1,2,3),(0,1))),
                    'Expected three splits and two directions per seed')
            by_seed.append({'seed':seed,'metrics':{k:float(np.mean([value(r,k) for r in rows])) for k in keys}})
        out[kind]={'per_seed':by_seed,'metrics':{k:describe([s['metrics'][k] for s in by_seed]) for k in keys}}
    return out


def render_report(result,path):
    pc=lambda v:f'{100*v:.4f}%'
    lines=['# 冻结发送者的可区分性诊断','',
        '本诊断在v0.8完成后提出，仅使用已保存的验证协议频数和接收解码表，没有新训练、模型推理、照片采样或重编码评分。'
        '它为下一步发送/接收路径适应实验提供事后能力上界，不是新增自然语言成绩。','',
        '## 计算口径','',
        '每个方向固定当前贪心发送函数，按实际delivered_messages建立码m与地图z的频数n(m,z)。'
        'JSON字段sender_goal只取0，并逐图核对1索引只是同一隐藏目标消息及双目标结果的重复。'
        '每图16个验证照片对，所以训练24图分母384、留出6图96、全30图480；每个码只能解成同一张完整食物/水地图。','',
        '分别对每个支持集G计算 C_S(G)=Σ_m max_{z∈G} n(m,z) / N_G。'
        '这一上界允许分析者按地图真值重新赋义全部码，已超出当前接收器。训练、留出和全图三种最佳赋义通常不同，不能同时当作一个可实现的解码器。','',
        '| 效用条件 | 训练C_S | 留出C_S | 全图C_S | 当前训练J | 当前留出J |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind,g in result['groups'].items():
        m=g['metrics'];lines.append(f'| {kind} | '+' | '.join(pc(m[k]['mean']) for k in ('train_C_S','held_C_S','full_C_S','natural_train_J','natural_held_J'))+' |')
    lines += ['', '上表先在每个训练种子内平均三划分和两方向，再等权平均4个种子。'
        '验证照片口径不同于每运行9600世界的终点评估；频数中存在同一照片或地图的重复测量，不能增加独立样本量。','',
        '## 保持训练正确总数的二维选择动态规划','',
        '对每个码计算t_m=max训练地图频数、h_m=max留出地图频数。解为训练地图贡献(t_m,0)，解为留出地图贡献(0,h_m)。'
        '同一支持内任何较低频地图都被该支持的最大项支配，因此每码这两个选择足以求最优。'
        'DP逐码保存可达到的训练正确数T及其最大留出正确数H，再在T≥原接收器训练正确总数的状态中最大化H。'
        '留出最优并列时选择训练正确更多的见证。','',
        '| 条件 | 当前留出J | 保持训练总数的留出上界 | 上界−当前（百分点） | 独立留出上界−受约束上界（百分点） | 见证训练J |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind,g in result['groups'].items():
        m=g['metrics'];lines.append(f"| {kind} | {pc(m['natural_held_J']['mean'])} | {pc(m['held_C_S_without_train_loss']['mean'])} | "
            f"{100*m['oracle_held_gain']['mean']:.4f} | {100*m['train_retention_cost']['mean']:.4f} | {pc(m['witness_train_J']['mean'])} |")
    lines += ['', '这项约束保持训练正确**总数**，允许超过原值，也允许原本正确的具体训练案例转错、其他案例转对。'
        '它允许重用已经被训练地图占用的码，没有限定只能使用空闲码。因此它既不是逐例无遗忘保证，也不是神经接收器可学习性。'
        '每方向的整数前沿、49码重新赋义见证及重算结果保存在JSON中。','',
        '| 条件 | 种子 | 训练C_S | 留出C_S | 全图C_S | 保持训练总数的留出上界 | 当前留出J |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for kind,g in result['groups'].items():
        for s in g['per_seed']:
            m=s['metrics'];lines.append(f"| {kind} | {s['seed']} | "+' | '.join(pc(m[k]) for k in ('train_C_S','held_C_S','full_C_S','held_C_S_without_train_loss','natural_held_J'))+' |')
    lines += ['', '## 对后续适应实验的约束','',
        'C_S低于1表示当前发送函数在这个经验分布上把部分不同地图压到同一实际码，单靠重赋义无法全部恢复。'
        'C_S高于自然J则表明理想查表接收器仍有改善空间，不能据此断言真实主体通过标量反馈能够学会。'
        '仅查看留出C_S可能夸大无遗忘适应空间，必须同时考虑全图C_S和训练总数约束。','',
        '上述上界只固定当前贪心发送函数及这些验证照片。后续若视觉编码、发送策略、取样方式或输入分布变化，上界需重新定义。'
        '仅冻结发送头而继续更新共享视觉/状态路径，并不保证发送函数固定。实际receiver_only实验应核验完整发送路径，'
        'sender_only实验则受既有接收可达性约束；两者差距都不能直接解释成视觉识别、绑定或语义理解的单一原因。','',
        f"独立检查：36个留出运行、72个方向、{result['validation_cases_after_source_goal_deduplication']:,}个去重后记账案例；"
        f"{result['compatibility_codebook_rows_checked']:,}个源目标兼容行均核对，自然F/W/双目标整数计数重建一致。"
        f"DP通过{result['dp_self_test']['exhaustive_toy_thresholds']}个小规模穷举阈值对照，72个最优赋义见证逐项重算。",'',
        f"来源：[v0.8 protocol_analysis.json]({result['source']})；SHA-256：`{result['source_sha256']}`。",'',
        f"复现脚本：[analyze_sender_bounds.py]({Path(__file__).resolve()})；[完整JSON]({path.with_suffix('.json')})。",'']
    path.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT.parent/'redesign_v0.8/results/complementarity_001/protocol_analysis.json')
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();source=args.source.resolve();data=json.loads(source.read_text());before=sha(source)
    require(data['status']=='complete' and len(data['runs'])==60,'Expected complete v0.8 source')
    runs=[r for r in data['runs'] if r['plan']['split'] in (1,2,3)]
    require(len(runs)==36,'Expected 36 split runs')
    expected={(s,split,k) for s in SEEDS for split in (1,2,3) for k in KINDS}
    require({(r['seed'],r['plan']['split'],r['plan']['reward_kind']) for r in runs}==expected,'Source run scope mismatch')
    directions=[];test=self_test()
    for run in runs:
        require(not run['plan']['blocked'] and not run['plan']['known'],'Expected unblocked hidden-target source')
        require(len(run['directions'])==2 and {d['scout'] for d in run['directions']}=={0,1},'Both source directions')
        for direction in run['directions']:
            pairs=direction['phases']['validation']['photo_pairs']
            require(pairs==data['photo_splits']['validation'] and len(pairs)==len(set(map(tuple,pairs)))==16,'Fixed validation photos')
            directions.append(analyze_direction(run,direction,len(pairs)))
    result={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
        'source_sha256':before,'script_sha256':sha(__file__),'timing':'post-hoc v0.8 diagnostic before choosing v0.9 adaptation',
        'scope':{'runs':36,'directions':72,'independent_training_pair_seeds':list(SEEDS),
                 'phase':'validation','source_goal_field':'sender_goal','source_goal_kept':0,'photos_per_map':16},
        'code_order':[list(c) for c in CODES],'map_order':[list(m) for m in MAPS],
        'dp_self_test':test,'validation_cases_after_source_goal_deduplication':sum(d['checks']['unique_validation_cases'] for d in directions),
        'compatibility_codebook_rows_checked':len(directions)*60,'groups':aggregate(directions),'directions':directions,
        'interpretation':'Empirical analyst-oracle joint decoding ceilings; no learning claim; constrained DP retains aggregate train successes, not each example.'}
    require(sha(source)==before,'Source file changed while reading')
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output/'发送可区分性诊断.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    render_report(result,out.with_suffix('.md'))
    print(json.dumps({'status':result['status'],'directions':len(directions),'output':str(out.resolve()),
        'macro':{k:{m:100*v['mean'] for m,v in g['metrics'].items()} for k,g in result['groups'].items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
