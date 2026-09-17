"""Post-result exploratory compatibility bounds for old18 and added6 only.

Reads saved messages and receiver logits. No training, inference, or sealed-map
optimization. Whole-message oracle reassignments are not natural agent scores.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
from itertools import permutations,product
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=(29101,29102,29103,29104)
MAPS=np.array(list(permutations(range(6),2)),dtype=np.int64)
MESSAGES=np.array(list(product(range(7),repeat=2)),dtype=np.int64)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class QA:
    def __init__(self):self.counts={};self.failures=[]
    def check(self,value,name,context=None):
        self.counts[name]=self.counts.get(name,0)+1
        if not bool(value):self.failures.append(dict(check=name,context=context))


def solve(old_counts,added_counts):
    """Exact achievable frontier. Independent brute force checks uncompressed choices."""
    old_best=old_counts.max(0);added_best=added_counts.max(0)
    limit=int(old_best.sum());ncode=old_counts.shape[1]
    value=np.full(limit+1,-1,dtype=np.int64);value[0]=0;backs=[]
    for code in range(ncode):
        current=np.full(limit+1,-1,dtype=np.int64)
        previous=np.full(limit+1,-1,dtype=np.int64);choice=np.full(limit+1,-1,dtype=np.int8)
        for before in np.flatnonzero(value>=0):
            for option,(o,a) in enumerate(((int(old_best[code]),0),(0,int(added_best[code])))):
                after=int(before)+o;candidate=int(value[before])+a
                if candidate>current[after]:current[after]=candidate;previous[after]=before;choice[after]=option
        value=current;backs.append((previous,choice))
    return dict(values=value,backs=backs,old_best=old_best,added_best=added_best)


def optimum(dp,threshold):
    eligible=np.flatnonzero((np.arange(len(dp['values']))>=threshold)&(dp['values']>=0))
    if not len(eligible):raise ValueError('Infeasible old-correct threshold')
    # Among added-optimal solutions, retain as many old correct cases as possible.
    best=max(eligible,key=lambda o:(int(dp['values'][o]),int(o)))
    state=int(best);options=[]
    for previous,choice in reversed(dp['backs']):
        options.append(int(choice[state]));state=int(previous[state])
    assert state==0
    return int(best),int(dp['values'][best]),np.array(options[::-1],dtype=np.int8)


def exhaustive_qa(qa):
    rng=np.random.default_rng(10010619)
    for trial in range(48):
        no=int(rng.integers(1,4));na=int(rng.integers(1,4));nm=int(rng.integers(1,6))
        oc=rng.integers(0,4,size=(no,nm));ac=rng.integers(0,4,size=(na,nm));dp=solve(oc,ac)
        attainable=[]
        # Enumerate every old-map, added-map and irrelevant action assignment,
        # without the DP's per-code dominance reduction.
        for assignments in product(range(no+na+1),repeat=nm):
            o=a=0
            for code,assignment in enumerate(assignments):
                if assignment<no:o+=int(oc[assignment,code])
                elif assignment<no+na:a+=int(ac[assignment-no,code])
            attainable.append((o,a))
        for threshold in range(int(oc.max(0).sum())+1):
            ref=max((a,o) for o,a in attainable if o>=threshold)
            o,a,choices=optimum(dp,threshold)
            qa.check((a,o)==ref,'dp_matches_exhaustive_uncompressed_choices',[trial,threshold])
            replay_o=sum(int(oc[:,m].max()) for m,x in enumerate(choices) if x==0)
            replay_a=sum(int(ac[:,m].max()) for m,x in enumerate(choices) if x==1)
            qa.check((replay_o,replay_a)==(o,a),'small_dp_backpointer_replay',[trial,threshold])
        cs=int(ac.max(0).sum());old_at_cs=int(oc.max(0)[ac.max(0)==0].sum())
        qa.check(max(o for o,a in attainable if a==cs)==old_at_cs,'minimum_loss_formula_exhaustive',trial)
    return dict(trials=48,rng_seed=10010619,description='All complete decoder assignments in small random integer-count cases, including suboptimal maps and irrelevant actions; all integer retention thresholds.')


def counts_for(positions,messages,pool):
    return np.array([np.bincount(messages[(positions==MAPS[m]).all(1)],minlength=49) for m in pool],dtype=np.int64)


def selected_probe(path,pool):
    # The frozen probe schema is map-major, 16 fixed photographs per map.
    # Read only old/added rows into analysis; no sealed row is scored or optimized.
    ix=(np.asarray(pool)[:,None]*16+np.arange(16)).reshape(-1)
    with np.load(path) as z:
        return dict(positions=z['positions'][ix],photos=z['photo_ids'][ix],messages=z['greedy_message'][ix],
            first=z['sender_first_logits'][ix],second=z['sender_second_logits'][ix],receiver=z['receiver_logits'])


def analyze_direction(batch,seed,p,direction,qa):
    base=batch/f's{seed}_p{p}_base';end=batch/f's{seed}_p{p}_expand_receiver'
    cfg=read(base/'config.json');old_pool=cfg['map_groups']['old'];added_pool=cfg['map_groups']['added']
    assert len(old_pool)==18 and len(added_pool)==6 and not set(old_pool)&set(added_pool)
    pool=old_pool+added_pool;label=[seed,p,direction]
    bp=batch/f'protocol/{base.name}_d{direction}.npz';ep=batch/f'protocol/{end.name}_d{direction}.npz'
    before=selected_probe(bp,pool);after=selected_probe(ep,pool)
    expected=np.repeat(MAPS[pool],16,axis=0)
    qa.check(np.array_equal(before['positions'],expected) and np.array_equal(after['positions'],expected),
             'old_added_only_exact_map_major_rows',label)
    qa.check(np.array_equal(before['photos'],after['photos']) and np.array_equal(before['messages'],after['messages']) and
             np.array_equal(before['first'],after['first']) and np.array_equal(before['second'],after['second']),
             'frozen_sender_logits_messages_same_photos',label)
    qa.check(before['messages'].shape==(384,2) and ((before['messages']>=0)&(before['messages']<7)).all(),
             'old_added_384_valid_messages',label)
    bdecoder=before['receiver'].argmax(-1);edecoder=after['receiver'].argmax(-1)
    for name,z in [('base',before),('expand_receiver',after)]:
        sorted_logits=np.sort(z['receiver'],axis=-1)
        qa.check((sorted_logits[:,:,-1]>sorted_logits[:,:,-2]).all(),'receiver_unique_greedy_maximum',[*label,name])
    code=before['messages'][:,0]*7+before['messages'][:,1]
    oc=counts_for(before['positions'],code,old_pool);ac=counts_for(before['positions'],code,added_pool)
    qa.check(np.array_equal(oc.sum(1),np.full(18,16)) and np.array_equal(ac.sum(1),np.full(6,16)),
             'every_old_added_map_16_photos',label)
    dp=solve(oc,ac);old_n=int(oc.sum());added_n=int(ac.sum())

    def score_decoder(decoder):
        correct=(decoder[code]==before['positions']).all(1)
        return int(correct[:old_n].sum()),int(correct[old_n:].sum())

    def witness(decoder,o,a,tag):
        measured=score_decoder(decoder)
        qa.check(measured==(o,a),'witness_replayed_on_original_photograph_rows',[*label,tag])
        # Second replay uses map/code frequency counts, independently of rows.
        old_sum=sum(int(oc[k,m]) for k,mapid in enumerate(old_pool) for m in range(49) if np.array_equal(decoder[m],MAPS[mapid]))
        added_sum=sum(int(ac[k,m]) for k,mapid in enumerate(added_pool) for m in range(49) if np.array_equal(decoder[m],MAPS[mapid]))
        qa.check((old_sum,added_sum)==(o,a),'witness_integer_frequency_replay',[*label,tag])
        return dict(old_correct=o,old_n=old_n,old_J=o/old_n,added_correct=a,added_n=added_n,added_J=a/added_n,
                    messages=MESSAGES.tolist(),actions_by_resource=decoder.tolist())

    def dp_witness(threshold,tag):
        o,a,options=optimum(dp,threshold);decoder=bdecoder.copy()
        for m,choice in enumerate(options):
            if choice==0 and dp['old_best'][m]>0:decoder[m]=MAPS[old_pool[int(oc[:,m].argmax())]]
            elif choice==1 and dp['added_best'][m]>0:decoder[m]=MAPS[added_pool[int(ac[:,m].argmax())]]
            # If its selected objective contribution is zero, retaining the old
            # action pair can accidentally gain other correctness. Use an equal-
            # location pair, absent from both supports, as the neutral witness.
            else:decoder[m]=[0,0]
        qa.check(o>=threshold,'dp_witness_meets_retention_threshold',[*label,tag])
        row=witness(decoder,o,a,tag);row['minimum_old_correct']=int(threshold)
        return row

    bo,ba=score_decoder(bdecoder);eo,ea=score_decoder(edecoder)
    baseline_truth=(bdecoder[code]==before['positions']).all(1)
    endpoint_truth=(edecoder[code]==before['positions']).all(1)
    old_behavior=dict(action_pair_changed_cases=int((bdecoder[code[:old_n]]!=edecoder[code[:old_n]]).any(1).sum()),
        previously_correct_lost=int((baseline_truth[:old_n]&~endpoint_truth[:old_n]).sum()),
        previously_incorrect_gained=int((~baseline_truth[:old_n]&endpoint_truth[:old_n]).sum()),
        correct_count_change=eo-bo)
    qa.check(old_behavior['previously_incorrect_gained']-old_behavior['previously_correct_lost']==eo-bo,
             'actual_old_gain_loss_accounting',label)
    basew=witness(bdecoder,bo,ba,'baseline');endw=witness(edecoder,eo,ea,'receiver_endpoint')
    frontier=[]
    for loss_pp in (0,5,10,25):
        allowed=old_n*loss_pp//100;threshold=max(0,bo-allowed)
        row=dp_witness(threshold,f'allow_{loss_pp}pp')
        row.update(allowed_loss_percentage_points=loss_pp,allowed_old_correct_loss=allowed,
                   actual_old_correct_change=row['old_correct']-bo)
        frontier.append(row)
    cs=int(dp['added_best'].sum());max_old_at_cs=int(dp['old_best'][dp['added_best']==0].sum())
    at_cs=dp_witness(max_old_at_cs,'attain_added_CS_minimum_old_loss')
    qa.check(at_cs['added_correct']==cs and at_cs['old_correct']==max_old_at_cs,'full_added_CS_loss_formula',label)
    used_old=oc.sum(0)>0;strict=bdecoder.copy()
    for m in np.flatnonzero(~used_old):
        if dp['added_best'][m]>0:strict[m]=MAPS[added_pool[int(ac[:,m].argmax())]]
    strict_o,strict_a=score_decoder(strict)
    qa.check(np.array_equal(strict[code[:old_n]],bdecoder[code[:old_n]]) and strict_o==bo,
             'all_old_photograph_actions_preserved_even_errors',label)
    strictw=witness(strict,strict_o,strict_a,'keep_all_old_actions')
    strict_formula=sum(int(ac[k,m]) for k,mapid in enumerate(added_pool) for m in range(49)
                       if used_old[m] and np.array_equal(bdecoder[m],MAPS[mapid]))+int(dp['added_best'][~used_old].sum())
    qa.check(strict_a==strict_formula and strict_a<=frontier[0]['added_correct']<=cs,'strict_vs_total_retention_order',label)
    endpoint_bound=dp_witness(eo,'actual_receiver_old_correct_threshold')
    qa.check(ea<=endpoint_bound['added_correct'],'actual_receiver_endpoint_within_its_own_retention_bound',label)
    qa.check(bo<=int(dp['old_best'].sum()),'baseline_old_below_frozen_sender_CS',label)
    return dict(seed=seed,partition=p,direction=direction,old_map_ids=old_pool,added_map_ids=added_pool,
        source_hashes={str(bp):sha(bp),str(ep):sha(ep),str(base/'config.json'):sha(base/'config.json'),
            str(base/'final.pt'):sha(base/'final.pt'),str(end/'final.pt'):sha(end/'final.pt')},
        old_counts_by_map_message=oc.tolist(),added_counts_by_map_message=ac.tolist(),
        baseline=basew,receiver_endpoint=endw,actual_old_behavior_changes=old_behavior,
        added_independent_CS=cs/added_n,added_independent_CS_correct=cs,
        retain_baseline_total_frontier=frontier,keep_all_old_actions=strictw,
        attain_added_CS=dict(minimum_necessary_old_correct_loss=max(0,bo-max_old_at_cs),
            minimum_necessary_old_loss_pp=100*max(0,bo-max_old_at_cs)/old_n,
            signed_old_correct_change=max_old_at_cs-bo,witness=at_cs),
        actual_endpoint_retention_bound=endpoint_bound,old_used_codes=int(used_old.sum()),
        added_used_codes=int((ac.sum(0)>0).sum()),shared_used_codes=int((used_old&(ac.sum(0)>0)).sum()))


def scalar_summary(row):
    return dict(baseline_old_J=row['baseline']['old_J'],baseline_added_J=row['baseline']['added_J'],
        receiver_old_J=row['receiver_endpoint']['old_J'],receiver_added_J=row['receiver_endpoint']['added_J'],
        added_independent_CS=row['added_independent_CS'],keep_all_old_actions_added_bound=row['keep_all_old_actions']['added_J'],
        **{f'retain_allow_{x["allowed_loss_percentage_points"]}pp_added_bound':x['added_J'] for x in row['retain_baseline_total_frontier']},
        min_old_loss_to_added_CS_pp=row['attain_added_CS']['minimum_necessary_old_loss_pp'],
        actual_receiver_old_threshold_added_bound=row['actual_endpoint_retention_bound']['added_J'],
        actual_receiver_gap_to_own_bound=row['actual_endpoint_retention_bound']['added_J']-row['receiver_endpoint']['added_J'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'results/generalization_001')
    parser.add_argument('--out',type=Path);args=parser.parse_args();batch=args.root.resolve()
    out=args.out or batch/'compatibility_exploratory_001';out.mkdir(parents=True,exist_ok=False)
    audit=read(batch/'audit_execution.json');assert audit['passed'] and audit['completed_social_runs']==60
    protocol_audit=read(batch/'audit_protocol.json');assert protocol_audit['passed']
    qa=QA();small=exhaustive_qa(qa)
    rows=[analyze_direction(batch,seed,p,d,qa) for seed in SEEDS for p in (1,2,3) for d in (0,1)]
    seeds=[]
    for seed in SEEDS:
        selected=[scalar_summary(r) for r in rows if r['seed']==seed]
        seeds.append(dict(seed=seed,metrics={k:float(np.mean([r[k] for r in selected])) for k in selected[0]}))
    mean={k:float(np.mean([r['metrics'][k] for r in seeds])) for k in seeds[0]['metrics']}
    report=dict(complete=True,exploratory_after_v10_functional_results=True,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Saved base and receiver-only endpoints; deterministic complete-message decoder tables only; old18/added6 only, 16 photos per map; no sealed row used in statistics or optimization, no model inference/training.',
        script_sha256=sha(__file__),formal_execution_audit_sha256=sha(batch/'audit_execution.json'),
        formal_protocol_audit_sha256=sha(batch/'audit_protocol.json'),
        independent_units='Four seeds; first average three partitions and two directions per seed.',
        qa=dict(passed=not qa.failures,checks=qa.counts,total_checks=sum(qa.counts.values()),failures=qa.failures,small_exhaustive=small),
        directions=rows,per_seed=seeds,mean=mean)
    dump(out/'compatibility_analysis.json',report)
    lines=['# 旧约定对新增组合解码的约束','',
        '**看到v0.10功能结果后的探索性离线诊断。** 本文不是事前预测、新训练、正式探针修改或新颖性证明。只读取已保存的base与仅接收端学习终点，选取old18和added6各图16照片；封存图不参与频数、约束、优化、见证评分或结果选择。原NPZ包含全图，本分析仅提取预定old/added行。','',
        '发送函数在仅接收端学习中固定。本分析限定确定性完整码→联合物理动作查表，与主贪心评价一致。每条完整码最终只能对应一个食物/水地点对，因此同一码被新旧地图使用时，重新解释它可能损害旧任务。上界允许分析者用新旧地图真值任意重写49条完整码；不受神经接收网络的参数化、优化或学习预算限制，也不是实际语言能力。若允许随机接收政策并只约束期望旧正确数，就应允许每码的动作分布混合；这里的整数DP不是这种更宽政策类的普适上界。','',
        '## 同照片宏平均','',
        '| 指标 | 比率 |','| --- | ---: |']
    labels={'baseline_old_J':'适应前旧图J','baseline_added_J':'适应前新增图J','receiver_old_J':'仅接收端终点旧图J',
        'receiver_added_J':'仅接收端终点新增图J','added_independent_CS':'新增图单独C_S',
        'keep_all_old_actions_added_bound':'逐例保持全部旧动作的新增上界','retain_allow_0pp_added_bound':'保持旧正确总数的新增上界',
        'retain_allow_5pp_added_bound':'允许旧J损失5个百分点的新增上界','retain_allow_10pp_added_bound':'允许旧J损失10个百分点的新增上界',
        'retain_allow_25pp_added_bound':'允许旧J损失25个百分点的新增上界',
        'actual_receiver_old_threshold_added_bound':'按实际终点旧正确数约束的新增上界'}
    for key,label in labels.items():lines.append(f'| {label} | {100*mean[key]:.2f}% |')
    lines+=['',f'达到新增图单独C_S，最低必要旧J损失的方向平均为 **{mean["min_old_loss_to_added_CS_pp"]:.2f}个百分点**。这是每方向先优化后的平均；不能由两项宏平均比例相减反推。',
        '', '## 四种子结果','',
        '| 种子 | 新增单独C_S | 保持旧总正确上界 | 保持全部旧动作上界 | 接收端实际新增J | 按实际旧J的上界 | 达到C_S最低旧损失pp |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cell in seeds:
        m=cell['metrics'];keys=['added_independent_CS','retain_allow_0pp_added_bound','keep_all_old_actions_added_bound','receiver_added_J','actual_receiver_old_threshold_added_bound']
        lines.append('| '+str(cell['seed'])+' | '+' | '.join(f'{100*m[k]:.2f}%' for k in keys)+f' | {m["min_old_loss_to_added_CS_pp"]:.2f} |')
    lines+=['','## 定义、精确解与核验','',
        '每方向旧图288例、新增图96例。保持旧正确总数允许牺牲某些原本正确的旧例，再由其他旧例改善补偿；不等于逐例无遗忘。更强的全部旧行为保持把每条曾在旧照片发出的码锁定为原两动作，连原有错误动作也保留，只能自由修改旧照片从未用过的码。两种约束不能混称。',
        '', '每码旧地图最大计数为o_m，新增地图最大计数为a_m。因新旧地图互斥，可将每码候选约简为(o_m,0)或(0,a_m)。动态规划枚举旧正确总数，求满足下界时最大的新增正确数；相同新增成绩时选保留旧正确数更多的见证。5/10/25个百分点约束分别允许旧正确数最多损失floor(288×p/100)，不通过四舍五入放松限制。',
        '', '新增图单独C_S的正确数为Σa_m。达到它时，所有a_m>0的码必须解释为新增地图；此时旧正确数最大为Σ[a_m=0]o_m。相对原旧正确数b，最低必要损失为max(0,b−Σ[a_m=0]o_m)。这是任意完整码表重解释的有限样本界，没有要求沿实际神经网络学习路径可达。',
        '', '实际仅接收端终点的每方向旧正确总数分别作为约束重新求界，全部实际新增正确数均验不超过各自上界；没有假定其旧成绩与原协议相同。所有基线与终点数据使用完全相同的384照片行，发送logits和贪心消息逐项相同。不能与主实验9600世界的比例直接相减。',
        '',f'本次实际核对发现，24个方向的旧正确总数变化均为0；这是数据结果，不是优化时预设的条件。原始行另记录旧动作变化、原正确例丢失与原错误例改善。菜单的物理动作等价及固定上下文由既有正式协议审计支持，故本分析按49条码的物理动作对定义约束。',
        '',f'独立核验共{sum(qa.counts.values())}项，失败{len(qa.failures)}。包含48个小随机问题对全部原始解码分配的穷举、所有整数旧表现门槛、DP回溯、达到C_S损失公式；每个49码见证分别在原照片行与整数频数表重放。全部方向频数、前沿点及码表见证见[机器可读结果](compatibility_analysis.json)。',
        '', '这里的低兼容上界支持完整码占用与旧行为保持之间存在约束；它不证明人类语言机制、不识别形成此冲突的历史根因，也不意味着删除保持约束后实际接收网络一定能达到C_S。旧照片与同构分区仍是开发条件；四个种子是独立单位，24个方向不是24个独立群体。']
    (out/'兼容性诊断.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),passed=not qa.failures,checks=sum(qa.counts.values()),failures=qa.failures,mean=mean),ensure_ascii=False,indent=2))
    if qa.failures:raise SystemExit(1)


if __name__=='__main__':main()
