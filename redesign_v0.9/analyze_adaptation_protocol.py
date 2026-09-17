"""Fixed before adaptation: read-only endpoint protocol and reachability probes."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.8'))
import analyze_protocols as p8
from camp import MAPS, ImageBank, remake_agents, projected_banks, split_maps, write_json
from run_adaptation import SEEDS, ARMS, SOURCE


def bounds(messages, decoder, pools):
    assert decoder.shape == (49,2,1)
    assert np.array_equal(messages[:,:,0],messages[:,:,1])
    codes=p8.code_ids(messages[:,:,0],7)
    counts=np.asarray([np.bincount(row,minlength=49) for row in codes])
    outputs=decoder[:,:,0]
    reachable=(outputs[None,:,:]==MAPS[:,None,:]).all(-1).any(-1)
    natural=(outputs[codes]==MAPS[:,None,:]).all(-1)
    result={}
    for name,ids in pools.items():
        c=counts[ids];n=int(c.sum());j=int(natural[ids].sum())
        cr=int(reachable[ids].sum())/len(ids)
        cs=int(c.max(0).sum())/n
        assert j/n <= cr+1e-12 and j/n <= cs+1e-12
        result[name]=dict(n=n,natural_correct=j,natural_j=j/n,
            receiver_coverage=cr,sender_optimal_decoder=cs,
            unavailable_receiver=1-cr,available_but_not_used=cr-j/n)
    return result


@torch.no_grad()
def endpoint(seed,split,arm,path,bank,photos):
    prepared=torch.load(SOURCE/f'prepared_{seed}.pt',weights_only=True)
    agents=remake_agents(seed,prepared,7,2,'identity')
    for agent,state in zip(agents,torch.load(path,weights_only=True)):agent.load_state_dict(state)
    banks=projected_banks(agents,bank)
    old,new=split_maps(split)
    plan=dict(known=False,blocked=False)
    directions=[]
    for scout in range(2):
        decoder,table,audit=p8.receiver_table(agents[1-scout])
        assert audit['all_menu_permutations_physically_equivalent']
        messages={};phase_records={}
        for phase,pairs in photos.items():
            emitted,delivered=p8.natural_messages(agents[scout],banks[scout],pairs,plan)
            messages[phase]=delivered
            phase_records[phase]=dict(photo_pairs=pairs.tolist(),messages=delivered[:,:,0].tolist(),
                bounds=bounds(delivered,decoder,{'old':old,'new':new,'all':np.arange(30)}),
                codebook=p8.codebook(emitted,delivered,decoder,7))
        support=set(p8.code_ids(messages['calibration'][old],7).flatten().tolist())
        assignment,candidates=p8.choose_assignment(messages['calibration'],decoder,7,old,support)
        fragments=p8.fragment_report(messages['validation'],decoder,7,assignment,np.arange(30),support)
        stitch=p8.heldout_stitch_report(messages['validation'],decoder,7,assignment,support,old,new)
        reference_seed=int(np.random.SeedSequence([9010,seed,split,scout]).generate_state(1,dtype=np.uint64)[0] >> np.uint64(1))
        reference=p8.whole_message_recoding_reference(messages,decoder,assignment,old,
            seed=reference_seed,replicates=100,heldout_maps=new)
        directions.append(dict(scout=scout,collector=1-scout,receiver_decoder_table=table,
            menu_audit=audit,phases=phase_records,assignment=list(assignment),calibration_candidates=candidates,
            fragments=fragments,fragments_old=p8.fragment_report(messages['validation'],decoder,7,assignment,old,support),
            old_donor_new_target_stitch=stitch,whole_message_recoding=reference))
    return dict(seed=seed,split=split,arm=arm,checkpoint=str(path),checkpoint_sha256=p8.sha(path),directions=directions)


def summarize(runs):
    rows=[]
    for r in runs:
        for d in r['directions']:
            reference=d['whole_message_recoding']
            row=dict(seed=r['seed'],split=r['split'],arm=r['arm'],scout=d['scout'],
                strict=d['fragments']['strict_transfer_all']['rate'],
                stitch=d['old_donor_new_target_stitch']['both_goals_correct_all']['rate'],
                strict_excess=reference['strict_transfer_all']['observed_minus_reference_mean'],
                stitch_excess=reference['heldout_stitch_both_goals_correct_all']['observed_minus_reference_mean'])
            for name,b in d['phases']['validation']['bounds'].items():
                for k in ('natural_j','receiver_coverage','sender_optimal_decoder'):
                    row[f'{name}_{k}']=b[k]
            rows.append(row)
    baseline={(r['seed'],r['split'],d['scout']):d for r in runs if r['arm']=='baseline' for d in r['directions']}
    for r in runs:
        for d in r['directions']:
            before=baseline[(r['seed'],r['split'],d['scout'])]
            bmsg=np.asarray(before['phases']['validation']['messages'])
            msg=np.asarray(d['phases']['validation']['messages'])
            bdecoder=np.asarray([v['actions_by_goal'] for v in before['receiver_decoder_table']])
            decoder=np.asarray([v['actions_by_goal'] for v in d['receiver_decoder_table']])
            drift=dict(message_changed=float((msg!=bmsg).any(-1).mean()),decoder_changed=float((decoder!=bdecoder).any(-1).mean()))
            if r['arm']=='sender_only':assert drift['decoder_changed']==0
            if r['arm']=='receiver_only':assert drift['message_changed']==0
            d['drift_from_baseline']=drift
            row=next(v for v in rows if (v['seed'],v['split'],v['arm'],v['scout'])==(r['seed'],r['split'],r['arm'],d['scout']))
            row.update(drift)
    keys=[k for k in rows[0] if k not in ('seed','split','arm','scout')]
    per_seed=[]
    for seed,arm in product(SEEDS,['baseline']+ARMS):
        selected=[r for r in rows if r['seed']==seed and r['arm']==arm]
        assert len(selected)==6
        per_seed.append(dict(seed=seed,arm=arm,**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
    aggregate={arm:{k:float(np.mean([r[k] for r in per_seed if r['arm']==arm])) for k in keys} for arm in ['baseline']+ARMS}
    return dict(direction_rows=rows,per_seed=per_seed,aggregate=aggregate)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/adaptation_001')
    args=parser.parse_args();torch.set_num_threads(1)
    cache=args.out/'protocol_cache';cache.mkdir(exist_ok=True)
    sources={str(f):p8.sha(f) for f in (Path(__file__),ROOT/'协议追踪方案.md',ROOT.parent/'redesign_v0.8/analyze_protocols.py',ROOT.parent/'redesign_v0.8/camp.py')}
    bank=ImageBank()
    photos={phase:np.asarray(list(product(bank.pools['test',0][ss],bank.pools['test',1][ss])),dtype=np.int64)
        for phase,ss in [('calibration',slice(0,4)),('validation',slice(4,8))]}
    assert not (set(photos['calibration'].flatten()) & set(photos['validation'].flatten()))
    runs=[]
    for seed,split,arm in product(SEEDS,[1,2,3],['baseline']+ARMS):
        path=(SOURCE/f's{seed}_split{split}_mixed' if arm=='baseline' else args.out/f's{seed}_split{split}_{arm}')/'final.pt'
        assert path.exists(),path
        dest=cache/f's{seed}_split{split}_{arm}.json'
        fingerprints=dict(sources=sources,checkpoint_sha256=p8.sha(path))
        if dest.exists():
            saved=json.loads(dest.read_text());assert saved['fingerprints']==fingerprints
            run=saved['run']
        else:
            run=endpoint(seed,split,arm,path,bank,photos)
            write_json(dest,dict(fingerprints=fingerprints,run=run))
        runs.append(run)
        print(f'PROTOCOL {seed} split{split} {arm}',flush=True)
    summary=summarize(runs)
    result=dict(complete=True,runs=runs,summary=summary,sources=sources,
        independent_training_seed_count=4,directions=96,recoding_scores=9600,unique_reference_rng_seeds=24,
        references_paired_across_arms_and_endpoints=True,
        interpretation='New maps were trained during adaptation; oracle bounds and artificial stitching are analyst interventions, not autonomous generalization.')
    write_json(args.out/'protocol_analysis.json',result)
    write_json(args.out/'protocol_summary.json',summary)
    labels={'baseline':'适应前','sender_only':'仅发送端学习','receiver_only':'仅接收端学习','both':'双方学习'}
    lines=['# 新组合适应前后的协议结构','',
        '四个继承种子；先种子内平均三个划分与两个方向，再等权平均种子。下表为固定验证16照片对枚举，分母与主实验9600随机照片世界不同。终点新6图已参加学习，不是零样本测试。','',
        '| 条件 | 新图自然双成功 | 接收可达C_R | 发送最优解码C_S | 严格片段替换 | 旧供体拼接新图 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,row in summary['aggregate'].items():
        values=[row[k] for k in ('new_natural_j','new_receiver_coverage','new_sender_optimal_decoder','strict','stitch')]
        lines.append('| '+labels[arm]+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
    lines+=['','| 条件 | 片段率减重编码均值 | 拼接率减重编码均值 | 自然消息改变比例 | 完整码解码改变比例 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm,row in summary['aggregate'].items():
        values=[row[k] for k in ('strict_excess','stitch_excess','message_changed','decoder_changed')]
        lines.append('| '+labels[arm]+' | '+' | '.join(f'{100*v:.2f}'+('个百分点' if i<2 else '%') for i,v in enumerate(values))+' |')
    lines+=['','C_R枚举当前接收者的49码是否有任一码能对目标地图完成两次选择；C_S允许分析者按评价标签重新赋义完整码，并非主体实际学到的能力。两种上界均不能等同认知能力的必要条件。',
        '', '严格片段替换在只改变一种资源位置时替换一位消息，要求该资源行动正确迁移、另一资源原有正确行动保持；拼接由分析者选择旧地图消息成分形成新图消息。两者均是结构诊断。',
        '', '每方向端点100个完整码可逆重编码参照保持所有自然动作与频数。24个来源方向共享参考序列用于初末/三臂配对，共9600条件化评分，绝非9600个独立训练样本。初末均只用旧图calibration选位置对应，validation评分。',
        '', '[完整枚举和参考记录](protocol_analysis.json)；[逐种子和逐方向汇总](protocol_summary.json)。']
    (args.out/'协议结构报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['aggregate'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
