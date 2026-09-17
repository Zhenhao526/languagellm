"""Frozen-plan endpoint probe: natural use, greedy coverage, exact soft-policy expectations."""
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_generalization as run
import analyze_protocols as p8
from camp import MAPS,ImageBank,remake_agents,projected_banks,scene_visual,write_json


def softmax(logits):
    x=np.asarray(logits,dtype=np.float64)
    exp=np.exp(x-x.max(-1,keepdims=True))
    return exp/exp.sum(-1,keepdims=True)


def summarize(z,p):
    positions=z['positions'];messages=z['greedy_message'];decoder=z['receiver_logits'].argmax(-1)
    first=softmax(z['sender_first_logits']);second=softmax(z['sender_second_logits'])
    sp=(first[:,:,None]*second).reshape(-1,49)
    rp=softmax(z['receiver_logits'])
    assert np.allclose(sp.sum(1),1,atol=1e-12,rtol=0)
    g1=first.argmax(-1);g2=second[np.arange(len(g1)),g1].argmax(-1)
    assert np.array_equal(np.column_stack((g1,g2)),messages)
    ids=messages[:,0]*7+messages[:,1]
    natural=(decoder[ids]==positions).all(-1)
    reachable=(decoder[None,:,:]==positions[:,None,:]).all(-1).any(-1)
    pf=rp[:,0,positions[:,0]].T;pw=rp[:,1,positions[:,1]].T
    joint=pf*pw
    expected_joint=(sp*joint).sum(1);best_joint=joint.max(1)
    expected_reward=(sp*(.25*(pf+pw)+.5*joint)).sum(1)
    assert np.all(expected_joint <= best_joint+1e-12) and np.all(natural<=reachable)
    assert ((expected_joint>=0)&(expected_joint<=1)).all()
    mids=positions[:,0]*5+positions[:,1]-(positions[:,1]>positions[:,0])
    counts=np.zeros((30,49),dtype=np.int64)
    np.add.at(counts,(mids,ids),1)
    result={}
    for group,pool in dict(run.partition(p),all=np.arange(30)).items():
        mask=np.isin(mids,pool);n=int(mask.sum())
        cs=float(counts[pool].max(0).sum()/n)
        assert natural[mask].mean() <= cs+1e-12
        result[group]=dict(n=n,natural_correct=int(natural[mask].sum()),N=float(natural[mask].mean()),
            U=float(reachable[mask].mean()),Q=float(expected_joint[mask].mean()),B=float(best_joint[mask].mean()),
            expected_mixed_reward=float(expected_reward[mask].mean()),C_S=cs)
    return result


@torch.no_grad()
def probe(folder,bank,photos,out):
    cfg=json.loads((folder/'config.json').read_text())
    seed,p,arm=cfg['seed'],cfg['partition'],cfg['arm']
    prepared=torch.load(folder.parent/f'prepared_{seed}.pt',weights_only=True)
    agents=remake_agents(seed,prepared,7,2,'identity')
    for a,state in zip(agents,torch.load(folder/'final.pt',weights_only=True)):a.load_state_dict(state)
    banks=projected_banks(agents,bank)
    indices=np.asarray(list(product(range(30),range(len(photos)))))
    positions=MAPS[indices[:,0]];photo_ids=photos[indices[:,1]]
    directions=[]
    for scout in (0,1):
        sender=agents[scout];receiver=agents[1-scout]
        physical,table,menu_audit=p8.receiver_table(receiver)
        assert menu_audit['all_menu_permutations_physically_equivalent']
        allmessages=torch.tensor(list(product(range(7),repeat=2)),dtype=torch.int64)
        logits=[]
        for goal in (0,1):
            g=torch.zeros(49,2);g[:,goal]=1
            y,_=receiver.receive(allmessages,g,torch.zeros(49,2),torch.zeros(49,18),torch.arange(6).repeat(49,1))
            logits.append(y.numpy())
        receiver_logits=np.stack(logits,axis=1)
        assert np.array_equal(receiver_logits.argmax(-1),physical[:,:,0])
        n=len(positions)
        h=sender.observe(scene_visual(positions,photo_ids,banks[scout]))
        state=sender.send_context(torch.cat((h,torch.zeros(n,4)),1))
        first=sender.send_out(state).numpy()
        tokens=torch.arange(7).repeat(n)
        second_state=sender.send_recur(sender.send_embedding(tokens),state.repeat_interleave(7,0))
        second=sender.send_out(second_state).numpy().reshape(n,7,7)
        greedy,_,_,_=sender.send(h,torch.zeros(n,2),torch.zeros(n,2),np.random.default_rng(100100),True)
        arrays=dict(positions=positions,photo_ids=photo_ids,greedy_message=greedy.numpy(),
            sender_first_logits=first,sender_second_logits=second,receiver_logits=receiver_logits)
        filename=f'{folder.name}_d{scout}.npz'
        np.savez_compressed(out/filename,**arrays)
        directions.append(dict(scout=scout,collector=1-scout,arrays=filename,menu_audit=menu_audit,
            bounds=summarize(arrays,p),sha256=run.v8.sha(out/filename)))
    return dict(seed=seed,partition=p,arm=arm,source_checkpoint=str(folder/'final.pt'),
        source_sha256=run.v8.sha(folder/'final.pt'),directions=directions)


def main():
    torch.set_num_threads(1)
    batch=ROOT/'results/generalization_001'
    expected=[batch/f's{s}_p{p}_{arm}' for s,p,arm in product(run.SEEDS,(1,2,3),['base']+run.ARMS)]
    assert all((folder/'result.json').exists() for folder in expected),'Only probe after all planned training is complete'
    manifest=json.loads((ROOT/'protocol_fixed_manifest.json').read_text())
    for path,digest in manifest['hashes'].items():assert run.v8.sha(path)==digest
    out=batch/'protocol';out.mkdir(exist_ok=False)
    bank=ImageBank();photos=np.asarray(list(product(bank.pools['test',0][4:8],bank.pools['test',1][4:8])))
    records=[]
    for folder in expected:
        records.append(probe(folder,bank,photos,out));print(f'PROBE {folder.name}',flush=True)
    rows=[]
    for seed,arm in product(run.SEEDS,['base']+run.ARMS):
        selected=[d for r in records if r['seed']==seed and r['arm']==arm for d in r['directions']]
        assert len(selected)==6
        rows.append(dict(seed=seed,arm=arm,groups={group:{key:float(np.mean([d['bounds'][group][key] for d in selected]))
            for key in ('N','U','Q','B','expected_mixed_reward','C_S')} for group in ('old','added','sealed','all')}))
    aggregates={arm:{group:{key:float(np.mean([r['groups'][group][key] for r in rows if r['arm']==arm]))
        for key in ('N','U','Q','B','expected_mixed_reward','C_S')} for group in ('old','added','sealed','all')} for arm in ['base']+run.ARMS}
    result=dict(complete=True,runs=records,seed_cells=rows,aggregate=aggregates,photo_pairs=photos.tolist(),
        source_hashes=manifest['hashes'],
        probability_numerics='float64 stable softmax from saved native float32 logits; analytic categorical-policy expectation, not a bit-exact model of finite PRNG bins',
        unit='four seeds; average three partitions and two directions within each seed')
    write_json(batch/'protocol_analysis.json',result)
    lines=['# 始终未训练组合的协议探针','',
        '本表只报告每模型封存6图，固定16照片对。N为自然贪心双成功，U为冻结贪心接收覆盖，Q为随机分类政策的解析联合成功概率，B为固定随机接收器的最优整码概率上界，C_S为按封存图标签单独最优重新赋义上界。B与C_S由分析者使用目标真值，不能当作主体自然能力。','',
        '| 条件 | N | U | Q | B | 封存单独C_S |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ['base']+run.ARMS:
        row=aggregates[arm]['sealed'];lines.append('| '+arm+' | '+' | '.join(f'{100*row[k]:.2f}%' for k in ('N','U','Q','B','C_S'))+' |')
    lines+=['','所有源logits、自然消息与照片映射在protocol/*.npz，使用float64稳定softmax重算数学政策下的概率。并非逐比特模拟有限精度随机数抽样。主实验9600世界使用不同照片配对口径，二者不直接相减。','',
        '[全部数据与各图组](protocol_analysis.json)；[预先固定口径](../../协议探针方案.md)。']
    (batch/'协议探针报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregates,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
