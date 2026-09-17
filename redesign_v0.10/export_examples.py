"""Descriptive fixed-order display from audited endpoint probes, after results."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/generalization_001'
SYMBOLS='@#%&+=?'
ARMS=['base','stay_old_both','expand_sender','expand_receiver','expand_both']
LABELS=['共同起点','只练旧图','仅发送端','仅接收端','两端学习']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=json.loads((BATCH/'audit_protocol.json').read_text());assert audit['passed']
    protocol=json.loads((BATCH/'protocol_analysis.json').read_text())
    seed=29101;p=1;photo=protocol['photo_pairs'][0]
    config=json.loads((BATCH/f's{seed}_p{p}_base/config.json').read_text())
    group={m:name for name,ids in config['map_groups'].items() for m in ids}
    rows=[];digests={};lookups={}
    for direction in (0,1):
        for arm in ARMS:
            entry=next(r for r in protocol['runs'] if (r['seed'],r['partition'],r['arm'])==(seed,p,arm))
            item=entry['directions'][direction];path=BATCH/'protocol'/item['arrays']
            assert sha(path)==item['sha256'];digests[str(path)]=sha(path)
            with np.load(path) as z:
                decoder=z['receiver_logits'].argmax(-1)
                selected=np.flatnonzero((z['photo_ids']==photo).all(1));assert len(selected)==30
                for mapid,i in enumerate(selected):
                    pos=z['positions'][i].tolist();msg=z['greedy_message'][i].tolist()
                    assert pos[0]*5+pos[1]-(pos[1]>pos[0])==mapid
                    action=decoder[msg[0]*7+msg[1]].tolist()
                    row=dict(seed=seed,partition=p,direction=direction,map_id=mapid,
                        group=group[mapid],photo_ids=photo,arm=arm,positions=pos,
                        message=msg,display=''.join(SYMBOLS[t] for t in msg),
                        actions=action,joint_correct=pos==action)
                    rows.append(row);lookups[direction,arm,mapid]=row
    assert len(rows)==300
    lines=['# 固定情境中的实际消息与动作','',
        f'描述性展示，选例规则在功能结果之后指定：第一种子{seed}、第一划分{p}、固定照片表第一对{photo}，两个方向、全部30种地图及五个阶段全部列出，不按成功筛选。这里只展示一个来源和一个照片对，不能替代四种子统计。',
        '', '地点按环境编号0—5。字符只是整数0—6的可读显示 `@ # % & + = ?`，不是主体接收的自然语言。每格为“实际消息 → 食物/水选择”，✓表示两者均正确，×表示至少一项失败。一条自然消息支持两次独立选择，没有人工拼接或中间反馈。','']
    names={'old':'旧','added':'新增','sealed':'始终未训练'}
    counts=[]
    for d in (0,1):
        lines += [f'## 方向{d}→{1-d}','', '| 图组 | 真地图（食物/水） | '+' | '.join(LABELS)+' |',
                  '| --- | --- | '+' | '.join(['---']*5)+' |']
        for m in range(30):
            first=lookups[d,'base',m];cells=[]
            for arm in ARMS:
                r=lookups[d,arm,m];a=r['actions'];cells.append(f'`{r["display"]}` → {a[0]}/{a[1]} '+('✓' if r['joint_correct'] else '×'))
            pos=first['positions'];lines.append('| '+names[first['group']]+f' | {pos[0]}/{pos[1]} | '+' | '.join(cells)+' |')
        lines+=['', '| 阶段 | 旧18图正确数 | 新增6图正确数 | 始终未训练6图正确数 |','| --- | ---: | ---: | ---: |']
        for arm,label in zip(ARMS,LABELS):
            count={g:sum(int(lookups[d,arm,m]['joint_correct']) for m in config['map_groups'][g]) for g in ('old','added','sealed')}
            counts.append(dict(direction=d,arm=arm,correct=count))
            lines.append(f'| {label} | {count["old"]}/18 | {count["added"]}/6 | {count["sealed"]}/6 |')
        lines.append('')
    lines += ['新增图只在三个扩展条件参与训练；始终未训练图不参与任何社会训练。任一消息形似两段可拆分，不足以证明两位具有稳定独立语义；上表只支持在给定地点框架中的完整码和动作观察。','',
              '[数值记录与来源哈希](消息实例.json)；[全部协议数据](protocol_analysis.json)。']
    result=dict(descriptive_after_results=True,selection=dict(seed=seed,partition=p,photo_ids=photo,
        directions=[0,1],maps='all30',arms=ARMS),symbols=list(SYMBOLS),examples=rows,counts=counts,
        source_hashes=digests,script_sha256=sha(Path(__file__)),protocol_audit_sha256=sha(BATCH/'audit_protocol.json'))
    (BATCH/'消息实例.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (BATCH/'消息实例.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=len(rows),selection=result['selection'],counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
