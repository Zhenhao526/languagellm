"""Display every newly introduced map for a fixed source, direction and photo pair."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/adaptation_001'
SOURCE=OUT/'protocol_analysis.json'
ALPHABET=['@','#','%','&','+','=','?']
data=json.loads(SOURCE.read_text())
assert data['complete']
runs={r['arm']:r for r in data['runs'] if r['seed']==27101 and r['split']==1}
arms=['baseline','sender_only','receiver_only','both']
labels={'baseline':'适应前','sender_only':'仅发送端学习','receiver_only':'仅接收端学习','both':'双方学习'}
directions={arm:next(d for d in runs[arm]['directions'] if d['scout']==0) for arm in arms}
phase=directions['baseline']['phases']['validation']
pair=phase['photo_pairs'][0]
new_maps=[row for row in phase['codebook'] if row['sender_goal']==0 and {row['food_location'],row['water_location']} in ({0,1},{2,3},{4,5})]
assert len(new_maps)==6
lines=['# 固定新地图中的真实消息变化','',
       f'固定来源种子27101、划分1、方向0→1、协议验证第一对照片{pair}，完整列出全部6种新地图及三种适应终点，没有按成功筛选。数字地点0–5对应环境固定地点；下列字符仅将实际整数0–6显示为`@ # % & + = ?`，主体并不读取字符名称。','',
       '| 地图（食物,水） | 阶段 | 实际消息 | 接收食物选择 | 接收水选择 | 两者均正确 |',
       '| --- | --- | --- | ---: | ---: | --- |']
records=[]
for row in new_maps:
    map_id=row['map_id'];target=[row['food_location'],row['water_location']]
    for arm in arms:
        d=directions[arm]
        assert d['phases']['validation']['photo_pairs'][0]==pair
        message=d['phases']['validation']['messages'][map_id][0]
        code=7*message[0]+message[1]
        decoder=d['receiver_decoder_table'][code]
        assert decoder['message']==message
        actions=decoder['actions_by_goal']
        text=''.join(ALPHABET[i] for i in message)
        correct=actions==target
        lines.append(f'| {tuple(target)} | {labels[arm]} | `{text}` {message} | {actions[0]} | {actions[1]} | {"是" if correct else "否"} |')
        records.append(dict(seed=27101,split=1,scout=0,arm=arm,map_id=map_id,positions=target,
                            photo_ids=pair,message=message,actions=actions,both_correct=correct))
lines+=['','每一行的两个选择使用同一条消息且没有中间反馈。此表展示自然生成的消息；人工片段拼接另见协议报告。终点新地图已经参与适应训练，成功不能作为零样本组合泛化证据。',
        '', '[完整协议与计数](protocol_analysis.json)；[对应数值记录](消息实例.json)。']
(OUT/'消息实例.md').write_text('\n'.join(lines)+'\n')
(OUT/'消息实例.json').write_text(json.dumps(dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    selection='Fixed first seed, first split, first direction, first validation photo pair, all six new maps; no success selection',records=records),ensure_ascii=False,indent=2))
print(f'Wrote {len(records)} fixed example rows')
