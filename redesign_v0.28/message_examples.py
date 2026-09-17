"""Fixed complete examples, without selection for successful communication."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/formation_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert json.loads((OUT/'training_complete.json').read_text())['status']=='complete'
    selection=json.loads((ROOT/'data/selection.json').read_text())['images']
    food=min(i for i,r in enumerate(selection) if r['split']=='test' and r['category']=='food')
    water=min(i for i,r in enumerate(selection) if r['split']=='test' and r['category']=='water')
    symbols='@#¥%&*+';records=[];sources={}
    for arm in ('old_old','all_old','all_all'):
        p=OUT/'social'/f's34101_p1_{arm}/protocol_2400_d0.npz';raw=np.load(p);sources[str(p)]=sha(p)
        chosen=(raw['map_id']<6)&(raw['photo_ids'][:,0]==food)&(raw['photo_ids'][:,1]==water)
        assert int(chosen.sum())==12
        actions=raw['receiver_logits'].argmax(-1)
        for i in np.flatnonzero(chosen):
            tokens=raw['tokens'][i];code=int(tokens[0]*7+tokens[1]);target=raw['positions'][i]
            records.append(dict(arm=arm,seed=34101,partition=1,direction=0,map_id=int(raw['map_id'][i]),
                shown=int(raw['shown'][i]),photo_ids=raw['photo_ids'][i].tolist(),target_sites=target.tolist(),
                tokens=tokens.tolist(),display=''.join(symbols[int(t)] for t in tokens),
                receiver_sites=actions[code].tolist(),joint_correct=bool((actions[code]==target).all())))
    output=dict(selection_rule_sha256=sha(ROOT/'消息展示规则.md'),producer_sha256=sha(__file__),raw_hashes=sources,
        rows=records,all_preselected_examples_included=True,updates=0,model_calls=0,display_map=dict(enumerate(symbols)))
    (OUT/'message_examples.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# 固定场景的实际消息','',
        '来源34101、分区1、方向0，固定同一测试照片对和地图0–5，两种末帧mask。共36例，未经成功率筛选。地点为0–5；每一行只有同一条消息，接收者分别按食物、水需求行动。字符只是整数token的显示别名。','',
        '| 条件 | 地图 | 末帧可见 | 真实地点(食物,水) | 消息 | 接收地点(食物,水) | 双成功 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    labels={'old_old':'A','all_old':'B','all_all':'C'}
    for r in records:
        lines.append(f"| {labels[r['arm']]} | {r['map_id']} | {('食物','水')[r['shown']]} | {r['target_sites']} | `{r['display']}` | {r['receiver_sites']} | {'是' if r['joint_correct'] else '否'} |")
    lines+=['','相同消息在不同场景中的使用可能依赖训练支持；此表不赋予单个符号预定意义，也不是组合句法的证据。']
    (OUT/'实际消息示例.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(examples=len(records),all_preselected=True)))

if __name__=='__main__':main()
