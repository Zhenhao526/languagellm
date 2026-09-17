from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/spatial_001'

def main():
    assert json.loads((BATCH/'training_complete.json').read_text())['status']=='complete'
    cfg=json.loads((BATCH/'social_s31101_p1_control/config.json').read_text())
    selected=[(g,i) for g in ('old','sealed') for i in sorted(cfg['map_groups'][g])[:2]]
    glyphs='@#¥%&*+';data=[];hashes={}
    for arm in ('control','equivariant'):
        for group,mid in selected:
            row=dict(arm=arm,group=group,map_id=mid,steps={})
            for t in (0,600,2400):
                p=BATCH/f'social_s31101_p1_{arm}/protocol_{t:04d}_d0.npz'
                hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(p)
                idx=int(np.flatnonzero(z['map_ids']==mid)[0]);msg=z['greedy_message'][idx]
                code=int(msg[0]*7+msg[1]);action=z['receiver_logits'][code].argmax(-1)
                row['positions']=z['positions'][idx].tolist();row['photo_ids']=z['photo_ids'][idx].tolist()
                row['steps'][str(t)]=dict(message_ids=msg.tolist(),display=''.join(glyphs[int(m)] for m in msg),
                    chosen_places=action.tolist(),both_correct=bool(np.array_equal(action,z['positions'][idx])))
            data.append(row)
    (BATCH/'fixed_examples.json').write_text(json.dumps(dict(rows=data,source_hashes=hashes,
        specification_sha256=hashlib.sha256((ROOT/'展示样例预设.md').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    lines=['# 固定场景中的消息形成记录','','按[展示规范](../../展示样例预设.md)取固定来源和场景，不挑选成功例。地点编号为0–5，每格依次为消息、食物地点/水地点、共同成败。符号只是整数词表的显示，没有人工规定含义。','',
        '| 条件 | 组别 | 真食物/水地点 | 0步 | 600步 | 2400步 |','|---|---|---|---|---|---|']
    for r in data:
        cells=[]
        for t in (0,600,2400):
            s=r['steps'][str(t)];a=s['chosen_places'];cells.append(f"`{s['display']}` → {a[0]}/{a[1]}，"+('成功' if s['both_correct'] else '失败'))
        lines.append(f"| {r['arm']} | {r['group']} | {r['positions'][0]}/{r['positions'][1]} | "+' | '.join(cells)+' |')
    lines+=['','这些样例只展示具体过程。判断训练效应必须使用全部预定来源；单个消息的变化不证明词素或语法涌现。']
    (BATCH/'固定消息形成样例.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
