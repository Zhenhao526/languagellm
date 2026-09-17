"""Deterministic, non-selected examples of the trained agents' actual messages."""
import argparse,json
from pathlib import Path
import numpy as np
import support
ROOT=Path(__file__).resolve().parent
SYMBOLS='@#¥%&*+'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    inv=json.loads((out/'invocation.json').read_text());seed=inv['seeds'][0];p=inv['partitions'][0];d=0;t=inv['updates']
    text=['# 实际通信消息：固定展示样本','',
        f'固定第一来源{seed}、坐标重复p{p}、发送方向{d}、终点{t}，共同P6全部地图，每种mask均取字典序第一测试照片对。未按成功与否挑选。',
        '整数0..6仅为便于阅读映射为 `@ # ¥ % & * +`；这不赋予词义。地点编号0..5，资源角色来自模拟器。','',
        '| 训练支持 | 食物/水目标 | 末帧显示 | 自然消息 | 接收食物/水动作 | 双成功 |',
        '|---|---|---|---|---|---|']
    records=[]
    for arm in support.ARMS:
        path=out/'social'/f's{seed}_p{p}_{arm}'/f'protocol_{t:04d}_d{d}.npz'
        raw=dict(np.load(path));acts=raw['receiver_logits'].argmax(-1);ids=support.groups(p,arm)['common_target6']
        for mask in (0,1):
            for mid in ids:
                index=int(np.flatnonzero((raw['map_id']==mid)&(raw['shown']==mask))[0]);tok=raw['tokens'][index];code=int(7*tok[0]+tok[1])
                target=raw['positions'][index];action=acts[code];correct=bool(np.array_equal(target,action))
                text.append(f"| {'路径3' if arm.endswith('3') else '路径2'} | {target[0]}/{target[1]} | {'食物' if mask==0 else '水'} | `{' '.join(SYMBOLS[x] for x in tok)}` | {action[0]}/{action[1]} | {'是' if correct else '否'} |")
                records.append(dict(arm=arm,source_file=str(path.resolve()),row=index,map_id=int(mid),mask=mask,photo_ids=raw['photo_ids'][index].tolist(),target=target.tolist(),tokens=tok.tolist(),actions=action.tolist(),correct=correct))
    text+=['','这些是自然发送与接收的实际记录。两个token是否分别编码资源，需看预定片段干预与随机整码重编码参照；不能凭单条可读字符串判断。']
    (out/'实际消息示例.md').write_text('\n'.join(text)+'\n')
    (out/'message_examples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
