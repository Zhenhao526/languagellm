"""Deterministic, predeclared examples of v0.31 endpoint messages."""
import argparse, json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); import support
SYMBOLS = '@#¥%&*+'
PAIR_KEYS = ('i0_j1', 'i0_j3')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    out = args.out.resolve(); inv = json.loads((out / 'invocation.json').read_text())
    seed, panel, update = inv['seeds'][0], inv['partitions'][0], inv['updates']
    target = support.groups(panel, 'fixed_partners')['target12']
    text = [
        '# 实际通信消息：固定预先选取的样本', '',
        f'来源 {seed}、坐标面板 p{panel}、终点 {update}。展示两个发送者—接收者协议（`i0→j1` 与 `i0→j3`），每个协议取排序后最小的3条目标地图、两种遮罩各一条。选择规则只由地图编号、遮罩和照片编号决定，不按正确率筛选。',
        '整数0..6仅映射为 `@ # ¥ % & * +` 便于阅读；字符本身没有预设词义。接收动作由保存的49码 receiver 表取贪心最大值。', '',
        '| 条件 | 协议 | 目标食物/水 | 遮罩 | 图片对 | 消息 | 接收动作 | 双成功 |',
        '|---|---|---:|---|---|---|---:|---|'
    ]
    records = []
    for condition in support.CONDITIONS:
        for pair in PAIR_KEYS:
            sender_id, receiver_id = pair.split('_')
            path = out / 'social' / f's{seed}_p{panel}_{condition}' / f'protocol_{update:04d}_{pair}.npz'
            with np.load(path, allow_pickle=False) as z:
                raw = {key: z[key] for key in z.files}
            chosen = []
            for mask in (0, 1):
                for mid in sorted(map(int, target))[:3]:
                    idx = np.flatnonzero((raw['map_id'] == mid) & (raw['shown'] == mask))
                    if len(idx) == 0: raise ValueError('target example missing')
                    idx = sorted(idx.tolist(), key=lambda i: tuple(raw['photo_ids'][i]))[0]
                    chosen.append((mask, mid, idx))
            for mask, mid, index in chosen:
                tokens = raw['tokens'][index]; code = int(7 * tokens[0] + tokens[1])
                action = raw['receiver_logits'].argmax(-1)[code]; target_pos = raw['positions'][index]
                correct = bool(np.array_equal(action, target_pos))
                msg = ' '.join(SYMBOLS[int(token)] for token in tokens)
                text.append(f'| {"固定伙伴" if condition == "fixed_partners" else "轮换伙伴"} | `{sender_id[1:]}→{receiver_id[1:]}` | {target_pos[0]}/{target_pos[1]} | {"食物帧" if mask == 0 else "水帧"} | {tuple(raw["photo_ids"][index])} | `{msg}` | {action[0]}/{action[1]} | {"是" if correct else "否"} |')
                records.append(dict(condition=condition, pair=pair, source_file=str(path), row=int(index), map_id=mid, mask=mask, photo_ids=raw['photo_ids'][index].tolist(), target=target_pos.tolist(), tokens=tokens.tolist(), message=msg, actions=action.tolist(), correct=correct))
    text += ['', '这些是保存的自然发送与接收记录。单条消息不能证明两个 token 各自编码某种关系；片段重组、整码双射参照和跨主体一致率需要结合正式指标解释。']
    (out / '实际消息示例.md').write_text('\n'.join(text) + '\n')
    (out / 'message_examples.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'records': len(records), 'selection': 'first three sorted target maps × two masks × two protocols × two conditions'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
