"""Deterministic endpoint message examples for topology transfer schedules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import support


SYMBOLS = "@#¥%&*+"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve(); invocation = json.loads((out / "invocation.json").read_text())
    seed, partition, update = invocation["seeds"][0], invocation["partitions"][0], invocation["updates"]
    target = support.groups(partition, "dual_complementary")["target12"]
    lines = ["# 实际通信消息：v0.34 伙伴拓扑迁移", "", f"来源 {seed}、坐标面板 p{partition}、终点 {update}。只展示 dual_complementary 的 team0/team1 在 A（native）、B（训练见过）和 C（训练未见）三种 schedule 下的保存消息。每个 schedule 取排序后最小的3条目标地图与两种时间帧，选择规则不读取正确率。", "整数0..6 映射为 `@ # ¥ % & * +`；两个 token 仍是有序的 food sender→water sender 角色槽。", "", "| schedule | team | 角色 | 目标食物/水 | 帧 | 图片对 | token消息 | 接收动作 | 双成功 |", "|---|---:|---|---:|---:|---|---|---:|---|"]
    records = []
    for schedule in ("A", "B", "C"):
        prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
        teams = support.team_slots("dual_complementary", schedule=schedule)
        for team_id in (0, 1):
            path = out / "social" / f"s{seed}_p{partition}_dual_complementary" / f"{prefix}_{update:04d}_team{team_id}.npz"
            with np.load(path, allow_pickle=False) as z: raw = {key: z[key] for key in z.files}
            decoder = raw["receiver_logits"].argmax(-1)
            for shown in (0, 1):
                for map_id in sorted(map(int, target))[:3]:
                    indices = np.flatnonzero((raw["map_id"] == map_id) & (raw["shown"] == shown))
                    if len(indices) == 0: raise ValueError("target example missing")
                    index = sorted(indices.tolist(), key=lambda i: tuple(raw["photo_ids"][i]))[0]
                    tokens = raw["tokens"][index]; code = int(7 * tokens[0] + tokens[1]); action = decoder[code]; target_pos = raw["positions"][index]; correct = bool(np.array_equal(action, target_pos)); msg = " ".join(SYMBOLS[int(token)] for token in tokens); role = f"r{teams[team_id][0]} ← sf{teams[team_id][1]}, sw{teams[team_id][2]}"
                    lines.append(f"| {schedule} | {team_id} | `{role}` | {target_pos[0]}/{target_pos[1]} | {shown} | {tuple(raw['photo_ids'][index])} | `{msg}` | {action[0]}/{action[1]} | {'是' if correct else '否'} |")
                    records.append({"schedule":schedule,"team":team_id,"roles":list(teams[team_id]),"source_file":str(path),"row":int(index),"map_id":map_id,"shown":shown,"photo_ids":raw["photo_ids"][index].tolist(),"target":target_pos.tolist(),"tokens":tokens.tolist(),"message":msg,"actions":action.tolist(),"correct":correct})
    lines += ["", "这些是保存的自然 sender/receiver 记录。C schedule 的消息来自终点评估而非训练期交互；跨拓扑成功率和跨团队替换指标决定协议是否可迁移。"]
    (out / "实际消息示例.md").write_text("\n".join(lines) + "\n"); (out / "message_examples.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"complete","records":len(records),"selection":"A/B/C schedules × two teams × first three target maps × two frames"},ensure_ascii=False))


if __name__ == "__main__": main()
