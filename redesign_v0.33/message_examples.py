"""Deterministic endpoint examples for v0.33 team protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import support


ROOT = Path(__file__).resolve().parent
SYMBOLS = "@#¥%&*+"


def label(condition: str) -> str:
    return {"single_full": "单发送者·完整视图", "dual_same_full": "双发送者·冗余完整视图",
            "dual_complementary": "双发送者·互补视图"}[condition]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    invocation = json.loads((out / "invocation.json").read_text())
    seed, partition, update = invocation["seeds"][0], invocation["partitions"][0], invocation["updates"]
    target = support.groups(partition, "dual_complementary")["target12"]
    lines = [
        "# 实际通信消息：v0.33互补观察实验终点", "",
        f"来源 {seed}、坐标面板 p{partition}、终点 {update}。每个条件展示 team0 和 team1，各取排序后最小的3条目标地图与两种时间帧。选择规则只由地图编号、帧和图片编号决定，不按正确率筛选。",
        "整数0..6映射为 `@ # ¥ % & * +`，字符没有预设词义。双发送者条件的两个整数分别是 food sender 与 water sender 的第一 token；接收者将有序 token 对解码成食物位置和水位置。", "",
        "| 条件 | team | 角色 | 目标食物/水 | 帧 | 图片对 | token消息 | 接收动作 | 双成功 |",
        "|---|---:|---|---:|---:|---|---|---:|---|",
    ]
    records = []
    for condition in support.CONDITIONS:
        teams = support.team_slots(condition)
        for team_id in (0, 1):
            folder = out / "social" / f"s{seed}_p{partition}_{condition}"
            path = folder / f"protocol_{update:04d}_team{team_id}.npz"
            with np.load(path, allow_pickle=False) as z:
                raw = {key: z[key] for key in z.files}
            chosen = []
            for shown in (0, 1):
                for map_id in sorted(map(int, target))[:3]:
                    indices = np.flatnonzero((raw["map_id"] == map_id) & (raw["shown"] == shown))
                    if len(indices) == 0:
                        raise ValueError("target example missing")
                    index = sorted(indices.tolist(), key=lambda i: tuple(raw["photo_ids"][i]))[0]
                    chosen.append((shown, map_id, index))
            decoder = raw["receiver_logits"].argmax(-1)
            for shown, map_id, index in chosen:
                tokens = raw["tokens"][index]
                code = int(7 * tokens[0] + tokens[1])
                action = decoder[code]
                target_pos = raw["positions"][index]
                correct = bool(np.array_equal(action, target_pos))
                message = " ".join(SYMBOLS[int(token)] for token in tokens)
                if condition == "single_full":
                    role = f"r{teams[team_id][0]} ← s{teams[team_id][1]}"
                else:
                    role = f"r{teams[team_id][0]} ← sf{teams[team_id][1]}, sw{teams[team_id][2]}"
                lines.append(f"| {label(condition)} | {team_id} | `{role}` | {target_pos[0]}/{target_pos[1]} | {shown} | {tuple(raw['photo_ids'][index])} | `{message}` | {action[0]}/{action[1]} | {'是' if correct else '否'} |")
                records.append({"condition": condition, "team": team_id, "roles": list(teams[team_id]), "source_file": str(path),
                                "row": int(index), "map_id": map_id, "shown": shown, "photo_ids": raw["photo_ids"][index].tolist(),
                                "target": target_pos.tolist(), "tokens": tokens.tolist(), "message": message,
                                "actions": action.tolist(), "correct": correct})
    lines += ["", "这些是保存的自然发送与接收记录。字符本身没有词义标签；跨团队 token 替换、留出位置和伙伴对正确率才是判断协议能否群体复用的依据。"]
    (out / "实际消息示例.md").write_text("\n".join(lines) + "\n")
    (out / "message_examples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "records": len(records), "selection": "first three target maps × two frames × two teams × three conditions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
