"""Audit and summarize one Qwen pilot result without raw completions."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .environment import AGENTS, make_episode, score_episode


def _mutual_information(rows: list[dict], left: str, right: str) -> float:
    n = len(rows)
    joint = Counter((row[left], row[right]) for row in rows)
    left_counts = Counter(row[left] for row in rows)
    right_counts = Counter(row[right] for row in rows)
    return sum(
        count / n * math.log2(count * n / (left_counts[x] * right_counts[y]))
        for (x, y), count in joint.items()
    )


def analyze_result(data: dict) -> dict:
    rows = []
    for record in data["records"]:
        episode = make_episode(record["episode"], data["seed"])
        if (episode["owner"] != record["owner"]
                or episode["meaning_id"] != record["meaning_id"]
                or episode["goal"] != record["goal"]):
            raise ValueError(f"episode reconstruction mismatch at {record['episode']}")
        recomputed = score_episode(episode, record["actions"])
        for key in ("success", "reward", "designated_helper_correct", "unassigned_helper_waited"):
            if recomputed[key] != record["outcome"][key]:
                raise ValueError(f"outcome mismatch at episode {record['episode']}: {key}")

        designated = episode["goal"]["partner"]
        other = next(agent for agent in episode["helpers"] if agent != designated)
        action = record["actions"].get(designated)
        other_action = record["actions"].get(other)
        row = dict(record)
        row["item_correct"] = bool(
            action is not None and action.get("item_id") == episode["target_item_id"]
        )
        row["destination_correct"] = bool(
            action is not None and action.get("destination") == episode["goal"]["destination"]
        )
        row["other_waited"] = other_action is None
        if bool(row["item_correct"] and row["destination_correct"] and row["other_waited"]) != record["outcome"]["success"]:
            raise ValueError(f"joint outcome mismatch at episode {record['episode']}")
        rows.append(row)

    by_block = {}
    for block in sorted({row["block"] for row in rows}):
        subset = [row for row in rows if row["block"] == block]
        n = len(subset)
        by_block[str(block)] = {
            "n": n,
            "item_correct": sum(row["item_correct"] for row in subset),
            "destination_correct": sum(row["destination_correct"] for row in subset),
            "designated_item_and_destination_correct": sum(
                row["outcome"]["designated_helper_correct"] for row in subset
            ),
            "other_waited": sum(row["other_waited"] for row in subset),
            "team_success": sum(row["outcome"]["success"] for row in subset),
        }

    pair_rows = defaultdict(list)
    for row in rows:
        pair_rows[(row["owner"], row["meaning_id"])].append(row)
    stable_pairs = sum(
        len(pair) == 2 and all(row["message_valid"] for row in pair)
        and pair[0]["message"] == pair[1]["message"]
        for pair in pair_rows.values()
    )
    final_by_meaning = defaultdict(list)
    for (owner, meaning_id), pair in pair_rows.items():
        final_by_meaning[meaning_id].append(max(pair, key=lambda row: row["block"]))
    cross_sender_agreement = sum(
        len(final_by_meaning[meaning]) == len(AGENTS)
        and all(row["message_valid"] for row in final_by_meaning[meaning])
        and len({row["message"] for row in final_by_meaning[meaning]}) == 1
        for meaning in range(6)
    )
    final_codes_by_sender = {
        owner: sorted({
            row["message"] for row in rows
            if row["owner"] == owner and row["block"] == max(r["block"] for r in rows)
        })
        for owner in AGENTS
    }

    return {
        "episodes": len(rows),
        "valid_messages": sum(row["message_valid"] for row in rows),
        "owner_json_valid": sum(row["owner_json_valid"] for row in rows),
        "helper_json_valid": sum(
            result["json_valid"] for row in rows for result in row["helper_results"].values()
        ),
        "helper_actions_valid": sum(
            result["action_valid"] for row in rows for result in row["helper_results"].values()
        ),
        "distinct_messages": len({row["message"] for row in rows if row["message_valid"]}),
        "stable_sender_meaning_pairs": stable_pairs,
        "sender_meaning_pairs": len(pair_rows),
        "cross_sender_meaning_agreement": cross_sender_agreement,
        "n_meanings": 6,
        "message_owner_mutual_information_bits": _mutual_information(rows, "message", "owner"),
        "message_meaning_mutual_information_bits": _mutual_information(rows, "message", "meaning_id"),
        "block_metrics": by_block,
        "final_codes_by_sender": final_codes_by_sender,
    }


def render_report(data: dict, result_path: str) -> str:
    stats = analyze_result(data)
    model = data["model"]
    revision = data["model_revision"]
    rows = data["records"]
    n = stats["episodes"]
    successes = sum(row["outcome"]["success"] for row in rows)
    lines = [
        "# Qwen3.5 三上下文协作预实验：结果",
        "",
        f"结果文件：`{result_path}`。这是单模型、单种子、36轮的可行性预实验，没有显著性检验。",
        "",
        "## 运行设置",
        "",
        f"模型：`{model}`；固定 revision：`{revision}`。模型 shard 的预期大小和 SHA-256 校验记录见 `model_lock.json`。",
        f"种子：`{data['seed']}`；区组：每个区组18轮；总轮数：{n}；完成时间：{data['wall_time_seconds']:.1f}秒。",
        f"消息通道：`{data['message_length'][0]}–{data['message_length'][1]}` 个字符，字符集 `{data['message_alphabet']}`；三个独立上下文；108次模型调用。",
        "",
        "## 结果",
        "",
        "| 指标 | 观察值 |",
        "|---|---:|",
        f"| requester 消息符合字符协议 | {stats['valid_messages']}/{n} |",
        f"| helper JSON / action 格式有效 | {stats['helper_json_valid']}/{2*n} / {stats['helper_actions_valid']}/{2*n} |",
        f"| 全批次不同消息字符串 | {stats['distinct_messages']} |",
        f"| 同一 requester–含义两次发送完全相同 | {stats['stable_sender_meaning_pairs']}/{stats['sender_meaning_pairs']} |",
        f"| 第二个区组跨 requester 对同一含义使用同一消息 | {stats['cross_sender_meaning_agreement']}/{stats['n_meanings']} |",
        f"| 经验互信息：消息与 requester 身份 | {stats['message_owner_mutual_information_bits']:.3f} bit |",
        f"| 经验互信息：消息与含义编号 | {stats['message_meaning_mutual_information_bits']:.3f} bit |",
        f"| 精确 item 选择 | {sum(x['item_correct'] for x in stats['block_metrics'].values())}/{n} |",
        f"| 精确 destination 选择 | {sum(x['destination_correct'] for x in stats['block_metrics'].values())}/{n} |",
        f"| 指定 helper 同时选对 item 和 destination | {sum(x['outcome']['designated_helper_correct'] for x in rows)}/{n} |",
        f"| 非指定 helper 等待 | {sum(x['outcome']['unassigned_helper_waited'] for x in rows)}/{n} |",
        f"| 联合成功 | {successes}/{n} ({successes/n:.1%}) |",
        "",
        "区组联合成功数：",
        "",
    ]
    for block, values in stats["block_metrics"].items():
        lines.append(
            f"- 区组 {block}：{values['team_success']}/{values['n']}；"
            f"item {values['item_correct']}/{values['n']}，"
            f"destination {values['destination_correct']}/{values['n']}，"
            f"非指定者等待 {values['other_waited']}/{values['n']}。"
        )
    lines += [
        "",
        "第二个区组每个 requester 的最终消息在其六种含义上都收敛成单一字符串：",
        "",
    ]
    for owner, codes in stats["final_codes_by_sender"].items():
        lines.append(f"- `{owner}`：`{', '.join(codes)}`")
    lines += [
        "",
        "## 解释与边界",
        "",
        "这次运行显示模型能够稳定输出满足符号字符约束的 JSON 动作协议，但没有形成可区分六种含义的共享消息码。全批次只有两种消息；第二个区组中 A、C 对所有含义都用同一字符串，B 对所有含义都用另一字符串。同一 requester–含义组合的重复率虽为83.3%，不能解释为语义约定，因为消息主要随 requester 身份而不是含义变化。经验互信息也与此一致：消息—身份高于消息—含义。",
        "",
        f"联合成功只有 {successes}/{n}。区组成功从0/18到1/18不足以证明学习；没有空消息或已知码本基线，不能判断低成功来自通信失败、任务能力不足、奖励稀疏还是曝光轮数太少。该批次也没有环境/任务条件变化，因此不支持因果结论，更不是人类语言起源的模型证据。",
        "",
        "item/destination 分项正确率由固定种子和版本化环境生成器重建候选板后计算；重建时核对了每轮 requester、含义和目标，并逐轮复算原有联合结果。未来 runner 已补上场景与 target 字段，避免继续依赖重建。原始模型 completion 和思维文本均未保留。",
        "",
        "## 下一步",
        "",
        "先做匹配的任务校准，而不是直接扩大自由符号批次：用相同平衡含义、角色和种子，对比空消息、双方已知的固定共享码本、自由符号三种通道。空消息估计无沟通基线，固定码本检验模型能否完成任务和解码约定，自由符号臂才检验约定能否靠互动形成。先用若干配对种子做筛查，再根据结果规划有功效分析的确认性矩阵；随后增加每个发送者–含义组合的曝光次数，并操纵反馈信息量和行动依赖。组合泛化需另用因子平衡的训练/留出含义，不能从当前六个整体含义推断。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    report = render_report(data, args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(json.dumps({"status": "audited", "episodes": len(data["records"]),
                      "successes": sum(r["outcome"]["success"] for r in data["records"]),
                      "raw_model_completions_read": False}, ensure_ascii=False))
