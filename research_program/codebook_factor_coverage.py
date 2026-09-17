"""Post-hoc marginal/joint coverage and same-photo reward ceilings for v0.6.

Pure saved-JSON/NumPy analysis: no policy import, checkpoint access or inference.
All-code ceilings hold the greedy (logits.argmax) receiver fixed and allow an oracle to choose one code
per map without knowing the private uniform goal. They are not learned-sender
performance and, for a blocked channel, are not attainable under that channel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .codebook_audit import decoder_from_saved, message_counts, fraction, require
else:
    from codebook_audit import decoder_from_saved, message_counts, fraction, require


WORK = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORK / "redesign_v0.6/results/recombination_001/protocol_analysis.json"
DEFAULT_OUTPUT = WORK / "research_program/codebook_factor_coverage_v06_20260915"
CATEGORIES = ("same_code_both_correct", "both_marginals_but_no_joint_code",
              "exactly_one_marginal_reachable", "neither_marginal_reachable")


def aggregate_maps(rows):
    """Retain exact map cells and matched photo/goal reward counts."""
    counts = Counter(r["category"] for r in rows)
    categories = {name: fraction(counts[name], len(rows)) for name in CATEGORIES}
    require(sum(counts.values()) == len(rows), "Coverage classes do not partition maps")
    reward_fields = ("natural_uniform_private_goal_reward", "full_code_uniform_goal_ceiling",
                     "channel_admissible_uniform_goal_ceiling", "natural_same_message_both_correct")
    out = {name: fraction(sum(r[name]["numerator"] for r in rows),
                          sum(r[name]["denominator"] for r in rows)) for name in reward_fields}
    n = out["natural_uniform_private_goal_reward"]["denominator"]
    natural = out["natural_uniform_private_goal_reward"]["numerator"]
    ceiling = out["full_code_uniform_goal_ceiling"]["numerator"]
    reachable = counts["same_code_both_correct"]
    out.update({"map_direction_cells": len(rows), "coverage_classes": categories,
        "food_marginal_reachable": fraction(sum(r["food_marginal_reachable"] for r in rows), len(rows)),
        "water_marginal_reachable": fraction(sum(r["water_marginal_reachable"] for r in rows), len(rows)),
        "joint_gap_share_of_joint_unreachable_maps": fraction(counts["both_marginals_but_no_joint_code"], len(rows)-reachable),
        "full_code_ceiling_minus_natural": fraction(ceiling-natural, n),
        "one_minus_full_code_ceiling": fraction(n-ceiling, n),
        "distinct_map_photo_contexts": sum(r["photo_pair_count"] for r in rows),
        "goal_axis_multiplier": 2})
    require(0 <= natural <= ceiling <= n, "Natural reward exceeds its fixed-receiver code ceiling")
    return out


def audit_factor_coverage(data):
    """Pure JSON value -> per-map coverage, matched rewards and seed summaries."""
    require(data["status"] == "complete" and not data["missing_runs"], "Source report is incomplete")
    maps = list(permutations(range(6), 2))
    runs = []
    checks = {"receiver_tables": 0, "map_phase_records": 0, "native_goal_counts_verified": 0,
              "uniform_goal_rewards_matched_to_saved_photos": 0}
    for source_run in data["runs"]:
        plan = source_run["plan"]
        require(plan["known"] is False, "The proposed oracle ceiling assumes sender does not know the goal")
        train, heldout = set(source_run["train_map_ids"]), set(source_run["heldout_map_ids"])
        require(not train & heldout and train | heldout == set(range(30)), "Invalid map split")
        run = {"seed": source_run["seed"], "condition": source_run["condition"], "plan": deepcopy(plan),
               "train_map_ids": sorted(train), "heldout_map_ids": sorted(heldout), "directions": []}
        for source_direction in source_run["directions"]:
            decoder = decoder_from_saved(source_direction, plan["vocab"], plan["length"])
            checks["receiver_tables"] += 1
            direction = {"scout": source_direction["scout"], "collector": source_direction["collector"], "phases": {}}
            for phase_name, phase in source_direction["phases"].items():
                photos = phase["photo_pairs"]
                require(photos == data["photo_splits"][phase_name] and len(photos) == 16,
                        "Unexpected calibration/validation photo subset")
                require(phase["cross_goal"]["message_unchanged_across_sender_goal"]["rate"] == 1,
                        "Sender goal copies do not contain the same message")
                book = {(r["map_id"], r["sender_goal"]): r for r in phase["codebook"]}
                require(len(book) == 60 and len(phase["codebook"]) == 60, "Incomplete natural codebook")
                rows = []
                for map_id, target in enumerate(maps):
                    food_codes = [code for code, actions in decoder.items() if actions[0] == target[0]]
                    water_codes = [code for code, actions in decoder.items() if actions[1] == target[1]]
                    joint_codes = [code for code, actions in decoder.items() if actions == target]
                    marginal_food, marginal_water = bool(food_codes), bool(water_codes)
                    require(set(joint_codes) == set(food_codes) & set(water_codes), "Joint code set is inconsistent")
                    category = ("same_code_both_correct" if joint_codes else
                        "both_marginals_but_no_joint_code" if marginal_food and marginal_water else
                        "exactly_one_marginal_reachable" if marginal_food or marginal_water else
                        "neither_marginal_reachable")
                    code_goal_correct = {code: int(actions[0] == target[0]) + int(actions[1] == target[1])
                                         for code, actions in decoder.items()}
                    best = max(code_goal_correct.values())
                    require(best == (2 if joint_codes else 1 if marginal_food or marginal_water else 0),
                            "Reward ceiling and coverage classes disagree")
                    all_counts = []
                    native_per_goal = []
                    for goal in (0, 1):
                        saved = book[map_id, goal]
                        require((saved["food_location"], saved["water_location"]) == target, "Map labels differ")
                        delivered = message_counts(saved["delivered_messages"], decoder, len(photos))
                        emitted = message_counts(saved["emitted_messages"], decoder, len(photos))
                        require(delivered == emitted if not plan["blocked"] else
                                delivered == {(0,) * plan["length"]: len(photos)}, "Unexpected channel delivery")
                        correct = sum(n for code, n in delivered.items() if decoder[code][goal] == target[goal])
                        require(saved["native"] == fraction(correct, len(photos)), "Saved native reward does not recompute")
                        all_counts.append(delivered); native_per_goal.append(correct)
                        checks["native_goal_counts_verified"] += 1
                    require(all_counts[0] == all_counts[1], "Uniform-goal pairing has differing messages")
                    histogram = Counter()
                    for code, count in all_counts[0].items():
                        histogram[code_goal_correct[code]] += count
                    natural_correct = histogram[1] + 2 * histogram[2]
                    require(natural_correct == sum(native_per_goal), "Per-code and native-goal reward differ")
                    n = 2 * len(photos)
                    require(book[map_id, 0]["both"] == book[map_id, 1]["both"] == fraction(histogram[2], len(photos)),
                            "Strict double-goal count differs from the natural code histogram")
                    admissible_best = code_goal_correct[(0,) * plan["length"]] if plan["blocked"] else best
                    require(natural_correct <= admissible_best * len(photos) <= best * len(photos),
                            "Natural reward exceeds channel-admissible or full-code ceiling")
                    row = {"map_id": map_id, "food_location": target[0], "water_location": target[1],
                        "map_subset": "unseen" if map_id in heldout else "seen", "category": category,
                        "food_marginal_reachable": marginal_food, "water_marginal_reachable": marginal_water,
                        "food_correct_codes": [list(c) for c in food_codes],
                        "water_correct_codes": [list(c) for c in water_codes],
                        "same_code_both_correct_codes": [list(c) for c in joint_codes],
                        "uniform_goal_best_value": best / 2,
                        "uniform_goal_optimal_codes": [list(c) for c, value in code_goal_correct.items() if value == best],
                        "photo_pair_count": len(photos), "native_correct_by_private_goal": native_per_goal,
                        "natural_code_photo_counts_by_number_of_correct_goals": {str(i): histogram[i] for i in (0, 1, 2)},
                        "natural_uniform_private_goal_reward": fraction(natural_correct, n),
                        "full_code_uniform_goal_ceiling": fraction(best * len(photos), n),
                        "channel_admissible_uniform_goal_ceiling": fraction(admissible_best * len(photos), n),
                        "natural_same_message_both_correct": fraction(2 * histogram[2], n)}
                    rows.append(row)
                    checks["map_phase_records"] += 1
                subsets = {}
                for name, ids in (("all", set(range(30))), ("seen", train), ("unseen", heldout)):
                    selected = [r for r in rows if r["map_id"] in ids]
                    summary = aggregate_maps(selected)
                    saved = phase["cross_goal"] if name == "all" else phase["cross_goal_groups"][name]
                    require((not selected and saved is None) or (saved is not None
                            and saved["native_goal"] == summary["natural_uniform_private_goal_reward"]
                            and saved["same_message_correct_for_both_goals"] == summary["natural_same_message_both_correct"]),
                            "Matched-photo aggregate differs from the original report")
                    subsets[name] = summary
                    if selected:
                        checks["uniform_goal_rewards_matched_to_saved_photos"] += 1
                direction["phases"][phase_name] = {"photo_pairs": deepcopy(photos), "maps": rows, "subsets": subsets}
            run["directions"].append(direction)
        runs.append(run)

    grouped = defaultdict(list)
    for run in runs:
        family = run["condition"].split("_", 1)[1] if run["condition"].startswith("split") else run["condition"]
        for d in run["directions"]:
            for phase, phase_data in d["phases"].items():
                for row in phase_data["maps"]:
                    grouped[family, run["seed"], phase, row["map_subset"]].append(row)
    by_seed = [{"family": family, "seed": seed, "phase": phase, "map_subset": subset,
                **aggregate_maps(rows)} for (family, seed, phase, subset), rows in sorted(grouped.items())]
    family_keys = sorted({(family, phase, subset) for family, _, phase, subset in grouped})
    by_family = []
    for family, phase, subset in family_keys:
        seed_rows = [r for r in by_seed if (r["family"], r["phase"], r["map_subset"]) == (family, phase, subset)]
        require(len(seed_rows) == 4 and {r["seed"] for r in seed_rows} == set(data["expected_seeds"]),
                "An aggregate does not retain exactly four trained seeds")
        pooled = [r for (f, s, p, u), values in grouped.items() if (f, p, u) == (family, phase, subset) for r in values]
        fields = ("natural_uniform_private_goal_reward", "full_code_uniform_goal_ceiling",
                  "full_code_ceiling_minus_natural", "natural_same_message_both_correct")
        by_family.append({"family": family, "phase": phase, "map_subset": subset,
            "training_seeds": [r["seed"] for r in seed_rows], **aggregate_maps(pooled),
            "seed_values": {name: [r[name]["rate"] for r in seed_rows] for name in fields},
            "seed_means": {name: float(np.mean([r[name]["rate"] for r in seed_rows])) for name in fields}})
    return {"status": "completed", "analysis_type": "post_hoc_exploratory_factor_coverage",
        "new_training_or_inference_calls": 0, "source_run_count": len(runs), "checks": checks,
        "receiver_policy": "frozen greedy physical action obtained from logits.argmax; not stochastic policy support",
        "definitions": {
            "marginal_reachable": "At least one of 49 codes selects the target resource location under that resource's private goal; food and water codes may differ.",
            "joint_gap": "Both marginal predicates hold but no single code is correct under both goals.",
            "oracle_ceiling": "max_code [(food_goal_correct + water_goal_correct)/2], fixed greedy logits.argmax receiver, same empty inventory/history, verified menu invariance; oracle knows the map but not the private goal.",
            "same_photo_comparison": "Use the exact 16 saved photo pairs per phase/map/direction and both uniform private goals; natural reward is rebuilt from delivered message frequencies and verified against source native_goal.",
            "photo_identity_limit": "Photo denominators and reward sums are exact; per-photo message identities were not saved and are not reconstructed.",
            "strict_vs_native": "If a natural code is correct under k in {0,1,2} goals, native uniform-goal reward is k/2; strict success is 1 only when k=2.",
            "independence": "Four trained seeds. Splits and directions are within-seed repeats; 16 photos and duplicate sender-goal axes are not independent training units.",
            "blocked_channel": "full_blocked full-code ceiling is hypothetical and requires lifting the imposed constant channel. Its separate channel-admissible ceiling uses only the zero code.",
            "causal_limit": "Coverage and residual selection gaps describe the final trained protocol. They do not identify which partner or training mechanism caused them, or show that a realizable sender can reach the oracle ceiling.",
            "greedy_limit": "Unreachable means absent from the frozen greedy 49-code action lookup. It does not mean zero stochastic success probability, nor insufficient physical or channel capacity.",
            "scope": "DINO v0.6 static task only; no inference about Qwen history trials, new symbols, or human language origin."},
        "runs": runs, "by_seed": by_seed, "by_family": by_family}


def markdown(result):
    fmt = lambda p: f"{p['numerator']}/{p['denominator']} ({p['rate']:.2%})" if p["denominator"] else "不适用"
    lines = ["# DINO v0.6：边际覆盖、组合缺口与原生回报上界", "",
        "这是训练完成后新增的探索性检验，仅重算保存的接收表和自然发信频数，未加载权重或调用模型。旧码空间审计未改。", "",
        "本文所有可达性与上界均限定冻结的贪心接收策略：源接收表通过logits.argmax选择动作。不可达不表示随机接收策略的成功概率为0，也不表示物理环境或49码信道容量不足。", "",
        "对每个贪心接收者和地图分别检查：有没有一个码能在食物目标下指向正确地点；有没有一个码能在水目标下指向正确地点；是否存在同一个码同时满足二者。前两者都成立而第三者不成立，表示边际动作可达但组合没有共同码；至少一边际未覆盖另行计数。", "",
        "| 留出条件 | 同码双目标可达 | 两边际可达、缺同码 | 仅一边际可达 | 两边际均不可达 | 同照片自然原生回报 | 完整码最优原生回报 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    heldout = [r for r in result["by_family"] if r["phase"] == "validation" and r["map_subset"] == "unseen"]
    for row in heldout:
        c = row["coverage_classes"]
        cells = " | ".join(str(c[name]["numerator"]) for name in CATEGORIES)
        lines.append(f"| {row['family']} | {cells} | {fmt(row['natural_uniform_private_goal_reward'])} | {fmt(row['full_code_uniform_goal_ceiling'])} |")
    lines += ["", "每行覆盖分类分母为144个地图—方向—种子—划分单元。原生回报分母为4608 = 144×16个验证照片对×2个均匀私有目标；每类仅有4个独立训练种子。表中的两目标是同一消息接受两种目标询问，不能当作新的独立照片样本。", "",
        "完整码上界按每图 max_m[(食物正确+水正确)/2] 计算，只能取0、0.5或1。若无同码双正确，但至少一边际可达，上界为0.5；若两边际均不可达，上界为0。同一条自然消息只答对一个目标时，严格双目标指标记0，原生平均回报仍为0.5。因此，低严格成功率不等于原生回报接近0。", "",
        "这项检验补充了上一轮分解：边际覆盖并非全部完整，未见地图的失败同时包含组合缺口和地点/资源动作本身的缺口。组合缺口在四种留出条件中占双目标不可达地图的多数，但不能据此称所有资源地点都已被正确表征。", "",
        "自然回报与上界使用完全相同的协议分析验证照片及地图分布。direct的自然回报是2096/4608=45.486%，不能拿另一个随机9600例评估的45.39%替代或直接作差。JSON保留校准和验证、训练及留出、各方向与四种子的全部结果。", "",
        "完整码上界假设一个知道真实地图的外部选择器能选任意码，保持贪心接收者不变且仍不知道私有目标。它不证明当前视觉发送者可以达到该回报，也不能把上界与自然回报之间的差距单独归因于发送者学习。full_blocked另保存只允许零码的通道上界；其完整49码上界仅是假设解除封锁后的参考。", "",
        "训练地图和完整地图控制的同照片结果：", "",
        "| 条件 | 地图单元 | 自然原生回报 | 完整码上界 | 通道约束上界 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["by_family"]:
        if row["phase"] == "validation" and row["map_subset"] == "seen":
            lines.append(f"| {row['family']} | {row['map_direction_cells']} | {fmt(row['natural_uniform_private_goal_reward'])} | {fmt(row['full_code_uniform_goal_ceiling'])} | {fmt(row['channel_admissible_uniform_goal_ceiling'])} |")
    lines += ["", "该结果不涉及Qwen历史续跑，也不证明新的语言或语法形成。生成过程、伙伴迁移和新组合的形成仍需要另行预先设计实验。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit_factor_coverage(json.loads(args.source.read_bytes()))
    files = (args.source, Path(__file__), Path(__file__).with_name("codebook_audit.py"),
             WORK / "redesign_v0.6/analyze_protocols.py", WORK / "redesign_v0.6/analysis_core.py")
    result["source_files_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result["runtime_check"] = {"torch_loaded": "torch" in sys.modules,
                               "mlx_loaded": any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)}
    require(not any(result["runtime_check"].values()), "Unexpected model runtime import")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "factor_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "边际与组合覆盖审计.md").write_text(markdown(result))
    print(json.dumps({"output": str(args.output.resolve()), "checks": result["checks"],
        "heldout_validation": [r for r in result["by_family"] if r["phase"] == "validation" and r["map_subset"] == "unseen"],
        "runtime_check": result["runtime_check"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
