"""Offline v0.6 receiver-code-space audit using saved JSON, NumPy and no policy.

No Torch/MLX import, checkpoint access, inference or training. The decomposition
concerns a frozen receiver in the saved zero-inventory/zero-history context.
Natural photo-level counts are exact; photo identities cannot be recovered from
the frequency-only codebooks. Donor-stitch reachability is a map-level existential
over the already evaluated donor/photo/goal grid, not the complete 49-code space.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np


WORK = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORK / "redesign_v0.6/results/recombination_001/protocol_analysis.json"
DEFAULT_OUTPUT = WORK / "research_program/codebook_audit_v06_20260915"
COUNT_FIELDS = ("natural_success", "receiver_unreachable_failure", "available_code_not_delivered_failure")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "rate": int(numerator) / int(denominator) if denominator else None}


def counts_summary(rows):
    total = sum(r["denominator"] for r in rows)
    out = {name: fraction(sum(r[name] for r in rows), total) for name in COUNT_FIELDS}
    require(sum(out[name]["numerator"] for name in COUNT_FIELDS) == total,
            "Failure categories do not partition the natural-message denominator")
    failure = total - out["natural_success"]["numerator"]
    out["unreachable_share_of_natural_failures"] = fraction(
        out["receiver_unreachable_failure"]["numerator"], failure)
    out["selection_failure_share_of_natural_failures"] = fraction(
        out["available_code_not_delivered_failure"]["numerator"], failure)
    out["natural_failure"] = fraction(failure, total)
    return out


def decoder_from_saved(direction, vocab, length):
    table, audit = direction["receiver_decoder_table"], direction["menu_audit"]
    messages = list(product(range(vocab), repeat=length))
    require(len(messages) == 49 and len(table) == 49, "Audit requires all 49 possible codes")
    require([tuple(r["message"]) for r in table] == messages, "Code enumeration is incomplete or unordered")
    require(audit["enumerated_messages"] == 49 and audit["goals"] == 2
            and audit["menus"] == 720 and audit["cases"] == 49 * 2 * 720,
            "Unexpected original full-menu enumeration")
    require(audit["all_menu_permutations_physically_equivalent"] is True
            and audit["menus_used_after_check"] == 1,
            "This audit must not silently collapse a menu-dependent receiver")
    actions = np.asarray([r["actions_by_goal"] for r in table], dtype=np.int64)
    require(actions.shape == (49, 2) and ((actions >= 0) & (actions < 6)).all(),
            "Invalid receiver actions")
    expected_counts = np.eye(6, dtype=np.int64)[actions] * 720
    require(np.array_equal(expected_counts, [r["menu_action_counts_by_goal"] for r in table]),
            "Saved action counts contradict physical menu invariance")
    full_actions = np.repeat(actions[:, :, None], 720, axis=2)
    require(hashlib.sha256(full_actions.tobytes()).hexdigest() == audit["physical_action_sha256"],
            "Reconstructed all-menu receiver hash differs from source")
    return dict(zip(messages, map(tuple, actions.tolist())))


def message_counts(rows, decoder, nphotos):
    result = {}
    for row in rows:
        message, n = tuple(row["message"]), row["count"]
        require(message in decoder and message not in result and type(n) is int and n > 0,
                "Invalid, duplicate or out-of-space natural message")
        result[message] = n
    require(sum(result.values()) == nphotos, "Natural photo denominator differs from frequency total")
    return result


def audit_protocol(data):
    """Pure data -> auditable map/goal counts plus descriptive seed summaries."""
    require(data["status"] == "complete" and not data["missing_runs"], "Use the completed v0.6 report")
    map_table = list(permutations(range(6), 2))
    runs, checks = [], {"full_receiver_tables": 0, "natural_map_goal_rows": 0,
                       "saved_scalar_metrics_recomputed": 0, "stitch_target_rows": 0}
    run_keys = set()
    for source_run in data["runs"]:
        seed, condition, plan = source_run["seed"], source_run["condition"], source_run["plan"]
        require((seed, condition) not in run_keys, "Duplicate trained run")
        run_keys.add((seed, condition))
        train, heldout = set(source_run["train_map_ids"]), set(source_run["heldout_map_ids"])
        require(not train & heldout and train | heldout == set(range(30)), "Incomplete map split")
        run = {"seed": seed, "condition": condition, "plan": deepcopy(plan),
               "train_map_ids": sorted(train), "heldout_map_ids": sorted(heldout), "directions": []}
        require({d["scout"] for d in source_run["directions"]} == {0, 1}, "Missing communication direction")
        for source_direction in source_run["directions"]:
            decoder = decoder_from_saved(source_direction, plan["vocab"], plan["length"])
            checks["full_receiver_tables"] += 1
            correct_codes = {m: [list(code) for code, output in decoder.items() if output == m]
                             for m in map_table}
            direction = {"scout": source_direction["scout"], "collector": source_direction["collector"],
                         "decoder_capacity": 49, "menus_enumerated": 720, "menus_after_verified_collapse": 1,
                         "receiver_pair_coverage": {"valid_distinct_resource_maps": sum(bool(x) for x in correct_codes.values()),
                                                    "valid_map_denominator": 30,
                                                    "same_location_outputs": sorted({a for a in decoder.values() if a[0] == a[1]})},
                         "phases": {}}
            for phase_name, phase in source_direction["phases"].items():
                pairs = phase["photo_pairs"]
                require(pairs == data["photo_splits"][phase_name] and len(pairs) == 16
                        and len({tuple(p) for p in pairs}) == 16, "Unexpected photo subset")
                indexed = {(r["map_id"], r["sender_goal"]): r for r in phase["codebook"]}
                require(len(indexed) == len(phase["codebook"]) == 60
                        and set(indexed) == set(product(range(30), range(2))), "Incomplete codebook grid")
                map_rows, all_goal_rows = [], []
                for map_id, target in enumerate(map_table):
                    reachable = bool(correct_codes[target])
                    goal_rows, delivered_by_goal = [], []
                    for goal in (0, 1):
                        saved = indexed[map_id, goal]
                        require((saved["food_location"], saved["water_location"]) == target,
                                "Codebook map ID does not match its physical locations")
                        delivered = message_counts(saved["delivered_messages"], decoder, len(pairs))
                        emitted = message_counts(saved["emitted_messages"], decoder, len(pairs))
                        if not plan["blocked"]:
                            require(delivered == emitted, "Unblocked sent/delivered frequencies differ")
                        else:
                            require(delivered == {(0,) * plan["length"]: len(pairs)}, "Blocked channel is not constant")
                        native = sum(n for code, n in delivered.items() if decoder[code][goal] == target[goal])
                        switched = sum(n for code, n in delivered.items() if decoder[code][1-goal] == target[1-goal])
                        success = sum(n for code, n in delivered.items() if decoder[code] == target)
                        for key, count in (("native", native), ("switched", switched), ("both", success)):
                            require(saved[key] == fraction(count, len(pairs)), "Saved natural metric does not recompute")
                            checks["saved_scalar_metrics_recomputed"] += 1
                        row = {"sender_goal": goal, "denominator": len(pairs),
                               "natural_success": success,
                               "receiver_unreachable_failure": 0 if reachable else len(pairs),
                               "available_code_not_delivered_failure": len(pairs)-success if reachable else 0,
                               "natural_native_success": native, "natural_switched_success": switched,
                               "delivered_messages": deepcopy(saved["delivered_messages"])}
                        goal_rows.append(row); all_goal_rows.append(row); delivered_by_goal.append(delivered)
                        checks["natural_map_goal_rows"] += 1
                    if not plan["known"]:
                        require(delivered_by_goal[0] == delivered_by_goal[1], "Hidden-goal frequency tables differ")
                    map_rows.append({"map_id": map_id, "food_location": target[0], "water_location": target[1],
                                     "map_subset": "unseen" if map_id in heldout else "seen",
                                     "any_of_49_codes_correct_for_both_goals": reachable,
                                     "correct_messages": correct_codes[target], "sender_goals": goal_rows,
                                     "decomposition": counts_summary(goal_rows)})
                subsets = {}
                for name, ids in (("all", set(range(30))), ("seen", train), ("unseen", heldout)):
                    selected = [m for m in map_rows if m["map_id"] in ids]
                    rows = [g for m in selected for g in m["sender_goals"]]
                    summary = counts_summary(rows)
                    summary["receiver_reachable_maps"] = fraction(sum(m["any_of_49_codes_correct_for_both_goals"] for m in selected), len(selected))
                    summary["map_count"] = len(selected)
                    summary["distinct_map_photo_contexts"] = len(selected) * len(pairs)
                    summary["sender_goal_axis_multiplier"] = 2
                    saved = phase["cross_goal"] if name == "all" else phase["cross_goal_groups"][name]
                    require((not rows and saved is None) or (saved is not None and
                            saved["same_message_correct_for_both_goals"] == summary["natural_success"]),
                            "Cross-goal aggregate does not match codebook counts")
                    if saved is not None and not plan["known"]:
                        require(saved["message_unchanged_across_sender_goal"]["rate"] == 1,
                                "Hidden sender-goal axis is not actually identical")
                    subsets[name] = summary
                direction["phases"][phase_name] = {"photo_pairs": deepcopy(pairs),
                    "per_photo_message_identity_available": False, "maps": map_rows, "subsets": subsets}

            stitch_rows = []
            stitch = source_direction.get("heldout_stitch_validation")
            if stitch is not None:
                require({r["map_id"] for r in stitch["target_maps"]} == heldout, "Stitch map coverage differs")
                for saved in stitch["target_maps"]:
                    map_id = saved["map_id"]
                    target = map_table[map_id]
                    validation = direction["phases"]["validation"]["maps"][map_id]
                    food_sources = sorted(i for i in train if map_table[i][0] == target[0])
                    water_sources = sorted(i for i in train if map_table[i][1] == target[1])
                    require(saved["food_source_maps"] == food_sources and saved["water_source_maps"] == water_sources,
                            "Donor source set differs from all matching training maps")
                    ndonors = len(food_sources) * len(water_sources)
                    n = ndonors * 16 * 2
                    correct = saved["both_goals_correct_all"]
                    require(saved["contexts_without_menu"] == n and saved["menus_per_context"] == 1
                            and correct["denominator"] == n and 0 <= correct["numerator"] <= n
                            and correct == fraction(correct["numerator"], n), "Invalid saved stitching denominator")
                    require(saved["natural_target_message_both_goals_correct"] == fraction(
                        validation["decomposition"]["natural_success"]["numerator"] * ndonors, n),
                        "Stitch-expanded natural target count differs from direct photo counts")
                    any_stitch = correct["numerator"] > 0
                    reachable = validation["any_of_49_codes_correct_for_both_goals"]
                    require(not any_stitch or reachable, "A successful hybrid lies outside the claimed receiver reachable set")
                    stitch_rows.append({"map_id": map_id, "food_location": target[0], "water_location": target[1],
                        "any_of_49_codes_correct": reachable,
                        "any_evaluated_donor_stitch_correct": any_stitch,
                        "donor_stitch_success": deepcopy(correct), "food_source_maps": food_sources,
                        "water_source_maps": water_sources, "photo_pairs": 16, "sender_goals": 2,
                        "donor_map_pairs": ndonors, "menus_after_verified_collapse": 1,
                        "category": "stitch_reachable" if any_stitch else
                            "49_reachable_but_no_evaluated_stitch_correct" if reachable else "49_unreachable"})
                    checks["stitch_target_rows"] += 1
                require(stitch["both_goals_correct_all"] == fraction(
                    sum(r["donor_stitch_success"]["numerator"] for r in stitch_rows),
                    sum(r["donor_stitch_success"]["denominator"] for r in stitch_rows)),
                    "Saved stitching pooled count differs")
            direction["heldout_stitch_map_reachability"] = stitch_rows
            direction["stitch_status"] = "saved_validation_grid" if stitch is not None else "not_applicable"
            run["directions"].append(direction)
        runs.append(run)

    # Pool exact descriptive counts, while keeping the four trained seed values.
    # Heldout splits and two directions are repeated measures, not new seeds.
    grouped = defaultdict(list)
    for run in runs:
        family = run["condition"].split("_", 1)[1] if run["condition"].startswith("split") else run["condition"]
        for d in run["directions"]:
            grouped[family, run["seed"]].append(d)
    by_seed = []
    for (family, seed), directions in sorted(grouped.items()):
        selected = [m for d in directions for m in d["phases"]["validation"]["maps"] if m["map_subset"] == "unseen"]
        if not selected:
            continue
        rows = [g for m in selected for g in m["sender_goals"]]
        hybrids = [r for d in directions for r in d["heldout_stitch_map_reachability"]]
        by_seed.append({"family": family, "seed": seed, "direction_count": len(directions),
            "map_direction_cells": len(selected), "distinct_photo_map_direction_contexts": 16 * len(selected),
            "natural_decomposition": counts_summary(rows),
            "receiver_reachable_maps": fraction(sum(m["any_of_49_codes_correct_for_both_goals"] for m in selected), len(selected)),
            "stitch_reachable_maps": fraction(sum(r["any_evaluated_donor_stitch_correct"] for r in hybrids), len(hybrids)),
            "49_reachable_but_no_stitch_maps": fraction(sum(r["category"] == "49_reachable_but_no_evaluated_stitch_correct" for r in hybrids), len(hybrids)),
            "stitch_success": fraction(sum(r["donor_stitch_success"]["numerator"] for r in hybrids),
                                       sum(r["donor_stitch_success"]["denominator"] for r in hybrids))})
    family_rows = []
    for family in sorted({r["family"] for r in by_seed}):
        rows = [r for r in by_seed if r["family"] == family]
        require(len(rows) == 4 and {r["seed"] for r in rows} == set(data["expected_seeds"]),
                "Family does not retain four independent training seeds")
        pooled = {}
        for field in COUNT_FIELDS:
            pooled[field] = fraction(sum(r["natural_decomposition"][field]["numerator"] for r in rows),
                                     sum(r["natural_decomposition"][field]["denominator"] for r in rows))
        failures = sum(r["natural_decomposition"]["natural_failure"]["numerator"] for r in rows)
        pooled["unreachable_share_of_natural_failures"] = fraction(pooled["receiver_unreachable_failure"]["numerator"], failures)
        family_rows.append({"family": family, "training_seeds": [r["seed"] for r in rows],
            "pooled_descriptive_counts": pooled,
            "receiver_reachable_maps": fraction(sum(r["receiver_reachable_maps"]["numerator"] for r in rows), sum(r["receiver_reachable_maps"]["denominator"] for r in rows)),
            "stitch_reachable_maps": fraction(sum(r["stitch_reachable_maps"]["numerator"] for r in rows), sum(r["stitch_reachable_maps"]["denominator"] for r in rows)),
            "seed_values": {field: [r["natural_decomposition"][field]["rate"] for r in rows] for field in COUNT_FIELDS},
            "seed_mean": {field: float(np.mean([r["natural_decomposition"][field]["rate"] for r in rows])) for field in COUNT_FIELDS}})
    return {"status": "completed", "analysis_type": "post_hoc_offline_frozen_codebook_audit",
            "training_or_model_calls": 0, "source_run_count": len(runs), "checks": checks,
            "definitions": {
                "target": "Same fixed delivered message routes to food location under food goal and water location under water goal.",
                "receiver_unreachable_failure": "No code in the complete 49-code receiver table is simultaneously correct for this map's two goals.",
                "available_code_not_delivered_failure": "At least one such code exists, but the naturally delivered code is not one of them.",
                "blocked_condition": "For full_blocked, not-delivered failure includes the imposed channel intervention and must not be called sender-only error.",
                "stitch_reachability": "At least one previously evaluated training-donor/photo/sender-goal hybrid succeeds; this is a map-level existential.",
                "photo_identity_limit": "Frequencies retain exact photo denominators but not which message belonged to which photo pair. No per-photo pairing or hybrid-success attribution is inferred.",
                "independence": "Four training seeds; map splits, directions, goals and photos are repeated observations. The two sender-goal axes are identical in all current hidden-goal runs.",
                "interpretation": "Receiver unreachability is a property of the final joint-training solution, not proof that receiver training alone caused failure. Existing-code selection failure may include perception, encoding and coordination mismatch.",
                "scope": "DINO v0.6 static protocol; independent of Qwen language-history experiments. No new language-formation claim or confirmatory preregistration."},
            "runs": runs, "heldout_validation_by_seed": by_seed, "heldout_validation_by_family": family_rows}


def markdown(result):
    lines = ["# DINO v0.6：未见地图的完整码空间审计", "",
        "本次是已完成训练后的纯 JSON/NumPy 分析，未加载模型或更新参数。判据为同一条消息在两个目标下都指向正确地点；它比单个原生目标成功更严格。", "",
        f"核对 {result['source_run_count']} 个运行、{result['checks']['full_receiver_tables']} 个方向接收表；每表含全部 49 码×2 目标×720 菜单。保存的动作频数及原完整表哈希均一致，菜单等价后才按 1 个物理结果计数。", "",
        "对每张地图，先枚举接收表是否存在双目标正确消息，再用自然发信频数分解：自然成功 + 完整码空间不可达失败 + 有正确码但自然未传递该码的失败 = 全部照片记录。后两项是冻结协议的精确分类，不是训练机制的因果归因。", "",
        "| 条件 | 自然双目标成功 | 49码不可达失败 | 有正确码但未选中 | 49码可达地图单元 | 人工拼接可达地图单元 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: f"{x['numerator']}/{x['denominator']}" if x['denominator'] else "不适用"
    for row in result["heldout_validation_by_family"]:
        d = row["pooled_descriptive_counts"]
        lines.append(f"| {row['family']} | {fmt(d['natural_success'])} | {fmt(d['receiver_unreachable_failure'])} | {fmt(d['available_code_not_delivered_failure'])} | {fmt(row['receiver_reachable_maps'])} | {fmt(row['stitch_reachable_maps'])} |")
    lines += ["", "各条件自然消息分母 4608 = 4 种子×3 划分×2 方向×6 未见地图×16 验证照片对×2 发送目标。发送者不知道目标，两目标下消息相同，因此独特的照片—地图—方向记录为 2304；目标复制不增加独立证据。地图单元分母 144 = 4×3×2×6，也不是独立训练样本。JSON 同时保存各种子结果、校准/验证照片列表、每图每方向的精确计数和正确码集合。", "",
        "人工拼接可达只表示：在原先固定的符号位置分工与全部训练地图供体组合中，至少一个已评拼接消息对该图成功。每图的拼接频率分母另为 4×4 供体地图对×16 照片对×2 发送目标 = 512。它不能替代完整49码可达性，也不能表示每张照片都有可用拼接。原子49码没有符号片段拼接指标。", "",
        "在四种留出条件中，多数自然双目标失败对应接收者现有码表没有正确码。只改发送者选择、保持接收表不变，无法消除这些失败；有正确码但未选中的部分才具备在固定接收者下修复的可能。该结论不区分训练期间是发送者未覆盖相关码、接收者未形成映射，还是双方共同训练路径造成了缺口。", "",
        "现有 JSON 不保留逐照片消息的对应关系，不能恢复同一照片下的供体拼接轨迹。若需这一层诊断，最小补充为一次固定评估导出 messages[phase,map,photo_pair,sender_goal,token] 和同一已冻结接收表；不需训练。不得从边际频数任意配对照片后称为原评估的可达性。", "",
        "本分析为探索性证据审计，不能将病例分母当样本量，也不将自然失败的分类直接解释为语言、语法或组合能力的形成机制。Qwen 的历史续跑实验另行记录。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = audit_protocol(json.loads(raw))
    files = [args.source, Path(__file__), WORK / "redesign_v0.6/analyze_protocols.py",
             WORK / "redesign_v0.6/analysis_core.py"]
    result["source_files_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result["runtime_check"] = {"torch_loaded": "torch" in sys.modules,
                               "mlx_loaded": any(x == "mlx" or x.startswith("mlx.") for x in sys.modules)}
    require(not any(result["runtime_check"].values()), "An inference runtime was unexpectedly imported")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "codebook_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "码空间审计.md").write_text(markdown(result))
    print(json.dumps({"output": str(args.output.resolve()), "checks": result["checks"],
                      "families": result["heldout_validation_by_family"], "runtime": result["runtime_check"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
