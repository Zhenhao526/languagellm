"""Read-only source/result audit using JSON and NumPy; never deserialize weights.

Writes only 独立核验.json and 独立核验.md beside this script.
"""
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
SEEDS = [25101, 25102, 25103, 25104]
UPDATES = [0, 100, 300, 600, 1200, 1800, 2100, 2400]
CONDITIONS = [f"split{s}_{k}" for s in (1, 2, 3)
              for k in ("course", "mixed", "direct", "atomic_direct")] + ["full_direct", "full_blocked"]
CATS = ["same_code_both_correct", "both_marginals_but_no_joint_code",
        "exactly_one_marginal_reachable", "neither_marginal_reachable"]
FIELDS = CATS + ["food_marginal_reachable", "water_marginal_reachable",
                 "full_code_uniform_goal_ceiling", "channel_admissible_uniform_goal_ceiling"]
MAPS = list(permutations(range(6), 2))
MENUS = np.asarray(list(permutations(range(6))), dtype=np.int64)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            out.update(chunk)
    return out.hexdigest()


def ratio(n, d):
    return {"numerator": int(n), "denominator": int(d), "rate": n / d if d else None}


def recompute_coverage(actions, train_ids, heldout_ids, blocked):
    """Direct per-menu sets and best code scores, independent of producer helpers."""
    nmenus = actions.shape[-1]
    entries = []
    for map_id, (food, water) in enumerate(MAPS):
        classes, food_counts, water_counts, best_sum, allowed_sum = [], 0, 0, 0, 0
        for menu in range(nmenus):
            food_codes = set(np.flatnonzero(actions[:, 0, menu] == food).tolist())
            water_codes = set(np.flatnonzero(actions[:, 1, menu] == water).tolist())
            category = (0 if food_codes & water_codes else 1 if food_codes and water_codes
                        else 2 if food_codes or water_codes else 3)
            classes.append(category)
            food_counts += bool(food_codes)
            water_counts += bool(water_codes)
            scores = [int(m in food_codes) + int(m in water_codes) for m in range(49)]
            best_sum += max(scores)
            allowed_sum += scores[0] if blocked else max(scores)
        counts = Counter(classes)
        entries.append({"map_id": map_id, "food_location": food, "water_location": water,
            "subset": "heldout" if map_id in heldout_ids else "train", "menus": nmenus,
            "food_marginal_reachable": ratio(food_counts, nmenus),
            "water_marginal_reachable": ratio(water_counts, nmenus),
            "coverage_classes": {name: ratio(counts[i], nmenus) for i, name in enumerate(CATS)},
            "full_code_uniform_goal_ceiling": ratio(best_sum, 2 * nmenus),
            "channel_admissible_uniform_goal_ceiling": ratio(allowed_sum, 2 * nmenus),
            "identity_menu_category": CATS[classes[0]],
            "identity_menu_uniform_goal_ceiling": [1., .5, .5, 0.][classes[0]]})
    subsets = {}
    for name, ids in (("all", range(30)), ("train", train_ids), ("heldout", heldout_ids)):
        selected = [entries[i] for i in ids]
        subset = {"map_count": len(selected), "menus_per_map": nmenus,
            "coverage_classes": {cat: ratio(sum(e["coverage_classes"][cat]["numerator"] for e in selected),
                                            len(selected) * nmenus) for cat in CATS}}
        for field in FIELDS[4:]:
            subset[field] = ratio(sum(e[field]["numerator"] for e in selected),
                                  sum(e[field]["denominator"] for e in selected))
        subsets[name] = subset
    return {"maps": entries, "subsets": subsets}


def same_numbers(expected, actual, path=""):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(
            same_numbers(value, actual[key], path + "/" + key) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            same_numbers(x, y, path) for x, y in zip(expected, actual))
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and np.isfinite(actual) and abs(expected - actual) < 1e-12
    return type(expected) is type(actual) and expected == actual


def main():
    status = load(ROOT / "execution/status.json")
    if status.get("status") != "completed":
        raise SystemExit("Execution is not completed; audit not run.")
    plan, freeze = load(ROOT / "plan.json"), load(ROOT / "freeze.json")
    results = load(ROOT / "execution/results.json")
    rows = [json.loads(line) for line in (ROOT / "execution/records.jsonl").read_text().splitlines() if line.strip()]
    source = load(plan["source_protocol"])
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)

    check(sha(ROOT / "plan.json") == freeze["plan_sha256"] == results["plan_sha256"], "plan digest")
    check(sha(ROOT / "冻结计划.md") == freeze["plan_markdown_sha256"], "plan text digest")
    check(sha(ROOT / "execution/records.jsonl") == results["records_sha256"], "record digest")
    frozen_sources = {name: sha(name) == digest for name, digest in plan["source_files_sha256"].items()}
    check(len(frozen_sources) == 686 and all(frozen_sources.values()), "686 source digests")
    snapshots = {name: sha(ROOT / "code_snapshot" / name) == plan["source_files_sha256"][str(WORK / name)]
                 for name in plan["code_files"]}
    check(len(snapshots) == 8 and all(snapshots.values()), "eight source snapshots")
    old_dir = ROOT.parent / "receiver_coverage_trajectory_v06_20260915"
    old_plan = load(old_dir / "plan.json")
    compatibility = {"all_source_hashes_same": old_plan["source_files_sha256"] == plan["source_files_sha256"],
                     "run_list_same": old_plan["runs"] == plan["runs"],
                     "old_package_unexecuted": not (old_dir / "execution").exists(),
                     "new_python": plan["prepare_runtime"]["python"]}
    check(all(compatibility[k] for k in ("all_source_hashes_same", "run_list_same", "old_package_unexecuted")),
          "3.9 to 3.12 preparation equivalence")
    check(plan["seeds"] == SEEDS and plan["checkpoints"] == UPDATES and plan["conditions"] == CONDITIONS,
          "frozen condition, seed and checkpoint grid")
    run_index = {(r["seed"], r["condition"]): r for r in plan["runs"]}
    source_index = {(r["seed"], r["condition"]): r for r in source["runs"]}
    check(len(run_index) == 56 and set(run_index) == set(product(SEEDS, CONDITIONS)), "56 source runs")
    expected_grid = list(product(SEEDS, CONDITIONS, UPDATES, (0, 1)))
    actual_grid = [(r["seed"], r["condition"], r["update"], r["receiver"]) for r in rows]
    check(actual_grid == expected_grid and len(rows) == 896, "complete ordered 56 x 8 x 2 record grid")

    # Static reading only: receiver menu must exclusively permute final location logits.
    import ast
    tree = ast.parse((WORK / "redesign_v0.6/camp.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CampAgent")
    receive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "receive")
    menu_refs = [n for n in ast.walk(receive) if isinstance(n, ast.Name) and n.id == "menu"]
    assign = next(n for n in receive.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "logits" for t in n.targets))
    menu_gather = (len(menu_refs) == 1 and isinstance(assign.value, ast.Call)
                   and isinstance(assign.value.func, ast.Attribute) and assign.value.func.attr == "gather"
                   and ast.dump(assign.value.args[1]) == ast.dump(menu_refs[0]))
    check(menu_gather, "reviewed final menu gather")

    row_checks, grouped = [], defaultdict(list)
    min_margin, tie_rows, forward_calls, evaluated_rows = float("inf"), [], 0, 0
    endpoint_count = 0
    for row in rows:
        key = (row["seed"], row["condition"], row["update"], row["receiver"])
        label = "/".join(map(str, key))
        before = len(errors)
        run = run_index[key[:2]]
        check(row["collector"] == row["receiver"] and row["scout"] == 1-row["receiver"], label + " direction")
        checkpoint = run["checkpoint_files"][str(row["update"])]
        check(row["checkpoint_file"] == checkpoint and row["checkpoint_sha256"] == plan["source_files_sha256"][checkpoint],
              label + " checkpoint source")
        config = load(run["config_file"])
        check(config["train_map_ids"] == run["train_map_ids"] and config["heldout_map_ids"] == run["heldout_map_ids"]
              and set(run["train_map_ids"]) | set(run["heldout_map_ids"]) == set(range(30))
              and not set(run["train_map_ids"]) & set(run["heldout_map_ids"]), label + " map split")
        lookup = row["lookup"]
        logits = np.asarray(lookup["identity_menu_logits"], dtype=np.float32)
        check(logits.shape == (49, 2, 6) and np.isfinite(logits).all(), label + " logits finite/shape")
        count = (logits == logits.max(-1, keepdims=True)).sum(-1)
        margins = np.sort(logits, axis=-1)[..., -1] - np.sort(logits, axis=-1)[..., -2]
        ties = int((count > 1).sum())
        has_tie = ties > 0
        if has_tie:
            tie_rows.append({"key": list(key), "code_goal_ties": ties})
        check(count.tolist() == lookup["maximum_multiplicity"] and margins.tolist() == lookup["top_two_margin"]
              and float(margins.min()) == lookup["minimum_top_two_margin"], label + " multiplicity/margins")
        min_margin = min(min_margin, float(margins.min()))
        check(ties == lookup["tie_code_goal_count"] and has_tie == lookup["complete_menu_enumeration_triggered"], label + " ties")
        table = lookup["receiver_decoder_table"]
        messages = [list(m) for m in product(range(run["plan"]["vocab"]), repeat=run["plan"]["length"])]
        check(len(table) == 49 and [t["message"] for t in table] == messages, label + " complete code list")
        # Reconstruct all 720 physical choices, preserving source first-index argmax at ties.
        ordered = logits[:, :, MENUS]
        predicted_full = MENUS[np.arange(720)[None, None, :], ordered.argmax(-1)].astype(np.int64)
        if has_tie and not lookup["physical_menu_invariant"]:
            full = np.asarray([t["physical_actions_by_goal_and_menu"] for t in table], dtype=np.int64)
            check(full.shape == (49, 2, 720) and np.array_equal(full, predicted_full), label + " all tied menus")
        else:
            full = np.repeat(np.asarray([t["actions_by_goal"] for t in table], dtype=np.int64)[:, :, None], 720, axis=2)
            check(np.array_equal(full, predicted_full), label + " all-menu argmax equivalence")
        check(hashlib.sha256(full.tobytes()).hexdigest() == lookup["physical_action_sha256"], label + " physical table digest")
        invariant = bool((full == full[:, :, :1]).all())
        check(invariant == lookup["physical_menu_invariant"] and (not has_tie) == lookup["unique_argmax_equivalence_proof_used"],
              label + " invariance")
        check(all(t["actions_by_goal"] == full[i, :, 0].tolist()
                  and t["menu_action_counts_by_goal"] == [np.bincount(full[i, g], minlength=6).tolist() for g in range(2)]
                  for i, t in enumerate(table)), label + " table actions/counts")
        nmenus = 720 if has_tie else 1
        calls = 1 + (18 if has_tie else 0)
        check(lookup["menus_retained_for_metrics"] == lookup["menus_evaluated_per_code_goal"] == nmenus
              and lookup["receiver_forward_calls"] == calls
              and lookup["evaluated_code_goal_menu_rows"] == 98 + (70560 if has_tie else 0), label + " evaluation counts")
        forward_calls += calls
        evaluated_rows += 98 + (70560 if has_tie else 0)
        coverage = recompute_coverage(full if has_tie else full[:, :, :1], run["train_map_ids"],
                                      run["heldout_map_ids"], run["plan"]["blocked"])
        check(coverage == row["coverage"], label + " complete maps, four classes and oracle metrics")
        if row["update"] == 2400:
            endpoint_count += 1
            anchor = next(d for d in source_index[key[:2]]["directions"] if d["collector"] == row["receiver"])
            check(table == anchor["receiver_decoder_table"] and lookup["physical_action_sha256"]
                  == anchor["menu_audit"]["physical_action_sha256"], label + " 2400 source table and SHA")
            expected_anchor = {"status": "exact_match", "code_rows": 49, "goal_rows": 98,
                "all_menu_physical_cases": 70560,
                "source_physical_action_sha256": anchor["menu_audit"]["physical_action_sha256"],
                "checkpoint_final_state_dict_exact": True}
            check(row["endpoint_anchor"] == expected_anchor, label + " endpoint report")
        else:
            check(row["endpoint_anchor"] is None, label + " no premature endpoint anchor")
        family = row["condition"].split("_", 1)[1] if row["condition"].startswith("split") else row["condition"]
        for subset in ("train", "heldout"):
            s = coverage["subsets"][subset]
            if s["map_count"]:
                grouped[family, row["seed"], row["update"], subset].append(s)
        row_checks.append({"key": list(key), "passed": len(errors) == before,
                           "physical_table_sha256": lookup["physical_action_sha256"], "tie_code_goal_count": ties})
    by_seed = []
    for (family, seed, update, subset), values in sorted(grouped.items()):
        means = {}
        for field in FIELDS:
            items = [s["coverage_classes"][field] if field in CATS else s[field] for s in values]
            means[field] = float(sum((Fraction(x["numerator"], x["denominator"]) for x in items), Fraction()) / len(items))
        count = 2 if family.startswith("full_") else 6
        check(len(values) == count, f"{family}/{seed}/{update}/{subset} direction denominator")
        by_seed.append({"family": family, "seed": seed, "update": update, "subset": subset,
                        "receiver_direction_count": len(values), "equal_weight_means": means})
    by_family = []
    for family, update, subset in sorted({(r["family"], r["update"], r["subset"]) for r in by_seed}):
        selected = [r for r in by_seed if (r["family"], r["update"], r["subset"]) == (family, update, subset)]
        check([r["seed"] for r in selected] == SEEDS, f"{family}/{update}/{subset} four seeds")
        by_family.append({"family": family, "update": update, "subset": subset, "training_seeds": SEEDS,
            "seed_values": {f: [r["equal_weight_means"][f] for r in selected] for f in FIELDS},
            "means": {f: sum(r["equal_weight_means"][f] for r in selected) / 4 for f in FIELDS}})
    check(same_numbers({"by_seed": by_seed, "by_family": by_family}, results["summaries"]), "all seed/family aggregates")
    check(endpoint_count == results["endpoint_anchors_exact"] == status["endpoint_anchors_exact"] == 112, "112 endpoints")
    check(results["receiver_checkpoint_count"] == status["receiver_checkpoint_count"] == 896, "896 count")
    check(results["tie_receiver_checkpoint_count"] == len(tie_rows) and results["receiver_forward_calls"] == forward_calls
          and results["evaluated_code_goal_menu_rows"] == evaluated_rows, "total evaluation accounting")
    check(all(results[k] == 0 for k in ("visual_backbones_loaded", "feature_banks_loaded", "sender_forward_calls",
                                      "training_updates", "natural_reward_sample_count")), "declared probe scope")
    trends = [r for r in by_family if r["subset"] == ("train" if r["family"].startswith("full_") else "heldout")]
    audit = {"audited_at": datetime.now().astimezone().isoformat(), "status": "verified" if not errors else "discrepancies",
        "errors": errors, "scope": {"runs": 56, "conditions": 14, "training_seeds": 4, "checkpoints_per_run": 8,
            "receivers_per_checkpoint": 2, "records": len(rows), "source_file_hashes": len(frozen_sources),
            "source_snapshots": len(snapshots), "endpoint_tables": endpoint_count,
            "complete_code_goal_inputs": evaluated_rows, "audit_model_calls": 0, "audit_weights_deserialized": 0},
        "source_files": frozen_sources, "snapshots": snapshots, "python_preparation_equivalence": compatibility,
        "menu_gather_source_verified": menu_gather, "record_checks": row_checks,
        "tie_records": tie_rows, "minimum_saved_top_two_margin": min_margin,
        "all_aggregate_rows_verified": {"by_seed": len(by_seed), "by_family": len(by_family)},
        "recomputed_summaries": {"by_seed": by_seed, "by_family": by_family},
        "selected_trajectory_rows": trends,
        "denominator_note": "Four trained seeds; within a split family, six directions (three splits x two receivers) per seed. Full-map controls use two directions and one split. Heldout maps=6 and train maps=24 per split; full-map train maps=30. Menus and checkpoints are not independent training samples.",
        "tie_interpretation": "If ties exist, mean_menu(max_code reward) allows menu-dependent code selection and can exceed max_code(mean_menu reward). No ties in these records means these two quantities agree here.",
        "limitations": ["Saved logits, not weight-to-logit execution, independently recomputed; weight files only byte-hashed.",
            "Checkpoint2400/final tensor equality was asserted by frozen execution; this audit independently checks their byte provenance and output table anchors without tensor deserialization.",
            "Full-code greedy receiver coverage is not learned sender success, stochastic support, language semantics or channel capacity.",
            "Checkpoint0 starts a newly initialized social interface after prepared individual-task projection training, with a frozen DINO backbone; it is neither end-to-end training from scratch nor an inherited social protocol. Wide random decoder coverage is not pre-existing communication.",
            "Eight saved checkpoints and a post-hoc probe do not identify first emergence, monotonic learning or an ecological causal mechanism.",
            "Full-blocked complete-code values are counterfactual; only zero-code values are channel-admissible."],
        "file_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "plan.json", ROOT / "execution/records.jsonl",
                         ROOT / "execution/results.json", ROOT / "execution/status.json", Path(__file__))}}
    (ROOT / "独立核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 接收覆盖轨迹独立核验", "", f"核验时间：{audit['audited_at']}。状态：{audit['status']}。仅读取 JSON、源码及文件字节，使用 NumPy 重算；未反序列化权重或追加推理。", "",
        "56 个运行 × 8 个检查点 × 2 个接收者的 896 条清单完整。686 份源文件 SHA、8 份源码快照、记录文件 SHA 及 112 个终点接收表与完整表 SHA 全部核对。Python 3.12 包与未执行的 3.9 包具有相同源文件摘要和运行清单。", "",
        "逐条从保存的 49×2×6 身份菜单 logits 重算 argmax、前二名间隔、全部 720 排列的物理动作及其 SHA。所有地图的四类覆盖、两边际可达性、完整码上界和封锁通道允许码上界均独立重算；320 条种子汇总与 80 条条件汇总一致。", "",
        f"实测并列接收者检查点为 {len(tie_rows)}，最小前二名间隔为 {min_margin:.10g}。本批所有最大值唯一，菜单只重排地点 logits，因此物理 argmax 不随菜单改变。若存在并列，冻结指标 mean_menu(max_code reward) 会允许选择器按菜单选码，是较乐观的参考；它一般不等于 max_code(mean_menu reward)。本批没有触发该差异。", "",
        "|条件／评估地图|检查点|同码双目标可达比例|完整码均匀目标上界|通道允许码上界|", "|---|---:|---:|---:|---:|"]
    for r in trends:
        m = r["means"]
        lines.append(f"|{r['family']}／{r['subset']}|{r['update']}|{m[CATS[0]]:.4%}|{m['full_code_uniform_goal_ceiling']:.4%}|{m['channel_admissible_uniform_goal_ceiling']:.4%}|")
    lines += ["", "表中每行先在种子内等权平均接收方向与划分，再平均四个训练种子；原始四种子值保留在独立核验.json。留出条件每种子有三个划分、六个接收方向，full 控制每种子只有一个划分、两个方向。各划分训练/留出地图分别为24/6，全地图控制训练地图30、无留出；地图、菜单、接收者和检查点均不增加独立训练种子数。", "",
        "轨迹显示：这些留出条件的完整码覆盖上界在训练早期出现下降，随后不同程度波动；并非单调增加。所有四个留出条件终点的四种子平均上界均低于各自 update0。随机初始化的接收映射也可覆盖许多地点组合；这种覆盖不表示发送者会选择相应码，更不能据此把 update0 解释为已有共同语言。update0 的社交接口为新初始化，但 DINO 骨干已预训练且冻结，project 已经过 prepared 个人任务训练，因此整个系统也不是端到端从零开始。full_direct 的完整地图终点覆盖高于其初值；full_blocked 虽仍有较高的49码反事实覆盖，仅零码可用时上界始终为1/6。", "",
        "本轮只测固定接收者的贪心响应范围，没有测自然发送消息、同照片任务回报或语言语义。它能描述保存检查点上的接收覆盖变化，不能定位两个存盘点之间的首次形成、证明生态因素导致变化，或把菜单/地图重复计为额外样本。权重来源已逐文件核对；权重到 logits 的数值生成没有在本独立审计中重跑，2400与final逐张量相等来自冻结执行的断言，本审计另行核对终点输出表。", "",
        "复核入口：[独立核验.py](独立核验.py)；逐项结果：[独立核验.json](独立核验.json)。"]
    if errors:
        lines += ["", "发现的不一致："] + ["- " + e for e in errors]
    (ROOT / "独立核验.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": audit["status"], "errors": errors, "scope": audit["scope"],
                      "tie_records": len(tie_rows), "minimum_margin": min_margin}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
