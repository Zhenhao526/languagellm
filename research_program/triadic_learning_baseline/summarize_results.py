"""Read only a completed four-seed baseline; summarize final update6000 only.

Standard library, no training-source import or checkpoint deserialization.
The full physical/forward audit and behavior analysis are separate artifacts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

SEEDS = (43101, 43102, 43103, 43104)
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
NAMES = {"train": "训练需求／训练布局", "new_needs": "新需求／训练布局",
         "new_layouts": "训练需求／新布局", "new_needs_and_layouts": "新需求／新布局"}
METRICS = ("greedy_full_success_rate", "greedy_reward_mean", "exact_stochastic_expected_reward_mean",
           "exact_stochastic_full_success_probability_mean", "exact_stochastic_physical_execution_probability_mean")
CHECKPOINTS = [0, 100, 500, 1500, 3000, 6000]
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def integer(value, name):
    require(type(value) is int and value >= 0, f"Invalid count: {name}")
    return value


def near(left, right, name):
    require(math.isfinite(left) and math.isfinite(right) and abs(left-right) <= 1e-12, f"Inconsistent {name}")


def check_cell(cell, world_count):
    n = integer(cell["worlds"], "worlds")
    require(n == world_count and n > 0, "Incomplete/incorrect partition denominator")
    counts = cell["greedy_reward_counts"]
    require(set(counts) == {"0.0", "0.5", "1.0"}, "Incorrect native reward support")
    c0, ch, c1 = [integer(counts[k], "reward count") for k in ("0.0", "0.5", "1.0")]
    require(c0 + ch + c1 == n, "Native reward counts do not cover all worlds")
    require(integer(cell["greedy_full_successes"], "full successes") == c1, "Full-success integer differs")
    require(integer(cell["greedy_non_full_success_worlds"], "failures") == c0 + ch, "Failure count differs")
    near(cell["greedy_full_success_rate"], c1/n, "full-success rate")
    near(cell["greedy_reward_sum"], c1 + .5*ch, "reward sum")
    near(cell["greedy_reward_mean"], (c1 + .5*ch)/n, "mean reward")
    require(integer(cell["greedy_satisfied_agents"], "satisfied agents") == 2*c1+ch, "Satisfied-agent count differs")
    for field in METRICS:
        require(math.isfinite(cell[field]) and -1e-12 <= cell[field] <= 1+1e-12, "Invalid probability/rate")
    ef, er, ep = [cell[k] for k in ("exact_stochastic_full_success_probability_mean",
        "exact_stochastic_expected_reward_mean", "exact_stochastic_physical_execution_probability_mean")]
    require(ef <= er + 1e-12 and er <= ep + 1e-12, "Exact stochastic quantities violate reward bounds")
    require(integer(cell["greedy_agent_argmax_ties"], "argmax ties") <= 3*n, "Too many actor ties")
    failures = cell["greedy_failure_categories"]
    require(sum(integer(v, "failure category") for v in failures.values()) == c0+ch, "Failure categories differ")


def aggregate(seed_results, world_counts, threshold):
    require([r["seed"] for r in seed_results] == list(SEEDS), "All four fixed seeds in their fixed order are required")
    require(set(world_counts) == set(PARTITIONS), "Four fixed partitions required")
    require(threshold == .99, "Frozen candidate threshold changed")
    rows, pass_cells, per_seed_pass = [], [], []
    for result in seed_results:
        require(result["updates"] == 6000 and result["training_world_samples"] == 1536000,
                "Not the fixed final training budget")
        require(set(result["final"]) == set(PARTITIONS), "Missing final partition")
        require([c["update"] for c in result["monitor"]] == CHECKPOINTS, "Missing fixed checkpoint record")
        # Monitor scores are intentionally never used in final aggregation.
        for name in PARTITIONS:
            cell = result["final"][name]
            check_cell(cell, world_counts[name])
            row = {"seed": result["seed"], "partition": name, "label": NAMES[name], "final_update": 6000,
                   **deepcopy(cell)}
            rows.append(row)
            if name != "train":
                pass_cells.append({"seed": result["seed"], "partition": name,
                    "threshold": threshold, "full_success_rate": cell["greedy_full_success_rate"],
                    "passes": cell["greedy_full_success_rate"] >= threshold})
        passed = all(x["passes"] for x in pass_cells if x["seed"] == result["seed"])
        require(result["candidate_threshold_met"] is passed, "Per-seed candidate flag differs")
        per_seed_pass.append({"seed": result["seed"], "all_three_heldout_cells_pass": passed})
    by_partition = []
    for name in PARTITIONS:
        cells = [r for r in rows if r["partition"] == name]
        summaries = {}
        for metric in METRICS:
            values = [r[metric] for r in cells]
            summaries[metric] = {"seed_order": list(SEEDS), "values": values,
                "mean": statistics.fmean(values), "min": min(values), "max": max(values)}
        by_partition.append({"partition": name, "label": NAMES[name], "worlds_per_seed": world_counts[name],
            "training_team_initializations": 4, "metrics": summaries,
            "non_full_success_counts_by_seed": [r["greedy_non_full_success_worlds"] for r in cells]})
    return {"rows": rows, "by_partition": by_partition, "candidate_cells": pass_cells,
            "candidate_seed_flags": per_seed_pass, "candidate_cells_passed": sum(x["passes"] for x in pass_cells),
            "candidate_cells_total": 12, "all_seed_candidates_meet_threshold": all(x["passes"] for x in pass_cells),
            "symbolic_phase_unlocked": False}


def analyze(run):
    run = Path(run).resolve()
    # This is the first result read: partial runs fail before any seed score is read.
    require((run / "execution/status.json").is_file(), "No completed batch status; do not summarize partial scores")
    status = read(run / "execution/status.json")
    require(status.get("status") == "completed" and status.get("completed_seeds") == 4, "Batch is not fully completed")
    inputs = {}

    def tracked(relative):
        path = run / relative
        inputs[str(path)] = sha(path)
        return read(path)

    status = tracked("execution/status.json")
    result, plan, prepared, freeze = [tracked(name) for name in
        ("execution/results.json", "plan.json", "prepared.json", "freeze.json")]
    require(result["status"] == "completed" and result["completed_seed_count"] == 4,
            "Four seed results are not complete")
    require(inputs[str(run / "plan.json")] == freeze["plan_sha256"] == result["plan_sha256"], "Plan SHA differs")
    require(inputs[str(run / "prepared.json")] == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared SHA differs")
    require(plan["config"] == prepared["config"] and plan["config"]["seeds"] == list(SEEDS), "Fixed configuration differs")
    require(result["updates_total"] == 24000 and result["training_world_samples_total"] == 6144000,
            "Total budget differs")
    for name, digest in plan["sources"].items():
        for path in (ROOT / name, run / "source_snapshot" / name):
            inputs[str(path)] = sha(path)
            require(inputs[str(path)] == digest, "Frozen source changed")
    seed_results = []
    require(sorted(p.name for p in (run / "execution").glob("seed_*")) == [f"seed_{s}" for s in SEEDS],
            "Unexpected or missing seed directory")
    for i, seed in enumerate(SEEDS):
        base = f"execution/seed_{seed}"
        one = tracked(base + "/result.json")
        require(one == result["seeds"][i], "Root/seed final result differs")
        paths = {base + "/training.jsonl": one["training_log_sha256"],
                 base + "/checkpoint_6000.npz": one["final_checkpoint_sha256"]}
        paths.update({base + f"/final_{name}.npz": one["final"][name]["data_sha256"] for name in PARTITIONS})
        for relative, expected in paths.items():
            path = run / relative
            inputs[str(path)] = sha(path)
            require(inputs[str(path)] == expected, "Final endpoint/log data hash differs")
        seed_results.append(one)
    counts = {name: prepared["partitions"][name]["world_count"] for name in PARTITIONS}
    require(sum(counts.values()) == 143424, "Development support denominator differs")
    summary = aggregate(seed_results, counts, plan["config"]["candidate_min_full_success_rate_each_heldout_partition_each_seed"])
    require(result["all_seed_candidates_meet_threshold"] is summary["all_seed_candidates_meet_threshold"]
            and result["symbolic_phase_unlocked"] is False, "Gate flags differ")
    summary.update(status="complete_final_6000_summary", completed_at=datetime.now(timezone.utc).isoformat(),
        run_directory=str(run), source_sha256=inputs, summary_source_sha256=sha(__file__),
        total_elapsed_seconds=result["elapsed_seconds"], seed_elapsed_seconds={r["seed"]: r["elapsed_seconds"] for r in seed_results},
        training_team_initializations=4, fixed_demand_and_layout_splits=1,
        training_state_samples=6144000, weighted_structural_action_contributions=147456000,
        offline_reward_table_entries=3442176, complete_final_world_evaluations=573696,
        endpoint_update=6000, statistical_tests_performed=0, checkpoint_weight_deserializations=0,
        policy_forward_calls=0, training_updates=0,
        scope=["Final summaries and byte/hash anchors only; full per-world/action/probability audit is separate.",
               "Four independently initialized teams share one fixed demand/layout split and the same world sets.",
               "Exact stochastic expectations are not additional native action samples or observed success rates.",
               "All final seeds and partitions are retained; no monitor or best-seed substitution.",
               "No communication channel or newly formed convention is measured."])
    return summary


def span(metric, *, percent=False):
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    return f"{scale*metric['mean']:.3f}{suffix}（{scale*metric['min']:.3f}–{scale*metric['max']:.3f}{suffix}）"


def markdown(summary):
    passed = summary["all_seed_candidates_meet_threshold"]
    conclusion = ("全部12个种子×留出格达到预定99%低残余失败候选阈值。" if passed else
                  f"12个种子×留出格中有{summary['candidate_cells_passed']}格达到预定99%阈值；整批未达到低残余失败候选条件。")
    out = ["# 三主体可训练能力基线：结果与下一步（报告草稿）", "",
        conclusion + "这不是后续互动能力门槛，符号阶段保持未解锁。", "",
        "本报告只使用4个固定种子各自更新6000的完整终点。没有挑选最佳种子，也没有用固定监测点的成绩替代正式终点。行为分类、固定搭档界与独立执行审计由根任务补充对应记录；本草稿不预先宣称那些核验已完成。", "",
        "## 全部16个终点格", "",
        "失败数指未得满分，包括0分和0.5分；原生R均值因此不同于满分率。每行的分母包含6种公开私有地点分配，不能把这些分配当成独立训练种子。", "",
        "| 种子 | 需求／布局 | 世界数 | R=0 | R=0.5 | R=1 | 未满分 | 满分率 | 原生R均值 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["rows"]:
        c = r["greedy_reward_counts"]
        out.append(f"| {r['seed']} | {r['label']} | {r['worlds']} | {c['0.0']} | {c['0.5']} | {c['1.0']} | {r['greedy_non_full_success_worlds']} | {100*r['greedy_full_success_rate']:.3f}% | {r['greedy_reward_mean']:.6f} |")
    out += ["", "## 同一政策分布的精确随机量", "",
        "这些量对三人独立从完整17项分布抽样的联合动作做解析求和。训练后的部署动作仍为各人argmax；下表不是额外随机执行次数，也没有包含抽样置信区间。", "",
        "| 种子 | 需求／布局 | 精确随机E[R] | 精确随机满分概率 | 精确随机物理执行概率 |", "| --- | --- | ---: | ---: | ---: |"]
    for r in summary["rows"]:
        out.append(f"| {r['seed']} | {r['label']} | {r['exact_stochastic_expected_reward_mean']:.6f} | {100*r['exact_stochastic_full_success_probability_mean']:.3f}% | {100*r['exact_stochastic_physical_execution_probability_mean']:.3f}% |")
    out += ["", "## 每格4个团队初始化的均值与范围", "",
        "每个种子等权。括号为4个种子的最小值–最大值，不是置信区间；没有进行显著性检验。", "",
        "| 需求／布局 | argmax满分率 | argmax原生R均值 | 随机E[R] | 随机满分概率 |", "| --- | --- | --- | --- | --- |"]
    for p in summary["by_partition"]:
        m = p["metrics"]
        out.append(f"| {p['label']} | {span(m['greedy_full_success_rate'],percent=True)} | {span(m['greedy_reward_mean'])} | {span(m['exact_stochastic_expected_reward_mean'])} | {span(m['exact_stochastic_full_success_probability_mean'],percent=True)} |")
    out += ["", "## 预算与可解释范围", "",
        f"完整固定预算为24000次更新、6144000次训练状态抽样及147456000次24项加权贡献；离线原生奖励表共3442176项。4种子的完整终点评估共573696个世界。整批记录耗时{summary['total_elapsed_seconds']/60:.2f}分钟。6144000不能写成每次只收到一条反馈的环境互动次数。", "",
        "独立训练单位是4个三人团队初始化；它们共用一套固定需求多重集拆分与布局拆分。新需求格表示未见需求多重集及其三人排列，新布局格表示未见物资排列；不是新的基本属性、新物资种类或独立重抽的多套数据划分。", "",
        "本轮优化器集中使用反事实原生奖励，actor各自参数独立、执行时各自选择完整17项动作。它可检验这种富反馈方法在固定预算内的能力与泛化，不能直接推广到仅靠单次稀疏反馈的学习者。和Qwen相比，策略类别、训练方式、输入和输出接口均不同，不能将分数差异归因于某一处提示或接口变化。", "",
        "没有通信频道、新符号词典或互动约定的学习，因此本结果没有语言或沟通现象结论。任务表现即使很高，也不能代替后续局部信息、互动机会和新约定的独立实验。", "",
        "## 下一步边界", "", conclusion,
        "本批不据结果追加训练、改变阈值或重选种子。先结合全部失败世界的物理分类、需求兼容性与搭档使用情况判断结果，再决定是否需要另一个完整、事先固定的研究设计。不能用这里的99%筛查替代通信任务的能力门槛，也不自动开启符号阶段。", "",
        "本脚本只读终点JSON并校验冻结来源、训练日志及终点NPZ的字节哈希；没有加载权重或进行策略前向。完整独立动作、概率和训练执行核验应引用另项审计，不能将本汇总当作它。机器可读逐种子向量、所有计数与输入SHA见同目录 `summary.json`。", ""]
    return "\n".join(out)


def execute(run, out):
    out = Path(out).resolve()
    require(not out.exists(), "Refuse to overwrite summary output")
    summary = analyze(run)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (out / "结果与下一步.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(summary))
    return {"status": summary["status"], "output": str(out), "summary_sha256": sha(out / "summary.json")}


def self_test():
    # Entirely synthetic counts; never reads any run or checkpoint.
    world_counts = {p: 100 for p in PARTITIONS}
    results = []
    for i, seed in enumerate(SEEDS):
        cells = {}
        for p in PARTITIONS:
            c0, ch, c1 = i, 4, 96-i
            cells[p] = {"worlds": 100, "greedy_reward_counts": {"0.0": c0, "0.5": ch, "1.0": c1},
                "greedy_full_successes": c1, "greedy_non_full_success_worlds": c0+ch,
                "greedy_full_success_rate": c1/100, "greedy_reward_sum": c1+.5*ch,
                "greedy_reward_mean": (c1+.5*ch)/100, "greedy_satisfied_agents": 2*c1+ch,
                "exact_stochastic_expected_reward_mean": .7, "exact_stochastic_full_success_probability_mean": .6,
                "exact_stochastic_physical_execution_probability_mean": .8, "greedy_agent_argmax_ties": 0,
                "greedy_failure_categories": {"single_transport_proposal": c0, "matched_one_need_satisfied": ch}}
        results.append({"seed": seed, "updates": 6000, "training_world_samples": 1536000, "final": cells,
            "monitor": [{"update": u, "score": 1} for u in CHECKPOINTS], "candidate_threshold_met": False})
    a = aggregate(results, world_counts, .99)
    require(len(a["rows"]) == 16 and a["candidate_cells_passed"] == 0, "Complete matrix fixture")
    near(a["by_partition"][0]["metrics"]["greedy_full_success_rate"]["mean"], .945, "Mean fixture")
    require(a["by_partition"][0]["non_full_success_counts_by_seed"] == [4, 5, 6, 7], "Failure preservation")
    b = deepcopy(results)
    b[0]["monitor"][0]["score"] = -999
    require(aggregate(b, world_counts, .99) == a, "Monitor affected final summary")
    for bad in (results[:3], list(reversed(results))):
        try: aggregate(bad, world_counts, .99)
        except ValueError: pass
        else: raise AssertionError("Missing/reordered seed accepted")
    corrupt = deepcopy(results)
    corrupt[0]["final"]["train"]["greedy_non_full_success_worlds"] = 0
    try: aggregate(corrupt, world_counts, .99)
    except ValueError: pass
    else: raise AssertionError("Lost failures accepted")
    with tempfile.TemporaryDirectory() as tmp:
        try: analyze(Path(tmp))
        except ValueError as e: require("completed batch" in str(e), "Wrong partial refusal")
        else: raise AssertionError("Incomplete run accepted")
    return {"status": "passed", "synthetic_checks": 8, "real_run_reads": 0, "policy_forward_calls": 0,
            "summary_source_sha256": sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summarize", "self-test"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
    else:
        require(args.run is not None and args.out is not None, "--run and --out required")
        result = execute(args.run, args.out)
    print(json.dumps(result, ensure_ascii=False))
