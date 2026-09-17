"""Audit and plot saved four-agent partner experiments, without model updates."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
PAIRS = tuple(itertools.combinations(range(4), 2))
ORIGINAL = {(0, 1), (2, 3)}
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
NAMES = {"fixed": "固定伙伴", "rotating": "轮换伙伴"}
COLORS = {"fixed": "#526d91", "rotating": "#00917c"}
GROUPS = ("original", "cross", "all")
GROUP_NAMES = {"original": "参考原配：AB、CD", "cross": "其余四对：AC、AD、BC、BD", "all": "全部六对"}
MODES = ("normal", "shuffle", "blank", "stochastic")


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {path}: {value}")))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(values):
    values = [float(v) for v in values]
    return {"n_population_seeds": len(values), "mean": float(np.mean(values)), "min": min(values), "max": max(values)}


def success(evaluation, group, mode="normal", task="full"):
    return evaluation["aggregates"][group]["tasks"][task][mode]["mean_reward_per_step"]


def gap(evaluation, group, task="full"):
    return evaluation["aggregates"][group]["tasks"][task]["normal_minus_shuffle"]["mean"]


def audit_training(rows, config, condition):
    total = sum(config[k] for k in ("course_updates", "exploratory_full_updates", "pure_reward_updates"))
    require([r["update"] for r in rows] == list(range(1, total + 1)), "Training update sequence is incomplete")
    counts = np.zeros((4, 4), dtype=np.int64)
    snapshots = {0: counts.copy()}
    for row in rows:
        update = row["update"]
        course_end = config["course_updates"]
        exploration_end = course_end + config["exploratory_full_updates"]
        expected_stage = "course" if update <= course_end else "full_exploration" if update <= exploration_end else "pure_reward"
        require(row["stage"] == expected_stage, "Unexpected training stage")
        require(row["task"] == ("curriculum" if update <= course_end else "full"), "Unexpected training task")
        require(row["aux_weight"] == 0., "Unexpected signalling auxiliary loss")
        require(row["action_entropy_weight"] == (config["action_entropy_coefficient"] if update <= exploration_end else 0.), "Unexpected action exploration schedule")
        index, bits = row["matching_index"], row["layout_bits"]
        require(index in (0, 1, 2) and len(bits) == 3 and set(bits) <= {0, 1}, "Invalid matching/layout record")
        if condition == "fixed":
            require(index == 0, "Fixed population used a nonfixed matching")
        expected = [list(pair) for pair in MATCHINGS[index]]
        if bits[0]:
            expected.reverse()
        for slot in range(2):
            if bits[slot + 1]:
                expected[slot].reverse()
        require(expected == row["slot_assignment"], "Slot assignment does not follow saved matching/layout bits")
        agents = {a["agent"]: a for a in row["agents"]}
        require(set(agents) == set(range(4)) and len(row["agents"]) == 4, "Missing or duplicate agent metrics")
        outcomes = {p["slot"]: p for p in row["pair_outcomes"]}
        require(set(outcomes) == {0, 1}, "Missing camp outcome")
        for slot, (left, right) in enumerate(expected):
            require(outcomes[slot]["agents"] == [left, right], "Camp identities differ from assignment")
            for who, partner in ((left, right), (right, left)):
                require(agents[who]["partner"] == partner, "Incorrect message recipient")
                require(agents[who]["received_sha256"] == agents[partner]["sent_sha256"], "Received message hash differs from partner sent hash")
                require(np.isclose(agents[who]["sampled_success"], outcomes[slot]["success"], atol=1e-8), "Agent target differs from its camp outcome")
                counts[who, partner] += 1
        if update in config["checkpoints"]:
            snapshots[update] = counts.copy()
    if condition == "rotating":
        require(all(sorted(r["matching_index"] for r in rows[i:i + 3]) == [0, 1, 2]
                    for i in range(0, total, 3)), "Rotation is not balanced in each three-update block")
    require(np.array_equal(counts.sum(axis=1), np.full(4, total)), "Unequal per-agent exposure budget")
    return counts, snapshots


def audit_evaluation(evaluation, n, final=False, run_path=None):
    require([tuple(p["agents"]) for p in evaluation["pairs"]] == list(PAIRS), "Pair set/order mismatch")
    expected_modes = set(MODES if final else ("normal", "shuffle", "stochastic"))
    expected_tasks = {"full", "curriculum"} if final else {"full"}
    traced = 0
    for pair in evaluation["pairs"]:
        ids = tuple(pair["agents"])
        require(pair["group"] == ("original" if ids in ORIGINAL else "cross"), "Incorrect reference pair group")
        require(set(pair["tasks"]) == expected_tasks, "Missing evaluation task")
        for task, modes in pair["tasks"].items():
            require(set(modes) == expected_modes, "Missing evaluation mode")
            require(len({m["external_cases_sha256"] for m in modes.values()}) == 1, "Message modes evaluated on unpaired cases")
            for name, metrics in modes.items():
                require(metrics["cases"] == n, "Evaluation case count mismatch")
                require(metrics["population_agents_in_local_order"] == list(ids), "Incorrect global identity metadata")
                require(metrics["greedy"] == (name != "stochastic"), "Greedy/stochastic evaluation label mismatch")
                require(metrics["mode"] == ("normal" if name == "stochastic" else name), "Message mode label mismatch")
                require(0 <= metrics["mean_reward_per_step"] <= 1, "Invalid success rate")
                for direction in metrics["by_direction"]:
                    require(direction["population_sender"] == ids[direction["restricted_sender"]]
                            and direction["population_receiver"] == ids[direction["mixed_receiver"]], "Incorrect direction mapping")
            if final:
                for mode in MODES:
                    record = pair["traces"][f"{task}_{mode}"]
                    trace_path = Path(record["path"]).resolve()
                    require(trace_path.parent == (run_path / "traces").resolve(), "Trace path lies outside run folder")
                    require(record["cases"] == n and sha(trace_path) == record["sha256"], "Trace file hash/count metadata mismatch")
                    traced += 1
        if final:
            intervention = pair["intervention"]
            require(intervention["population_agents_in_local_order"] == list(ids), "Intervention global identity mismatch")
            require({d["population_sender"] for d in intervention["directions"]} == set(ids), "Missing intervention direction")
    for group in GROUPS:
        chosen = [p for p in evaluation["pairs"] if group == "all" or p["group"] == group]
        require(evaluation["aggregates"][group]["pair_count"] == len(chosen), "Incorrect group pair count")
        for task in expected_tasks:
            for mode in expected_modes:
                expected = np.mean([p["tasks"][task][mode]["mean_reward_per_step"] for p in chosen])
                require(np.isclose(expected, success(evaluation, group, mode, task), atol=1e-12), "Aggregate success differs from equal pair mean")
            expected_gap = np.mean([p["tasks"][task]["normal"]["mean_reward_per_step"] -
                                    p["tasks"][task]["shuffle"]["mean_reward_per_step"] for p in chosen])
            require(np.isclose(expected_gap, gap(evaluation, group, task), atol=1e-12), "Aggregate shuffle gap mismatch")
    return traced


def eval_hashes(evaluation):
    hashes = {}
    for pair in evaluation["pairs"]:
        label = "".join(map(str, pair["agents"]))
        for task, modes in pair["tasks"].items():
            for mode, metric in modes.items():
                hashes[f"{label}/{task}/{mode}"] = metric["external_cases_sha256"]
        if "intervention" in pair:
            hashes[f"{label}/intervention"] = pair["intervention"]["external_cases_sha256"]
    if "sender_alignment" in evaluation:
        hashes["sender_alignment"] = evaluation["sender_alignment"]["external_cases_sha256"]
    return hashes


def load_and_audit(folder):
    config, completion = read(folder / "config.json"), read(folder / "completed.json")
    require(config["conditions"] == ["fixed", "rotating"], "Unexpected comparison conditions")
    require(completion["status"] == "completed" and completion["runs"] == len(config["seeds"]) * 2, "Declared experiment is incomplete")
    source_hashes = read(folder / "source_hashes.json")
    require(set(source_hashes) == {p.name for p in (folder / "source").iterdir() if p.is_file()}, "Saved source manifest is incomplete")
    for name, expected in source_hashes.items():
        require(sha(folder / "source" / name) == expected, f"Saved source hash mismatch: {name}")
    listed_results = read(folder / "results.json")
    require(len(listed_results) == completion["runs"], "Top-level results list is incomplete")
    listed = {(r["condition"], r["seed"]): r for r in listed_results}
    require(len(listed) == len(listed_results), "Duplicate top-level run result")
    preparations = read(folder / "preparations.json")
    runs, training_rows, audit = {}, {}, {"source_hashes_verified": source_hashes, "paired_seeds": [], "runs": []}
    total = sum(config[k] for k in ("course_updates", "exploratory_full_updates", "pure_reward_updates"))
    for seed in config["seeds"]:
        preparation = read(folder / f"preparation_s{seed}.json")
        require(preparation == preparations[str(seed)], "Preparation report copies differ")
        prepared_hash = sha(folder / f"prepared_s{seed}.pt")
        require(preparation["passed"] and prepared_hash == preparation["prepared_sha256"], "Prepared population failed gate or hash verification")
        require(all(r["heldout_need_sensitive_choice"] >= config["preparation_accuracy_gate"]
                    for r in preparation["reused_practice"] + preparation["new_practice"]), "Preparation accuracy is below configured gate")
        require(all(r["sender_unchanged_during_preparation"] for r in preparation["new_practice"]), "Sender changed in new individual preparation")
        for condition in config["conditions"]:
            path = folder / f"{condition}_s{seed}"
            result, curve = read(path / "result.json"), read(path / "learning_curve.json")
            require(result == listed[condition, seed], "Run result differs from top-level saved result")
            require((result["condition"], result["seed"], result["updates"]) == (condition, seed, total), "Run identity or budget mismatch")
            require(sha(path / "initial.pt") == prepared_hash, "Initial checkpoint differs from common prepared population")
            require([r["update"] for r in curve] == config["checkpoints"], "Checkpoint list is incomplete")
            rows = [json.loads(line) for line in (path / "training_metrics.jsonl").read_text().splitlines()]
            counts, snapshots = audit_training(rows, config, condition)
            require(np.array_equal(counts, result["edge_training_updates"]), "Final exposure matrix differs from training log")
            require(np.array_equal(counts * config["batch_size_per_agent"], result["edge_joint_cases"]), "Edge case counts differ from exposure matrix")
            for row in curve:
                require(np.array_equal(snapshots[row["update"]], row["edge_training_updates"]), "Checkpoint exposure counts differ from logs")
                audit_evaluation(row["evaluation"], config["checkpoint_evaluation_n"])
            traced = audit_evaluation(result["evaluation"], config["evaluation_n"], True, path)
            require(all(p["intervention"]["cases"] == config["intervention_n"] for p in result["evaluation"]["pairs"]), "Intervention budget mismatch")
            require(result["evaluation"]["sender_alignment"]["cases"] == config["alignment_n"], "Alignment budget mismatch")
            runs[condition, seed] = {"path": path, "result": result, "curve": curve}
            training_rows[condition, seed] = rows
            audit["runs"].append({"condition": condition, "seed": seed, "source_and_result_metadata_valid": True,
                                   "result_sha256": sha(path / "result.json"), "training_updates_verified": len(rows),
                                   "trace_files_sha256_verified": traced, "edge_training_updates": counts.tolist()})
        left, right = training_rows["fixed", seed], training_rows["rotating", seed]
        require(all(a["world_slots_sha256"] == b["world_slots_sha256"] for a, b in zip(left, right)), "Training world resource/photo slots differ across conditions")
        require(all(a["layout_bits"] == b["layout_bits"] for a, b in zip(left, right)), "Random layout bits differ across conditions")
        identical_private = [0] * 4
        same_assignments = 0
        for a, b in zip(left, right):
            aa, bb = {x["agent"]: x for x in a["agents"]}, {x["agent"]: x for x in b["agents"]}
            for who in range(4):
                identical_private[who] += aa[who]["private_input_sha256"] == bb[who]["private_input_sha256"]
            if a["slot_assignment"] == b["slot_assignment"]:
                same_assignments += 1
                require(all(aa[who]["private_input_sha256"] == bb[who]["private_input_sha256"] for who in range(4)), "Identical assignments received differing private inputs")
        for a, b in zip(runs["fixed", seed]["curve"], runs["rotating", seed]["curve"]):
            require(eval_hashes(a["evaluation"]) == eval_hashes(b["evaluation"]), "Checkpoint holdout cases differ across conditions")
        require(eval_hashes(runs["fixed", seed]["result"]["evaluation"]) ==
                eval_hashes(runs["rotating", seed]["result"]["evaluation"]), "Final/intervention/alignment cases differ across conditions")
        audit["paired_seeds"].append({"seed": seed, "common_initial_sha256": prepared_hash,
                                       "initial_weights_identical": True, "training_world_slots_identical": True,
                                       "layout_bits_identical": True, "same_assignment_updates": same_assignments,
                                       "same_private_input_updates_per_agent": identical_private,
                                       "checkpoint_and_final_evaluation_cases_identical": True,
                                       "interpretation": "Conditions share world slots and layout randomness; changing the pairing changes which agent occupies each slot, so per-agent observation histories need not be identical."})
    return config, completion, runs, audit


def make_summary(config, completion, runs, audit):
    result = {"status": "complete_saved_results_audited", "completion": completion, "integrity": audit,
              "replicate_unit": "independently prepared/trained population seed; not six pairs, two directions, or test cases",
              "scope": config["scope"], "conditions": config["conditions"], "seeds": config["seeds"],
              "population_no_message_expected_upper_bounds": config["population_no_message_expected_upper_bounds"],
              "primary_metric": config["primary_metric"], "runs": [], "condition_summary": {}, "paired_seed_differences": []}
    stages = [config["course_updates"], config["course_updates"] + config["exploratory_full_updates"],
              sum(config[k] for k in ("course_updates", "exploratory_full_updates", "pure_reward_updates"))]
    criteria = config["engineering_edge_criteria"]
    for (condition, seed), run in runs.items():
        final = run["result"]
        evaluation = final["evaluation"]
        entry = {"condition": condition, "seed": seed, "seconds": final["seconds"],
                 "groups": evaluation["aggregates"], "edges": [], "sender_alignment": evaluation["sender_alignment"],
                 "stages": [{"update": row["update"], "stage": row["stage"],
                             "action_entropy_weight": row["action_entropy_weight"], "groups": row["evaluation"]["aggregates"]}
                            for row in run["curve"] if row["update"] in stages]}
        for pair in evaluation["pairs"]:
            i, j = pair["agents"]
            edge = {"agents": [i, j], "group": pair["group"], "training_updates": final["edge_training_updates"][i][j],
                    "training_joint_cases": final["edge_joint_cases"][i][j], "tasks": {}, "intervention": pair["intervention"]}
            for task, modes in pair["tasks"].items():
                task_result = {mode: metric["mean_reward_per_step"] for mode, metric in modes.items()}
                task_result["normal_minus_shuffle"] = task_result["normal"] - task_result["shuffle"]
                task_result["directions"] = []
                for local_sender in (0, 1):
                    directions = {m: next(d for d in stats["by_direction"] if d["restricted_sender"] == local_sender)
                                  for m, stats in modes.items()}
                    d = {"sender": [i, j][local_sender], "receiver": [i, j][1 - local_sender],
                         **{m: value["success"] for m, value in directions.items()}}
                    d["normal_minus_shuffle"] = d["normal"] - d["shuffle"]
                    task_result["directions"].append(d)
                edge["tasks"][task] = task_result
            full = edge["tasks"]["full"]
            checks = {"full_success": full["normal"] >= criteria["full_success"],
                      "shuffle_gap": full["normal_minus_shuffle"] >= criteria["shuffle_gap"],
                      "both_direction_shuffle_gaps": all(d["normal_minus_shuffle"] >= criteria["direction_shuffle_gap"] for d in full["directions"])}
            edge["engineering_checks"] = {**checks, "all_pass": all(checks.values())}
            entry["edges"].append(edge)
        entry["engineering_edge_pass_count"] = sum(e["engineering_checks"]["all_pass"] for e in entry["edges"])
        entry["engineering_edge_criteria"] = criteria
        result["runs"].append(entry)
    for condition in config["conditions"]:
        selected = [r for r in result["runs"] if r["condition"] == condition]
        output = {"groups": {}}
        for group in GROUPS:
            output["groups"][group] = {}
            for task in ("full", "curriculum"):
                output["groups"][group][task] = {
                    **{mode: aggregate(r["groups"][group]["tasks"][task][mode]["mean_reward_per_step"] for r in selected) for mode in MODES},
                    "normal_minus_shuffle": aggregate(r["groups"][group]["tasks"][task]["normal_minus_shuffle"]["mean"] for r in selected)}
        output["all_four_raw_symbol_agreement"] = aggregate(r["sender_alignment"]["all_four_greedy_agreement"] for r in selected)
        output["engineering_edge_pass_counts_by_seed"] = {str(r["seed"]): r["engineering_edge_pass_count"] for r in selected}
        result["condition_summary"][condition] = output
    for seed in config["seeds"]:
        fixed = runs["fixed", seed]["result"]["evaluation"]
        rotating = runs["rotating", seed]["result"]["evaluation"]
        result["paired_seed_differences"].append({"seed": seed, "rotating_minus_fixed": {
            group: {task: {**{mode: success(rotating, group, mode, task) - success(fixed, group, mode, task) for mode in MODES},
                           "normal_minus_shuffle": gap(rotating, group, task) - gap(fixed, group, task)}
                    for task in ("full", "curriculum")} for group in GROUPS}})
    return result


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def plot_learning(folder, config, runs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    course = config["course_updates"]
    explore = course + config["exploratory_full_updates"]
    total = explore + config["pure_reward_updates"]
    for col, group in enumerate(GROUPS):
        for condition in config["conditions"]:
            for row in (0, 1):
                curves = []
                stochastic = []
                for seed in config["seeds"]:
                    records = runs[condition, seed]["curve"]
                    x = [r["update"] for r in records]
                    y = [success(r["evaluation"], group) if row == 0 else gap(r["evaluation"], group) for r in records]
                    curves.append(y)
                    stochastic.append([success(r["evaluation"], group, "stochastic") for r in records])
                    axes[row, col].plot(x, y, color=COLORS[condition], lw=.7, alpha=.25)
                axes[row, col].plot(x, np.mean(curves, axis=0), "o-", color=COLORS[condition], ms=3, lw=2, label=NAMES[condition])
                if row == 0:
                    axes[row, col].plot(x, np.mean(stochastic, axis=0), color=COLORS[condition], lw=1.3, ls="--", alpha=.8)
        for row in (0, 1):
            ax = axes[row, col]
            ax.axvline(course, color="#999999", ls=":", lw=.8)
            ax.axvline(explore, color="#999999", ls=":", lw=.8)
            ax.axvspan(course, explore, color="#dfb573", alpha=.08)
            ax.axvspan(explore, total, color="#a0bed0", alpha=.09)
            ax.set(xlim=(0, total), xlabel="完成的训练更新", title=GROUP_NAMES[group],
                   ylabel="完整场景成功率" if row == 0 else "正常消息 − 打乱消息", ylim=(0, 1.03) if row == 0 else (-.52, .52))
            bound = config["population_no_message_expected_upper_bounds"]["full"][group]
            ax.axhline(bound if row == 0 else 0,
                       color="#555555", ls=":" if row == 0 else "-", lw=.8)
            if row == 0:
                ax.text(.98, bound + .014, f"无有效消息上界 {bound:.1%}", transform=ax.get_yaxis_transform(),
                        ha="right", fontsize=8, color="#555555")
            ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle("伙伴条件：细实线为各训练种子，粗实线为均值；虚线为按策略采样，发送和行动均按学习概率采样\n固定群体仅训练 AB、CD；轮换群体训练全部六对。每对在独立营地中评估", fontsize=13)
    save(fig, folder, "partner_learning_curves")


def plot_edge_matrices(folder, config, runs, alignment=False):
    fig, axes = plt.subplots(len(config["seeds"]), 4, figsize=(14, 10), constrained_layout=True)
    images = [None, None]
    for row, seed in enumerate(config["seeds"]):
        for column in range(4):
            condition = config["conditions"][column % 2]
            metric_index = column // 2
            evaluation = runs[condition, seed]["result"]["evaluation"]
            pairs = evaluation["sender_alignment"]["pairs"] if alignment else evaluation["pairs"]
            matrix = np.full((4, 4), np.nan)
            for pair in pairs:
                i, j = pair["agents"]
                if alignment:
                    value = pair["greedy_agreement" if metric_index == 0 else "mean_total_variation"]
                else:
                    modes = pair["tasks"]["full"]
                    value = modes["normal"]["mean_reward_per_step"]
                    if metric_index == 1:
                        value -= modes["shuffle"]["mean_reward_per_step"]
                matrix[i, j] = matrix[j, i] = value
            ax = axes[row, column]
            divergent = metric_index == 1 and not alignment
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r" if divergent else "YlGnBu",
                              vmin=-.5 if divergent else 0, vmax=.5 if divergent else 1)
            images[metric_index] = image
            for i, j in PAIRS:
                for y, x in ((i, j), (j, i)):
                    value = matrix[y, x]
                    dark = abs(value) > .32 if divergent else value > .55
                    ax.text(x, y, f"{100 * value:.1f}", ha="center", va="center", fontsize=10,
                            color="white" if dark else "#152126")
                    if (i, j) in ORIGINAL:
                        ax.add_patch(Rectangle((x - .48, y - .48), .96, .96, fill=False, edgecolor="#dc921f", lw=1.3))
            ax.set(xticks=range(4), xticklabels=list("ABCD"), yticks=range(4), yticklabels=list("ABCD"))
            if column == 0:
                ax.set_ylabel(f"训练种子 {seed}")
            if row == 0:
                metric_name = ("原始符号一致率" if metric_index == 0 else "平均概率分布差异 TV") if alignment else ("正常消息成功率" if metric_index == 0 else "正常 − 打乱成功率")
                ax.set_title(NAMES[condition] + "\n" + metric_name, fontsize=11)
    for index in range(2):
        fig.colorbar(images[index], ax=axes[:, index * 2:index * 2 + 2], shrink=.72, pad=.015,
                     label="比例 / 概率差（格内数字为百分数）")
    if alignment:
        title = "每个种子内部，四主体看到完全相同的照片和公共输入；直接比较原始符号编号\n未置换符号，未跨种子对齐。高一致率也可能来自无信息的固定符号；橙框为 AB、CD"
    else:
        title = "六条配对边逐种子评估：橙框为 AB、CD，矩阵对称格是同一测量\n固定群体其余四对未共同训练；轮换群体全部配对均已训练，不属于相同接触量的泛化对照"
    fig.suptitle(title, fontsize=13)
    save(fig, folder, "partner_sender_alignment" if alignment else "partner_edge_matrices")


def plot_group_summary(folder, config, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, group in zip(axes, GROUPS):
        for offset, condition in zip((-.14, .14), config["conditions"]):
            for index, mode in enumerate(MODES):
                values = [success(runs[condition, seed]["result"]["evaluation"], group, mode) for seed in config["seeds"]]
                mean = np.mean(values)
                ax.errorbar(index + offset, mean, yerr=[[mean - min(values)], [max(values) - mean]],
                            color=COLORS[condition], marker="o", ms=6, capsize=4, lw=1.8,
                            label=NAMES[condition] if index == 0 else None)
                for jitter, seed, value in zip(np.linspace(-.045, .045, len(values)), config["seeds"], values):
                    ax.scatter(index + offset + jitter, value, color=COLORS[condition], s=13, alpha=.7)
        ax.set(xticks=range(4), xticklabels=["正常消息", "打乱消息", "固定符号 0", "按策略采样"],
               ylim=(0, 1.045), ylabel="完整场景成功率", title=GROUP_NAMES[group])
        bound = config["population_no_message_expected_upper_bounds"]["full"][group]
        ax.axhline(bound, color="#666666", ls=":", lw=.9)
        ax.text(.98, bound + .014, f"无有效消息上界 {bound:.1%}", transform=ax.get_yaxis_transform(),
                ha="right", fontsize=8, color="#555555")
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("最终表现：每个点是一整个群体种子的配对均值；大点与误差线为三种子均值和范围\n按策略采样：发送和行动均按学习概率采样。六条边共享主体，不能作为六个独立重复", fontsize=13)
    save(fig, folder, "partner_group_summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results/partners_001")
    args = parser.parse_args()
    folder = args.folder.resolve()
    config, completion, runs, audit = load_and_audit(folder)
    summary = make_summary(config, completion, runs, audit)
    summary["analysis_script_sha256"] = sha(Path(__file__))
    plt.rcParams.update({"font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot_learning(folder, config, runs)
    plot_edge_matrices(folder, config, runs)
    plot_group_summary(folder, config, runs)
    plot_edge_matrices(folder, config, runs, alignment=True)
    (folder / "partner_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"folder": str(folder), "runs": len(runs), "summary": summary["condition_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
