"""Audit saved contact readaptation and describe paired population trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_partners import (PAIRS, ORIGINAL, GROUPS, MODES, read, sha, require,
                              aggregate, success, gap, audit_training,
                              audit_evaluation, eval_hashes, save)


ROOT = Path(__file__).resolve().parent
FF, FR, RR = "fixed_to_fixed", "fixed_to_rotating", "rotating_to_rotating"
NAMES = {FF: "固定起点→继续固定", FR: "固定起点→轮换", RR: "轮换起点→继续轮换（参照）"}
COLORS = {FF: "#526d91", FR: "#00917c", RR: "#929292"}
GROUP_NAMES = {"original": "原固定配对 AB、CD", "cross": "原固定组间四对", "all": "全部六对"}


def verify_manifest(folder, name):
    manifest = read(folder / name)
    for filename, expected in manifest.items():
        require(sha(folder / filename) == expected, f"Artifact hash mismatch: {folder.name}/{filename}")
    return len(manifest)


def drift_value(drift, kind, metric, context="mixed"):
    if kind == "sender":
        values = [(s["overall"] if context == "overall" else s["by_own_context"][context])[metric] for s in drift["senders"]]
    else:
        values = [s["by_own_context"][context]["initial_received_support_summary"]["initial_usage_weighted_mean"][metric]
                  for s in drift["listeners"]]
    require(all(value is not None for value in values), "Missing drift measurements in the declared probe")
    return float(np.mean(values))


def drift_metrics(drift):
    return {"sender_overall_tv": drift_value(drift, "sender", "mean_probability_total_variation", "overall"),
            "sender_overall_raw_symbol_change": drift_value(drift, "sender", "greedy_raw_symbol_change_rate", "overall"),
            "sender_mixed_tv": drift_value(drift, "sender", "mean_probability_total_variation"),
            "sender_mixed_raw_symbol_change": drift_value(drift, "sender", "greedy_raw_symbol_change_rate"),
            "sender_food_food_tv": drift_value(drift, "sender", "mean_probability_total_variation", "food_food"),
            "sender_food_food_raw_symbol_change": drift_value(drift, "sender", "greedy_raw_symbol_change_rate", "food_food"),
            "sender_water_water_tv": drift_value(drift, "sender", "mean_probability_total_variation", "water_water"),
            "sender_water_water_raw_symbol_change": drift_value(drift, "sender", "greedy_raw_symbol_change_rate", "water_water"),
            "listener_mixed_initial_support_weighted_tv": drift_value(drift, "listener", "mean_action_probability_total_variation"),
            "listener_mixed_initial_support_weighted_resource_change": drift_value(drift, "listener", "greedy_resource_change_rate")}


def edge_checks(pair, criteria):
    modes = pair["tasks"]["full"]
    directions = []
    for sender in (0, 1):
        normal = next(d for d in modes["normal"]["by_direction"] if d["restricted_sender"] == sender)
        shuffled = next(d for d in modes["shuffle"]["by_direction"] if d["restricted_sender"] == sender)
        directions.append({"sender": normal["population_sender"], "receiver": normal["population_receiver"],
                           "normal": normal["success"], "shuffle": shuffled["success"],
                           "normal_minus_shuffle": normal["success"] - shuffled["success"]})
    normal = modes["normal"]["mean_reward_per_step"]
    shuffled = modes["shuffle"]["mean_reward_per_step"]
    checks = {"full_success": normal >= criteria["full_success"],
              "shuffle_gap": normal - shuffled >= criteria["shuffle_gap"],
              "both_direction_shuffle_gaps": all(d["normal_minus_shuffle"] >= criteria["direction_shuffle_gap"] for d in directions)}
    return {"agents": pair["agents"], "group": pair["group"], "normal": normal, "shuffle": shuffled,
            "normal_minus_shuffle": normal - shuffled, "directions": directions,
            "checks": {**checks, "all_pass": all(checks.values())}}


def load_and_audit(folder):
    config, completion = read(folder / "config.json"), read(folder / "completed.json")
    require(config["conditions"] == [FF, FR, RR], "Unexpected contact conditions")
    require(completion["status"] == "completed" and completion["runs"] == 3 * len(config["seeds"]), "Incomplete contact batch")
    batch_hash_count = verify_manifest(folder, "batch_artifact_hashes.json")
    source_hashes = read(folder / "source_hashes.json")
    for name, expected in source_hashes.items():
        require(sha(folder / "source" / name) == expected, f"Saved source differs: {name}")
    require(read(folder / "source_validation.json")["passed"], "Origin source validation failed")
    origin_batch = folder.parent / config["source_batch"]
    origin_config = read(origin_batch / "config.json")
    bounds = origin_config["population_no_message_expected_upper_bounds"]
    listed = {(r["condition"], r["seed"]): r for r in read(folder / "results.json")}
    require(len(listed) == completion["runs"], "Incomplete top-level result list")
    normalized_config = {**config, "course_updates": 0, "exploratory_full_updates": 0,
                         "pure_reward_updates": config["additional_updates"]}
    runs, rows_by_run = {}, {}
    audit = {"batch_artifact_hashes_verified": batch_hash_count, "saved_source_hashes_verified": source_hashes,
             "runs": [], "paired_seeds": [],
             "scope": "Artifact, identity, schedule, exposure and evaluation pairing audit; no model training or new behavioral trials."}
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            path = folder / f"{condition}_s{seed}"
            artifacts = verify_manifest(path, "artifact_hashes.json")
            final, curve = read(path / "result.json"), read(path / "learning_curve.json")
            origin, old_result = read(path / "origin.json"), read(path / "origin_result.json")
            require(final == listed[condition, seed] and final["origin"] == origin, "Result/origin metadata copies differ")
            require((final["seed"], final["condition"], final["updates"]) == (seed, condition, config["additional_updates"]), "Contact run identity/budget mismatch")
            expected_origin = "rotating" if condition == RR else "fixed"
            old_folder = origin_batch / f"{expected_origin}_s{seed}"
            old_checkpoint = old_folder / f"checkpoint_{config['origin_update']:04d}.pt"
            require(sha(old_checkpoint) == origin["origin_sha256"] == origin["initial_sha256"] == sha(path / "initial.pt") == sha(path / "checkpoint_0000.pt"), "Initial state differs from declared old endpoint")
            require(old_result == read(old_folder / "result.json") and sha(path / "origin_result.json") == origin["origin_result_sha256"], "Copied origin result differs")
            require(origin["fresh_adam"] and origin["initial_optimizer_state_entries"] == [0, 0, 0, 0], "Unexpected optimizer origin")
            require(sha(path / "protocol_reference.pt") == origin["protocol_reference_sha256"], "Protocol reference artifact changed")
            require([row["update"] for row in curve] == config["checkpoints"], "Checkpoint list is incomplete")
            old_curve = read(old_folder / "learning_curve.json")
            zero = read(path / "zero_update_validation.json")
            require(zero["passed"] and zero["origin_learning_curve_sha256"] == sha(old_folder / "learning_curve.json"), "Zero-update verification metadata failed")
            require(curve[0]["evaluation"] == old_curve[-1]["evaluation"], "Zero-update full checkpoint evaluation differs from old endpoint")
            rows = [json.loads(line) for line in (path / "training_metrics.jsonl").read_text().splitlines()]
            for row in rows:
                require(row["stage"] == "contact_pure_reward" and row["lifetime_update"] == config["origin_update"] + row["update"], "Contact stage/lifetime counter mismatch")
            counts, snapshots = audit_training([dict(row, stage="pure_reward") for row in rows], normalized_config,
                                               "fixed" if condition == FF else "rotating")
            old_edges = np.array(old_result["edge_training_updates"])
            require(np.array_equal(old_edges, origin["origin_edge_training_updates"]), "Origin exposure matrix mismatch")
            require(np.array_equal(counts, final["edge_training_updates"]), "Additional exposure matrix mismatch")
            require(np.array_equal(counts + old_edges, final["cumulative_edge_training_updates"]), "Cumulative exposure matrix mismatch")
            require(np.array_equal(counts * config["batch_size_per_agent"], final["edge_joint_cases"]), "Additional edge case budget mismatch")
            for row in curve:
                require(row["lifetime_update"] == config["origin_update"] + row["update"], "Checkpoint lifetime counter mismatch")
                require(np.array_equal(snapshots[row["update"]], row["edge_training_updates"]), "Checkpoint additional edge count mismatch")
                require(np.array_equal(snapshots[row["update"]] + old_edges, row["cumulative_edge_training_updates"]), "Checkpoint cumulative edge count mismatch")
                audit_evaluation(row["evaluation"], config["checkpoint_evaluation_n"])
                drift = row["protocol_drift"]
                encountered = [list(pair) for pair in PAIRS if (snapshots[row["update"]] + old_edges)[pair[0], pair[1]] > 0]
                require(drift["current_trained_pairs"] == encountered, "Current protocol-support graph does not match actually encountered edges")
                require(drift["probe_sha256"] == origin["protocol_probe_sha256"] and
                        drift["support_probe_sha256"] == origin["protocol_support_probe_sha256"], "Protocol probe inputs changed")
                require(drift["initial_received_support"] == curve[0]["protocol_drift"]["initial_received_support"], "Initial listener support mask changed over time")
            require(all(value == 0 for value in drift_metrics(curve[0]["protocol_drift"]).values()), "Zero-update drift is nonzero")
            require(final["protocol_drift"] == curve[-1]["protocol_drift"], "Final drift differs from final checkpoint")
            traces = audit_evaluation(final["evaluation"], config["evaluation_n"], True, path)
            require(eval_hashes(final["evaluation"]) == eval_hashes(old_result["evaluation"]), "Final and original endpoint holdout cases differ")
            runs[condition, seed] = {"path": path, "final": final, "curve": curve, "origin": origin, "old_result": old_result}
            rows_by_run[condition, seed] = rows
            audit["runs"].append({"condition": condition, "seed": seed, "artifact_hashes_verified": artifacts,
                                   "trace_files_sha256_verified": traces, "training_updates_verified": len(rows),
                                   "origin_checkpoint_exact_match": True, "zero_update_evaluation_exact_match": True,
                                   "protocol_probe_and_initial_support_fixed": True,
                                   "additional_edge_training_updates": counts.tolist(), "cumulative_edge_training_updates": (counts + old_edges).tolist()})
        initial_equal = runs[FF, seed]["origin"]["initial_sha256"] == runs[FR, seed]["origin"]["initial_sha256"]
        require(initial_equal and runs[FF, seed]["curve"][0]["evaluation"] == runs[FR, seed]["curve"][0]["evaluation"], "Primary contrast does not share one fixed origin")
        reference_different = runs[FF, seed]["origin"]["initial_sha256"] != runs[RR, seed]["origin"]["initial_sha256"]
        require(reference_different, "Reference unexpectedly shares primary initial state")
        for comparison in (FR, RR):
            left, right = rows_by_run[FF, seed], rows_by_run[comparison, seed]
            require(all(a["world_slots_sha256"] == b["world_slots_sha256"] and a["layout_bits"] == b["layout_bits"] for a, b in zip(left, right)), "World/photo slots or layout randomness are unpaired")
            for a, b in zip(runs[FF, seed]["curve"], runs[comparison, seed]["curve"]):
                require(eval_hashes(a["evaluation"]) == eval_hashes(b["evaluation"]), "Checkpoint cases differ across conditions")
                require(a["protocol_drift"]["probe_sha256"] == b["protocol_drift"]["probe_sha256"] and
                        a["protocol_drift"]["support_probe_sha256"] == b["protocol_drift"]["support_probe_sha256"], "Protocol probe cases differ across conditions")
            require(eval_hashes(runs[FF, seed]["final"]["evaluation"]) == eval_hashes(runs[comparison, seed]["final"]["evaluation"]), "Final cases differ across conditions")
        require(runs[FF, seed]["curve"][0]["protocol_drift"]["initial_received_support"] ==
                runs[FR, seed]["curve"][0]["protocol_drift"]["initial_received_support"], "Primary contrast listener drift uses different initial symbol support")
        audit["paired_seeds"].append({"seed": seed, "primary_initial_checkpoint_identical": True,
                                       "reference_initial_checkpoint_different": True,
                                       "world_slots_and_layouts_paired": True,
                                       "checkpoint_final_and_drift_probe_cases_paired": True,
                                       "primary_initial_listener_support_identical": True})
    return config, completion, runs, audit, bounds


def make_summary(config, completion, runs, audit, bounds):
    criteria = config["engineering_edge_criteria"]
    output = {"status": "complete_saved_results_audited", "completion": completion, "integrity": audit,
              "primary_comparison": config["primary_comparison"], "reference_condition": config["reference_condition"],
              "optimizer_limitation": config["optimizer_limitation"], "policy_sampling": config["policy_sampling"],
              "replicate_unit": config["replicate_unit"], "population_no_message_expected_upper_bounds": bounds,
              "bound_interpretation": "The population bound concerns policies without any partner information. Within-pair message shuffling preserves sender-specific marginal symbol use and can retain partner identity cues; the no-partner-information bound does not constrain shuffled-message performance.",
              "engineering_edge_criteria": criteria,
              "threshold_interpretation": "First pass means first recorded checkpoint meeting engineering thresholds at n=512; unobserved updates are not assessed. Final endpoint uses a separate n=4096 paired holdout sample. These are not language-emergence or significance thresholds.",
              "drift_interpretation": "Same observations and received symbol, relative to each run's own starting policy. Primary listener drift uses the unchanged initial partners' estimated symbol support and initial usage weights. RR has different starting policies and support, so it is descriptive reference only.",
              "runs": [], "condition_summary": {}, "primary_paired_seed_differences": []}
    for (condition, seed), run in runs.items():
        final, initial = run["final"]["evaluation"], run["old_result"]["evaluation"]
        trajectory = []
        for row in run["curve"]:
            edges = [edge_checks(p, criteria) for p in row["evaluation"]["pairs"]]
            trajectory.append({"update": row["update"], "lifetime_update": row["lifetime_update"],
                               "groups": row["evaluation"]["aggregates"], "edges": edges,
                               "cross_pass_count": sum(e["checks"]["all_pass"] for e in edges if e["group"] == "cross"),
                               "original_pass_count": sum(e["checks"]["all_pass"] for e in edges if e["group"] == "original"),
                               "drift_population_mean": drift_metrics(row["protocol_drift"])})
        hits = [row["update"] for row in trajectory if row["cross_pass_count"] == 4]
        first = hits[0] if hits else None
        edges = []
        for index, pair in enumerate(final["pairs"]):
            i, j = pair["agents"]
            observed_passes = [row["update"] for row in trajectory if row["edges"][index]["checks"]["all_pass"]]
            edges.append({**edge_checks(pair, criteria),
                          "first_recorded_pass_update": observed_passes[0] if observed_passes else None,
                          "recorded_pass_updates": observed_passes,
                          "additional_training_updates": run["final"]["edge_training_updates"][i][j],
                          "cumulative_training_updates": run["final"]["cumulative_edge_training_updates"][i][j],
                          "tasks": pair["tasks"]})
        entry = {"condition": condition, "seed": seed, "seconds": run["final"]["seconds"],
                 "origin": run["origin"], "initial_groups": initial["aggregates"], "final_groups": final["aggregates"],
                 "edges": edges, "trajectory": trajectory,
                 "first_recorded_all_four_cross_pass_update": first,
                 "all_later_recorded_cross_checks_pass": all(row["cross_pass_count"] == 4 for row in trajectory if row["update"] >= first) if first is not None else None,
                 "final_cross_pass_count": sum(e["checks"]["all_pass"] for e in edges if e["group"] == "cross"),
                 "final_original_pass_count": sum(e["checks"]["all_pass"] for e in edges if e["group"] == "original"),
                 "final_protocol_drift": run["final"]["protocol_drift"],
                 "final_sender_alignment": final["sender_alignment"],
                 "endpoint_change_from_own_origin": {g: {**{m: success(final, g, m) - success(initial, g, m) for m in MODES},
                                                       "normal_minus_shuffle": gap(final, g) - gap(initial, g)} for g in GROUPS},
                 "largest_recorded_original_success_drop_from_checkpoint_origin": max(0., success(run["curve"][0]["evaluation"], "original") -
                                                                                      min(success(row["evaluation"], "original") for row in run["curve"]))}
        output["runs"].append(entry)
    for condition in config["conditions"]:
        selected = [r for r in output["runs"] if r["condition"] == condition]
        result = {"groups": {}, "cross_first_pass_by_seed": {str(r["seed"]): r["first_recorded_all_four_cross_pass_update"] for r in selected},
                  "cross_final_pass_count_by_seed": {str(r["seed"]): r["final_cross_pass_count"] for r in selected}}
        for group in GROUPS:
            result["groups"][group] = {**{m: aggregate(r["final_groups"][group]["tasks"]["full"][m]["mean_reward_per_step"] for r in selected) for m in MODES},
                                       "normal_minus_shuffle": aggregate(r["final_groups"][group]["tasks"]["full"]["normal_minus_shuffle"]["mean"] for r in selected)}
        result["final_drift_population_mean"] = {key: aggregate(r["trajectory"][-1]["drift_population_mean"][key] for r in selected)
                                                 for key in selected[0]["trajectory"][-1]["drift_population_mean"]}
        output["condition_summary"][condition] = result
    for seed in config["seeds"]:
        baseline, contact = runs[FF, seed], runs[FR, seed]
        by_update = []
        for a, b in zip(baseline["curve"], contact["curve"]):
            by_update.append({"update": a["update"], "groups": {
                g: {"normal": success(b["evaluation"], g) - success(a["evaluation"], g),
                    "normal_minus_shuffle": gap(b["evaluation"], g) - gap(a["evaluation"], g)} for g in GROUPS}})
        output["primary_paired_seed_differences"].append({"seed": seed, "direction": "fixed_to_rotating minus fixed_to_fixed", "trajectory": by_update,
            "final_groups": {g: {**{m: success(contact["final"]["evaluation"], g, m) - success(baseline["final"]["evaluation"], g, m) for m in MODES},
                                 "normal_minus_shuffle": gap(contact["final"]["evaluation"], g) - gap(baseline["final"]["evaluation"], g)} for g in GROUPS}})
    return output


def plot_process(folder, config, runs, bounds, early=False):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    measured_gaps = [gap(row["evaluation"], group) for run in runs.values() for row in run["curve"]
                     if not early or row["update"] <= 50 for group in GROUPS]
    gap_limits = (min(-.05, np.floor(min(measured_gaps) * 20) / 20 - .02),
                  max(.45, np.ceil(max(measured_gaps) * 20) / 20 + .02))
    for column, group in enumerate(GROUPS):
        for condition in config["conditions"]:
            for row in (0, 1):
                ys = []
                for seed in config["seeds"]:
                    records = [r for r in runs[condition, seed]["curve"] if not early or r["update"] <= 50]
                    x = [r["update"] for r in records]
                    y = [success(r["evaluation"], group) if row == 0 else gap(r["evaluation"], group) for r in records]
                    ys.append(y)
                    axes[row, column].plot(x, y, color=COLORS[condition], alpha=.25, lw=.7, ls="--" if condition == RR else "-")
                axes[row, column].plot(x, np.mean(ys, axis=0), "o", ls="--" if condition == RR else "-", color=COLORS[condition], lw=2, ms=3, label=NAMES[condition])
        for row in (0, 1):
            ax = axes[row, column]
            reference = bounds["full"][group] if row == 0 else 0
            ax.axhline(reference, color="#555555", lw=.8, ls=":" if row == 0 else "-")
            if row == 0:
                ax.text(.97, reference + .015, f"无伙伴信息上界 {reference:.1%}", transform=ax.get_yaxis_transform(),
                        ha="right", fontsize=8, color="#555555")
            ax.set(title=GROUP_NAMES[group], xlabel="从保存起点继续训练的更新次数", ylabel="完整场景成功率" if row == 0 else "正常消息 − 打乱消息",
                   xlim=(0, 50 if early else config["additional_updates"]), ylim=(0, 1.04) if row == 0 else gap_limits)
            ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(("接触最初 50 次更新" if early else "已有约定的接触再适应") + "：细线为每个群体种子，粗线为三种子均值\n蓝、绿两组来自同一固定伙伴终点；灰色参照起点不同。训练按已有策略采样，无额外熵奖励\n点线只约束没有伙伴信息的策略；打乱消息仍可能保留伙伴身份线索，不受此界约束", fontsize=12)
    save(fig, folder, "contact_early_process" if early else "contact_process")


def plot_paired(folder, config, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    seed_colors = ("#386cb0", "#ed8a35", "#8b65a7")
    for col, group in enumerate(GROUPS):
        for row, metric in enumerate(("normal", "normal_minus_shuffle")):
            all_y = []
            for color, entry in zip(seed_colors, summary["primary_paired_seed_differences"]):
                x = [r["update"] for r in entry["trajectory"]]
                y = [r["groups"][group][metric] for r in entry["trajectory"]]
                all_y.append(y)
                axes[row, col].plot(x, y, "o-", lw=1.3, ms=3, color=color, label=f"种子 {entry['seed']}")
            axes[row, col].plot(x, np.mean(all_y, axis=0), color="#202d35", lw=2.2, label="三种子均值")
            axes[row, col].axhline(0, color="#777777", lw=.8)
            axes[row, col].grid(axis="y", alpha=.15)
            axes[row, col].set(title=GROUP_NAMES[group], xlabel="继续训练的更新次数",
                              ylabel="轮换 − 固定：成功率差" if row == 0 else "轮换 − 固定：消息收益差", xlim=(0, config["additional_updates"]))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("主要配对对比：固定起点→轮换，减去同一起点→继续固定\n仅比较两种同起点处理；未将不同起点的轮换参照纳入差值。点间未记录的变化不作推断", fontsize=13)
    save(fig, folder, "contact_paired_changes")


def plot_drift(folder, config, runs):
    specs = [("sender_mixed_tv", "发送端：符号概率分布 TV"),
             ("sender_mixed_raw_symbol_change", "发送端：最大概率符号改变比例"),
             ("listener_mixed_initial_support_weighted_tv", "接收端：资源选择概率分布 TV"),
             ("listener_mixed_initial_support_weighted_resource_change", "接收端：最大概率资源选择改变比例")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, specs):
        for condition in config["conditions"]:
            ys = []
            for seed in config["seeds"]:
                records = runs[condition, seed]["curve"]
                x = [r["update"] for r in records]
                y = [drift_metrics(r["protocol_drift"])[key] for r in records]
                ys.append(y)
                ax.plot(x, y, color=COLORS[condition], lw=.7, alpha=.25, ls="--" if condition == RR else "-")
            ax.plot(x, np.mean(ys, axis=0), "o", ls="--" if condition == RR else "-", color=COLORS[condition], ms=3, lw=2, label=NAMES[condition])
        ax.set(title=title, xlabel="继续训练的更新次数", ylabel="相对各自起点的概率差 / 比例", ylim=(0, 1.03), xlim=(0, config["additional_updates"]))
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("相同照片、公共输入和收到的符号：只看自己同时拥有两种资源的情境\n接收端仅在起点伙伴估计常用的编号上比较，按起点频率加权；不代表对新增编号的反应\n先平均四主体，再按群体种子作图。漂移不等于语义创新", fontsize=12)
    save(fig, folder, "contact_protocol_drift")


def plot_sender_contexts(folder, config, runs):
    contexts = (("food_food", "只有食物：FF"), ("mixed", "两种资源：FW/WF"), ("water_water", "只有水：WW"))
    metrics = (("mean_probability_total_variation", "发送符号概率分布 TV"),
               ("greedy_raw_symbol_change_rate", "最大概率发送符号改变比例"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (context, title) in enumerate(contexts):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            for condition in config["conditions"]:
                values = []
                for seed in config["seeds"]:
                    records = runs[condition, seed]["curve"]
                    x = [r["update"] for r in records]
                    y = [drift_value(r["protocol_drift"], "sender", metric, context) for r in records]
                    values.append(y)
                    ax.plot(x, y, color=COLORS[condition], ls="--" if condition == RR else "-", lw=.7, alpha=.25)
                ax.plot(x, np.mean(values, axis=0), "o", ls="--" if condition == RR else "-", color=COLORS[condition], ms=3, lw=2, label=NAMES[condition])
            ax.set(title=title, xlabel="继续训练的更新次数", ylabel=ylabel, ylim=(0, 1.03), xlim=(0, config["additional_updates"]))
            ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("发送端在三类私人资源情境中的变化：相同照片和公共输入，相对各自保存起点\n细线为群体种子，粗线为三种子均值；每个群体先平均四主体。灰色参照起点不同；原始编号变化不等于语义创新", fontsize=12)
    save(fig, folder, "contact_sender_contexts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path, default=ROOT / "results/contact_001")
    args = parser.parse_args()
    folder = args.folder.resolve()
    config, completion, runs, audit, bounds = load_and_audit(folder)
    summary = make_summary(config, completion, runs, audit, bounds)
    summary["analysis_script_sha256"] = sha(Path(__file__))
    plt.rcParams.update({"font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot_process(folder, config, runs, bounds)
    plot_process(folder, config, runs, bounds, early=True)
    plot_paired(folder, config, summary)
    plot_drift(folder, config, runs)
    plot_sender_contexts(folder, config, runs)
    (folder / "contact_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"folder": str(folder), "runs": len(runs), "condition_summary": summary["condition_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
