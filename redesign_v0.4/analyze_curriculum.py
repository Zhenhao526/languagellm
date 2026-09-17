"""Plot completed curriculum runs and summarize paired holdout interventions.

Reads saved measurements only. Incomplete/missing runs fail explicitly instead
of silently changing the preregistered seed set. Run with the experiment folder
as its positional argument, or use the default results/curriculum_001.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITION_NAMES = {"baseline": "无信号辅助项", "signalling": "有信号辅助项"}
COLORS = {"baseline": "#55708d", "signalling": "#008b7c"}
MODES = ("normal", "shuffle", "blank", "stochastic")
METRIC = "mean_reward_per_step"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def aggregate(values):
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(values), "mean": float(np.mean(values)),
            "min": min(values), "max": max(values)}


def direction(metrics, sender):
    entries = [r for r in metrics["by_direction"] if r["restricted_sender"] == sender]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one direction for sender {sender}")
    return entries[0]


def paired_trace(path, task, replacement):
    """Pair exact saved cases; report binary gains and losses without a p-value."""
    paths = [path / f"final_{task}_{mode}_trace.jsonl" for mode in ("normal", replacement)]
    traces = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    if len(traces[0]) != len(traces[1]) or not traces[0]:
        raise ValueError(f"Mismatched or empty paired traces: {path.name}/{task}/{replacement}")
    external_keys = ("case", "remaining", "inventory", "kinds", "image_ids", "sent")
    pairs = []
    for left, right in zip(*traces):
        if any(left[k] != right[k] for k in external_keys):
            raise ValueError(f"External cases or sent messages changed at {path.name}/{task}/{left['case']}")
        pairs.append((left, right))

    def summarize(selected):
        if not selected:
            return {"n": 0, "normal_minus_intervened_success": None}
        a = np.array([left["success"] for left, _ in selected], dtype=bool)
        b = np.array([right["success"] for _, right in selected], dtype=bool)
        return {"n": len(selected), "normal_success": float(a.mean()),
                "intervened_success": float(b.mean()),
                "normal_minus_intervened_success": float((a.astype(float) - b).mean()),
                "success_lost_cases": int((a & ~b).sum()),
                "success_gained_cases": int((~a & b).sum()),
                "both_success_cases": int((a & b).sum()),
                "both_failure_cases": int((~a & ~b).sum())}

    result = {"external_cases_and_sent_messages_identical": True,
              "overall": summarize(pairs), "by_direction": []}
    for sender in range(2):
        receiver = 1 - sender
        subset = [(a, b) for a, b in pairs
                  if a["kinds"][sender][0] == a["kinds"][sender][1]
                  and a["kinds"][receiver][0] != a["kinds"][receiver][1]]
        result["by_direction"].append({"restricted_sender": sender,
                                        "mixed_receiver": receiver, **summarize(subset)})
    return result


def load_runs(folder):
    config = read_json(folder / "config.json")
    completion = read_json(folder / "completed.json")
    expected = len(config["conditions"]) * len(config["seeds"])
    if completion.get("status") != "completed" or completion.get("runs") != expected:
        raise ValueError("Only analyze the complete declared experiment")
    total = sum(config[k] for k in ("course_updates", "transition_updates", "pure_reward_updates"))
    runs = {}
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            path = folder / f"{condition}_s{seed}"
            result, curve = read_json(path / "result.json"), read_json(path / "learning_curve.json")
            if (result["condition"], result["seed"], result["updates"]) != (condition, seed, total):
                raise ValueError(f"Inconsistent run identity or budget: {path}")
            if [r["update"] for r in curve] != config["checkpoints"]:
                raise ValueError(f"Incomplete or duplicate learning-curve checkpoints: {path}")
            for task in ("curriculum", "full"):
                hashes = {result[task][m]["external_cases_sha256"] for m in MODES}
                if len(hashes) != 1:
                    raise ValueError(f"Final external cases not paired: {path}/{task}")
                for row in curve:
                    if len({row[task][m]["external_cases_sha256"] for m in ("normal", "shuffle", "stochastic")}) != 1:
                        raise ValueError(f"Checkpoint cases not paired: {path}/{row['update']}/{task}")
            runs[condition, seed] = {"path": path, "result": result, "curve": curve}
    return config, completion, runs


def make_summary(config, completion, runs):
    stages = [config["course_updates"], config["course_updates"] + config["transition_updates"],
              sum(config[k] for k in ("course_updates", "transition_updates", "pure_reward_updates"))]
    criteria = config["engineering_criteria"]
    output = {"status": "complete_saved_results_only", "seeds": config["seeds"],
              "conditions": config["conditions"], "completed": completion,
              "condition_scope": "Both conditions have communication, the same curriculum and the same action exploration. Baseline has no signalling auxiliary loss; it is not a no-message training group.",
              "interpretation": "Across-seed mean and range; three seeds are not a population confidence interval. Positive normal-minus-shuffle gaps show functional message use, not a shared language.",
              "blank_interpretation": "All received messages replaced by symbol 0; this symbol may carry learned meaning and is not assumed neutral.",
              "engineering_criteria": criteria, "runs": [], "condition_summary": {},
              "paired_condition_differences": [], "pairing_audit": []}
    for (condition, seed), run in runs.items():
        final = run["result"]
        entry = {"condition": condition, "seed": seed, "seconds": final["seconds"],
                 "tasks": {}, "stages": [], "symbol_intervention": final["intervention"]}
        for task in ("curriculum", "full"):
            task_summary = {m: final[task][m][METRIC] for m in MODES}
            task_summary["normal_minus_shuffle"] = task_summary["normal"] - task_summary["shuffle"]
            task_summary["normal_minus_blank"] = task_summary["normal"] - task_summary["blank"]
            task_summary["directions"] = []
            for sender in range(2):
                row = {"restricted_sender": sender, "mixed_receiver": 1 - sender,
                       **{m: direction(final[task][m], sender)["success"] for m in MODES}}
                row["n"] = direction(final[task]["normal"], sender)["n"]
                row["normal_minus_shuffle"] = row["normal"] - row["shuffle"]
                task_summary["directions"].append(row)
            task_summary["paired_traces"] = {m: paired_trace(run["path"], task, m) for m in ("shuffle", "blank")}
            for mode, trace in task_summary["paired_traces"].items():
                if not np.isclose(trace["overall"]["normal_minus_intervened_success"], task_summary[f"normal_minus_{mode}"], atol=1e-12):
                    raise ValueError(f"Trace/aggregate mismatch: {condition}/{seed}/{task}/{mode}")
            entry["tasks"][task] = task_summary
        for update in stages:
            row = next(r for r in run["curve"] if r["update"] == update)
            entry["stages"].append({"update": update, "stage": row["stage"], "aux_weight": row["aux_weight"],
                                     "action_entropy_weight": row["action_entropy_weight"],
                                     **{task: {m: row[task][m][METRIC] for m in ("normal", "shuffle", "stochastic")}
                                        for task in ("curriculum", "full")}})
        earlier, later = entry["stages"][-2:]
        entry["after_auxiliary_withdrawal_change"] = {
            "from_update": earlier["update"], "to_update": later["update"],
            "note": "Both signalling loss and action entropy are absent after the transition stage; changes cannot be attributed to withdrawal of only one term.",
            **{task: {m: later[task][m] - earlier[task][m] for m in ("normal", "shuffle", "stochastic")}
               for task in ("curriculum", "full")}}
        checks = {"course_success": entry["tasks"]["curriculum"]["normal"] >= criteria["course_success"],
                  "full_success": entry["tasks"]["full"]["normal"] >= criteria["full_success"],
                  "full_shuffle_gap": entry["tasks"]["full"]["normal_minus_shuffle"] >= criteria["full_shuffle_gap"],
                  "both_direction_shuffle_gaps": all(d["normal_minus_shuffle"] >= criteria["direction_shuffle_gap"]
                                                       for d in entry["tasks"]["full"]["directions"])}
        entry["engineering_checks"] = {**checks, "all_pass": all(checks.values())}
        output["runs"].append(entry)
    for condition in config["conditions"]:
        selected = [r for r in output["runs"] if r["condition"] == condition]
        output["condition_summary"][condition] = {
            "engineering_pass_seeds": [r["seed"] for r in selected if r["engineering_checks"]["all_pass"]],
            "tasks": {task: {m: aggregate(r["tasks"][task][m] for r in selected)
                             for m in (*MODES, "normal_minus_shuffle", "normal_minus_blank")}
                      for task in ("curriculum", "full")},
            "full_direction_gaps": {str(sender): aggregate(r["tasks"]["full"]["directions"][sender]["normal_minus_shuffle"] for r in selected)
                                    for sender in range(2)}}
    for seed in config["seeds"]:
        baseline = next(r for r in output["runs"] if r["condition"] == "baseline" and r["seed"] == seed)
        signalling = next(r for r in output["runs"] if r["condition"] == "signalling" and r["seed"] == seed)
        difference = {"seed": seed, "signalling_minus_baseline": {
            task: {m: signalling["tasks"][task][m] - baseline["tasks"][task][m]
                   for m in (*MODES, "normal_minus_shuffle", "normal_minus_blank")} for task in ("curriculum", "full")}}
        output["paired_condition_differences"].append(difference)
        pair = [runs[c, seed]["path"] for c in ("baseline", "signalling")]
        final_cases_equal = all(runs["baseline", seed]["result"][task][mode]["external_cases_sha256"] ==
                                runs["signalling", seed]["result"][task][mode]["external_cases_sha256"]
                                for task in ("curriculum", "full") for mode in MODES)
        if not final_cases_equal:
            raise ValueError(f"Condition final cases differ for seed {seed}")
        initial = [hashlib.sha256((p / "initial.pt").read_bytes()).hexdigest() for p in pair]
        metrics = [[json.loads(line) for line in (p / "training_metrics.jsonl").read_text().splitlines()] for p in pair]
        world_equal = (len(metrics[0]) == len(metrics[1]) and all(
            a["update"] == b["update"] and a["world_input_sha256"] == b["world_input_sha256"]
            for a, b in zip(*metrics)))
        if not world_equal:
            raise ValueError(f"Conditions did not receive identical training scenes/photos for seed {seed}")
        output["pairing_audit"].append({"seed": seed, "initial_serialized_file_sha256": initial,
                                       "serialized_initial_files_identical": initial[0] == initial[1],
                                       "training_scene_and_photo_hashes_identical": world_equal,
                                       "final_cases_identical_across_conditions": final_cases_equal,
                                       "training_updates_compared": len(metrics[0]),
                                       "note": "Different serialized file hashes alone do not imply different parameters; checkpoint initialization is specified in config."})
    return output


def save_figure(fig, folder, stem):
    fig.savefig(folder / f"{stem}.png", dpi=180)
    fig.savefig(folder / f"{stem}.pdf")
    plt.close(fig)


def stage_background(ax, config):
    course = config["course_updates"]
    transition = course + config["transition_updates"]
    end = transition + config["pure_reward_updates"]
    ax.axvspan(course, transition, color="#e7b267", alpha=.09, zorder=0)
    ax.axvspan(transition, end, color="#a5b8cc", alpha=.09, zorder=0)
    for boundary in (course, transition):
        ax.axvline(boundary, color="#9a9a9a", lw=.9, ls=":", zorder=0)
    ax.set_xlim(0, end)


def plot_curves(folder, config, runs):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    for row, task in enumerate(("curriculum", "full")):
        for col, mode in enumerate(("normal", "stochastic")):
            ax = axes[row, col]
            for condition in config["conditions"]:
                values = []
                for seed in config["seeds"]:
                    curve = runs[condition, seed]["curve"]
                    x = [r["update"] for r in curve]
                    y = [r[task][mode][METRIC] for r in curve]
                    values.append(y)
                    ax.plot(x, y, color=COLORS[condition], lw=.9, alpha=.28)
                ax.plot(x, np.mean(values, axis=0), "o-", color=COLORS[condition], lw=2.2, ms=3,
                        label=CONDITION_NAMES[condition])
            bound = config["task_bounds"][task]["no_message_expected_success_upper_bound"]
            ax.axhline(bound, color="#555555", lw=1, ls="--", label="无有效消息的期望上界")
            stage_background(ax, config)
            ax.set(xlabel="完成的训练更新", ylabel="留出照片上的资源配合成功率", ylim=(0, 1.04),
                   title=("课程场景" if task == "curriculum" else "完整场景") + " · " +
                         ("取最大概率动作" if mode == "normal" else "按概率采样动作"))
            ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    course = config["course_updates"]
    withdrawal = course + config["transition_updates"]
    fig.suptitle(f"两组均有通信：细线为单个种子，粗线为三种子均值\n{course} 次后恢复完整场景；{withdrawal} 次后两组均只使用资源奖励", fontsize=14)
    save_figure(fig, folder, "learning_curves")


def plot_final(folder, config, runs):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    mode_labels = ["正常消息", "打乱消息", "固定为符号 0"]
    panels = [("curriculum", None, "课程场景"), ("full", None, "完整场景"),
              ("full", 0, "完整场景：A 受限 → B 选择"), ("full", 1, "完整场景：B 受限 → A 选择")]
    for ax, (task, sender, title) in zip(axes.flat, panels):
        for offset, condition in zip((-.16, .16), config["conditions"]):
            means, lower, upper = [], [], []
            for ix, mode in enumerate(("normal", "shuffle", "blank")):
                values = [runs[condition, seed]["result"][task][mode][METRIC] if sender is None
                          else direction(runs[condition, seed]["result"][task][mode], sender)["success"]
                          for seed in config["seeds"]]
                means.append(np.mean(values)); lower.append(np.mean(values) - min(values)); upper.append(max(values) - np.mean(values))
                jitter = np.linspace(-.045, .045, len(values))
                ax.scatter(ix + offset + jitter, values, s=20, color=COLORS[condition], alpha=.6, zorder=5)
            ax.bar(np.arange(3) + offset, means, width=.29, color=COLORS[condition], alpha=.55,
                   yerr=[lower, upper], capsize=4, label=CONDITION_NAMES[condition], error_kw={"lw": 1.1})
        bound = config["task_bounds"][task]["no_message_expected_success_upper_bound"] if sender is None else .5
        ax.axhline(bound, color="#555555", ls="--", lw=.9)
        ax.set(xticks=np.arange(3), xticklabels=mode_labels, ylim=(0, 1.05), ylabel="资源配合成功率", title=title)
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle("两组均有通信：最终均值、三种子范围与各个种子\n消息干预使用完全相同的场景和照片；固定符号 0 不等于行为中性的沉默", fontsize=14)
    save_figure(fig, folder, "final_communication")


def plot_interventions(folder, config, runs):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)
    for sender in range(2):
        for row, (metric, title) in enumerate((
            ("fraction_cases_resource_changes_for_some_used_symbol", "更换已使用符号后，所选资源改变的案例比例"),
            ("mean_food_probability_range", "更换已使用符号后，取食物概率的平均变化范围"))):
            ax = axes[row, sender]
            for ix, condition in enumerate(config["conditions"]):
                for jitter, seed in zip(np.linspace(-.09, .09, len(config["seeds"])), config["seeds"]):
                    entries = runs[condition, seed]["result"]["intervention"]["directions"]
                    d = next(d for d in entries if d["sender"] == sender)
                    value = d["observed_symbol_sensitivity"][metric]
                    if value is None:
                        ax.text(ix + jitter, .045, f"{seed}\n未具备 ≥2 个常用符号", fontsize=7, ha="center", color=COLORS[condition])
                    else:
                        ax.scatter(ix + jitter, value, s=35, color=COLORS[condition])
                        ax.annotate(str(seed), (ix + jitter, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
            ax.set(xticks=[0, 1], xticklabels=[CONDITION_NAMES[c] for c in config["conditions"]],
                   xlim=(-.45, 1.45), ylim=(-.02, 1.13), ylabel="比例 / 概率差", title=("A → B" if sender == 0 else "B → A") + "：" + title)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("固定接收者观察和伙伴行动，仅替换收到的离散符号\n仅统计一方受限、另一方可选的案例；符号范围由独立照片样本中的发送频次确定", fontsize=14)
    save_figure(fig, folder, "symbol_interventions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results/curriculum_001")
    args = parser.parse_args()
    folder = args.folder.resolve()
    config, completion, runs = load_runs(folder)
    summary = make_summary(config, completion, runs)
    plt.rcParams.update({"font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot_curves(folder, config, runs)
    plot_final(folder, config, runs)
    plot_interventions(folder, config, runs)
    write_json(folder / "paired_summary.json", summary)
    print(json.dumps({"output": str(folder), "runs": len(runs),
                      "conditions": summary["condition_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
