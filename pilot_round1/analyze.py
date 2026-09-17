"""Analyze recorded pilot runs without training, imputing, or selecting seeds.

Usage: python -m pilot_round1.analyze --results DIR --output DIR

Counts contract:
    message_counts[direction][true_class][symbol]
    response_counts[direction][symbol][selected_class]
Direction 0 is A -> B, direction 1 is B -> A. Symbol 0 is displayed as S0;
its interpretation as silence must be established by the recorded config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

THRESHOLD = 0.90
CONSECUTIVE_CHECKPOINTS = 3
CHANCE = 0.25
CLASSES = ("circle", "square", "triangle", "cross")
DIRECTIONS = ("A -> B", "B -> A")
TASK_NAMES = {"emergent": "自发协议", "fixed": "固定信号", "no_comm": "无通信"}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def pct(value: Any) -> str:
    number = finite_number(value)
    return "未记录" if number is None else f"{number * 100:.2f}%"


def intervention_pct(final: dict[str, Any], key: str) -> str:
    if final.get("intervention_n_trials") == 0:
        return "不可估（n=0）"
    return pct(final.get(key))


def number_text(value: Any) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def table_cell(value: Any) -> str:
    return number_text(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(table_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def performance_confirmation(evaluations: list[dict[str, Any]]) -> int | float | None:
    """Return the THIRD checkpoint of the first uninterrupted passing triple.

    Missing/invalid accuracy interrupts a triple. Terminal intervention results
    are deliberately not consulted, and failed runs remain unconfirmed.
    """
    streak = 0
    for point in evaluations:
        accuracy = finite_number(point.get("accuracy"))
        streak = streak + 1 if accuracy is not None and accuracy >= THRESHOLD else 0
        if streak >= CONSECUTIVE_CHECKPOINTS:
            return point["step"]
    return None


def load_runs(results: Path) -> list[dict[str, Any]]:
    paths = sorted(results.rglob("summary.json"))
    if not paths:
        raise ValueError(f"未发现实际结果：{results} 下没有 summary.json；未生成图或报告。")
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: summary.json 必须是对象")
        for key in ("seed", "condition", "task", "steps", "final_eval"):
            if key not in data:
                raise ValueError(f"{path}: 缺少 {key}")
        if data["condition"] not in ("H0", "H1"):
            raise ValueError(f"{path}: 未知 condition {data['condition']!r}")
        if data["task"] not in TASK_NAMES:
            raise ValueError(f"{path}: 未知 task {data['task']!r}")
        if not isinstance(data["final_eval"], dict):
            raise ValueError(f"{path}: final_eval 必须是对象")
        points = data.get("evaluations", [])
        if not isinstance(points, list):
            raise ValueError(f"{path}: evaluations 必须是列表")
        if any(not isinstance(point, dict) or finite_number(point.get("step")) is None for point in points):
            raise ValueError(f"{path}: 每个检查点必须包含数值 step")
        steps = [point["step"] for point in points]
        if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
            raise ValueError(f"{path}: 检查点 step 必须严格递增，不能静默重排或合并")
        for point in [*points, data["final_eval"]]:
            for key in ("accuracy", "cleared_accuracy", "intervention_accuracy", "intervention_follow_rate", "original_accuracy_on_intervened"):
                value = point.get(key)
                if value is not None and (finite_number(value) is None or not 0 <= value <= 1):
                    raise ValueError(f"{path}: {key} 必须是 [0,1] 内概率或 null")
        run = dict(data)
        run["evaluations"] = points
        run["_path"] = path.resolve()
        run["_name"] = str(path.parent.relative_to(results))
        run["_confirmation"] = performance_confirmation(points)
        runs.append(run)
    return runs


def optional_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def figure_link(label: str, path: Path) -> str:
    return f"![{label}](<{path.resolve()}>)"


def direction_values(final: dict[str, Any], cleared: bool = False) -> list[Any]:
    keys = ("direction_cleared_accuracy",) if cleared else ("direction_accuracy", "accuracy_by_direction", "direction_accuracies")
    for key in keys:
        value = final.get(key)
        if isinstance(value, list) and len(value) == 2:
            return value
    return [None, None]


def count_array(raw: Any, name: str):
    import numpy as np

    if raw is None:
        return None
    array = np.asarray(raw, dtype=float)
    if array.shape != (2, 4, 4) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name}: 计数矩阵必须是非负有限数值 [2][4][4]，收到 {array.shape}")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name}: 计数矩阵不能包含小数计数")
    return array


def row_normalize(array):
    import numpy as np

    totals = array.sum(axis=-1, keepdims=True)
    return np.divide(array, totals, out=np.full_like(array, np.nan), where=totals != 0)


def save_figure(figure, destination: Path) -> None:
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")


def trajectory_plot(runs: list[dict[str, Any]], destination: Path) -> bool:
    import matplotlib.pyplot as plt

    available = [run for run in runs if run["evaluations"]]
    if not available:
        return False
    columns = 2
    rows = math.ceil(len(available) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.15 * rows), squeeze=False)
    for axis, run in zip(axes.flat, available):
        points = run["evaluations"]
        for key, label, color, style in (
            ("accuracy", "Recorded-history evaluation", "#245ca6", "-"),
            ("cleared_accuracy", "Cleared-history evaluation", "#cd7023", "--"),
        ):
            valid = [(p["step"], p[key]) for p in points if finite_number(p.get(key)) is not None]
            if valid:
                axis.plot(*zip(*valid), color=color, linestyle=style, marker="o", markersize=3, label=label)
        axis.axhline(CHANCE, color="0.65", linestyle=":", label="Chance (0.25)")
        axis.axhline(THRESHOLD, color="0.35", linestyle="--", linewidth=0.8, label="Threshold (0.90)")
        if run["_confirmation"] is not None:
            axis.axvline(run["_confirmation"], color="#38834b", alpha=0.7, label="Performance confirmation")
        axis.set(title=f"{run['task']} | {run['condition']} | seed {run['seed']}", xlabel="Training steps", ylabel="Accuracy", ylim=(-0.03, 1.04))
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7, loc="lower right")
    for axis in list(axes.flat)[len(available):]:
        axis.set_visible(False)
    figure.tight_layout()
    save_figure(figure, destination)
    plt.close(figure)
    return True


def endpoint_plot(runs: list[dict[str, Any]], destination: Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np

    if not any(finite_number(run["final_eval"].get("accuracy")) is not None for run in runs):
        return False
    figure, axis = plt.subplots(figsize=(max(9, len(runs) * 0.72), 4.8))
    locations = np.arange(len(runs))
    fields = (("accuracy", "Final", "#245ca6"), ("cleared_accuracy", "History cleared", "#cd7023"))
    for offset, (field, label, color) in zip((-0.16, 0.16), fields):
        positions = [i for i, run in enumerate(runs) if finite_number(run["final_eval"].get(field)) is not None]
        axis.bar([locations[i] + offset for i in positions], [runs[i]["final_eval"][field] for i in positions], width=0.30, color=color, label=label)
    axis.axhline(CHANCE, color="0.4", linestyle=":", label="Chance (0.25)")
    axis.set_xticks(locations, [f"{r['task']}\n{r['condition']} / seed {r['seed']}" for r in runs], rotation=45, ha="right", fontsize=8)
    axis.set(ylabel="Accuracy", ylim=(0, 1.05), title="Every recorded run; no pooled inferential estimate")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    save_figure(figure, destination)
    plt.close(figure)
    return True


def main_condition_plot(runs: list[dict[str, Any]], destination: Path) -> bool:
    """Compact paired main-condition curves, retaining each seed separately."""
    import matplotlib.pyplot as plt

    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run["task"] == "emergent" and run["evaluations"]:
            grouped[str(run["seed"])][run["condition"]].append(run)
    seeds = sorted(grouped, key=lambda seed: (0, int(seed)) if seed.lstrip("-").isdigit() else (1, seed))
    if not seeds:
        return False
    figure, axes = plt.subplots(1, len(seeds), figsize=(max(4, len(seeds) * 4), 3.5), sharey=True, squeeze=False)
    for axis, seed in zip(axes.flat, seeds):
        for condition, color, style in (("H0", "#245ca6", "-"), ("H1", "#cd7023", "--")):
            matches = grouped[seed][condition]
            for index, run in enumerate(matches):
                points = [(point["step"], point["accuracy"]) for point in run["evaluations"] if finite_number(point.get("accuracy")) is not None]
                if points:
                    label = condition if len(matches) == 1 else f"{condition} run {index + 1}"
                    axis.plot(*zip(*points), color=color, linestyle=style, marker="o", markersize=3, linewidth=1.5, label=label)
        axis.axhline(CHANCE, color="0.5", linestyle=":", linewidth=0.9, label="Chance (0.25)")
        axis.axhline(THRESHOLD, color="0.65", linestyle="--", linewidth=0.9, label="Threshold (0.90)")
        axis.set(title=f"Emergent | seed {seed}", xlabel="Training interactions", ylim=(-0.03, 1.04))
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7, loc="lower right")
    axes[0, 0].set_ylabel("Accuracy")
    figure.tight_layout()
    save_figure(figure, destination)
    plt.close(figure)
    return True


def intervention_plot(runs: list[dict[str, Any]], destination: Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np

    if not any(finite_number(run["final_eval"].get("original_accuracy_on_intervened")) is not None for run in runs):
        return False
    figure, axes = plt.subplots(2, 1, figsize=(max(10, len(runs) * 0.80), 8), sharex=True)
    locations = np.arange(len(runs))
    for offset, field, label, color in ((-0.16, "original_accuracy_on_intervened", "Original message, eligible trials", "#245ca6"), (0.16, "intervention_accuracy", "Replaced message, same eligible trials", "#884c9b")):
        indices = [i for i, run in enumerate(runs) if finite_number(run["final_eval"].get(field)) is not None]
        axes[0].bar([locations[i] + offset for i in indices], [runs[i]["final_eval"][field] for i in indices], width=0.30, label=label, color=color)
    indices = [i for i, run in enumerate(runs) if finite_number(run["final_eval"].get("intervention_follow_rate")) is not None]
    axes[1].bar([locations[i] for i in indices], [runs[i]["final_eval"]["intervention_follow_rate"] for i in indices], width=0.50, color="#38834b", label="Follow replaced symbol's reference mapping")
    for i, run in enumerate(runs):
        n = run["final_eval"].get("intervention_n_trials")
        axes[1].text(i, 1.02, f"n={n}" if n is not None else "n unknown", ha="center", va="bottom", fontsize=7)
        if n == 0:
            axes[0].text(i, 0.03, "Not estimable", rotation=90, ha="center", va="bottom", fontsize=7, color="0.35")
    axes[0].set_ylabel("Target accuracy")
    axes[0].set_title("Message replacement at the final snapshot only", pad=34)
    axes[1].set(ylabel="Reference-mapping follow rate")
    axes[1].set_xticks(locations, [f"{r['task']}\n{r['condition']} / seed {r['seed']}" for r in runs], rotation=45, ha="right", fontsize=8)
    for axis in axes:
        axis.set_ylim(0, 1.12)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.set_xlim(-0.6, len(runs) - 0.4)
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2, borderaxespad=0)
    figure.tight_layout()
    save_figure(figure, destination)
    plt.close(figure)
    return True


def plot_count_matrix(axis, counts, title: str, xlabels, ylabels, xlabel: str, ylabel: str):
    normalized = row_normalize(counts)
    picture = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(4), xlabels, rotation=25, ha="right")
    axis.set_yticks(range(4), ylabels)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for row in range(4):
        for column in range(4):
            value = normalized[row, column]
            label = "n=0" if math.isnan(value) else f"{value:.2f}\n({int(counts[row, column])})"
            axis.text(column, row, label, ha="center", va="center", fontsize=8, color="white" if not math.isnan(value) and value > 0.6 else "black")
    return picture


def matrix_plot(run: dict[str, Any], destination: Path) -> bool:
    import matplotlib.pyplot as plt

    final = run["final_eval"]
    messages = count_array(final.get("message_counts"), f"{run['_name']}.message_counts")
    responses = count_array(final.get("response_counts"), f"{run['_name']}.response_counts")
    if messages is None and responses is None:
        return False
    columns = int(messages is not None) * 2 + int(responses is not None)
    figure, axes = plt.subplots(2, columns, figsize=(4.15 * columns, 8.0), squeeze=False)
    symbols = [f"S{i}" for i in range(4)]
    for direction in range(2):
        column = 0
        if messages is not None:
            plot_count_matrix(axes[direction, column], messages[direction], f"{DIRECTIONS[direction]}: production", symbols, CLASSES, "Sent symbol", "True class")
            column += 1
            plot_count_matrix(axes[direction, column], messages[direction].T, f"{DIRECTIONS[direction]}: target association", CLASSES, symbols, "True class", "Sent symbol")
            column += 1
        if responses is not None:
            plot_count_matrix(axes[direction, column], responses[direction], f"{DIRECTIONS[direction]}: receiver choices", CLASSES, symbols, "Selected class", "Received symbol")
    figure.suptitle(f"{run['task']} | {run['condition']} | seed {run['seed']}\nRow proportions; raw counts in parentheses; empty rows are undefined", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, destination)
    plt.close(figure)
    return True


def mapping_trajectory_plot(run: dict[str, Any], destination: Path) -> bool:
    """Show raw symbol identities over time without post-hoc permutations."""
    import matplotlib.pyplot as plt
    import numpy as np

    points = []
    for point in run["evaluations"]:
        counts = count_array(point.get("message_counts"), f"{run['_name']} step {point['step']}.message_counts")
        if counts is not None:
            points.append((point["step"], row_normalize(counts)))
    if not points:
        return False
    figure, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True, squeeze=False)
    colors = ("#245ca6", "#cd7023", "#38834b", "#884c9b")
    for direction in range(2):
        for true_class in range(4):
            axis = axes[direction, true_class]
            for symbol in range(4):
                values = [p[1][direction, true_class, symbol] for p in points]
                if np.isfinite(values).any():
                    axis.plot([p[0] for p in points], values, color=colors[symbol], marker=".", linewidth=1.3, label=f"S{symbol}")
            axis.set(title=f"{DIRECTIONS[direction]} | {CLASSES[true_class]}", xlabel="Training steps", ylim=(-0.04, 1.04))
            axis.grid(alpha=0.15)
            if true_class == 0:
                axis.set_ylabel("P(symbol | true class)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{run['task']} | {run['condition']} | seed {run['seed']} — unchanged symbol identities", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, destination)
    plt.close(figure)
    return True


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def intervention_drop(final: dict[str, Any]) -> float | None:
    original = finite_number(final.get("original_accuracy_on_intervened"))
    changed = finite_number(final.get("intervention_accuracy"))
    return None if original is None or changed is None else original - changed


def intervention_coverage(final: dict[str, Any]) -> float | None:
    eligible = finite_number(final.get("intervention_n_trials"))
    total = finite_number(final.get("n_trials"))
    return None if eligible is None or total is None or total <= 0 else eligible / total


def two_values(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) and len(raw) == 2 else [None, None]


def export_plot_data(runs: list[dict[str, Any]], output: Path) -> None:
    """Export every plotted observation and exact matrix count as tidy CSV."""
    metric_keys = ["accuracy", "cleared_accuracy", "n_trials", "original_accuracy_on_intervened", "intervention_accuracy", "intervention_follow_rate", "intervention_n_trials", "intervention_follow_n_trials", "reference_n", "reference_seed"]
    observations, counts, mappings = [], [], []
    for run in runs:
        checkpoints = [("checkpoint", point["step"], point) for point in run["evaluations"]]
        checkpoints.append(("final", run["steps"], run["final_eval"]))
        for phase, step, point in checkpoints:
            prefix = [run["_name"], run["seed"], run["condition"], run["task"], phase, step]
            observations.append([*prefix, *[point.get(key) for key in metric_keys], *direction_values(point), *direction_values(point, cleared=True), *two_values(point.get("direction_n")), intervention_drop(point), intervention_coverage(point)])
            for key, label in (("message_counts", "production"), ("response_counts", "receiver_choices"), ("reference_message_counts", "reference_production")):
                array = count_array(point.get(key), f"{run['_name']}.{phase}.{key}")
                if array is None:
                    continue
                views = [(label, array)]
                if key != "response_counts":
                    views.append(("reference_target_association" if key.startswith("reference") else "target_association", array.transpose(0, 2, 1)))
                for matrix_name, matrices in views:
                    for direction in range(2):
                        matrix = matrices[direction]
                        probabilities = row_normalize(matrix)
                        is_production = matrix_name.endswith("production")
                        for row in range(4):
                            for column in range(4):
                                probability = probabilities[row, column]
                                counts.append([*prefix, direction, DIRECTIONS[direction], matrix_name, CLASSES[row] if is_production else f"S{row}", f"S{column}" if is_production else CLASSES[column], int(matrix[row, column]), int(matrix[row].sum()), None if math.isnan(probability) else float(probability)])
            reference_maps = two_values(point.get("reference_message_to_class"))
            reference_used = two_values(point.get("reference_used_symbols"))
            for direction in range(2):
                mapping = reference_maps[direction]
                if not isinstance(mapping, list) or len(mapping) != 4:
                    continue
                for symbol, category in enumerate(mapping):
                    mappings.append([*prefix, direction, symbol, category, CLASSES[category] if isinstance(category, int) and 0 <= category < 4 else None, symbol in reference_used[direction] if isinstance(reference_used[direction], list) else None, point.get("reference_n"), point.get("reference_seed")])
    prefix_headers = ["run", "seed", "condition", "task", "phase", "step"]
    write_csv(output / "evaluation_metrics.csv", [*prefix_headers, *metric_keys, "a_to_b_accuracy", "b_to_a_accuracy", "a_to_b_cleared_accuracy", "b_to_a_cleared_accuracy", "a_to_b_n", "b_to_a_n", "original_minus_replaced_accuracy_same_trials", "intervention_coverage"], observations)
    write_csv(output / "communication_matrix_data.csv", [*prefix_headers, "sender_direction", "direction_label", "matrix", "row", "column", "count", "row_total", "row_probability"], counts)
    write_csv(output / "reference_mappings.csv", [*prefix_headers, "sender_direction", "symbol", "mapped_class_id", "mapped_class_name", "used_in_reference", "reference_n", "reference_seed"], mappings)


def reference_description(final: dict[str, Any]) -> str:
    n = finite_number(final.get("reference_n"))
    if n is None or n <= 0:
        return "独立参考批次未记录；不能声称该运行已通过独立参考映射验证。"
    rows = []
    maps = two_values(final.get("reference_message_to_class"))
    used = two_values(final.get("reference_used_symbols"))
    ns = two_values(final.get("reference_direction_n"))
    counts = count_array(final.get("reference_message_counts"), "reference_message_counts")
    for direction in range(2):
        mapping = maps[direction]
        meanings = []
        if isinstance(mapping, list) and len(mapping) == 4:
            for symbol, category in enumerate(mapping):
                meaning = f"S{symbol}→{CLASSES[category]}" if isinstance(category, int) and 0 <= category < 4 else f"S{symbol}→未估计"
                if counts is not None:
                    symbol_n = int(counts[direction, :, symbol].sum())
                    purity = float(counts[direction, :, symbol].max() / symbol_n) if symbol_n else None
                    meaning += f"（n={symbol_n}，多数占比{pct(purity) if symbol_n else '不可估'}）"
                meanings.append(meaning)
        rows.append([DIRECTIONS[direction], ns[direction], used[direction], "；".join(meanings) or "未记录"])
    return f"独立参考批次：n={int(n)}，seed={final.get('reference_seed', '未记录')}。参考批次确定可替换符号及其多数对应对象；测试批次随后比较原始与替换消息，参考观察和奖励不进入策略更新。跟随率仅指跟随这个独立参考集中的多数对应，不是完整词义的测量。部分符号可能只出现极少次，须连同逐符号n和多数占比解读；高占比若仅来自少量样本，不能说明对应稳定。\n\n" + markdown_table(["方向", "参考试次", "参考中使用的符号", "符号→对象多数对应及样本量"], rows)


def matrix_tables(run: dict[str, Any]) -> list[str]:
    """Retain exact counts in the report, including directions and empty rows."""
    sections = []
    final = run["final_eval"]
    sources = (
        ("message_counts", "发送符号→真实目标（观察关联）", True),
        ("response_counts", "接收符号→所选对象（接收行为）", False),
    )
    for key, name, transpose in sources:
        array = count_array(final.get(key), f"{run['_name']}.{key}")
        if array is None:
            sections.append(f"{name}：未记录。")
            continue
        for direction in range(2):
            counts = array[direction].T if transpose else array[direction]
            normalized = row_normalize(counts)
            rows = []
            for symbol in range(4):
                cells = ["—（0）" if math.isnan(normalized[symbol, category]) else f"{normalized[symbol, category]:.2f}（{int(counts[symbol, category])}）" for category in range(4)]
                rows.append([f"S{symbol}", *cells, int(counts[symbol].sum())])
            sections.append(f"**{DIRECTIONS[direction]}：{name}**\n\n" + markdown_table(["符号", *CLASSES, "行样本数"], rows))
    return sections


def observed_summary(runs: list[dict[str, Any]]) -> list[str]:
    paragraphs = ["## 本批主要观察"]
    main = [run for run in runs if run["task"] == "emergent"]
    fixed = [run for run in runs if run["task"] == "fixed"]
    silent = [run for run in runs if run["task"] == "no_comm"]
    if main:
        accuracies = [run["final_eval"]["accuracy"] for run in main if finite_number(run["final_eval"].get("accuracy")) is not None]
        if accuracies:
            confirmed = sum(run["_confirmation"] is not None for run in main)
            paragraphs.append(f"自发协议条件共 {len(main)} 组，终点准确率为 {pct(min(accuracies))}–{pct(max(accuracies))}；其中 {confirmed} 组在预算内达到预定性能确认标准。未确认不表示没有有效通信，含义是没有满足连续三个检查点均达到90%的操作标准。")
        paired = defaultdict(dict)
        for run in main:
            paired[str(run["seed"])][run["condition"]] = run
        differences = []
        for seed, conditions in sorted(paired.items()):
            if len(conditions) == 2:
                a, b = (finite_number(conditions[c]["final_eval"].get("accuracy")) for c in ("H0", "H1"))
                if a is not None and b is not None:
                    differences.append(f"seed {seed}：{(b - a) * 100:+.2f} 个百分点")
        if differences:
            paragraphs.append("H1−H0 的终点配对差值为：" + "；".join(differences) + "。这些差值仅描述本批配对，不能据此认定总体记忆效应。")
        clear_deltas = [abs(run["final_eval"]["accuracy"] - run["final_eval"]["cleared_accuracy"]) for run in main if finite_number(run["final_eval"].get("accuracy")) is not None and finite_number(run["final_eval"].get("cleared_accuracy")) is not None]
        if clear_deltas:
            paragraphs.append(f"主条件终点保留与清空近期历史的准确率差异，绝对值最大为 {max(clear_deltas) * 100:.3f} 个百分点。这只描述终点策略对当前记录的依赖，不能推出训练过程中的历史没有作用。")
        drops = [intervention_drop(run["final_eval"]) for run in main]
        drops = [value for value in drops if value is not None]
        follows = [run["final_eval"]["intervention_follow_rate"] for run in main if finite_number(run["final_eval"].get("intervention_follow_rate")) is not None]
        if drops and follows:
            paragraphs.append(f"终点在同一可干预样本上替换消息后，主条件目标准确率下降 {min(drops) * 100:.2f}–{max(drops) * 100:.2f} 个百分点；跟随独立参考集多数对应的比例为 {pct(min(follows))}–{pct(max(follows))}。这支持消息在终点影响接收选择；多数对应及其样本量需逐符号检查，不能等同完整词义，也不能回溯到性能确认点。")
    for selected, label in ((fixed, "固定信号"), (silent, "无通信")):
        values = [run["final_eval"]["accuracy"] for run in selected if finite_number(run["final_eval"].get("accuracy")) is not None]
        if values:
            confirmation_steps = sorted({run["_confirmation"] for run in selected if run["_confirmation"] is not None})
            confirmation_text = f"；已确认运行的确认点为 {confirmation_steps} steps" if confirmation_steps else "；预算内没有运行达到性能确认标准"
            paragraphs.append(f"{label}条件共 {len(selected)} 组，终点准确率为 {pct(min(values))}–{pct(max(values))}{confirmation_text}。这些对照须结合其独立种子数量和协议审计解释。")
    return paragraphs


def analyze(results: Path, output: Path) -> Path:
    runs = load_runs(results)
    # Validate all count data before creating output artifacts.
    for run in runs:
        for point in [*run["evaluations"], run["final_eval"]]:
            for key in ("message_counts", "response_counts", "reference_message_counts"):
                count_array(point.get(key), f"{run['_name']}.{key}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metadata_paths = [results / name for name in ("batch_config.json", "batch_status.json", "machine.json", "config.json", "benchmark.json")]
    metadata_paths.append(results.parent / "benchmark_confirm" / "benchmark.json")
    metadata = [(path, optional_json(path)) for path in metadata_paths if path.exists()]
    export_plot_data(runs, output)
    report = ["# 首轮先导实验结果", "本报告直接读取实际保存的 summary.json，保留全部运行。独立单位是一个双智能体群体；逐轮试次和检查点不是独立群体重复。少量配对种子只作描述，不据此推断统计显著、记忆的必要性或人类语言起源。", f"输入目录：`{results}`。共读取 {len(runs)} 次运行，{len({str(r['seed']) for r in runs})} 个不同 seed 标识。不同任务即使 seed 相同也不能自动视为等预算对照。", "## 口径与边界", "“性能确认时间”定义为首次连续三个已记录检查点的准确率均不低于 90% 时，第三个检查点的 step。预算内未出现该事件的运行保留为未确认；不将它们删除，也不把训练终点当作形成时间。终点消息干预只说明终点状态，不能回溯证明此前检查点已有因果通信或稳定语义。", "图中的机会水平为四选一任务的 25%。各运行均展示原始结果；没有根据三个 seed 计算总体效应的显著性或把测试样本误当作跨群体重复。图未提供误差带；如日志未记录测试试次数，报告明确保留缺失。", "S0–S3 是固定符号编号，跨检查点不重新排列。符号编号本身没有语义；沉默的具体编号以实验配置为准。发送关联矩阵统计 P(真实对象|发送符号)，接收矩阵统计 P(所选对象|收到符号)。二者均是描述性分布，不能替代控制其他输入的消息干预。零样本行未定义，不显示成零概率。", "## 每次运行的终点结果"]
    report[3:3] = observed_summary(runs)
    headers = ["运行", "seed", "条件", "任务", "steps", "终点", "清空历史", "替换消息后目标准确率", "跟随替换消息", "性能确认时间"]
    rows = []
    csv_rows = []
    for run in runs:
        final = run["final_eval"]
        confirmation = run["_confirmation"]
        rows.append([run["_name"], run["seed"], run["condition"], TASK_NAMES[run["task"]], run["steps"], pct(final.get("accuracy")), pct(final.get("cleared_accuracy")), intervention_pct(final, "intervention_accuracy"), intervention_pct(final, "intervention_follow_rate"), f"{confirmation}" if confirmation is not None else "预算内未确认"])
        csv_rows.append([run["_name"], run["seed"], run["condition"], run["task"], run["steps"], final.get("accuracy"), final.get("cleared_accuracy"), final.get("intervention_accuracy"), final.get("intervention_follow_rate"), confirmation, confirmation is None, run.get("train_seconds"), run.get("params_per_agent"), str(run["_path"]), final.get("original_accuracy_on_intervened"), intervention_drop(final), final.get("intervention_n_trials"), intervention_coverage(final), *direction_values(final), *two_values(final.get("direction_n")), final.get("reference_n")])
    report.append(markdown_table(headers, rows))
    write_csv(output / "run_metrics.csv", ["run", "seed", "condition", "task", "steps", "final_accuracy", "cleared_accuracy", "intervention_accuracy", "intervention_follow_rate", "performance_confirmation_step", "unconfirmed", "train_seconds", "params_per_agent", "source", "original_accuracy_on_intervened", "original_minus_replaced_accuracy_same_trials", "intervention_n_trials", "intervention_coverage", "a_to_b_accuracy", "b_to_a_accuracy", "a_to_b_n", "b_to_a_n", "reference_n"], csv_rows)
    report.append("消息替换后的目标准确率与跟随替换消息比例回答不同问题：前者是对原目标的成功率，后者是转向预先估计的替换符号所指对象的比例。下降本身不充分证明符号意义；跟随率还须结合干预可用样本、映射覆盖率及映射估计数据的独立性审查。若只有部分方向可替换，干预分母只是可干预子集，不能直接将其与全体终点准确率相减解释为同样本效应。无记录或零可干预试次的指标不当作零准确率。")
    if endpoint_plot(runs, figures / "endpoint_comparison"):
        report.append(figure_link("每次运行终点结果", figures / "endpoint_comparison.png"))
    report.append("## 终点消息替换：同一可干预试次的比较")
    intervention_rows = []
    for run in runs:
        final = run["final_eval"]
        drop = intervention_drop(final)
        intervention_rows.append([run["_name"], final.get("n_trials"), final.get("intervention_n_trials"), pct(intervention_coverage(final)), intervention_pct(final, "original_accuracy_on_intervened"), intervention_pct(final, "intervention_accuracy"), "不可估" if drop is None else f"{drop * 100:+.2f} 个百分点", intervention_pct(final, "intervention_follow_rate"), final.get("reference_n")])
    report.append(markdown_table(["运行", "全部测试n", "可干预n", "覆盖率", "原始消息", "替换消息", "原始−替换", "跟随参考多数对应", "独立参考n"], intervention_rows))
    report.append("此处原始和替换准确率均使用相同的可干预试次，故可直接作描述性差值；不把这一差值当作跨群体显著效应。替换资格、替换符号和符号的多数对应对象由独立参考批次确定。未出现至少两个参考使用符号的方向不能做此替换，n=0时表现不可估。跟随率针对参考映射，需结合映射纯度、覆盖率和未干预行为解释，不能单独证明完整语义。")
    if intervention_plot(runs, figures / "message_intervention"):
        report.append(figure_link("同一可干预试次的消息替换结果", figures / "message_intervention.png"))
    report.append("## H0/H1 逐种子配对差值")
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[(run["task"], str(run["seed"]))][run["condition"]].append(run)
    paired_rows = []
    paired_csv = []
    for (task, seed), conditions in sorted(grouped.items()):
        if len(conditions["H0"]) != 1 or len(conditions["H1"]) != 1:
            paired_rows.append([TASK_NAMES[task], seed, "未构成唯一配对", "—", "—"])
            continue
        h0, h1 = conditions["H0"][0], conditions["H1"][0]
        deltas = []
        for key in ("accuracy", "cleared_accuracy"):
            a, b = finite_number(h0["final_eval"].get(key)), finite_number(h1["final_eval"].get(key))
            deltas.append(None if a is None or b is None else b - a)
        time_difference = None if h0["_confirmation"] is None or h1["_confirmation"] is None else h1["_confirmation"] - h0["_confirmation"]
        paired_rows.append([TASK_NAMES[task], seed, "未记录" if deltas[0] is None else f"{100 * deltas[0]:+.2f} 个百分点", "未记录" if deltas[1] is None else f"{100 * deltas[1]:+.2f} 个百分点", "至少一方未确认" if time_difference is None else f"{time_difference:+g} steps"])
        paired_csv.append([task, seed, *deltas, time_difference, h0["steps"], h1["steps"]])
    report.append(markdown_table(["任务", "seed", "H1−H0 终点准确率", "H1−H0 清空历史准确率", "H1−H0 确认时间"], paired_rows))
    report.append("差值方向只描述本批运行。确认时间差为负表示 H1 更早；仅当双方均达到确认标准时计算，不用成功运行的均值代替未确认群体。是否为真正匹配的配对，还需核验初始权重、外生环境序列和训练预算。")
    write_csv(output / "paired_differences.csv", ["task", "seed", "h1_minus_h0_accuracy", "h1_minus_h0_cleared_accuracy", "h1_minus_h0_confirmation_step", "h0_steps", "h1_steps"], paired_csv)
    report.append("## 性能轨迹")
    if main_condition_plot(runs, figures / "main_condition_paired_curves"):
        report.append(figure_link("主条件逐种子配对学习曲线", figures / "main_condition_paired_curves.png"))
    if trajectory_plot(runs, figures / "learning_curves"):
        report.append(figure_link("逐运行检查点轨迹", figures / "learning_curves.png"))
    else:
        report.append("未记录检查点轨迹，不能估计性能确认时间。")
    report.append("## 逐运行通信映射与方向结果")
    for index, run in enumerate(runs, 1):
        final = run["final_eval"]
        stem = f"{index:02d}_" + re.sub(r"[^A-Za-z0-9_-]+", "_", f"{run['task']}_{run['condition']}_seed{run['seed']}")
        report.append(f"### {run['_name']}（{run['condition']}，seed {run['seed']}，{TASK_NAMES[run['task']]}）")
        direction_acc = direction_values(final)
        direction_clear = direction_values(final, cleared=True)
        direction_ns = two_values(final.get("direction_n"))
        report.append(markdown_table(["方向", "测试n", "终点准确率", "清空历史准确率"], [[DIRECTIONS[d], direction_ns[d], pct(direction_acc[d]), pct(direction_clear[d])] for d in range(2)]))
        report.append(f"视觉练习准确率：{table_cell(run.get('pretrain_accuracy'))}。每个 agent 参数量：{table_cell(run.get('params_per_agent'))}；训练用时：{table_cell(run.get('train_seconds'))} 秒；两个方向实际使用符号数：{table_cell(final.get('used_symbols'))}。")
        sample_fields = {key: final[key] for key in ("n_trials", "n_eval", "direction_n", "direction_trials", "intervention_n_trials", "intervention_follow_n_trials", "intervention_eligible_trials", "intervention_coverage", "intervention_direction_accuracy", "intervention_direction_follow_rate") if key in final}
        report.append("测试及干预分母/覆盖率记录：" + (table_cell(sample_fields) if sample_fields else "未记录；不推算置信区间或覆盖率。"))
        report.append(reference_description(final))
        if matrix_plot(run, figures / f"{stem}_matrices"):
            report.append(figure_link("方向性通信矩阵", figures / f"{stem}_matrices.png"))
        report.extend(matrix_tables(run))
        if mapping_trajectory_plot(run, figures / f"{stem}_symbol_trajectories"):
            report.append(figure_link("固定编号符号使用轨迹", figures / f"{stem}_symbol_trajectories.png"))
            report.append("该轨迹用于观察对象类别与符号的对应及漂移。优势符号改变本身不证明意义类别发生拆分/合并，也不代表组合结构形成。")
    report.append("## 资源与来源")
    report.append("各运行的实际 steps、训练秒数及参数量已保存在 run_metrics.csv。以下转录批次目录及已保存的基准测试目录中的资源与配置记录；没有文件时不估计硬件、时长或成本。")
    if not metadata:
        report.append("结果目录及指定基准目录未找到批次元数据文件。")
    for path, payload in metadata:
        report.append(f"### {path.name}\n\n来源：`{path.resolve()}`。")
        report.append("```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```")
    report.append("## 本轮证据能够支持的范围")
    report.append("本轮可检验当前视觉指称任务下，双方是否获得有效通信，以及保留近期私人交互记录与清空记录的运行表现。形成过程分析以检查点轨迹和固定编号映射为依据；对符号作用的解释须结合终点干预。即使成功，也不能据此认定自然语言、组合语法或人类语言的历史起源已被模拟。")
    report.append("若无通信准确率持续明显高于四选一机会水平，应先复核信息泄露、采样依赖和测试流程；若某方向失败，不能用平均值遮蔽。固定信号对照衡量关联学习能力，其解释受训练预算和信号来源限制。H0/H1 差异也可能反映优化、视觉学习或历史编码负担，需结合协议审计后再归因。")
    report.append("图仅输出 PNG。run_metrics.csv 保存每次运行汇总；paired_differences.csv 保存配对差值；evaluation_metrics.csv 保存每个检查点和终点的全部绘图指标与方向分母；communication_matrix_data.csv 保存各方向矩阵的原始计数、行分母和归一化概率；reference_mappings.csv 保存独立参考批次的符号映射。未对符号做事后对齐，未筛选成功种子。原始 summary.json 保持不变。")
    report.append("输入来源：\n\n" + markdown_table(["运行", "原始结果文件"], [[run["_name"], str(run["_path"])] for run in runs]))
    report_path = output / "首轮实验结果报告.md"
    report_path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="用实际保存的首轮实验结果生成科学图、逐seed表和中文报告。")
    parser.add_argument("--results", type=Path, required=True, help="递归读取各 run/summary.json；根目录可有 config.json、benchmark.json")
    parser.add_argument("--output", type=Path, required=True, help="报告、CSV和图的输出目录")
    args = parser.parse_args()
    try:
        report_path = analyze(args.results.resolve(), args.output.resolve())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(f"已生成：{report_path}")


if __name__ == "__main__":
    main()
