"""Describe a language pilot run without model dependencies or semantic inference.

Usage: python -m qwen_language_v3.analyze PATH_TO_RUN
The independent experimental unit is a group, not an episode or message.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any


ALPHABET = "@#%&*+=~"
MESSAGE_LIMIT = 32
STEP_LIMIT = 64
MISSING = "(missing)"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values), "mean": _mean(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _window_number(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"(?:window[_ -]?|w)?(\d+)", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def _episode_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [value for row in episodes if (value := _number(row.get("score"))) is not None]
    steps = [value for row in episodes if (value := _number(row.get("steps"))) is not None]
    successes = [int(row["success"]) for row in episodes
                 if isinstance(row.get("success"), bool)
                 or (_number(row.get("success")) in (0.0, 1.0))]
    success_steps = [value for row in episodes
                     if row.get("success") in (True, 1)
                     and (value := _number(row.get("steps"))) is not None]
    return {
        "episode_rows": len(episodes), "score": _summary(scores),
        "steps_all_episodes": _summary(steps),
        "success_recorded": len(successes), "success_count": sum(successes),
        "success_rate": _mean(successes),
        "steps_successful_episodes_only": _summary(success_steps),
    }


def _message_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    natural = bool(episodes) and all(row.get("condition") in ("natural", "full_information", "natural_language")
                                     for row in episodes)
    texts: list[str] = []
    window_transitions: Counter[str] = Counter()
    edit_distances: list[float] = []
    length_changes: list[float] = []
    nonempty_pair_equal = 0
    nonempty_pairs = 0
    repeated_window_keys = 0
    unusable_adjacency_rows = 0
    step_totals: list[float] = []
    step_complete_totals: list[float] = []
    for episode in episodes:
        by_step: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        totals: Counter[tuple[str, str]] = Counter()
        for row in episode.get("_valid_messages", []):
            text = row["text"]
            texts.append(text)
            agent, step = row.get("agent", row.get("sender")), row.get("step")
            window = _window_number(row.get("window"))
            if agent is None or step is None:
                unusable_adjacency_rows += 1
                continue
            key = (str(agent), str(step))
            totals[key] += len(text)
            if window is None:
                unusable_adjacency_rows += 1
                continue
            by_step[key].append((window, text))
        step_totals.extend(totals.values())
        for key, windows in by_step.items():
            counts = Counter(window for window, _ in windows)
            repeated_window_keys += sum(count - 1 for count in counts.values() if count > 1)
            # Ambiguous duplicate windows cannot establish a temporal pair.
            unique_windows = sorted((w, text) for w, text in windows if counts[w] == 1)
            if len(unique_windows) == 4 and all(
                unique_windows[i + 1][0] - unique_windows[i][0] == 1 for i in range(3)
            ):
                step_complete_totals.append(totals[key])
            for (window_a, text_a), (window_b, text_b) in zip(unique_windows, unique_windows[1:]):
                if window_b != window_a + 1:
                    continue
                if not text_a and not text_b:
                    transition = "both_empty"
                elif not text_a:
                    transition = "empty_to_nonempty"
                elif not text_b:
                    transition = "nonempty_to_empty"
                elif text_a == text_b:
                    transition = "same_nonempty"
                else:
                    transition = "changed_nonempty"
                window_transitions[transition] += 1
                if text_a and text_b:
                    nonempty_pairs += 1
                    nonempty_pair_equal += int(text_a == text_b)
                edit_distances.append(float(_edit_distance(text_a, text_b)))
                length_changes.append(float(len(text_b) - len(text_a)))
    nonempty = [text for text in texts if text]
    frequencies = Counter(nonempty)
    chars = Counter("".join(texts))
    length_histogram = Counter(len(text) for text in texts)
    top = frequencies.most_common(20)
    return {
        "channel_type": "natural_language_control" if natural else "symbolic",
        "message_rows": len(texts), "nonempty_count": len(nonempty),
        "nonempty_ratio": _ratio(len(nonempty), len(texts)),
        "length_all": _summary([float(len(text)) for text in texts]),
        "length_nonempty": _summary([float(len(text)) for text in nonempty]),
        "length_histogram": {str(k): value for k, value in sorted(length_histogram.items())},
        "at_32_count": length_histogram[MESSAGE_LIMIT],
        "at_32_ratio_all": _ratio(length_histogram[MESSAGE_LIMIT], len(texts)),
        "at_32_ratio_nonempty": _ratio(length_histogram[MESSAGE_LIMIT], len(nonempty)),
        "over_32_count": None if natural else sum(len(text) > MESSAGE_LIMIT for text in texts),
        "total_characters": sum(chars.values()),
        "alphabet_counts": {char: chars[char] for char in ALPHABET},
        "alphabet_frequencies_among_legal_characters": {
            char: _ratio(chars[char], sum(chars[symbol] for symbol in ALPHABET)) for char in ALPHABET
        },
        "invalid_character_counts": None if natural else {char: count for char, count in sorted(chars.items()) if char not in ALPHABET},
        "unique_nonempty_strings": len(frequencies),
        "repeated_nonempty_occurrences": len(nonempty) - len(frequencies),
        "repeated_nonempty_occurrence_ratio": _ratio(len(nonempty) - len(frequencies), len(nonempty)),
        "top_nonempty_strings": [{"text": text, "count": count} for text, count in top],
        "agent_step_character_totals_observed": _summary(step_totals),
        "agent_steps_over_64": None if natural else sum(value > STEP_LIMIT for value in step_totals),
        "complete_four_window_agent_steps": len(step_complete_totals),
        "complete_four_window_agent_step_character_totals": _summary(step_complete_totals),
        "adjacent_windows": {
            "pair_count": sum(window_transitions.values()),
            "transition_counts": {key: window_transitions[key] for key in (
                "both_empty", "empty_to_nonempty", "nonempty_to_empty", "same_nonempty", "changed_nonempty"
            )},
            "both_nonempty_pair_count": nonempty_pairs,
            "same_string_ratio_when_both_nonempty": _ratio(nonempty_pair_equal, nonempty_pairs),
            "character_edit_distance": _summary(edit_distances),
            "signed_length_change": _summary(length_changes),
            "duplicate_window_rows_excluded_from_pairs": repeated_window_keys,
            "rows_without_usable_agent_step_window": unusable_adjacency_rows,
        },
    }


def _read_json(path: Path, warnings: list[str]) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        warnings.append(f"{path.name} 无法读取：{error}")
        return None


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    """Return descriptive data. Missing data are null, never failed trials."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    warnings: list[str] = []
    metadata = {name: _read_json(run_dir / f"{name}.json", warnings)
                for name in ("manifest", "status", "backend_stats")}
    episodes: list[dict[str, Any]] = []
    malformed_lines = 0
    invalid_message_rows = 0
    duplicate_keys: Counter[tuple[str, ...]] = Counter()
    source = run_dir / "episodes.jsonl"
    if source.exists():
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("episode不是对象")
            except (ValueError, json.JSONDecodeError) as error:
                malformed_lines += 1
                warnings.append(f"episodes.jsonl 第{line_number}行未纳入：{error}")
                continue
            for field in ("condition", "group", "phase", "episode", "seed"):
                if row.get(field) is None:
                    warnings.append(f"第{line_number}行缺少{field}；保留记录，分组字段缺失时标为{MISSING}。")
            for field in ("condition", "group", "phase"):
                if row.get(field) is None:
                    row[field] = MISSING
            for field in ("score", "steps"):
                if _number(row.get(field)) is None:
                    warnings.append(f"第{line_number}行{field}没有有限数值；不纳入该指标分母。")
            success = row.get("success")
            if not isinstance(success, bool) and _number(success) not in (0.0, 1.0):
                warnings.append(f"第{line_number}行success不是布尔值或0/1；不纳入成功率分母。")
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list):
                warnings.append(f"第{line_number}行messages缺失或不是列表；不解释为主体全部沉默。")
                raw_messages = []
            valid_messages = []
            for message in raw_messages:
                if isinstance(message, dict) and isinstance(message.get("text"), str):
                    valid_messages.append(message)
                else:
                    invalid_message_rows += 1
            row["_valid_messages"] = valid_messages
            row["_line"] = line_number
            episodes.append(row)
            duplicate_keys[tuple(str(row.get(field, MISSING)) for field in
                                 ("condition", "group", "phase", "episode"))] += 1
    else:
        warnings.append("episodes.jsonl不存在；尚无可分析的任务段记录。")
    duplicates = [{"key": list(key), "rows": count} for key, count in duplicate_keys.items() if count > 1]
    if duplicates:
        warnings.append("存在相同condition/group/phase/episode的重复记录；未擅自去重，汇总可能被重复写入影响。")
    if invalid_message_rows:
        warnings.append(f"跳过{invalid_message_rows}条非对象或text不是字符串的消息记录。")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(str(row.get("condition", MISSING)), str(row.get("group", MISSING)))].append(row)
    groups = []
    for (condition, group), rows in sorted(grouped.items()):
        phases = sorted(set(str(row.get("phase", MISSING)) for row in rows))
        groups.append({
            "condition": condition, "group": group,
            "seeds": sorted(set(str(row.get("seed", MISSING)) for row in rows)),
            "overall": _episode_stats(rows), "messages": _message_stats(rows),
            "phases": {phase: {
                "episodes": _episode_stats([row for row in rows if str(row.get("phase", MISSING)) == phase]),
                "messages": _message_stats([row for row in rows if str(row.get("phase", MISSING)) == phase]),
            } for phase in phases},
            "episode_results": [{key: row.get(key) for key in
                                 ("condition", "group", "phase", "episode", "seed", "score", "steps", "success")}
                                for row in rows],
        })
    conditions = []
    for condition in sorted(set(group["condition"] for group in groups)):
        selected = [group for group in groups if group["condition"] == condition]
        phases = sorted(set(phase for group in selected for phase in group["phases"]))
        phase_stats = {}
        for phase in phases:
            parts = [group["phases"][phase]["episodes"] for group in selected if phase in group["phases"]]
            phase_stats[phase] = {
                "groups_observed": len(parts),
                "group_mean_score": _summary([p["score"]["mean"] for p in parts if p["score"]["mean"] is not None]),
                "group_success_rate": _summary([p["success_rate"] for p in parts if p["success_rate"] is not None]),
                "episode_rows": sum(p["episode_rows"] for p in parts),
            }
        conditions.append({
            "condition": condition, "groups_observed": len(selected),
            "group_ids": [group["group"] for group in selected], "phases": phase_stats,
            "messages_pooled_descriptive_only": _message_stats(
                [row for row in episodes if str(row.get("condition", MISSING)) == condition]),
        })
    return {
        "schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir), "metadata": metadata,
        "data_quality": {"episode_rows": len(episodes), "malformed_jsonl_lines": malformed_lines,
                         "invalid_message_rows": invalid_message_rows, "duplicate_episode_keys": duplicates,
                         "warnings": warnings},
        "groups": groups, "conditions": conditions,
        "interpretation_limits": [
            "独立实验单位是群体，任务段和消息不作为独立重复；本报告不做显著性检验。",
            "条件成绩按阶段分别汇总各群体均值，群体等权；消息汇总仅描述已记录字符串。",
            "成功、重复、变长或相邻窗口改变不能证明共同意义、澄清、组合性或语法。",
            "相邻窗口仅比较同一任务段、主体、物理动作步内编号连续的窗口；不比较跨段相邻文本。",
            "尚未出现、丢失或损坏的任务段不算失败，缺失消息不算沉默；运行中报告可能不完整。",
            "同时报告所有任务段步数与成功段步数；不据不同完成度的平均步数直接判断效率。",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_report(analysis: dict[str, Any]) -> str:
    """Render Markdown, retaining every group's episode results."""
    lines = ["# 语言形成先导运行：描述性分析", "",
             f"运行目录：`{analysis['run_directory']}`", "",
             f"分析时间：{analysis['generated_at_utc']}", "",
             "本报告描述任务完成和字符串使用，不判定语言已经形成。独立实验单位为群体；没有显著性检验。", "",
             "## 运行状态与数据完整性", ""]
    for name, label in (("manifest", "运行配置"), ("status", "运行状态"), ("backend_stats", "推理统计")):
        value = analysis["metadata"][name]
        lines.extend([f"{label}（{name}.json）：", ""])
        if value is None:
            lines.extend(["未提供或无法读取。", ""])
        else:
            lines.extend(["```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""])
    quality = analysis["data_quality"]
    lines.extend([f"已纳入{quality['episode_rows']}条任务段记录、{len(analysis['groups'])}个条件内群体。", ""])
    if quality["warnings"]:
        lines.extend([f"- {_cell(warning)}" for warning in quality["warnings"]])
        lines.append("")
    lines.extend(["缺失值用“—”显示。未完成运行不能据此判断两个条件的最终差异。", "",
                  "## 条件比较", "",
                  "各阶段分别比较；下列成绩先在群体内求均值，再对有记录的群体等权汇总。最小—最大是群体取值范围，不是置信区间。", "",
                  "| 条件 | 阶段 | 有记录群体 | 任务段 | 群体平均score的均值（范围） | 群体成功率的均值（范围） |",
                  "|---|---|---:|---:|---|---|"])
    for condition in analysis["conditions"]:
        for phase, value in condition["phases"].items():
            score, success = value["group_mean_score"], value["group_success_rate"]
            lines.append(f"| {_cell(condition['condition'])} | {_cell(phase)} | {value['groups_observed']} | {value['episode_rows']} | "
                         f"{_fmt(score['mean'])}（{_fmt(score['min'])}—{_fmt(score['max'])}；有效n={score['n']}） | "
                         f"{_pct(success['mean'])}（{_pct(success['min'])}—{_pct(success['max'])}；有效n={success['n']}） |")
    lines.extend(["", "score沿用环境记录的尺度；本分析器不将其重新定义为语言能力分数。", "",
                  "## 各群体结果", ""])
    for group in analysis["groups"]:
        lines.extend([f"### {_cell(group['condition'])} / {_cell(group['group'])}", "",
                      f"记录中的seed：{', '.join(group['seeds'])}", "",
                      "| 阶段 | 任务段 | 平均score | 成功/有效记录 | 所有段平均步数 | 仅成功段平均步数 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for phase, part in group["phases"].items():
            value = part["episodes"]
            lines.append(f"| {_cell(phase)} | {value['episode_rows']} | {_fmt(value['score']['mean'])} | "
                         f"{value['success_count']}/{value['success_recorded']} | {_fmt(value['steps_all_episodes']['mean'])} | "
                         f"{_fmt(value['steps_successful_episodes_only']['mean'])} |")
        lines.extend(["", "逐任务段记录：", "",
                      "| 阶段 | 任务段 | seed | score | 步数 | success |", "|---|---|---|---|---|---|"])
        for row in group["episode_results"]:
            lines.append("| " + " | ".join(_cell(row.get(key)) if row.get(key) is not None else "—"
                                              for key in ("phase", "episode", "seed", "score", "steps", "success")) + " |")
        messages = group["messages"]
        lines.extend(["", f"消息：{messages['message_rows']}条，非空{messages['nonempty_count']}条（{_pct(messages['nonempty_ratio'])}）；"
                      f"平均长度{_fmt(messages['length_all']['mean'])}，非空消息平均长度{_fmt(messages['length_nonempty']['mean'])}。", ""])
        if messages["channel_type"] == "symbolic":
            lines.extend([f"恰好32字符：{messages['at_32_count']}条；超过32字符：{messages['over_32_count']}条；"
                          f"每主体每步超过64字符：{messages['agent_steps_over_64']}次。", ""])
        else:
            lines.extend(["自然语言能力控制不适用8字符字母表、32字符单条上限和64字符每步额度；字符长度不等于模型token数。", ""])
    lines.extend(["## 字符串使用与相邻窗口变化", "",
                  "以下按条件合并消息，只描述所记录文本。更长的任务段会贡献更多消息，不能把消息条数当作独立样本量。", ""])
    for condition in analysis["conditions"]:
        value = condition["messages_pooled_descriptive_only"]
        adjacent = value["adjacent_windows"]
        lines.extend([f"### {_cell(condition['condition'])}", "",
                      f"共{value['message_rows']}条消息，非空比例{_pct(value['nonempty_ratio'])}。所有消息平均长度"
                      f"{_fmt(value['length_all']['mean'])}，非空平均长度{_fmt(value['length_nonempty']['mean'])}。", "",
                      "长度分布（字符数: 条数）：" + ", ".join(f"{length}: {count}" for length, count in value["length_histogram"].items()), ""])
        if value["channel_type"] == "symbolic":
            lines.extend([f"恰好32字符{value['at_32_count']}条，占全部消息{_pct(value['at_32_ratio_all'])}、"
                          f"非空消息{_pct(value['at_32_ratio_nonempty'])}。触顶不等于发生了截断；须结合后端原始记录。", "",
                          "| 字符 | 次数 | 占合法字符比例 |", "|---|---:|---:|"])
            for char in ALPHABET:
                lines.append(f"| `{char}` | {value['alphabet_counts'][char]} | {_pct(value['alphabet_frequencies_among_legal_characters'][char])} |")
            lines.extend(["", f"非法字符计数：`{json.dumps(value['invalid_character_counts'], ensure_ascii=False)}`。", ""])
        else:
            lines.extend(["自然语言能力控制不适用符号字母表与符号预算审计；生成token数另见推理日志。", ""])
        lines.extend([
                      f"非空完整字符串共有{value['unique_nonempty_strings']}种，首次出现以后的重复记录"
                      f"{value['repeated_nonempty_occurrences']}条（占非空消息{_pct(value['repeated_nonempty_occurrence_ratio'])}）。"
                      "该统计把同一主体和不同主体的重复都计入，不证明共享意义。", "",
                      "出现最多的非空完整字符串（最多20种）：", "",
                      "| 字符串 | 次数 |", "|---|---:|"])
        for row in value["top_nonempty_strings"]:
            lines.append(f"| `{_cell(row['text'])}` | {row['count']} |")
        lines.extend(["", f"同一任务段、主体、动作步内，编号连续的窗口对共{adjacent['pair_count']}对。", "",
                      "| 相邻窗口文本变化 | 对数 |", "|---|---:|"])
        labels = {"both_empty": "两条均为空", "empty_to_nonempty": "空→非空",
                  "nonempty_to_empty": "非空→空", "same_nonempty": "非空且完全相同",
                  "changed_nonempty": "两条非空且不同"}
        for key, label in labels.items():
            lines.append(f"| {label} | {adjacent['transition_counts'][key]} |")
        lines.extend(["", f"两条均非空时完全相同比例：{_pct(adjacent['same_string_ratio_when_both_nonempty'])}；"
                      f"所有窗口对平均字符编辑距离：{_fmt(adjacent['character_edit_distance']['mean'])}；"
                      f"平均长度变化（后减前）：{_fmt(adjacent['signed_length_change']['mean'])}。",
                      "这些变化不能自动标注为回应、修复或新的语法结构；它们也可能来自预算耗尽、任务状态或重复策略。",
                      f"缺少可用主体/步/窗口信息的消息{adjacent['rows_without_usable_agent_step_window']}条；"
                      f"重复窗口键的额外消息{adjacent['duplicate_window_rows_excluded_from_pairs']}条，相关窗口不作配对。", ""])
    lines.extend(["## 解释边界", ""])
    lines.extend(f"- {limit}" for limit in analysis["interpretation_limits"])
    lines.extend(["", "逐阶段、逐群体的完整消息统计和读取的运行元数据见analysis.json。", ""])
    return "\n".join(lines)


def write_report(run_dir: str | Path) -> dict[str, Any]:
    analysis = analyze_run(run_dir)
    path = Path(run_dir)
    for filename, content in (
        ("analysis.json", json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
        ("report.md", render_report(analysis)),
    ):
        temporary = path / f".{filename}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path / filename)
    return analysis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args(argv)
    analysis = write_report(arguments.run_dir)
    print(json.dumps({"run_directory": analysis["run_directory"],
                      "episodes": analysis["data_quality"]["episode_rows"],
                      "groups": len(analysis["groups"]),
                      "warnings": len(analysis["data_quality"]["warnings"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
