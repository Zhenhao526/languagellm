"""Read-only aggregation of v3 runs into a Chinese Markdown/JSON report.

Usage: python3 -m qwen_language_v3.report_suite RUN_DIR ... --out REPORT_DIR
Only the selected output files are written. No model or third-party dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SYMBOL_CONDITIONS = {"immediate", "delayed"}
CALIBRATION_PHASE = "calibration"
AGENTS = ("A", "B", "C")
CORE_SOURCE_FILES = ("environment.py", "agents.py", "backend.py", "protocol.py", "run.py")


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _json(path, warnings):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        warnings.append(f"{path.name}未读取：{error}")
        return None


def _jsonl(path, warnings, *, fields=None):
    rows = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("记录不是对象")
                except ValueError as error:
                    warnings.append(f"{path.name}第{index}行未纳入：{error}")
                    continue
                rows.append(row if fields is None else {key: row.get(key) for key in fields})
    except (OSError, UnicodeError) as error:
        warnings.append(f"{path.name}读取中断：{error}")
    return rows


def _key(row, default_phase):
    return tuple(str(row.get(field) if row.get(field) is not None else default_phase if field == "phase" else "未知")
                 for field in ("phase", "condition", "group", "episode"))


def _goals_delivered(state):
    if not isinstance(state, dict) or not isinstance(state.get("goals"), list):
        return None
    values = [_number(goal.get("delivered")) for goal in state["goals"] if isinstance(goal, dict)]
    return sum(values) if len(values) == len(state["goals"]) and all(value is not None for value in values) else None


def _holding_in_state(state):
    if not isinstance(state, dict) or not isinstance(state.get("inventory"), dict):
        return None
    return any(item is not None for inv in state["inventory"].values() if isinstance(inv, dict) for item in inv.values())


def _source_version(path, manifest):
    snapshot = path / "code_snapshot"
    hashes = {}
    for name in CORE_SOURCE_FILES:
        source = snapshot / name
        if source.is_file():
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() if hashes else None
    return {
        "snapshot_directory": str(snapshot), "core_file_sha256": hashes,
        "core_snapshot_sha256": digest,
        "missing_core_files": [name for name in CORE_SOURCE_FILES if name not in hashes],
        "model_path_recorded": manifest.get("model"),
        "model_commit_recorded": manifest.get("model_commit"),
    }


def _action_summary(steps):
    kinds, succeeded = Counter(), Counter()
    failures, feedback_missing, anomalous_wait_failures = 0, 0, 0
    holding_values, delivered_deltas = [], []
    for row in steps:
        feedback = row.get("feedback") if isinstance(row.get("feedback"), dict) else {}
        actions = row.get("actions") if isinstance(row.get("actions"), dict) else {}
        for agent, action in actions.items():
            kind = action.get("kind", "未知") if isinstance(action, dict) else "未知"
            kinds[str(kind)] += 1
            result = feedback.get(agent)
            ok = result.get("action_succeeded") if isinstance(result, dict) else None
            if ok is True:
                succeeded[str(kind)] += 1
            elif ok is False:
                if kind == "wait":
                    anomalous_wait_failures += 1
                else:
                    failures += 1
            else:
                feedback_missing += 1
        for name in ("state_before", "state_after"):
            holding = _holding_in_state(row.get(name))
            if holding is not None:
                holding_values.append(holding)
        observations = row.get("observations") if isinstance(row.get("observations"), dict) else {}
        for observation in observations.values():
            if isinstance(observation, dict) and isinstance(observation.get("carried_items"), list):
                holding_values.append(bool(observation["carried_items"]))
        before, after = (_goals_delivered(row.get(name)) for name in ("state_before", "state_after"))
        if before is not None and after is not None:
            delivered_deltas.append(after - before)
    return {
        "action_kind_counts": dict(sorted(kinds.items())),
        "successful_action_kind_counts": dict(sorted(succeeded.items())),
        "actions_recorded": sum(kinds.values()),
        "wait_actions": kinds["wait"],
        "failed_nonwait_actions": failures,
        "actions_missing_boolean_feedback": feedback_missing,
        "wait_actions_with_false_feedback": anomalous_wait_failures,
        "holding_observed": any(holding_values) if holding_values else None,
        "successful_pickups": succeeded["pickup"],
        "pickup_attempts": kinds["pickup"],
        "successful_processing_actions": succeeded["cut"],
        "processing_attempts": kinds["cut"],
        "accepted_delivery_actor_actions": succeeded["deliver"] + succeeded["deliver_together"],
        "delivery_actor_attempts": kinds["deliver"] + kinds["deliver_together"],
        "newly_delivered_units_from_states": sum(delivered_deltas) if delivered_deltas else None,
        "processing_observed": True if succeeded["cut"] else False if steps and not feedback_missing else None,
        "accepted_delivery_observed": True if succeeded["deliver"] + succeeded["deliver_together"] else False if steps and not feedback_missing else None,
    }


def _message_reference(analysis, phase, condition, group):
    if condition not in SYMBOL_CONDITIONS or not isinstance(analysis, dict):
        return None
    for row in analysis.get("groups", []):
        if str(row.get("condition")) != condition or str(row.get("group")) != group:
            continue
        messages = row.get("phases", {}).get(phase, {}).get("messages")
        if not isinstance(messages, dict):
            return None
        return {key: messages.get(key) for key in (
            "message_rows", "nonempty_count", "nonempty_ratio", "length_all", "length_nonempty",
            "at_32_count", "over_32_count", "agent_steps_over_64", "invalid_character_counts",
            "complete_four_window_agent_steps", "complete_four_window_agent_step_character_totals",
        )}
    return None


def summarize_run(run_dir):
    path = Path(run_dir).resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    warnings = []
    manifest = _json(path / "manifest.json", warnings) or {}
    status = _json(path / "status.json", warnings) or {}
    backend_stats = _json(path / "backend_stats.json", warnings)
    power = _json(path / "power_pause.json", warnings)
    analysis = _json(path / "analysis.json", warnings)
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise ValueError(f"{path}: manifest/status应为对象")
    if not isinstance(backend_stats, dict):
        backend_stats = status.get("backend") if isinstance(status.get("backend"), dict) else {}
    phase = manifest.get("phase", "未知")
    episodes = _jsonl(path / "episodes.jsonl", warnings, fields=(
        "phase", "condition", "group", "episode", "seed", "variant", "score", "steps", "success", "seconds"))
    steps = _jsonl(path / "steps.jsonl", warnings)
    calls = _jsonl(path / "inference.jsonl", warnings, fields=(
        "call", "label", "mode", "seconds", "prompt_tokens", "generation_tokens", "peak_memory_gb", "valid"))
    case_calls = Counter(_key(row["label"], phase) for row in calls if isinstance(row.get("label"), dict))

    cases = {}
    for condition in manifest.get("conditions", []):
        for group in manifest.get("groups", []):
            for scenario in manifest.get("scenarios", []):
                row = {"phase": phase, "condition": condition, "group": group, **scenario}
                cases[_key(row, phase)] = {"planned": row, "completed": None, "steps": []}
    for row in episodes:
        key = _key(row, phase)
        case = cases.setdefault(key, {"planned": None, "completed": None, "steps": []})
        if case["completed"] is not None:
            warnings.append(f"任务段{key}有重复完成记录；表格显示最后一条，不合并成额外重复。")
        case["completed"] = row
    for row in steps:
        key = _key(row, phase)
        cases.setdefault(key, {"planned": None, "completed": None, "steps": []})["steps"].append(row)

    rows = []
    for key, case in sorted(cases.items()):
        current_phase, condition, group, episode = key
        complete = case["completed"]
        recorded_steps = case["steps"]
        numbers = [row["step"] for row in recorded_steps if _number(row.get("step")) is not None]
        latest = max(recorded_steps, key=lambda row: _number(row.get("step")) or 0) if recorded_steps else None
        planned = case["planned"] or {}
        row = {
            "phase": current_phase, "condition": condition, "group": group, "episode": episode,
            "seed": (complete or planned).get("seed"), "variant": (complete or planned).get("variant"),
            "completion_recorded": complete is not None,
            "final_score": _number(complete.get("score")) if complete is not None else None,
            "final_steps": _number(complete.get("steps")) if complete is not None else None,
            "success": complete.get("success") if complete is not None and isinstance(complete.get("success"), bool) else None,
            "steps_recorded": len(recorded_steps), "last_step_recorded": max(numbers) if numbers else None,
            "complete_inference_records": case_calls[key],
            "last_progress_score_not_final": _number(latest.get("score")) if latest else None,
            "recorded_episode_seconds": _number(complete.get("seconds")) if complete else None,
            "actions": _action_summary(recorded_steps),
        }
        if complete is not None:
            row["record_status"] = "已有任务段完成记录"
        elif not recorded_steps and not case_calls[key]:
            row["record_status"] = "尚无完整调用或动作记录"
        elif str(status.get("status", "")).startswith("stopped"):
            row["record_status"] = "已中止，任务段未完成"
        elif status.get("status") in ("failed", "interrupted", "cancelled"):
            row["record_status"] = "运行终止，任务段未完成"
        elif status.get("status") == "completed":
            row["record_status"] = "缺少任务段完成记录（运行标为完成）"
            warnings.append(f"{key}已经执行但没有episodes完成记录，虽然运行status为completed；最终分数留空。")
        else:
            row["record_status"] = "进行中或未写入完成记录"
        rows.append(row)

    interrupted = power.get("interrupted_calls", []) if isinstance(power, dict) else []
    interrupted_ids = {str(row["call"]) for row in interrupted if isinstance(row, dict) and row.get("call") is not None}
    uninterrupted = [row for row in calls if str(row.get("call")) not in interrupted_ids]
    uninterrupted_times = [row["seconds"] for row in uninterrupted if _number(row.get("seconds")) is not None]
    interrupted_observed = [row for row in calls if str(row.get("call")) in interrupted_ids]
    call_ids = Counter(str(row.get("call")) for row in calls)
    if any(count > 1 for count in call_ids.values()):
        warnings.append("inference.jsonl存在重复call编号；耗时按现有记录逐条求和，须核对是否重启追加。")
    peaks = [row["peak_memory_gb"] for row in calls if _number(row.get("peak_memory_gb")) is not None]
    if _number(backend_stats.get("peak_mlx_memory_gb")) is not None:
        peaks.append(backend_stats["peak_mlx_memory_gb"])
    token_counts = Counter()
    for row in calls:
        if _number(row.get("generation_tokens")) is not None:
            token_counts[str(row.get("mode", "未知"))] += row["generation_tokens"]
    timing = {
        "inference_records": len(calls),
        "reported_backend_calls": backend_stats.get("calls"),
        "invalid_output_records": sum(row.get("valid") is False for row in calls),
        "uninterrupted_call_records": len(uninterrupted),
        "uninterrupted_call_records_with_time": len(uninterrupted_times),
        "uninterrupted_seconds_sum": sum(uninterrupted_times) if uninterrupted_times else None,
        "interrupted_call_ids": sorted(interrupted_ids),
        "interrupted_call_records_observed": len(interrupted_observed),
        "interrupted_calls_reported": interrupted,
        "interrupted_active_seconds": None,
        "recorded_backend_inference_seconds": _number(backend_stats.get("inference_seconds")),
        "recorded_episode_seconds_sum": sum(row["recorded_episode_seconds"] for row in rows if row["recorded_episode_seconds"] is not None)
            if any(row["recorded_episode_seconds"] is not None for row in rows) else None,
        "peak_mlx_memory_gb": max(peaks) if peaks else None,
        "model_load_seconds": _number(backend_stats.get("load_seconds")),
        "generation_tokens_by_mode": dict(token_counts),
        "power_pause_record": power,
        "stopped_with_unfinished_episode": any(not row["completion_recorded"] and
            (row["steps_recorded"] or row["complete_inference_records"]) for row in rows)
            and (str(status.get("status", "")).startswith("stopped") or
                 status.get("status") in ("failed", "interrupted", "cancelled")),
        "timing_limit": "中断调用的有效推理时间无法准确还原；未从perf_counter耗时减去epoch暂停时长。"
            if power else "调用耗时合计为原始记录的perf_counter时长，不等于整个任务的墙钟总时长。",
    }

    message_refs = []
    for current_phase, condition, group in sorted({(row["phase"], row["condition"], row["group"]) for row in rows}):
        if condition in SYMBOL_CONDITIONS:
            message_refs.append({"phase": current_phase, "condition": condition, "group": group,
                "statistics_from_existing_analysis": _message_reference(analysis, current_phase, condition, group)})
    if analysis and isinstance(analysis, dict):
        warnings.extend(f"原分析器：{warning}" for warning in analysis.get("data_quality", {}).get("warnings", []))
    probes = []
    for report in sorted(path.glob("frozen_probe*/report.md")):
        probes.append({"report": str(report), "results": _json(report.parent / "results.json", warnings)})
    return {
        "name": path.name, "directory": str(path), "manifest": manifest, "status": status,
        "source_version": _source_version(path, manifest),
        "episodes": rows, "actions_all_recorded_steps": _action_summary(steps), "timing": timing,
        "symbol_statistics_references": message_refs,
        "existing_report": str(path / "report.md") if (path / "report.md").exists() else None,
        "existing_analysis": str(path / "analysis.json") if (path / "analysis.json").exists() else None,
        "probe_reports": probes, "warnings": warnings,
    }


def build_suite(run_dirs):
    runs = [summarize_run(path) for path in run_dirs]
    for previous, current in zip(runs, runs[1:]):
        before = previous["source_version"]["core_file_sha256"]
        after = current["source_version"]["core_file_sha256"]
        common = set(before) & set(after)
        current["source_difference_from_previous"] = {
            "previous_run": previous["name"],
            "changed_files": sorted(name for name in common if before[name] != after[name]),
            "comparable_files": sorted(common),
        }
    return {
        "schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
        "runs": runs,
        "interpretation_limits": [
            "校准与符号先导分开，不将控制条件的自然语言成绩算作符号语言的形成。",
            "最终完成度只取episodes.jsonl的任务段完成记录；进行中分数不作最终结果。",
            "独立重复单位是群体；同群体任务段、动作和消息不能当作独立群体。",
            "消息长度、重复和主体自述不能确认共同意义、组合性、语法或误解修复。",
        ],
        "language_evidence_not_confirmed_by_this_summary": [
            "跨情境的共同意义及稳定理解，且区分发送者编码与伙伴理解。",
            "预先固定留出组合中的首次理解，排除测试中重新协商。",
            "在独立确认情境中对熟悉片段替换/重组所得的选择性行为改变。",
            "成对情境中的关系与参与者敏感性，排除整场景编号。",
            "对特定可观察误解的回应干预及选择性修复。",
            "新成员学习、文化传递和代际变化。",
        ],
    }


def _fmt(value, digits=2):
    return "—" if _number(value) is None else f"{value:.{digits}f}"


def _percent(value):
    return "—" if _number(value) is None else f"{100 * value:.1f}%"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _yes(value):
    return "观察到" if value is True else "未观察到" if value is False else "未知"


def _link(label, path):
    return f"[{label}](<{path}>)"


def render_report(suite):
    runs = suite["runs"]
    lines = ["# 三主体语言形成工程先导：运行总报告", "", f"汇总时间：{suite['generated_at']}", "",
        "本报告汇总实际记录的能力检查与符号实验。它描述合作任务、动作和通信使用，不宣布语言已经形成。", "",
        "## 运行与版本范围", "", "| 运行 | phase | 条件 | 群体 | 当前状态 |", "|---|---|---|---|---|"]
    for run in runs:
        manifest = run["manifest"]
        lines.append(f"| {_cell(run['name'])} | {_cell(manifest.get('phase', '未知'))} | "
            f"{_cell(', '.join(map(str, manifest.get('conditions', []))))} | "
            f"{_cell(', '.join(map(str, manifest.get('groups', []))))} | {_cell(run['status'].get('status', '未知'))} |")
    lines += ["", "每次运行的配置和源码快照各自保存；不同校准版本不能合并成同一训练过程。表中群体编号在不同条件下代表各自独立的私有历史。", ""]
    for run in runs:
        mode = run['manifest'].get('history_mode', '未单独记录')
        detail = '每段从三个空白私有历史开始；段号不代表连续学习。' if mode == 'independent_episodes' else '同条件、同群体跨段保留自身经历。' if mode == 'continuous' else '请查阅该次运行清单。'
        lines.append(f"- **{_cell(run['name'])}**：history_mode={_cell(mode)}；{detail}")
    lines.append('')
    lines += ["natural为局部观察下的自然语言能力控制；full_information为完整当前物理信息下的独立自然语言能力控制，均不与符号条件匹配带宽。", "",
        "| 运行 | 核心源码快照摘要 | 相对上一列出运行的源码差异 | 模型revision记录 |",
        "|---|---|---|---|"]
    for run in runs:
        version = run["source_version"]
        difference = run.get("source_difference_from_previous")
        comparison = "首个列出运行"
        if difference is not None:
            comparison = ", ".join(difference["changed_files"]) or "已能比较的文件相同"
            if not difference["comparable_files"]:
                comparison = "没有可比源码文件"
        digest = version["core_snapshot_sha256"]
        lines.append(f"| {_cell(run['name'])} | {digest[:12] if digest else '缺失'} | {_cell(comparison)} | "
            f"{_cell(version['model_commit_recorded'] or '未记录')} |")
    lines += ["", "摘要只用于识别已保存的核心源码版本；文本差异不自动意味着物理引擎行为不同，也不是对模型权重文件的再次校验。完整逐文件SHA256保存在summary.json。", ""]
    for run in runs:
        if run["status"].get("status") == "stopped_for_rule_clarity":
            lines.append(f"**{_cell(run['name'])}已为补齐规则说明而手动中止。** 仅报告停止前已记录动作；未完成任务段不作为已完成的失败结果，未启动条件不计实验结果。")
        elif run["status"].get("stop_reason"):
            lines.append(f"{_cell(run['name'])}停止原因记录：{_cell(run['status']['stop_reason'])}。")
    lines.append("")
    categories = [
        ("能力检查（calibration）", lambda row: row["phase"] == CALIBRATION_PHASE),
        ("符号工程先导", lambda row: row["phase"] != CALIBRATION_PHASE and row["condition"] in SYMBOL_CONDITIONS),
        ("其他阶段与条件", lambda row: row["phase"] != CALIBRATION_PHASE and row["condition"] not in SYMBOL_CONDITIONS),
    ]
    for title, include in categories:
        selected = [(run, row) for run in runs for row in run["episodes"] if include(row)]
        if not selected:
            continue
        lines += [f"## {title}", "", "| 运行 | 条件 | 群体 | 段 | variant | 最终完成度 | 最终步数 | 已记录步 | 记录状态 |",
            "|---|---|---|---|---|---:|---:|---:|---|"]
        for run, row in selected:
            lines.append(f"| {_cell(run['name'])} | {row['condition']} | {row['group']} | {row['episode']} | "
                f"{_cell(row['variant']) if row['variant'] is not None else '—'} | {_percent(row['final_score'])} | "
                f"{_fmt(row['final_steps'], 0)} | {_fmt(row['last_step_recorded'], 0)} | {row['record_status']} |")
        lines += ["", "没有任务段完成记录时，最终完成度和最终步数留空；已记录步数只表示执行进度。任务超时但正常写入完成记录时，其实际完成度仍计入。", ""]

    lines += ["## 动作与实际发生的过程", "", "| 运行 | 条件/群体/段 | 等待 | 失败的非等待动作 | 曾携带物品 | 拿取成功/尝试 | 加工成功/尝试 | 交付成功动作/尝试 | 新交付单位 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for run in runs:
        for row in run["episodes"]:
            actions = row["actions"]
            observed_actions = actions["actions_recorded"] > 0
            pickups = f"{actions['successful_pickups']}/{actions['pickup_attempts']}" if observed_actions else "—"
            processing = f"{actions['successful_processing_actions']}/{actions['processing_attempts']}" if observed_actions else "—"
            delivery = f"{actions['accepted_delivery_actor_actions']}/{actions['delivery_actor_attempts']}" if observed_actions else "—"
            lines.append(f"| {_cell(run['name'])} | {row['condition']}/{row['group']}/{row['episode']} | "
                f"{actions['wait_actions']} | {actions['failed_nonwait_actions']} | {_yes(actions['holding_observed'])} | "
                f"{pickups} | {processing} | {delivery} | "
                f"{_fmt(actions['newly_delivered_units_from_states'], 0)} |")
    lines += ["", "v3不设工具或加工动作；加工列沿用通用统计字段，应为0/0。成功与尝试分别列出；提交动作、实际拿到资源、完成交付是不同判据。曾携带来自实际状态或观察，也可能由交接或共同搬运产生。", "",
        "等待单列，不因未交付而把等待记成执行失败。失败只计明确反馈action_succeeded=false的非等待动作；缺失反馈不算失败。共同交付可产生两人的成功动作记录，但交付单位从状态变化计算，不重复算两份。未完成段的动作也按已有记录展示。", ""]
    for run in runs:
        actions = run["actions_all_recorded_steps"]
        counts = "，".join(f"{kind}={count}" for kind, count in actions["action_kind_counts"].items()) or "尚无动作记录"
        lines += [f"- **{_cell(run['name'])}**：{counts}。缺少布尔执行反馈的动作{actions['actions_missing_boolean_feedback']}次。"]
    lines.append("")

    lines += ["## 符号长度与额度检查", "", "下列统计直接引用各运行现有analysis.json，不把自然语言控制按符号字母表判为非法。统计以该分析器已纳入的任务段为准，运行中可能尚不包含未结束任务段。", "",
        "| 运行 | 阶段/条件/群体 | 消息数 | 平均字符数 | 非空比例 | 超32字符 | 每人每步超64 | 非法字符 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for run in runs:
        for ref in run["symbol_statistics_references"]:
            stats = ref["statistics_from_existing_analysis"]
            label = f"{ref['phase']}/{ref['condition']}/{ref['group']}"
            if stats is None:
                lines.append(f"| {_cell(run['name'])} | {label} | — | — | — | — | — | 未有对应分析 |")
            else:
                invalid = stats.get("invalid_character_counts")
                invalid_text = json.dumps(invalid, ensure_ascii=False) if invalid is not None else "未知"
                mean = (stats.get("length_all") or {}).get("mean")
                lines.append(f"| {_cell(run['name'])} | {label} | {_fmt(stats.get('message_rows'), 0)} | {_fmt(mean)} | "
                    f"{_percent(stats.get('nonempty_ratio'))} | {_fmt(stats.get('over_32_count'), 0)} | "
                    f"{_fmt(stats.get('agent_steps_over_64'), 0)} | {_cell(invalid_text)} |")
    lines += ["", "触及上限不等于存在被截去的语义内容；长度、重复和相邻窗口变化不作语言结构判据。", ""]

    lines += ["## 调用、内存与耗时", "", "| 运行 | 已记录完整调用 | 其中输出校验失败 | MLX峰值GB | 未受已知中断影响的调用 | 这些调用耗时合计秒 | 中断调用 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for run in runs:
        timing = run["timing"]
        lines.append(f"| {_cell(run['name'])} | {timing['inference_records']} | {timing['invalid_output_records']} | "
            f"{_fmt(timing['peak_mlx_memory_gb'])} | {timing['uninterrupted_call_records']} | "
            f"{_fmt(timing['uninterrupted_seconds_sum'])} | {', '.join(timing['interrupted_call_ids']) or '无已知标记'} |")
    lines.append("")
    for run in runs:
        timing = run["timing"]
        line = f"- **{_cell(run['name'])}**：模型加载记录{_fmt(timing['model_load_seconds'])}秒；"
        if timing["power_pause_record"]:
            line += (f"call {', '.join(timing['interrupted_call_ids']) or '未标定'}受SIGSTOP/供电影响，已从上表其余调用耗时合计中排除。"
                "中断调用的有效推理时间无法准确还原；epoch暂停时长与perf_counter时长不可直接相减。"
                f"原始backend推理耗时记录为{_fmt(timing['recorded_backend_inference_seconds'])}秒，含暂停影响，不标作有效推理总时长。")
        else:
            line += f"原始backend推理耗时记录{_fmt(timing['recorded_backend_inference_seconds'])}秒。"
        line += f" 已完成任务段原始耗时合计{_fmt(timing['recorded_episode_seconds_sum'])}秒"
        line += "（含暂停影响）。" if timing["power_pause_record"] else "。"
        if timing["stopped_with_unfinished_episode"]:
            line += " 该运行中止时仍有未完成任务段；尚未完成并写入日志的调用耗时未包含在完整调用合计内。"
        lines.append(line)
    lines += ["", "调用耗时合计不是整个实验的墙钟总耗时；未完成调用尚未写入日志时不计入完整调用数。token分模式统计和原始计时字段保存在summary.json。", ""]

    lines += ["## 冻结状态检查与语言证据", ""]
    found_probe = False
    for run in runs:
        for probe in run["probe_reports"]:
            found_probe = True
            lines.append(f"- {_cell(run['name'])}：{_link('冻结状态探针报告', probe['report'])}。")
            if isinstance(probe["results"], list):
                mismatches = sum(row.get("original_reproduced_exactly") is False for row in probe["results"] if isinstance(row, dict))
                lines.append(f"  记录{len(probe['results'])}个冻结案例；原动作未完全重现{mismatches}例，需结合该报告的版本与复现检查解释。")
    if not found_probe:
        lines.append("这些运行目录中尚未发现冻结探针报告，本汇总不推断频道具有因果作用。")
    lines += ["", "若已有本步消息文字置空检查，它最多支持该文本对给定状态下一步行动的影响；过去消息仍保留，长度也变化，不能当作整段无通信基线或语义、组合性确认。", "",
        "下列证据本汇总未确认；只有另行实施并报告相应检查后才能作更强结论：", ""]
    lines += [f"- {item}" for item in suite["language_evidence_not_confirmed_by_this_summary"]]
    lines += ["", "同一条件只有一个群体时，没有群体层面的独立重复。多任务段提供过程记录，不能补成多个独立群体。即使任务完成度不同，也可能来自任务理解、动作执行、规划或通信使用差异。", "",
        "## 原始资料与数据质量", ""]
    for run in runs:
        links = [_link("运行配置", str(Path(run["directory"]) / "manifest.json"))]
        if run["existing_report"]:
            links.append(_link("单运行报告", run["existing_report"]))
        if run["existing_analysis"]:
            links.append(_link("字符串统计JSON", run["existing_analysis"]))
        lines += [f"- **{_cell(run['name'])}**：{'；'.join(links)}。"]
        lines += [f"  读取提示：{_cell(warning)}" for warning in run["warnings"]]
    lines.append("")
    return "\n".join(lines)


def write_suite(run_dirs, out):
    suite = build_suite(run_dirs)
    target = Path(out).resolve()
    if target.is_dir() or not target.suffix:
        target.mkdir(parents=True, exist_ok=True)
        markdown, structured = target / "report.md", target / "summary.json"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown, structured = target, target.with_suffix(".summary.json")
    for path, content in ((markdown, render_report(suite)), (structured, json.dumps(suite, ensure_ascii=False, indent=2, allow_nan=False) + "\n")):
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return {"report": str(markdown), "summary": str(structured), "runs": len(suite["runs"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(write_suite(args.run_dirs, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
