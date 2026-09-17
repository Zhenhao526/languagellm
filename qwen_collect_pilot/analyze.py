"""Build an auditable Chinese report from an actual pilot result directory.

Uses only Python's standard library. No model execution or result fabrication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


CONDITIONS = ("original", "empty", "permuted")
CONDITION_NAMES = {
    "original": "原消息",
    "empty": "空消息",
    "permuted": "跨需求置换消息",
}


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} 第 {number} 行不是完整 JSON；请待写入结束后重试") from error
    return rows


def message_text(message):
    if message == "":
        return "∅（空消息）"
    # A code span prevents the message alphabet from becoming Markdown syntax.
    return "`" + str(message).replace("`", "\\`") + "`"


def proportion(hits, n):
    return f"{hits}/{n}（{100 * hits / n:.1f}%）" if n else "未运行"


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        **{role: sum(bool(row["scores"][role]) for row in rows) for role in ("A", "B", "overall")},
        "success_rate": sum(bool(row["scores"]["overall"]) for row in rows) / n if n else None,
        "unique_worlds": len({tuple(row["world"][key] for key in ("d_a", "d_b", "p_a", "p_b")) for row in rows}),
    }


def validate_trial(row):
    world, actions, scores = row["world"], row["actions"], row["scores"]
    if any(type(world[key]) is not int or world[key] not in (0, 1)
           for key in ("d_a", "d_b", "p_a", "p_b")):
        raise ValueError("记录中存在非法世界状态")
    if any(type(actions[role]) is not int or actions[role] not in (0, 1) for role in ("A", "B")):
        raise ValueError("记录中存在非法动作")
    expected = {"A": actions["A"] == world["d_a"] ^ world["p_a"],
                "B": actions["B"] == world["d_b"] ^ world["p_b"]}
    expected["overall"] = expected["A"] and expected["B"]
    if scores != expected:
        raise ValueError(f"记录得分与世界/动作不一致：{row['phase']} / {row['group']} / {row['index']}")
    msg = row["message"]
    if not isinstance(msg, str) or len(msg) > 2 or any(char not in "@#%&" for char in msg):
        raise ValueError("记录中存在超出信道约束的消息")


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def fmt_number(value, decimals=2):
    return "未记录" if value is None else f"{value:,.{decimals}f}"


def inference_role(row):
    """Recover sender identity from older frozen-encoder log labels."""
    label = row.get("label") or {}
    if label.get("role") in ("A", "B", "C"):
        return label["role"]
    if label.get("phase") == "frozen_encoder":
        return "C"
    return None


def build_report(result_dir: Path):
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"未找到 {summary_path}；实验尚未写出摘要")
    summary = read_json(summary_path)
    config = read_json(result_dir / "config.json", {})
    trials = read_jsonl(result_dir / "trials.jsonl")
    inference = read_jsonl(result_dir / "inference.jsonl")
    private_calls = [row for row in inference if row.get("mode") == "private_reasoning"]
    private_reasoning = bool(config.get("private_reasoning", private_calls))
    private_limit = config.get("private_reasoning_max_tokens", 256)
    calibration_outputs = [row for row in inference if row.get("mode") == "action"
                           and (row.get("label") or {}).get("phase") == "known_protocol"]
    short_calibration = bool(calibration_outputs) and all(
        row.get("messages") and row["messages"][0]["role"] == "user" for row in calibration_outputs)
    for row in trials:
        validate_trial(row)

    groups = list(dict.fromkeys(
        [str(seed) for seed in config.get("seeds", [])]
        + list(summary.get("groups", {}))
        + [str(row["group"]) for row in trials if row["phase"] == "interaction"]
    ))
    audits = []
    derived = {"completed": bool(summary.get("completed")), "private_reasoning": private_reasoning, "groups": {}}
    calibration = summarize([row for row in trials if row["phase"] == "known_protocol"])
    derived["calibration"] = calibration
    if calibration["n"] and summary.get("calibration", {}).get("overall") != calibration["overall"]:
        audits.append("能力对照 summary.json 与 trials.jsonl 的成功数不一致。")

    evaluations = {}
    for group in ["zero_interaction"] + groups:
        evals = {}
        for condition in CONDITIONS:
            rows = [row for row in trials if str(row["group"]) == group and row["phase"] == "frozen_" + condition]
            stats = summarize(rows)
            evals[condition] = stats
            if stats["n"] and (stats["n"] != 16 or stats["unique_worlds"] != 16):
                audits.append(f"{group} / {condition} 未完整覆盖 16 个互异状态，不能作为完整枚举结果。")
            if condition == "empty" and stats["n"] == stats["unique_worlds"] == 16 and stats["overall"] != 4:
                audits.append(f"{group} 空消息结果偏离理论 4/16，需要检查状态或信息泄漏。")
            source = summary.get("zero_interaction", {}) if group == "zero_interaction" else summary.get("groups", {}).get(group, {}).get("evaluation", {})
            recorded = source.get("conditions", {}).get(condition)
            if recorded and any(recorded.get(key) != stats[key] for key in ("n", "A", "B", "overall")):
                audits.append(f"{group} / {condition} 摘要与逐轮记录不一致。")
        evaluations[group] = evals
    derived["evaluations"] = evaluations

    complete = bool(summary.get("completed"))
    status = "已完成" if complete else "未完成或仅完成部分阶段"
    lines = ["# 三主体采集预实验报告", "", f"运行目录：`{result_dir.name}`。状态：**{status}**。", ""]
    if not complete:
        lines += ["以下仅汇总已写入日志的实际结果；未运行阶段不补值。", ""]
    elif groups:
        interaction_count = sum(row["phase"] == "interaction" for row in trials)
        original_results = {group: evaluations[group]["original"] for group in evaluations}
        zero = original_results["zero_interaction"]
        result_text = "，".join(
            f"组 {group} 为 {proportion(original_results[group]['overall'], original_results[group]['n'])}"
            for group in groups)
        lines += [
            f"完成 {len(groups)} 组、合计 {interaction_count} 轮互动。已知协议能力对照为 {proportion(calibration['overall'], calibration['n'])}。冻结原消息评估中，零互动为 {proportion(zero['overall'], zero['n'])}；互动后{result_text}。", ""]
        if all(original_results[group]["n"] == 16 and original_results[group]["success_rate"] <= 0.5 for group in groups):
            gains = "、".join(
                f"组 {group} {'多' if (gain := original_results[group]['overall'] - evaluations[group]['empty']['overall']) >= 0 else '少'} {abs(gain)} 个成功状态"
                for group in groups)
            lines += [f"相较空消息，原消息下{gains}，且整体成功率均未超过一半。消息在当前任务中有一定作用，但作用有限，尚未观察到可靠的共享协议。这是少量种子上的任务结果，不是语言起源或统计稳健性的结论。", ""]
        zero_codes = summary.get("zero_interaction", {}).get("codes_by_demand", {})
        for group in groups:
            codes = summary.get("groups", {}).get(group, {}).get("evaluation", {}).get("codes_by_demand", {})
            if len(zero_codes) != 4 or len(codes) != 4:
                continue
            count, initial = len(set(codes.values())), len(set(zero_codes.values()))
            if count > initial and original_results[group]["overall"] == zero["overall"]:
                lines += [f"组 {group} 的 C 在四种需求上的冻结输出从 {initial} 种增至 {count} 种，群体成功数仍为 {zero['overall']}/{zero['n']}；发送端区分更多需求，没有自动转化为接收端的正确协作。", ""]
            elif count < 4:
                lines += [f"组 {group} 的 C 在四种需求上只产生 {count} 种冻结消息，仍有不同需求使用同一编码。", ""]
    if audits:
        lines += ["## 记录核对", ""] + [f"- {item}" for item in audits] + [""]
    else:
        lines += [f"已从世界状态与动作重新计算并核对 {len(trials)} 条任务记录的得分。", ""]

    decision_mode = (
        f"本次启用 `--private-reasoning`：每次决策先用温度 0 生成最多 {private_limit} token 的私有分析，再作受限输出。分析可以使用自然语言，只在当前主体本轮决策中使用，不发送给其他主体，也不写入后续互动历史。它会保存在实验者推理日志中用于审计。C 的正式消息在互动阶段仍用温度 0.7，采集动作与所有冻结评估输出为贪心解码。"
        if private_reasoning else
        "本次采用直接受限输出，没有显式私有分析步骤；模型从私有观察和历史直接生成正式消息或动作。")
    calibration_description = (
        "本次使用独立的简短单 user 提示，只包含当前采集者所需的已知消息映射与本地槽位。它提供一个容易执行条件下的能力检查；自由编码阶段另有完整任务说明与互动历史，因此校准通过不代表复杂历史中的每次动作选择都正确。"
        if short_calibration else
        "本次使用运行日志所记录的已知协议提示。通过这一能力对照不能保证自由编码提示和复杂互动历史中的动作选择无误。")
    lines += ["## 任务与解释范围", "",
        "A 采纤维，B 采燃料，C 知道营地分别需要哪一种。A、B 各自只看到两个资源所在的槽位，C 看不到槽位排列。需求与两处排列构成四个独立二值变量，共 16 个等概率状态。C 向两名采集者广播同一条消息；随后 A、B 独立选择槽位，双方都匹配需求才算群体成功。", "",
        "消息字母表为 `@#%&`，允许 0–2 个字符。每个可用字符均验证为独立的单个 token，并用 logits 硬屏蔽限制输出；关闭模型的 thinking 模式。采集者结算后只获得自身匹配与群体成功反馈，C 只获得群体成功反馈。所有观察均为文字描述的离散资源，没有图片。角色和分工由实验者预设，信道为单向广播。", "",
        decision_mode, "",
        "由于采集者完整看到两个候选并且只有二选一，“自己匹配”的事后反馈足以反推出该轮目标资源。这是较强的学习信号；它在动作提交后才提供，不泄漏待决策轮次的目标。", "",
        "三个主体共用同一份固定模型权重与分词器，观察与互动历史分开保存，每次推理重新建立缓存。这里的“训练”指累积私有上下文，未更新模型权重；三个上下文仍有相同的预训练语言知识。零互动条件也可能出现由共同先验产生的编码。", "",
        "无通信时，在固定历史、独立需求与完整状态枚举下，任意确定性的本地决策组合均有 4/16 的群体成功率。训练阶段使用独立随机抽样，有限轮次成功率可以偏离 25%。", "",
        "## 已知协议能力对照", "",
        "实验程序按明确给定的四项协议产生消息，仅检验 A、B 能否解码并按随机槽位选择。此阶段没有调用 C 生成消息，其历史也不会进入自由编码阶段。", "",
        calibration_description, "",
        "| 阶段 | 状态数 | A 匹配 | B 匹配 | 群体成功 |",
        "|---|---:|---:|---:|---:|",
        f"| 已知协议 | {calibration['n']} | {proportion(calibration['A'], calibration['n'])} | {proportion(calibration['B'], calibration['n'])} | {proportion(calibration['overall'], calibration['n'])} |", "",
        "## 冻结策略的完整状态评估", "",
        "每个阶段先冻结历史，再以贪心解码评估全部 16 个状态，不返回评估反馈。完全相同的私有提示词可以复用确定性输出，因此 16 个任务状态不等于 16 次独立随机试验。下表是当前冻结策略的枚举成功率，不附抽样置信区间。", "",
        "| 上下文 | 原消息 | 空消息 | 跨需求置换消息 | 原消息－空消息 |",
        "|---|---:|---:|---:|---:|"]
    for group, evals in evaluations.items():
        label = "零互动" if group == "zero_interaction" else f"组 {group} 互动后"
        entries = [proportion(evals[condition]["overall"], evals[condition]["n"]) for condition in CONDITIONS]
        orig, empty = evals["original"], evals["empty"]
        difference = f"{100 * (orig['success_rate'] - empty['success_rate']):+.1f} 个百分点" if orig["n"] == orig["unique_worlds"] == empty["n"] == empty["unique_worlds"] == 16 else "未获得完整配对评估"
        lines.append(f"| {label} | " + " | ".join(entries) + f" | {difference} |")
    lines += ["", "置换把本轮需求 `(dₐ,dᵦ)` 的消息替换成同一编码者对 `(1−dₐ,1−dᵦ)` 产生的消息，保持四种需求上的消息分布。如果不同需求得到同一消息，部分置换不会改变输入；需要结合下表的实际变化比例判断。置换与空消息干预可以检验接收者是否利用消息，但单独不能证明组合语义或人类语言能力。", ""]

    lines += ["## 冻结编码与符号使用", "",
        "表中列出 C 在同一份冻结历史下对四种需求的实际贪心输出；它们由模型产生，并非自由编码阶段给定的词典。", "",
        "| 上下文 | F0 / G0 | F0 / G1 | F1 / G0 | F1 / G1 | 不同消息数 | 置换实际改变消息 |",
        "|---|---|---|---|---|---:|---:|"]
    for group in evaluations:
        source = summary.get("zero_interaction", {}) if group == "zero_interaction" else summary.get("groups", {}).get(group, {}).get("evaluation", {})
        codes = source.get("codes_by_demand", {})
        entries = [message_text(codes[need]) if need in codes else "未记录" for need in ("00", "01", "10", "11")]
        distinct = len(set(codes.values())) if codes else "未记录"
        changed = sum(codes[key] != codes[f"{1-int(key[0])}{1-int(key[1])}"] for key in codes) / 4 if set(codes) == {"00", "01", "10", "11"} else None
        label = "零互动" if group == "zero_interaction" else f"组 {group} 互动后"
        changed_text = f"{100 * changed:.0f}%" if changed is not None else "未记录"
        lines.append(f"| {label} | " + " | ".join(entries) + f" | {distinct} | {changed_text} |")
    lines += ["", "## 互动阶段的随机样本结果", "",
        "各组从空白私有互动历史开始。需求与位置按组内随机种子独立抽样；C 的采样温度为 0.7，采集者为贪心解码。前后半段是不同的随机任务样本，成功率差异也可能来自任务抽样和消息探索，不能直接解释为学习效应。", "",
        "互动中的 C 使用随机采样，冻结评估使用贪心解码，两者并非同一行为策略。学习前后的可比结果是零互动与互动后在相同贪心设置下的完整状态评估，不能把随机互动成功率直接与冻结成功率相减来量化学习。", "",
        "| 组 | 前半段群体成功 | 后半段群体成功 | 全程群体成功 | 使用消息数 | 空消息次数 | 平均消息长度 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    group_counts = {}
    for group in groups:
        rows = sorted([row for row in trials if str(row["group"]) == group and row["phase"] == "interaction"], key=lambda row: row["index"])
        mid = len(rows) // 2
        counts = Counter(row["message"] for row in rows)
        chars = Counter(char for row in rows for char in row["message"])
        stats, first, second = summarize(rows), summarize(rows[:mid]), summarize(rows[mid:])
        lengths = [len(row["message"]) for row in rows]
        derived["groups"][group] = {"interaction": stats, "first_half": first, "second_half": second,
            "message_counts": dict(sorted(counts.items())), "character_counts": dict(sorted(chars.items())),
            "mean_message_chars": statistics.mean(lengths) if lengths else None}
        group_counts[group] = counts
        average = f"{statistics.mean(lengths):.2f}" if lengths else "未运行"
        lines.append(f"| {group} | {proportion(first['overall'], first['n'])} | {proportion(second['overall'], second['n'])} | {proportion(stats['overall'], stats['n'])} | {len(counts)} | {counts['']} | {average} |")
    for group, counts in group_counts.items():
        if not counts:
            continue
        usage = "；".join(f"{message_text(msg)}：{count} 次" for msg, count in counts.most_common())
        char_counts = derived["groups"][group]["character_counts"]
        characters = "、".join(f"`{char}` {char_counts.get(char, 0)} 次" for char in "@#%&")
        lines += ["", f"组 {group} 的完整消息频数：{usage}。字符频数：{characters}。"]
    lines += ["", "四种需求只需要四个可区分的整体编码。当前信道含空串在内共有 21 种可用消息，因此高成功率无需组合语法。符号数、长度和频数仅是描述统计；本次任务也没有保留未见过的概念组合用于组合泛化检验。", ""]

    backend = summary.get("backend", {})
    timings = [row["seconds"] for row in inference]
    peaks = [row["peak_memory_gb"] for row in inference if row.get("peak_memory_gb") is not None]
    invalid = [row["call"] for row in inference if row.get("valid") is False]
    derived["inference"] = {"calls": len(inference), "seconds": sum(timings),
        "median_seconds": statistics.median(timings) if timings else None,
        "p95_seconds": percentile(timings, 0.95),
        "peak_mlx_memory_gb": backend.get("peak_mlx_memory_gb", max(peaks, default=None)),
        "max_prompt_tokens": max((row.get("prompt_tokens", 0) for row in inference), default=0),
        "invalid_calls": invalid,
        "private_reasoning_calls": len(private_calls),
        "private_reasoning_seconds": sum(row["seconds"] for row in private_calls),
        "private_reasoning_generation_tokens": sum(row.get("generation_tokens", 0) for row in private_calls),
        "private_reasoning_length_truncations": sum(row.get("finish_reason") == "length" for row in private_calls),
        "private_reasoning_truncated_call_ids": [row["call"] for row in private_calls if row.get("finish_reason") == "length"],
        "private_reasoning_by_role": {role: {
            "calls": sum(inference_role(row) == role for row in private_calls),
            "length_truncations": sum(inference_role(row) == role and row.get("finish_reason") == "length" for row in private_calls)
        } for role in ("A", "B", "C")},
        "private_reasoning_unattributed_calls": [row["call"] for row in private_calls if inference_role(row) is None]}
    metrics = derived["inference"]
    hardware = config.get("hardware", {}).get("hardware", {})
    lines += ["## 本地运行与复现记录", "",
        f"模型：`{config.get('model', '未记录')}`；固定版本：`{config.get('revision', '未记录')}`。",
        "", f"设备记录：{hardware.get('chip_type', '未记录芯片')}，{hardware.get('physical_memory', '未记录内存')}。", "",
        "| 指标 | 实际记录 |", "|---|---:|",
        f"| 模型加载耗时 | {fmt_number(backend.get('load_seconds'))} 秒 |",
        f"| 全流程耗时（不含模型下载） | {fmt_number(summary.get('elapsed_seconds'))} 秒 |",
        f"| 模型推理调用数（含启动检查与所有条件） | {len(inference)} |",
        f"| 其中私有分析调用数 | {metrics['private_reasoning_calls']} |",
        f"| 私有分析累计耗时 | {fmt_number(metrics['private_reasoning_seconds'])} 秒 |",
        f"| 私有分析累计生成 token | {metrics['private_reasoning_generation_tokens']:,} |",
        f"| 私有分析因长度上限结束 | {proportion(metrics['private_reasoning_length_truncations'], metrics['private_reasoning_calls'])} |",
        f"| 推理累计耗时 | {fmt_number(sum(timings))} 秒 |",
        f"| 单次调用耗时中位数 / P95 | {fmt_number(metrics['median_seconds'])} / {fmt_number(metrics['p95_seconds'])} 秒 |",
        f"| 最长输入 | {metrics['max_prompt_tokens']:,} token |",
        f"| MLX 峰值内存 | {fmt_number(metrics['peak_mlx_memory_gb'])} GB |",
        f"| 信道校验失败调用数 | {len(invalid)} |", "",
        "私有分析按角色统计：" + "；".join(
            f"{role} 共 {counts['calls']} 次、长度截断 {counts['length_truncations']} 次"
            for role, counts in metrics["private_reasoning_by_role"].items()) +
            f"。无法归属角色的调用有 {len(metrics['private_reasoning_unattributed_calls'])} 次；旧日志中未写 role 的 `frozen_encoder` 阶段按任务定义归属 C。", "",
        "MLX 内存为框架记录的十进制 GB，不是整机内存使用量或进程 RSS。单次耗时统计混合了启动检查、正式输出及启用时的私有分析；输入历史预填充和私有分析生成均计入推理耗时，不能由极短正式输出的速度推断长文本吞吐。推理次数不含冻结评估中复用的相同私有提示词。", "",
        "私有分析截断按 `mode=private_reasoning` 且 `finish_reason=length` 统计；它表示达到 token 上限，不保证分析已经完成。正式消息与动作仍受原有信道限制。截断可能影响决策质量，需要与协议执行错误区分；未启用私有分析的旧运行没有这一额外步骤。", "",
        "每次推理的完整提示词、输出、生成参数、耗时和缓存配置保存在 [inference.jsonl](inference.jsonl)；世界状态、动作及评分保存在 [trials.jsonl](trials.jsonl)。[config.json](config.json) 记录模型版本、文件校验、软件版本、代码摘要与实验参数；[summary.json](summary.json) 是原始运行摘要，[analysis.json](analysis.json) 是本脚本从逐轮记录重新计算的统计。组内 `history_A.json`、`history_B.json`、`history_C.json` 保存三个主体各自实际获得的互动记录。实验者日志可以包含全局状态，但不回填到主体上下文。", "",
        "## 可以据此判断什么", "",
        "该预实验用于检查本地运行、任务执行、私有上下文和受限信道，并初步观察共同预训练先验与短期互动如何影响编码。可比较零互动和互动后的冻结原消息结果，再检查空消息与置换对照；差异只对应当前任务、提示词、模型和少量随机种子。", "",
        "它不能据此证明语言从无到有、内生分工、代际传承或语法涌现。更强的结论需要重复种子、改变信息结构与任务依赖、控制既有符号和资源标签偏好，并设置未见组合与新主体迁移等独立检验。", ""]
    derived["audit_warnings"] = audits
    return "\n".join(lines), derived


def main():
    parser = argparse.ArgumentParser(description="从真实采集预实验目录生成中文报告")
    parser.add_argument("result_dir", type=Path, help="具体的 results/<timestamp> 目录")
    args = parser.parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    report, derived = build_report(result_dir)
    (result_dir / "report.md").write_text(report, encoding="utf-8")
    (result_dir / "analysis.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result_dir / "report.md")


if __name__ == "__main__":
    main()
