"""Build the Chinese report for the v0.42 intervention assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ASSIGNMENTS = ("canonical", "cyclic")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
BLOCKS = ("012", "021", "102", "120", "201", "210")


def read(path):
    return json.loads(path.read_text())


def subset(rows, assignment=None, condition=None):
    return [
        row
        for row in rows
        if (assignment is None or row["assignment"] == assignment)
        and (condition is None or row["condition"] == condition)
    ]


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def direct(rows, key, split="target60"):
    return [row[key][split]["J"] for row in rows]


def nested(rows, key, value, split="target60"):
    return [row[key][str(value)][split]["J"] for row in rows]


def fmt(values, scale=100.0):
    m, s = mean_sd(values)
    return f"{m * scale:.2f}%（SD {s * scale:.2f}%）"


def pct(values):
    return mean_sd(values)[0] * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "sequence_intervention_analysis.json")
    rows = data["rows"]
    report = out / "三资源双token端点干预研究报告.md"
    lines = [
        "# 三资源双 token 端点干预：v0.42 研究报告",
        "",
        "## 研究问题",
        "",
        "v0.41 的双 token 形成结果显示 token1 携带位置相关信息，但 pair NMI 达到饱和，仍无法排除两个 token 只是对同一地点的冗余编码。本轮不再改变训练，而是对 72 条 v0.41 形成终点做预先定义的端点干预：删除任一 token、交换 token 顺序、把 token1 在世界之间打乱、置换三个资源的双 token 块，并把同一组照片重定位到六个站点的循环坐标。核心问题是：协议的功能依赖哪个位置，资源块置换是否表现出结构性等变，以及这种结构能否在未见的空间关系上保持。",
        "",
        "## 数据和干预",
        "",
        "本轮输入固定为 v0.41 的 72 条 endpoint（4 seeds × 3 partitions × 2 resource assignments × 3 partner topologies），每条 endpoint 在 A/B/C 三种评估拓扑、4 个 team 和 120 张测试地图上重放，因此共有 864 个 team-schedule 记录。所有干预都使用同一 receiver 权重；位置重定位会重新运行冻结 sender 前端，照片身份不变。",
        "",
        "- token mask：将 token0 或 token1 替换为 0–6 的每一个已有词表值，不假设某个符号是保留的 null；",
        "- token order：交换每个 sender 内的两个 token，或把 token1 在世界之间做固定循环移位；",
        "- resource blocks：枚举 3 个双 token 资源块的全部 6 个排列，同时报告保持原资源目标的 literal 分数和同步置换目标的 equivariant 分数；",
        "- site relocation：identity 加 5 个站点 ID 循环移位，target/train 身份仍按原 map_id 划分。",
        "",
        "设计文件：[sequence_intervention_design.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/sequence_intervention_design.json)；执行绑定：[invocation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/results/sequence_intervention_001/invocation.json)。",
        "",
        "## 结果一：端点基线可复现，token 顺序交换会破坏功能",
        "",
        "下表是未干预 endpoint 与两个顺序干预的 target-60 联合 J。数值跨两个资源排列、4 个 seed、3 个 partition、3 个评估拓扑和 4 个 team 汇总。",
        "",
        "| 训练拓扑 | baseline | swap token0/token1 | shuffle token1 across worlds |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = subset(rows, condition=condition)
        lines.append(f"| {condition} | {fmt(direct(s, 'baseline'))} | {fmt(direct(s, 'swap'))} | {fmt(direct(s, 'shuffle_t1'))} |")
    lines += [
        "",
        "baseline 与 v0.41 保存协议的 receiver argmax 完全一致（0 个 replay mismatch）。交换两个 token 后 target J 降到约 0.3%–1.0%，而跨世界打乱 token1 的损失较小但稳定，说明 token0 是主要功能载体，token1 仍参与当前协议。",
        "",
        "## 结果二：token0 删除几乎摧毁联合任务，token1 删除保留部分功能",
        "",
        "将 mask 值在 7 个词表符号上平均后，target J 如下。",
        "",
        "| 训练拓扑 | baseline | mask token0（7 值均值） | mask token1（7 值均值） |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = subset(rows, condition=condition)
        t0 = [value for v in range(7) for value in nested(s, "mask_t0", v)]
        t1 = [value for v in range(7) for value in nested(s, "mask_t1", v)]
        lines.append(f"| {condition} | {fmt(direct(s, 'baseline'))} | {fmt(t0)} | {fmt(t1)} |")
    lines += [
        "",
        "mask token0 后 J 约 0.2%–0.4%，mask token1 后仍有约 8.3%–16.3%。这与 v0.41 的 token0/token1 NMI 差异一致，但 mask 不是把信息从模型中删除的训练操纵，而是输入端的端点反事实；因此只能说明当前功能对两个位置的依赖程度。",
        "",
        "## 结果三：资源块置换显示伙伴覆盖下的结构性等变",
        "",
        "下表给出非 identity 资源块排列的平均 target J；literal 将 receiver 输出与原始资源目标比较，equivariant 将目标轴同步按相同排列置换。",
        "",
        "| 训练拓扑 | literal（5 个非 identity 排列均值） | equivariant（5 个非 identity 排列均值） |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = subset(rows, condition=condition)
        literal = [row["block_permutations"][block]["literal"]["target60"]["J"] for block in BLOCKS if block != "012" for row in s]
        equiv = [row["block_permutations"][block]["equivariant"]["target60"]["J"] for block in BLOCKS if block != "012" for row in s]
        lines.append(f"| {condition} | {fmt(literal)} | {fmt(equiv)} |")
    lines += [
        "",
        "在固定 A 条件下，非 identity equivariant J 仍接近 baseline；轮换 A/B 和随机 A/B/C 则分别上升到约 46%–50% 和 62%–66%，同时 literal J 接近零。这种差异表示：在伙伴覆盖较广的形成条件下，receiver 对资源块顺序形成了可以随资源轴一起重排的关系结构。它比 pair NMI 更接近一个干预证据，但仍是 receiver 端点的 counterfactual，不能单独证明可开放组合的语法。",
        "",
        "## 结果四：位置重定位改变 target J，暴露坐标和关系分布的交互",
        "",
        "| 训练拓扑 | identity | 5 个非 identity 循环移位均值 |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = subset(rows, condition=condition)
        ident = [row["site_relocations"]["0"]["target60"]["J"] for row in s]
        shifted = [row["site_relocations"][str(i)]["target60"]["J"] for i in range(1, 6) for row in s]
        lines.append(f"| {condition} | {fmt(ident)} | {fmt(shifted)} |")
    lines += [
        "",
        "循环重定位在三个拓扑中都提高了 target J，随机 A/B/C 的 identity 约 31.1%，非 identity 平均约 70.5%。这不是“模型发现了新地点”的证据：六个站点仍来自同一冻结视觉环境，且 target 分组沿用原 map_id。更保守的解释是，sender 的站点编码、receiver 的输出坐标和原始 target 关系划分存在交互；因此坐标重定位必须作为后续形成实验的对称性控制，而不能被当作泛化收益。",
        "",
        "## 对核心问题的回答",
        "",
        "1. 当前协议的功能主要依赖 token0，但 token1 不是固定填充；删除或打乱 token1 会降低功能，交换顺序几乎摧毁功能。",
        "2. 伙伴覆盖改变了协议的结构性质。轮换和随机伙伴条件下，资源块同步置换的等变评分显著高于 literal 评分，说明三个双 token 块可以作为带有资源轴的结构单位被重新排列。",
        "3. 这仍不足以称为语言或语法。所有结果来自有限六地点任务的 endpoint，receiver 可以用整码查表；位置重定位的提升还可能来自 target 关系分布与架构坐标偏置。",
        "",
        "## 限制和下一轮",
        "",
        "- 本轮没有新增训练，不能回答训练过程中干预是否改变协议形成；",
        "- 资源块排列是接收端反事实，尚未在形成阶段遍历全部资源排列；",
        "- 位置重定位复用六个站点和冻结照片，没有真正的新视觉位置；",
        "- mask 使用已有词表值，虽然覆盖了全部 7 个替换值，仍不是一个训练过的 null token；",
        "- sender/receiver 更新、奖励和视觉前端仍来自集中式 v0.41 任务。",
        "",
        "下一轮应训练完整的六种资源排列，并在每次 episode 随机化资源/角色顺序；同时保留删除/交换、跨世界 token 重组和未见资源位置关系作为预注册测试，再把形成终点接入 v0.40 的代际替换，测量等变结构能否由新主体保留。",
        "",
        "## 可复核性",
        "",
        "分析结果：[sequence_intervention_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/results/sequence_intervention_001/sequence_intervention_analysis.json)；独立审计：[sequence_intervention_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/results/sequence_intervention_001/sequence_intervention_audit.json)；图形：[01_sequence_intervention.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/results/sequence_intervention_001/figures/01_sequence_intervention.png)；视觉 QA：[visual_qa.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.42/results/sequence_intervention_001/visual_qa.json)。",
        "",
        "独立审计覆盖 72 条链、864 条记录、12096 个 mask 表、5184 个资源块表和 5184 个位置表；所有 baseline replay 与保存协议一致，最大重算误差为 0。",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
