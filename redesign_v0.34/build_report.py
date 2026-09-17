"""Build the v0.34 paired topology report from sealed analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (34101, 34102, 34103, 34104)


def read(path: Path):
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.3f} 个百分点"


def pooled(aggregate, condition: str, split: str, key: str = "J") -> float:
    return aggregate[condition]["scores"][split]["pooled"][key]


def mean_sd(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def seed_level_delta(rows, schedule: str, metric: str = "J"):
    values = []
    for seed in SEEDS:
        values.append(sum(row[schedule]["rotating_minus_fixed"][metric] for row in rows if row["seed"] == seed) / 3.0)
    return mean_sd(values), values


def build(out: Path) -> Path:
    analysis = read(out / "analysis.json")
    transfer = read(out / "rotation_transfer.json")
    topology = read(out / "cross_topology.json")
    validation = read(out / "raw_validation.json")
    audit = read(out / "audit_execution.json")
    complete = read(out / "training_complete.json")

    fixed_out = out.parent / "fixed_001"
    fixed = read(fixed_out / "fixed_control_analysis.json")
    fixed_cross = read(fixed_out / "fixed_control_cross.json")
    fixed_validation = read(fixed_out / "fixed_control_raw_validation.json")
    fixed_audit = read(fixed_out / "fixed_control_audit.json")
    fixed_complete = read(fixed_out / "training_complete.json")
    fixed_receipt = read(fixed_out / "terminal_receipt.json")

    if not (analysis.get("formal") and analysis.get("status") == "complete" and transfer.get("formal") and topology.get("formal")):
        raise ValueError("formal rotating analyses required")
    if not (validation.get("passed") and audit.get("passed") and complete.get("status") == "complete"):
        raise ValueError("formal rotating validation, audit, and training completion must pass")
    if not (fixed.get("formal") and fixed.get("status") == "complete" and fixed.get("control") == "fixed_A"):
        raise ValueError("formal fixed-A analysis required")
    if not (fixed_cross.get("formal") and fixed_cross.get("status") == "complete"):
        raise ValueError("formal fixed-A cross analysis required")
    if not (fixed_validation.get("passed") and fixed_audit.get("passed") and fixed_complete.get("status") == "complete" and fixed_receipt.get("exit_code") == 0):
        raise ValueError("formal fixed-A validation, audit, and receipt must pass")
    if not fixed["paired_with_rotation"]["same_namespace"] or not fixed_cross["paired_with_rotation"]["same_namespace"]:
        raise ValueError("same-namespace paired binding required")

    agg = analysis["aggregate"]
    rotating_transfer = transfer["summary"]
    fixed_transfer = fixed["schedule_summary"]
    rotating_topology = topology["summary"]
    fixed_topology = fixed_cross["summary"]
    comp_a = rotating_transfer["A"]["J"]["mean"]
    comp_b = rotating_transfer["B"]["J"]["mean"]
    comp_c = rotating_transfer["C"]["J"]["mean"]
    fixed_a = fixed_transfer["A"]["target12"]["J"]["mean"]
    paired_rows = fixed["paired_with_rotation"]["rows"]
    paired_summary = fixed["paired_with_rotation"]["summary"]
    seed_delta_a, seed_values_a = seed_level_delta(paired_rows, "A")
    old_replication = transfer["replication_vs_v033"]
    old_delta = sum(row["difference"] for row in old_replication) / len(old_replication)
    agreement_comp = agg["dual_complementary"]["agreement"][-1]["agreement"]["full_message_agreement"]
    fixed_agreement = fixed["agreement"]["fixed_A_endpoint"]["mean"]

    fixed_curve = {}
    for row in fixed["rows"]:
        for item in row["trajectory"]:
            fixed_curve.setdefault(item["update"], []).append(item["scores"]["target12"]["J"])
    rotating_curve = {item["update"]: item["scores"]["target12"]["pooled"]["J"] for item in agg["dual_complementary"]["curve"]}

    lines = [
        "# 伙伴拓扑轮换与跨伙伴迁移：v0.34 同命名空间配对实验报告",
        "",
        "## 研究问题",
        "",
        "上一轮的互补观察实验表明，food sender 和 water sender 可以在同一团队中形成有 grounded 后果的双 token 协议，但协议是否依赖固定伙伴仍不清楚。本轮考察伙伴拓扑变化的两个可能后果：它是否提高对新伙伴组合的可读性，以及这种泛化是否以 native 协议性能为代价。",
        "",
        "## 实验设计与因果对照",
        "",
        "四个主体使用两种冻结的私有视觉接口 [0,1,0,1]。food sender 只看 food-only，water sender 只看 water-only，各发送一个 token；receiver 读取有序 token 对并恢复食物和水的位置。每个拓扑都让四个主体轮流担任 receiver、food sender 和 water sender。A、B、C 是三套伙伴排列；训练中的 rotating 批次在偶数更新使用 A、奇数更新使用 B，C 只在终点评估。",
        "",
        "为隔离伙伴轮换效应，新增同一 34034 fixture 命名空间下的 fixed-A control。它保留相同 seed、坐标面板、world、照片、初始化、优化器、学习率和 2400 更新预算，只把每次更新固定为 A。两批次的 source_hashes 和 input_hashes 完全相同；主要统计单位是 seed，三个坐标面板先在 seed 内平均。",
        "",
        "两批次都使用双发送者互补条件。rotating 批次另外保留 single_full 和 dual_same_full 作为任务结构参照，但它们不进入 fixed-A 因果差分。",
        "",
        "## 结果一：固定伙伴与伙伴轮换的 grounded 性能",
        "",
        f"fixed-A 的 native A 目标12 J 为 **{pct(fixed_a)}**；rotating-A 为 **{pct(comp_a)}**。同一 seed×panel 配对的 rotating−fixed 差值为 {pp(paired_summary['A']['J']['mean'])}（SD {pct(paired_summary['A']['J']['sd'])}），四个 seed 先在面板内平均后的差值为 {pp(seed_delta_a[0])}（SD {pct(seed_delta_a[1])}）。四个 seed 的 seed-level 差值依次为 {', '.join(pp(x) for x in seed_values_a)}。",
        "",
        "这个对照把此前无法解释的跨批次差异固定下来了：在当前训练预算下，轮换并没有提高 native A 的 grounded 成功率，平均反而下降。它更像是一种把单一伙伴协议扩展到多种拓扑的训练折衷，而不是免费的泛化增益。",
        "",
        "| 终点评估拓扑 | fixed-A control | rotating | rotating−fixed |",
        "|---|---:|---:|---:|",
        f"| A（native） | {pct(fixed_transfer['A']['target12']['J']['mean'])} | {pct(comp_a)} | {pp(paired_summary['A']['J']['mean'])} |",
        f"| B（训练见过） | {pct(fixed_transfer['B']['target12']['J']['mean'])} | {pct(comp_b)} | {pp(paired_summary['B']['J']['mean'])} |",
        f"| C（训练未见） | {pct(fixed_transfer['C']['target12']['J']['mean'])} | {pct(comp_c)} | {pp(paired_summary['C']['J']['mean'])} |",
        "",
        f"fixed-A 的 train12、target12 和 held18 J 分别为 {pct(fixed_transfer['A']['train12']['J']['mean'])}、{pct(fixed_transfer['A']['target12']['J']['mean'])} 和 {pct(fixed_transfer['A']['held18']['J']['mean'])}；rotating native A 的对应 target12 和 held18 为 {pct(comp_a)} 和 {pct(agg['dual_complementary']['scores']['held18']['pooled']['J'])}。fixed-A 中保存的 B/C transfer 表与 A 逐项相同；这说明本批次终点在这些排列下产生了相同的输出，但不能据此断言主体学到了抽象拓扑规则。rotating 则出现 A≈B、C 明显下降的结构。",
        "",
        "## 结果二：形式一致率与语义性能分离",
        "",
        f"同私有类型主体在 fixed-A 终点的完整双 token 一致率为 **{pct(fixed_agreement)}**，rotating 为 **{pct(agreement_comp)}**，轮换提高 {pp(agreement_comp - fixed_agreement)}。single_full 和 dual_same_full 的 rotating 一致率分别为 {pct(agg['single_full']['agreement'][-1]['agreement']['full_message_agreement'])} 和 {pct(agg['dual_same_full']['agreement'][-1]['agreement']['full_message_agreement'])}。",
        "",
        "形式上的 token 收敛与 grounded 成功率沿相反方向变化：轮换让不同主体更常发出相同形式，但同时降低 native 目标的联合恢复率。因而本实验不能把字符串一致率单独当作共同语言；至少需要同时报告任务后果和跨伙伴替换后的可读性。",
        "",
        "| 更新 | fixed-A target J（描述性） | rotating target J（描述性） |",
        "|---:|---:|---:|",
    ]
    for update in sorted(fixed_curve):
        lines.append(f"| {update} | {pct(sum(fixed_curve[update]) / len(fixed_curve[update]))} | {pct(rotating_curve[update])} |")

    lines += [
        "",
        "## 结果三：跨团队 token 兼容性",
        "",
        f"fixed-A 的 A within J 为 {pct(fixed_topology['A']['within']['mean'])}，same-type food 替换为 {pct(fixed_topology['A']['same_type_food']['mean'])}，same-type water 替换为 {pct(fixed_topology['A']['same_type_water']['mean'])}，16 种 food/water 跨团队组合的 all-cross J 为 **{pct(fixed_topology['A']['all_cross']['mean'])}**。rotating-A 的对应值为 {pct(rotating_topology['A']['within']['mean'])}、{pct(rotating_topology['A']['same_type_food']['mean'])}、{pct(rotating_topology['A']['same_type_water']['mean'])} 和 **{pct(rotating_topology['A']['all_cross']['mean'])}**。",
        "",
        f"同命名空间配对的 all-cross rotating−fixed 为 {pp(fixed_cross['paired_with_rotation']['summary']['A']['all_cross']['mean'])}（SD {pct(fixed_cross['paired_with_rotation']['summary']['A']['all_cross']['sd'])}），而 native within 差值为 {pp(fixed_cross['paired_with_rotation']['summary']['A']['within']['mean'])}。这给出一个清楚的权衡：轮换提高了不同团队 token 的互相可读性，却降低了每个团队使用自身协议的可靠性。",
        "",
        "| 兼容性指标（A） | fixed-A | rotating | rotating−fixed |",
        "|---|---:|---:|---:|",
        f"| within | {pct(fixed_topology['A']['within']['mean'])} | {pct(rotating_topology['A']['within']['mean'])} | {pp(fixed_cross['paired_with_rotation']['summary']['A']['within']['mean'])} |",
        f"| same-type food | {pct(fixed_topology['A']['same_type_food']['mean'])} | {pct(rotating_topology['A']['same_type_food']['mean'])} | {pp(fixed_cross['paired_with_rotation']['summary']['A']['same_type_food']['mean'])} |",
        f"| same-type water | {pct(fixed_topology['A']['same_type_water']['mean'])} | {pct(rotating_topology['A']['same_type_water']['mean'])} | {pp(fixed_cross['paired_with_rotation']['summary']['A']['same_type_water']['mean'])} |",
        f"| all-cross | {pct(fixed_topology['A']['all_cross']['mean'])} | {pct(rotating_topology['A']['all_cross']['mean'])} | {pp(fixed_cross['paired_with_rotation']['summary']['A']['all_cross']['mean'])} |",
        "",
        f"v0.33 历史固定批次的 all-cross J 为 {pct(topology['v033_fixed_all_cross'])}，但那一比较跨越了 fixture 命名空间，因此只作为背景线索。当前报告的 fixed-A/rotating 差值完全在 34034 命名空间内计算。",
        "",
        "## 机制解释与边界",
        "",
        "在这个两发送者协议中，互补观察提供了形成有意义双 token 的信息条件；固定伙伴让主体可以把同一套配对特定协议优化得更好，伙伴轮换则施加了跨拓扑可读性的压力。结果表明，压力确实能改变协议的社会可迁移性，但短预算下它会与 native 任务性能竞争。当前证据支持“共同符号的形式稳定性、语义后果和社会可迁移性是可分离维度”，不支持已经出现了与伙伴身份无关的公共词典、组合语法或意向性通信。",
        "",
        "四个主体仍是固定人口，私有视觉接口只有两种，拓扑由实验者预先规定，训练同步且没有出生、死亡、迁移或代际替换。C 的失败只说明当前小型受控协议的拓扑泛化有限，不能直接外推到人类语言起源。",
        "",
        "## 结果质量与可复核性",
        "",
        f"rotating 批次的独立 NumPy 重算通过 {validation['checks']:,} 项检查和 {validation['scalar_comparisons']:,} 个标量比较，最大绝对误差为 {validation['maximum_metric_absolute_difference']:.1g}；覆盖 {validation['coverage']['protocol_tables']:,} 张协议表、{validation['coverage']['protocol_worlds']:,} 个协议世界和 {validation['coverage']['null_target_scores']:,} 个结构空参照目标分数。边界审计通过 {audit['checks']:,} 项检查，覆盖 {audit['coverage']['training_trace_rows']:,} 行 trace。",
        f"fixed-A 分析通过 {fixed_validation['checks']:,} 项检查和 {fixed_validation['scalar_comparisons']:,} 个标量比较，最大绝对误差为 {fixed_validation['maximum_metric_absolute_difference']:.1g}；覆盖 {fixed_validation['coverage']['protocol_tables']:,} 张协议表、{fixed_validation['coverage']['transfer_tables']:,} 张 transfer 表和 {fixed_validation['coverage']['protocol_worlds']:,} 个协议世界。固定控制的边界审计通过 {fixed_audit['checks']:,} 项检查，覆盖 {fixed_audit['coverage']['training_trace_rows']:,} 行 trace、{fixed_audit['coverage']['agreement_files']} 个 agreement 文件和 {fixed_audit['coverage']['trace_files']} 个 trace 文件。",
        f"rotating 正式批次包含 {complete['social_runs']} 组运行；fixed-A control 包含 {fixed_complete['social_runs']} 组运行。两者均没有新的私有接口拟合和视觉前向，fixed-A 终端回执记录退出码 {fixed_receipt['exit_code']}。审计没有声称重放所有梯度与 Adam 状态。",
        "",
        "## 下一步实验",
        "",
        "下一轮应把“伙伴轮换”拆成两个因素：伙伴拓扑变化和主体身份变化。保留 fixed-A 与 A/B 轮换的 paired 框架，再加入一个训练期见过同样角色拓扑、但终点更换 sender 或 receiver 身份的留出条件；同时保留 C 这种未见拓扑。这样可以区分协议是否依赖角色位置、私有观察类型或具体主体身份。主指标仍应同时包括 native J、held-out J、all-cross J、token 一致率和形成速度 AUC。",
        "",
        "只有当身份留出仍能维持 grounded 的跨伙伴读出时，才有理由把当前 all-cross 上升解释为更一般的共同符号；如果身份留出失败而形式一致率继续上升，则更可能是类型级模式复用或输出分布收敛。",
        "",
        "## 可复核文件",
        "",
        f"- v0.34 设计：[固定执行方案_v0.34.md]({(ROOT / '固定执行方案_v0.34.md').resolve()})；任务设计：[support_design.json]({(ROOT / 'support_design.json').resolve()})；实现审查：[implementation_review.json]({(ROOT / 'implementation_review.json').resolve()})。",
        f"- rotating 独立统计：[analysis.json]({(out / 'analysis.json').resolve()})；拓扑迁移：[rotation_transfer.json]({(out / 'rotation_transfer.json').resolve()})；跨团队检查：[cross_topology.json]({(out / 'cross_topology.json').resolve()})。",
        f"- fixed-A 独立统计：[fixed_control_analysis.json]({(fixed_out / 'fixed_control_analysis.json').resolve()})；跨团队检查：[fixed_control_cross.json]({(fixed_out / 'fixed_control_cross.json').resolve()})。",
        f"- rotating 门禁：[raw_validation.json]({(out / 'raw_validation.json').resolve()})；rotating 审计：[audit_execution.json]({(out / 'audit_execution.json').resolve()})；fixed-A 门禁：[fixed_control_raw_validation.json]({(fixed_out / 'fixed_control_raw_validation.json').resolve()})；fixed-A 审计：[fixed_control_audit.json]({(fixed_out / 'fixed_control_audit.json').resolve()})。",
        f"- rotating 训练完成：[training_complete.json]({(out / 'training_complete.json').resolve()})；fixed-A 训练完成：[training_complete.json]({(fixed_out / 'training_complete.json').resolve()})；固定控制回执：[terminal_receipt.json]({(fixed_out / 'terminal_receipt.json').resolve()})。",
        f"- 图形：[01_rotation_design.png]({(out / 'figures/01_rotation_design.png').resolve()})、[02_rotation_outcomes.png]({(out / 'figures/02_rotation_outcomes.png').resolve()})；图形 QA：[visual_qa.json]({(out / 'figures/visual_qa.json').resolve()})。",
        f"- 终点消息样本：[实际消息示例.md]({(out / '实际消息示例.md').resolve()})；样本 JSON：[message_examples.json]({(out / 'message_examples.json').resolve()})。",
        "",
        "本报告把 v0.33 只作为历史参照；关于伙伴轮换的主要结论来自同一 34034 命名空间下的 rotating 与 fixed-A 配对结果。",
    ]
    report = out / "伙伴拓扑轮换与跨伙伴迁移研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); build(args.out.resolve())


if __name__ == "__main__": main()
