"""Build the v0.35 identity and online-adaptation report from sealed JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.3f}%"


def pp(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{100 * value:.3f} 个百分点"


def link(path: Path, label: str | None = None):
    return f"[{label or path.name}]({path.resolve()})"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); identity_out = out.parent / "identity_001"
    adaptation = read(out / "adaptation_analysis.json")
    identity = read(identity_out / "identity_analysis.json")
    validation = read(out / "adaptation_raw_validation.json")
    audit = read(out / "adaptation_audit.json")
    training = read(out / "training_complete.json")
    invocation = read(out / "invocation.json")
    if not (adaptation["formal"] and adaptation["status"] == "complete" and identity["formal"] and identity["status"] == "complete" and validation["passed"] and audit["passed"] and training["formal"]):
        raise ValueError("formal identity and adaptation outputs are required")

    lines = []
    lines += [
        "# 新主体的共同符号获取与拓扑迁移：v0.35 身份留出—在线适应实验报告",
        "",
        "## 研究问题",
        "",
        "v0.34 说明伙伴拓扑轮换会提高跨团队 token 兼容性，同时牺牲固定拓扑下的 native 性能。本轮把“具体主体身份”单独拿出来：一个与居民共享私有视觉类型、但通信模块被重新初始化的新主体，能否零样本读懂居民协议？如果不能，它能否只靠在线任务反馈学会协议？伙伴轮换是否改变这种社会学习的速度、最终性能和未见拓扑迁移？",
        "",
        "本轮检验三个可区分的命题：零样本可读性、在线文化获取和拓扑覆盖带来的迁移—性能权衡。这里的“语言”仍严格指有 grounded 后果的离散 token 协议，不把字符串一致率直接等同于自然语言或公共词典。",
        "",
        "## 实验条件",
        "",
        "四个主体的私有类型为 `[0, 1, 0, 1]`。food sender 只看 food-only 视图，water sender 只看 water-only 视图，各发送一个 7 值 token；receiver 读取有序的 food-token—water-token 对，并分别输出食物和水的位置。收益为 `.25*(cF+cW)+.5*(cF*cW)`，其中 `cF`、`cW` 是两个位置是否都恢复正确。A、B、C 是三套伙伴排列；fixed-A 每一步训练 A，rotating-AB 在偶数步训练 A、奇数步训练 B，C 只在终点评估。",
        "",
        "居民从 v0.34 同命名空间的 fixed-A 或 rotating-AB 终点导入并冻结。新主体保留相同私有视觉编码器类型，只重置通信模块；在线适应时只优化新主体通信参数 600 次，居民、私有视觉编码器、世界和测试集均不更新。正式矩阵为 4 个 seed × 3 个坐标面板 × 2 个训练条件 × 2 个留出身份，共 48 个运行；在更新 0、100、300、600 评估 A/B/C。",
        "",
        f"形式执行记录见 {link(out / 'invocation.json')}，训练完成记录见 {link(out / 'training_complete.json')}。",
        "",
        "## 结果一：零样本身份留出几乎不能读出居民协议",
        "",
        "零样本实验只替换一个角色模型，其他角色保持居民终点模型。下表是 target12 的联合 J 均值（括号内为跨运行 SD）；均匀随机选择两个位置的理论联合基线约为 2.778%。",
        "",
        "| 训练条件 | 终点评估 | receiver newcomer | food sender newcomer | water sender newcomer |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in identity["conditions"]:
        for schedule in identity["schedules"]:
            values = []
            for role in identity["roles"]:
                x = identity["summary"][condition][schedule][role]["newcomer"]["target12"]["J"]
                values.append(f"{pct(x['mean'])}（{pct(x['sd'])}）")
            lines.append(f"| {condition} | {schedule} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "fixed-A 的三个角色在 A 中只有 3.646%–8.681% 的 J，在 B/C 中约 1%–3%；rotating-AB 的 A/B 约 4%–6%，C 约 4%。这与居民已经形成的 29%–41% 级别目标性能相差很大，说明伙伴轮换带来的形式或跨团队兼容性不会自动变成“新主体一看就懂”的公共协议。",
        "",
        f"零样本原始统计：{link(identity_out / 'identity_analysis.json')}；独立审计：{link(identity_out / 'identity_audit.json')}。",
        "",
        "## 结果二：在线适应能获取居民协议，但获取对象由训练拓扑决定",
        "",
        "下表报告新主体目标 J 的初始均值、600 次更新终点均值和离散学习曲线 AUC。每个单元包含 24 个运行（4 seed × 3 panel × 2 held identity）。",
        "",
        "| 训练条件 | 评估拓扑 | 初始 J | 600 更新 J | AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in adaptation["conditions"]:
        for schedule in adaptation["schedules"]:
            x = adaptation["summary"][condition][schedule]
            lines.append(f"| {condition} | {schedule} | {pct(x['initial_target_J']['mean'])}（{pct(x['initial_target_J']['sd'])}） | {pct(x['final_target_J']['mean'])}（{pct(x['final_target_J']['sd'])}） | {pct(x['auc_target_J']['mean'])} |")
    lines += [
        "",
        "fixed-A newcomer 在 A 上从 13.166% 上升到 45.920%，100 次更新已经达到 42.925%；但 B、C 在 600 次更新仍为 2.387% 和 1.577%，没有形成可迁移的抽象协议。rotating-AB newcomer 在 A、B 分别达到 31.105% 和 31.829%，C 达到 14.815%；A/B 的获得都明显高于随机，C 也高于零样本但仍低于训练见过的拓扑。",
        "",
        "这种曲线把两个能力分开：新主体可以通过 grounded 反馈学习 resident code，但固定训练只教会它一套局部协议；训练时覆盖 A/B 会降低 A 的专门化上限，却把一部分能力带到 B，并改善未见 C 的读出。",
        "",
        "## 结果三：轮换相对固定的配对差异",
        "",
        "在相同 seed、面板和 held identity 上配对比较 rotating-AB 与 fixed-A 的终点差异：",
        "",
        "| 评估拓扑 | rotating−fixed 终点 J |",
        "|---|---:|",
    ]
    for schedule in adaptation["schedules"]:
        x = adaptation["paired_rotating_minus_fixed"][schedule]["600"]
        lines.append(f"| {schedule} | {pp(x['mean'])}（SD {pct(x['sd'])}）|")
    lines += [
        "",
        "A 的差异为 −14.815 个百分点，B 为 +29.442 个百分点，C 为 +13.238 个百分点。因而“轮换是否更好”没有单一答案：它损害 native A 的适应结果，却显著改善新主体对 B 和未见 C 的可读性。该对照在同一 v0.34-34034 世界、照片、私有类型和输出接口内完成。",
        "",
        "## 结果四：私有类型不是主要解释",
        "",
        "rotating-AB 条件下，held identity 的私有类型为 0 或 1。600 更新后的 A/B/C J 均值分别为：",
        "",
        "| 私有类型 | A | B | C |",
        "|---|---:|---:|---:|",
    ]
    for private_type in ("0", "1"):
        vals = [pct(adaptation["private_type_summary"]["rotating_AB"][private_type][schedule]["final_target_J"]["mean"]) for schedule in adaptation["schedules"]]
        lines.append(f"| {private_type} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += [
        "",
        "两种类型的差异小于拓扑条件差异（例如 C 为 14.265% 对 15.365%）。在当前任务中，社会训练拓扑比这两个私有视觉类型更能解释新主体最终读出的协议范围；这只是本矩阵内的描述性结果，不能推出视觉表征完全不重要。",
        "",
        "## 机制解释",
        "",
        "1. **零样本身份泛化不足。** 共享私有视觉类型并不能让新主体直接进入居民符号系统；通信头的社会历史是必要变量。",
        "2. **社会学习是可行的。** 新主体在固定居民不变的情况下，仅靠任务反馈就能把一个离散协议重新对齐到 resident sender/receiver 的 grounded 后果。",
        "3. **拓扑覆盖塑造可迁移性。** A/B 轮换把优化目标从单一团队协议改成多个伙伴排列共同可读的协议，代价是 A 的专门化性能；对 C 的提升说明覆盖压力会改变协议的外推结构，但 C 仍没有达到 A/B。",
        "4. **形式一致、任务成功和新主体可读性应分开报告。** v0.34 已显示 token 形式一致率可以上升而 grounded 性能下降；v0.35 进一步显示 resident 之间的兼容性也不等于 newcomer 的零样本可读性。",
        "",
        "## 与“语言诞生”问题的关系",
        "",
        "本轮还没有模拟从无到有的群体共创，也没有让主体自己选择是否通信。它回答的是更窄但可检验的机制问题：在已有 grounded 协议的社会中，何种社会经历足以让新主体获取该协议，以及训练拓扑如何决定协议的可迁移范围。这个结果为后续的代际实验提供了基线：若新主体的通信模块从零开始，必须把“接入既有协议”和“群体共同发明协议”分开，否则形成成功可能只是从教师复制而来。",
        "",
        "## 限制",
        "",
        "- 模型是小型视觉—通信模块机制探针，不是大规模开源 VLM/LLM；本轮没有声称具有人类世界知识或自然语言能力。",
        "- 居民是 v0.34 的固定终点策略；没有出生、死亡、人口替换、代际瓶颈或文化漂移。",
        "- 新主体保留私有视觉编码器，只重置通信模块；因此不能测量从零学习知觉，也不能把结果解释成端到端语言能力。",
        "- 任务只有两个资源、两个发送 token、一个 receiver 和六个位置，不涉及词汇增长、组合语法、指称意向、协商成本或时间延迟。",
        "- A/B/C 拓扑、food-only/water-only 观察掩码和收益函数均由实验者设定；当前结果是受控因果探针，不是人类语言起源的直接重建。",
        "- 审计重放了端点表、离散采样 trace、奖励算术、训练日程和文件哈希，没有重放每一步 Adam 状态。",
        "",
        "## 可复核性",
        "",
        f"在线适应独立重算通过 {adaptation['checks']} 项检查和 {adaptation['scalar_comparisons']} 个标量比较，最大指标误差为 {adaptation['maximum_metric_absolute_difference']}；覆盖 2,304 张协议表。边界审计通过 {audit['checks']} 项检查，覆盖 {audit['coverage']['trace_rows']} 行 trace、{audit['coverage']['trace_files']} 个 trace 文件。",
        "",
        f"图形：{link(out / 'figures/01_adaptation_outcomes.png')}；图形源数据：{link(out / 'figures/figure_source.json')}；视觉 QA：{link(out / 'figures/visual_qa.json')}。",
        f"在线适应分析：{link(out / 'adaptation_analysis.json')}；独立重算：{link(out / 'adaptation_raw_validation.json')}；轨迹审计：{link(out / 'adaptation_audit.json')}。",
        f"零样本分析：{link(identity_out / 'identity_analysis.json')}；零样本审计：{link(identity_out / 'identity_audit.json')}。",
        f"设计文件：{link(ROOT / 'adaptation_design.json')}；执行源文件：{link(ROOT / 'newcomer_adaptation.py')}。",
        "",
        "## 下一步实验",
        "",
        "下一轮应把“已有协议适应”推进到真正的文化传递：先让一代居民在 grounded 任务中形成协议，再逐个替换居民；每一代只允许新主体通过有限 episode 观察和反馈更新通信模块，并将其作为下一代的唯一社会输入。需要同时保留 fixed-A、rotating-AB 和随机拓扑三种教师历史，记录代际保真度、协议漂移、跨身份可读性和新词/组合结构的增长。只有在没有原始教师直接参与的代际中仍能保持 grounded 传递，才可以把结果解释为文化演化，而不只是一次性适应。",
        "",
        "本报告的总体结论仍是机制性和条件性的：共同符号的获取需要社会学习；拓扑轮换会把协议从局部最优推向更宽的可迁移范围，但会牺牲固定伙伴下的专门化性能。",
    ]
    report = out / "新主体身份留出与在线适应研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
