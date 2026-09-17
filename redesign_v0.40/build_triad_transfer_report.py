"""Build the Chinese v0.40 triad transfer research report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMATIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
FORMATION_LABELS = {
    "origin_fixed_A": "形成固定 A",
    "origin_rotating_AB": "形成轮换 A/B",
    "origin_random_ABC": "形成随机 A/B/C",
}
TRANSMISSION_LABELS = {"fixed_A": "传递固定 A", "rotating_AB": "传递轮换 A/B", "random_ABC": "传递随机 A/B/C"}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x):
    return f"{100 * x:.3f}%"


def pp(x):
    return f"{100 * x:+.3f} 个百分点"


def link(path: Path, label: str | None = None):
    return f"[{label or path.name}]({path.resolve()})"


def mean_sd(item):
    return f"{pct(item['mean'])}（SD {pct(item['sd'])}）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    d = read(out / "triad_transfer_analysis.json")
    v = read(out / "triad_transfer_raw_validation.json")
    a = read(out / "triad_transfer_audit.json")
    q = read(out / "visual_qa.json")
    t = read(out / "training_complete.json")
    i = read(out / "invocation.json")
    design = read(ROOT / "triad_transfer_design.json")
    if not (
        d["formal"]
        and d["status"] == "complete"
        and v["passed"]
        and a["status"] == "complete"
        and q["status"] == "passed_visual_qa"
        and t["formal"]
        and i["formal"]
    ):
        raise ValueError("formal triad transfer outputs required")

    effects = d["condition_effects"]
    summary = d["summary"]
    lines = [
        "# 三资源三 token 共同协议的代际传递：v0.40 研究报告",
        "",
        "## 研究问题",
        "",
        "前一轮 v0.39 从随机通信头形成了三资源、三 token 的有限 grounded 协议，但还没有回答这种协议能否在主体替换后继续存在。本轮把每个形成终点放入四次连续替换：每一代移除一个主体，加入一个没有通信知识的新主体，只允许该主体通过共享收益学习 300 次更新。核心问题是：形成时的伙伴覆盖和传递时的伙伴覆盖，是否共同决定协议的跨代保真度、资源槽位一致性与符号漂移。",
        "",
        "这里的“协议”指有限的三槽位共同通信系统，不等同于人类语言。联合成功、逐资源成功率和 token 形式一致率分别衡量功能、槽位功能和形式收敛；三者分开报告，避免把相同 token 或常量输出误判为语言能力。",
        "",
        "## 实验设计",
        "",
        "v0.40 使用 v0.39 的三个形成终点（形成固定 A、形成轮换 A/B、形成随机 A/B/C）作为代际 0。每个形成终点分别接入固定 A、轮换 A/B、随机 A/B/C 三种传递日程，构成 3×3 因子矩阵。正式矩阵有 4 个随机种子、3 个坐标面板和 9 个形成×传递组合，共 108 条链；每条链依次替换主体 0、1、2、3，得到代际 0–4。",
        "",
        "每个主体的私有视觉前端与 v0.39 相同并冻结。新主体的发送器和接收器通信模块全部重新初始化，只训练这些通信模块；居民主体不更新。三种评估拓扑 A、B、C 都在每一代运行，测试覆盖 120 个三地点有序地图，其中每个 partition 的 60 个地图作为训练集合、其余 60 个作为 target 集合。每个资源 sender 发送一个 7 值 token，receiver 读取长度为 3 的有序 tuple（343 个可能码），收益为 `(1/6)*sum(correct_resources)+0.5*all_three_correct`。",
        "",
        f"设计文件：{link(ROOT / 'triad_transfer_design.json')}；执行绑定：{link(out / 'invocation.json')}；训练完成记录：{link(out / 'training_complete.json')}。",
        "",
        "## 主要结果一：形成覆盖决定协议是否能被传递",
        "",
        "下面的数值是把三种评估拓扑、12 个 seed×panel 运行合并后的 target-60 联合 J。代际 0 是 v0.39 的形成终点，代际 4 已完成四次主体替换。括号内为跨链 SD。",
        "",
        "| 形成条件 | 传递条件 | 代际 0 | 代际 4 | 变化 |",
        "|---|---|---:|---:|---:|",
    ]
    for formation in FORMATIONS:
        for transmission in TRANSMISSIONS:
            x0 = effects[formation][transmission]["0"]["mean"]
            x4 = effects[formation][transmission]["4"]["mean"]
            delta = x4 - x0
            lines.append(
                f"| {FORMATION_LABELS[formation]} | {TRANSMISSION_LABELS[transmission]} | {mean_sd(effects[formation][transmission]['0'])} | {mean_sd(effects[formation][transmission]['4'])} | {pp(delta)} |"
            )
    lines += [
        "",
        "形成固定 A 的协议在传递固定 A 时基本保持（10.579%→10.648%），一旦新主体在轮换或随机拓扑中学习，联合 J 降到 4.479% 或 3.912%。形成轮换 A/B 的协议在轮换和随机传递下略有提升（21.019%→22.697%/23.542%）；形成随机 A/B/C 的协议在三种传递下都保持或提升（26.806%→27.465%–29.097%）。这说明“能否形成”与“能否传递”不是同一个属性：形成阶段覆盖的伙伴关系越广，协议越不依赖单一角色排列。",
        "",
        "## 主要结果二：A、B、C 评估揭示了关系特定协议的瓦解",
        "",
        "下表给出代际 4 的 target-60 联合 J；每个单元格仍是 12 个 seed×panel 运行的均值（SD）。",
        "",
        "| 形成条件 | 传递条件 | 评估 A | 评估 B | 评估 C |",
        "|---|---|---:|---:|---:|",
    ]
    for formation in FORMATIONS:
        for transmission in TRANSMISSIONS:
            vals = [summary[formation][transmission]["4"][s]["target_J"] for s in SCHEDULES]
            lines.append(
                f"| {FORMATION_LABELS[formation]} | {TRANSMISSION_LABELS[transmission]} | {mean_sd(vals[0])} | {mean_sd(vals[1])} | {mean_sd(vals[2])} |"
            )
    lines += [
        "",
        "形成固定 A 的起点在 A 评估下有约 30.45% 的 target J，但 B/C 评估接近零；四代后固定传递仍保持这一关系特定性能，而轮换/随机传递会把 A 的性能带到接近随机水平。形成轮换 A/B 的起点在 A/B 上都可读，在随机传递下三种评估的差距变小但总体仍低于随机形成。形成随机 A/B/C 的协议在代际 4 仍保持较接近的三拓扑读出。",
        "",
        "## 主要结果三：形式一致性随伙伴覆盖而上升，但功能仍有组合瓶颈",
        "",
        "代际 4 的三 token 联合一致率按三种评估拓扑平均如下；它要求同一资源的同类型 sender 对在 120 个测试地图上同时使用相同 token。",
        "",
        "| 形成条件 \\ 传递条件 | 三 token 联合一致率 | 资源 0 | 资源 1 | 资源 2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for formation in FORMATIONS:
        for transmission in TRANSMISSIONS:
            ag = [summary[formation][transmission]["4"][s]["agreement_joint"] for s in SCHEDULES]
            ar = [
                [summary[formation][transmission]["4"][s][f"agreement_resource{k}"] for s in SCHEDULES]
                for k in range(3)
            ]
            lines.append(
                f"| {FORMATION_LABELS[formation]} \\ {TRANSMISSION_LABELS[transmission]} | {pct(sum(x['mean'] for x in ag)/3)} | {pct(sum(x[0]['mean'] for x in ar)/3)} | {pct(sum(x[1]['mean'] for x in ar)/3)} | {pct(sum(x[2]['mean'] for x in ar)/3)} |"
            )
    lines += [
        "",
        "随机形成的三 token 联合一致率在代际 4 达到约 67.1%–71.3%，轮换形成约 14.6%–23.8%，固定形成接近 0%。形式一致率与 target J 同向变化，但并不相等：相同 token 只有在 receiver 的组合解码正确时才有功能。",
        "",
        "代际 4 的 target-60 逐资源准确率（按 A/B/C 评估平均）如下。",
        "",
        "| 形成条件 | 传递固定 A | 传递轮换 A/B | 传递随机 A/B/C |",
        "|---|---:|---:|---:|",
    ]
    for formation in FORMATIONS:
        cells = []
        for transmission in TRANSMISSIONS:
            vals = [
                sum(summary[formation][transmission]["4"][s][f"resource{k}"]["mean"] for s in SCHEDULES) / 3
                for k in range(3)
            ]
            cells.append("/".join(f"{100*x:.1f}%" for x in vals))
        lines.append(f"| {FORMATION_LABELS[formation]}（资源 0/1/2） | " + " | ".join(cells) + " |")
    lines += [
        "",
        "资源 2 在多数条件下准确率最高，而资源 0/1 更容易成为联合瓶颈。这是当前模型与图像特征初始化共同造成的结果，不能直接解释为某种人类语言的语法层级；它提示后续必须做资源置换和对称性控制。",
        "",
        "## 机制解释",
        "",
        "1. **伙伴覆盖产生传递鲁棒性。** 固定 A 形成的协议可以把角色关系编码进 token—receiver 配对，因此面对新拓扑时迅速失配。轮换和随机形成迫使同一槽位在多种主体关系中可读，代际替换后更不容易崩溃。",
        "2. **代际学习不只是复制符号。** 新主体的通信头完全随机，必须利用共享 grounded 反馈恢复现有槽位。固定传递下它可以成为一个关系特定的替代者；轮换/随机传递下则需要恢复跨伙伴的公共映射。",
        "3. **三槽位组合放大了功能瓶颈。** 逐资源准确率可以达到中高水平，但联合 J 仍低于逐资源平均值，因为 receiver 必须同时正确解码 343 个组合码。",
        "4. **形式与功能需同时观察。** 随机形成条件的高 token 一致率伴随更高 target J，说明这轮不是单纯的常量收敛；但二者仍有明显差距，因此不能只用符号相似度定义共同语言。",
        "",
        "## 对核心问题的回答",
        "",
        "在当前受控模型中，已经形成的有限共同符号系统能否跨主体传递，主要取决于形成阶段是否覆盖多种伙伴拓扑，以及传递阶段是否继续施加这种覆盖压力。结果支持一个可检验的机制假设：非语言的视觉表征、离散符号采样、共享后果反馈和重复伙伴交互，足以支持有限槽位的共同符号与代际重建；伙伴覆盖不足时，系统仍能在局部关系中取得成功，却无法维持公共协议。这个结论只适用于本实验定义的 grounded 三 token 协议，不能外推为人类语言起源的充分条件。",
        "",
        "## 限制",
        "",
        "- 代际 0 不是从零形成，而是 v0.39 的三资源形成终点；本轮研究的是传递与漂移。",
        "- 私有视觉前端继承自 v0.28 的图像特征训练并冻结，尚未测试视觉表征本身的共同学习或主体间差异。",
        "- 每个资源 sender 只有一个 token，未涉及同一主体内的序列语法、组合词法、指称扩展或开放词汇。",
        "- 共享奖励、批量采样和梯度更新仍由实验者集中实现；尚未模拟完全分布式、异步和带噪声的社会学习。",
        "- 资源槽位的准确率不对称，说明还需要交换资源标签、随机化角色顺序和增加独立视觉种类来排除模型偏置。",
        "",
        "## 可复核性",
        "",
        f"正式执行包含 {t['runs']} 条链、{t['replacement_events']} 次替换、{t['runs']*5*3*4} 张协议表。独立 NumPy 重分析通过 {d['checks']} 项结构与指标检查、{d['scalar_comparisons']} 个标量比较，最大保存指标差异为 {d['maximum_metric_absolute_difference']}；轨迹审计通过 {a['checks']} 项检查，覆盖 {a['trace_files']} 个轨迹文件和 {a['trace_rows']} 行，最大重放误差为 {a['maximum_replay_absolute_difference']}。",
        "",
        f"结果图：{link(out / 'figures/01_triad_transfer_outcomes.png')}；视觉 QA：{link(out / 'visual_qa.json')}；图形元数据：{link(out / 'plot_metadata.json')}。",
        f"独立分析：{link(out / 'triad_transfer_analysis.json')}；重分析验证：{link(out / 'triad_transfer_raw_validation.json')}；轨迹审计：{link(out / 'triad_transfer_audit.json')}。",
        f"训练源：{link(ROOT / 'triad_transfer_train.py')}；分析源：{link(ROOT / 'triad_transfer_analysis.py')}；审计源：{link(ROOT / 'triad_transfer_audit.py')}。",
        "",
        "## 下一步",
        "",
        "下一轮应先做资源和角色的完全对称化，并把每个 sender 扩展为两个 token：第一 token 检验槽位稳定性，第二 token 检验同一主体内是否出现可复用的顺序结构。随后可在同一形成×传递矩阵中加入新资源、新主体和通信噪声，比较组合容量、伙伴覆盖与代际保真度的交互。只有在这些控制通过后，才适合讨论更接近原始社会的分工任务和更开放的符号结构。",
    ]
    report = out / "三资源三token代际传递研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report), "design_version": design["version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
