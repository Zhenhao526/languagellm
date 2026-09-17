"""Build the Chinese v0.45 newcomer social-learning report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CULTURES = (
    "static_role__fixed_A",
    "static_role__rotating_AB",
    "static_role__random_ABC",
    "random_role__fixed_A",
    "random_role__rotating_AB",
    "random_role__random_ABC",
)
LABELS = {
    "static_role__fixed_A": "静态/A",
    "static_role__rotating_AB": "静态/A-B",
    "static_role__random_ABC": "静态/A-B-C",
    "random_role__fixed_A": "随机/A",
    "random_role__rotating_AB": "随机/A-B",
    "random_role__random_ABC": "随机/A-B-C",
}
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
UPDATES = (0, 100, 300)


def read(path: Path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def item(summary, culture, update, metric):
    return summary[culture][str(update)][metric]


def fmt(summary, culture, update, metric, digits=2):
    value = item(summary, culture, update, metric)
    return f"{pct(value['mean']):.{digits}f}%（SD {pct(value['sd']):.{digits}f}%）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "newcomer_adaptation_analysis.json")
    audit = read(out / "newcomer_adaptation_audit.json")
    complete = read(out / "training_complete.json")
    summary = data["summary"]
    report = out / "陌生主体社会学习与协议恢复研究报告.md"

    lines = [
        "# v0.45 陌生主体社会学习与协议恢复：共同符号的可继承性",
        "",
        "## 摘要",
        "",
        "v0.44 的零适应实验表明，独立训练的文化会形成各自的离散码本，陌生 sender 与 resident receiver 之间几乎不能直接互读。本轮把问题推进到真正的社会学习：保留一个 v0.43 已形成的 resident 文化，清空一个 newcomer 的通信模块，只让它在 resident 共同回报下训练 300 次更新；视觉前端、resident 参数和任务世界保持不变。正式批次包含 432 条链，覆盖 4 个 seed、3 个视觉 partition、6 种资源列排列、6 类 resident 文化和 3 个检查点。",
        "",
        "新人确实从低兼容性起点开始，并在反馈中恢复功能。身份角色排列的等变 target-60 联合 J 在 update 0 为 1.62%–7.01%，update 300 为 9.29%–43.69%。随机角色顺序的 resident 文化平均恢复水平最高，且角色 spread 更稳定：固定 A、轮换 A/B、随机 A/B/C 分别达到 43.69%、42.79%、39.38%，而静态角色顺序分别只有 9.29%、22.02%、27.77%。角色 spread 同时呈现相反变化：静态文化在 update 300 达到 37.22–70.60 个百分点，随机角色文化为 8.19–16.86 个百分点。",
        "",
        "这轮结果支持一个更具体的机制判断：共同后果反馈可以使陌生主体恢复 resident 协议的功能，但角色随机化决定了恢复后的协议是否跨角色排列保持稳定。功能恢复、表面 token 一致性和位置编码并不同步；因此“新人学会做对任务”不能直接等同于“新人复刻了同一套符号形式”。",
        "",
        "## 研究问题和预注册式假设",
        "",
        "本轮问：一个通信头被重置的陌生主体，能否只凭共享任务后果恢复已形成的共同符号？resident 形成时的角色顺序和伙伴拓扑会否改变恢复速度、角色对称性及消息形式的一致性？",
        "",
        "假设为：",
        "",
        "- newcomer 在 update 0 应接近 v0.44 的跨文化零适应基线，而不是 resident native endpoint；",
        "- resident 形成时使用随机角色顺序的文化应具有更小的角色 spread，并给新人更稳定的恢复；",
        "- 伙伴覆盖更广的 resident 文化可能提供更多可观察的约定，但本轮把适应阶段统一固定为随机 A/B/C，以隔离 resident 文化因素。",
        "",
        "## 实验设计",
        "",
        "每条链从 v0.43 的 resident endpoint 开始。resident 包含 3 个冻结通信主体；identity 0 被替换为 newcomer。newcomer 复制同一视觉组的基础主体，但只重置 sender/receiver 通信模块，视觉前端继承并冻结；因此本轮测量的是通信协议的社会恢复，不是从像素学习新概念。newcomer 的 sender 和 receiver 通信参数可训练，三个 resident 的全部参数冻结。",
        "",
        f"形成阶段的 6 类 resident 文化为 `{', '.join(CULTURES)}`：静态或随机角色顺序 × 固定 A、轮换 A/B、随机 A/B/C 伙伴拓扑。适应阶段对所有文化统一使用随机 A/B/C 伙伴日程，每条链训练 {complete['updates_per_run']} 次更新，检查点为 {', '.join(map(str, complete['checkpoints']))}。每个通信 sender 看到一个资源图片并输出 2 个离散 token（词表 7），receiver 根据 3 组 token 预测 3 个资源的站点（6 个站点）；联合回报为 `(1/6)×正确资源数 + 0.5×三个资源全对`。",
        "",
        f"正式因子为 4 seed × 3 partition × 6 资源列排列 × 6 resident 文化 = {complete['runs']} 条链。每个检查点保存 A/B/C 三种评估拓扑、4 个 team 和 6 种角色排列；主要统计固定身份排列 `012`，同时用同步置换 target 资源轴的 equivariant J 检查角色关系。设计文件：[newcomer_adaptation_design.json]({(ROOT / 'newcomer_adaptation_design.json').resolve()})；训练绑定：[invocation.json]({(out / 'invocation.json').resolve()})。",
        "",
        "## 结果一：陌生主体从低基线开始并恢复联合功能",
        "",
        "下表为身份角色排列 `012` 的等变 target-60 联合 J。均值跨该文化的 seed、partition、资源列排列、3 个评估拓扑和 4 个 team 汇总；SD 反映这些链和评估单元的离散程度。",
        "",
        "| resident 文化 | update 0 | update 100 | update 300 | 0→300 增益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for culture in CULTURES:
        j0 = item(summary, culture, 0, "identity_equivariant_target60_J")["mean"]
        j3 = item(summary, culture, 300, "identity_equivariant_target60_J")["mean"]
        lines.append(
            f"| {LABELS[culture]} | {fmt(summary, culture, 0, 'identity_equivariant_target60_J')} | {fmt(summary, culture, 100, 'identity_equivariant_target60_J')} | {fmt(summary, culture, 300, 'identity_equivariant_target60_J')} | {pct(j3-j0):.2f} 个百分点 |"
        )
    lines += [
        "",
        "六类文化都比 update 0 有功能增益，说明共享回报确实给 newcomer 提供了可用的社会学习信号。增益并不只由适应阶段的随机伙伴日程决定，因为所有链都使用同一适应日程；更合理的解释是 resident endpoint 本身携带了不同的角色和拓扑结构。随机角色/A 的最终 J 最高（43.69%），随机角色/A-B 与随机角色/A-B-C 也达到 42.79% 和 39.38%。静态角色/A 的恢复最弱，只从 1.62% 到 9.29%。",
        "",
        "update 0 的低分也是本轮设计的关键对照：newcomer 与 resident 共享视觉前端和任务世界，但通信头被重新初始化，因此视觉相同并不会自动带来码本兼容性。这与 v0.44 的跨独立文化 all-sender 零适应结果方向一致。",
        "",
        "## 结果二：随机角色 resident 让恢复后的协议更稳定",
        "",
        "role spread 定义为同一评估单元中六种角色排列的等变 target-60 J 的最大值减最小值。它越小，说明协议越少依赖某一个 sender—资源角色排列。",
        "",
        "| resident 文化 | update 0 | update 100 | update 300 |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(f"| {LABELS[culture]} | {fmt(summary, culture, 0, 'role_spread_target60_J')} | {fmt(summary, culture, 100, 'role_spread_target60_J')} | {fmt(summary, culture, 300, 'role_spread_target60_J')} |")
    lines += [
        "",
        "静态角色 resident 在适应后出现很大的角色扩散：轮换 A/B 和随机 A/B/C 分别为 69.46 和 70.60 个百分点；随机角色 resident 则保持在 8.19–16.86 个百分点。这个差异比最终 J 的差异更直接地揭示了形成压力的作用：随机角色不是简单地把所有文化都推到更高分，而是让新人学到一个在角色轴变化下更均匀的协议。",
        "",
        "需要谨慎解读静态文化的高 spread。它表示新人可以在某些角色排列上取得较高分，同时在另一些排列上失败；它不表示形成了一个更复杂或更像自然语言的系统。",
        "",
        "## 结果三：功能恢复与表面符号复刻存在时间差",
        "",
        "下面比较 newcomer 与 resident sender 在同一资源、同一测试世界上的双 token pair agreement，并报告 newcomer 双 token 与其负责位置之间的 NMI。pair agreement 要求两个 token 同时相同；独立随机双 token 的机会水平约为 1/49=2.04%。",
        "",
        "| resident 文化 | pair agreement（0） | pair agreement（300） | pair-position NMI（0） | pair-position NMI（300） |",
        "|---|---:|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(
            f"| {LABELS[culture]} | {fmt(summary, culture, 0, 'newcomer_resident_pair_agreement')} | {fmt(summary, culture, 300, 'newcomer_resident_pair_agreement')} | {fmt(summary, culture, 0, 'newcomer_pair_position_nmi')} | {fmt(summary, culture, 300, 'newcomer_pair_position_nmi')} |"
        )
    lines += [
        "",
        "随机角色/A-B-C 的 pair agreement 从 4.63% 上升到 42.16%，随机角色/A-B 从 5.56% 上升到 35.44%；这说明适应不仅提高了任务分数，也使 newcomer 的表面消息更接近 resident。固定 A 条件的 pair agreement 仍接近 5.25%，但随机角色/A 的任务 J 已达到 43.69%，显示功能恢复不必依赖逐 token 复刻。",
        "",
        "pair-position NMI 在 update 0 已经较高，并在许多文化接近 1。它不能单独当作“新人学会了语义”的证据：有限的六站点任务、离散 token 和确定性网络可以在形式一致性很低时仍保留位置相关信息；NMI 还可能受到低熵消息和任务查表结构影响。更稳妥的结论是，功能、形式和位置统计需要分开报告，不能用任一指标替代共同符号本身。",
        "",
        "## 结果四：resident 文化的角色模式比拓扑标签更强地影响新人恢复",
        "",
        "在同一适应日程下，静态角色三种拓扑的 update-300 J 为 9.29%、22.02%、27.77%；随机角色三种拓扑为 43.69%、42.79%、39.38%。因此随机角色模式在三种 resident 拓扑下都保持高恢复，拓扑从固定 A 扩展到 A/B/C 并没有同等幅度的额外收益。",
        "",
        "这不是拓扑无关性的证明。适应阶段的伙伴日程被固定为随机 A/B/C，拓扑因素只存在于 resident 形成阶段；本轮能够支持的是：在给定的形成和适应设置中，角色随机化与 newcomer 恢复之间的差异比 resident 拓扑标签更明显。要估计独立的拓扑主效应，下一轮必须把适应阶段也完整交叉固定、轮换和随机伙伴日程，并保持更新量与随机预算一致。",
        "",
        "## 对核心问题的回答",
        "",
        "这轮首次把“共同符号能否被陌生主体继承”从 endpoint 兼容性变成了反馈驱动的学习问题。结果表明：",
        "",
        "1. 非语言视觉表征和共同回报足以给 newcomer 提供恢复信号；新人可以在不共享通信初始化的情况下提高联合任务成功率。",
        "2. resident 形成时的角色随机化改变了恢复后的结构稳定性。随机角色压力让新人得到更低的角色 spread，而静态角色文化容易让新人学到固定槽位约定。",
        "3. 功能恢复不等同于符号形式复刻。某些文化的 J 提升远高于 pair agreement，说明新人可能形成了可工作的等价或折衷码本；这正是后续需要研究的“社会学习如何塑造公共符号”的入口。",
        "",
        "因此，本轮还不能称为从零产生了自然语言。它给出的是一个更窄而可检验的结果：在有限 grounded coordination task 中，共同后果反馈可以把陌生通信主体拉回 resident 协议的功能吸引域，形成期的角色对称性决定这个吸引域对角色变化的稳定程度。",
        "",
        "## 限制",
        "",
        "- 适应阶段只使用随机 A/B/C 日程，不能单独识别适应拓扑的因果效应；",
        "- newcomer 只有 identity 0，且视觉前端继承同一视觉组并被冻结，未测试新感知主体、感知噪声或视觉概念学习；",
        "- 任务为六站点、三资源、双 token、7 类词表的有限协议，存在查表式解；没有开放词汇、指称扩展、组合生产性或真实时间延迟；",
        "- resident 参数完全冻结，只有 newcomer 通信模块更新；没有模拟 resident 对新成员的适应、谈判、冲突或共同修复；",
        "- reward 是集中式即时联合回报，没有资源库存、互补分工、生存压力或代际瓶颈，因此不能推出分工是人类语言的必要条件；",
        "- SD 较大，尤其是随机角色/A 的端点 J，后续需要以 seed×partition 为层级单位的统计模型和预注册主要对比；",
        "- NMI 在低适应点已较高，说明当前位置信息指标需要熵校正、置换检验和更丰富的未见组合测试。",
        "",
        "## 下一轮实验",
        "",
        "下一步应完整交叉适应阶段的固定、轮换和随机伙伴日程，同时保留本轮六类 resident 文化。首先比较单 sender、单 receiver 和全群体 newcomer；其次把更新量扩展到 30、100、300、600，并以恢复曲线下面积、达到预设 J 阈值所需更新数、pair agreement 和角色 spread 作为预注册主指标。",
        "",
        "在确认通信头社会学习稳定后，再引入真正的社会压力：两种资源需要由不同主体采集、库存会随时间消耗、错误沟通产生延迟成本，随后加入新主体观察旧主体、代际 bottleneck 和有限示范。每一步都应保留零适应、随机符号、receiver 替换和跨文化 endpoint 作为对照，避免把任务复杂度增加后出现的性能下降误判为语言结构。",
        "",
        "## 可复核性",
        "",
        f"独立 NumPy 分析：[newcomer_adaptation_analysis.json]({(out / 'newcomer_adaptation_analysis.json').resolve()})，覆盖 {data['runs']} 条链、{data['coverage']['protocol_files']:,} 个协议文件，完成 {data['checks']:,} 项检查和 {data['scalar_comparisons']:,} 个标量比较，保存指标最大绝对差异为 {data['maximum_metric_absolute_difference']:.1e}。",
        f"独立动作、采样和分数审计：[newcomer_adaptation_audit.json]({(out / 'newcomer_adaptation_audit.json').resolve()})，覆盖 {audit['runs']} 条链、{audit['coverage']['traces']:,} 条训练轨迹和 {audit['coverage']['protocol_files']:,} 个协议文件；完成 {audit['checks']:,} 项检查，最大重放误差 {audit['maximum_replay_absolute_difference']:.1e}，未导入生产模块。",
        f"训练完成记录：[training_complete.json]({(out / 'training_complete.json').resolve()})；图形：[01_newcomer_adaptation.png]({(out / 'figures' / '01_newcomer_adaptation.png').resolve()})；绘图元数据：[plot_metadata.json]({(out / 'plot_metadata.json').resolve()})；视觉 QA：[visual_qa.json]({(out / 'visual_qa.json').resolve()})。",
        f"研究审查：[结果审查.md]({(ROOT / '结果审查.md').resolve()})；审查记录：[结果审查.json]({(ROOT / '结果审查.json').resolve()})。",
        "",
        "本轮使用的是本地 PyTorch 受控 agent 环境：视觉特征和通信模块来自已封存的 v0.28/v0.43 训练链，未调用 LLM 或外部 API。结果的适用范围是协议社会学习机制，而不是具备世界知识的大模型语言能力。",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
