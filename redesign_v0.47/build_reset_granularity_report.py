"""Build the Chinese report for the v0.47 reset-granularity batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CULTURES = (
    "static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC",
    "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC",
)
RESET_MODES = ("sender_only", "receiver_only", "both")
LABELS = {
    "static_role__fixed_A": "静态/A", "static_role__rotating_AB": "静态/A-B", "static_role__random_ABC": "静态/A-B-C",
    "random_role__fixed_A": "随机/A", "random_role__rotating_AB": "随机/A-B", "random_role__random_ABC": "随机/A-B-C",
}
MODE_LABELS = {"sender_only": "只重置发送端", "receiver_only": "只重置接收端", "both": "两端都重置"}
UPDATES = (0, 100, 300)


def read(path: Path):
    return json.loads(path.read_text())


def pct(value):
    return 100.0 * float(value)


def item(summary, culture, mode, update, metric):
    return summary[culture][mode][str(update)][metric]


def progression(summary, culture, mode, metric="identity_equivariant_target60_J"):
    return " → ".join(f"{pct(item(summary, culture, mode, update, metric)['mean']):.2f}%" for update in UPDATES)


def fmt(summary, culture, mode, update, metric):
    value = item(summary, culture, mode, update, metric)
    return f"{pct(value['mean']):.2f}%（SD {pct(value['sd']):.2f}%）"


def delta(value):
    return f"{pct(value['mean']):+.2f} pp（95% CI {pct(value['ci95'][0]):+.2f}, {pct(value['ci95'][1]):+.2f}）"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve()
    analysis = read(out / "reset_granularity_analysis.json"); stats = read(out / "reset_granularity_statistics.json"); audit = read(out / "reset_granularity_audit.json"); complete = read(out / "training_complete.json"); summary = analysis["summary"]
    report = out / "通信模块重置粒度与陌生主体恢复研究报告.md"
    lines = [
        "# v0.47 通信模块重置粒度与陌生主体恢复：发送端和接收端的非对称作用",
        "",
        "## 摘要",
        "",
        "v0.46 已把 resident 形成文化与 newcomer 适应伙伴日程交叉，下一步需要区分通信系统的哪一侧真正承担恢复压力。本轮从同一 resident endpoint 复制 identity 0，只重置发送模块、只重置接收模块，或同时重置两端；适应阶段固定使用随机 A/B/C 伙伴日程。实验包含 4 个 seed × 3 个视觉 partition × 6 个资源列排列 × 6 类 resident 文化 × 3 种重置模式，共 1,296 条链，每条训练 300 次更新。",
        "",
        "本轮的构念是通信侧恢复，不是从零学习世界。newcomer 继承 resident 的视觉前端和未重置通信模块；resident 全部冻结。因而结果回答的是：在共同任务回报、已有 resident 约定和有限消息空间给定时，发送端或接收端的可塑性是否足以恢复功能，以及两者是否有不同的时间尺度。",
        "",
        "## 研究问题和预注册比较",
        "",
        "1. 只重置发送端与只重置接收端，哪一种恢复更快、端点 J 更高？",
        "2. 两端都重置是否产生最大的早期缺口，并在 300 次更新内接近单端恢复？",
        "3. resident 形成期的角色随机化是否仍降低恢复后的 role spread？",
        "",
        "三种重置模式在同一 seed、partition、资源列排列和 resident culture 内配对；reset seed 不含重置模式，因此可比较的随机初始化流保持一致。所有链都使用随机 A/B/C 适应日程、相同训练预算和相同评估协议。",
        "",
        "## 实验设计",
        "",
        "每条链加载 v0.43 的 resident endpoint。identity 0 被复制为 newcomer。`sender_only` 重置 `send_context`、`send_embedding`、`send_recur`、`send_out` 并只训练 sender 组；`receiver_only` 重置 `receive_embedding`、`actor` 并只训练 receiver 组；`both` 重置两组并同时训练。视觉投影、记忆和 resident 参数冻结。",
        "",
        "每次更新从随机 A/B/C 日程抽取一个四人 team，三种资源各发送两个离散 token（词表 7），接收者输出三个站点动作（6 个站点）。联合回报与 v0.43 相同。检查点为 0、100、300；每个检查点保存 A/B/C 评估拓扑、4 个 team 和 6 种角色排列。主要指标为 identity-012 的等变 target-60 joint J，辅助指标为 role spread、newcomer–resident pair agreement 和 position NMI。",
        "",
        f"正式输出为 {complete['runs']} 条链、{complete['updates_per_run']} 次更新/链、{complete['trace_files_expected']} 条轨迹和 {complete['protocol_files_expected']} 个端点协议文件。设计：[reset_granularity_design.json]({(ROOT / 'reset_granularity_design.json').resolve()})；训练绑定：[invocation.json]({(out / 'invocation.json').resolve()})。",
        "",
        "## 结果一：三种重置模式的恢复曲线",
        "",
        "每格为 update 0 → 100 → 300 的 identity-012 等变 target-60 J。",
        "",
        "| resident 形成文化 | 只重置发送端 | 只重置接收端 | 两端都重置 |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(f"| {LABELS[culture]} | {progression(summary, culture, 'sender_only')} | {progression(summary, culture, 'receiver_only')} | {progression(summary, culture, 'both')} |")
    lines += [
        "",
        "端点表给出 300 次更新的链级均值和 SD。每个格子有 72 个配对视觉组单位，并先在四个评估 team 与三种拓扑上求平均。",
        "",
        "| resident 形成文化 | 只重置发送端 | 只重置接收端 | 两端都重置 |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(f"| {LABELS[culture]} | {fmt(summary, culture, 'sender_only', 300, 'identity_equivariant_target60_J')} | {fmt(summary, culture, 'receiver_only', 300, 'identity_equivariant_target60_J')} | {fmt(summary, culture, 'both', 300, 'identity_equivariant_target60_J')} |")
    lines += [
        "",
        "若 receiver-only 高于 sender-only，说明已有 resident sender 流可以在共同回报下为接收端提供稳定学习信号；若 sender-only 更高，则说明保留 resident decoder 比保留 sender codebook 更关键。这个比较应结合起点差异和恢复增量阅读，不能只看端点。",
        "",
        "## 结果二：重置模式的配对端点效应",
        "",
        "以下差异均为同一 resident 文化内的链级配对比较。每个比较 72 个配对单位；区间用固定 seed 的 20,000 次 bootstrap 描述。正值表示右侧模式的 J 更高。",
        "",
        "| resident 形成文化 | 接收端 − 发送端 | 两端 − 发送端 | 两端 − 接收端 |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        e = stats["paired_reset_effects"][culture]
        lines.append(f"| {LABELS[culture]} | {delta(e['receiver_only_minus_sender_only'])} | {delta(e['both_minus_sender_only'])} | {delta(e['both_minus_receiver_only'])} |")
    lines += [
        "",
        "这些是配对效应，不是跨 culture 的独立样本检验；bootstrap 只描述视觉组层面的不确定性。若两端重置的端点接近单端模式，但 update-100 仍有明显缺口，说明恢复瓶颈主要发生在早期同步，而非最终可达协议。",
        "",
        "## 结果三：角色稳定性与表面形式",
        "",
        "| resident 形成文化 | 只重置发送端 spread | 只重置接收端 spread | 两端都重置 spread |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(f"| {LABELS[culture]} | {fmt(summary, culture, 'sender_only', 300, 'role_spread_target60_J')} | {fmt(summary, culture, 'receiver_only', 300, 'role_spread_target60_J')} | {fmt(summary, culture, 'both', 300, 'role_spread_target60_J')} |")
    lines += [
        "",
        "随机角色 resident 的 spread 若在三种重置模式下都较低，将延续 v0.43–v0.46 的解释：形成期的角色随机化改变了协议对角色轴的稳定性。重置模式若只改变 J 而不改变 spread，则表示它主要影响恢复速度或功能可达性；若同时改变 spread，则说明通信侧恢复也会重新选择表面码本结构。",
        "",
        "下表给出 update-300 的 newcomer–resident 双 token pair agreement。",
        "",
        "| resident 形成文化 | 只重置发送端 | 只重置接收端 | 两端都重置 |",
        "|---|---:|---:|---:|",
    ]
    for culture in CULTURES:
        lines.append(f"| {LABELS[culture]} | {fmt(summary, culture, 'sender_only', 300, 'newcomer_resident_pair_agreement')} | {fmt(summary, culture, 'receiver_only', 300, 'newcomer_resident_pair_agreement')} | {fmt(summary, culture, 'both', 300, 'newcomer_resident_pair_agreement')} |")
    lines += [
        "",
        "pair agreement 只是表面形式指标。功能 J 可以通过等价码本恢复，因此 pair agreement 较低并不等价于没有共同符号；position NMI 在有限离散任务上也不能单独解释为语义。",
        "",
        "## 对核心问题的回答",
        "",
        "本轮把“陌生主体能否恢复协议”拆成了发送端和接收端两个可塑性问题。结果应优先解释为通信系统的侧向依赖：哪一侧被重置会改变早期反馈路径、端点功能和表面一致性。它仍然没有测试没有语言先验的开放主体，也没有证明自然语言的词义、句法或人类语言起源条件。",
        "",
        "v0.46 的伙伴覆盖结论在本轮被固定为随机 A/B/C；因此本轮新增的是 reset granularity 的机制维度，而不是新的伙伴拓扑主效应。",
        "",
        "## 限制",
        "",
        "- newcomer 继承同一 visual group 的视觉前端，且未重置的通信模块直接继承 resident endpoint；这不是全新主体；",
        "- resident 冻结，没有双向协商、repair、冲突或群体共同改码；",
        "- 任务仍是六站点、三资源、双 token 的有限 grounded protocol，没有开放词汇、生产性组合、延迟库存、生存压力、互补分工或代际传递；",
        "- reset mode 的 bootstrap 区间是描述性配对区间，正式论文仍需预注册层级模型、多重比较和更长时间点；",
        "- NMI、pair agreement 和 target-60 J 的机会基线仍需独立置换和熵校正。",
        "",
        "## 下一轮实验",
        "",
        "下一步应在不改变本轮 reset 模式的前提下加入更长的 600 更新检查点，并把 resident 从完全冻结改为有界适应：先允许被重置侧和一个 resident 侧各更新少量步数，再比较单向 repair 与全群体共同重构。随后再引入延迟资源库存和互补分工，检验通信恢复是否会转化为跨回合的公共协调约定。",
        "",
        "## 可复核性",
        f"独立 NumPy 分析：[reset_granularity_analysis.json]({(out / 'reset_granularity_analysis.json').resolve()})，覆盖 {analysis['runs']} 条链、{analysis['coverage']['protocol_files']:,} 个协议文件，完成 {analysis['checks']:,} 项检查和 {analysis['scalar_comparisons']:,} 个标量比较，保存指标最大绝对差异 {analysis['maximum_metric_absolute_difference']:.1e}。",
        f"配对统计：[reset_granularity_statistics.json]({(out / 'reset_granularity_statistics.json').resolve()})，每个 resident 文化的重置比较使用 72 个链级配对单位和 20,000 次 bootstrap。",
        f"独立轨迹/采样审计：[reset_granularity_audit.json]({(out / 'reset_granularity_audit.json').resolve()})，覆盖 {audit['runs']} 条链、{audit['coverage']['traces']:,} 条轨迹和 {audit['coverage']['protocol_files']:,} 个协议文件，完成 {audit['checks']:,} 项检查，最大重放误差 {audit['maximum_replay_absolute_difference']:.1e}，未导入生产模块。",
        f"图形：[01_reset_granularity.png]({(out / 'figures' / '01_reset_granularity.png').resolve()})；绘图元数据：[plot_metadata.json]({(out / 'plot_metadata.json').resolve()})；视觉 QA：[visual_qa.json]({(out / 'visual_qa.json').resolve()})。",
        f"研究审查：[结果审查.md]({(ROOT / '结果审查.md').resolve()})；审查记录：[结果审查.json]({(ROOT / '结果审查.json').resolve()})。",
        "",
        "本轮使用本地 PyTorch 受控 agent，未调用 LLM 或外部 API。结论适用于有限协议的通信侧恢复机制，不能外推为大模型或自然语言能力。",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
