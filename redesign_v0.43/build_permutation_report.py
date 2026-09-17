"""Build the Chinese v0.43 formation report from sealed JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
MODES = ("static_role", "random_role")
ROLES = ("012", "021", "102", "120", "201", "210")
LABEL = {"fixed_A": "固定 A", "rotating_AB": "轮换 A/B", "random_ABC": "随机 A/B/C"}
MODE_LABEL = {"static_role": "静态角色顺序", "random_role": "随机角色顺序"}


def read(path):
    return json.loads(path.read_text())


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def pct(x):
    return 100.0 * float(x)


def fmt(m, s=None, digits=2):
    if isinstance(m, (list, tuple, np.ndarray)):
        m, s = mean_sd(m)
    if s is None:
        return f"{pct(m):.{digits}f}%"
    return f"{pct(m):.{digits}f}%（SD {pct(s):.{digits}f}%）"


def endpoint(data, mode, condition, role="012", kind="equivariant"):
    return data["summary"][mode][condition]["600"][role][kind]


def values(data, mode, condition, role="012", kind="equivariant"):
    return endpoint(data, mode, condition, role, kind)["values"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "permutation_analysis.json")
    stats = read(out / "permutation_statistics.json")
    complete = read(out / "training_complete.json")
    audit = read(out / "permutation_audit.json")
    report = out / "三资源排列与角色随机化形成研究报告.md"

    lines = [
        "# v0.43 三资源排列与角色随机化：共同符号形成研究报告",
        "",
        "## 摘要",
        "",
        "本轮把“角色顺序是否必须被共同体对称化”放进形成阶段直接检验。每个 sender 仍只看到自己的资源图片并输出两个离散 token，receiver 根据三组 token 预测三个资源的位置；改变的是训练时三个资源角色是否始终固定，或在每个 team、每次更新随机重排。结果显示，随机角色顺序显著降低了协议对某一固定角色排列的依赖：端点六种角色排列的等变联合 J 的跨排列 spread 在固定 A、轮换 A/B、随机 A/B/C 中分别从 25.20%、66.12%、71.13% 降到 2.45%、15.87%、14.44%。在身份角色排列上，随机角色训练相对静态训练的配对增益分别为 +23.77、+16.87、+7.60 个百分点（bootstrap 95% CI 均不跨 0）。",
        "",
        "这轮结果支持一个更窄、可检验的结论：在有限的 grounded coordination task 中，角色置换压力足以诱导近似的资源轴等变性，并改变双 token 的序列结构；它还没有产生开放式语言，也不能单独解释人类语言起源。对 ICLR 级别论文而言，当前最有价值的对象是“非语言的共同任务、角色对称性和伙伴覆盖如何塑造可迁移的共同符号协议”，而不是把有限码表直接称为语言。",
        "",
        "## 研究问题和分析预期",
        "",
        "本轮沿用前几轮的三资源协作环境，新增两个形成因素：",
        "",
        "1. 资源列排列：六种排列 `012/021/102/120/201/210`，把资源身份放入不同的视觉输入列；",
        "2. 角色顺序：`static_role` 每次都使用 `(0,1,2)`，`random_role` 在每个 team、每次更新从六种排列中确定性随机抽取，并同步重排 sender 观察和 target 位置轴。",
        "",
        "由此检验四个问题：角色随机化是否提高身份排列的功能；是否降低对角色排列的敏感性；伙伴拓扑的覆盖是否与角色对称压力交互；资源列排列是否会改变最终协议。序列指标用于判断这种对称化是否伴随 token0/token1 的信息重新分配。这里的“等变”是预先规定的评分：把 receiver 输出的资源轴与被测试的角色排列同步置换后再算 J；它表示结构关系随角色轴一起移动，不等同于自然语言的组合语法。",
        "",
        "## 实验环境",
        "",
        f"正式批次包含 {complete['runs']} 条独立链：4 个 seed × 3 个视觉 partition × 6 个资源列排列 × 2 种角色模式 × 3 种伙伴拓扑。每条链训练 {complete['updates_per_run']} 次更新，检查点为 0、100、300、600；每次更新包含四个 team、每个 team 240 个训练世界。通信模块重新随机初始化，视觉投影、记忆和图片特征从 v0.28/v0.39 输入继承并冻结。训练总量为 {complete['messages']:,} 次三资源消息、{complete['token_instances']:,} 个 token 实例和 {complete['actions']:,} 次位置动作采样。",
        "",
        "训练世界来自每个 partition 的 60 个有序地图和每地图 8 个图片组合（480 个训练世界）；测试使用 120 个地图。每个 sender 的消息长度为 2，词表大小为 7；receiver 对三个资源输出 6 个站点中的一个。回报为 `(1/6)×正确资源数 + 0.5×三项全对`，优势基线为 0.1。伙伴拓扑 A、B、C 分别改变四个 agent 的 receiver/sender 配对；形成时只在指定拓扑中训练，评估时保存 A/B/C 三种拓扑。",
        "",
        f"设计文件：[permutation_design.json]({(ROOT / 'permutation_design.json').resolve()})；训练调用和输入哈希：[invocation.json]({(out / 'invocation.json').resolve()})。",
        "",
        "## 结果一：随机角色顺序提高身份排列的端点功能",
        "",
        "下表汇总身份角色排列 `012` 的 target-60 联合 J，样本跨 4 个 seed、3 个 partition、6 个资源列排列、3 个评估拓扑和 4 个 team。",
        "",
        "| 训练拓扑 | 静态角色顺序 | 随机角色顺序 | 随机−静态（配对均值，95% bootstrap CI） |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = endpoint(data, "static_role", condition)
        r = endpoint(data, "random_role", condition)
        delta = stats["identity_role_mode_differences"][condition]["random_minus_static_equivariant_target_J"]
        ci_text = f"{pct(delta['mean']):.2f} 个百分点（{pct(delta['ci95_bootstrap'][0]):.2f}, {pct(delta['ci95_bootstrap'][1]):.2f}）"
        lines.append(f"| {LABEL[condition]} | {fmt(s['mean'], s['sd'])} | {fmt(r['mean'], r['sd'])} | {ci_text} |")
    lines += [
        "",
        "静态角色模式下，伙伴拓扑从固定 A 到随机 A/B/C 时身份 J 从 11.41% 上升到 26.81%；随机角色模式下三种拓扑都在 34.42%–37.72%。这说明随机角色训练提供了比单纯扩大伙伴拓扑更直接的角色对称压力。配对差异的 bootstrap 区间以链的 seed、partition、资源排列、评估拓扑和 team 为配对单位；它是描述性不确定性，不替代对多因素交互的正式层级模型。",
        "",
        "## 结果二：角色随机化降低六种角色排列的敏感性",
        "",
        "对每个 endpoint 单元计算六种角色排列的等变 target J 的最大值减最小值，再在链和评估 team 上汇总。spread 越小，说明同一协议对资源角色轴越均匀。",
        "",
        "| 训练拓扑 | 静态角色 spread | 随机角色 spread | 随机−静态（95% bootstrap CI） |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        st = stats["role_spread"][condition]["static_role"]
        rr = stats["role_spread"][condition]["random_role"]
        dd = stats["role_spread"][condition]["random_minus_static"]
        lines.append(f"| {LABEL[condition]} | {fmt(st['mean'], st['sd'])} | {fmt(rr['mean'], rr['sd'])} | {fmt(dd['mean'], dd['sd'])}（CI {pct(dd['ci95_bootstrap'][0]):.2f}–{pct(dd['ci95_bootstrap'][1]):.2f} 个百分点） |")
    lines += [
        "",
        "端点角色排列的详细曲线见图。静态模式在轮换和随机伙伴条件下出现某些非 identity 排列的高等变分数，但排列之间差异很大；随机角色模式把六种排列压到相近水平。这个结果更像形成过程中的对称性正则化，而不是某一种排列偶然获得高分。",
        "",
        "## 结果三：literal 与等变评分揭示资源轴关系",
        "",
        "下面把五个非 identity 角色排列合并。literal 将输出和原始资源目标比较；等变评分先同步置换 target 资源轴。",
        "",
        "| 训练拓扑 | 静态 literal | 静态等变 | 随机 literal | 随机等变 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = []
        for mode in MODES:
            row.append(stats["nonidentity_literal_equivariant"][condition][mode])
        lines.append(f"| {LABEL[condition]} | {fmt(row[0]['literal']['mean'], row[0]['literal']['sd'])} | {fmt(row[0]['equivariant']['mean'], row[0]['equivariant']['sd'])} | {fmt(row[1]['literal']['mean'], row[1]['literal']['sd'])} | {fmt(row[1]['equivariant']['mean'], row[1]['equivariant']['sd'])} |")
    lines += [
        "",
        "静态模式的非 identity 等变分数在轮换 A/B 和随机 A/B/C 中达到 60.42% 和 69.49%，literal 几乎为零；随机角色模式则在三种拓扑中都保持约 35%–37% 的等变 J，literal 仍接近零。等变分数的上升说明 receiver 学到的是“消息资源块—目标资源轴”的关系，而非把三个块绑定到不可移动的绝对槽位；随机角色训练则把这项关系在六个轴排列之间均匀化。由于这些都是同一 endpoint 的反事实重排，不能把它解释成新任务或开放组合上的泛化。",
        "",
        "## 结果四：六种资源列排列的影响有限但不为零",
        "",
        "下表是随机角色模式、身份角色排列 `012` 的 endpoint 等变 target J；每格为跨 seed、partition、评估拓扑和 team 的均值。",
        "",
        "| 资源列排列 | 固定 A | 轮换 A/B | 随机 A/B/C |",
        "|---|---:|---:|---:|",
    ]
    for assignment in ROLES:
        cells = []
        for condition in CONDITIONS:
            m = stats["assignment_identity_endpoint"][condition]["random_role"]["means"][assignment]
            cells.append(f"{pct(m):.2f}%")
        lines.append(f"| `{assignment}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "六种排列的范围在固定 A、轮换 A/B、随机 A/B/C 中分别为 2.10、6.81、7.72 个百分点。资源列交换不是完全无影响，但其效应小于角色模式和伙伴拓扑引起的整体变化；这支持把资源名称视为需要通过协作学习的身份，而不是实验者直接提供的语义标签。由于三类图片 strata 的视觉统计并不严格相同，不能把小范围差异当作完全的资源置换不变性。",
        "",
        "## 结果五：双 token 的序列结构随角色压力改变",
        "",
        "序列统计只取身份角色排列下的 sender 端点输出，并在同一资源、同一伙伴拓扑内比较 sender 的 token。NMI 是 token 与该 sender 负责的位置之间的归一化互信息；pair NMI 接近 1 需要谨慎，因为六地点有限任务允许查表式编码。",
        "",
        "| 训练拓扑 | 角色模式 | token0 agreement | token1 agreement | pair agreement | token0 NMI | token1 NMI | pair NMI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for mode in MODES:
            s = data["sequence_summary"][mode][condition]
            lines.append(f"| {LABEL[condition]} | {MODE_LABEL[mode]} | {pct(s['token0_agreement']['mean']):.2f}% | {pct(s['token1_agreement']['mean']):.2f}% | {pct(s['pair_agreement']['mean']):.2f}% | {s['nmi_token0_position']['mean']:.3f} | {s['nmi_token1_position']['mean']:.3f} | {s['nmi_pair_position']['mean']:.3f} |")
    lines += [
        "",
        "随机角色模式下，随机 A/B/C 的 token0 agreement 从静态模式的 80.56% 降到 70.83%，但 token1 agreement 从 45.29% 升到 54.98%，pair agreement 从 34.26% 升到 43.06%；pair NMI 也从 1.000 降到 0.967。较保守的解释是，角色置换使信息不再全部依赖早期 token 或某个固定 sender—slot 配对，序列中的信息分配更分散；这不是“语法出现”的证据。",
        "",
        "## 结果六：形成曲线显示角色随机化改变了学习时间尺度",
        "",
        "身份角色排列的等变 target J（跨所有资源排列、seed、partition、评估拓扑和 team）如下；完整曲线见图。",
        "",
        "| 训练拓扑 | 角色模式 | update 0 | update 100 | update 300 | update 600 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for mode in MODES:
            vals = [data["summary"][mode][condition][str(update)]["012"]["equivariant"]["mean"] for update in (0, 100, 300, 600)]
            lines.append(f"| {LABEL[condition]} | {MODE_LABEL[mode]} | " + " | ".join(f"{pct(v):.2f}%" for v in vals) + " |")
    lines += [
        "",
        "静态模式在 update 100 出现短暂的低分平台，随机角色模式从早期开始持续上升；在 600 更新时随机角色模式达到约 34%–38%。但固定学习预算只有 600 更新，且跨链 SD 较大，不能据此断言随机角色在任意优化器或训练长度下都更快。",
        "",
        "## 解释：这轮实验实际支持什么",
        "",
        "第一，角色随机化提供了一个明确的非语言对称性压力：agent 必须根据当前资源块的关系完成任务，不能只记住某个固定角色槽位。它提高身份排列的功能，并使六种角色排列的等变性能更接近。这个因素比简单增加伙伴拓扑覆盖更直接地改变了协议的结构。",
        "",
        "第二，伙伴拓扑仍然重要，但作用取决于是否有角色随机化。静态模式需要轮换或随机伙伴来暴露多种资源—receiver 配对，因此非 identity 等变评分会升高；随机角色模式已经在每次更新内遍历角色轴，拓扑差异随之缩小。这里的“更均匀”不是更高的绝对性能：静态模式在某些非 identity 反事实上有更高峰值，随机模式的优势是跨排列稳定。",
        "",
        "第三，资源列排列的影响相对较小但可测量，说明当前图片输入列仍携带架构偏置或视觉 strata 差异。后续必须把资源身份、图片统计和站点几何进一步解耦，才能把差异归因于社会学习而不是数据排列。",
        "",
        "## 不能从这轮结果推出的结论",
        "",
        "- 词表是 7 个离散值，消息长度为 2，任务只有六个站点和三个资源；receiver 可以在有限世界上查表，不能称为自然语言或开放式语法。",
        "- 所有 agent 的视觉前端继承并冻结，形成阶段只训练通信头；本轮测试的是共同符号协议的形成压力，不是从像素到概念的完整语言起源。",
        "- 奖励是集中式、即时的联合回报，训练中没有生存、资源库存、冲突、谈判或代际文化传递。它能表示协作信息价值，不能单独证明“分工是人类语言的必要条件”。",
        "- `equivariant` 是同步置换 target 的端点评分，不是未见资源或新空间关系上的迁移；六种资源列排列的视觉统计也不完全相同。",
        "- 角色随机模式把 sender 观察和 target 轴一起重排，等价于形成阶段的数据增强/对称性训练；它没有改变网络结构，也没有引入可解释的符号语义。",
        "",
        "## 面向高质量论文的下一步",
        "",
        "当前最有潜力的论文主张应限定为：在一个可完全审计的 grounded coordination 环境中，角色对称性、伙伴覆盖和共同回报足以改变离散共同符号协议的公共性与轴等变结构；双 token 的序列分工是这些压力的可观测结果。要把它推进到 ICLR 级别，下一轮需要把 endpoint 反事实变成形成期和迁移期的因果测试：",
        "",
        "1. 预注册训练期干预与测试期干预的分离，加入真正未见的资源—位置组合和未见 team；",
        "2. 保持总更新量和随机预算一致，完整交叉 `伙伴拓扑 × 角色随机化 × 资源列排列`，并用 seed/partition 的层级模型报告交互和置信区间；",
        "3. 做伙伴替换和新 agent 加入，测量协议保留、重学速度、歧义修复和新组合泛化，而不只看 receiver argmax；",
        "4. 逐步加入可共享资源库存、互补分工、延迟回报和代际 bottleneck，检验哪些非语言能力真正产生公共符号，哪些只是在有限任务上形成查表协议；",
        "5. 对消息长度、词表大小和连续/离散通道做容量控制，并保留本轮的 mask、swap、block permutation 和 role spread 作为统一因果指标。",
        "",
        "## 可复核性和审计",
        "",
        f"独立 NumPy 分析：[permutation_analysis.json]({(out / 'permutation_analysis.json').resolve()})，共完成 {data['checks']:,} 项结构检查和 {data['scalar_comparisons']:,} 个标量比较，重算指标最大绝对误差为 {data['maximum_metric_absolute_difference']:.1e}。",
        f"轨迹审计：[permutation_audit.json]({(out / 'permutation_audit.json').resolve()})，覆盖 432 条链、{audit['coverage']['traces']:,} 个训练轨迹文件（{audit['coverage']['trace_rows']:,} 行）、{audit['coverage']['protocols']:,} 个协议文件；固定随机数、角色排列、类别采样、reward、advantage、log-probability 和 entropy 均被独立重放，最大误差为 {audit['maximum_replay_absolute_difference']:.1e}。",
        f"配对统计：[permutation_statistics.json]({(out / 'permutation_statistics.json').resolve()})；图形：[01_permutation_formation.png]({(out / 'figures' / '01_permutation_formation.png').resolve()})；视觉 QA：[visual_qa.json]({(out / 'visual_qa.json').resolve()})。",
        "",
        "本轮没有调用生产模型做二次推理；分析和审计均以保存的 NumPy/NPZ 数组为输入。训练源文件和冻结输入的 SHA-256 绑定在 invocation.json 与 training_complete.json 中。",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
