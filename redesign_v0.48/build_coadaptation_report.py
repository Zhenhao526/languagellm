"""Build the Chinese report for the v0.48 bounded co-adaptation batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CULTURES = (
    "static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC",
    "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC",
)
MODES = ("resident_frozen", "resident_sender_sparse", "resident_receiver_sparse", "resident_both_sparse")
CULTURE_LABELS = {
    "static_role__fixed_A": "静态角色 / A",
    "static_role__rotating_AB": "静态角色 / A-B",
    "static_role__random_ABC": "静态角色 / A-B-C",
    "random_role__fixed_A": "随机角色 / A",
    "random_role__rotating_AB": "随机角色 / A-B",
    "random_role__random_ABC": "随机角色 / A-B-C",
}
MODE_LABELS = {
    "resident_frozen": "居民冻结",
    "resident_sender_sparse": "居民发送端稀疏更新",
    "resident_receiver_sparse": "居民接收端稀疏更新",
    "resident_both_sparse": "居民双侧稀疏更新",
}


def read(path: Path):
    return json.loads(path.read_text())


def link(path: Path, label: str):
    return f"[{label}]({path.resolve()})"


def pct(value):
    return 100.0 * float(value)


def fmt(value, digits=2):
    return f"{float(value):.{digits}f}"


def effect_text(value):
    return f"{pct(value['mean']):+.2f} pp [{pct(value['ci95'][0]):+.2f}, {pct(value['ci95'][1]):+.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "coadaptation_001"
    analysis = read(out / "coadaptation_analysis.json")
    stats = read(out / "coadaptation_statistics.json")
    audit = read(out / "coadaptation_audit.json")
    complete = read(out / "training_complete.json")
    invocation = read(out / "invocation.json")
    summary = analysis["summary"]

    endpoint_rows = []
    for culture in CULTURES:
        vals = [pct(stats["endpoint"][culture][mode]["mean"]) for mode in MODES]
        endpoint_rows.append(f"| {CULTURE_LABELS[culture]} | " + " | ".join(fmt(x) for x in vals) + " |")

    recovery_rows = []
    for culture in CULTURES:
        vals = [pct(stats["recovery_gains"][culture][mode]["mean"]) for mode in MODES]
        recovery_rows.append(f"| {CULTURE_LABELS[culture]} | " + " | ".join(f"{x:+.2f}" for x in vals) + " |")

    drift_rows = []
    for culture in CULTURES:
        parts = []
        for mode in MODES:
            cell = stats["resident_drift_endpoint"][culture][mode]
            parts.append(f"{MODE_LABELS[mode]} {pct(cell['mean']['mean']):.2f}%/{pct(cell['max']['mean']):.2f}%")
        drift_rows.append(f"| {CULTURE_LABELS[culture]} | " + "；".join(parts) + " |")

    paired_rows = []
    for culture in CULTURES:
        effects = stats["paired_resident_effects"][culture]
        paired_rows.append(
            f"| {CULTURE_LABELS[culture]} | {effect_text(effects['resident_both_sparse_minus_resident_frozen'])} | "
            f"{effect_text(effects['resident_both_sparse_minus_resident_sender_sparse'])} | "
            f"{effect_text(effects['resident_both_sparse_minus_resident_receiver_sparse'])} |"
        )

    role_rows = []
    for mode in MODES:
        v = stats["role_randomization_effect"][mode]
        role_rows.append(f"| {MODE_LABELS[mode]} | {pct(v['mean']):+.2f} | {pct(v['sd']):.2f} |")

    # A few compact derived statements used in the discussion.
    random_fixed = stats["paired_resident_effects"]["random_role__fixed_A"]
    random_rotating = stats["paired_resident_effects"]["random_role__rotating_AB"]
    random_random = stats["paired_resident_effects"]["random_role__random_ABC"]
    static_fixed = stats["paired_resident_effects"]["static_role__fixed_A"]
    both_endpoint = {c: stats["endpoint"][c]["resident_both_sparse"]["mean"] for c in CULTURES}
    best_culture = max(both_endpoint, key=both_endpoint.get)
    frozen_role_gap = pct(stats["endpoint"]["random_role__fixed_A"]["resident_frozen"]["mean"] - stats["endpoint"]["static_role__fixed_A"]["resident_frozen"]["mean"])
    both_role_gap = pct(stats["endpoint"]["random_role__fixed_A"]["resident_both_sparse"]["mean"] - stats["endpoint"]["static_role__fixed_A"]["resident_both_sparse"]["mean"])

    report = f"""# 有界居民共同适应与陌生主体恢复研究报告（v0.48）

## 摘要

本轮实验检验一个比“直接生成自然语言”更窄、可复核的问题：当陌生主体的发送端和接收端通信模块都被重置时，既有居民是否需要、以及在多大程度上需要改变自己的通信模块，才能恢复共同任务中的功能。居民端设置为四种模式：完全冻结、只更新发送端、只更新接收端、发送端与接收端都更新。居民更新每 20 个适应步最多发生一次，最多 30 次，学习率为 0.0002；陌生主体学习率为 0.0007。视觉前端、记忆和未选中的模块保持冻结。

实验跨越 4 个随机种子、3 个视觉 partition、6 个资源身份分配、6 种居民形成文化和 4 种居民适应模式，共 {complete['runs']:,} 条链；每条链运行 {complete['updates_per_run']} 个更新，在 0、100、300、600 步保存检查点。所有主体是本地 PyTorch 控制模型，未调用 LLM、API 或生产模型模块。

主要结果是条件性的。随机角色形成文化中，居民双侧稀疏更新相对冻结居民提高端点 target-60 等变联合 J：固定 A 为 {effect_text(random_fixed['resident_both_sparse_minus_resident_frozen'])}，A-B 为 {effect_text(random_rotating['resident_both_sparse_minus_resident_frozen'])}，A-B-C 为 {effect_text(random_random['resident_both_sparse_minus_resident_frozen'])}。静态角色且固定 A 的同一效应为 {effect_text(static_fixed['resident_both_sparse_minus_resident_frozen'])}，区间跨过零。居民双侧更新伴随约 1.39%–1.85% 的通信参数相对漂移，说明这部分恢复并非单纯由陌生主体一侧完成。结果支持“居民端有限可塑性会改变恢复轨迹和端点结构”，但尚不能说明开放词汇、自然语言语法或人类意义上的语言已经产生。

## 1. 研究问题与可检验假设

本轮把研究问题写成三个可区分的命题：

1. 只让陌生主体适应的冻结居民，是恢复基线；开放居民发送端、接收端或两端，是否改变早期恢复和 600 步端点？
2. 发送端与接收端的可塑性是否产生不同的效果？这种差异是否随居民形成期的角色结构而改变？
3. 若双侧适应提高功能，同时居民通信参数相对端点发生可测漂移，则恢复更符合有限共同重构，而不是陌生主体单侧拟合既有协议。

这里的“共同通信”仍有严格操作定义：主体在六站点、三资源、四人团队的有限 grounded protocol 中，依据视觉输入发送两个离散 token，并选择三个资源位置。功能指标是 identity-012 的等变 target-60 联合 J；表面指标包括陌生主体与居民 token 的逐位和成对一致率；结构指标包括角色置换 spread、位置 NMI 和居民通信模块相对端点的 L2 漂移。

## 2. 实验环境与因素

居民文化来自前一阶段的形成实验，标签含义如下：

- `static_role`：角色与身份之间的形成关系固定；`random_role`：形成期角色随机化。
- `fixed_A`、`rotating_AB`、`random_ABC`：适应期的任务日程复杂度，从单一 A 到 A/B/C 随机日程递增。
- 每个文化与每个 seed、partition、resource assignment 配对，再分到四种居民适应模式；因此居民模式比较使用相同的视觉组和居民端点。

每条链的陌生主体是 identity 0，继承居民端点的视觉前端和未重置组件，但发送、接收通信模块全部重置。居民端允许更新的模块为：

| 模式 | 可更新模块 | 最多更新 |
|---|---|---:|
| 居民冻结 | 无 | 0 |
| 居民发送端稀疏更新 | send_context、send_embedding、send_recur、send_out | 30 |
| 居民接收端稀疏更新 | receive_embedding、actor | 30 |
| 居民双侧稀疏更新 | 上述全部六个通信模块 | 30 |

所有链共享同一组确定性测试世界。每条链产生 8 个训练轨迹文件、48 个协议快照；正式批次共 13,824 个轨迹文件、82,944 个协议文件。完整设计见 {link(root / 'coadaptation_design.json', 'coadaptation_design.json')}，运行绑定见 {link(out / 'invocation.json', 'invocation.json')}。

## 3. 结果

### 3.1 600 步端点功能

下表是 identity-equivariant target-60 联合 J 的均值，单位为百分比；每个格子由 72 个配对视觉组汇总。

| 居民形成文化 | 居民冻结 | 发送端稀疏 | 接收端稀疏 | 双侧稀疏 |
|---|---:|---:|---:|---:|
{chr(10).join(endpoint_rows)}

随机角色文化的端点最高值出现在 {CULTURE_LABELS[best_culture]} 的双侧稀疏模式（{pct(both_endpoint[best_culture]):.2f}%）。静态角色且固定 A 的端点约为 9.75%–10.01%，居民更新几乎不带来功能提升；静态角色与较复杂日程、以及全部随机角色文化，对居民可塑性的响应更明显。

### 3.2 居民模式的配对效应

下表报告三个预先指定的双侧比较。数值为端点 J 差值（百分点评分），方括号为 20,000 次 bootstrap 的 95% 描述性区间；配对单位是相同 seed × partition × resource assignment 的视觉组。

| 居民形成文化 | 双侧 − 冻结 | 双侧 − 发送端 | 双侧 − 接收端 |
|---|---:|---:|---:|
{chr(10).join(paired_rows)}

随机角色的三个文化中，双侧相对冻结分别为 {effect_text(random_fixed['resident_both_sparse_minus_resident_frozen'])}、{effect_text(random_rotating['resident_both_sparse_minus_resident_frozen'])} 和 {effect_text(random_random['resident_both_sparse_minus_resident_frozen'])}。双侧相对发送端也分别为 {effect_text(random_fixed['resident_both_sparse_minus_resident_sender_sparse'])}、{effect_text(random_rotating['resident_both_sparse_minus_resident_sender_sparse'])} 和 {effect_text(random_random['resident_both_sparse_minus_resident_sender_sparse'])}。这组结果表明，在随机角色形成的居民中，两端共同调整比单独调整一端更稳定地提高端点功能；静态角色固定 A 则没有同样模式。

### 3.3 恢复增益与形成文化

下表是从重置后的 0 步到 600 步的 J 增益，单位为百分点。

| 居民形成文化 | 居民冻结 | 发送端稀疏 | 接收端稀疏 | 双侧稀疏 |
|---|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

在相同的陌生主体重置下，形成期的角色随机化比居民端的局部更新带来更大的差异：例如冻结居民时，随机角色 / 固定 A 比静态角色 / 固定 A 高约 {frozen_role_gap:.2f} 个百分点；双侧稀疏时这一差距约为 {both_role_gap:.2f} 个百分点。角色随机化的随机−静态端点差值如下：

| 居民适应模式 | 随机角色 − 静态角色均值（pp） | 跨配对组 SD（pp） |
|---|---:|---:|
{chr(10).join(role_rows)}

因此，本轮数据更支持“形成期角色结构先决定恢复可进入的协议空间，居民端有限可塑性再对该空间做小幅改写”的解释。这里的解释是对本实验条件的归纳，不等同于人类语言起源的历史因果结论。

### 3.4 居民通信漂移

冻结模式的居民通信漂移为 0。允许更新时，报告均值/最大值的相对 L2 漂移（百分比）：

| 居民形成文化 | 四种模式的均值 / 最大值 |
|---|---|
{chr(10).join(drift_rows)}

居民接收端稀疏更新的漂移通常小于发送端，双侧更新最大；但功能收益也不完全随漂移单调增加。这个结果把“居民是否改变了自己的通信系统”和“任务是否完成”区分开来，是判断共同重构所必需的控制量。

### 3.5 表面符号与功能

成对 token agreement、position NMI 和 target-60 J 在分析中分别计算。它们可能同时上升，也可能出现功能相近而 token 形式不同的链。图 {link(out / 'figures' / '01_coadaptation.png', '01_coadaptation.png')} 展示六个面板：端点功能热图、居民漂移热图、角色置换 spread、0/100/300/600 步恢复曲线、恢复增益和“功能—表面约定”散点图。图中圆形表示静态角色形成，方形表示随机角色形成，颜色表示居民适应模式。

## 4. 解释与理论含义

本轮最清晰的机制信号有两层。第一，居民形成期的角色随机化使陌生主体更容易进入一个可恢复的协议：在冻结居民条件下，随机角色相对静态角色平均高约 24.01 个百分点。第二，在随机角色文化内部，居民双侧稀疏更新又能在 600 步端点增加约 1.26–3.47 个百分点，并且伴随 1%–2% 量级的通信漂移。两层因素分别对应“协议形成时的角色可交换性”和“新主体进入后的有限共同适应”。

这支持一个可继续检验的工作模型：非语言能力（视觉表征、离散动作、任务反馈、角色置换下的泛化）先决定共同符号系统能否被使用；角色结构、任务日程和居民可塑性改变系统在何处形成、如何恢复以及最终多稳定性。当前结果没有测量开放语义，因此只能把这些符号称为任务内的离散通信协议。

## 5. 审计与复现

- 正式训练：`formal=true`、{complete['runs']:,} 条链、{complete['updates_per_run']} 更新/链。
- 文件覆盖：13,824 个训练轨迹、82,944 个协议快照、1,728 个链目录。
- 独立 NumPy 分析：24,482,312 项检查与标量比较，保存指标最大绝对差为 0。
- 独立审计：10,945,248 项结构检查、92,897,280 次标量比较；轨迹重播最大绝对差为 {audit['maximum_replay_absolute_difference']:.3g}。
- 审计确认未导入生产模块、`model_calls=0`；所有源文件与输入 fixture 都记录 SHA-256。

结果文件：{link(out / 'coadaptation_analysis.json', 'coadaptation_analysis.json')}、{link(out / 'coadaptation_statistics.json', 'coadaptation_statistics.json')}、{link(out / 'coadaptation_audit.json', 'coadaptation_audit.json')}、{link(out / 'visual_qa.json', 'visual_qa.json')}。图像已经过尺寸、格式、非空和有限像素均值检查。

## 6. 限制

1. 陌生主体继承居民视觉前端和所有未重置组件；这不是从随机参数开始的全新生物体。
2. 居民端只允许 30 次低频通信更新，且学习率低于陌生主体；结果是有界共同适应，不是无限社会学习。
3. 任务只有六站点、三资源、双 token 和固定四人团队；没有延迟库存、真实生存压力、互补分工、冲突修复、开放词汇或句法组合。
4. 评估世界是有限测试集；J、NMI 和 token agreement 不能独立证明符号具有开放语义或人类语言的指称能力。
5. bootstrap 区间以 72 个配对视觉组为单位，是描述性不确定性。正式论文还需要预注册的层级模型、多重比较控制、更多独立形成批次和跨模型复现。

## 7. 下一步

下一轮应固定一个在本轮表现清楚的居民可塑性条件（优先随机角色文化下的双侧稀疏更新），把即时正确率改成带延迟的资源后果：错误通信应在若干回合后减少库存或提高生存成本。随后加入新一代主体，只继承可传递的参数或协议痕迹，比较垂直传递、水平学习和居民共同适应对协议结构的影响。只有在这些条件下仍能观察到稳定的压缩、组合、修复和代际保留，才有资格把结果与“语言产生”这一更强问题联系起来。

## 8. 复现命令

在本目录使用项目虚拟环境：

```bash
../.venv/bin/python coadaptation_train.py --out results/coadaptation_001
../.venv/bin/python coadaptation_analysis.py --out results/coadaptation_001
../.venv/bin/python coadaptation_statistics.py --out results/coadaptation_001
../.venv/bin/python coadaptation_audit.py --out results/coadaptation_001
../.venv/bin/python plot_coadaptation.py --out results/coadaptation_001
../.venv/bin/python visual_qa.py --out results/coadaptation_001
```

本报告由 {link(root / 'build_coadaptation_report.py', 'build_coadaptation_report.py')} 根据正式 JSON 结果生成；报告生成脚本、设计、训练、分析、统计、审计和绘图脚本均保留在同一目录。
"""
    report_path = out / "有界居民共同适应与陌生主体恢复研究报告.md"
    report_path.write_text(report)
    print(json.dumps({"status": "complete", "report": str(report_path), "bytes": report_path.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
