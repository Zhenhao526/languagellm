"""Build the Chinese v0.41 two-token formation report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSIGNMENTS = ("canonical", "cyclic")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
ASSIGNMENT_LABELS = {"canonical": "资源顺序 canonical", "cyclic": "资源顺序 cyclic（1→2→0）"}
CONDITION_LABELS = {"fixed_A": "固定 A", "rotating_AB": "轮换 A/B", "random_ABC": "随机 A/B/C"}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x):
    return f"{100 * x:.3f}%"


def pp(x):
    return f"{100 * x:+.3f} 个百分点"


def mean_sd(item):
    return f"{pct(item['mean'])}（SD {pct(item['sd'])}）"


def link(path: Path, label: str | None = None):
    return f"[{label or path.name}]({path.resolve()})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    d = read(out / "two_token_analysis.json")
    v = read(out / "two_token_raw_validation.json")
    a = read(out / "two_token_audit.json")
    q = read(out / "visual_qa.json")
    t = read(out / "training_complete.json")
    i = read(out / "invocation.json")
    if not (
        d["formal"]
        and d["status"] == "complete"
        and v["passed"]
        and a["status"] == "complete"
        and q["status"] == "passed_visual_qa"
        and t["formal"]
        and i["formal"]
    ):
        raise ValueError("formal two-token outputs required")

    summary = d["summary"]
    effects = d["condition_effects"]
    assignment_effects = d["assignment_effects"]
    lines = [
        "# 三资源双 token 协议形成：v0.41 研究报告",
        "",
        "## 研究问题",
        "",
        "v0.39 只允许每个资源 sender 发送一个 token，已经能检验三槽位共同编码，但无法判断同一 sender 内是否出现顺序结构。本轮让每个 sender 连续发送两个 token，并加入资源身份置换控制。要回答的问题是：第二个 token 是否携带独立的位置相关信息，两个 token 是否形成依赖于前缀的序列分布，以及这些性质是否受伙伴覆盖和资源槽位不对称影响。",
        "",
        "本轮仍研究有限的 grounded 通信协议，不把两个离散 token 直接称为自然语言。功能指标是 receiver 在留出地图上的联合动作成功；形式指标是同类型 sender 的 token 一致率；序列指标包括 token 位置的信息量、token1 对 token0 前缀的敏感性和 token0/token1 的统计依赖。",
        "",
        "## 实验设计",
        "",
        "正式矩阵有 4 个 seed、3 个 partition、3 种伙伴拓扑和 2 种资源分配，共 72 条独立链。伙伴拓扑为固定 A、轮换 A/B、随机 A/B/C。资源分配 canonical 使用 apple、banana、orange；cyclic 将资源槽位映射为 banana、orange、apple，保持地点世界与训练/测试地图不变，用来检查结果是否随资源身份绑定在某个架构槽位上。",
        "",
        "三个 sender 各连续发送两个 token，消息顺序是 resource0-token0、resource0-token1、resource1-token0、resource1-token1、resource2-token0、resource2-token1。名义消息空间为 7⁶=117649；为避免保存大量无行为意义的全码表，评估文件只物化 120 张测试地图上实际出现的六 token 码，并保存每一行到唯一码的索引。这不会改变已报告的行为分数。",
        "",
        "私有视觉前端继承自 v0.28 并冻结，所有通信模块从随机状态初始化。训练地图为每个 partition 的 60 个三地点排列，target 集合为其余 60 个排列；每个 checkpoint 对 A、B、C 三种评估拓扑都完整扫描 120 张地图。收益为 `(1/6)*sum(correct_resources)+0.5*all_three_correct`，没有 token 监督。",
        "",
        f"设计文件：{link(ROOT / 'two_token_design.json')}；执行绑定：{link(out / 'invocation.json')}；训练完成：{link(out / 'training_complete.json')}。",
        "",
        "## 结果一：双 token 协议可以形成，但伙伴覆盖仍决定跨拓扑功能",
        "",
        "下表是代际 1200 步、三种评估拓扑合并后的 target-60 联合 J；括号为 12 条 seed×partition 链的 SD。",
        "",
        "| 资源分配 | 训练拓扑 | target J（0） | target J（1200） | 变化 |",
        "|---|---|---:|---:|---:|",
    ]
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            x0 = effects[assignment][condition]["0"]["mean"]
            x4 = effects[assignment][condition]["1200"]["mean"]
            lines.append(f"| {ASSIGNMENT_LABELS[assignment]} | {CONDITION_LABELS[condition]} | {mean_sd(effects[assignment][condition]['0'])} | {mean_sd(effects[assignment][condition]['1200'])} | {pp(x4 - x0)} |")
    lines += [
        "",
        "canonical 资源顺序下，固定 A、轮换 A/B、随机 A/B/C 的 pooled target J 分别从约 0.44%、0.49%、0.31% 上升到 11.91%、25.15%、32.77%。cyclic 顺序给出相近的学习曲线，最终为 12.05%、27.01%、29.48%。因此增加第二个 token 没有破坏 grounded 学习，伙伴覆盖仍然是最主要的功能因素。",
        "",
        "## 结果二：评估拓扑区分局部关系码与公共协议",
        "",
        "代际 1200 步按评估拓扑拆分的 target J 如下。",
        "",
        "| 资源分配 | 训练拓扑 | 评估 A | 评估 B | 评估 C |",
        "|---|---|---:|---:|---:|",
    ]
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            vals = [summary[assignment][condition]["1200"][s]["target_J"] for s in SCHEDULES]
            lines.append(f"| {ASSIGNMENT_LABELS[assignment]} | {CONDITION_LABELS[condition]} | {mean_sd(vals[0])} | {mean_sd(vals[1])} | {mean_sd(vals[2])} |")
    lines += [
        "",
        "固定 A 的双 token 协议仍主要服务于 A 关系；B/C 评估接近零。轮换 A/B 提高 A/B 可读性，但 C 仍较弱。随机 A/B/C 在三种评估拓扑上最接近。这与 v0.39 的三 token 结果一致，说明序列长度增加后，伙伴覆盖压力仍能决定协议是关系特定还是跨伙伴共享。",
        "",
        "## 结果三：第二个 token 携带位置相关信息，但尚不能称为语法",
        "",
        "下表将 A/B/C 评估拓扑平均，报告代际 1200 步的序列诊断。NMI 是 token 对 sender 所见地点的归一化互信息；prefix TV 是所有 token0 前缀下 p(token1|token0) 相对其前缀均值的平均总变差。",
        "",
        "| 资源分配 | 训练拓扑 | token0 NMI | token1 NMI | token pair NMI | token0–token1 MI | prefix TV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            values = {
                key: sum(summary[assignment][condition]["1200"][s][key]["mean"] for s in SCHEDULES) / 3
                for key in ("nmi_token0_position", "nmi_token1_position", "nmi_pair_position", "token_dependency_mi", "prefix_sensitivity_tv")
            }
            lines.append(f"| {ASSIGNMENT_LABELS[assignment]} | {CONDITION_LABELS[condition]} | {pct(values['nmi_token0_position'])} | {pct(values['nmi_token1_position'])} | {pct(values['nmi_pair_position'])} | {pct(values['token_dependency_mi'])} | {pct(values['prefix_sensitivity_tv'])} |")
    lines += [
        "",
        "在六个条件中，token0 对地点的 NMI 约为 93%–96%，token1 约为 77%–81%，完整 token pair 达到 100%。这表明第二个 token 不是纯粹的空白或固定填充，而是携带了额外的可预测位置信息；token0–token1 的高依赖和 6.6%–8.1% 的 prefix TV 也说明第二个 token 的分布会随第一 token 改变。",
        "",
        "但这还不是序列语法的证据。当前任务只有六个地点，sender 可以让两个 token 冗余地编码同一个地点；pair NMI 的 100% 反而说明存在这种简单的完整编码。要证明顺序结构具有可组合的意义，必须加入新位置、新资源、token 删除/交换干预以及跨主体迁移测试。",
        "",
        "## 结果四：资源置换揭示了槽位不对称",
        "",
        "cyclic−canonical 的代际 1200 步 target J 变化如下（按评估拓扑拆分）。",
        "",
        "| 训练拓扑 | 评估 A | 评估 B | 评估 C |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        vals = [assignment_effects[condition][s]["cyclic_minus_canonical_target_J"] for s in SCHEDULES]
        lines.append(f"| {CONDITION_LABELS[condition]} | {pp(vals[0])} | {pp(vals[1])} | {pp(vals[2])} |")
    lines += [
        "",
        "资源置换的影响依赖伙伴拓扑：轮换 A/B 下 cyclic 提高 A/B/C 评估约 1.6–2.2 个百分点，而随机 A/B/C 下降低约 2.7–4.2 个百分点。这个交互说明资源身份和伙伴覆盖共同影响学习，而不是一个可以忽略的实现细节。由于当前只有一个 cyclic 置换，下一轮需要遍历全部六种资源排列，并把角色排列也随机化。",
        "",
        "## 机制解释与核心问题",
        "",
        "1. **非语言表征可以支持双 token 共同协议。** 冻结的视觉表征、离散采样和共享 grounded 后果足以让 sender pair 编码地点，并让 receiver 在留出地图上恢复三个资源位置。",
        "2. **第二 token 已经有信息功能，但还没有被证明具有词法或语法功能。** 当前结果支持“序列中的后续位置可以承载增量信息”，不支持“模型已经产生了语言语法”。",
        "3. **伙伴覆盖比消息长度更先决定公共性。** 固定 A 能形成局部关系码；轮换和随机拓扑使同一 token 位置必须跨主体保持可读，并提高公共协议的机会。",
        "4. **资源对称性是必要控制。** cyclic 与 canonical 的结果差异表明，若不交换资源身份，资源 2 的较高准确率可能被误读为通信结构本身。",
        "",
        "因此，本轮对核心问题的回答是：已有非语言视觉能力、离散序列记忆和共同后果反馈，足以形成一个有限的双 token grounded 协议；第二 token 能携带与地点相关的信息，并表现出前缀依赖。是否形成可传递、可重组的“语言”仍需跨主体迁移和干预实验。",
        "",
        "## 限制",
        "",
        "- 代际 0 是本轮从随机通信头开始的形成过程，不是 v0.40 的代际 endpoint transfer；两者回答不同问题。",
        "- 私有视觉前端继承自 v0.28 并冻结；没有测试视觉概念的共同学习。",
        "- 名义六 token 码空间为 117649，但行为评估只需 120 张测试地图上出现的稀疏码；尚未测试未见组合码的系统性泛化。",
        "- 共享奖励和梯度更新集中实现，主体没有异步、噪声或局部观察限制。",
        "- 只使用 canonical 与一个 cyclic 资源置换，尚未形成完整的资源/角色对称性基线。",
        "",
        "## 可复核性",
        "",
        f"正式执行包含 {t['runs']} 条链、{t['updates_per_run']} 次更新/链、{t['protocol_tables']} 张协议表和 {t['trace_files_expected']} 个轨迹文件。独立 NumPy 重分析通过 {d['checks']} 项检查、{d['scalar_comparisons']} 个标量比较，最大保存指标差异为 {d['maximum_metric_absolute_difference']}；轨迹审计通过 {a['checks']} 项检查，覆盖 {a['trace_files']} 个轨迹文件和 {a['trace_rows']} 行，最大重放误差为 {a['maximum_replay_absolute_difference']}。",
        "",
        f"结果图：{link(out / 'figures/01_two_token_outcomes.png')}；视觉 QA：{link(out / 'visual_qa.json')}；图形元数据：{link(out / 'plot_metadata.json')}。",
        f"独立分析：{link(out / 'two_token_analysis.json')}；重分析验证：{link(out / 'two_token_raw_validation.json')}；轨迹审计：{link(out / 'two_token_audit.json')}。",
        f"训练源：{link(ROOT / 'two_token_train.py')}；分析源：{link(ROOT / 'two_token_analysis.py')}；审计源：{link(ROOT / 'two_token_audit.py')}。",
        "",
        "## 下一步实验",
        "",
        "先遍历全部资源和角色排列，加入 token 删除、交换和新地点干预，检验第二 token 的信息是否可组合而不是冗余编码；随后把这个双 token endpoint 接入 v0.40 的代际替换，直接比较序列协议的跨主体保真度、prefix TV 漂移和 pair code 重编码。",
    ]
    report = out / "三资源双token协议形成研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
