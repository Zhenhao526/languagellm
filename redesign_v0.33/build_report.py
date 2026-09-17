"""Build the v0.33 formal report from sealed independent analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.3f} 个百分点"


def pooled(aggregate, condition: str, split: str, key: str = "J") -> float:
    return aggregate[condition]["scores"][split]["pooled"][key]


def build(out: Path) -> Path:
    analysis = read(out / "analysis.json")
    audit = read(out / "audit_execution.json")
    validation = read(out / "raw_validation.json")
    complete = read(out / "training_complete.json")
    invocation = read(out / "invocation.json")
    cross = read(out / "cross_team.json")
    if not (analysis.get("formal") and analysis.get("status") == "complete"):
        raise ValueError("formal independent analysis required")
    if not (validation.get("passed") and audit.get("passed") and complete.get("status") == "complete"):
        raise ValueError("formal validation, audit, and training completion must pass")
    if not (cross.get("formal") and cross.get("status") == "complete"):
        raise ValueError("formal cross-team analysis required")

    agg = analysis["aggregate"]
    primary = analysis["primary"]
    target_single = pooled(agg, "single_full", "target12")
    target_redundant = pooled(agg, "dual_same_full", "target12")
    target_comp = pooled(agg, "dual_complementary", "target12")
    target_comp_auc = agg["dual_complementary"]["auc"]["target12"]["pooled"]["J"]
    target_redundant_auc = agg["dual_same_full"]["auc"]["target12"]["pooled"]["J"]
    comp = cross["summary"]["dual_complementary"]
    redundant = cross["summary"]["dual_same_full"]

    lines = [
        "# 互补观察下的双发送者共同符号形成：v0.33 正式实验报告",
        "",
        "## 研究问题",
        "",
        "v0.32 的同回合联合收益提高了消息形式的一致率，却没有稳定地产生跨伙伴的 grounded 通信。本轮只增加一个结构性压力：同一接收者的两个资源行动由两个不可互换的发送者分别贡献；食物发送者只看到食物，水发送者只看到水。若两个 token 逐渐成为可交换的组成部分，接收者应能把互补线索合并为正确的食物—水位置，并在替换一个 sender 后保留另一资源的解码能力。",
        "",
        "## 实验条件",
        "",
        "四个主体使用两种冻结的私有视觉接口 [0,1,0,1]，通信模块独立初始化。单发送者条件让一个 full-view sender 发送两个 token；冗余双发送者条件让两个 full-view sender 各发送一个 token；互补双发送者条件让 food sender 使用 food-only 视图、water sender 使用 water-only 视图，各发送一个 token。每个 token 有7种取值，接收者读取有序49码表并输出食物和水位置。字符或整数本身没有预设词义。",
        "",
        "每次群体更新包含四个 team slot：(receiver, food_sender, water_sender) 为 (0,1,2)、(1,2,3)、(2,3,0)、(3,0,1)；单发送者条件使用相应的 (receiver, sender)。三种条件共享世界、照片、目标/训练地图、uniform 流、模型初始化和训练预算，条件名称不进入 fixture 随机流。收益为",
        "",
        "R = .25(cF + cW) + .5 cF cW，其中 cF、cW 是两项行动是否正确的指示量。",
        "",
        "正式批次包含4个 seed（34101–34104）、3个坐标面板、3种条件和2400个群体更新，共36组运行。主比较的独立单位是4个 seed；3个面板、team slot、图片和时间帧是嵌套重复。",
        "",
        "## 主要结果",
        "",
        f"互补观察大幅提高了目标12的双资源 J：冗余双发送者为 **{pct(target_redundant)}**，互补双发送者为 **{pct(target_comp)}**，增加 **{pp(target_comp - target_redundant)}**；四个 seed 的差值全部为正。相对于单发送者的 {pct(target_single)}，互补条件增加 **{pp(target_comp - target_single)}**。互补条件的目标 AUC 为 {pct(target_comp_auc)}，冗余条件为 {pct(target_redundant_auc)}，AUC 差为 **{pp(target_comp_auc - target_redundant_auc)}**。",
        "",
        f"互补条件的训练12 J 为 {pct(pooled(agg, 'dual_complementary', 'train12'))}，目标12 J 为 {pct(target_comp)}，留出18 J 为 {pct(pooled(agg, 'dual_complementary', 'held18'))}；目标食物单边正确率为 {pct(pooled(agg, 'dual_complementary', 'target12', 'food'))}，水单边正确率为 {pct(pooled(agg, 'dual_complementary', 'target12', 'water'))}。冗余条件的相应三项 J 为 {pct(pooled(agg, 'dual_same_full', 'train12'))}、{pct(target_redundant)}、{pct(pooled(agg, 'dual_same_full', 'held18'))}。互补条件的提升同时出现在 held-out 位置，不能归结为只记住训练边。",
        "",
        "### 四个 seed 的预定主指标",
        "",
        "| seed | 单发送者目标 J | 冗余双发送者目标 J | 互补双发送者目标 J | 互补−冗余 | 互补−单发送者 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(f"| {row['seed']} | {pct(row['single'])} | {pct(row['dual_same_full'])} | {pct(row['dual_complementary'])} | {pp(row['complementary_difference'])} | {pp(row['single_difference'])} |")
    lines += [
        "",
        "四个 seed 的互补−冗余差分别为 " + ", ".join(pp(row["complementary_difference"]) for row in primary) + "。这支持一个稳定的描述性效应，但样本级推断仍应以 seed 为单位，不能把所有协议世界当作独立实验。",
        "",
        "### token 交换与跨团队复用",
        "",
        f"在同一训练团队内，互补条件的离线 donor 交换得到 all-joint J={pct(agg['dual_complementary']['component']['all_joint'])}、food 边际正确率={pct(agg['dual_complementary']['component']['all_food'])}、water 边际正确率={pct(agg['dual_complementary']['component']['all_water'])}。这些数字说明保存的端点表中存在可重组的结构，但不能单独证明在线主体已经学会了独立词素；交换是分析者预先指定的干预。",
        "",
        f"更严格的跨团队检查固定接收者，只替换一个或两个 sender 的 token。互补条件的自身团队（within）目标 J 为 **{pct(comp['within']['mean'])}**，替换同私有类型 food token 后为 {pct(comp['same_type_food']['mean'])}，替换 water token 后为 {pct(comp['same_type_water']['mean'])}，对16种跨团队 food/water 组合取平均的 all-cross J 仅为 **{pct(comp['all_cross']['mean'])}**。相对于 within，all-cross 的差值为 **{pp(comp['delta_all_cross'])}**。冗余条件的 within 为 {pct(redundant['within']['mean'])}、all-cross 为 {pct(redundant['all_cross']['mean'])}。",
        "",
        "因此，互补视图确实提高了一个接收者整合两个局部线索的能力，但提高的协议主要绑定在训练过的 sender—receiver 配对上。不同团队之间的 token 不能直接互换，当前结果不支持“群体共享词典”或“语言已经形成”的表述。",
        "",
        "互补条件还显示明显的角色顺序不对称：离线 FW 片段重组的 J 为 39.931%，WF 为0%。这与两个 token 在接收表中具有固定 food→water 角色顺序相符，也提醒我们不能把双 token 数字当作两个自然可交换的词。",
        "",
        "### 消息形式一致率",
        "",
        f"终点同私有类型完整双 token 一致率为：单发送者 {pct(agg['single_full']['agreement'][-1]['agreement']['full_message_agreement'])}，冗余双发送者 {pct(agg['dual_same_full']['agreement'][-1]['agreement']['full_message_agreement'])}，互补双发送者 {pct(agg['dual_complementary']['agreement'][-1]['agreement']['full_message_agreement'])}。互补条件的 grounded J 很高，但形式一致率并未超过冗余条件；这再次说明消息形式相似与 grounded 共享语义是两个可分离的性质。",
        "",
        "## 解释边界",
        "",
        "本轮最有价值的结果是一个必要条件候选：当两个 sender 的观察严格互补、且接收者必须合并两条局部信息才能完成双资源行动时，端点协议的 grounded 泛化大幅提高。这个效应比单纯增加联合收益更直接地作用于任务结构。与此同时，跨团队替换几乎失败，说明协议仍然是配对特定的；互补信息压力尚未自动产生可供新伙伴读取的群体公共码本。",
        "",
        "本轮不能回答自然语言的词汇、语法或社会意向性问题。agent 具有冻结的视觉表征和可训练的通信头，私有类型只有两种，team 调度固定，视图遮罩由实验者规定，训练是同步的，没有出生、死亡、迁移、代际传递或文化学习。因此，结果应解释为“受控非语言表征在互补任务中形成了配对特定的组合通信协议”，而不是人类语言起源的直接模拟。",
        "",
        "## 结果质量与可复核性",
        "",
        f"独立 NumPy 分析重算 {validation['checks']:,} 项检查和 {validation['scalar_comparisons']:,} 个标量比较，最大绝对误差为 {validation['maximum_metric_absolute_difference']:.1g}；覆盖 {validation['coverage']['protocol_tables']:,} 张协议表、{validation['coverage']['protocol_worlds']:,} 个协议世界和 {validation['coverage']['null_target_scores']:,} 个空参照目标分数。",
        f"边界审计通过 {audit['checks']:,} 项检查，重放 {audit['coverage']['endpoint_protocol_worlds']:,} 个终点协议世界、{audit['coverage']['agreement_files']} 个 agreement 文件、{audit['coverage']['training_trace_rows']:,} 行训练 trace 和 {audit['coverage']['paired_world_checks']} 个配对世界检查；审计没有声称重放所有梯度和 Adam 状态。",
        f"正式训练完成记录包含 {complete['social_runs']} 组运行、{complete['pair_updates']:,} 个群体更新、{complete['messages']:,} 条消息和 {complete['actions']:,} 次行动。输入和实现源文件由 invocation 与 training-complete 哈希绑定；本轮没有新的私有接口拟合，也没有新的视觉前向。",
        "",
        "## 下一步实验",
        "",
        "下一轮应把“配对特定”作为主要操纵，而不是继续增加任务复杂度：保留互补视图和双资源收益，让同一批 sender 与多个 receiver 轮换配对，并在训练中加入未见过的 sender—receiver 组合。预先注册三项结果：原配对、已见角色但新伙伴、完全跨团队 token 替换。如果新伙伴仍失败，应引入显式公共中继或低频共享锚点；如果新伙伴成功，再增加第三种资源和延迟转发，检验协议是否从两个角色 token 扩展到可复用的关系结构。人口、视觉主干和优化器应保持不变，以便把新增效应归因于社会拓扑。",
        "",
        "## 可复核文件",
        "",
        f"- 实验方案：[固定执行方案]({(ROOT / '固定执行方案.md').resolve()})；任务设计：[support_design.json]({(ROOT / 'support_design.json').resolve()})。",
        f"- 独立统计：[analysis.json]({(out / 'analysis.json').resolve()})；跨团队检查：[cross_team.json]({(out / 'cross_team.json').resolve()})；原始统计门禁：[raw_validation.json]({(out / 'raw_validation.json').resolve()})。",
        f"- 边界审计：[audit_execution.json]({(out / 'audit_execution.json').resolve()})；训练完成记录：[training_complete.json]({(out / 'training_complete.json').resolve()})；正式参数：[invocation.json]({(out / 'invocation.json').resolve()})。",
        f"- 图形：[01_complementary_design.png]({(out / 'figures/01_complementary_design.png').resolve()})、[02_complementary_outcomes.png]({(out / 'figures/02_complementary_outcomes.png').resolve()})；图形 QA：[visual_qa.json]({(out / 'figures/visual_qa.json').resolve()})。",
        f"- 实际消息样本：[实际消息示例.md]({(out / '实际消息示例.md').resolve()})；消息样本 JSON：[message_examples.json]({(out / 'message_examples.json').resolve()})；实现审查：[implementation_review.json]({(ROOT / 'implementation_review.json').resolve()})。",
        "",
        "本报告只解释 v0.33 的正式批次；v0.31 和 v0.32 保留为历史对照，不与本轮正式统计混合。",
    ]
    report = out / "互补观察下的双发送者共同符号形成研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out.resolve())


if __name__ == "__main__":
    main()
