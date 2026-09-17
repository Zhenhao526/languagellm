"""Build the v0.32 formal report from the sealed independent analysis."""
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


def score(aggregate, condition: str, split: str, key: str = "J") -> float:
    return aggregate[condition]["scores"][split]["pooled"][key]


def curve_score(aggregate, condition: str, update: int, split: str = "target12", key: str = "J") -> float:
    row = next(item for item in aggregate[condition]["curve"] if item["update"] == update)
    return row["scores"][split]["pooled"][key]


def build(out: Path) -> Path:
    analysis = read(out / "analysis.json")
    audit = read(out / "audit_execution.json")
    validation = read(out / "raw_validation.json")
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    if not (analysis.get("formal") and analysis.get("status") == "complete"):
        raise ValueError("formal independent analysis required")
    if not (validation.get("passed") and audit.get("passed") and complete.get("status") == "complete"):
        raise ValueError("formal validation, audit, and training completion must pass")

    agg = analysis["aggregate"]
    primary = analysis["primary"]
    fixed_target = score(agg, "fixed_partners", "target12")
    rotating_target = score(agg, "rotating_partners", "target12")
    fixed_train = score(agg, "fixed_partners", "train12")
    rotating_train = score(agg, "rotating_partners", "train12")
    fixed_held = score(agg, "fixed_partners", "held18")
    rotating_held = score(agg, "rotating_partners", "held18")
    fixed_food = score(agg, "fixed_partners", "target12", "food")
    rotating_food = score(agg, "rotating_partners", "target12", "food")
    fixed_water = score(agg, "fixed_partners", "target12", "water")
    rotating_water = score(agg, "rotating_partners", "target12", "water")
    fixed_pair = (score(agg, "fixed_partners", "target12", "food_partner_pair_J") + score(agg, "fixed_partners", "target12", "water_partner_pair_J")) / 2
    rotating_pair = (score(agg, "rotating_partners", "target12", "food_partner_pair_J") + score(agg, "rotating_partners", "target12", "water_partner_pair_J")) / 2
    fixed_agree = agg["fixed_partners"]["agreement"][-1]["agreement"]
    rotating_agree = agg["rotating_partners"]["agreement"][-1]["agreement"]
    fixed_auc = agg["fixed_partners"]["auc"]["target12"]["pooled"]["J"]
    rotating_auc = agg["rotating_partners"]["auc"]["target12"]["pooled"]["J"]
    fixed_recomb = agg["fixed_partners"]["recombination_null"]["pooled"]
    rotating_recomb = agg["rotating_partners"]["recombination_null"]["pooled"]

    lines = [
        "# 联合收益压力下的共同符号形成：固定伙伴与轮换伙伴",
        "",
        "## 研究目的",
        "",
        "本轮把上一轮的伙伴轮换操纵与同回合共同收益结合起来，检验一个更严格的条件：一对主体是否会因为只有共同完成任务才能获得额外回报，而形成能跨伙伴复用的离散通信协议。核心区分是消息形式是否趋同，以及消息是否仍然把冻结视觉状态传递为正确的资源位置。前者提高而后者不提高时，不能把结果解释为 grounded 语言形成。",
        "",
        "## 预注册式实验条件",
        "",
        "四个主体的私有视觉接口固定为 `[0,1,0,1]`，通信模块独立初始化。每个主体看到同一套冻结的96维视觉特征，只能发送两个 token；每个 token 有7种取值，共49种完整消息。目标是从消息中恢复食物位置和水位置。训练图与目标图、坐标面板、消息容量、Adam预算、熵权重和冻结输入均沿用 v0.31 的配对设计。",
        "",
        "每个匹配对的两个方向使用同一批世界和同一组资源位置，但方向使用独立的预先生成采样流。若 `c_iF,c_iW,c_jF,c_jW` 表示四个动作是否正确，本轮的同回合收益为",
        "",
        "`R = 0.125(c_iF+c_iW+c_jF+c_jW) + 0.5 c_iF c_iW c_jF c_jW`，基线为 `0.5`。",
        "",
        "前一项提供密集信号，后一项只在四个动作全部正确时支付额外奖励。固定伙伴条件始终训练 `(0,1)+(2,3)`；轮换伙伴条件在偶数步使用该配对，在奇数步使用 `(0,3)+(2,1)`。每组2400个群体更新，共4个来源、3个坐标面板和2个条件，即24组正式运行。",
        "",
        "## 主要结果",
        "",
        f"正式批次包含 {complete['social_runs']} 组运行、{complete['pair_updates']} 个群体更新、{complete['messages']:,} 条消息和 {complete['actions']:,} 次资源动作。目标12终点自然 J 在固定伙伴条件为 **{pct(fixed_target)}**，轮换伙伴条件为 **{pct(rotating_target)}**，差值为 **{pp(rotating_target - fixed_target)}**。四个独立来源的差值分别为 " + ", ".join(pp(row["difference"]) for row in primary) + "；其中两个为正、两个为负，因此不能把终点差异概括为稳定的伙伴轮换效应。",
        "",
        f"消息形式的一致率却出现了明显变化。同一私有类型、相同冻结视觉状态上的完整双 token 一致率从固定伙伴的 **{pct(fixed_agree['full_message_agreement'])}** 增至轮换伙伴的 **{pct(rotating_agree['full_message_agreement'])}**，增加 **{pp(rotating_agree['full_message_agreement'] - fixed_agree['full_message_agreement'])}**；四个来源的方向全部为正。token0 一致率为 {pct(fixed_agree['token0_agreement'])} → {pct(rotating_agree['token0_agreement'])}，token1 一致率为 {pct(fixed_agree['token1_agreement'])} → {pct(rotating_agree['token1_agreement'])}。",
        "",
        f"这形成了本轮最清楚的结果：轮换伙伴能使消息形式趋同，但这种趋同没有稳定转化为目标语义的跨伙伴传递。训练12 J 为 {pct(fixed_train)} → {pct(rotating_train)}，留出18 J 为 {pct(fixed_held)} → {pct(rotating_held)}；目标单边食物正确率为 {pct(fixed_food)} → {pct(rotating_food)}，水正确率为 {pct(fixed_water)} → {pct(rotating_water)}。同一资源需要两个伙伴都正确时，伙伴对均值仍只有 {pct(fixed_pair)} → {pct(rotating_pair)}。",
        "",
        "### 四个来源的预定主指标",
        "",
        "| 来源 | 固定伙伴目标 J | 轮换伙伴目标 J | 轮换−固定 | 固定伙伴 AUC | 轮换伙伴 AUC | AUC差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        fixed_row = next(x for x in analysis["seed_rows"] if x["seed"] == row["seed"] and x["condition"] == "fixed_partners")
        rotating_row = next(x for x in analysis["seed_rows"] if x["seed"] == row["seed"] and x["condition"] == "rotating_partners")
        lines.append(
            f"| {row['seed']} | {pct(row['fixed'])} | {pct(row['rotating'])} | {pp(row['difference'])} | "
            f"{pct(fixed_row['auc']['target12']['pooled']['J'])} | {pct(rotating_row['auc']['target12']['pooled']['J'])} | {pp(row['auc_difference'])} |"
        )
    lines += [
        "",
        f"按四个来源平均，目标终点差为 **{pp(analysis['primary_mean'])}**，目标 AUC 差为 **{pp(analysis['primary_auc_mean'])}**。目标 AUC 的描述性均值从 {pct(fixed_auc)} 增至 {pct(rotating_auc)}，但来源级 AUC 差仍为 " + ", ".join(pp(row["auc_difference"]) for row in primary) + "，所以它不能替代来源级异质性报告。",
        "",
        "### 终点协议与结构参照",
        "",
        "| 有序协议 | 固定伙伴 J | 轮换伙伴 J |",
        "|---|---:|---:|",
    ]
    pair_keys = sorted(agg["fixed_partners"]["pair_endpoint_J"])
    for key in pair_keys:
        lines.append(f"| `{key}` | {pct(agg['fixed_partners']['pair_endpoint_J'][key])} | {pct(agg['rotating_partners']['pair_endpoint_J'][key])} |")
    lines += [
        "",
        "FW/WF 片段重组只作为结构参照，不能被当作主体在线组合能力：",
        "",
        "| 条件 | FW 原始 | FW 199项整码双射参照均值 | FW 超出 | WF 原始 | WF 参照均值 | WF 超出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 固定伙伴 | {pct(fixed_recomb['recombine_FW_J']['observed'])} | {pct(fixed_recomb['recombine_FW_J']['null_mean'])} | {pp(fixed_recomb['recombine_FW_J']['excess'])} | {pct(fixed_recomb['recombine_WF_J']['observed'])} | {pct(fixed_recomb['recombine_WF_J']['null_mean'])} | {pp(fixed_recomb['recombine_WF_J']['excess'])} |",
        f"| 轮换伙伴 | {pct(rotating_recomb['recombine_FW_J']['observed'])} | {pct(rotating_recomb['recombine_FW_J']['null_mean'])} | {pp(rotating_recomb['recombine_FW_J']['excess'])} | {pct(rotating_recomb['recombine_WF_J']['observed'])} | {pct(rotating_recomb['recombine_WF_J']['null_mean'])} | {pp(rotating_recomb['recombine_WF_J']['excess'])} |",
        "",
        "两个条件的 FW 和 WF 上尾比例均为 0.005（199 个固定整码双射参照，观测值高于全部参照行）。这一结果说明保存的端点协议中存在可重组的结构信号，但重组是在离线分析中用人工指定的 donor 片段完成的，不能证明主体已自主形成词法或组合规则。",
        "",
        "## 结果质量与审计",
        "",
        f"独立统计从保存的 sender/receiver 表重算了 {validation['checks']:,} 个检查和 {validation['scalar_comparisons']:,} 个标量比较，最大绝对误差为 {validation['maximum_metric_absolute_difference']:.1g}。统计覆盖 {validation['coverage']['protocol_tables']:,} 个协议表、{validation['coverage']['protocol_worlds']:,} 个协议世界和 {validation['coverage']['null_target_scores']:,} 个空参照目标分数。",
        f"边界审计通过 {audit['checks']:,} 项检查，覆盖 {audit['coverage']['social_runs']} 组运行、{audit['coverage']['endpoint_sender_worlds_replayed']:,} 个终点 sender 世界、{audit['coverage']['endpoint_receiver_tables_replayed']} 个 receiver 表、{audit['coverage']['training_log_rows_counted']:,} 行训练日志和 {audit['coverage']['sampled_message_trajectories']:,} 条抽样通信轨迹。审计重放端点分类采样、receiver 解码、联合奖励和指定训练检查点的 fixture/trace；没有声称重放全部梯度与 Adam 状态。",
        "",
        "固定与轮换条件的初始通信状态、匹配槽世界、目标图、训练图、冻结特征和私有 checkpoint 均由正式输入哈希绑定。正式批次没有新 DINO 前向、没有新的私有能力拟合，也没有按成绩筛选来源或条件。",
        "",
        "## 结论边界",
        "",
        "本轮支持一个有限而清晰的机制判断：伙伴轮换和同回合联合收益足以推动不同主体副本采用更相似的消息形式，但在当前两 token、受控资源位置任务中，消息一致并不足以产生稳定的 grounded 多伙伴通信。联合收益本身也不是充分条件；四动作伙伴对正确率仍接近零，目标跨伙伴 J 的来源方向不一致。",
        "",
        "因此，本轮不能支持“已经产生语言”的结论，也不能把消息一致率当作语义共享的替代指标。更合适的解释是：社会压力先改变了协议的形式稳定性，而目标语义还缺少让多个发送者必须共同贡献到同一行动的因果结构。",
        "",
        "## 下一步实验",
        "",
        "下一轮只增加一个结构性压力：把同一接收者的一个行动拆成两个不可互换的发送者贡献（例如两个 sender 分别看到互补线索，receiver 只有合并两个消息才能定位资源），并保留固定伙伴、轮换伙伴、单发送者和随机消息四类对照。这样可以直接检验“共享收益”之外的“信息互补是否迫使消息出现可组合的部分”。若该操纵仍只提升形式一致率而不提升 held-out grounded 任务，再引入延迟第三方转发；不同时改变 agent 数、视觉接口和代际替换。",
        "",
        "## 可复核文件",
        "",
        f"- 实验方案：[固定执行方案]({(ROOT / '固定执行方案.md').resolve()})；任务设计：[support_design.json]({(ROOT / 'support_design.json').resolve()})。",
        f"- 独立统计：[analysis.json]({(out / 'analysis.json').resolve()})；边界审计：[audit_execution.json]({(out / 'audit_execution.json').resolve()})；原始统计门禁：[raw_validation.json]({(out / 'raw_validation.json').resolve()})。",
        f"- 训练完成记录：[training_complete.json]({(out / 'training_complete.json').resolve()})；正式运行参数：[invocation.json]({(out / 'invocation.json').resolve()})。",
        f"- 图形：[01_partner_design.png]({(out / 'figures/01_partner_design.png').resolve()})、[02_partner_outcomes.png]({(out / 'figures/02_partner_outcomes.png').resolve()})；图形 QA：[visual_qa.json]({(out / 'figures/visual_qa.json').resolve()})。",
        f"- 实际消息样本：[实际消息示例.md]({(out / '实际消息示例.md').resolve()})；实现审查：[implementation_review.json]({(ROOT / 'implementation_review.json').resolve()})。",
        "",
        "本报告只解释 v0.32 的正式批次；v0.31 和早期版本保留为历史对照，不与本轮的正式统计混合。",
    ]
    report = out / "联合收益压力下的共同符号形成研究报告.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "bytes": report.stat().st_size}, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.out.resolve())


if __name__ == "__main__":
    main()
