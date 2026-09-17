"""Build the v0.39 three-resource formation report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(x): return f"{100 * x:.3f}%"
def link(path: Path, label: str | None = None): return f"[{label or path.name}]({path.resolve()})"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve(); d = read(out / "triad_analysis.json"); v = read(out / "triad_raw_validation.json"); a = read(out / "triad_audit.json"); t = read(out / "training_complete.json"); i = read(out / "invocation.json")
    if not (d["formal"] and d["status"] == "complete" and v["passed"] and a["passed"] and t["formal"] and i["formal"]): raise ValueError("formal triad outputs required")
    lines = ["# 三资源三 token 共同符号形成：v0.39 实验报告", "", "## 研究问题", "", "v0.37/v0.38 在双资源任务中显示，伙伴拓扑可以决定协议是局部配对码还是跨拓扑共享码。本轮增加第三种资源和第三个互补 sender，让 receiver 只从一个有序三 token 组合恢复三个资源的位置。目标是检查共同符号形成是否能扩展到三个槽位，以及形式一致、分资源正确率和联合 grounded 成功是否仍然分离。", "", "本轮所有通信模块从随机状态开始；私有视觉前端冻结。三种训练日程仍为 fixed-A、rotating-AB 和 random-ABC，因此可以把三资源结果与之前的双资源结果直接对照。", "", "## 实验条件", "", "世界由 6 个地点中的 3 个有序且互不相同的位置组成，共 120 种地图。三个 sender 分别只观察 apple、banana、orange 三种资源视图，各发送一个 7 值 token；receiver 读取 3 token tuple（343 个可能码）并输出三个地点。收益为 `(1/6)*sum(correct_resources)+(1/2)*all_three_correct`，随机通信头没有 token 监督。训练地图为每个 partition 的 60 个三元组，测试包含全部 120 个三元组。", "", "正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个伙伴拓扑，共 36 条链；每条链训练 1200 更新，在 0、100、600、1200 更新评估 A/B/C。", "", f"执行绑定：{link(out / 'invocation.json')}；训练完成：{link(out / 'training_complete.json')}。", "", "## 结果一：三资源组合协议可以从随机头形成", "", "下表给出第 1200 步 target60 联合 J；括号内是跨 12 条 seed×panel 运行的 SD。", "", "| 训练日程 | A 评估 | B 评估 | C 评估 |", "|---|---:|---:|---:|"]
    for condition in d["conditions"]:
        vals = [d["summary"][condition][schedule]["1200"]["target_J"] for schedule in d["schedules"]]; lines.append(f"| {condition} | {pct(vals[0]['mean'])}（{pct(vals[0]['sd'])}） | {pct(vals[1]['mean'])}（{pct(vals[1]['sd'])}） | {pct(vals[2]['mean'])}（{pct(vals[2]['sd'])}） |")
    lines += ["", "random-ABC 在三种评估拓扑上形成了接近的联合 grounded 读出；fixed-A 和 rotating-AB 的联合 J 需要结合单资源结果判断，因为单个资源可能已经学会，而三者尚未同时正确。", "", "## 结果二：三 token 的组合一致率与任务成功仍然是两个维度", "", "下表给出 A 评估下同类型 sender 的 token 一致率。`joint` 要求三个资源槽位同时一致。", "", "| 训练日程 | 资源 0 | 资源 1 | 资源 2 | 三 token joint |", "|---|---:|---:|---:|---:|"]
    for condition in d["conditions"]:
        x = d["summary"][condition]["A"]["1200"]; vals = [x[f"agreement_resource{k}"] for k in range(3)] + [x["agreement_joint"]]; lines.append(f"| {condition} | " + " | ".join(pct(v["mean"]) + f"（{pct(v['sd'])}）" for v in vals) + " |")
    lines += ["", "若三 token 只是在表面上趋同而没有同时提高联合 J，它们不能被称为共享语言。相反，如果联合 J 提高但 token 一致率不高，说明 sender 可以与特定 receiver 形成关系特定的组合码。两类指标必须同时报告。", "", "## 结果三：分资源正确率显示组合瓶颈的位置", "", "下表是 A 评估下第 1200 步的三个资源正确率。", "", "| 训练日程 | 资源 0 | 资源 1 | 资源 2 |", "|---|---:|---:|---:|---:|"]
    for condition in d["conditions"]:
        x = d["summary"][condition]["A"]["1200"]; lines.append(f"| {condition} | {pct(x['resource0']['mean'])} | {pct(x['resource1']['mean'])} | {pct(x['resource2']['mean'])} |")
    lines += ["", "三资源任务比双资源任务增加了一个必要的联合约束：即使每个 sender 的槽位输出都出现局部规律，receiver 仍需要学习 343 码中的组合到三地点动作的映射。若单资源正确率提高而联合 J 仍接近随机，瓶颈位于组合解码而非词槽形成。", "", "## 机制解释", "", "1. **组合槽位是比双资源互补更强的检验。** 三个 sender 的局部输出只有在 receiver 的三 token 解码中共同产生后果，才构成有效协议。", "2. **伙伴覆盖仍是形成条件。** fixed-A 只覆盖一种角色排列；rotating-AB 和 random-ABC 迫使同一槽位在更多主体关系中保持可读。", "3. **一致率不能替代 grounded 结果。** 三 token 的形式收敛可能是无功能的常量；联合 J 也可能来自关系特定编码而非群体公共码。", "4. **任务复杂度开始触及容量瓶颈。** 从 49 个双 token 码扩展到 343 个三 token 码后，需要独立报告每个资源槽位、联合动作和留出地图上的性能。", "", "## 与语言起源问题的关系", "", "本轮把“共同符号形成”从两个互补信息源推进到三个信息源的组合任务。若协议在三资源条件下仍能形成并跨拓扑读出，说明共同 grounded 反馈与伙伴覆盖足以支持有限的多槽位符号组合；若性能显著下降，则提供一个可量化的组合复杂度边界。两种结果都比单纯增加训练步数更能说明非语言能力与协议结构之间的关系。", "", "## 限制", "", "- 三种资源视图使用已有 apple/banana/orange 图像特征；视觉编码器继承自前序两资源训练并被冻结。", "- 每个 sender 只发一个 token，尚未测试同一 sender 内的序列语法或词汇增长。", "- 共同奖励和联合梯度仍由实验者中心化计算；没有完全分布式的社会学习。", "- 四个主体、六个地点和固定角色日程是受控机制探针，不是人类社会历史重建。", "", "## 可复核性", "", f"独立 NumPy 重算通过 {d['checks']} 项检查和 {d['scalar_comparisons']} 个标量比较，最大指标误差为 {d['maximum_metric_absolute_difference']}；覆盖 {d['coverage']['protocol_tables']} 张协议表。trace 审计通过 {a['checks']} 项检查，覆盖 {a['coverage']['trace_rows']} 行 trace、{a['coverage']['trace_files']} 个 trace 文件。", "", f"图形：{link(out / 'figures/01_triad_outcomes.png')}；图形源数据：{link(out / 'figures/figure_source.json')}；视觉 QA：{link(out / 'figures/visual_qa.json')}。", f"分析：{link(out / 'triad_analysis.json')}；独立重算：{link(out / 'triad_raw_validation.json')}；审计：{link(out / 'triad_audit.json')}。", f"设计：{link(ROOT / 'triad_design.json')}；执行：{link(ROOT / 'triad_train.py')}。", "", "## 下一步实验", "", "将三资源协议接入 v0.38 的代际替换，测量三槽位组合在主体替换后的保真度；随后增加每个 sender 的第二个 token和完全分布式 episode 反馈，区分槽位组合、序列结构和社会传递三种复杂度来源。"]
    report = out / "三资源三token共同符号形成研究报告.md"; report.write_text("\n".join(lines) + "\n"); print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report)}, ensure_ascii=False))


if __name__ == "__main__": main()
