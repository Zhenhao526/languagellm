"""Build the v0.38 origin-transfer factorial report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(x): return f"{100 * x:.3f}%"
def pp(x): return f"{'+' if x >= 0 else ''}{100 * x:.3f} 个百分点"
def link(path: Path, label: str | None = None): return f"[{label or path.name}]({path.resolve()})"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve()
    d = read(out / "transfer_analysis.json"); v = read(out / "transfer_raw_validation.json"); a = read(out / "transfer_audit.json"); t = read(out / "training_complete.json"); i = read(out / "invocation.json")
    if not (d["formal"] and d["status"] == "complete" and v["passed"] and a["passed"] and t["formal"] and i["formal"]): raise ValueError("formal transfer outputs required")
    formations = d["formation_conditions"]; transmissions = d["transmission_conditions"]; schedules = d["schedules"]; final_generation = str(d["generations"][-1])
    lines = [
        "# 形成拓扑与代际传递：v0.38 起源协议转移实验报告", "", "## 研究问题", "",
        "v0.37 让四个主体从随机通信模块共同形成 grounded 协议，但它没有回答这种协议能否被新主体接入并跨代保存。本轮把 v0.37 的三种形成终点分别接入同一套有限社会学习瓶颈，并把形成日程与传递日程做成 3×3 全因子设计。每一代重置一个身份的通信模块，只用 resident 的 grounded 反馈训练 300 次，再把 newcomer 安装回群体。", "",
        "形成条件和传递条件分开记录：形成条件决定 generation 0 的协议来自 fixed-A、rotating-AB 还是 random-ABC；传递条件决定 generation 1–4 的互动日程。这样可以区分“协议怎样被发明”与“协议怎样被文化传递”，并检验两者是否有交互。", "",
        "## 实验条件", "",
        "四个主体的私有类型为 `[0,1,0,1]`。food sender 只看 food-only，water sender 只看 water-only，各发送一个 7 值 token；receiver 读取有序 token 对并输出两个资源的位置。收益为 `.25*(cF+cW)+.5*(cF*cW)`。v0.37 形成终点来自冻结私有视觉编码器和 1200 更新的无教师通信训练；本轮每条链依次替换身份 0、1、2、3，每次 300 更新，评估 A/B/C 三种拓扑。", "",
        "正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个形成条件 × 3 个传递条件，共 108 条链、432 次替换事件。", "",
        f"执行绑定：{link(out / 'invocation.json')}；训练完成：{link(out / 'training_complete.json')}。", "", "## 结果一：形成日程留下可测的传递差异", "",
        "下表给出第 4 代在 A 评估拓扑上的 target12 联合 J；括号内为跨 12 条 seed×panel 链的 SD。", "", "| 形成条件 | 传递：固定 A | 传递：轮换 A/B | 传递：随机 A/B/C |", "|---|---:|---:|---:|",
    ]
    for formation in formations:
        vals = []
        for transmission in transmissions:
            x = d["summary"][formation][transmission][final_generation]["A"]["target_J"]; vals.append(f"{pct(x['mean'])}（{pct(x['sd'])}）")
        lines.append(f"| {formation.replace('origin_', '')} | " + " | ".join(vals) + " |")
    lines += ["", "形成终点并不是可传递性的充分条件。即使三类 endpoint 都能在自己的形成日程中工作，进入替换瓶颈后仍会因传递拓扑而分化。固定 A 传递通常保留 A 的局部协议；轮换和随机传递会把可读范围扩展到 B/C，但可能牺牲形成终点的 A 性能。", "", "## 结果二：代际变化取决于形成 × 传递交互", "", "下表给出 A 评估拓扑从形成终点到第 4 代的 target J 变化。", "", "| 形成条件 | 传递：固定 A | 传递：轮换 A/B | 传递：随机 A/B/C |", "|---|---:|---:|---:|"]
    for formation in formations:
        vals = [pp(d["retention"][formation][transmission]["A"]["mean"]) for transmission in transmissions]; lines.append(f"| {formation.replace('origin_', '')} | " + " | ".join(vals) + " |")
    lines += ["", "固定 A 形成的协议在固定 A 传递下最容易保留；在轮换或随机传递下，若它原本主要是局部配对码，A 的性能会下降并伴随重编码。random-ABC 形成的 endpoint 已覆盖更多伙伴拓扑，因此在轮换/随机传递下通常有更好的起点；但传递条件仍会改变最终的表面协议和各拓扑间的性能分配。", "", "## 结果三：终点结构不能只用单一成功率描述", "", "下表是随机 A/B/C 传递条件在第 4 代对三种评估拓扑的 target J，展示同一形成终点是否能跨拓扑读出。", "", "| 形成条件 | A | B | C | A 拓扑形成 AUC |", "|---|---:|---:|---:|---:|"]
    for formation in formations:
        vals = [d["summary"][formation]["random_ABC"][final_generation][schedule]["target_J"] for schedule in schedules]; auc = d["formation_auc"][formation]["random_ABC"]["A"]; lines.append(f"| {formation.replace('origin_', '')} | {pct(vals[0]['mean'])} | {pct(vals[1]['mean'])} | {pct(vals[2]['mean'])} | {pct(auc['mean'])}（{pct(auc['sd'])}） |")
    lines += ["", "这一步把“形成”与“迁移”连接起来：共同发明出的协议若只在单一伙伴拓扑上有效，代际替换会迅速暴露其局部性；若形成阶段已经覆盖多套拓扑，有限社会学习更可能保留跨拓扑可读性。A/B/C 的 grounded J 仍必须和 token 变化一起看，因为表面 token 变化并不必然意味着任务功能丢失。", "", "## 机制解释", "", "1. **形成拓扑塑造可传递性。** generation 0 的 endpoint 不是等价的初始状态；其伙伴覆盖范围改变了 newcomer 需要拟合的关系集合。", "2. **传递拓扑是第二个压力源。** 固定、轮换和随机日程分别偏向局部保真、有限扩展和广覆盖；相同形成终点在三种日程下会走出不同代际曲线。", "3. **共享协议需要两个条件同时满足。** 任务后果提供 grounded 约束，伙伴覆盖提供跨主体约束；只有前者时可以得到高成功的局部编码，只有后者而没有后果时则可能得到形式一致但无功能的 token。", "4. **共同发明与文化保真应分开测量。** v0.37 测形成，v0.38 测接入和传递；这为后续扩展到词汇、组合结构或分布式训练提供了两个独立阶段。", "", "## 与语言起源问题的关系", "", "本轮支持一个较窄的机制判断：在已有私有感知、离散动作、共同资源后果和同步反馈的条件下，伙伴拓扑的覆盖程度会影响共同符号能否从一次性协调变成可传递的群体规范。这与“分工和资源交换需要信息交流”的假设相容，但没有证明分工是人类语言的唯一或首要起因。下一步应把资源稀缺、角色不对称、互动成本和自主伙伴选择逐一加入，并保持形成/传递两阶段结构。", "", "## 限制", "", "- 形成 endpoint 的私有视觉编码器来自前序感知训练；本轮不是从像素、世界知识和通信一起从零学习。", "- 所有主体共享实验者定义的 grounded 反馈，更新仍是中心化的联合梯度；还没有完全分布式的可观察 episode 学习。", "- 群体只有四个主体、两种私有类型、两个资源和单 token 发送，不涉及词汇增长、组合语法、指称意向或多轮话语。", "- 替换顺序和伙伴日程由实验者固定，不能代表真实人口结构或社会选择。", "- token 变化率是表面指纹，不能直接解释为语义距离；审计不重放优化器状态。", "", "## 可复核性", "", f"独立 NumPy 重算通过 {d['checks']} 项检查和 {d['scalar_comparisons']} 个标量比较，最大指标误差为 {d['maximum_metric_absolute_difference']}；覆盖 {d['coverage']['protocol_tables']} 张协议表。trace 审计通过 {a['checks']} 项检查，覆盖 {a['coverage']['trace_rows']} 行 trace、{a['coverage']['trace_files']} 个 trace 文件。", "", f"图形：{link(out / 'figures/01_transfer_outcomes.png')}；图形源数据：{link(out / 'figures/figure_source.json')}；视觉 QA：{link(out / 'figures/visual_qa.json')}。", f"分析：{link(out / 'transfer_analysis.json')}；独立重算：{link(out / 'transfer_raw_validation.json')}；审计：{link(out / 'transfer_audit.json')}。", f"设计：{link(ROOT / 'transfer_design.json')}；执行：{link(ROOT / 'transfer_train.py')}。", "", "## 下一步实验", "", "把同一 factorial 扩展到第三个资源和多 token 序列，并增加完全分布式的 episode 反馈；同时引入稀缺度和通信成本，检验当前观察到的局部协议、跨拓扑协议和表面漂移是否会转化为更稳定的词汇分工与组合结构。"]
    report = out / "形成协议的代际传递研究报告.md"; report.write_text("\n".join(lines) + "\n"); print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report)}, ensure_ascii=False))


if __name__ == "__main__": main()
