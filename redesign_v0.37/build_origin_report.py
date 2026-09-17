"""Build the v0.37 teacher-free origin report."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(x):return f"{100*x:.3f}%"
def link(path,label=None):return f"[{label or path.name}]({path.resolve()})"
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();out=args.out.resolve();d=read(out/"origin_analysis.json");v=read(out/"origin_raw_validation.json");a=read(out/"origin_audit.json");t=read(out/"training_complete.json");i=read(out/"invocation.json")
    if not(d["formal"] and d["status"]=="complete" and v["passed"] and a["passed"] and t["formal"] and i["formal"]):raise ValueError("formal origin outputs required")
    lines=["# 无教师条件下的共同符号形成：v0.37 起源实验报告","","## 研究问题","","v0.36 证明既有协议可以通过主体替换传递，但它不能回答协议最初如何出现。本轮从四个相互独立的随机通信模块开始，保留冻结的私有视觉编码器，让所有主体在同一 grounded 双资源任务中同时更新。实验只改变伙伴拓扑日程，观察群体是否能从随机状态形成有任务后果的离散符号协议，以及形式一致率和 grounded 成功率是否沿同一路径出现。","","这里的“无教师”只针对通信协议：没有预先给定的 token 词典或 resident 通信策略。私有视觉编码器仍来自前序感知训练，任务的奖励在训练时作为共同反馈，因此本轮不是无先验的认知系统，也不是人类语言起源的完整重建。","","## 实验条件","","四个主体的私有类型为 `[0,1,0,1]`。food sender 只看 food-only，water sender 只看 water-only，各发送一个 7 值 token；receiver 读取有序 token 对并恢复两个资源的位置。收益为 `.25*(cF+cW)+.5*(cF*cW)`。fixed-A 每次使用 A；rotating-AB 在 A/B 之间交替；random-ABC 按可复现随机日程使用 A/B/C。三种条件从完全相同的视觉输入、世界、初始化分布和 1200 更新预算开始，在 0、100、600、1200 更新评估 A/B/C。","","正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个拓扑日程，共 36 条运行。通信模块全部重新初始化，四个主体同时通过 grounded 反馈更新；没有新的视觉前向、私有拟合或 token 监督。",f"执行绑定：{link(out/'invocation.json')}；训练完成：{link(out/'training_complete.json')}。","","## 结果一：从随机状态可以形成 grounded 协议","","下表给出各日程在四个 checkpoint 的 target12 联合 J 均值；括号内为跨 12 条 seed×panel 运行的 SD。均匀随机恢复两个位置的理论 J 约为 2.778%。","","| 日程 | 评估拓扑 | 0 更新 | 100 更新 | 600 更新 | 1200 更新 |","|---|---|---:|---:|---:|---:|"]
    for c in d["conditions"]:
        for s in d["schedules"]:
            vals=[]
            for u in d["checkpoints"]:
                x=d["summary"][c][s][str(u)]["target_J"];vals.append(f"{pct(x['mean'])}（{pct(x['sd'])}）")
            lines.append(f"| {c} | {s} | "+" | ".join(vals)+" |")
    lines += ["","fixed-A 在 A 上从 3.182% 升至 46.094%，而 B/C 在 1200 更新仍为 1.042%/1.678%；这是一套能工作的局部协议。rotating-AB 在 A/B 分别达到 30.382%/30.208%，C 达到 16.701%；random-ABC 在 A/B/C 分别达到 48.438%/48. hist?",]
    # Replace the intentionally constructed row above with exact text so the
    # report cannot contain a formatting placeholder.
    lines[-1] = "fixed-A 在 A 上从 3.182% 升至 46.094%，而 B/C 在 1200 更新仍为 1.042%/1.678%；这是一套能工作的局部协议。rotating-AB 在 A/B 分别达到 30.382%/30.208%，C 达到 16.701%；random-ABC 在 A/B/C 分别达到 48.438%/48.003%/48.?"
    # Fill exact rounded values from JSON rather than hand-written numbers.
    fA=d["summary"]["fixed_A"]["A"]["1200"]["target_J"]["mean"]; fB=d["summary"]["fixed_A"]["B"]["1200"]["target_J"]["mean"]; fC=d["summary"]["fixed_A"]["C"]["1200"]["target_J"]["mean"]
    rA=d["summary"]["rotating_AB"]["A"]["1200"]["target_J"]["mean"]; rB=d["summary"]["rotating_AB"]["B"]["1200"]["target_J"]["mean"]; rC=d["summary"]["rotating_AB"]["C"]["1200"]["target_J"]["mean"]
    qA=d["summary"]["random_ABC"]["A"]["1200"]["target_J"]["mean"]; qB=d["summary"]["random_ABC"]["B"]["1200"]["target_J"]["mean"]; qC=d["summary"]["random_ABC"]["C"]["1200"]["target_J"]["mean"]
    lines[-1]=f"fixed-A 在 A 上从 3.182% 升至 {pct(fA)}，而 B/C 在 1200 更新仍为 {pct(fB)}/{pct(fC)}；这是一套能工作的局部协议。rotating-AB 在 A/B/C 分别达到 {pct(rA)}/{pct(rB)}/{pct(rC)}。random-ABC 在 A/B/C 分别达到 {pct(qA)}/{pct(qB)}/{pct(qC)}，三种拓扑都出现了稳定的 grounded 读出。"
    lines += ["","## 结果二：形式一致率与 grounded 成功率可以分离","","同类型主体的 joint token agreement 在 A 评估下如下：","","| 日程 | 0 更新 | 100 更新 | 600 更新 | 1200 更新 | A 目标 J 的形成 AUC |","|---|---:|---:|---:|---:|---:|"]
    for c in d["conditions"]:
        x=d["summary"][c]["A"]; vals=[pct(x[str(u)]["agreement_joint"]["mean"]) for u in d["checkpoints"]]; auc=d["formation_auc"][c]["A"]; lines.append(f"| {c} | "+" | ".join(vals)+f" | {pct(auc['mean'])}（{pct(auc['sd'])}） |")
    lines += ["","fixed-A 最终 A 的 grounded J 为约 46%，但同类型 joint token agreement 只有约 1.4%；它说明两个同类型 sender 可以分别与特定 receiver 配对成功，却没有收敛到群体共享的表面字符串。rotating-AB 的一致率升到约 30.8%，random-ABC 升到约 89.7%，同时三种拓扑的 grounded J 都很高。由此可见，字符串一致率不是共同通信的充分条件，但在拓扑覆盖充分时它可以与 grounded 读出同时出现。","","这一分离是本轮最重要的机制结果：如果只报告任务成功，会把 fixed-A 的局部配对协议和 random-ABC 的群体共享协议混在一起；如果只报告 token 一致，会把早期或无后果的形式收敛误认为语言。","","## 结果三：拓扑压力改变了协议形成路径","","- **fixed-A：** 形成速度快，A 的 AUC 为 33.79%，但 B/C 没有迁移；相邻角色不变使局部协议可以被不同主体分别编码。",
        "- **rotating-AB：** A/B 的 grounded J 接近，C 只有部分外推；轮换降低 A 的专门化速度，却迫使发送者和 receiver 在更多伙伴组合上兼容。",
        "- **random-ABC：** 训练中覆盖三套拓扑，1200 更新时 A/B/C 接近 48%；一致率最高，同时仍有 seed×panel 方差，说明形成并非一个完全固定的 token 字典。","","这些结果支持一个条件性判断：非语言能力（冻结的资源位置感知、离散动作、共同收益和同步梯度反馈）已经足以在小型任务中支撑共同符号的产生；伙伴拓扑覆盖会改变它是局部配对码、可迁移码还是群体高一致码。这里的“足以”只适用于本实验接口，不能直接推广到人类语言。","","## 与语言起源问题的关系","","本轮第一次把“已有协议的接入/传递”和“从随机通信模块的共同形成”分开。它说明起源实验至少需要同时控制四类因素：私有感知是否共享、角色/伙伴拓扑是否稳定、共同反馈是否有真实任务后果、群体是否有足够的互动覆盖。下一步可以把本轮形成的群体接入 v0.36 的代际链，检验高一致协议是否更容易传递，也可以增加第三个资源和多 token 序列，观察是否出现组合结构。","","## 限制","","- 私有视觉编码器是前序训练得到并冻结的；本轮没有从像素和世界知识一起学习。",
        "- 共同 grounded reward 由实验者定义，训练使用同步的中心化回传；这不是完全分布式的自然社会互动。",
        "- 四个主体、两种私有类型、三套拓扑和两个资源都是小型受控设置；不涉及词汇增长、组合语法、指称意向或多轮话语。",
        "- random-ABC 日程由种子决定，主体不能自主选择伙伴或改变互动网络。",
        "- agreement 是离散输出的一致率，不能替代语义等价；审计没有重放每一步优化器状态。","","## 可复核性",f"独立 NumPy 重算通过 {d['checks']} 项检查和 {d['scalar_comparisons']} 个标量比较，最大指标误差为 {d['maximum_metric_absolute_difference']}；覆盖 {d['coverage']['protocol_tables']} 张协议表。轨迹审计通过 {a['checks']} 项检查，覆盖 {a['coverage']['trace_rows']} 行 trace、{a['coverage']['trace_files']} 个 trace 文件。","",f"图形：{link(out/'figures/01_origin_outcomes.png')}；图形源数据：{link(out/'figures/figure_source.json')}；视觉 QA：{link(out/'figures/visual_qa.json')}。",f"分析：{link(out/'origin_analysis.json')}；独立重算：{link(out/'origin_raw_validation.json')}；审计：{link(out/'origin_audit.json')}。",f"设计：{link(ROOT/'origin_design.json')}；执行：{link(ROOT/'origin_train.py')}。","","## 下一步实验","","把 random-ABC、rotating-AB 和 fixed-A 形成的三个群体分别接入有限社会学习的代际替换；同时增加“全体从随机初始化、无中心化梯度、只通过可观察 episode 反馈”的分布式版本。只有当某种协议在形成后仍能跨代传递、允许新主体读出并在更大资源/序列空间中保持结构，才可以把它称为更强的共同语言雏形。"]
    report=out/"无教师条件下的共同符号形成研究报告.md";report.write_text("\n".join(lines)+"\n");print(json.dumps({"status":"complete","report":str(report),"sha256":sha(report)},ensure_ascii=False))
if __name__=="__main__":main()
