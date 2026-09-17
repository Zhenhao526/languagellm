"""Build the v0.36 iterated-transmission report from independent JSON."""
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
    analysis = read(out / "iterated_analysis.json"); validation = read(out / "iterated_raw_validation.json"); audit = read(out / "iterated_audit.json"); training = read(out / "training_complete.json"); invocation = read(out / "invocation.json")
    if not (analysis["formal"] and analysis["status"] == "complete" and validation["passed"] and audit["passed"] and training["formal"] and invocation["formal"]): raise ValueError("formal iterated output required")
    lines = [
        "# 有限社会学习瓶颈下的代际传递：v0.36 迭代替换实验报告", "", "## 研究问题", "",
        "v0.35 说明新主体可以在线获取 resident 协议，但那仍是一次性适应。语言要成为群体文化，符号必须在主体替换后继续被传递。本轮让四主体群体从同一个 v0.34 fixed-A 终点开始，依次重置身份 0、1、2、3、0、1、2、3 的通信模块；每一代只训练被替换主体 300 次，其他三名居民和全部私有视觉编码器冻结，然后把新主体安装回群体。这样可以直接观察有限社会学习瓶颈下的 grounded 保真度、拓扑泛化和表面协议漂移。", "",
        "本轮比较三种训练日程：fixed-A 每次只使用 A；rotating-AB 在全局偶数更新使用 A、奇数更新使用 B；random-ABC 在预先确定的随机日程中均衡抽取 A/B/C。所有条件共享 generation 0 的初始居民状态，条件差异从第一次替换开始。", "",
        "## 实验条件", "",
        "任务沿用 v0.34/v0.35：food sender 只看 food-only，water sender 只看 water-only，各发一个 7 值 token；receiver 读取有序 token 对并输出两个资源的位置。收益为 `.25*(cF+cW)+.5*(cF*cW)`。每一代的 newcomer 保留被替换身份的私有视觉类型，只重置通信模块并通过 grounded 反馈更新 300 次。", "",
        "正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个日程，共 36 条链、288 次替换事件；每条链包含共同初始群体和 8 个替换代次，并在每代评估 A/B/C 三种终点评估拓扑。", "",
        f"执行绑定：{link(out / 'invocation.json')}；训练完成：{link(out / 'training_complete.json')}。", "",
        "## 结果一：fixed-A 传递并增强局部协议", "",
        "下表给出各条件在 generation 0、1、4、8 的 target12 联合 J 均值；括号内为跨 12 条 seed×panel 链的 SD。", "",
        "| 日程 | 评估拓扑 | g0 | g1 | g4 | g8 |", "|---|---|---:|---:|---:|---:|",
    ]
    for condition in analysis["conditions"]:
        for schedule in analysis["schedules"]:
            vals = []
            for generation in (0, 1, 4, 8):
                x = analysis["summary"][condition][str(generation)][schedule]["target_J"]; vals.append(f"{pct(x['mean'])}（{pct(x['sd'])}）")
            lines.append(f"| {condition} | {schedule} | " + " | ".join(vals) + " |")
    lines += [
        "", "fixed-A 链在 A 上从共同初始群体的 40.625% 升到第 8 代 58.536%，而 B/C 始终约 2%。这表示在每代只允许一个 newcomer 学习时，固定伙伴压力可以维持甚至强化局部 grounded 协议，但不能把它传递成拓扑无关的公共协议。", "",
        "## 结果二：轮换扩大可读范围，但会先破坏 native A", "",
        "rotating-AB 链的 A 从 40.625% 降到第 4 代 15.799%，随后回升到 23.264%；B 从 1.910% 单调升到 20.920%，C 从 1.190% 升到 5.122%。在同一共同初始群体和相同 300 更新瓶颈下，A/B 轮换把传递目标从一套专门协议改成多个拓扑都能部分读取的协议，代价是 A 的专门化性能和早期代际保真度。", "",
        "random-ABC 链在第 8 代的 A/B/C 分别为 36.111%、34.201% 和 34.722%，三种拓扑趋于接近；它的相邻代 A-token 变化率第 1 代为 18.400%，第 8 代仍为 9.031%，明显高于 fixed-A 的 3.941% 和 2.370%。随机拓扑覆盖因此带来最宽的终点可读范围，同时保留最大的表面漂移。", "",
        "| 日程 | 第 8 代 A | 第 8 代 B | 第 8 代 C | 第 8 代 A 相邻代 token 变化 |", "|---|---:|---:|---:|---:|",
    ]
    for condition in analysis["conditions"]:
        vals = [analysis["summary"][condition]["8"][s]["target_J"] for s in analysis["schedules"]]; drift = analysis["summary"][condition]["8"]["A"]["token_change_prev"]
        lines.append(f"| {condition} | {pct(vals[0]['mean'])} | {pct(vals[1]['mean'])} | {pct(vals[2]['mean'])} | {pct(drift['mean'])}（{pct(drift['sd'])}）|")
    lines += [
        "", "## 结果三：代际漂移和 grounded 保真度是两个维度", "",
        "token_change_prev 是同一评估拓扑下相邻代四个团队 token 输出的变化率；token_change_base 是相对 generation 0 的累计表面变化。它们是协议指纹，不是语义距离。", "",
        "| 日程 | g1 相邻代变化 | g4 相邻代变化 | g8 相邻代变化 | g8 相对 g0 变化 | g8 A held18 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in analysis["conditions"]:
        x1 = analysis["summary"][condition]["1"]["A"]; x4 = analysis["summary"][condition]["4"]["A"]; x8 = analysis["summary"][condition]["8"]["A"]
        lines.append(f"| {condition} | {pct(x1['token_change_prev']['mean'])} | {pct(x4['token_change_prev']['mean'])} | {pct(x8['token_change_prev']['mean'])} | {pct(x8['token_change_base']['mean'])} | {pct(x8['held_J']['mean'])} |")
    lines += [
        "", "fixed-A 的累计 token 变化到第 8 代为 13.630%，但 A 的 grounded J 升至 58.542%；因此表面变化并不等于文化失效。相反，random-ABC 的累计变化约 76.270%，同时 A/B/C 的 grounded J 都升高并接近；这说明在当前任务中可以出现“表面协议不断重排、任务后果仍被保持”的路径。需要用更强的语义等价和组合结构指标，才能判断这是同一语言的漂移还是一套套不同的再编码。", "",
        "## 机制解释", "",
        "1. **有限传递瓶颈足以产生明显日程效应。** 同一初始居民群体、相同 newcomer 预算下，fixed-A、rotating-AB、random-ABC 的代际曲线分化，说明社会互动拓扑本身会塑造可传递的符号结构。",
        "2. **局部高保真和广泛迁移存在权衡。** fixed-A 把 grounded 性能集中在 A；A/B 轮换把一部分性能移到 B，并对 C 有小幅外推；随机 A/B/C 让三类拓扑接近，但伴随更高的表面漂移。",
        "3. **代际传递不能只看 token 是否相同。** fixed-A 的 token 每代变化较小但 B/C 仍不可读；random-ABC 的 token 变化较大但三种拓扑都能工作。共同符号的评价必须同时包含 grounded 后果、跨拓扑读出和跨代稳定性。",
        "4. **这仍是文化传递，不是从无到有的语言发明。** generation 0 已有 resident 协议，newcomer 通过任务反馈重新对齐；下一步需要让初始群体也从随机通信模块开始，或让多个新主体在没有教师协议的条件下共同协商。",
        "",
        "## 与“语言诞生”问题的关系", "",
        "这轮实验把研究对象从“一个主体能否学会符号”推进到“符号能否通过人口替换和有限学习预算持续存在”。它支持把代际瓶颈、伙伴拓扑覆盖和 grounded 后果作为语言演化实验中的独立超参数。最有价值的下一步不是继续增加 token 数，而是建立一个两阶段设计：先测无教师群体的协议起源，再将形成的协议放入本轮的代际传递框架，区分发明、接入、保真和漂移。",
        "",
        "## 限制", "",
        "- 初始 generation 0 使用 v0.34 fixed-A 的既有协议；本轮不能单独证明语言从零产生。",
        "- newcomer 保留私有视觉编码器，只重置通信模块；没有测试端到端知觉学习或世界知识。",
        "- 群体固定为四个主体，替换顺序由实验者指定；没有出生、死亡、迁移、教师选择或真实代际生命周期。",
        "- 任务只有两个资源和两个单 token sender，不涉及词汇扩展、组合语法、指称意向、协商成本或多轮话语。",
        "- random-ABC 的随机日程由可复现种子预先决定；它不是在线主体自主改变互动网络。",
        "- token 变化率是表面输出诊断，不能直接解释为语义漂移；审计没有重放每一步优化器状态。",
        "",
        "## 可复核性", "",
        f"独立 NumPy 重算通过 {analysis['checks']} 项检查和 {analysis['scalar_comparisons']} 个标量比较，最大指标误差为 {analysis['maximum_metric_absolute_difference']}；覆盖 {analysis['coverage']['protocol_tables']} 张协议表。trace 审计通过 {audit['checks']} 项检查，覆盖 {audit['coverage']['trace_rows']} 行 trace、{audit['coverage']['trace_files']} 个 trace 文件。", "",
        f"图形：{link(out / 'figures/01_iterated_outcomes.png')}；图形源数据：{link(out / 'figures/figure_source.json')}；视觉 QA：{link(out / 'figures/visual_qa.json')}。", f"分析：{link(out / 'iterated_analysis.json')}；独立重算：{link(out / 'iterated_raw_validation.json')}；审计：{link(out / 'iterated_audit.json')}。", f"设计：{link(ROOT / 'iterated_design.json')}；执行：{link(ROOT / 'iterated_replacement.py')}。", "",
        "## 下一步实验", "",
        "建立无教师起源条件：从随机通信模块开始，让四个主体在相同 grounded 双资源任务中共同训练；比较固定角色、轮换角色和无固定角色三种互动组织，并把得到的群体终点接入 v0.36 的迭代替换链。这样可以把“共同发明”与“代际传递”放在同一套可审计指标上比较。",
    ]
    report = out / "有限社会学习瓶颈下的代际传递研究报告.md"; report.write_text("\n".join(lines) + "\n"); print(json.dumps({"status": "complete", "report": str(report), "sha256": sha(report)}, ensure_ascii=False))


if __name__ == "__main__": main()
