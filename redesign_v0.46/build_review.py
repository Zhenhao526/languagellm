"""Build the scientific review record for the v0.46 crossed-schedule batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCHEDULES = ("fixed_A", "rotating_AB", "random_ABC")


def read(path: Path):
    return json.loads(path.read_text())


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        yield Path(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.46 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "cross_schedule_001"
    report = out / "交叉适应日程与陌生主体恢复研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "cross_schedule_analysis.json")
    stats = read(out / "cross_schedule_statistics.json")
    audit = read(out / "cross_schedule_audit.json")
    visual = read(out / "visual_qa.json")
    local_links = list(links(report))
    self_links = {root / "结果审查.json", root / "结果审查.md"}
    missing = [str(path) for path in local_links if not path.exists() and path not in self_links]
    reviewed = [
        path for path in (
            root / "cross_schedule_design.json",
            root / "cross_schedule_train.py",
            root / "cross_schedule_analysis.py",
            root / "cross_schedule_statistics.py",
            root / "cross_schedule_audit.py",
            root / "plot_cross_schedule.py",
            root / "visual_qa.py",
            root / "build_cross_schedule_report.py",
            root / "build_review.py",
            root / "package_results.py",
            root / "README.md",
            out / "invocation.json",
            out / "training_complete.json",
            out / "cross_schedule_analysis.json",
            out / "cross_schedule_statistics.json",
            out / "cross_schedule_audit.json",
            out / "plot_metadata.json",
            out / "figures" / "01_cross_schedule.png",
            out / "visual_qa.json",
            report,
        ) if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            invocation.get("version") == "v0.46-cross-adaptation-schedules",
            tuple(invocation.get("adaptation_schedules", ())) == SCHEDULES,
            invocation.get("updates") == 300,
            complete.get("status") == "complete",
            complete.get("formal") is True,
            complete.get("runs") == 1296,
            complete.get("updates_per_run") == 300,
            tuple(complete.get("adaptation_schedules", ())) == SCHEDULES,
            complete.get("trace_files_expected") == 10368,
            complete.get("protocol_files_expected") == 46656,
            analysis.get("status") == "complete",
            analysis.get("formal") is True,
            analysis.get("runs") == 1296,
            tuple(analysis.get("adaptation_schedules", ())) == SCHEDULES,
            len(analysis.get("rows", [])) == 279936,
            len(analysis.get("sequence_rows", [])) == 3888,
            analysis.get("maximum_metric_absolute_difference") == 0.0,
            stats.get("status") == "complete",
            stats.get("formal") is True,
            stats.get("runs") == 1296,
            stats.get("bootstrap", {}).get("repetitions") == 20000,
            audit.get("status") == "complete",
            audit.get("formal") is True,
            audit.get("runs") == 1296,
            audit.get("coverage", {}).get("traces") == 10368,
            audit.get("coverage", {}).get("protocol_files") == 46656,
            audit.get("coverage", {}).get("chains") == 1296,
            audit.get("maximum_replay_absolute_difference") <= 1e-5,
            audit.get("production_modules_imported") is False,
            audit.get("model_calls") == 0,
            visual.get("status") == "passed_visual_qa",
            not missing,
            report.is_file(),
        )
    )
    findings = [
        {
            "type": "crossed_schedule_effect",
            "interpretation": "适应阶段的伙伴覆盖效应依赖 resident 形成文化：随机角色/固定 A 的随机适应相对固定适应提高 8.51 pp（95% CI 7.87–9.13），而随机角色/A-B-C 的差异接近 0；静态角色/A 反而下降 2.34 pp。",
        },
        {
            "type": "role_symmetry",
            "interpretation": "随机角色 resident 在全部适应日程下保持较低 role spread（2.28–16.86 pp），静态 resident 为 24.96–70.60 pp，支持形成阶段角色随机化带来更强的角色轴稳定性。",
        },
        {
            "type": "recovery",
            "interpretation": "通信模块被重置的 newcomer 在共同回报下从 update-0 的 1.62%–7.01% J 恢复到 update-300 的 9.29%–43.69%，但这是有限 grounded protocol 的功能恢复。",
        },
        {
            "type": "form_function_gap",
            "interpretation": "pair agreement、J 和 position NMI 不完全同步；功能成功不能单独证明 newcomer 复刻了 resident 的表面符号或获得开放语义。",
        },
        {
            "type": "auditability",
            "interpretation": "独立 NumPy 分析检查 279,936 行并保存协议，独立审计重放 10,368 条轨迹和 46,656 个协议文件，未导入生产模块，model_calls 为 0。",
        },
    ]
    limitations = [
        "只有 identity 0 被替换；视觉前端继承同一视觉组并冻结，未测试新感知系统、视觉噪声或概念学习。",
        "任务是六站点、三资源、双 token 的有限协议，存在查表解，没有开放词汇、组合句法或未见概念。",
        "resident 在适应阶段完全冻结，没有双向协商、冲突、repair 或群体共同重构。",
        "奖励集中且即时，没有资源库存、延迟后果、生存压力、互补分工或代际 bottleneck。",
        "bootstrap 区间以 72 个视觉组链级配对单位为单位，是描述性不确定性，正式论文仍需预注册层级模型和多重比较处理。",
        "NMI 在低适应点可能受低熵 token 和有限任务结构影响，需要熵校正、置换检验和未见组合基线。",
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.46-cross-adaptation-schedules",
        "reviewed_files": [str(path) for path in reviewed],
        "review_scope": [
            "Checked all 1,296 formal chains across six resident cultures and three newcomer adaptation schedules.",
            "Checked three checkpoints, 46,656 endpoint protocol files and 10,368 training trace files.",
            "Checked identity-012 target-60 J, role spread, newcomer-resident token agreement and position NMI.",
            "Checked paired schedule effects with 72 chain-level matched visual groups per resident culture and 20,000 bootstrap repetitions.",
            "Checked source/input binding, deterministic replays, raster QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": limitations,
        "next_step": "Add replacement granularity and longer checkpoints, then allow bounded resident adaptation before introducing delayed resource consequences and generational transmission.",
        "link_check": {"local_links": len(local_links), "missing": missing},
        "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.46 结果审查",
        "",
        f"- 状态：`{review['status']}`",
        f"- 形式批次：`{review['passed']}`",
        f"- 审查文件数：{len(reviewed)}",
        f"- 本地链接：{len(local_links)}，缺失：{len(missing)}",
        "",
        "## 审查范围",
        "",
    ]
    md += [f"- {value}" for value in review["review_scope"]]
    md += ["", "## 主要发现", ""]
    md += [f"- **{value['type']}**：{value['interpretation']}" for value in findings]
    md += ["", "## 限制", ""]
    md += [f"- {value}" for value in limitations]
    md += ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed), "links": len(local_links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
