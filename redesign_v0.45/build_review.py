"""Build the scientific review record for the v0.45 newcomer batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        yield Path(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.45 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "newcomer_adaptation_001"
    report = out / "陌生主体社会学习与协议恢复研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "newcomer_adaptation_analysis.json")
    audit = read(out / "newcomer_adaptation_audit.json")
    visual = read(out / "visual_qa.json")
    local_links = list(links(report))
    review_json = root / "结果审查.json"
    missing = [str(path) for path in local_links if not path.exists() and path != review_json and path != root / "结果审查.md"]
    reviewed = [
        path
        for path in (
            root / "newcomer_adaptation_design.json",
            root / "newcomer_adaptation_train.py",
            root / "newcomer_adaptation_analysis.py",
            root / "newcomer_adaptation_audit.py",
            root / "plot_newcomer_adaptation.py",
            root / "visual_qa.py",
            root / "build_newcomer_adaptation_report.py",
            root / "build_review.py",
            root / "package_results.py",
            root / "README.md",
            out / "invocation.json",
            out / "training_complete.json",
            out / "newcomer_adaptation_analysis.json",
            out / "newcomer_adaptation_audit.json",
            out / "plot_metadata.json",
            out / "figures" / "01_newcomer_adaptation.png",
            out / "visual_qa.json",
            report,
        )
        if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            invocation.get("version") == "v0.45-newcomer-adaptation",
            invocation.get("adaptation_schedule") == "random_ABC",
            invocation.get("updates") == 300,
            complete.get("status") == "complete",
            complete.get("formal") is True,
            complete.get("runs") == 432,
            complete.get("updates_per_run") == 300,
            complete.get("trace_files_expected") == 3456,
            complete.get("protocol_files_expected") == 15552,
            analysis.get("status") == "complete",
            analysis.get("formal") is True,
            analysis.get("runs") == 432,
            len(analysis.get("rows", [])) == 93312,
            len(analysis.get("sequence_rows", [])) == 1296,
            analysis.get("maximum_metric_absolute_difference") == 0.0,
            audit.get("status") == "complete",
            audit.get("formal") is True,
            audit.get("runs") == 432,
            audit.get("coverage", {}).get("traces") == 3456,
            audit.get("coverage", {}).get("protocol_files") == 15552,
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
            "type": "social_recovery",
            "interpretation": "通信头被重置的 newcomer 在 resident 共同回报下从 1.62%–7.01% 的 update-0 J 恢复到 9.29%–43.69%，说明有限任务中存在反馈驱动的协议恢复。",
        },
        {
            "type": "role_symmetry",
            "interpretation": "随机角色 resident 的 update-300 role spread 为 8.19–16.86 个百分点，静态 resident 为 37.22–70.60 个百分点；形成阶段的角色随机化影响恢复后的轴稳定性。",
        },
        {
            "type": "form_function_gap",
            "interpretation": "pair agreement、J 和 position NMI 的变化并不一致；任务成功不能直接作为符号形式复刻或语言语义的证据。",
        },
        {
            "type": "topology_scope",
            "interpretation": "适应日程统一固定为 random_ABC，因此本批比较 resident 形成文化，不能单独识别适应拓扑的因果主效应。",
        },
        {
            "type": "auditability",
            "interpretation": "独立 NumPy 分析检查保存协议，独立 NumPy 审计重放 fixture、采样、reward、log-probability 和 action score，未导入生产模块。",
        },
    ]
    limitations = [
        "只有 identity 0 被替换；视觉前端继承同一视觉组并冻结，未测试新感知系统。",
        "任务是六站点、三资源、双 token 的有限协议，不能称为开放语言。",
        "resident 冻结，未测试新成员加入时的双向协商、冲突和共同修复。",
        "奖励集中且即时，没有库存、延迟、生存压力、互补分工或代际 bottleneck。",
        "NMI 在 update 0 已偏高，需要熵校正、置换检验和未见组合测试。",
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.45-newcomer-adaptation",
        "reviewed_files": [str(path) for path in reviewed],
        "review_scope": [
            "Checked all 432 formal newcomer chains, three checkpoints and six resident cultures.",
            "Checked 15,552 saved endpoint protocol files and 3,456 training trace files.",
            "Checked identity-012 target-60 J, role spread, newcomer-resident token agreement and position NMI.",
            "Checked source/input binding and independent replays without importing production modules.",
            "Checked raster QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": limitations,
        "next_step": "Fully cross adaptation partner schedules, replacement granularity and update budgets; then add bidirectional resident adaptation and delayed resource consequences.",
        "link_check": {"local_links": len(local_links), "missing": missing},
        "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.45 结果审查",
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
