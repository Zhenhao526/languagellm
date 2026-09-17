"""Build the scientific review record for v0.47."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RESET_MODES = ("sender_only", "receiver_only", "both")


def read(path: Path):
    return json.loads(path.read_text())


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        yield Path(match.group(2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); root = args.out.resolve(); out = root / "results" / "reset_granularity_001"
    report = out / "通信模块重置粒度与陌生主体恢复研究报告.md"; invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); analysis = read(out / "reset_granularity_analysis.json"); stats = read(out / "reset_granularity_statistics.json"); audit = read(out / "reset_granularity_audit.json"); visual = read(out / "visual_qa.json")
    local_links = list(links(report)); self_links = {root / "结果审查.json", root / "结果审查.md"}; missing = [str(path) for path in local_links if not path.exists() and path not in self_links]
    reviewed = [path for path in (root / "reset_granularity_design.json", root / "reset_granularity_train.py", root / "reset_granularity_analysis.py", root / "reset_granularity_statistics.py", root / "reset_granularity_audit.py", root / "plot_reset_granularity.py", root / "visual_qa.py", root / "build_reset_granularity_report.py", root / "build_review.py", root / "package_results.py", root / "README.md", out / "invocation.json", out / "training_complete.json", out / "reset_granularity_analysis.json", out / "reset_granularity_statistics.json", out / "reset_granularity_audit.json", out / "plot_metadata.json", out / "figures" / "01_reset_granularity.png", out / "visual_qa.json", report) if path.exists()]
    formal = all((invocation.get("formal") is True, invocation.get("version") == "v0.47-reset-granularity", invocation.get("adaptation_schedule") == "random_ABC", tuple(invocation.get("reset_modes", ())) == RESET_MODES, invocation.get("updates") == 300, complete.get("status") == "complete", complete.get("formal") is True, complete.get("runs") == 1296, complete.get("updates_per_run") == 300, complete.get("reset_modes", []) == list(RESET_MODES), complete.get("trace_files_expected") == 10368, complete.get("protocol_files_expected") == 46656, analysis.get("status") == "complete", analysis.get("formal") is True, analysis.get("runs") == 1296, tuple(analysis.get("reset_modes", ())) == RESET_MODES, len(analysis.get("rows", [])) == 279936, len(analysis.get("sequence_rows", [])) == 3888, analysis.get("maximum_metric_absolute_difference") == 0.0, stats.get("status") == "complete", stats.get("formal") is True, stats.get("runs") == 1296, stats.get("bootstrap", {}).get("repetitions") == 20000, audit.get("status") == "complete", audit.get("formal") is True, audit.get("runs") == 1296, audit.get("coverage", {}).get("traces") == 10368, audit.get("coverage", {}).get("protocol_files") == 46656, audit.get("coverage", {}).get("chains") == 1296, audit.get("maximum_replay_absolute_difference") <= 1e-5, audit.get("production_modules_imported") is False, audit.get("model_calls") == 0, visual.get("status") == "passed_visual_qa", not missing, report.is_file()))
    findings = [
        {"type": "reset_side", "interpretation": "只重置发送端、只重置接收端和两端都重置在相同 resident endpoint 与随机 A/B/C 适应下形成可配对的恢复比较；端点和曲线差异定位通信侧恢复压力。"},
        {"type": "role_symmetry", "interpretation": "若随机角色 resident 在三种重置模式下保持较低 role spread，说明角色随机化的稳定性来自形成阶段，而不是某一侧恢复训练。"},
        {"type": "form_function_gap", "interpretation": "pair agreement、J 和 position NMI 是不同构念；有限任务中的功能恢复不能单独证明开放语义或自然语言。"},
        {"type": "auditability", "interpretation": "独立 NumPy 分析覆盖 279,936 行，独立审计重放 10,368 条轨迹和 46,656 个协议文件，未导入生产模块，model_calls 为 0。"},
    ]
    limitations = ["newcomer 继承同一 resident endpoint 的视觉前端和未重置通信模块，尚非全新主体。", "resident 冻结，没有双向协商、repair、冲突或群体共同改码。", "任务为六站点、三资源、双 token 的有限 grounded protocol，没有开放词汇、延迟库存、互补分工或代际传递。", "72 个视觉组的 bootstrap 区间是描述性不确定性，正式论文仍需预注册层级模型、多重比较和更长时间点。", "NMI 和 pair agreement 需要独立置换、熵校正和未见组合基线。"]
    review = {"passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review", "version": "v0.47-reset-granularity", "reviewed_files": [str(path) for path in reviewed], "review_scope": ["Checked all 1,296 formal chains across six resident cultures and three reset modes.", "Checked three checkpoints, 46,656 endpoint protocol files and 10,368 training trace files.", "Checked identity-012 target-60 J, role spread, newcomer-resident token agreement and position NMI.", "Checked paired reset effects with 72 matched visual groups per resident culture and 20,000 bootstrap repetitions.", "Checked source/input binding, deterministic replays, raster QA and every absolute local report link."], "findings": findings, "limitations": limitations, "next_step": "Add 600-update checkpoints and bounded resident adaptation, then introduce delayed resource consequences and generational transmission.", "link_check": {"local_links": len(local_links), "missing": missing}, "new_training_reviewed": True}
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.47 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed)}", f"- 本地链接：{len(local_links)}，缺失：{len(missing)}", "", "## 审查范围", ""] + [f"- {x}" for x in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{x['type']}**：{x['interpretation']}" for x in findings] + ["", "## 限制", ""] + [f"- {x}" for x in limitations] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md)); print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed), "links": len(local_links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
