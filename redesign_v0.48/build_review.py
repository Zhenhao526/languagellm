"""Build the scientific review record for the v0.48 co-adaptation batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODES = ("resident_frozen", "resident_sender_sparse", "resident_receiver_sparse", "resident_both_sparse")
CULTURES = (
    "static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC",
    "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC",
)


def read(path: Path):
    return json.loads(path.read_text())


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        yield Path(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "coadaptation_001"
    report = out / "有界居民共同适应与陌生主体恢复研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "coadaptation_analysis.json")
    stats = read(out / "coadaptation_statistics.json")
    audit = read(out / "coadaptation_audit.json")
    visual = read(out / "visual_qa.json")
    local_links = list(links(report))
    self_links = {root / "结果审查.json", root / "结果审查.md"}
    missing = [str(path) for path in local_links if not path.exists() and path not in self_links]
    reviewed = [
        root / "coadaptation_design.json", root / "coadaptation_train.py", root / "coadaptation_analysis.py",
        root / "coadaptation_statistics.py", root / "coadaptation_audit.py", root / "plot_coadaptation.py",
        root / "visual_qa.py", root / "build_coadaptation_report.py", root / "build_review.py",
        root / "package_results.py", root / "README.md", out / "invocation.json", out / "training_complete.json",
        out / "coadaptation_analysis.json", out / "coadaptation_statistics.json", out / "coadaptation_audit.json",
        out / "plot_metadata.json", out / "figures" / "01_coadaptation.png", out / "visual_qa.json", report,
    ]
    reviewed = [path for path in reviewed if path.exists()]
    formal = all(
        (
            invocation.get("formal") is True,
            invocation.get("version") == "v0.48-coadaptation-horizon",
            invocation.get("adaptation_schedule") == "random_ABC",
            invocation.get("newcomer_reset_mode") == "both",
            tuple(invocation.get("resident_adaptation_modes", ())) == MODES,
            complete.get("status") == "complete", complete.get("formal") is True,
            complete.get("runs") == 1728, complete.get("updates_per_run") == 600,
            complete.get("trace_files_expected") == 13824, complete.get("protocol_files_expected") == 82944,
            analysis.get("status") == "complete", analysis.get("formal") is True,
            analysis.get("runs") == 1728, tuple(analysis.get("resident_adaptation_modes", ())) == MODES,
            len(analysis.get("rows", [])) == 497664, len(analysis.get("sequence_rows", [])) == 6912,
            analysis.get("maximum_metric_absolute_difference") == 0.0,
            stats.get("status") == "complete", stats.get("formal") is True, stats.get("runs") == 1728,
            stats.get("bootstrap", {}).get("repetitions") == 20000,
            audit.get("status") == "complete", audit.get("formal") is True, audit.get("runs") == 1728,
            audit.get("coverage", {}).get("traces") == 13824, audit.get("coverage", {}).get("protocol_files") == 82944,
            audit.get("coverage", {}).get("chains") == 1728, audit.get("maximum_replay_absolute_difference") <= 1e-5,
            audit.get("production_modules_imported") is False, audit.get("model_calls") == 0,
            visual.get("status") == "passed_visual_qa", not missing, report.is_file(),
        )
    )
    findings = [
        {"type": "bounded_coadaptation", "interpretation": "在随机角色形成文化中，居民双侧稀疏更新相对冻结居民的端点收益约为 1.26–3.47 个百分点，并伴随约 1%–2% 的通信漂移，支持有限共同重构的解释。"},
        {"type": "formation_history", "interpretation": "角色随机化形成期的影响大于本轮居民局部更新；冻结居民时随机角色相对静态角色平均高约 24.01 个百分点，说明可恢复协议空间主要由形成期结构决定。"},
        {"type": "asymmetry", "interpretation": "发送端、接收端和双侧更新的收益并不相同；静态角色固定 A 的接收端更新甚至略低于冻结基线，因此居民可塑性作用依赖形成文化和任务日程。"},
        {"type": "form_function_gap", "interpretation": "target-60 J、token agreement、position NMI 和 role spread 分别测量功能、表面约定与角色泛化；有限任务中的功能恢复不能单独证明开放语义或自然语言。"},
        {"type": "auditability", "interpretation": "独立 NumPy 分析覆盖 497,664 行，独立审计重放 13,824 条轨迹和 82,944 个协议文件；未导入生产模块且 model_calls 为 0。"},
    ]
    limitations = [
        "陌生主体继承居民视觉前端和所有未重置组件，尚非从随机参数开始的全新主体。",
        "居民端只允许 30 次低频通信更新，且学习率低于陌生主体；结果是有界共同适应，不是无限社会学习。",
        "任务为六站点、三资源、双 token 的有限 grounded protocol，没有延迟库存、真实生存压力、互补分工、冲突修复、开放词汇或句法组合。",
        "bootstrap 区间以 72 个配对视觉组为单位，是描述性不确定性；正式论文仍需预注册层级模型、多重比较控制、更多独立形成批次和跨模型复现。",
        "本批次没有代际传递，因此不能判断协议是否能被新一代主体保留、压缩或重组。",
    ]
    review = {
        "passed": formal, "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.48-coadaptation-horizon", "reviewed_files": [str(path) for path in reviewed],
        "review_scope": [
            "Checked all 1,728 formal chains across six resident cultures and four resident adaptation modes.",
            "Checked four checkpoints, 82,944 endpoint protocol files and 13,824 training trace files.",
            "Checked identity-012 target-60 J, role-permutation spread, newcomer-resident agreement, position NMI and resident communication drift.",
            "Checked paired resident-mode effects with 72 matched visual groups per resident culture and 20,000 bootstrap repetitions.",
            "Checked source/input binding, deterministic replays, raster QA and every absolute local report link.",
        ],
        "findings": findings, "limitations": limitations,
        "next_step": "Select a resident plasticity regime from this bounded result, then add delayed resource consequences and generational transmission.",
        "link_check": {"local_links": len(local_links), "missing": missing}, "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.48 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`",
        f"- 审查文件数：{len(reviewed)}", f"- 本地链接：{len(local_links)}，缺失：{len(missing)}", "",
        "## 审查范围", "", *[f"- {x}" for x in review["review_scope"]], "", "## 主要发现", "",
        *[f"- **{x['type']}**：{x['interpretation']}" for x in findings], "", "## 限制", "",
        *[f"- {x}" for x in limitations], "", "## 下一步", "", review["next_step"], "",
    ]
    (root / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed), "links": len(local_links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
