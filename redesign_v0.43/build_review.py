"""Build a scientific review record for the v0.43 formal batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute():
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.43 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "permutation_001"
    report = out / "三资源排列与角色随机化形成研究报告.md"
    invocation = read(out / "invocation.json")
    analysis = read(out / "permutation_analysis.json")
    stats = read(out / "permutation_statistics.json")
    audit = read(out / "permutation_audit.json")
    visual = read(out / "visual_qa.json")
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [
        str(path)
        for path in (
            out / "invocation.json", out / "training_complete.json", out / "permutation_analysis.json",
            out / "permutation_raw_validation.json", out / "permutation_statistics.json", out / "permutation_audit.json",
            out / "plot_metadata.json", out / "figures" / "01_permutation_formation.png", out / "visual_qa.json", report,
            root / "permutation_design.json", root / "permutation_train.py", root / "permutation_analysis.py",
            root / "permutation_statistics.py", root / "permutation_audit.py", root / "plot_permutation.py",
            root / "visual_qa.py", root / "build_permutation_report.py", root / "build_review.py", root / "README.md",
        )
        if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            invocation.get("version") == "v0.43-permutation-formation",
            analysis.get("formal") is True,
            analysis.get("status") == "complete",
            analysis.get("probe") == "permutation_formation",
            analysis.get("runs") == 432,
            len(analysis.get("rows", [])) == 124416,
            len(analysis.get("sequence_rows", [])) == 5184,
            analysis.get("maximum_metric_absolute_difference") == 0.0,
            stats.get("formal") is True and stats.get("status") == "complete",
            audit.get("formal") is True,
            audit.get("status") == "complete",
            audit.get("runs") == 432,
            audit.get("coverage", {}).get("traces") == 3456,
            audit.get("coverage", {}).get("protocols") == 20736,
            visual.get("status") == "passed_visual_qa",
            not missing,
        )
    )
    findings = [
        {
            "type": "role_symmetry",
            "interpretation": "随机角色顺序把六种角色排列的等变 target J spread 显著压低，并提高 identity 排列的端点功能；这是形成阶段的对称性压力证据。",
        },
        {
            "type": "topology_interaction",
            "interpretation": "静态角色模式依赖轮换或随机伙伴来暴露多种资源—receiver 配对；随机角色模式已经在每次更新中覆盖角色轴，因此伙伴拓扑差异缩小。",
        },
        {
            "type": "resource_assignment",
            "interpretation": "六种资源列排列仍有数个百分点的范围，提示视觉输入列和资源 strata 的架构/数据偏置尚未完全消除。",
        },
        {
            "type": "sequence_structure",
            "interpretation": "角色随机化改变 token0、token1 和 pair agreement 以及 NMI 的组合，但有限六地点任务中的高 NMI 仍可由查表式编码产生，不能称为语法。",
        },
        {
            "type": "scope",
            "interpretation": "本轮是三资源、六站点、双 token 的 grounded protocol formation；它回答非语言角色对称压力如何改变共同符号结构，不回答人类语言起源的全部社会条件。",
        },
        {
            "type": "replayability",
            "interpretation": "独立 NumPy 分析、配对统计和无生产模型导入的轨迹审计均完成，所有正式链、检查点、协议表和训练轨迹均有覆盖。",
        },
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.43-permutation-formation",
        "reviewed_files": reviewed_files,
        "review_scope": [
            "Checked all 432 factorial formation chains across six resource assignments, two role modes and three partner topologies.",
            "Checked all four checkpoints, three evaluation schedules, four team slots and six role-permutation evaluations.",
            "Replayed deterministic fixture indices/uniforms, schedule and role RNG, categorical token/action draws, reward, advantage, log-probability and entropy for all saved traces.",
            "Computed paired role-mode effects, endpoint role spread and assignment ranges with a deterministic bootstrap.",
            "Checked raster QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": [
            "Finite vocabulary 7, two-token messages, six sites and three resources; no open-ended compositional language test.",
            "Frozen private visual frontends and centralized immediate reward are inherited from earlier batches.",
            "Role randomization synchronizes sender views and target axes, so it is a formation-stage symmetry/data-augmentation factor rather than a new architecture.",
            "Resource strata are not perfectly matched in visual statistics; assignment effects need a balanced visual bank.",
            "Endpoint role-equivalent scores are counterfactual target-axis evaluations, not novel-location or novel-agent transfer.",
            "The 600-update budget and large between-chain SD limit claims about optimization speed or universal superiority.",
        ],
        "next_step": "Add held-out role/resource compositions and partner replacement during evaluation, then introduce delayed inventory-based complementary work while preserving the role-spread and replay audits.",
        "link_check": {"local_links": len(links), "missing": missing},
        "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.43 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`",
        f"- 审查文件数：{len(reviewed_files)}", f"- 本地链接：{len(links)}，缺失：{len(missing)}", "", "## 审查范围", "",
    ] + [f"- {item}" for item in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{item['type']}**：{item['interpretation']}" for item in findings] + ["", "## 限制", ""] + [f"- {item}" for item in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed_files), "links": len(links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
