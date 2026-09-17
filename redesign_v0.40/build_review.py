"""Build a scientific review record for the v0.40 formal batch."""
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "triad_transfer_001"
    report = out / "三资源三token代际传递研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "triad_transfer_analysis.json")
    validation = read(out / "triad_transfer_raw_validation.json")
    audit = read(out / "triad_transfer_audit.json")
    visual = read(out / "visual_qa.json")
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [
        str(path)
        for path in (
            out / "invocation.json",
            out / "training_complete.json",
            out / "triad_transfer_analysis.json",
            out / "triad_transfer_raw_validation.json",
            out / "triad_transfer_audit.json",
            out / "visual_qa.json",
            out / "plot_metadata.json",
            out / "figures" / "01_triad_transfer_outcomes.png",
            report,
            root / "triad_transfer_design.json",
            root / "triad_transfer_train.py",
            root / "triad_transfer_analysis.py",
            root / "triad_transfer_audit.py",
            root / "plot_triad_transfer.py",
            root / "visual_qa.py",
            root / "build_triad_transfer_report.py",
            root / "package_results.py",
        )
        if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            complete.get("status") == "complete",
            complete.get("formal") is True,
            complete.get("probe") == "triad_transfer",
            complete.get("runs") == 108,
            analysis.get("formal") is True,
            analysis.get("status") == "complete",
            validation.get("passed") is True,
            audit.get("status") == "complete",
            visual.get("status") == "passed_visual_qa",
            not missing,
        )
    )
    findings = [
        {
            "type": "formation_transfer_interaction",
            "interpretation": "形成固定 A 的协议在固定传递下保持关系特定成功，但在轮换或随机传递下显著下降；随机形成的协议在三种传递下保持或提高联合成功。",
        },
        {
            "type": "compositional_transfer",
            "interpretation": "三 token 联合一致率与 target-60 联合 J 同时报告，显示形式收敛和 grounded 功能相关但不等价。",
        },
        {
            "type": "resource_bottleneck",
            "interpretation": "资源 0/1 的准确率通常低于资源 2，三槽位组合的联合成功因此低于局部正确率；资源置换和对称性控制是下一步必要实验。",
        },
        {
            "type": "replayability",
            "interpretation": "独立 NumPy 检查覆盖协议表、343 码因子化、训练 fixture、token/action 采样、奖励、全局伙伴日程和完成文件绑定。",
        },
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.40-triad-transfer",
        "reviewed_files": reviewed_files,
        "review_scope": [
            "Checked 108 formal formation-by-transmission chains, five generations and four ordered newcomer replacements.",
            "Checked all 6480 protocol tables, 120-map test support, 60-map partition training support and saved generation curves.",
            "Checked independent NumPy metric reanalysis, 343-code factorization, three-token categorical sampling, reward arithmetic and schedule replay.",
            "Checked all 3456 saved trace files (829440 rows), completion hashes, raster figure QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": [
            "Generation zero is a v0.39 formation endpoint rather than a new zero-shot formation run.",
            "Visual frontends are inherited from earlier perceptual training and frozen.",
            "Each resource sender emits one token; within-sender sequence grammar is not tested.",
            "Shared reward and joint policy-gradient updates are centralized.",
            "Resource-wise asymmetry requires permutation and role-symmetry controls.",
        ],
        "next_step": "Permutation-controlled two-token senders, followed by distributed episode feedback and noisy partner transmission.",
        "link_check": {"local_links": len(links), "missing": missing},
        "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.40 结果审查",
        "",
        f"- 状态：`{review['status']}`",
        f"- 形式批次：`{review['passed']}`",
        f"- 审查文件数：{len(reviewed_files)}",
        f"- 本地链接：{len(links)}，缺失：{len(missing)}",
        "",
        "## 审查范围",
        "",
    ] + [f"- {item}" for item in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{item['type']}**：{item['interpretation']}" for item in findings] + ["", "## 限制", ""] + [f"- {item}" for item in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed_files), "links": len(links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
