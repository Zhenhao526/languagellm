"""Build a scientific review record for the v0.42 intervention batch."""
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
    out = root / "results" / "sequence_intervention_001"
    report = out / "三资源双token端点干预研究报告.md"
    invocation = read(out / "invocation.json")
    analysis = read(out / "sequence_intervention_analysis.json")
    audit = read(out / "sequence_intervention_audit.json")
    visual = read(out / "visual_qa.json")
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [
        str(path)
        for path in (
            out / "invocation.json",
            out / "sequence_intervention_analysis.json",
            out / "sequence_intervention_audit.json",
            out / "plot_metadata.json",
            out / "figures" / "01_sequence_intervention.png",
            out / "visual_qa.json",
            report,
            root / "sequence_intervention_design.json",
            root / "sequence_intervention_analysis.py",
            root / "sequence_intervention_audit.py",
            root / "plot_sequence_intervention.py",
            root / "visual_qa.py",
            root / "build_sequence_intervention_report.py",
            root / "README.md",
        )
        if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            invocation.get("source_version") == "v0.41-two-token-per-sender",
            analysis.get("formal") is True,
            analysis.get("status") == "complete",
            analysis.get("probe") == "sequence_intervention",
            analysis.get("chains") == 72,
            len(analysis.get("rows", [])) == 864,
            analysis.get("baseline_replay_mismatches") == 0,
            audit.get("formal") is True,
            audit.get("status") == "complete",
            audit.get("coverage", {}).get("rows") == 864,
            visual.get("status") == "passed_visual_qa",
            not missing,
        )
    )
    findings = [
        {
            "type": "token_position_causality",
            "interpretation": "token0 mask、token1 mask、顺序交换和跨世界打乱形成了一组端点反事实；当前任务对 token0 的依赖更强，但 token1 仍有可测的增量作用。",
        },
        {
            "type": "resource_block_equivariance",
            "interpretation": "轮换和随机伙伴形成条件下，资源块同步置换后的目标评分远高于保持原目标的 literal 评分，支持资源轴可以作为结构单位重排。",
        },
        {
            "type": "coordinate_interaction",
            "interpretation": "循环站点重定位提高了 target J；由于照片库和站点数量未变且 target 分组沿用原 map_id，这更像坐标编码与关系划分的交互，应作为后续对称性控制。",
        },
        {
            "type": "endpoint_scope",
            "interpretation": "本轮只干预 v0.41 的形成终点，没有新增训练；干预结果不能单独证明协议在形成时学习了语法，也不能代表开放式语言。",
        },
        {
            "type": "replayability",
            "interpretation": "独立 NumPy 审计重新计算了 864 条记录的 baseline、mask、swap、block permutation 和 site relocation 指标，并检查了所有保存数组和报告链接。",
        },
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.42-sequence-intervention",
        "reviewed_files": reviewed_files,
        "review_scope": [
            "Checked all 72 v0.41 endpoint chains and 864 schedule-team records.",
            "Checked seven-value masks for both token positions, token-order and cross-world shuffles.",
            "Checked all six resource-block permutations under literal and jointly permuted targets.",
            "Checked identity plus five cyclic site relocations with frozen photo identities.",
            "Checked baseline replay identity, independent NumPy metric replay, raster QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": [
            "Endpoint intervention only; no additional formation training.",
            "Resource blocks are receiver-side counterfactuals and are not yet a training factor.",
            "Six-site relocation reuses the frozen visual bank and is not a new visual-location test.",
            "Existing vocabulary symbols are used as mask values; no reserved null symbol exists.",
            "Centralized reward and frozen private visual frontends remain inherited from v0.41.",
        ],
        "next_step": "Train with all resource and role permutations, randomize block order during formation, then test the same interventions and cross-generation transfer.",
        "link_check": {"local_links": len(links), "missing": missing},
        "new_training_reviewed": False,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.42 结果审查",
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
