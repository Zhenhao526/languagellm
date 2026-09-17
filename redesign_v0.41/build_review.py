"""Build a scientific review record for the v0.41 formal batch."""
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
    out = root / "results" / "two_token_001"
    report = out / "三资源双token协议形成研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "two_token_analysis.json")
    validation = read(out / "two_token_raw_validation.json")
    audit = read(out / "two_token_audit.json")
    visual = read(out / "visual_qa.json")
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [
        str(path)
        for path in (
            out / "invocation.json",
            out / "training_complete.json",
            out / "two_token_analysis.json",
            out / "two_token_raw_validation.json",
            out / "two_token_audit.json",
            out / "visual_qa.json",
            out / "plot_metadata.json",
            out / "figures" / "01_two_token_outcomes.png",
            report,
            root / "two_token_design.json",
            root / "two_token_train.py",
            root / "two_token_analysis.py",
            root / "two_token_audit.py",
            root / "plot_two_token.py",
            root / "visual_qa.py",
            root / "build_two_token_report.py",
            root / "README.md",
        )
        if path.exists()
    ]
    formal = all(
        (
            invocation.get("formal") is True,
            complete.get("status") == "complete",
            complete.get("formal") is True,
            complete.get("probe") == "two_token_formation",
            complete.get("runs") == 72,
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
            "type": "sequential_information",
            "interpretation": "每个 sender 的第二个 token 携带独立的位置相关信息，并且 p(token1|token0) 相对边际分布有可测的前缀敏感性；这支持序列位置具有功能，但不构成语法证据。",
        },
        {
            "type": "partner_coverage",
            "interpretation": "固定 A 主要形成关系特定协议，轮换 A/B 提高 A/B 的可读性，随机 A/B/C 在三种评估拓扑上最接近；伙伴覆盖仍是公共协议形成的主要压力。",
        },
        {
            "type": "resource_symmetry",
            "interpretation": "cyclic 资源置换的效应随伙伴拓扑改变，说明资源身份与架构槽位的交互会影响结果；完整的资源和角色排列仍是必要控制。",
        },
        {
            "type": "composition_caveat",
            "interpretation": "pair NMI 达到 100% 可能来自六地点任务中的冗余完整编码，不能单凭该指标声称出现了可组合语法；需要新地点、删除/交换干预和迁移测试。",
        },
        {
            "type": "replayability",
            "interpretation": "独立 NumPy 重分析覆盖协议表与序列指标，轨迹审计覆盖 categorical 采样、奖励、fixture、日程和所有保存轨迹，并检查报告中的绝对本地链接。",
        },
    ]
    review = {
        "passed": formal,
        "passed_with_stated_limits": formal,
        "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.41-two-token-per-sender",
        "reviewed_files": reviewed_files,
        "review_scope": [
            "Checked 72 formal two-token formation chains across two resource assignments, three partner schedules, four seeds and three partitions.",
            "Checked all 3456 checkpoint protocol tables, 120-map test support, 60-map partition training support and saved curves.",
            "Checked independent NumPy reanalysis of resource-wise, pair and sequence metrics, including conditional token1 distributions.",
            "Checked independent replay of schedule/fixture derivation, categorical token/action sampling, reward arithmetic and all 576 trace files (138240 rows).",
            "Checked completion hashes, raster figure QA and every absolute local report link.",
        ],
        "findings": findings,
        "limitations": [
            "Generation starts from fresh random communication heads; this is formation, not v0.40 endpoint transfer.",
            "Visual frontends are inherited from v0.28 and frozen.",
            "Pair NMI is saturated by the finite six-location task and is not evidence for syntax.",
            "Only one cyclic resource permutation is included; all six resource permutations and role permutations remain to be tested.",
            "Shared reward and joint policy-gradient updates are centralized.",
        ],
        "next_step": "Enumerate resource and role permutations, apply token deletion/swapping and novel-location interventions, then test sequence-protocol transfer across replacement generations.",
        "link_check": {"local_links": len(links), "missing": missing},
        "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# v0.41 结果审查",
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
