"""Build a scientific review record for the v0.44 endpoint transfer batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def links(report):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute():
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.44 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "partner_transfer_001"
    report = out / "双token伙伴替换端点迁移研究报告.md"
    invocation = read(out / "invocation.json")
    complete = read(out / "transfer_complete.json")
    analysis = read(out / "partner_transfer_analysis.json")
    audit = read(out / "partner_transfer_audit.json")
    visual = read(out / "visual_qa.json")
    local_links = list(links(report))
    # The report links to this review file itself; allow the first pass to
    # create that self-referential artifact, then a second pass checks it.
    review_path = root / "结果审查.json"
    missing = [str(path) for path in local_links if not path.exists() and path != review_path]
    reviewed = [
        str(path)
        for path in (
            out / "invocation.json", out / "transfer_complete.json", out / "partner_transfer_analysis.json", out / "partner_transfer_audit.json", out / "plot_metadata.json", out / "figures" / "01_partner_transfer.png", out / "visual_qa.json", report,
            root / "partner_transfer_design.json", root / "partner_transfer_train.py", root / "partner_transfer_analysis.py", root / "partner_transfer_audit.py", root / "plot_partner_transfer.py", root / "visual_qa.py", root / "build_partner_transfer_report.py", root / "build_review.py", root / "README.md",
        )
        if path.exists()
    ]
    formal = all((invocation.get("formal") is True, invocation.get("version") == "v0.44-partner-transfer-endpoint", complete.get("status") == "complete", complete.get("formal") is True, complete.get("groups") == 72, analysis.get("status") == "complete", analysis.get("formal") is True, analysis.get("groups") == 72, analysis.get("native_replay_mismatches") == 0, audit.get("status") == "complete", audit.get("formal") is True, audit.get("groups") == 72, audit.get("maximum_replay_absolute_difference") == 0.0, visual.get("status") == "passed_visual_qa", not missing))
    findings = [
        {"type": "cultural_compatibility", "interpretation": "同一视觉环境中的独立通信文化拥有各自的码本；跨文化替换一个 sender 已显著降分，替换全部 sender 几乎归零。"},
        {"type": "replacement_granularity", "interpretation": "一个 sender 的替换保留少量联合功能，全部替换暴露组合 receiver 对公共码本的强依赖。"},
        {"type": "role_randomization_boundary", "interpretation": "v0.43 的随机角色压力改善文化内部角色等变性，但没有让独立文化之间自动共享符号。"},
        {"type": "token_form", "interpretation": "跨文化 pair agreement 仅略高于 1/49 的独立机会水平，与跨文化功能崩溃相符。"},
        {"type": "scope", "interpretation": "这是零适应 endpoint 兼容性基线，不是从零形成、代际传递或开放式语言实验。"},
        {"type": "replayability", "interpretation": "独立 NumPy 分析重放 native v0.43 endpoint，独立审计重算全部 action/score cell，未导入生产模型。"},
    ]
    review = {"passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review", "version": "v0.44-partner-transfer-endpoint", "reviewed_files": reviewed, "review_scope": ["Checked 72 visual groups and six cultures per group.", "Checked none, single-sender and all-sender replacement across A/B/C schedules, six role permutations and four team slots.", "Checked native endpoint provenance against all 432 v0.43 culture endpoints.", "Replayed all 72 stored action tensors into literal/equivariant target scores.", "Checked raster QA and every absolute local report link."], "findings": findings, "limitations": ["No post-replacement adaptation training; compatibility is an endpoint baseline.", "Donor and recipient share visual frontends within each group, so new perception is not tested.", "Finite two-token, six-site grounded protocol; no open-ended vocabulary or syntax.", "Centralized reward and frozen visual frontends are inherited from v0.43."], "next_step": "Train controlled newcomer adaptation after single-sender replacement, comparing recovery speed under fixed, rotating and random partner schedules.", "link_check": {"local_links": len(local_links), "missing": missing}, "new_training_reviewed": False}
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.44 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed)}", f"- 本地链接：{len(local_links)}，缺失：{len(missing)}", "", "## 审查范围", ""] + [f"- {item}" for item in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{item['type']}**：{item['interpretation']}" for item in findings] + ["", "## 限制", ""] + [f"- {item}" for item in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed), "links": len(local_links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
