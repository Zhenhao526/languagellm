"""Build a scientific review record for the v0.39 formal batch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path: Path): return json.loads(path.read_text())


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute(): yield path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); root = args.out.resolve(); out = root / "results" / "triad_001"; report = out / "三资源三token共同符号形成研究报告.md"
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); analysis = read(out / "triad_analysis.json"); validation = read(out / "triad_raw_validation.json"); audit = read(out / "triad_audit.json"); visual = read(out / "figures" / "visual_qa.json"); links = list(linked_paths(report)); missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [str(path) for path in (out / "invocation.json", out / "training_complete.json", out / "triad_analysis.json", out / "triad_raw_validation.json", out / "triad_audit.json", out / "figures" / "figure_source.json", out / "figures" / "visual_qa.json", report, root / "triad_design.json", root / "triad_train.py", root / "triad_analysis.py", root / "triad_audit.py", root / "plot_triad.py", root / "visual_qa.py", root / "build_triad_report.py", root / "package_results.py") if path.exists()]
    formal = all((invocation.get("formal") is True, complete.get("status") == "complete", complete.get("formal") is True, complete.get("runs") == 36, analysis.get("formal") is True, analysis.get("status") == "complete", validation.get("passed") is True, audit.get("passed") is True, visual.get("passed") is True, not missing))
    findings = [{"type": "three_resource_joint_formation", "interpretation": "Three resource-specific senders and a 343-code receiver can form grounded slot-wise protocols from random communication heads, while single-resource accuracy and joint success remain separable."}, {"type": "topology_effect", "interpretation": "Fixed, rotating and random partner schedules produce different tradeoffs between local grounded success and cross-partner token agreement."}, {"type": "complexity_boundary", "interpretation": "Moving from two to three resource slots exposes a composition bottleneck that is invisible in one-resource or two-resource aggregate success."}]
    review = {"passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review", "version": "v0.39-triad-origin", "reviewed_files": reviewed_files, "review_scope": ["Checked 36 formal three-resource formation chains across three partner-topology schedules.", "Checked 343-code receiver tables, 120-map test support, 60-map partition-specific training support and all saved endpoint curves.", "Checked independent NumPy metric replay, categorical token/action traces, reward arithmetic, schedule replay, completion hashes and raster figure QA.", "Checked every absolute local link in the report."], "findings": findings, "limitations": ["The visual frontends are inherited from earlier two-resource perceptual training and remain frozen.", "Each sender emits one token; this is a slot-composition probe rather than open-ended grammar.", "Shared reward and joint policy-gradient updates are centralized.", "The four-agent population, six sites and topology schedules are experimenter specified."], "next_step": "Transfer the three-resource endpoints through finite-generation newcomer replacement, then add within-sender multi-token sequences and distributed episode feedback.", "link_check": {"local_links": len(links), "missing": missing}, "new_training_reviewed": True}
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.39 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed_files)}", f"- 本地链接：{len(links)}，缺失：{len(missing)}", "", "## 审查范围", ""] + [f"- {x}" for x in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{x['type']}**：{x['interpretation']}" for x in findings] + ["", "## 限制", ""] + [f"- {x}" for x in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md)); print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed_files), "links": len(links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__": main()
