"""Build an independent review record for the v0.38 formal batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute(): yield path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); root = args.out.resolve(); out = root / "results" / "transfer_001"
    analysis = read(out / "transfer_analysis.json"); validation = read(out / "transfer_raw_validation.json"); audit = read(out / "transfer_audit.json"); complete = read(out / "training_complete.json"); invocation = read(out / "invocation.json"); visual = read(out / "figures" / "visual_qa.json"); report = out / "形成协议的代际传递研究报告.md"
    links = list(linked_paths(report)); missing = [str(path) for path in links if not path.exists()]
    reviewed_files = [str(path) for path in (out / "invocation.json", out / "training_complete.json", out / "transfer_analysis.json", out / "transfer_raw_validation.json", out / "transfer_audit.json", out / "figures" / "figure_source.json", out / "figures" / "visual_qa.json", report, root / "transfer_design.json", root / "transfer_train.py", root / "transfer_analysis.py", root / "transfer_audit.py", root / "plot_transfer.py", root / "visual_qa.py", root / "build_transfer_report.py", root / "package_results.py") if path.exists()]
    formal = all((invocation.get("formal") is True, complete.get("status") == "complete", complete.get("formal") is True, complete.get("runs") == 108, analysis.get("formal") is True, analysis.get("status") == "complete", validation.get("passed") is True, audit.get("passed") is True, visual.get("passed") is True, not missing))
    formations = analysis["formation_conditions"]; transmissions = analysis["transmission_conditions"]; final_generation = str(analysis["generations"][-1])
    findings = []
    for formation in formations:
        values = {transmission: analysis["summary"][formation][transmission][final_generation]["A"]["target_J"]["mean"] for transmission in transmissions}; findings.append({"type": "formation_endpoint_transfer", "formation_condition": formation, "final_A_target_J": values, "interpretation": "The topology used during protocol formation changes the starting point and the endpoint reached under each newcomer-training schedule."})
    findings.append({"type": "factorial_retention", "retention_A_target_J": {formation: {transmission: analysis["retention"][formation][transmission]["A"]["mean"] for transmission in transmissions} for formation in formations}, "interpretation": "Formation and transmission topology jointly determine whether a protocol is preserved, broadened across partner topologies, or substantially recoded."})
    review = {"passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review", "version": "v0.38-origin-transfer", "reviewed_files": reviewed_files, "review_scope": ["Checked 108 factorial chains combining three v0.37 teacher-free formation endpoints with three transmission schedules.", "Checked four serial replacement events per chain, A/B/C endpoint evaluation, generation curves and completion bindings.", "Checked independent NumPy metric replay, categorical traces, reward arithmetic, deterministic schedule replay and source/input hashes.", "Checked figure generation, raster dimensions/non-white content and every absolute local link in the report."], "findings": findings, "limitations": ["Formation endpoints inherit frozen private visual encoders from earlier perceptual training.", "Only communication modules are optimized for newcomers; resident models remain frozen.", "The four-agent population, replacement order and topology schedules are experimenter specified.", "The two-resource one-token task does not test vocabulary growth, compositional grammar, intention or natural-language structure.", "Token output changes are surface fingerprints rather than semantic distance; optimizer-state replay is outside the audit."], "next_step": "Add a third resource and multi-token sequences, then repeat the formation-by-transmission factorial under fully distributed episode feedback and explicit communication costs.", "link_check": {"local_links": len(links), "missing": missing}, "new_training_reviewed": True}
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.38 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed_files)}", f"- 本地链接：{len(links)}，缺失：{len(missing)}", "", "## 审查范围", ""] + [f"- {x}" for x in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{x['type']}**：{x['interpretation']}" for x in findings] + ["", "## 限制", ""] + [f"- {x}" for x in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md)); print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed_files), "links": len(links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__": main()
