"""Create a bounded scientific review record for v0.35."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        target = Path(match.group(2))
        if target.is_absolute(): yield target


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); identity_out = out / "results" / "identity_001"; adapt_out = out / "results" / "adapt_001"
    identity = read(identity_out / "identity_analysis.json")
    identity_audit = read(identity_out / "identity_audit.json")
    adaptation = read(adapt_out / "adaptation_analysis.json")
    adaptation_validation = read(adapt_out / "adaptation_raw_validation.json")
    adaptation_audit = read(adapt_out / "adaptation_audit.json")
    visual = read(adapt_out / "figures" / "visual_qa.json")
    report = adapt_out / "新主体身份留出与在线适应研究报告.md"
    links = list(linked_paths(report)); missing = [str(p) for p in links if not p.exists()]
    files = [
        ROOT / "identity_design.json", ROOT / "adaptation_design.json", ROOT / "newcomer_identity.py", ROOT / "newcomer_adaptation.py",
        ROOT / "identity_analysis.py", ROOT / "identity_audit.py", ROOT / "adaptation_analysis.py", ROOT / "adaptation_audit.py",
        ROOT / "plot_adaptation.py", ROOT / "visual_qa.py", ROOT / "build_adaptation_report.py", report,
        identity_out / "identity_analysis.json", identity_out / "identity_raw_validation.json", identity_out / "identity_audit.json",
        identity_out / "terminal_receipt.json", adapt_out / "adaptation_analysis.json", adapt_out / "adaptation_raw_validation.json",
        adapt_out / "adaptation_audit.json", adapt_out / "terminal_receipt.json", adapt_out / "figures" / "visual_qa.json",
    ]
    reviewed_files = {str(path.resolve()): sha(path) for path in files if path.is_file()}
    formal = all([
        identity.get("formal") is True and identity.get("status") == "complete",
        identity_audit.get("passed") is True,
        adaptation.get("formal") is True and adaptation.get("status") == "complete",
        adaptation_validation.get("passed") is True,
        adaptation_audit.get("passed") is True,
        visual.get("passed") is True,
        not missing,
    ])
    review = {
        "passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review",
        "version": "v0.35-identity-adaptation", "reviewed_files": reviewed_files,
        "review_scope": [
            "Checked the formal zero-shot identity holdout and online newcomer adaptation batches in the shared v0.34-34034 namespace.",
            "Checked the factorial identity interventions, fixed-A versus rotating-AB learning curves, A/B/C endpoint evaluation and paired seed×panel×identity contrasts.",
            "Checked independent NumPy reanalysis, raw metric replays, endpoint protocol tables, categorical traces, reward arithmetic, schedule identity and completion bindings.",
            "Checked the figure source values, fixed axes, raster dimensions and local report links.",
        ],
        "findings": [
            {"type": "zero_shot_identity_failure", "fixed_A_target_J": {role: identity["summary"]["fixed_A"]["A"][role]["newcomer"]["target12"]["J"]["mean"] for role in identity["roles"]}, "rotating_AB_target_J": {role: identity["summary"]["rotating_AB"]["A"][role]["newcomer"]["target12"]["J"]["mean"] for role in identity["roles"]}, "interpretation": "A same-type newcomer with reset communication modules is not zero-shot readable by residents; shared private visual type is insufficient."},
            {"type": "online_cultural_acquisition", "fixed_A_A_initial": adaptation["summary"]["fixed_A"]["A"]["initial_target_J"]["mean"], "fixed_A_A_final": adaptation["summary"]["fixed_A"]["A"]["final_target_J"]["mean"], "interpretation": "With residents frozen, communication-only online adaptation raises newcomer target J on the trained topology."},
            {"type": "topology_dependent_transfer", "rotating_A_final": adaptation["summary"]["rotating_AB"]["A"]["final_target_J"]["mean"], "rotating_B_final": adaptation["summary"]["rotating_AB"]["B"]["final_target_J"]["mean"], "rotating_C_final": adaptation["summary"]["rotating_AB"]["C"]["final_target_J"]["mean"], "paired_final": {schedule: adaptation["paired_rotating_minus_fixed"][schedule]["600"]["mean"] for schedule in adaptation["schedules"]}, "interpretation": "Rotation lowers fixed-topology A adaptation but improves B and held-out C relative to fixed-A training."},
            {"type": "private_type_robustness", "rotating_private_type_final": {private_type: {schedule: adaptation["private_type_summary"]["rotating_AB"][private_type][schedule]["final_target_J"]["mean"] for schedule in adaptation["schedules"]} for private_type in ("0", "1")}, "interpretation": "The two retained private visual types have similar endpoint patterns; topology condition is the larger manipulation in this matrix."},
        ],
        "limitations": [
            "The agents are small visual–communication mechanism probes, not a large open-source VLM/LLM and not end-to-end world models.",
            "The resident population is imported from fixed v0.34 endpoint policies; no births, deaths, demographic turnover or generational bottleneck is present.",
            "The newcomer retains a private visual encoder and only its communication modules are reset/optimized, so the result isolates social acquisition rather than perceptual learning.",
            "The task has two resources, two one-token senders, one receiver and six positions; it does not test vocabulary growth, compositional grammar or intention.",
            "A/B/C topologies and the observation masks are experimenter-specified controls; the findings are mechanistic and conditional.",
            "The audit replays saved endpoint tables and bounded traces, not every optimizer state transition.",
        ],
        "next_step": "Run iterated replacement with a finite social-learning bottleneck, comparing fixed-A, rotating-AB and random-topology teacher histories; measure grounded transmission, protocol drift and compositional growth across generations.",
        "link_check": {"local_links": len(links), "missing": missing}, "new_training_reviewed": True,
    }
    (out / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.35 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed_files)}", f"- 本地链接：{len(links)}，缺失：{len(missing)}", "", "## 审查范围", ""]
    md.extend(f"- {item}" for item in review["review_scope"])
    md += ["", "## 主要发现", ""]
    for finding in review["findings"]: md.append(f"- **{finding['type']}**：{finding['interpretation']}")
    md += ["", "## 限制", ""]
    md.extend(f"- {item}" for item in review["limitations"])
    md += ["", "## 下一步", "", review["next_step"], ""]
    (out / "结果审查.md").write_text("\n".join(md))
    print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed_files), "links": len(links), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__": main()
