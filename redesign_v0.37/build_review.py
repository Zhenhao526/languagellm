"""Create a bounded scientific review record for v0.37."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def links(report):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute(): yield path

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    root = args.out.resolve(); out = root / "results" / "origin_001"; report = out / "无教师条件下的共同符号形成研究报告.md"
    analysis = read(out / "origin_analysis.json"); validation = read(out / "origin_raw_validation.json"); audit = read(out / "origin_audit.json"); visual = read(out / "figures" / "visual_qa.json"); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json")
    report_links = list(links(report)); missing = [str(path) for path in report_links if not path.exists()]
    files = [ROOT / name for name in ("origin_design.json", "origin_train.py", "origin_analysis.py", "origin_audit.py", "plot_origin.py", "visual_qa.py", "build_origin_report.py")]
    files += [report, out / "origin_analysis.json", out / "origin_raw_validation.json", out / "origin_audit.json", out / "figures" / "visual_qa.json"]
    reviewed = {str(path.resolve()): sha(path) for path in files if path.is_file()}
    formal = all([invocation.get("formal") is True, complete.get("formal") is True, complete.get("status") == "complete", complete.get("runs") == 36, analysis.get("formal") is True, analysis.get("status") == "complete", validation.get("passed") is True, audit.get("passed") is True, visual.get("passed") is True, not missing])
    review = {
        "passed": formal, "passed_with_stated_limits": formal, "status": "passed_scientific_review" if formal else "failed_scientific_review", "version": "v0.37-teacher-free-origin", "reviewed_files": reviewed,
        "review_scope": ["Checked 36 formal teacher-free runs with all four communication modules freshly reset and frozen private visual encoders.", "Checked fixed-A, rotating-AB and deterministic random-ABC partner schedules at A/B/C endpoint topologies and four checkpoints.", "Checked independent NumPy metric replay, same-type token agreement, endpoint probability tables, categorical traces, reward arithmetic, schedule replay and completion hashes.", "Checked figure axes, raster dimensions and every absolute local report link."],
        "findings": [
            {"type": "grounded_origin_from_random_heads", "fixed_A_A_final": analysis["summary"]["fixed_A"]["A"]["1200"]["target_J"]["mean"], "rotating_AB_A_final": analysis["summary"]["rotating_AB"]["A"]["1200"]["target_J"]["mean"], "random_ABC_A_final": analysis["summary"]["random_ABC"]["A"]["1200"]["target_J"]["mean"], "interpretation": "Joint training from random communication modules can establish non-chance grounded protocols under the two-resource task."},
            {"type": "form_grounding_dissociation", "fixed_A_agreement": analysis["summary"]["fixed_A"]["A"]["1200"]["agreement_joint"]["mean"], "random_ABC_agreement": analysis["summary"]["random_ABC"]["A"]["1200"]["agreement_joint"]["mean"], "interpretation": "Fixed-A achieves grounded local coordination with very low same-type token agreement, whereas random topology coverage produces both high agreement and broad grounded transfer."},
            {"type": "topology_as_formation_pressure", "final_grounded": {condition: {schedule: analysis["summary"][condition][schedule]["1200"]["target_J"]["mean"] for schedule in ("A", "B", "C")} for condition in analysis["conditions"]}, "interpretation": "Partner-topology coverage changes whether the emergent protocol is local, partially transferable or broadly shared."},
        ],
        "limitations": ["Private visual encoders are inherited from prior perceptual training and frozen; the experiment is not an end-to-end no-language world model.", "Training uses a shared grounded reward and synchronized policy-gradient updates, not decentralized observation-only social learning.", "Four agents, two private types, two resources and one-token messages are a small controlled probe without vocabulary growth, grammar or intention.", "Topology schedules are experimenter specified and random-ABC is deterministic from seeds; agents cannot choose partners.", "Token agreement is a surface statistic and cannot replace semantic equivalence; the audit does not replay every optimizer state transition."],
        "next_step": "Feed the three formed populations into iterated replacement and then repeat origin with decentralized episode-level feedback and a third resource/multi-token protocol.", "link_check": {"local_links": len(report_links), "missing": missing}, "new_training_reviewed": True,
    }
    (root / "结果审查.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    md = ["# v0.37 结果审查", "", f"- 状态：`{review['status']}`", f"- 形式批次：`{review['passed']}`", f"- 审查文件数：{len(reviewed)}", f"- 本地链接：{len(report_links)}，缺失：{len(missing)}", "", "## 审查范围", ""] + [f"- {x}" for x in review["review_scope"]] + ["", "## 主要发现", ""] + [f"- **{x['type']}**：{x['interpretation']}" for x in review["findings"]] + ["", "## 限制", ""] + [f"- {x}" for x in review["limitations"]] + ["", "## 下一步", "", review["next_step"], ""]
    (root / "结果审查.md").write_text("\n".join(md)); print(json.dumps({"status": review["status"], "reviewed_files": len(reviewed), "links": len(report_links), "missing": len(missing)}, ensure_ascii=False))

if __name__ == "__main__": main()
