"""Seal the completed v0.34 rotating batch and its same-namespace control."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_hash(path: Path, expected: str):
    actual = sha(path)
    if actual != expected:
        raise AssertionError(f"hash mismatch: {path}: {actual} != {expected}")


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        target = Path(match.group(2))
        if target.is_absolute():
            yield target


def check_training(out: Path):
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    if not (invocation.get("formal") is True and complete.get("status") == "complete" and complete.get("formal") is True):
        raise AssertionError("formal training completion required")
    for key in ("source_hashes", "input_hashes"):
        if invocation[key] != complete[key]:
            raise AssertionError(f"{key} changed")
        for path, digest in invocation[key].items():
            assert_hash(Path(path), digest)
    for rel, digest in complete["files"].items():
        assert_hash(out / rel, digest)
    return invocation, complete


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve(); fixed_out = out.parent / "fixed_001"; manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)

    invocation, complete = check_training(out)
    fixed_invocation, fixed_complete = check_training(fixed_out)
    if fixed_invocation["source_hashes"] != invocation["source_hashes"] or fixed_invocation["input_hashes"] != invocation["input_hashes"]:
        raise AssertionError("fixed-A control is not in the same source/input namespace")
    if fixed_complete.get("control") != "fixed_A" or fixed_complete.get("social_runs") != 12:
        raise AssertionError("fixed-A control completion required")

    validation = read(out / "raw_validation.json")
    audit = read(out / "audit_execution.json")
    analysis = read(out / "analysis.json")
    transfer = read(out / "rotation_transfer.json")
    topology = read(out / "cross_topology.json")
    visual = read(out / "figures/visual_qa.json")
    fixed_analysis = read(fixed_out / "fixed_control_analysis.json")
    fixed_validation = read(fixed_out / "fixed_control_raw_validation.json")
    fixed_audit = read(fixed_out / "fixed_control_audit.json")
    fixed_cross = read(fixed_out / "fixed_control_cross.json")
    review = read(ROOT / "结果审查.json")
    report = out / "伙伴拓扑轮换与跨伙伴迁移研究报告.md"

    if not (analysis.get("formal") and analysis.get("status") == "complete" and validation.get("passed") and validation.get("analysis_sha256") == sha(out / "analysis.json")):
        raise AssertionError("formal rotating analysis/validation required")
    if not (audit.get("passed") and transfer.get("formal") and transfer.get("status") == "complete" and topology.get("formal") and topology.get("status") == "complete"):
        raise AssertionError("rotating transfer analyses required")
    if not (fixed_analysis.get("formal") and fixed_analysis.get("status") == "complete" and fixed_analysis.get("control") == "fixed_A" and fixed_validation.get("passed") and fixed_validation.get("analysis_sha256") == sha(fixed_out / "fixed_control_analysis.json") and fixed_audit.get("passed") and fixed_cross.get("formal") and fixed_cross.get("status") == "complete"):
        raise AssertionError("formal fixed-A analysis/validation/audit required")
    wrapper = read(fixed_out / "wrapper_receipt.json")
    receipt = read(fixed_out / "terminal_receipt.json")
    if not (wrapper.get("control") == "fixed_A" and wrapper.get("source_sha256") == sha(ROOT / "run_fixed_control.py") and receipt.get("exit_code") == 0):
        raise AssertionError("fixed-A wrapper and receipt required")
    if not visual.get("passed"):
        raise AssertionError("visual QA failed")
    if not (review.get("passed") and review.get("passed_with_stated_limits") and review.get("version") == "v0.34-paired"):
        raise AssertionError("scientific review failed")
    if not report.is_file():
        raise FileNotFoundError(report)
    for target in linked_paths(report):
        if not target.exists():
            raise FileNotFoundError(target)

    files: set[Path] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts or path.name == "completion_manifest.json":
            continue
        if "results" in path.parts:
            if not (path.is_relative_to(out) or path.is_relative_to(fixed_out)):
                continue
            if any(part.startswith("smoke_") for part in path.parts):
                continue
        files.add(path.resolve())
    files.update(Path(path) for path in invocation["source_hashes"])
    files.update(Path(path) for path in invocation["input_hashes"])
    for target in linked_paths(report):
        files.add(target.resolve())
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files)]
    record = {
        "status": "complete_development_batch",
        "version": "v0.34-paired",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "partner-topology rotation and same-namespace fixed-A control for complementary two-sender grounded communication",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "paired_control_output": str(fixed_out),
        "terminal": read(out / "terminal_receipt.json"),
        "fixed_control_terminal": receipt,
        "counts": {
            "rotating": {key: complete[key] for key in ("social_runs", "pair_updates", "messages", "actions", "new_private_fits", "new_dino_inferences")},
            "fixed_A": {key: fixed_complete[key] for key in ("social_runs", "pair_updates", "messages", "actions", "new_private_fits", "new_dino_inferences")},
        },
        "quality": {
            "rotating_independent_checks": validation["checks"],
            "rotating_scalar_comparisons": validation["scalar_comparisons"],
            "rotating_maximum_metric_absolute_difference": validation["maximum_metric_absolute_difference"],
            "rotating_audit_checks": audit["checks"],
            "fixed_A_independent_checks": fixed_validation["checks"],
            "fixed_A_scalar_comparisons": fixed_validation["scalar_comparisons"],
            "fixed_A_maximum_metric_absolute_difference": fixed_validation["maximum_metric_absolute_difference"],
            "fixed_A_audit_checks": fixed_audit["checks"],
            "visual_qa": visual["passed"],
            "scientific_review": review["status"],
            "rotation_transfer": transfer["status"],
            "cross_topology": topology["status"],
            "fixed_A_cross_topology": fixed_cross["status"],
        },
        "report": str(report),
        "review": str(ROOT / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Four population members reuse two frozen private encoders; topology rotation is a fixed four-agent schedule rather than demographic turnover.",
            "A/B/C schedules and food-only/water-only masks are experimenter-specified symbolic resource-location controls.",
            "The task has two senders, one receiver and two resources; it does not test vocabulary growth, grammar, intention or cultural transmission.",
            "Cross-team token substitution is an offline grounded-readout diagnostic, not an online communication episode.",
            "The fixed-A control has 12 social runs and the rotating batch has 36; paired inference uses the common four seeds and averages three panels within each seed.",
            "The bounded audits replay categorical traces, endpoint tables and agreement files, but not every gradient or Adam transition.",
        ],
        "next_step": "Hold topology coverage constant while holding out sender or receiver identities, then compare identity generalization with held-out C before adding a third resource or delayed relay.",
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__": main()
