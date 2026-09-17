"""Seal the formal v0.44 endpoint partner-transfer batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_hash(path, digest):
    if not path.is_file():
        raise AssertionError(f"missing input: {path}")
    actual = sha(path)
    if actual != digest:
        raise AssertionError(f"hash mismatch: {path}: {actual} != {digest}")


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
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation = read(out / "invocation.json")
    complete = read(out / "transfer_complete.json")
    analysis = read(out / "partner_transfer_analysis.json")
    audit = read(out / "partner_transfer_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "双token伙伴替换端点迁移研究报告.md"
    if not (invocation.get("formal") is True and invocation.get("version") == "v0.44-partner-transfer-endpoint" and complete.get("status") == "complete" and complete.get("formal") is True and complete.get("groups") == 72 and analysis.get("status") == "complete" and analysis.get("formal") is True and analysis.get("groups") == 72 and analysis.get("native_replay_mismatches") == 0 and audit.get("status") == "complete" and audit.get("formal") is True and audit.get("groups") == 72 and visual.get("status") == "passed_visual_qa" and review.get("passed") is True and review.get("version") == "v0.44-partner-transfer-endpoint"):
        raise AssertionError("formal partner-transfer analysis/audit/review required")
    for kind in ("source_hashes", "input_hashes"):
        for path, digest in invocation[kind].items():
            assert_hash(Path(path), digest)
    training_files = complete.get("files", {})
    for relative, digest in training_files.items():
        assert_hash(out / relative, digest)
    report_links = list(links(report))
    missing = [str(path) for path in report_links if not path.exists()]
    if missing:
        raise AssertionError(f"missing report links: {missing}")
    files = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name == "completion_manifest.json" or "failure" in path.name:
            continue
        if "results" in path.parts and any("smoke" in part or "probe" in part for part in path.parts):
            continue
        files.add(path.resolve())
    files.update(Path(path).resolve() for path in invocation["source_hashes"])
    files.update(Path(path).resolve() for path in invocation["input_hashes"])
    files.update(path.resolve() for path in report_links)
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files) if path.is_file()]
    record = {
        "status": "complete_development_batch", "version": "v0.44-partner-transfer-endpoint", "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "zero-adaptation endpoint compatibility after independent culture partner replacement", "overall_research_goal_complete": False, "formal_output": str(out), "terminal": None,
        "counts": {"visual_groups": complete["groups"], "cultures_per_group": complete["cultures"], "replacement_modes": len(complete["replacement_modes"]), "schedules": len(complete["schedules"]), "role_permutations": complete["role_permutations"], "team_slots": complete["team_slots"], "score_cells": audit["coverage"]["score_cells"]},
        "quality": {"analysis_checks": analysis["checks"], "analysis_scalar_comparisons": analysis["scalar_comparisons"], "analysis_maximum_metric_absolute_difference": analysis["maximum_metric_absolute_difference"], "audit_checks": audit["checks"], "audit_scalar_comparisons": audit["scalar_comparisons"], "audit_maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"], "visual_qa": visual["status"], "scientific_review": review["status"]},
        "report": str(report), "review": str(root / "结果审查.json"), "artifacts": artifacts,
        "limitations": ["No post-replacement adaptation training; this is a direct compatibility baseline.", "Donor and recipient share visual frontends within each group, so new perception is not tested.", "Finite two-token, six-site grounded protocol; no open-ended vocabulary or syntax.", "Centralized reward and frozen visual frontends are inherited from v0.43."],
        "next_step": "Train controlled newcomer adaptation after one-sender replacement, comparing recovery speed under fixed, rotating and random partner schedules.", "excluded": ["results/*smoke*", "results/*probe*", "__pycache__", ".venv", "failure records", "this completion manifest"], "link_check": {"report_links": len(report_links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "training_files_rehashed": len(training_files), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
