"""Seal the formal v0.45 newcomer social-learning batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_hash(path: Path, digest: str):
    if not path.is_file():
        raise AssertionError(f"missing input: {path}")
    actual = sha(path)
    if actual != digest:
        raise AssertionError(f"hash mismatch: {path}: {actual} != {digest}")


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute():
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.45 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "newcomer_adaptation_001"
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "newcomer_adaptation_analysis.json")
    audit = read(out / "newcomer_adaptation_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "陌生主体社会学习与协议恢复研究报告.md"
    required = (
        invocation.get("formal") is True,
        invocation.get("version") == "v0.45-newcomer-adaptation",
        complete.get("status") == "complete",
        complete.get("formal") is True,
        complete.get("runs") == 432,
        complete.get("trace_files_expected") == 3456,
        complete.get("protocol_files_expected") == 15552,
        analysis.get("status") == "complete",
        analysis.get("formal") is True,
        analysis.get("runs") == 432,
        len(analysis.get("rows", [])) == 93312,
        len(analysis.get("sequence_rows", [])) == 1296,
        audit.get("status") == "complete",
        audit.get("formal") is True,
        audit.get("runs") == 432,
        audit.get("coverage", {}).get("traces") == 3456,
        audit.get("coverage", {}).get("protocol_files") == 15552,
        visual.get("status") == "passed_visual_qa",
        review.get("passed") is True,
        review.get("version") == "v0.45-newcomer-adaptation",
        report.is_file(),
    )
    if not all(required):
        raise AssertionError("formal newcomer analysis/audit/review required")
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
        "status": "complete_development_batch",
        "version": "v0.45-newcomer-adaptation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "social recovery of a freshly reset communication newcomer inside frozen v0.43 resident cultures",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": None,
        "counts": {
            "chains": complete["runs"],
            "resident_cultures": len(invocation["cultures"]),
            "updates_per_chain": complete["updates_per_run"],
            "checkpoints": len(complete["checkpoints"]),
            "trace_files": complete["trace_files_expected"],
            "protocol_files": complete["protocol_files_expected"],
            "analysis_rows": len(analysis["rows"]),
            "sequence_rows": len(analysis["sequence_rows"]),
        },
        "quality": {
            "analysis_checks": analysis["checks"],
            "analysis_scalar_comparisons": analysis["scalar_comparisons"],
            "analysis_maximum_metric_absolute_difference": analysis["maximum_metric_absolute_difference"],
            "audit_checks": audit["checks"],
            "audit_scalar_comparisons": audit["scalar_comparisons"],
            "audit_maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"],
            "audit_production_modules_imported": audit["production_modules_imported"],
            "visual_qa": visual["status"],
            "scientific_review": review["status"],
        },
        "report": str(report),
        "review": str(root / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Adaptation schedule is fixed to random_ABC in this first batch; adaptation topology is not causally crossed.",
            "Only identity 0 is replaced and its visual frontend is inherited/frozen; new perception is not tested.",
            "Finite two-token, six-site grounded protocol; no open-ended vocabulary, syntax or compositional generalization.",
            "Residents are frozen; bidirectional negotiation and resident repair are deferred.",
            "Centralized immediate reward without inventory, delayed consequences, survival pressure or generational bottleneck.",
        ],
        "next_step": "Fully cross adaptation partner schedules and replacement granularity, then add resident adaptation, delayed resource consequences and generational transmission.",
        "excluded": ["results/*smoke*", "results/*probe*", "__pycache__", ".venv", "failure records", "this completion manifest"],
        "link_check": {"report_links": len(report_links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "training_files_rehashed": len(training_files), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
