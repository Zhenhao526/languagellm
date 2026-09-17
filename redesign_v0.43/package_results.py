"""Seal the formal v0.43 permutation-formation batch."""
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


def assert_hash(path: Path, expected: str):
    if not path.is_file():
        raise AssertionError(f"missing input: {path}")
    actual = sha(path)
    if actual != expected:
        raise AssertionError(f"hash mismatch: {path}: {actual} != {expected}")


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute():
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="v0.43 project directory")
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "permutation_001"
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    analysis = read(out / "permutation_analysis.json")
    stats = read(out / "permutation_statistics.json")
    audit = read(out / "permutation_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "三资源排列与角色随机化形成研究报告.md"
    if not (
        invocation.get("formal") is True
        and invocation.get("version") == "v0.43-permutation-formation"
        and complete.get("status") == "complete"
        and complete.get("formal") is True
        and complete.get("runs") == 432
        and analysis.get("status") == "complete"
        and analysis.get("formal") is True
        and analysis.get("runs") == 432
        and len(analysis.get("rows", [])) == 124416
        and stats.get("status") == "complete"
        and stats.get("formal") is True
        and audit.get("status") == "complete"
        and audit.get("formal") is True
        and audit.get("runs") == 432
        and visual.get("status") == "passed_visual_qa"
        and review.get("passed") is True
        and review.get("version") == "v0.43-permutation-formation"
    ):
        raise AssertionError("formal permutation analysis/audit/review required")
    for key in ("source_hashes", "input_hashes"):
        for path, digest in invocation[key].items():
            assert_hash(Path(path), digest)
    # Every file recorded by the trainer is rehashed before sealing.  New
    # analysis/report artifacts are added below and are not part of the
    # training-time manifest.
    training_files = complete.get("files", {})
    for relative, digest in training_files.items():
        assert_hash(out / relative, digest)

    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
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
    files.update(path.resolve() for path in links)
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files) if path.is_file()]
    coverage = audit["coverage"]
    record = {
        "status": "complete_development_batch",
        "version": "v0.43-permutation-formation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "formation-stage all-resource permutations and randomized resource-role order",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": None,
        "counts": {
            "formation_chains": complete["runs"],
            "updates_per_chain": complete["updates_per_run"],
            "resource_assignments": len(complete["assignments"]),
            "role_modes": len(complete["role_modes"]),
            "partner_conditions": len(invocation["conditions"]),
            "protocol_files": coverage["protocols"],
            "role_evaluation_tables": analysis["coverage"]["protocol_tables"],
            "trace_files": coverage["traces"],
            "trace_rows": coverage["trace_rows"],
            "messages": complete["messages"],
            "token_instances": complete["token_instances"],
            "actions": complete["actions"],
        },
        "quality": {
            "analysis_checks": analysis["checks"],
            "analysis_scalar_comparisons": analysis["scalar_comparisons"],
            "analysis_maximum_metric_absolute_difference": analysis["maximum_metric_absolute_difference"],
            "statistics_bootstrap_replicates": stats["bootstrap_replicates"],
            "audit_checks": audit["checks"],
            "audit_scalar_comparisons": audit["scalar_comparisons"],
            "audit_maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"],
            "visual_qa": visual["status"],
            "scientific_review": review["status"],
        },
        "report": str(report),
        "review": str(root / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Finite vocabulary 7, two-token messages, six sites and three resources; no open-ended language benchmark.",
            "Frozen private visual frontends and centralized immediate reward are inherited from prior batches.",
            "Role randomization synchronizes sender views and target axes and therefore acts as a formation-stage symmetry factor.",
            "Resource visual strata are not perfectly matched; assignment effects need a balanced bank.",
            "Endpoint role-equivalent scores are counterfactual target-axis evaluations, not novel-agent or novel-location transfer.",
        ],
        "next_step": "Add held-out role/resource compositions and partner replacement, then introduce delayed inventory-based complementary work while preserving role-spread and replay audits.",
        "excluded": ["results/permutation_smoke", "results/*probe*", "__pycache__", ".venv", "failure records", "prior completion manifests"],
        "link_check": {"report_links": len(links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "training_files_rehashed": len(training_files), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
