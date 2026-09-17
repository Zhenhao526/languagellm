"""Seal the formal v0.40 triad transfer batch."""
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
    if sha(path) != expected:
        raise AssertionError(f"hash mismatch: {path}")


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute():
            yield path


def check_training(out: Path):
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    if not (
        invocation.get("formal") is True
        and complete.get("status") == "complete"
        and complete.get("formal") is True
        and complete.get("probe") == "triad_transfer"
        and complete.get("runs") == 108
    ):
        raise AssertionError("formal training completion required")
    for key in ("source_hashes", "input_hashes"):
        if invocation[key] != complete[key]:
            raise AssertionError(f"{key} changed")
        for path, digest in invocation[key].items():
            assert_hash(Path(path), digest)
    for rel, digest in complete.get("files", {}).items():
        assert_hash(out / rel, digest)
    return invocation, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "triad_transfer_001"
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation, complete = check_training(out)
    analysis = read(out / "triad_transfer_analysis.json")
    validation = read(out / "triad_transfer_raw_validation.json")
    audit = read(out / "triad_transfer_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "三资源三token代际传递研究报告.md"
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    if not (
        analysis.get("formal")
        and analysis.get("status") == "complete"
        and validation.get("passed")
        and audit.get("status") == "complete"
        and visual.get("status") == "passed_visual_qa"
        and review.get("passed")
        and review.get("version") == "v0.40-triad-transfer"
        and not missing
    ):
        raise AssertionError("formal analysis/audit/review required")

    files = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name == "completion_manifest.json" or "failure" in path.name:
            continue
        if "results" in path.parts and any("smoke" in part for part in path.parts):
            continue
        files.add(path.resolve())
    files.update(Path(path).resolve() for path in invocation["source_hashes"])
    files.update(Path(path).resolve() for path in invocation["input_hashes"])
    files.update(path.resolve() for path in links)
    artifacts = [
        {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(files)
        if path.is_file()
    ]
    record = {
        "status": "complete_development_batch",
        "version": "v0.40-triad-transfer",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "generational transfer of a grounded three-resource three-token protocol",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": None,
        "counts": {
            "chains": complete["runs"],
            "generations_per_chain": complete["generations_per_run"],
            "updates_per_generation": complete["updates_per_generation"],
            "replacement_events": complete["replacement_events"],
            "protocol_tables": 6480,
            "trace_files": complete["trace_files_expected"],
            "trace_rows": complete["trace_files_expected"] * 240,
            "messages": complete["messages"],
            "token_instances": complete["token_instances"],
            "actions": complete["actions"],
        },
        "quality": {
            "independent_checks": validation["checks"],
            "scalar_comparisons": validation["scalar_comparisons"],
            "maximum_metric_absolute_difference": validation["maximum_metric_absolute_difference"],
            "trace_audit_checks": audit["checks"],
            "maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"],
            "visual_qa": visual["status"],
            "scientific_review": review["status"],
        },
        "report": str(report),
        "review": str(root / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Generation zero is inherited from v0.39 rather than newly formed in this batch.",
            "Visual frontends are inherited and frozen.",
            "One token per resource sender tests slot composition but not within-sender grammar.",
            "Shared reward and joint gradients are centralized.",
            "Resource-wise asymmetry requires permutation and role-symmetry controls.",
        ],
        "next_step": "Permutation-controlled two-token senders, distributed episode feedback and noisy partner transmission.",
        "excluded": ["results/triad_transfer_smoke", "__pycache__", ".venv", "failure records", "prior completion manifests"],
        "link_check": {"report_links": len(links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
