"""Seal the completed v0.33 formal batch without modifying its evidence."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)

    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    validation = read(out / "raw_validation.json")
    audit = read(out / "audit_execution.json")
    analysis = read(out / "analysis.json")
    cross = read(out / "cross_team.json")
    visual_qa = read(out / "figures/visual_qa.json")
    review = read(ROOT / "结果审查.json")
    report = out / "互补观察下的双发送者共同符号形成研究报告.md"

    if not (invocation.get("formal") is True and complete.get("status") == "complete" and complete.get("formal") is True):
        raise AssertionError("formal training completion required")
    for key in ("source_hashes", "input_hashes"):
        if invocation[key] != complete[key]:
            raise AssertionError(f"{key} changed between invocation and completion")
        for path, digest in invocation[key].items():
            assert_hash(Path(path), digest)
    for rel, digest in complete["files"].items():
        assert_hash(out / rel, digest)
    if not (analysis.get("formal") and analysis.get("status") == "complete"):
        raise AssertionError("formal analysis required")
    if not validation.get("passed") or validation.get("analysis_sha256") != sha(out / "analysis.json"):
        raise AssertionError("independent raw validation failed")
    if not audit.get("passed"):
        raise AssertionError("bounded audit failed")
    if not (cross.get("passed", True) and cross.get("formal") and cross.get("status") == "complete"):
        raise AssertionError("cross-team analysis failed")
    if not visual_qa.get("passed"):
        raise AssertionError("visual QA failed")
    if not (review.get("passed") and review.get("passed_with_stated_limits") and review.get("version") == "v0.33"):
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
            if not path.is_relative_to(out):
                continue
            if any(part.startswith("smoke_") for part in path.parts):
                continue
        files.add(path.resolve())
    files.update(Path(path) for path in invocation["source_hashes"])
    files.update(Path(path) for path in invocation["input_hashes"])
    for path in audit.get("audit_dependencies", {}):
        candidate = Path(path)
        if candidate.is_file() and ".venv" not in candidate.parts:
            files.add(candidate.resolve())
    for target in linked_paths(report):
        files.add(target.resolve())
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files)]

    record = {
        "status": "complete_development_batch",
        "version": "v0.33",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "complementary private observations in a two-sender grounded resource-location protocol",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": read(out / "terminal_receipt.json"),
        "counts": {key: complete[key] for key in ("social_runs", "pair_updates", "messages", "actions", "new_private_fits", "new_dino_inferences")},
        "quality": {
            "independent_checks": validation["checks"],
            "scalar_comparisons": validation["scalar_comparisons"],
            "maximum_metric_absolute_difference": validation["maximum_metric_absolute_difference"],
            "audit_checks": audit["checks"],
            "visual_qa": visual_qa["passed"],
            "scientific_review": review["status"],
            "cross_team_analysis": cross["status"],
        },
        "report": str(report),
        "review": str(ROOT / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Four population members reuse two frozen private encoders; this tests controlled private variation rather than natural demographic or generational change.",
            "Team roles and four-agent schedules are fixed, with no birth, death, migration, cultural transmission or generation turnover.",
            "Food-only and water-only masks are experimenter-specified and the task is a symbolic resource-location protocol, not a model of complete human ecology.",
            "The bounded audit replays endpoint sender worlds, receiver tables, agreement files and sampled traces, but not every gradient or Adam transition.",
            "Cross-team token substitution is a diagnostic intervention performed after training; it is not an online communication episode.",
            "The four seeds are independent units; panels, team slots, masks, photos and world rows are nested repeats.",
        ],
        "next_step": "Rotate sender-receiver pairings during training while keeping complementary views and the two-resource payoff, then test original-team, seen-role/new-partner and unseen-team transfer before adding a third resource or delayed relay.",
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
