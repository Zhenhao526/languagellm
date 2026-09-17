"""Seal the completed v0.32 formal batch without modifying its evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)

    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    validation = read(out / "raw_validation.json")
    audit = read(out / "audit_execution.json")
    analysis = read(out / "analysis.json")
    visual_qa = read(out / "figures/visual_qa.json")
    review = read(ROOT / "结果审查.json")
    report = out / "联合收益压力下的共同符号形成研究报告.md"

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
    if not visual_qa.get("passed"):
        raise AssertionError("visual QA failed")
    if not (review.get("passed") and review.get("passed_with_stated_limits") and review.get("version") == "v0.32"):
        raise AssertionError("scientific review failed")
    if not report.is_file():
        raise FileNotFoundError(report)
    for target in linked_paths(report):
        if not target.exists():
            raise FileNotFoundError(target)

    # The package contains the current source/design, the current formal output,
    # and every hash-bound inherited input. Superseded result directories are
    # left on disk but are excluded from this v0.32 package.
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
        "version": "v0.32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "same-world joint payoff with fixed versus rotating partners in a multi-partner target task",
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
        },
        "report": str(report),
        "review": str(ROOT / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Four population members reuse two frozen private encoders; this tests controlled private variation rather than natural demographic or generational change.",
            "Partner rotation is a fixed four-agent schedule without birth, death, migration, cultural transmission or generation turnover.",
            "The target is a symbolic resource-location protocol, not a reconstruction of human ecology.",
            "The bounded audit does not replay every gradient and Adam state transition; inherited visual and private inputs are hash-bound.",
            "Multi-partner simultaneous correctness remains low, and offline recombination cannot establish autonomous composition.",
            "The four initialization sources are the independent units; panels, directions, masks and photos are nested repeats.",
        ],
        "next_step": "Make one receiver action depend on two non-interchangeable senders with complementary observations, retaining fixed/rotating, single-sender and random-message controls.",
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
