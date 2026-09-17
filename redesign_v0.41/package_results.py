"""Seal the formal v0.41 two-token formation batch."""
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
        and complete.get("probe") == "two_token_formation"
        and complete.get("runs") == 72
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
    out = root / "results" / "two_token_001"
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation, complete = check_training(out)
    analysis = read(out / "two_token_analysis.json")
    validation = read(out / "two_token_raw_validation.json")
    audit = read(out / "two_token_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "三资源双token协议形成研究报告.md"
    links = list(linked_paths(report))
    missing = [str(path) for path in links if not path.exists()]
    if not (
        analysis.get("formal")
        and analysis.get("status") == "complete"
        and validation.get("passed")
        and audit.get("status") == "complete"
        and visual.get("status") == "passed_visual_qa"
        and review.get("passed")
        and review.get("version") == "v0.41-two-token-per-sender"
        and not missing
    ):
        raise AssertionError("formal analysis/audit/review required")

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
    artifacts = [
        {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(files)
        if path.is_file()
    ]
    record = {
        "status": "complete_development_batch",
        "version": "v0.41-two-token-per-sender",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "formation of a grounded six-token protocol with two autoregressive tokens per resource sender",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": None,
        "counts": {
            "chains": complete["runs"],
            "updates_per_chain": complete["updates_per_run"],
            "protocol_tables": complete["protocol_tables"],
            "trace_files": complete["trace_files_expected"],
            "trace_rows": complete["trace_rows_expected"],
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
            "The batch starts from fresh random communication heads rather than a transferred endpoint.",
            "Visual frontends are inherited from v0.28 and frozen.",
            "Pair NMI saturation in the six-location task does not establish syntax or open-ended compositionality.",
            "Only canonical and one cyclic resource permutation are included; full resource/role symmetry is pending.",
            "Shared reward and joint policy-gradient updates are centralized.",
        ],
        "next_step": "Enumerate resource and role permutations, perform token deletion/swapping and novel-location interventions, then test sequence-protocol transfer across replacement generations.",
        "excluded": ["results/two_token_smoke*", "results/two_token_probe*", "__pycache__", ".venv", "failure records", "prior completion manifests"],
        "link_check": {"report_links": len(links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
