"""Seal the formal v0.42 endpoint intervention batch."""
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    out = root / "results" / "sequence_intervention_001"
    manifest = out / "completion_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    invocation = read(out / "invocation.json")
    analysis = read(out / "sequence_intervention_analysis.json")
    audit = read(out / "sequence_intervention_audit.json")
    visual = read(out / "visual_qa.json")
    review = read(root / "结果审查.json")
    report = out / "三资源双token端点干预研究报告.md"
    if not (
        invocation.get("formal") is True
        and invocation.get("source_version") == "v0.41-two-token-per-sender"
        and analysis.get("formal") is True
        and analysis.get("status") == "complete"
        and analysis.get("probe") == "sequence_intervention"
        and analysis.get("chains") == 72
        and len(analysis.get("rows", [])) == 864
        and analysis.get("baseline_replay_mismatches") == 0
        and audit.get("status") == "complete"
        and audit.get("formal") is True
        and visual.get("status") == "passed_visual_qa"
        and review.get("passed") is True
        and review.get("version") == "v0.42-sequence-intervention"
    ):
        raise AssertionError("formal intervention analysis/audit/review required")
    for key in ("source_hashes", "input_hashes"):
        for path, digest in invocation[key].items():
            assert_hash(Path(path), digest)

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
    artifacts = [
        {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(files)
        if path.is_file()
    ]
    coverage = audit["coverage"]
    record = {
        "status": "complete_development_batch",
        "version": "v0.42-sequence-intervention",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "endpoint counterfactual tests of two-token grounded protocol structure",
        "overall_research_goal_complete": False,
        "formal_output": str(out),
        "terminal": None,
        "counts": {
            "source_endpoint_chains": 72,
            "schedule_team_records": 864,
            "mask_tables": coverage["mask_tables"],
            "resource_block_tables": coverage["block_tables"],
            "site_relocation_tables": coverage["site_tables"],
            "mask_values_per_position": 7,
            "resource_block_permutations": 6,
            "site_permutations": 6,
        },
        "quality": {
            "baseline_replay_mismatches": analysis["baseline_replay_mismatches"],
            "independent_audit_checks": audit["checks"],
            "scalar_comparisons": audit["scalar_comparisons"],
            "maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"],
            "visual_qa": visual["status"],
            "scientific_review": review["status"],
        },
        "report": str(report),
        "review": str(root / "结果审查.json"),
        "artifacts": artifacts,
        "limitations": [
            "Endpoint intervention only; no additional formation training.",
            "Resource-block permutations are receiver-side counterfactuals.",
            "Cyclic site relocation reuses the six-site frozen visual bank and is not a new visual-location test.",
            "Pairwise or full-code success in the finite task does not establish open-ended syntax.",
            "Centralized reward and frozen private visual frontends remain inherited from v0.41.",
        ],
        "next_step": "Train full resource and role permutation conditions, then test whether the intervention signatures survive formation and cross-generation transfer.",
        "excluded": ["results/sequence_intervention_smoke", "results/*probe*", "__pycache__", ".venv", "failure records", "prior completion manifests"],
        "link_check": {"report_links": len(links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
