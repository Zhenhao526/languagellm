"""Seal v0.35 identity and online-adaptation artifacts with hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_hash(path: Path, expected: str):
    actual = sha(path)
    if actual != expected: raise AssertionError(f"hash mismatch: {path}: {actual} != {expected}")


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        target = Path(match.group(2))
        if target.is_absolute(): yield target


def check_training(out: Path):
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json")
    if not (invocation.get("formal") is True and complete.get("status") == "complete" and complete.get("formal") is True): raise AssertionError(f"formal training completion required: {out}")
    for key in ("source_hashes", "input_hashes"):
        if invocation[key] != complete[key]: raise AssertionError(f"{key} changed: {out}")
        for path, digest in invocation[key].items(): assert_hash(Path(path), digest)
    for rel, digest in complete.get("files", {}).items(): assert_hash(out / rel, digest)
    return invocation, complete


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); root = args.out.resolve(); identity = root / "results" / "identity_001"; adaptation = root / "results" / "adapt_001"; manifest = adaptation / "completion_manifest.json"
    if manifest.exists(): raise FileExistsError(manifest)
    identity_inv, identity_complete = check_training(identity); adaptation_inv, adaptation_complete = check_training(adaptation)
    if not (identity_inv["formal"] and adaptation_inv["formal"]): raise AssertionError("formal invocations required")
    identity_analysis = read(identity / "identity_analysis.json"); identity_validation = read(identity / "identity_raw_validation.json"); identity_audit = read(identity / "identity_audit.json")
    adaptation_analysis = read(adaptation / "adaptation_analysis.json"); adaptation_validation = read(adaptation / "adaptation_raw_validation.json"); adaptation_audit = read(adaptation / "adaptation_audit.json")
    visual = read(adaptation / "figures" / "visual_qa.json"); review = read(root / "结果审查.json"); report = adaptation / "新主体身份留出与在线适应研究报告.md"
    if not (identity_analysis["formal"] and identity_analysis["status"] == "complete" and identity_validation["passed"] and identity_audit["passed"]): raise AssertionError("identity analysis incomplete")
    if not (adaptation_analysis["formal"] and adaptation_analysis["status"] == "complete" and adaptation_validation["passed"] and adaptation_audit["passed"]): raise AssertionError("adaptation analysis incomplete")
    if not (visual["passed"] and review["passed"] and review["version"] == "v0.35-identity-adaptation"): raise AssertionError("visual/scientific review incomplete")
    if not report.is_file(): raise FileNotFoundError(report)
    links = list(linked_paths(report)); missing = [str(path) for path in links if not path.exists()]
    if missing: raise FileNotFoundError("missing report links: " + ", ".join(missing))

    files: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts or path.name == "completion_manifest.json": continue
        if "results" in path.parts and any(part.startswith("smoke_") for part in path.parts): continue
        files.add(path.resolve())
    for invocation in (identity_inv, adaptation_inv):
        files.update(Path(path).resolve() for path in invocation["source_hashes"])
        files.update(Path(path).resolve() for path in invocation["input_hashes"])
    files.update(path.resolve() for path in links)
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files) if path.is_file()]
    record = {
        "status": "complete_development_batch", "version": "v0.35-identity-adaptation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "zero-shot identity holdout and online cultural acquisition under fixed versus rotating partner topologies",
        "overall_research_goal_complete": False, "formal_outputs": {"identity": str(identity), "adaptation": str(adaptation)},
        "terminal": {"identity": read(identity / "terminal_receipt.json"), "adaptation": read(adaptation / "terminal_receipt.json")},
        "counts": {"identity_endpoint_rows": len(identity_analysis.get("rows", [])), "adaptation_runs": adaptation_complete["runs"], "adaptation_updates_per_run": adaptation_complete["updates_per_run"], "adaptation_messages": adaptation_complete["messages"], "adaptation_actions": adaptation_complete["actions"]},
        "quality": {"identity_independent_checks": identity_validation["checks"], "identity_scalar_comparisons": identity_validation["scalar_comparisons"], "identity_maximum_metric_absolute_difference": identity_validation["maximum_metric_absolute_difference"], "identity_audit_checks": identity_audit["checks"], "adaptation_independent_checks": adaptation_validation["checks"], "adaptation_scalar_comparisons": adaptation_validation["scalar_comparisons"], "adaptation_maximum_metric_absolute_difference": adaptation_validation["maximum_metric_absolute_difference"], "adaptation_audit_checks": adaptation_audit["checks"], "visual_qa": visual["passed"], "scientific_review": review["status"]},
        "reports": [str(report), str(root / "README.md")], "review": str(root / "结果审查.json"), "artifacts": artifacts,
        "limitations": [
            "The agents are small visual–communication mechanism probes, not a large open-source VLM/LLM or an end-to-end world model.",
            "Residents are imported v0.34 endpoint policies; the batch has no births, deaths, population turnover or generational bottleneck.",
            "The newcomer keeps a private visual encoder and only its communication modules are reset/optimized.",
            "The two-resource task does not test vocabulary growth, compositional grammar, intention or natural-language structure.",
            "The audit replays bounded endpoint tables and categorical traces, not every optimizer state transition.",
        ],
        "next_step": "Iterated replacement with a finite social-learning bottleneck, comparing fixed-A, rotating-AB and random-topology teacher histories across generations.",
        "excluded": ["results/adapt_smoke_001", "__pycache__", ".venv", "prior completion manifests"], "link_check": {"report_links": len(links), "missing": missing},
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__": main()
