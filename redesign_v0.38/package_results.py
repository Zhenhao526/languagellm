"""Seal the formal v0.38 origin-transfer batch."""
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
    if sha(path) != expected: raise AssertionError(f"hash mismatch: {path}")


def linked_paths(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute(): yield path


def check_training(out: Path):
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json")
    if not (invocation.get("formal") is True and complete.get("status") == "complete" and complete.get("formal") is True and complete.get("probe") == "origin_transfer"): raise AssertionError("formal training completion required")
    for key in ("source_hashes", "input_hashes"):
        if invocation[key] != complete[key]: raise AssertionError(f"{key} changed")
        for path, digest in invocation[key].items(): assert_hash(Path(path), digest)
    for rel, digest in complete.get("files", {}).items(): assert_hash(out / rel, digest)
    return invocation, complete


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); root = args.out.resolve(); out = root / "results" / "transfer_001"; manifest = out / "completion_manifest.json"
    if manifest.exists(): raise FileExistsError(manifest)
    invocation, complete = check_training(out); analysis = read(out / "transfer_analysis.json"); validation = read(out / "transfer_raw_validation.json"); audit = read(out / "transfer_audit.json"); visual = read(out / "figures" / "visual_qa.json"); review = read(root / "结果审查.json"); report = out / "形成协议的代际传递研究报告.md"
    if not (analysis.get("formal") and analysis.get("status") == "complete" and validation.get("passed") and audit.get("passed") and visual.get("passed") and review.get("passed") and review.get("version") == "v0.38-origin-transfer"): raise AssertionError("formal analysis/audit/review required")
    links = list(linked_paths(report)); missing = [str(path) for path in links if not path.exists()]
    if missing: raise FileNotFoundError("missing report links: " + ", ".join(missing))
    files = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts or path.name == "completion_manifest.json": continue
        if "results" in path.parts and any(part.startswith("smoke_") for part in path.parts): continue
        files.add(path.resolve())
    files.update(Path(path).resolve() for path in invocation["source_hashes"]); files.update(Path(path).resolve() for path in invocation["input_hashes"]); files.update(path.resolve() for path in links)
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files) if path.is_file()]
    record = {"status": "complete_development_batch", "version": "v0.38-origin-transfer", "created_utc": datetime.now(timezone.utc).isoformat(), "study": "factorial transfer of teacher-free formed protocols through finite social learning", "overall_research_goal_complete": False, "formal_output": str(out), "terminal": read(out / "terminal_receipt.json") if (out / "terminal_receipt.json").is_file() else None, "counts": {"chains": complete["runs"], "replacement_events": complete["replacement_events"], "generations_per_chain": complete["generations_per_run"], "updates_per_generation": complete["updates_per_generation"], "messages": complete["messages"], "actions": complete["actions"]}, "quality": {"independent_checks": validation["checks"], "scalar_comparisons": validation["scalar_comparisons"], "maximum_metric_absolute_difference": validation["maximum_metric_absolute_difference"], "trace_audit_checks": audit["checks"], "visual_qa": visual["passed"], "scientific_review": review["status"]}, "report": str(report), "review": str(root / "结果审查.json"), "artifacts": artifacts, "limitations": ["Formation endpoints inherit frozen private visual encoders from v0.37.", "Only communication modules are optimized during replacement.", "The four-agent population, replacement order and topology schedules are experimenter specified.", "The task does not test vocabulary growth, compositional grammar, intention or natural-language structure.", "Token output changes are surface fingerprints, not semantic distance; optimizer-state replay is not included."], "next_step": "Add a third resource and multi-token sequences, then repeat the factorial under fully distributed episode feedback and explicit communication costs.", "excluded": ["results/transfer_smoke_001", "__pycache__", ".venv", "prior completion manifests"], "link_check": {"report_links": len(links), "missing": missing}}
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__": main()
